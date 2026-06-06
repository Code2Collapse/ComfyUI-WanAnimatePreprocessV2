"""Build list of custom-pack nodes present in a running ComfyUI instance."""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C2C = ROOT / "ComfyUI-CustomNodePacks"
WAN = ROOT / "ComfyUI-WanAnimatePreprocessV2"
COMFY_URL = "http://127.0.0.1:8188"
OUT = Path(__file__).resolve().parent / "_playwright_node_manifest.json"


def _keys_from_py(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore")
    keys: set[str] = set()
    if "NODE_CLASS_MAPPINGS" in text:
        for m in re.finditer(
            r'NODE_CLASS_MAPPINGS\s*=\s*\{([^;]+?)\n\}',
            text,
            re.S,
        ):
            keys.update(re.findall(r'"([A-Za-z0-9_]+)"\s*:', m.group(1)))
    if "EXTRA_NODE_CLASS_MAPPINGS" in text:
        for m in re.finditer(
            r'EXTRA_NODE_CLASS_MAPPINGS\s*=\s*\{([^;]+?)\n\}',
            text,
            re.S,
        ):
            keys.update(re.findall(r'"([A-Za-z0-9_]+)"\s*:', m.group(1)))
    keys.update(re.findall(r'^class\s+(Wan[A-Za-z0-9_]+)\s*:', text, re.M))
    return keys


def collect_pack_nodes() -> set[str]:
    ours: set[str] = set()
    for p in C2C.rglob("__init__.py"):
        ours |= _keys_from_py(p)
    for p in [WAN / "nodes_extras" / "__init__.py", WAN / "nodes.py"]:
        ours |= _keys_from_py(p)
    # MEC suffix heuristic for nodes defined only in node.py files
    for p in C2C.rglob("node.py"):
        t = p.read_text(encoding="utf-8", errors="ignore")
        ours.update(re.findall(r'^class\s+([A-Za-z0-9_]+)\s*:', t, re.M))
    return {k for k in ours if k and not k.startswith("_")}


def _is_pack_node(name: str) -> bool:
    """Heuristic: nodes belonging to CustomNodePacks / WanAnimatePreprocessV2."""
    if name.endswith("MEC") or name.endswith("C2C"):
        return True
    if name.endswith("V2") and (
        name.startswith("Wan")
        or name in {
            "PoseAndFaceDetectionV2",
            "DrawViTPoseV2",
            "DepthPoseCannyCombinedV2",
            "OnnxDetectionModelLoaderV2",
            "WanAnimateFaceQualityCheckV2",
        }
    ):
        return True
    if name in {
        "MaskEditMEC",
        "MaskOpsMEC",
        "MaskRefineMEC",
        "MaskTemporalMEC",
        "MaskTrackerMEC",
        "MaskFailureExplainerMEC",
        "MECAdvancedPaintCanvas",
        "MECContextInpainter",
        "MECFaceFixer",
        "MECToneRefiner",
        "MECBuilderSampler",
        "LocateAnythingGroundingMEC",
        "LocateAnythingToSAMMEC",
        "SAMMaskGeneratorMEC",
        "PointsMaskEditor",
        "WanDirectorC2C",
    }:
        return True
    return False


def main() -> None:
    ours = collect_pack_nodes()
    with urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=180) as resp:
        info = json.loads(resp.read().decode())
    from_api = sorted(k for k in info if _is_pack_node(k))
    loaded = sorted(set(from_api) | {k for k in ours if k in info})
    missing = sorted(k for k in ours if k not in info)
    payload = {
        "comfy_url": COMFY_URL,
        "pack_defs": len(ours),
        "loaded": loaded,
        "missing": missing,
        "heuristic_count": len(from_api),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"pack_defs={len(ours)} loaded={len(loaded)} missing={len(missing)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
