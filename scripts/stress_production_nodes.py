#!/usr/bin/env python3
"""Import + smoke-run stress test for WanAnimatePreprocessV2 production nodes."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMFYUI = Path(os.environ.get("COMFYUI_PATH", r"D:\PROJECT\ComfyUI_windows_portable\ComfyUI"))
PKG_NAME = "custom_nodes.ComfyUI-WanAnimatePreprocessV2"


def _bootstrap_pack() -> str:
    """Load the pack the same way ComfyUI does (parent-relative imports work)."""
    if str(COMFYUI) not in sys.path:
        sys.path.insert(0, str(COMFYUI))
    if PKG_NAME in sys.modules:
        return PKG_NAME
    spec = importlib.util.spec_from_file_location(
        PKG_NAME,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pack from {ROOT}")
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[PKG_NAME] = pkg
    spec.loader.exec_module(pkg)
    return PKG_NAME


def _face_meta(i: int, w: int = 512, h: int = 512) -> dict:
    rng = np.random.default_rng(i + 7)
    kf = rng.random((68, 2), dtype=np.float32)
    kf[:, 0] = 0.2 + kf[:, 0] * 0.6
    kf[:, 1] = 0.15 + kf[:, 1] * 0.7
    return {"keypoints_face": kf, "width": float(w), "height": float(h)}


def make_pose_data(n: int = 4) -> dict:
    return {
        "pose_metas": [_face_meta(i) for i in range(n)],
        "iris_data": [None] * n,
    }


def _mod(sub: str) -> str:
    return f"{PKG_NAME}.nodes_extras.{sub}"


WANIMATE_PRODUCTION = [
    (_mod("iris_controlnet"), "WanIrisControlNetV2", None),
    (_mod("sh_lighting"), "WanSHLightingTransferV2", None),
    (_mod("quality_scorer_jitter"), "WanQualityScorerJitterV2", None),
    (_mod("pose_format_convert"), "WanPoseFormatConvertV2", None),
    (_mod("pose_detect_vitpose"), "WanPoseDetectViTPoseV2", None),
    (_mod("face_controller_3d"), "WanFaceController3DV2", "face_controller"),
]


def check_schema(cls, label: str) -> list[str]:
    errs = []
    if not hasattr(cls, "INPUT_TYPES"):
        errs.append(f"{label}: missing INPUT_TYPES")
        return errs
    if not hasattr(cls, "RETURN_TYPES"):
        errs.append(f"{label}: missing RETURN_TYPES")
        return errs
    try:
        spec = cls.INPUT_TYPES()
    except Exception as exc:
        errs.append(f"{label}: INPUT_TYPES() failed: {exc}")
        return errs
    if "required" not in spec:
        errs.append(f"{label}: INPUT_TYPES missing 'required'")
    return errs


def smoke_face_controller(cls) -> list[str]:
    errs = []
    pose = make_pose_data(3)
    inst = cls()
    try:
        out = inst.run(
            pose,
            expression_coeffs_json=json.dumps({
                "frames": {"1": {"smile": 0.75}},
            }),
            expression_strength=1.0,
            unique_id="stress-test-node",
        )
    except Exception as exc:
        errs.append(f"WanFaceController3DV2.run failed: {exc}")
        traceback.print_exc()
        return errs
    if not isinstance(out, dict) or "result" not in out:
        errs.append("WanFaceController3DV2: run() must return dict with result")
        return errs
    result = out["result"]
    if not isinstance(result, tuple) or len(result) < 3:
        errs.append("WanFaceController3DV2: short result tuple")
        return errs
    bundle, info, coeff_json = result[0], result[1], result[2]
    if not isinstance(bundle, dict):
        errs.append("WanFaceController3DV2: pose_data output not dict")
    info_l = str(info).lower()
    if "expr" not in info_l and "frames=" not in info_l:
        errs.append(f"WanFaceController3DV2: unexpected info: {info!r}")
    try:
        coeffs = json.loads(coeff_json)
    except json.JSONDecodeError as exc:
        errs.append(f"WanFaceController3DV2: coeff JSON invalid: {exc}")
    else:
        if not coeffs:
            errs.append("WanFaceController3DV2: empty coeff_time_series")
    ui = out.get("ui") or {}
    om = ui.get("overlay_meta")
    if not om or not om[0]:
        errs.append("WanFaceController3DV2: missing overlay_meta ui")
    else:
        meta = json.loads(om[0])
        frames = meta.get("frames") or []
        if not frames or not frames[0].get("lms"):
            errs.append("WanFaceController3DV2: overlay_meta.frames missing lms")
    return errs


def main() -> int:
    failed = []
    passed = []

    try:
        _bootstrap_pack()
    except Exception as exc:
        print(f"=== WanAnimate production stress test ===\nFAIL: pack bootstrap: {exc}")
        return 1

    for mod_name, cls_name, smoke in WANIMATE_PRODUCTION:
        label = cls_name
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
        except Exception as exc:
            failed.append(f"{label}: import failed: {exc}")
            continue
        errs = check_schema(cls, label)
        if smoke == "face_controller":
            errs.extend(smoke_face_controller(cls))
        if errs:
            failed.extend(errs)
        else:
            passed.append(label)

    try:
        fps = importlib.import_module(_mod("_face_preview_server"))
        assert hasattr(fps, "cache_put") and hasattr(fps, "cache_get")
        passed.append("_face_preview_server")
    except Exception as exc:
        failed.append(f"_face_preview_server: {exc}")

    print("=== WanAnimate production stress test ===")
    print(f"PASS ({len(passed)}): {', '.join(passed)}")
    if failed:
        print(f"FAIL ({len(failed)}):")
        for e in failed:
            print(f"  - {e}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
