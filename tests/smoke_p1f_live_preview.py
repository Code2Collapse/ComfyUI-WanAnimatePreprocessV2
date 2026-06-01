"""
P1.F — Live-preview smoke test for FaceController3DV2.

Two phases:
  PHASE A (always runs, no server, no browser):
    1. Import nodes_extras → verify FaceController3DV2 INPUT_TYPES has the
       4 new DOF widgets from P1.A (head_scale, jaw_rot_deg,
       neck_yaw_deg, neck_pitch_deg) AND hidden unique_id.
    2. Verify RETURN_TYPES is the 7-tuple from P1.C.
    3. Import _face_preview_server → verify register_routes / cache APIs
       exist and the float whitelist contains the new DOFs.
    4. Import _face_overlay_render → run render_overlay_frame on a
       synthetic meta and assert tensor shape (H,W,3) uint8.
    5. node --check on face_controller_3d.js.

  PHASE B (only if ComfyUI is up):
    1. Probe GET /c2c/fc3d_preview/status (the only read-only HTTP we
       allow in this driver — it's a *status* probe, not a node-creation
       shortcut; the v2 ban targets POST /prompt and GET /object_info).
    2. Probe POST /c2c/fc3d_preview with a fake node_id to confirm the
       412 cache_miss path is wired correctly.
    3. (Optional) If playwright is installed and a chromium browser
       binary is present, open ComfyUI, double-click canvas, search
       for WanFaceController3DV2, add the node, and screenshot. This
       is the DOM-only proof per /memories/comfyui_qa_stress_test.md v2.

Exit code 0 = phase A passed (phase B reported but non-blocking).
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # ComfyUI-WanAnimatePreprocessV2/
PACK_DIR = ROOT.parent                                  # Custom_Nodes/
# Make the inner package importable as plain "nodes_extras".
sys.path.insert(0, str(ROOT))
SHOTS = ROOT / "tests" / "p1f_shots"
SHOTS.mkdir(exist_ok=True, parents=True)


def _import_submodule(modname: str, filename: str):
    """Import a single nodes_extras submodule WITHOUT running the package
    __init__.py (which would pull in heavy siblings like ViTPose detector
    that need ../pose_utils on sys.path)."""
    import importlib.util
    import types
    # Create an empty placeholder nodes_extras package so relative imports
    # like `from ._face_helpers import ...` resolve.
    if "nodes_extras" not in sys.modules:
        pkg = types.ModuleType("nodes_extras")
        pkg.__path__ = [str(ROOT / "nodes_extras")]  # type: ignore[attr-defined]
        sys.modules["nodes_extras"] = pkg
    full = f"nodes_extras.{modname}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, str(ROOT / "nodes_extras" / filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod

OK = "[OK]  "
NO = "[FAIL]"


def _hdr(msg: str) -> None:
    print(f"\n=== {msg} ===")


def _check_server_up(host="127.0.0.1", port=8188, timeout=1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# ───────────────────────── PHASE A — offline ─────────────────────────
def phase_a() -> bool:
    all_ok = True
    _hdr("PHASE A.1: FaceController3DV2 INPUT_TYPES / RETURN_TYPES")
    try:
        # Import via the package path so relative imports work.
        import importlib
        mod = _import_submodule("face_controller_3d", "face_controller_3d.py")
        cls = mod.WanFaceController3DV2
        it = cls.INPUT_TYPES()
        required = it.get("required", {})
        optional = it.get("optional", {})
        hidden   = it.get("hidden",   {})

        new_dofs = ["head_scale", "jaw_rot_deg", "neck_yaw_deg", "neck_pitch_deg"]
        missing = [k for k in new_dofs if k not in required and k not in optional]
        if missing:
            print(f"{NO} missing DOF widgets: {missing}")
            all_ok = False
        else:
            print(f"{OK} all 4 new DOF widgets present")

        if "unique_id" not in hidden:
            print(f"{NO} hidden input 'unique_id' missing"); all_ok = False
        else:
            print(f"{OK} hidden 'unique_id' present")

        rt = cls.RETURN_TYPES
        rn = cls.RETURN_NAMES
        expected_rt = ("POSEDATA", "STRING", "STRING", "IMAGE", "IMAGE", "STRING", "STRING")
        expected_rn = ("pose_data", "info", "coeff_time_series_json",
                       "preview_image", "overlay_video",
                       "keyframes_csv", "pose_diff_json")
        if tuple(rt) != expected_rt:
            print(f"{NO} RETURN_TYPES mismatch: got {rt}"); all_ok = False
        else:
            print(f"{OK} RETURN_TYPES = 7-tuple as expected")
        if tuple(rn) != expected_rn:
            print(f"{NO} RETURN_NAMES mismatch: got {rn}"); all_ok = False
        else:
            print(f"{OK} RETURN_NAMES = 7-tuple as expected")
    except Exception as ex:
        print(f"{NO} import / introspection failed: {ex!r}")
        all_ok = False

    _hdr("PHASE A.2: _face_preview_server module")
    try:
        fps = _import_submodule("_face_preview_server", "_face_preview_server.py")
        for name in ("register_routes", "try_register_routes_deferred",
                     "cache_put", "cache_get", "cache_clear", "cache_size",
                     "_FLOAT_KWARGS", "_BOOL_KWARGS", "_STR_KWARGS"):
            if not hasattr(fps, name):
                print(f"{NO} missing symbol: {name}"); all_ok = False
        floats = set(getattr(fps, "_FLOAT_KWARGS", ()))
        for k in ("head_scale", "jaw_rot_deg", "neck_yaw_deg", "neck_pitch_deg"):
            if k not in floats:
                print(f"{NO} _FLOAT_KWARGS missing {k}"); all_ok = False
        if all_ok:
            print(f"{OK} module exports + float whitelist complete")

        # Round-trip cache_put/cache_get/cache_clear.
        fps.cache_clear()
        fps.cache_put("test_node_99", {"hello": "world"})
        got = fps.cache_get("test_node_99")
        if got != {"hello": "world"}:
            print(f"{NO} cache round-trip failed: {got!r}"); all_ok = False
        else:
            print(f"{OK} bundle cache round-trip OK (size={fps.cache_size()})")
        fps.cache_clear()
    except Exception as ex:
        print(f"{NO} preview-server import failed: {ex!r}")
        all_ok = False

    _hdr("PHASE A.3: _face_overlay_render module")
    try:
        for_mod = _import_submodule("_face_overlay_render", "_face_overlay_render.py")
        # Build a synthetic meta with 68 evenly-spaced face landmarks.
        import numpy as np
        face_norm = np.array([
            [0.3 + 0.4 * (i % 8) / 7.0,
             0.3 + 0.4 * (i // 8) / 8.0] for i in range(68)
        ], dtype=np.float32)
        meta = {
            "width":  512,
            "height": 512,
            "face": face_norm.tolist(),
            "body": None,
            "gaze": {
                "left_gaze":  {"yaw_rad": 0.05, "pitch_rad": 0.02},
                "right_gaze": {"yaw_rad": 0.04, "pitch_rad": 0.01},
            },
        }
        img = for_mod.render_overlay_frame(
            meta=meta,
            iris_entry=meta["gaze"],
            body_xy=None,
            out_size=256,
            edges=[],
            face_norm=face_norm,
            frame_idx=0,
            n_frames=1,
        )
        if not (hasattr(img, "shape") and img.shape == (256, 256, 3) and img.dtype == np.uint8):
            print(f"{NO} render returned shape={getattr(img,'shape',None)}, dtype={getattr(img,'dtype',None)}")
            all_ok = False
        else:
            print(f"{OK} render_overlay_frame → (256,256,3) uint8")

        ten = for_mod.render_to_tensor(img)
        # ComfyUI IMAGE is (B,H,W,3) float32 [0,1].
        if not (hasattr(ten, "shape") and tuple(ten.shape) == (1, 256, 256, 3)):
            print(f"{NO} render_to_tensor wrong shape: {tuple(ten.shape)}")
            all_ok = False
        else:
            print(f"{OK} render_to_tensor → (1,256,256,3)")
    except Exception as ex:
        print(f"{NO} overlay-render check failed: {ex!r}")
        all_ok = False

    _hdr("PHASE A.4: node --check face_controller_3d.js")
    js = ROOT / "js" / "face_controller_3d.js"
    try:
        cp = subprocess.run(["node", "--check", str(js)],
                            capture_output=True, text=True, timeout=30)
        if cp.returncode != 0:
            print(f"{NO} JS syntax error:\n{cp.stderr}")
            all_ok = False
        else:
            print(f"{OK} JS clean ({js.name})")
    except FileNotFoundError:
        print(f"[WARN] node not on PATH; skipping JS syntax check")
    except Exception as ex:
        print(f"{NO} node --check failed: {ex!r}")
        all_ok = False

    return all_ok


# ───────────────────────── PHASE B — online ──────────────────────────
def phase_b() -> bool:
    if not _check_server_up():
        print("[SKIP] ComfyUI not up on 127.0.0.1:8188 — skipping phase B")
        return True
    all_ok = True
    _hdr("PHASE B.1: GET /c2c/fc3d_preview/status")
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8188/c2c/fc3d_preview/status", timeout=5
        ) as r:
            body = r.read().decode("utf-8", errors="replace")
            data = json.loads(body) if body else {}
            print(f"{OK} status {r.status} → {data}")
    except urllib.error.HTTPError as he:
        print(f"{NO} status endpoint HTTP {he.code}: {he.reason}")
        all_ok = False
    except Exception as ex:
        print(f"{NO} status endpoint unreachable: {ex!r}")
        all_ok = False

    _hdr("PHASE B.2: POST /c2c/fc3d_preview (expect 412 cache_miss)")
    try:
        payload = json.dumps({
            "node_id": "smoke_p1f_unknown",
            "frame_idx": 0,
            "head_yaw_deg": 5.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8188/c2c/fc3d_preview",
            data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                print(f"{NO} expected 412 cache_miss, got {r.status}")
                all_ok = False
        except urllib.error.HTTPError as he:
            if he.code == 412:
                try:
                    body = json.loads(he.read().decode("utf-8"))
                except Exception:
                    body = {}
                if body.get("error") == "cache_miss":
                    print(f"{OK} 412 cache_miss returned as designed")
                else:
                    print(f"{NO} 412 returned but body={body!r}")
                    all_ok = False
            else:
                print(f"{NO} unexpected HTTP {he.code}: {he.reason}")
                all_ok = False
    except Exception as ex:
        print(f"{NO} POST probe failed: {ex!r}")
        all_ok = False

    _hdr("PHASE B.3: Playwright DOM probe (best-effort, non-blocking)")
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except Exception:
        print("[SKIP] playwright not installed")
        return all_ok

    try:
        with sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(headless=True)
            except Exception as ex:
                print(f"[SKIP] chromium unavailable: {ex!r}")
                return all_ok
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            try:
                page.goto("http://127.0.0.1:8188/", wait_until="domcontentloaded", timeout=20000)
                page.wait_for_selector("canvas#graph-canvas", timeout=20000)
                page.wait_for_function("() => !!(window.app && window.app.graph)", timeout=20000)
                page.wait_for_timeout(1200)
                page.screenshot(path=str(SHOTS / "01_boot.png"))

                # Clear graph.
                page.evaluate("() => app.graph?.clear?.()")
                page.wait_for_timeout(150)

                # Double-click to invoke search palette and type class name.
                box = page.locator("canvas#graph-canvas").first.bounding_box()
                cx = box["x"] + box["width"] / 2
                cy = box["y"] + box["height"] / 2
                page.keyboard.press("Escape")
                page.mouse.dblclick(cx, cy)
                palette = None
                for sel in (
                    ".comfy-vue-node-search-container input",
                    ".litegraph-search-input",
                    ".litegraphcontextmenu input",
                    ".p-autocomplete-input",
                ):
                    try:
                        page.wait_for_selector(sel, timeout=1500, state="visible")
                        palette = page.locator(sel).first
                        break
                    except PWTimeout:
                        continue
                if not palette:
                    print(f"{NO} no search palette appeared")
                    page.screenshot(path=str(SHOTS / "02_no_palette.png"))
                    all_ok = False
                else:
                    palette.fill("")
                    palette.type("WanFaceController3DV2", delay=8)
                    page.wait_for_timeout(400)
                    page.keyboard.press("Enter")
                    page.wait_for_timeout(600)
                    page.screenshot(path=str(SHOTS / "03_after_add.png"))

                    info = page.evaluate(
                        "() => {\n"
                        "  const ns = app.graph?._nodes || [];\n"
                        "  for (let i = ns.length - 1; i >= 0; i--) {\n"
                        "    const n = ns[i];\n"
                        "    if (n.type === 'WanFaceController3DV2' || n.comfyClass === 'WanFaceController3DV2') {\n"
                        "      const widgets = (n.widgets||[]).map(w => w.name);\n"
                        "      return {ok:true, id:n.id, title:n.title, widgets};\n"
                        "    }\n"
                        "  }\n"
                        "  return {ok:false, count:ns.length};\n"
                        "}"
                    )
                    if not info or not info.get("ok"):
                        print(f"{NO} node not in graph: {info!r}")
                        all_ok = False
                    else:
                        print(f"{OK} node added: id={info['id']}, {len(info['widgets'])} widgets")
                        missing = [k for k in
                                   ("head_scale", "jaw_rot_deg", "neck_yaw_deg", "neck_pitch_deg")
                                   if k not in info["widgets"]]
                        if missing:
                            print(f"{NO} DOF widgets missing in DOM: {missing}")
                            all_ok = False
                        else:
                            print(f"{OK} all 4 new DOF widgets visible in node DOM")
                        # Check overlay div mounted.
                        has_overlay = page.evaluate(
                            "(id) => {\n"
                            "  const n = app.graph?._nodes.find(x => x.id === id);\n"
                            "  return !!(n && n._faceOverlay && n._faceOverlay.root);\n"
                            "}", info["id"]
                        )
                        if has_overlay:
                            print(f"{OK} face overlay DOM widget mounted")
                        else:
                            print(f"[WARN] overlay not yet mounted — non-blocking")
            finally:
                browser.close()
    except Exception as ex:
        print(f"[WARN] Playwright phase exception (non-blocking): {ex!r}")

    return all_ok


def main() -> int:
    a_ok = phase_a()
    b_ok = phase_b()
    print()
    print("=" * 60)
    print(f"PHASE A (offline): {'PASS' if a_ok else 'FAIL'}")
    print(f"PHASE B (online) : {'PASS' if b_ok else 'FAIL/WARN'}")
    print("=" * 60)
    return 0 if a_ok else 2


if __name__ == "__main__":
    sys.exit(main())
