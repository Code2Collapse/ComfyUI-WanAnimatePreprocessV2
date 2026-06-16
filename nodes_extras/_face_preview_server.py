"""Live-preview server for ``WanFaceController3DV2`` (Plan Phase 1.B).

Registers ``POST /c2c/fc3d_preview`` on the ComfyUI aiohttp server. The JS
gizmo (see ``js/face_controller_3d.js`` + ``js/face_3d_editor.js``) sends
the current widget state for a single frame and receives back the
per-frame edited landmarks / body keypoints / gaze in the same coordinate
system as ``overlay_meta`` so it can repaint without re-queueing the
whole graph.

Coupling contract:
  • ``WanFaceController3DV2.run()`` stashes the raw input ``pose_data``
    bundle into :data:`_BUNDLE_CACHE` keyed by the node's ``unique_id``.
  • The route looks up that bundle, re-runs the node's ``run()`` with
    ``frame_start = frame_end = frame_idx`` (so only the requested frame
    is touched), then returns a compact JSON payload.
  • Cache miss → HTTP 412 PRECONDITION_FAILED with ``{"error": "..."}``,
    telling the JS layer to queue the node once before requesting
    previews.

No fallbacks, no stubs. If aiohttp / ComfyUI server is unavailable the
module logs a warning and the function returns; nothing else breaks.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# In-memory bundle cache (node_id → raw input POSEDATA bundle)
# ----------------------------------------------------------------------
# A single bundle can be 50-200 MB for long shots, so we cap entries to
# 8 most-recent nodes. Eviction is LRU via dict insertion order. Cache
# access is guarded by a lock since the route handler runs on the
# aiohttp event loop while ``run()`` is on the prompt-worker thread.
_BUNDLE_CACHE_MAX = 8
_BUNDLE_CACHE: "dict[str, dict]" = {}
_BUNDLE_CACHE_LOCK = threading.Lock()


def cache_put(node_id: str, bundle: dict) -> None:
    """Stash a raw input POSEDATA bundle for later live-preview requests."""
    if not node_id:
        return
    with _BUNDLE_CACHE_LOCK:
        if node_id in _BUNDLE_CACHE:
            # Refresh LRU position.
            _BUNDLE_CACHE.pop(node_id, None)
        _BUNDLE_CACHE[node_id] = bundle
        while len(_BUNDLE_CACHE) > _BUNDLE_CACHE_MAX:
            # Pop oldest (FIFO order matches LRU here since we re-insert on hit).
            oldest_key = next(iter(_BUNDLE_CACHE))
            _BUNDLE_CACHE.pop(oldest_key, None)


def cache_get(node_id: str) -> Optional[dict]:
    if not node_id:
        return None
    with _BUNDLE_CACHE_LOCK:
        b = _BUNDLE_CACHE.get(node_id)
        if b is not None:
            # Refresh LRU.
            _BUNDLE_CACHE.pop(node_id, None)
            _BUNDLE_CACHE[node_id] = b
        return b


def cache_clear() -> None:
    with _BUNDLE_CACHE_LOCK:
        _BUNDLE_CACHE.clear()


def cache_size() -> int:
    with _BUNDLE_CACHE_LOCK:
        return len(_BUNDLE_CACHE)


# ----------------------------------------------------------------------
# Underlay-frame cache (node_id → uint8 [N,H,W,3]) — the INPUT frames the
# editor draws behind the pose, frame-by-frame. Downscaled + frame-capped
# by the node before caching so this stays small. Served lazily one frame
# at a time via GET /c2c/fc3d_underlay (no giant payloads, scales to long
# shots). A single image batch → one frame → constant backdrop.
# ----------------------------------------------------------------------
_UNDERLAY_CACHE_MAX = 6
_UNDERLAY_CACHE: "dict[str, np.ndarray]" = {}
_UNDERLAY_CACHE_LOCK = threading.Lock()


def underlay_put(node_id: str, frames_u8, idxs=None) -> None:
    """Stash downscaled uint8 input frames [N,H,W,3] + their original frame
    indices (for decimated long shots) for a node (LRU-capped)."""
    if not node_id or frames_u8 is None:
        return
    n = int(getattr(frames_u8, "shape", [0])[0])
    if idxs is None:
        idxs = list(range(n))
    with _UNDERLAY_CACHE_LOCK:
        _UNDERLAY_CACHE.pop(node_id, None)
        _UNDERLAY_CACHE[node_id] = {"frames": frames_u8, "idxs": list(idxs)}
        while len(_UNDERLAY_CACHE) > _UNDERLAY_CACHE_MAX:
            _UNDERLAY_CACHE.pop(next(iter(_UNDERLAY_CACHE)), None)


def underlay_get(node_id: str):
    with _UNDERLAY_CACHE_LOCK:
        a = _UNDERLAY_CACHE.get(node_id)
        if a is not None:           # refresh LRU
            _UNDERLAY_CACHE.pop(node_id, None)
            _UNDERLAY_CACHE[node_id] = a
        return a


# ----------------------------------------------------------------------
# Whitelist of run() kwargs that the route accepts. We deliberately
# enumerate them so junk fields in the POST body cannot reach run().
# ----------------------------------------------------------------------
_FLOAT_KWARGS = (
    "expression_strength", "expression_clamp",
    "head_yaw_deg", "head_pitch_deg", "head_roll_deg",
    "head_tx", "head_ty", "head_tz",
    "head_scale", "jaw_rot_deg", "neck_yaw_deg", "neck_pitch_deg",
    "gaze_yaw_deg", "gaze_pitch_deg",
    "blend_strength",
)
_BOOL_KWARGS = ("blend_mouth", "blend_brows", "blend_eyes", "blend_jaw")
_STR_KWARGS = (
    "expression_coeffs_json", "head_pose_json", "gaze_json",
    "use_metas", "propagate_expression", "propagate_head", "propagate_gaze",
    "landmark_overrides_json", "pose_overrides_json", "gaze_overrides_json",
)


def _coerce_payload(body: dict) -> Dict[str, Any]:
    """Return only the well-typed kwargs the node's ``run()`` accepts."""
    out: Dict[str, Any] = {}
    for k in _FLOAT_KWARGS:
        if k in body:
            try:
                out[k] = float(body[k])
            except (TypeError, ValueError):
                pass
    for k in _BOOL_KWARGS:
        if k in body:
            out[k] = bool(body[k])
    for k in _STR_KWARGS:
        if k in body and body[k] is not None:
            out[k] = str(body[k])
    return out


# ----------------------------------------------------------------------
# Per-frame extraction from the bundle produced by run()
# ----------------------------------------------------------------------
def _extract_frame_overlay(out_bundle: dict, frame_idx: int) -> Dict[str, Any]:
    """Pull face/body/gaze info for one frame from a freshly-edited bundle."""
    # Lazy imports to avoid a circular dependency at module load time.
    from .expression_3d_coeffs import _read_face_normalised, _N_LM
    from ._face_helpers import _get_body_kps

    metas = out_bundle.get("pose_metas") or []
    if not (0 <= frame_idx < len(metas)):
        return {"face": None, "body": None, "gaze": None}
    meta = metas[frame_idx]

    # Face: 68 landmarks → face-bbox-normalised [0..1] to match the
    # overlay_meta convention used by face_controller_3d.js.
    face_norm = None
    eye_l_norm = eye_r_norm = None
    xy = _read_face_normalised(meta)
    if xy is not None and xy.shape == (_N_LM, 2):
        mn = xy.min(axis=0); mx = xy.max(axis=0)
        bw = max(float(mx[0] - mn[0]), 1e-6)
        bh = max(float(mx[1] - mn[1]), 1e-6)
        fn = np.empty_like(xy)
        fn[:, 0] = (xy[:, 0] - mn[0]) / bw
        fn[:, 1] = (xy[:, 1] - mn[1]) / bh
        face_norm = [[round(float(fn[i, 0]), 5), round(float(fn[i, 1]), 5)]
                     for i in range(_N_LM)]
        eye_r_norm = [round(float(fn[36:42, 0].mean()), 5),
                      round(float(fn[36:42, 1].mean()), 5)]
        eye_l_norm = [round(float(fn[42:48, 0].mean()), 5),
                      round(float(fn[42:48, 1].mean()), 5)]

    # Body: 18 keypoints image-normalised (or None per joint if NaN).
    body_kps_out = None
    body_xy = _get_body_kps(meta)
    if body_xy is not None and body_xy.shape[0] >= 18:
        body_kps_out = []
        for j in range(18):
            x, y = float(body_xy[j, 0]), float(body_xy[j, 1])
            if math.isnan(x) or math.isnan(y):
                body_kps_out.append(None)
            else:
                body_kps_out.append([round(x, 5), round(y, 5)])

    # Gaze: pull the per-eye yaw/pitch radians from iris_data if present.
    iris_data = out_bundle.get("iris_data") or []
    gaze_out: Dict[str, Any] = {
        "l_eye_norm": eye_l_norm, "r_eye_norm": eye_r_norm,
        "l_gaze": None, "r_gaze": None,
    }
    if 0 <= frame_idx < len(iris_data) and isinstance(iris_data[frame_idx], dict):
        cur = iris_data[frame_idx]
        for side, key in (("l", "left_gaze"), ("r", "right_gaze")):
            g = cur.get(key)
            if isinstance(g, dict):
                try:
                    gaze_out[f"{side}_gaze"] = [
                        round(float(g.get("yaw_rad", 0.0)), 5),
                        round(float(g.get("pitch_rad", 0.0)), 5),
                    ]
                except (TypeError, ValueError):
                    pass

    return {"face": face_norm, "body": body_kps_out, "gaze": gaze_out}


# ----------------------------------------------------------------------
# Route registration
# ----------------------------------------------------------------------
_REGISTERED = False


def register_routes() -> bool:
    """Attach ``POST /c2c/fc3d_preview`` to the running ComfyUI server.

    Returns ``True`` on success, ``False`` if the server is not yet
    started or aiohttp is missing. Idempotent — safe to call multiple
    times; only the first call attaches the route.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import server as _comfy_server  # ComfyUI's server module
        from aiohttp import web
    except Exception as exc:                                            # noqa: BLE001
        msg = str(exc)
        # Headless smoke / pytest imports this module outside a live ComfyUI process.
        if "utils.install_util" in msg or "No module named 'server'" in msg:
            log.debug("[fc3d_preview] route registration skipped (offline): %s", exc)
        else:
            log.warning("[fc3d_preview] aiohttp / ComfyUI server unavailable: %s", exc)
        return False

    ps = getattr(_comfy_server, "PromptServer", None)
    inst = getattr(ps, "instance", None) if ps is not None else None
    if inst is None or not hasattr(inst, "routes"):
        log.warning("[fc3d_preview] PromptServer.instance not ready; will retry")
        return False

    routes = inst.routes

    @routes.post("/c2c/fc3d_preview")
    async def _preview(req):                                            # noqa: ANN001
        try:
            body = await req.json()
        except Exception as exc:                                        # noqa: BLE001
            return web.json_response(
                {"error": "bad_json", "message": str(exc)}, status=400,
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"error": "bad_json", "message": "body must be a JSON object"},
                status=400,
            )

        node_id = str(body.get("node_id", "")).strip()
        if not node_id:
            return web.json_response(
                {"error": "missing_node_id"}, status=400,
            )

        cached = cache_get(node_id)
        if cached is None:
            return web.json_response(
                {"error": "cache_miss",
                 "message": "Run the WanFaceController3DV2 node once "
                            "to populate the live-preview cache."},
                status=412,
            )

        try:
            frame_idx = int(body.get("frame_idx", 0))
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "bad_frame_idx"}, status=400,
            )

        kwargs = _coerce_payload(body)
        kwargs["frame_start"] = frame_idx
        kwargs["frame_end"] = frame_idx

        # Lazy import to avoid a circular dependency at module load.
        from .face_controller_3d import WanFaceController3DV2

        try:
            node = WanFaceController3DV2()
            # run() returns {"ui": ..., "result": (bundle, info, ts_json)}
            result = node.run(cached, **kwargs)
            if not isinstance(result, dict) or "result" not in result:
                raise RuntimeError("run() returned an unexpected envelope")
            res = result["result"]
            if not isinstance(res, tuple) or len(res) < 2:
                raise RuntimeError("run() result tuple too short")
            out_bundle, info = res[0], res[1]
        except Exception as exc:                                        # noqa: BLE001
            log.exception("[fc3d_preview] run() failed for node %s frame %d",
                          node_id, frame_idx)
            return web.json_response(
                {"error": "run_failed", "message": str(exc)}, status=500,
            )

        per_frame = _extract_frame_overlay(out_bundle, frame_idx)

        # Optional rendered preview (Face-Director plan P5): when the JS panel
        # asks for `return_image`, render the edited frame through the same
        # overlay renderer used by run()'s preview_image output and inline it
        # as base64 PNG so the node body can show a live WYSIWYG thumbnail.
        image_b64 = None
        if body.get("return_image"):
            try:
                import base64
                import io as _io

                from PIL import Image as _PILImage

                from . import _face_overlay_render as _r
                from ._face_helpers import _get_body_kps
                from .expression_3d_coeffs import _read_face_normalised
                from .face_controller_3d import _POSE18_EDGES_UI

                metas = out_bundle.get("pose_metas") or []
                if 0 <= frame_idx < len(metas):
                    meta = metas[frame_idx]
                    iris = out_bundle.get("iris_data") or []
                    iris_entry = (
                        iris[frame_idx]
                        if frame_idx < len(iris) and isinstance(iris[frame_idx], dict)
                        else None
                    )
                    size = max(96, min(512, int(body.get("preview_size", 256))))
                    frame_u8 = _r.render_overlay_frame(
                        meta=meta,
                        iris_entry=iris_entry,
                        body_xy=_get_body_kps(meta),
                        out_size=size,
                        edges=_POSE18_EDGES_UI,
                        face_norm=_read_face_normalised(meta),
                        frame_idx=frame_idx,
                        n_frames=len(metas),
                    )
                    buf = _io.BytesIO()
                    _PILImage.fromarray(frame_u8).save(buf, format="PNG")
                    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as exc:                                    # noqa: BLE001
                log.debug("[fc3d_preview] image render skipped: %s", exc)

        return web.json_response({
            "ok": True,
            "node_id": node_id,
            "frame_idx": frame_idx,
            "info": str(info),
            "face_norm": per_frame["face"],
            "body_kps":  per_frame["body"],
            "gaze":      per_frame["gaze"],
            "image_b64": image_b64,
        })

    @routes.get("/c2c/fc3d_preview/status")
    async def _status(_req):                                            # noqa: ANN001
        return web.json_response({
            "ok": True,
            "cached_nodes": cache_size(),
            "max": _BUNDLE_CACHE_MAX,
        })

    @routes.get("/c2c/fc3d_underlay")
    async def _underlay(req):                                           # noqa: ANN001
        """Serve one cached INPUT frame as PNG for the editor's backdrop.

        Query: node_id, frame (clamped to range). The editor draws this
        behind the pose so you see input + pose aligned, frame-by-frame.
        Cache miss → 412 (queue the node once).
        """
        node_id = str(req.query.get("node_id", "")).strip()
        data = underlay_get(node_id) if node_id else None
        frames = data["frames"] if data else None
        if frames is None or len(frames) == 0:
            return web.json_response(
                {"error": "no_underlay",
                 "message": "Wire `images` and queue the node once."},
                status=412,
            )
        try:
            want = int(req.query.get("frame", 0))
        except (TypeError, ValueError):
            want = 0
        # Map the requested ORIGINAL frame index to the nearest cached frame
        # (cache may be decimated for long shots). Single image → always 0.
        idxs = data.get("idxs") or list(range(len(frames)))
        f = min(range(len(idxs)), key=lambda k: abs(idxs[k] - want))
        try:
            import io as _io

            from PIL import Image as _PILImage
            buf = _io.BytesIO()
            _PILImage.fromarray(frames[f]).save(buf, format="PNG")
            return web.Response(body=buf.getvalue(), content_type="image/png",
                                headers={"Cache-Control": "no-store"})
        except Exception as exc:                                        # noqa: BLE001
            return web.json_response(
                {"error": "encode_failed", "message": str(exc)}, status=500,
            )

    @routes.post("/c2c/fc3d_cache_seed")
    async def _cache_seed(req):                                         # noqa: ANN001
        """Seed the live-preview LRU cache (E2E / dev tooling).

        Accepts ``{"node_id": "<graph id>", "pose_data": {<POSEDATA dict>}}``.
        The bundle must look like a real POSEDATA payload (``pose_metas`` key).
        """
        try:
            body = await req.json()
        except Exception as exc:                                        # noqa: BLE001
            return web.json_response(
                {"error": "bad_json", "message": str(exc)}, status=400,
            )
        if not isinstance(body, dict):
            return web.json_response(
                {"error": "bad_json", "message": "body must be a JSON object"},
                status=400,
            )
        node_id = str(body.get("node_id", "")).strip()
        pose = body.get("pose_data")
        if not node_id:
            return web.json_response({"error": "missing_node_id"}, status=400)
        if not isinstance(pose, dict) or not (
            pose.get("pose_metas") or pose.get("pose_metas_original")
        ):
            return web.json_response(
                {"error": "bad_pose_data",
                 "message": "pose_data must be a POSEDATA dict with pose_metas"},
                status=400,
            )
        cache_put(node_id, pose)
        return web.json_response({
            "ok": True,
            "node_id": node_id,
            "cached_nodes": cache_size(),
        })

    _REGISTERED = True
    log.info("[fc3d_preview] routes registered (POST /c2c/fc3d_preview, "
             "POST /c2c/fc3d_cache_seed, GET /c2c/fc3d_preview/status)")
    return True


def try_register_routes_deferred() -> None:
    """Attempt registration now; if the server isn't ready, retry on a
    background thread until it is (or 30 s elapse)."""
    if register_routes():
        return
    # Offline import (smoke tests) — do not spin a retry thread.
    try:
        import server as _comfy_server  # noqa: F401
    except Exception:
        return

    def _retry():
        import time
        for _ in range(30):
            time.sleep(1.0)
            if register_routes():
                return
        log.warning("[fc3d_preview] gave up after 30s waiting for PromptServer")

    t = threading.Thread(target=_retry, daemon=True, name="fc3d-preview-retry")
    t.start()
