"""Geometric (measured-iris) eye-in-head gaze — the deterministic engine.

W7-G1 of the gaze program. Instead of ESTIMATING gaze from appearance with a
neural network (L2CS ≈ 10° MAE, ETH-XGaze ≈ 4–5°), this engine MEASURES it:
MediaPipe's attention-mesh iris centre is ~1–2 px accurate, so the iris offset
inside the eye aperture (corner-to-corner, lid-to-lid) directly gives the
eye-in-head rotation. Composition with the solvePnP head pose + the Kalman
smoother is reused from the existing `blendshape_head_corrected` machinery in
nodes.py / gaze_3d.py — this module only supplies a better eye-in-head source.

Why this is the "real thing" for animation retargeting: Wan Animate needs the
character's eyeballs to move exactly like the source performer's — i.e. the
iris position within the socket — not a world-space gaze ray guess. Measuring
that position removes the NN's per-person appearance bias; the remaining
systematic offset is zeroed by the per-shot calibration (W7-G2).

Sign convention (matches gaze_3d.world_gaze_from_eye_in_head input):
    yaw   > 0  = subject looking to THEIR right  (iris moves toward image LEFT)
    pitch > 0  = subject looking UP              (iris moves toward image TOP)

Pure numpy; no models, no downloads. Apache-2.0, C2C/MEC.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# MediaPipe 478-landmark indices (FaceMesh attention mesh / FaceLandmarker).
_IDX = {
    "right": {"iris": 468, "outer": 33, "inner": 133, "lid_top": 159, "lid_bot": 145},
    "left":  {"iris": 473, "inner": 362, "outer": 263, "lid_top": 386, "lid_bot": 374},
}

# Max eyeball rotation mapped to the iris reaching the aperture edge.
# Anthropometric: the visible aperture half-width corresponds to roughly
# 40° of horizontal eyeball rotation and ~25° vertical. These are GAINS,
# not hard physics — the W7-G2 per-shot calibration removes residual bias,
# and the composition/Kalman downstream are gain-agnostic.
MAX_EYE_YAW_RAD = math.radians(40.0)
MAX_EYE_PITCH_RAD = math.radians(25.0)

# Below this fraction of the horizontal half-width, the lid aperture is
# treated as a blink/occlusion → measurement rejected (caller falls back).
_MIN_LID_FRACTION = 0.10

# Per-shot calibration offsets (W7-G2), settable via calibrate()/reset().
_CALIB: Dict[str, Dict[str, float]] = {
    "right": {"yaw": 0.0, "pitch": 0.0},
    "left":  {"yaw": 0.0, "pitch": 0.0},
}


def is_available() -> bool:
    return True


def eye_in_head_from_iris(mp_result: Dict[str, Any], eye: str) -> Optional[Dict[str, float]]:
    """Measured eye-in-head gaze for one eye.

    Args:
        mp_result: the dict from _run_mediapipe_on_face_crop /
            _run_face_landmarker_on_face_crop. Uses `landmarks_px_full`
            (precise lids) when present, else the corner keys.
        eye: "right" | "left".

    Returns:
        {'yaw_rad','pitch_rad','blink','confidence'} or None when the eye is
        closed/occluded or landmarks are missing (caller should fall back).
    """
    if not isinstance(mp_result, dict):
        return None
    idx = _IDX[eye]

    lm = mp_result.get("landmarks_px_full")
    if lm is not None and len(lm) >= 478:
        ix, iy = float(lm[idx["iris"]][0]), float(lm[idx["iris"]][1])
        ox, oy = float(lm[idx["outer"]][0]), float(lm[idx["outer"]][1])
        nx, ny = float(lm[idx["inner"]][0]), float(lm[idx["inner"]][1])
        ty = float(lm[idx["lid_top"]][1])
        by = float(lm[idx["lid_bot"]][1])
        lid_cy = 0.5 * (ty + by)
        hv = max(0.5 * abs(by - ty), 1e-6)
    else:
        # Corner-only fallback (older dict shapes): approximate the vertical
        # frame from the horizontal span (open-eye aspect ≈ 0.36).
        try:
            ix, iy = mp_result[f"{eye}_iris_px"]
            ox, oy = mp_result[f"{eye}_eye_outer_px"]
            nx, ny = mp_result[f"{eye}_eye_inner_px"]
        except Exception:
            return None
        lid_cy = 0.5 * (oy + ny)
        hv = max(0.36 * 0.5 * abs(nx - ox), 1e-6)
        ty = lid_cy - hv
        by = lid_cy + hv

    cx = 0.5 * (ox + nx)
    hw = max(0.5 * abs(nx - ox), 1e-6)

    # Blink / occlusion gate: aperture too small for a trustworthy vertical
    # measurement. (The lids track gaze, so a blink corrupts BOTH axes.)
    blink = float(max(0.0, 1.0 - (hv / (0.36 * hw + 1e-6))))
    if hv < _MIN_LID_FRACTION * hw:
        return None

    # Normalised iris offsets inside the aperture frame, image convention
    # (+x right, +y down), clamped to the physical range.
    rh = max(-1.0, min(1.0, (ix - cx) / hw))
    rv = max(-1.0, min(1.0, (iy - lid_cy) / hv))

    # Image-space → anatomical: iris toward image-right = subject's LEFT
    # (yaw < 0); iris downward = looking DOWN (pitch < 0).
    yaw = -rh * MAX_EYE_YAW_RAD - _CALIB[eye]["yaw"]
    pitch = -rv * MAX_EYE_PITCH_RAD - _CALIB[eye]["pitch"]

    # HYBRID VERTICAL (2026-07-24): the aperture-relative vertical measure
    # systematically UNDER-reads true vertical eyeball rotation, because the
    # lids travel WITH vertical gaze (upper lid lifts when looking up, drops
    # when looking down) — the iris barely moves relative to the lid frame
    # even during a large vertical saccade. Confirmed live on an eyes-up-
    # under-brows portrait: measured rv stayed near 0 while the blendshape
    # engine (appearance-trained eyeLookUp/Down, no such failure mode) read
    # the up-rotation correctly; composing the under-read vertical with the
    # head-down pose then produced a large spurious "down" world gaze.
    # Horizontal has no such problem (the canthi do not move with gaze), so:
    # keep the MEASURED yaw, take pitch from the blendshape coefficients
    # whenever they are available, and keep the aperture-measured pitch only
    # as the no-blendshape fallback (FaceMesh path).
    gaze_bs = mp_result.get("gaze_blendshape")
    if isinstance(gaze_bs, dict) and isinstance(gaze_bs.get(eye), dict):
        bs_pitch = gaze_bs[eye].get("pitch_rad")
        if bs_pitch is not None and math.isfinite(float(bs_pitch)):
            pitch = float(bs_pitch) - _CALIB[eye]["pitch"]

    # Confidence: high when the aperture is wide open and the iris radius is
    # sane; degrade toward the blink gate.
    conf = float(max(0.0, min(1.0, 1.0 - blink)))
    return {"yaw_rad": float(yaw), "pitch_rad": float(pitch),
            "blink": blink, "confidence": conf}


# ── W7-G2: per-shot calibration ───────────────────────────────────────────

def calibrate_from_measurement(measure_right: Optional[Dict[str, float]],
                               measure_left: Optional[Dict[str, float]]) -> None:
    """Zero the engine on a reference frame where the subject looks at the
    camera: whatever eye-in-head angles are measured there become the new
    origin. Removes per-person eye-shape/aperture bias for the whole shot."""
    for eye, m in (("right", measure_right), ("left", measure_left)):
        if m is None:
            continue
        _CALIB[eye]["yaw"] += float(m.get("yaw_rad", 0.0))
        _CALIB[eye]["pitch"] += float(m.get("pitch_rad", 0.0))


def reset_calibration() -> None:
    for eye in ("right", "left"):
        _CALIB[eye]["yaw"] = 0.0
        _CALIB[eye]["pitch"] = 0.0


def get_calibration() -> Dict[str, Dict[str, float]]:
    return {k: dict(v) for k, v in _CALIB.items()}
