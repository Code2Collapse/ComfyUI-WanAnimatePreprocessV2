"""WanFaceController3DV2 — single unified face controller.

This is the consolidated "face expression + 3D + everything" node the
user explicitly asked for. It subsumes:

  * WanExpression3DCoeffsV2          — 12 FACS-inspired expression dials.
  * 3-DoF head rotation              — yaw / pitch / roll applied to the
                                       iBUG-68 landmarks via a hardcoded
                                       canonical-face depth map (so they
                                       rotate as if they were 3D points).
  * Gaze offset                      — per-frame yaw / pitch applied to
                                       iris / pupil keypoints stored in
                                       the optional `iris_data` field.
  * Optional reference-shape blend   — region-wise (mouth / brows / eyes
                                       / jaw) linear blend toward a
                                       single-frame reference POSEDATA,
                                       same convention as the legacy
                                       WanFaceExpressionEditorV2.

All math is local, clean-room, and CPU-only. No EMOCA / DECA / FLAME /
LivePortrait code or weights — only the public FACS method and the
public iBUG-68 anatomy.

Pipeline order (per frame, when an override exists)
===================================================
    landmarks_in  ──► (1) reference blend (regions)
                  ──► (2) expression coeffs (FACS basis)
                  ──► (3) 3-DoF head rotation (canonical depth + R)
                  ──► (4) gaze offset (iris_data only)
                  ──► landmarks_out

Each stage is independent: leaving its controls at their defaults is a
no-op for that stage, so the node degrades gracefully to whichever
subset of features the user actually wants.

JSON schemas
============
expression_coeffs_json::

    {
      "frames":  {"4": {"smile": 0.7, "eye_close_L": 1.0}, ...},
      "ranges":  [{"start": 0, "end": 24, "coeffs": {"brow_furrow": 0.3}}]
    }

head_pose_json::

    {
      "frames": {"4": {"yaw": 8.0, "pitch": -3.0, "roll": 2.0}, ...},
      "ranges": [{"start": 0, "end": 47,
                   "pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}}]
    }
    # Angles in degrees. Sign convention:
    #   yaw   +  → subject looks to their LEFT  (image right).
    #   pitch +  → subject looks UP.
    #   roll  +  → head tilts to subject's LEFT (image clockwise).

gaze_json::

    {
      "frames": {"4": {"yaw": 5.0, "pitch": -2.0}, ...},
      "ranges": [...]
    }
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np

from .expression_3d_coeffs import (
    _AXIS_NAMES,
    _BASIS,
    _N_AXES,
    _N_LM,
    _expand_range_overrides,
    _face_bbox_size,
    _merge_overrides,
    _parse_coeffs_json,
    _read_face_normalised,
    _write_face_normalised,
)
from ._face_helpers import (
    _parse_overrides as _parse_face_overrides_ui,
    _parse_gaze_overrides as _parse_gaze_overrides_ui,
    _apply_gaze_override_to_iris_entry,
    _get_body_kps,
    _set_body_kps,
)

log = logging.getLogger(__name__)

# OpenPose-18 topology — shared with the JS overlay (face_controller_3d.js).
_POSE18_JOINT_NAMES_UI: Tuple[str, ...] = (
    "nose", "neck",
    "rshoulder", "relbow", "rwrist",
    "lshoulder", "lelbow", "lwrist",
    "rhip", "rknee", "rankle",
    "lhip", "lknee", "lankle",
    "reye", "leye", "rear", "lear",
)
_POSE18_EDGES_UI: Tuple[Tuple[int, int], ...] = (
    (1, 2), (1, 5),
    (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (1, 0),
    (0, 14), (14, 16),
    (0, 15), (15, 17),
)
_GAZE_MAX_YAW_RAD   = math.radians(30.0)
_GAZE_MAX_PITCH_RAD = math.radians(25.0)


def _parse_pose_overrides_ui(blob: Optional[str]) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Parse pose_overrides_json from the JS overlay.

    Shape: ``{"frames": {"<idx>": {"<joint_idx>": [x_norm, y_norm]}}}``.
    Coordinates are IMAGE-normalised [0,1] (NOT face-bbox-normalised).
    Returns ``{}`` on empty / invalid.
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("pose_overrides_json ignored — not valid JSON: %s", e)
        return {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for f_key, joint_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(joint_map, dict):
            continue
        per_frame: Dict[int, Tuple[float, float]] = {}
        for j_key, xy in joint_map.items():
            try:
                j_idx = int(j_key)
            except (TypeError, ValueError):
                continue
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                continue
            try:
                per_frame[j_idx] = (float(xy[0]), float(xy[1]))
            except (TypeError, ValueError):
                continue
        if per_frame:
            out[f_idx] = per_frame
    return out


def _parse_range_deltas(
    blob: Optional[str], n_frames: int,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Parse the ``ranges`` section of a face/pose override blob.

    Shape (additive over the per-frame ``frames`` section)::

        {"ranges": [{"start": 0, "end": -1,
                     "delta": {"<idx>": [dx_norm, dy_norm]}}, ...]}

    This is the contract the JS canvas overlay produces when the user
    edits one frame with "Δ propagate" enabled: a single drag yields a
    *delta* that is broadcast across every frame in ``[start, end]``.
    ``end < 0`` means "to the last frame" (resolved against ``n_frames``).

    Returns ``{frame_idx: {idx: (dx, dy)}}`` — the accumulated delta for
    every covered frame.  Per-frame absolute overrides (the ``frames``
    section) are applied *after* these and therefore win on conflicts.
    Coordinates are in the SAME normalised space as the per-frame section
    (face: bbox-normalised, pose: image-normalised).
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        return {}
    ranges = data.get("ranges") if isinstance(data, dict) else None
    if not isinstance(ranges, list):
        return {}
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for entry in ranges:
        if not isinstance(entry, dict):
            continue
        try:
            s = int(entry.get("start", 0))
            e = int(entry.get("end", -1))
        except (TypeError, ValueError):
            continue
        if e < 0:
            e = n_frames - 1
        s = max(0, s)
        e = min(n_frames - 1, e) if n_frames > 0 else e
        if e < s:
            continue
        delta_raw = entry.get("delta")
        if not isinstance(delta_raw, dict):
            continue
        delta: Dict[int, Tuple[float, float]] = {}
        for k, v in delta_raw.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                continue
            try:
                delta[idx] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue
        if not delta:
            continue
        for f in range(s, e + 1):
            acc = out.setdefault(f, {})
            for idx, (dx, dy) in delta.items():
                px, py = acc.get(idx, (0.0, 0.0))
                acc[idx] = (px + dx, py + dy)
    return out


# ----------------------------------------------------------------------
# Canonical iBUG-68 depth (z) map.  Public-knowledge generic model:
# values approximate the *relative* depth of each landmark on a frontal
# human face in face-bbox-normalised units (face_width ≈ 1.0).
# Convention: +z = AWAY from camera (deep), -z = forward (closer).
# Forward features (nose tip, lips) are negative; deep features (ears,
# outer jaw) are positive.  Numbers are intentionally smooth and small
# so the rotation behaviour stays stable across degraded landmarks.
# ----------------------------------------------------------------------
_CANONICAL_Z = np.zeros(_N_LM, dtype=np.float32)
# Jawline 0..16 — outermost (ears) at +0.18, chin at -0.04 (chin pokes
# forward of the ear plane).
_JAW_Z = [
    +0.18, +0.16, +0.13, +0.10, +0.06, +0.02, -0.02, -0.03,
    -0.04,
    -0.03, -0.02, +0.02, +0.06, +0.10, +0.13, +0.16, +0.18,
]
for i, z in enumerate(_JAW_Z):
    _CANONICAL_Z[i] = z
# Right brow 17..21 — slight forward arch from -0.04 (outer) to -0.06 (inner).
for i, z in enumerate([-0.04, -0.05, -0.06, -0.06, -0.05]):
    _CANONICAL_Z[17 + i] = z
# Left brow 22..26 — mirror.
for i, z in enumerate([-0.05, -0.06, -0.06, -0.05, -0.04]):
    _CANONICAL_Z[22 + i] = z
# Nose bridge 27..30 — bridge top to tip, tip is the most forward point.
for i, z in enumerate([-0.04, -0.07, -0.10, -0.13]):
    _CANONICAL_Z[27 + i] = z
# Nose base 31..35 — alar wings to tip and back.
for i, z in enumerate([-0.06, -0.09, -0.12, -0.09, -0.06]):
    _CANONICAL_Z[31 + i] = z
# Right eye outline 36..41 — outer corner slightly deeper.
for i, z in enumerate([+0.01, -0.01, -0.02, -0.03, -0.02, -0.01]):
    _CANONICAL_Z[36 + i] = z
# Left eye outline 42..47 — mirror.
for i, z in enumerate([-0.03, -0.02, -0.01, +0.01, -0.01, -0.02]):
    _CANONICAL_Z[42 + i] = z
# Mouth outer 48..59 — corners slightly forward of cheeks; centre most forward.
for i, z in enumerate([
    -0.05, -0.06, -0.07, -0.08, -0.07, -0.06,
    -0.05, -0.06, -0.07, -0.08, -0.07, -0.06,
]):
    _CANONICAL_Z[48 + i] = z
# Mouth inner 60..67.
for i, z in enumerate([
    -0.06, -0.07, -0.08, -0.07, -0.06, -0.07, -0.08, -0.07,
]):
    _CANONICAL_Z[60 + i] = z


# ----------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------
def _rot_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from intrinsic Euler angles (deg).

    Order: R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    Sign convention (image with +X right, +Y down, +Z away):
        yaw +  → subject rotates head to their LEFT (their face turns to
                 image right).  Implemented as Ry with sin=+sin.
        pitch+ → subject looks UP (chin lifts; landmarks rotate so Y of
                 the chin DECREASES).  Implemented as Rx with sin=-sin
                 so that +pitch yields chin-up in image space.
        roll + → head tilts to subject's LEFT (image clockwise).
                 Implemented as Rz with sin=+sin.
    """
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    # Note the -sp on Rx to get "+pitch ⇒ chin up in image".
    Rx = np.array([[1, 0, 0],
                   [0, cp, sp],
                   [0, -sp, cp]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cr, -sr, 0],
                   [sr, cr, 0],
                   [0, 0, 1]], dtype=np.float32)
    return Rz @ Ry @ Rx


def _apply_head_translation(
    xy_norm: np.ndarray,
    tx_face: float,
    ty_face: float,
    tz_scale: float,
) -> np.ndarray:
    """Translate + depth-scale the 68 landmarks rigidly.

    tx_face / ty_face : translation in FACE-BBOX-width units. +tx moves
        the face to image right; +ty moves it down. Typical user range
        ±0.5 (half a face width).
    tz_scale          : signed depth in face-bbox-width units. The face
        is uniformly scaled around its own centroid by ``1 / (1 + tz)``;
        +tz pushes the face away (smaller), -tz brings it forward
        (larger). Clamped so the divisor cannot reach zero or flip sign.

    All three are no-ops at 0.0 (fast path).
    """
    if abs(tx_face) < 1e-4 and abs(ty_face) < 1e-4 and abs(tz_scale) < 1e-4:
        return xy_norm
    bw, bh = _face_bbox_size(xy_norm)
    s = max(bw, bh)
    cx = float(xy_norm[:, 0].mean())
    cy = float(xy_norm[:, 1].mean())
    # Depth scale around centroid. Clamp tz so (1+tz) stays in [0.25, 4.0]
    # i.e. zoom factor in [0.25x, 4.0x] — beyond that the face is unusable.
    tz_clamped = float(np.clip(tz_scale, -0.75, 3.0))
    zoom = 1.0 / (1.0 + tz_clamped)
    out = xy_norm.copy()
    if abs(zoom - 1.0) > 1e-4:
        out[:, 0] = (out[:, 0] - cx) * zoom + cx
        out[:, 1] = (out[:, 1] - cy) * zoom + cy
    # Translation last, in IMAGE-normalised units (face-bbox * face_width).
    out[:, 0] = out[:, 0] + tx_face * s
    out[:, 1] = out[:, 1] + ty_face * s
    return out


def _propagate_head_overrides(
    base: Dict[int, Dict[str, float]],
    n_frames: int,
    mode: str,
    keys: Tuple[str, ...],
) -> Dict[int, Dict[str, float]]:
    """Spread sparse per-frame head/gaze pins across the whole timeline.

    base    : ``{frame_idx: {"yaw": ..., ...}}`` from the JSON parser
              (already merged with range-broadcast entries).
    n_frames: total number of frames in the clip.
    mode    : one of:
        ``"off"``               — return ``base`` unchanged.
        ``"hold_last"``         — every gap holds the previous pin's
                                   value (step-function). Before the
                                   first pin: flat-extrapolated to that
                                   pin's value (no implicit gap).
                                   Useful for discrete pose changes.
        ``"interpolate"``       — linearly interpolate each key between
                                   adjacent pins; flat-extrapolate at
                                   both ends. Smooth, DAW-style automation.
        ``"broadcast_first"``   — apply the first pin's values to every
                                   frame that has no explicit pin.

    Per-frame pins in ``base`` always win on output; only gaps are
    filled. Keys missing from a pin are left unfilled (so two keys can
    have independent timelines).
    """
    if mode == "off" or not base or n_frames <= 0:
        return base
    if mode not in ("hold_last", "interpolate", "broadcast_first"):
        return base

    # Per-key sorted pin list.
    per_key_pins: Dict[str, list[Tuple[int, float]]] = {k: [] for k in keys}
    for f_idx in sorted(base.keys()):
        if not (0 <= f_idx < n_frames):
            continue
        entry = base[f_idx]
        if not isinstance(entry, dict):
            continue
        for k in keys:
            if k in entry:
                try:
                    per_key_pins[k].append((f_idx, float(entry[k])))
                except (TypeError, ValueError):
                    pass

    # Walk every frame, compute the propagated value per key.
    out: Dict[int, Dict[str, float]] = {f: dict(v) for f, v in base.items() if isinstance(v, dict)}
    for k in keys:
        pins = per_key_pins[k]
        if not pins:
            continue
        if mode == "broadcast_first":
            v0 = pins[0][1]
            for f in range(n_frames):
                slot = out.setdefault(f, {})
                slot.setdefault(k, v0)
            continue
        # hold_last and interpolate share the same pin-walk skeleton.
        for f in range(n_frames):
            # Find left/right pins.
            left = None
            right = None
            for p in pins:
                if p[0] <= f:
                    left = p
                elif right is None:
                    right = p
                    break
            slot = out.setdefault(f, {})
            if k in slot:
                continue  # explicit pin — leave it alone.
            if left is None and right is None:
                continue
            if left is None:
                # Before any pin — flat at the first pin.
                slot[k] = right[1]
                continue
            if right is None or mode == "hold_last":
                slot[k] = left[1]
                continue
            # Linear interpolation between adjacent pins.
            span = right[0] - left[0]
            if span <= 0:
                slot[k] = left[1]
            else:
                t = (f - left[0]) / span
                slot[k] = left[1] + (right[1] - left[1]) * t
    return out


def _apply_head_rotation(
    xy_norm: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """Rotate the 68 landmarks in 3D using the canonical depth model.

    xy_norm  : (68, 2)  image-normalised coords (already in [0..1]).
    Returns  : (68, 2)  image-normalised coords after rotation, with the
                        face centroid + bbox-scale preserved.

    Steps:
      1. Translate to the face centroid so rotation is about the face.
      2. Scale by 1/face_width so x/y/z share the same unit.
      3. Add canonical z (also in face-width units).
      4. Apply R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
      5. Orthographic projection (drop new z).
      6. Undo scale and translation.
    """
    if abs(yaw_deg) < 1e-3 and abs(pitch_deg) < 1e-3 and abs(roll_deg) < 1e-3:
        return xy_norm  # no-op fast path
    bw, bh = _face_bbox_size(xy_norm)
    # Use the longer side so we don't squish under extreme aspect ratios.
    s = max(bw, bh)
    cx = float(xy_norm[:, 0].mean())
    cy = float(xy_norm[:, 1].mean())
    local_xy = (xy_norm - np.array([[cx, cy]], dtype=np.float32)) / s
    pts3d = np.empty((_N_LM, 3), dtype=np.float32)
    pts3d[:, 0] = local_xy[:, 0]
    pts3d[:, 1] = local_xy[:, 1]
    pts3d[:, 2] = _CANONICAL_Z
    R = _rot_matrix(yaw_deg, pitch_deg, roll_deg)
    rot = pts3d @ R.T  # (68, 3)
    out = np.empty_like(xy_norm)
    out[:, 0] = rot[:, 0] * s + cx
    out[:, 1] = rot[:, 1] * s + cy
    return out


# ----------------------------------------------------------------------
# Pose-JSON parsing
# ----------------------------------------------------------------------
_POSE_KEYS = ("yaw", "pitch", "roll", "tx", "ty", "tz")
_GAZE_KEYS = ("yaw", "pitch")


def _parse_keyed_json(
    blob: Optional[str],
    n_frames: int,
    keys: Tuple[str, ...],
    range_field: str,
) -> Dict[int, Dict[str, float]]:
    """Generic per-frame / ranges parser used for head_pose_json and gaze_json.

    Returns ``{frame_idx: {"yaw": ..., "pitch": ...}}``.
    Range entries are broadcast (last-write-wins per range, per-frame WINS).
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("%s: ignored — not valid JSON: %s", range_field, e)
        return {}
    if not isinstance(data, dict):
        return {}

    per_frame: Dict[int, Dict[str, float]] = {}
    frames = data.get("frames")
    if isinstance(frames, dict):
        for f_key, v in frames.items():
            try:
                f_idx = int(f_key)
            except (TypeError, ValueError):
                continue
            if f_idx < 0 or (n_frames > 0 and f_idx >= n_frames):
                continue
            if not isinstance(v, dict):
                continue
            entry: Dict[str, float] = {}
            for k in keys:
                if k in v:
                    try:
                        entry[k] = float(v[k])
                    except (TypeError, ValueError):
                        pass
            if entry:
                per_frame[f_idx] = entry

    from_ranges: Dict[int, Dict[str, float]] = {}
    ranges = data.get("ranges")
    if isinstance(ranges, list):
        for entry in ranges:
            if not isinstance(entry, dict):
                continue
            try:
                s = int(entry.get("start", 0))
                e = int(entry.get("end", n_frames - 1))
            except (TypeError, ValueError):
                continue
            s = max(0, s)
            e = min(n_frames - 1, e) if n_frames > 0 else e
            if e < s:
                continue
            pose = entry.get(range_field) or entry.get("pose") or entry.get("gaze")
            if not isinstance(pose, dict):
                continue
            vals: Dict[str, float] = {}
            for k in keys:
                if k in pose:
                    try:
                        vals[k] = float(pose[k])
                    except (TypeError, ValueError):
                        pass
            if not vals:
                continue
            for f in range(s, e + 1):
                from_ranges[f] = dict(vals)

    merged: Dict[int, Dict[str, float]] = {}
    merged.update(from_ranges)
    merged.update(per_frame)
    return merged


# ----------------------------------------------------------------------
# Reference-blend helpers (subset of WanFaceExpressionEditorV2 logic).
# ----------------------------------------------------------------------
_REGION_MOUTH = list(range(48, 68))
_REGION_BROWS = list(range(17, 27))
_REGION_EYES  = list(range(36, 48))
_REGION_JAW   = list(range(5, 12))
_EYE_EMPH_IDX = (37, 38, 43, 44)


def _ref_norm_from_bundle(ref_bundle: Optional[dict]) -> Optional[np.ndarray]:
    """Pick the first valid face in a reference POSEDATA and return its
    bbox-normalised iBUG-68 landmarks (68, 2)."""
    if not isinstance(ref_bundle, dict):
        return None
    metas = ref_bundle.get("pose_metas_original") or ref_bundle.get("pose_metas")
    if not metas:
        return None
    for m in metas:
        xy = _read_face_normalised(m)
        if xy is None:
            continue
        mn = xy.min(axis=0)
        mx = xy.max(axis=0)
        w = max(float(mx[0] - mn[0]), 1e-6)
        h = max(float(mx[1] - mn[1]), 1e-6)
        local = np.empty_like(xy)
        local[:, 0] = (xy[:, 0] - mn[0]) / w
        local[:, 1] = (xy[:, 1] - mn[1]) / h
        return local
    return None


def _apply_reference_blend(
    xy_norm: np.ndarray,
    ref_norm_local: np.ndarray,
    blend_strength: float,
    affect_mouth: bool,
    affect_brows: bool,
    affect_eyes: bool,
    affect_jaw: bool,
) -> np.ndarray:
    sel: list[int] = []
    if affect_mouth: sel += _REGION_MOUTH
    if affect_brows: sel += _REGION_BROWS
    if affect_eyes:  sel += _REGION_EYES
    if affect_jaw:   sel += _REGION_JAW
    if not sel or blend_strength <= 0.0:
        return xy_norm
    # Move xy_norm into its own face-local bbox, blend in that space, map back.
    mn = xy_norm.min(axis=0)
    mx = xy_norm.max(axis=0)
    w = max(float(mx[0] - mn[0]), 1e-6)
    h = max(float(mx[1] - mn[1]), 1e-6)
    local = np.empty_like(xy_norm)
    local[:, 0] = (xy_norm[:, 0] - mn[0]) / w
    local[:, 1] = (xy_norm[:, 1] - mn[1]) / h
    s = float(np.clip(blend_strength, 0.0, 1.0))
    for i in sel:
        if i >= _N_LM or i >= ref_norm_local.shape[0]:
            continue
        a = s * (1.5 if i in _EYE_EMPH_IDX else 1.0)
        a = float(np.clip(a, 0.0, 1.0))
        local[i] = (1.0 - a) * local[i] + a * ref_norm_local[i]
    out = np.empty_like(xy_norm)
    out[:, 0] = local[:, 0] * w + mn[0]
    out[:, 1] = local[:, 1] * h + mn[1]
    return out


# ----------------------------------------------------------------------
# Gaze offset for iris_data
# ----------------------------------------------------------------------
def _apply_gaze_offset(
    iris_entry: dict,
    yaw_deg: float,
    pitch_deg: float,
    eye_w_norm: float,
    eye_h_norm: float,
) -> dict:
    """Shift pupil/iris keypoints in iris_data by an angular offset.

    Maps yaw/pitch in degrees through a linear "pixels per degree"
    factor scaled by the eye opening width/height. At ±30° the pupil
    reaches the edge of the visible eye opening; beyond that it clamps.
    """
    if abs(yaw_deg) < 1e-3 and abs(pitch_deg) < 1e-3:
        return iris_entry
    if not isinstance(iris_entry, dict):
        return iris_entry
    out = dict(iris_entry)
    # ±30° → ±half eye-width / half eye-height.
    dx = float(np.clip(yaw_deg / 30.0, -1.0, 1.0)) * (eye_w_norm * 0.5)
    dy = float(np.clip(-pitch_deg / 30.0, -1.0, 1.0)) * (eye_h_norm * 0.5)
    for k in ("left_pupil", "right_pupil",
              "left_iris", "right_iris",
              "pupil_l", "pupil_r",
              "iris_l", "iris_r"):
        if k not in out:
            continue
        arr = np.asarray(out[k], dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] >= 2:
            arr = arr.copy()
            arr[0] += dx
            arr[1] += dy
            out[k] = arr.tolist()
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            arr = arr.copy()
            arr[:, 0] += dx
            arr[:, 1] += dy
            out[k] = arr.tolist()
    return out


def _eye_box_norm(xy_norm: np.ndarray) -> Tuple[float, float]:
    """Average eye opening width/height (image-normalised)."""
    L = xy_norm[_REGION_EYES[:6]]   # 36..41 right
    R = xy_norm[_REGION_EYES[6:]]   # 42..47 left
    def _wh(eye: np.ndarray) -> Tuple[float, float]:
        mn = eye.min(axis=0); mx = eye.max(axis=0)
        return float(max(mx[0] - mn[0], 1e-6)), float(max(mx[1] - mn[1], 1e-6))
    wL, hL = _wh(L)
    wR, hR = _wh(R)
    return (wL + wR) * 0.5, (hL + hR) * 0.5


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------
class WanFaceController3DV2:
    """Single unified face controller — expression + 3D head pose + gaze
    + optional reference-shape blend, all in one node."""

    CATEGORY    = "WanAnimatePreprocessV2/extras"
    DESCRIPTION = (
        "All-in-one face controller for the V2 pose pipeline.\n\n"
        "Combines four independent stages, applied in this order:\n"
        "  (1) Optional reference-shape blend (mouth/brows/eyes/jaw).\n"
        "  (2) 12 FACS-inspired expression dials via expression_coeffs_json.\n"
        "  (3) 3-DoF head rotation (yaw/pitch/roll) using a canonical\n"
        "      iBUG-68 depth map.\n"
        "  (4) Gaze offset applied to iris_data pupil/iris keypoints.\n\n"
        "Leaving a stage's controls at their defaults makes that stage a "
        "no-op, so the same node covers any subset.\n\n"
        "Schemas: see source file docstring."
    )
    RETURN_TYPES = ("POSEDATA", "STRING", "STRING")
    RETURN_NAMES = ("pose_data", "info", "coeff_time_series_json")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {
                    "tooltip": "Pose bundle with iBUG-68 face landmarks.",
                }),
            },
            "optional": {
                # ── (2) Expression coeffs ────────────────────────────
                "expression_coeffs_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-frame and/or range expression coefficient JSON. "
                               "Keys: " + ", ".join(_AXIS_NAMES),
                }),
                "expression_strength": ("FLOAT", {
                    "default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Global multiplier on expression coefficients.",
                }),
                "expression_clamp": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Symmetric clamp on each expression coefficient.",
                }),
                # ── (3) Head pose ────────────────────────────────────
                "head_pose_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-frame head pose JSON. Keys: yaw, pitch, roll (degrees).",
                }),
                "head_yaw_deg": ("FLOAT", {
                    "default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5,
                    "tooltip": "Constant yaw applied to every frame (added to JSON).",
                }),
                "head_pitch_deg": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 0.5,
                }),
                "head_roll_deg": ("FLOAT", {
                    "default": 0.0, "min": -60.0, "max": 60.0, "step": 0.5,
                }),
                # ── (3b) Head translation (rigid 5-DOF extension) ────
                # tx/ty in FACE-BBOX-WIDTH units (±0.5 ≈ half-face shift).
                # tz is a depth-zoom signal: +tz pushes the face away
                # (face shrinks); -tz brings it forward (grows). Clamped
                # in _apply_head_translation so the zoom factor stays in
                # [0.25x .. 4.0x].
                "head_tx": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Horizontal head shift in face-width units (+ = image right).",
                }),
                "head_ty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Vertical head shift in face-width units (+ = image down).",
                }),
                "head_tz": ("FLOAT", {
                    "default": 0.0, "min": -0.75, "max": 3.0, "step": 0.01,
                    "tooltip": "Depth shift: + pushes the face away (smaller), - brings it forward (larger).",
                }),
                # ── (3c) Rigid-pose propagation across the timeline ──
                "propagate_head": (["off", "hold_last", "interpolate", "broadcast_first"], {
                    "default": "off",
                    "tooltip": (
                        "How to fill timeline gaps between explicit head_pose_json pins:\n"
                        "  off            – use pins where given, constants elsewhere.\n"
                        "  hold_last      – step-function: each gap holds the previous pin.\n"
                        "  interpolate    – DAW-style linear interpolation between pins.\n"
                        "  broadcast_first – apply the first pin to every gap frame."
                    ),
                }),
                "propagate_gaze": (["off", "hold_last", "interpolate", "broadcast_first"], {
                    "default": "off",
                    "tooltip": "Same propagation semantics for gaze_json pins.",
                }),
                # ── (4) Gaze ─────────────────────────────────────────
                "gaze_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-frame gaze JSON. Keys: yaw, pitch (degrees, ±30 saturates).",
                }),
                "gaze_yaw_deg": ("FLOAT", {
                    "default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5,
                }),
                "gaze_pitch_deg": ("FLOAT", {
                    "default": 0.0, "min": -30.0, "max": 30.0, "step": 0.5,
                }),
                # ── (1) Reference blend ──────────────────────────────
                "reference_pose_data": ("POSEDATA", {
                    "tooltip": "Optional single-frame POSEDATA used as a region-wise shape target.",
                }),
                "blend_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Reference-blend strength (0 = no blend, 1 = fully match reference).",
                }),
                "blend_mouth": ("BOOLEAN", {"default": True}),
                "blend_brows": ("BOOLEAN", {"default": False}),
                "blend_eyes":  ("BOOLEAN", {"default": False}),
                "blend_jaw":   ("BOOLEAN", {"default": False}),
                # ── Common ───────────────────────────────────────────
                "use_metas": (["edited", "original"], {
                    "default": "edited",
                    "tooltip": "'edited' = stack on pose_metas. 'original' = restart from pose_metas_original.",
                }),
                "frame_start": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "If >=0, only frames in [frame_start..frame_end] are touched. -1 = no clamp.",
                }),
                "frame_end": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                }),
                # ── Hidden override JSON inputs driven by the in-canvas
                #    viewer UI (face landmark drags, body-joint drags,
                #    gaze-handle drags). All face-bbox-normalised /
                #    image-normalised / radian-encoded — see js/face_controller_3d.js.
                "landmark_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"<lm>":[x_bbox_norm,y_bbox_norm]}}} from the in-canvas face viewer. Empty = no override.',
                }),
                "pose_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"<joint>":[x_img_norm,y_img_norm]}}} from the in-canvas pose viewer (OpenPose-18).',
                }),
                "gaze_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"l":[yaw_rad,pitch_rad],"r":[yaw_rad,pitch_rad]}}} from the in-canvas gaze handles.',
                }),
            },
        }

    @classmethod
    def axis_names(cls) -> Tuple[str, ...]:
        return _AXIS_NAMES

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def run(self, pose_data,
            expression_coeffs_json: str = "",
            expression_strength: float = 1.0,
            expression_clamp: float = 1.5,
            head_pose_json: str = "",
            head_yaw_deg: float = 0.0,
            head_pitch_deg: float = 0.0,
            head_roll_deg: float = 0.0,
            head_tx: float = 0.0,
            head_ty: float = 0.0,
            head_tz: float = 0.0,
            propagate_head: str = "off",
            propagate_gaze: str = "off",
            gaze_json: str = "",
            gaze_yaw_deg: float = 0.0,
            gaze_pitch_deg: float = 0.0,
            reference_pose_data=None,
            blend_strength: float = 0.0,
            blend_mouth: bool = True,
            blend_brows: bool = False,
            blend_eyes: bool = False,
            blend_jaw: bool = False,
            use_metas: str = "edited",
            frame_start: int = -1,
            frame_end: int = -1,
            landmark_overrides_json: str = "",
            pose_overrides_json: str = "",
            gaze_overrides_json: str = ""):

        if not isinstance(pose_data, dict):
            raise ValueError("pose_data is not a POSEDATA bundle (expected dict).")

        bundle = deepcopy(pose_data)
        key = "pose_metas" if use_metas == "edited" else "pose_metas_original"
        if key not in bundle and "pose_metas" in bundle:
            key = "pose_metas"
        metas = bundle.get(key)
        if not isinstance(metas, list) or not metas:
            raise ValueError(f"pose_data has no usable '{key}' list of metas.")
        n_frames = len(metas)
        iris_data = bundle.get("iris_data") or [None] * n_frames
        if not isinstance(iris_data, list) or len(iris_data) < n_frames:
            iris_data = (list(iris_data) + [None] * n_frames)[:n_frames]

        # ── Parse stage controls ────────────────────────────────────
        expr_per_frame = _parse_coeffs_json(expression_coeffs_json, n_frames)
        expr_ranges    = _expand_range_overrides(expression_coeffs_json, n_frames)
        expr_overrides = _merge_overrides(expr_per_frame, expr_ranges)

        head_overrides = _parse_keyed_json(
            head_pose_json, n_frames, _POSE_KEYS, range_field="pose",
        )
        gaze_overrides = _parse_keyed_json(
            gaze_json, n_frames, _GAZE_KEYS, range_field="gaze",
        )
        # Spread sparse pins across the timeline when requested. This
        # widens `head_overrides`/`gaze_overrides` with synthetic
        # interpolated/held entries that the per-frame loop below picks
        # up exactly like real pins.
        head_overrides = _propagate_head_overrides(
            head_overrides, n_frames, str(propagate_head), _POSE_KEYS,
        )
        gaze_overrides = _propagate_head_overrides(
            gaze_overrides, n_frames, str(propagate_gaze), _GAZE_KEYS,
        )

        ref_norm_local = _ref_norm_from_bundle(reference_pose_data)
        do_blend = (
            ref_norm_local is not None
            and blend_strength > 0.0
            and (blend_mouth or blend_brows or blend_eyes or blend_jaw)
        )

        # Outer frame clamp.
        if frame_start >= 0 or frame_end >= 0:
            lo = frame_start if frame_start >= 0 else 0
            hi = frame_end   if frame_end   >= 0 else n_frames - 1
        else:
            lo, hi = 0, n_frames - 1

        # ── Apply per frame ─────────────────────────────────────────
        affected = 0
        skipped_no_face = 0
        n_rotated = 0
        n_gazed = 0
        n_expr = 0
        n_blended = 0
        coeff_series: Dict[int, Dict[str, float]] = {}

        clamp = float(expression_clamp)
        strn  = float(expression_strength)

        for f_idx in range(n_frames):
            if f_idx < lo or f_idx > hi:
                continue
            meta = metas[f_idx]
            xy_norm = _read_face_normalised(meta)
            if xy_norm is None:
                skipped_no_face += 1
                continue

            changed = False

            # (1) Reference blend.
            if do_blend:
                xy_norm = _apply_reference_blend(
                    xy_norm, ref_norm_local, float(blend_strength),
                    blend_mouth, blend_brows, blend_eyes, blend_jaw,
                )
                n_blended += 1
                changed = True

            # (2) Expression coeffs.
            c_vec = expr_overrides.get(f_idx)
            if c_vec is not None:
                c = np.clip(c_vec * strn, -clamp, clamp)
                if np.any(np.abs(c) > 1e-6):
                    bw, bh = _face_bbox_size(xy_norm)
                    delta_bbox = np.einsum("k,klc->lc", c, _BASIS)
                    delta_img = np.empty_like(delta_bbox)
                    delta_img[:, 0] = delta_bbox[:, 0] * bw
                    delta_img[:, 1] = delta_bbox[:, 1] * bh
                    xy_norm = xy_norm + delta_img
                    n_expr += 1
                    changed = True
                    coeff_series[f_idx] = {
                        _AXIS_NAMES[k]: float(c[k])
                        for k in range(_N_AXES) if abs(c[k]) > 1e-6
                    }

            # (3) Head pose.
            h_entry = head_overrides.get(f_idx, {})
            yaw   = float(head_yaw_deg)   + float(h_entry.get("yaw", 0.0))
            pitch = float(head_pitch_deg) + float(h_entry.get("pitch", 0.0))
            roll  = float(head_roll_deg)  + float(h_entry.get("roll", 0.0))
            if abs(yaw) > 1e-3 or abs(pitch) > 1e-3 or abs(roll) > 1e-3:
                xy_norm = _apply_head_rotation(xy_norm, yaw, pitch, roll)
                n_rotated += 1
                changed = True

            # (3b) Head translation — tx/ty in face-width units; tz as
            # depth zoom around the face centroid. Per-frame additive
            # overrides under the key "tx"/"ty"/"tz" in head_pose_json
            # are honoured (parser is permissive: any float-coerced
            # value through _parse_keyed_json works once the key is in
            # _POSE_KEYS_FULL).
            tx = float(head_tx) + float(h_entry.get("tx", 0.0))
            ty = float(head_ty) + float(h_entry.get("ty", 0.0))
            tz = float(head_tz) + float(h_entry.get("tz", 0.0))
            if abs(tx) > 1e-4 or abs(ty) > 1e-4 or abs(tz) > 1e-4:
                xy_norm = _apply_head_translation(xy_norm, tx, ty, tz)
                n_rotated += 1  # counted under rigid edits
                changed = True

            if changed:
                _write_face_normalised(meta, xy_norm.astype(np.float32))
                affected += 1

            # (4) Gaze — operates on iris_data, independent of landmark
            # writes. Compute eye dimensions from the (potentially updated)
            # xy_norm for plausible per-frame scaling.
            g_entry = gaze_overrides.get(f_idx, {})
            gy = float(gaze_yaw_deg)   + float(g_entry.get("yaw", 0.0))
            gp = float(gaze_pitch_deg) + float(g_entry.get("pitch", 0.0))
            if (abs(gy) > 1e-3 or abs(gp) > 1e-3) and iris_data[f_idx] is not None:
                ew, eh = _eye_box_norm(xy_norm)
                iris_data[f_idx] = _apply_gaze_offset(
                    iris_data[f_idx], gy, gp, ew, eh,
                )
                n_gazed += 1

        # ── Finalise bundle ─────────────────────────────────────────
        if key != "pose_metas":
            bundle["pose_metas"] = metas
        bundle["iris_data"] = iris_data

        bundle.setdefault("expression_coeffs", {})
        bundle["expression_coeffs"]["axis_names"]  = list(_AXIS_NAMES)
        bundle["expression_coeffs"]["per_frame"]   = {
            str(k): v for k, v in coeff_series.items()
        }
        bundle["expression_coeffs"]["strength"]    = strn
        bundle["expression_coeffs"]["coeff_range"] = clamp
        bundle["head_pose_controls"] = {
            "constant_deg": {"yaw": float(head_yaw_deg),
                              "pitch": float(head_pitch_deg),
                              "roll": float(head_roll_deg)},
            "constant_translation": {"tx": float(head_tx),
                                      "ty": float(head_ty),
                                      "tz": float(head_tz)},
            "propagate_mode": str(propagate_head),
            "per_frame": {str(k): v for k, v in head_overrides.items()},
        }
        bundle["gaze_controls"] = {
            "constant_deg": {"yaw": float(gaze_yaw_deg),
                              "pitch": float(gaze_pitch_deg)},
            "propagate_mode": str(propagate_gaze),
            "per_frame": {str(k): v for k, v in gaze_overrides.items()},
        }

        info = (
            f"WanFaceController3DV2: frames={n_frames} edited={affected} "
            f"skipped_no_face={skipped_no_face} | expr={n_expr} rot={n_rotated} "
            f"gaze={n_gazed} blend={n_blended} | "
            f"strength={strn:.2f} clamp=±{clamp:.2f} base={key}"
        )
        log.info(info)

        ts_json = json.dumps({
            "axis_names": list(_AXIS_NAMES),
            "frames": {str(k): v for k, v in coeff_series.items()},
            "head_pose": {str(k): v for k, v in head_overrides.items()},
            "gaze": {str(k): v for k, v in gaze_overrides.items()},
            "n_frames": n_frames,
        }, separators=(",", ":"))

        # ── (5) UI overrides — face / pose / gaze drags WIN over the
        #     algorithmic stages above so the user always sees their
        #     edits respected on the next queue.
        face_ov   = _parse_face_overrides_ui(landmark_overrides_json)
        pose_ov   = _parse_pose_overrides_ui(pose_overrides_json)
        gaze_ov_u = _parse_gaze_overrides_ui(gaze_overrides_json)
        face_delta = _parse_range_deltas(landmark_overrides_json, n_frames)
        pose_delta = _parse_range_deltas(pose_overrides_json, n_frames)
        n_face_ov = n_pose_ov = n_gaze_ov_u = 0
        n_face_delta = n_pose_delta = 0

        # (5a) Range-delta propagation — a single-frame edit broadcast as a
        #      delta across every frame in its range.  Applied BEFORE the
        #      per-frame absolute overrides so explicit pins always win.
        #      Face deltas are bbox-normalised → scaled by each frame's own
        #      face bbox so the edit tracks scale/pose changes per frame.
        for f_idx, lm_map in face_delta.items():
            if not (0 <= f_idx < n_frames):
                continue
            meta = metas[f_idx]
            xy_now = _read_face_normalised(meta)
            if xy_now is None:
                continue
            mn = xy_now.min(axis=0); mx = xy_now.max(axis=0)
            bw = max(float(mx[0] - mn[0]), 1e-6)
            bh = max(float(mx[1] - mn[1]), 1e-6)
            for lm_idx, (dxn, dyn) in lm_map.items():
                if not (0 <= lm_idx < _N_LM):
                    continue
                xy_now[lm_idx, 0] = float(xy_now[lm_idx, 0] + dxn * bw)
                xy_now[lm_idx, 1] = float(xy_now[lm_idx, 1] + dyn * bh)
                n_face_delta += 1
            _write_face_normalised(meta, xy_now.astype(np.float32))

        # Pose deltas are image-normalised → added directly, clamped to [0,1].
        for f_idx, joint_map in pose_delta.items():
            if not (0 <= f_idx < n_frames):
                continue
            meta = metas[f_idx]
            body_xy = _get_body_kps(meta)
            if body_xy is None:
                continue
            for j_idx, (dxn, dyn) in joint_map.items():
                if not (0 <= j_idx < 18):
                    continue
                body_xy[j_idx, 0] = float(min(1.0, max(0.0, body_xy[j_idx, 0] + dxn)))
                body_xy[j_idx, 1] = float(min(1.0, max(0.0, body_xy[j_idx, 1] + dyn)))
                n_pose_delta += 1
            _set_body_kps(meta, body_xy)

        for f_idx, lm_map in face_ov.items():
            if not (0 <= f_idx < n_frames):
                continue
            meta = metas[f_idx]
            xy_now = _read_face_normalised(meta)
            if xy_now is None:
                continue
            mn = xy_now.min(axis=0); mx = xy_now.max(axis=0)
            bw = max(float(mx[0] - mn[0]), 1e-6)
            bh = max(float(mx[1] - mn[1]), 1e-6)
            for lm_idx, (xn, yn) in lm_map.items():
                if not (0 <= lm_idx < _N_LM):
                    continue
                xy_now[lm_idx, 0] = float(mn[0] + xn * bw)
                xy_now[lm_idx, 1] = float(mn[1] + yn * bh)
                n_face_ov += 1
            _write_face_normalised(meta, xy_now.astype(np.float32))

        for f_idx, joint_map in pose_ov.items():
            if not (0 <= f_idx < n_frames):
                continue
            meta = metas[f_idx]
            body_xy = _get_body_kps(meta)
            if body_xy is None:
                continue
            for j_idx, (xn, yn) in joint_map.items():
                if not (0 <= j_idx < 18):
                    continue
                body_xy[j_idx, 0] = float(xn)
                body_xy[j_idx, 1] = float(yn)
                n_pose_ov += 1
            _set_body_kps(meta, body_xy)

        for f_idx, eye_map in gaze_ov_u.items():
            if not (0 <= f_idx < n_frames):
                continue
            if f_idx >= len(iris_data) or not isinstance(iris_data[f_idx], dict):
                continue
            for eye_key, (yaw_r, pitch_r) in eye_map.items():
                _apply_gaze_override_to_iris_entry(
                    iris_data[f_idx], eye_key, yaw_r, pitch_r,
                )
                n_gaze_ov_u += 1

        if face_ov or pose_ov or gaze_ov_u:
            info += (
                f" | ui_overrides face={n_face_ov} pose={n_pose_ov} gaze={n_gaze_ov_u}"
            )
        if face_delta or pose_delta:
            info += (
                f" | range_delta face={n_face_delta} pose={n_pose_delta}"
                f" over {len(face_delta)}f/{len(pose_delta)}p frames"
            )

        # ── (6) Build overlay_meta for the JS viewer ────────────────
        body_frames_ui: list = []
        gaze_overlay_frames_ui: list = []
        for f_idx, meta in enumerate(metas):
            body_xy = _get_body_kps(meta)
            if body_xy is not None:
                body_list = []
                for j in range(body_xy.shape[0]):
                    x, y = body_xy[j, 0], body_xy[j, 1]
                    if np.isnan(x) or np.isnan(y):
                        body_list.append(None)
                    else:
                        body_list.append([round(float(x), 5), round(float(y), 5)])
                body_frames_ui.append({
                    "i": f_idx, "ok": True,
                    "w": float(meta.get("width", 1.0)),
                    "h": float(meta.get("height", 1.0)),
                    "kps": body_list,
                })
            else:
                body_frames_ui.append({"i": f_idx, "ok": False})

            xy_now = _read_face_normalised(meta)
            l_eye_norm = r_eye_norm = None
            if xy_now is not None:
                _mn = xy_now.min(axis=0); _mx = xy_now.max(axis=0)
                _bw = max(float(_mx[0] - _mn[0]), 1e-6)
                _bh = max(float(_mx[1] - _mn[1]), 1e-6)
                fn = np.empty_like(xy_now)
                fn[:, 0] = (xy_now[:, 0] - _mn[0]) / _bw
                fn[:, 1] = (xy_now[:, 1] - _mn[1]) / _bh
                r_eye_norm = [float(fn[36:42, 0].mean()), float(fn[36:42, 1].mean())]
                l_eye_norm = [float(fn[42:48, 0].mean()), float(fn[42:48, 1].mean())]
            cur_iris = iris_data[f_idx] if (
                f_idx < len(iris_data) and isinstance(iris_data[f_idx], dict)
            ) else None

            def _gp(g):
                if not isinstance(g, dict):
                    return None
                try:
                    return [round(float(g.get("yaw_rad", 0.0)), 5),
                            round(float(g.get("pitch_rad", 0.0)), 5)]
                except (TypeError, ValueError):
                    return None

            gaze_overlay_frames_ui.append({
                "i": f_idx,
                "l_eye_norm": l_eye_norm,
                "r_eye_norm": r_eye_norm,
                "l_gaze": _gp((cur_iris or {}).get("left_gaze")),
                "r_gaze": _gp((cur_iris or {}).get("right_gaze")),
            })

        selectable_lms = (
            list(range(0, 17))               # jaw
            + list(range(17, 27))            # brows
            + list(range(27, 36))            # nose
            + list(range(36, 48))            # eyes
            + list(range(48, 68))            # mouth
        )
        overlay_meta = json.dumps({
            "selected": selectable_lms,
            "eye_emph": list(_EYE_EMPH_IDX),
            "d_norm":   [[0.0, 0.0] for _ in range(_N_LM)],
            "frames":   [{"i": i, "ok": True} for i in range(n_frames)],
            "strength": 1.0,
            "pose": {
                "format":      "openpose_18",
                "joint_names": list(_POSE18_JOINT_NAMES_UI),
                "edges":       [list(e) for e in _POSE18_EDGES_UI],
                "frames":      body_frames_ui,
            },
            "gaze": {
                "max_yaw_rad":   _GAZE_MAX_YAW_RAD,
                "max_pitch_rad": _GAZE_MAX_PITCH_RAD,
                "frames":        gaze_overlay_frames_ui,
            },
        }, separators=(",", ":"))

        return {
            "ui":     {"overlay_meta": [overlay_meta]},
            "result": (bundle, info, ts_json),
        }
