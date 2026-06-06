"""WanExpression3DCoeffsV2 — clean-room AU-based expression coefficient editor.

Background
==========
Modern open-source 3D-aware face animators (EMOCA, DECA, LivePortrait,
AE Face Director) all expose a low-dimensional **implicit blendshape**
control surface that lives on top of an explicit 3DMM such as FLAME or
BFM. The implementation details vary, but the user-facing UX converges
on the same idea:

  *   8-20 named "expression dials" whose meaning is human-readable
      (smile, brow-raise, jaw-drop, blink, ...).
  *   Each dial maps to a fixed displacement field on the face landmark
      set.  Dial value ∈ [-1, +1] linearly scales the displacement.
  *   Edits compose linearly with each other and additively with the
      detected per-frame landmarks (so identity / head pose are
      preserved while expression is re-targeted).

This file is a **clean-room** implementation of that idea, with the
basis hand-authored from the Facial Action Coding System (FACS, Ekman
1978 — public method) and standard iBUG-68 landmark anatomy. No code
or weights from EMOCA / DECA / FLAME / LivePortrait are used; only the
high-level concept ("named expression dials apply linear landmark
deltas") and basic facial anatomy literature.

Design summary
==============
*   Input:    POSEDATA bundle (with per-frame ``keypoints_face`` in
             iBUG-68 layout, normalised [0,1] coords).
*   Control:  JSON of per-frame coefficient overrides, ``{"frames": {
             "<idx>": {"smile": 0.6, "brow_furrow": -0.2, ...}}}``.
*   Math:    For each affected frame i:
              1. Compute the face bbox (min/max of all 68 landmarks).
              2. The basis ``B`` is expressed in **face-bbox-normalised
                 coords** (so it generalises across face scales).  We
                 rescale ``B @ c`` by the bbox size to image-normalised
                 coords, then add to the current landmarks.
              3. Write the new landmarks back into the meta in the
                 same normalised storage convention.
*   Output:  Modified POSEDATA + STRING info + STRING coeff_time_series
             (JSON of per-frame coefficient vectors after clipping,
             usable by the editor UI to draw sliders / curves).
*   Sentinel: the basis is intentionally small (12 axes).  More axes
             can be added by appending to ``_AXIS_NAMES`` and
             ``_BASIS_TABLE`` — both validated at module import time.

Anatomical conventions
======================
iBUG-68 indexing (verified against image-coordinates, X→right, Y→down):
    0-16   jawline (idx 0 = subject's right ear bottom (image left if
                   frontal), idx 8 = chin tip, idx 16 = subject's left
                   ear bottom (image right)).
    17-21  right brow  (subject's right; left side of image, lower X).
    22-26  left brow   (subject's left; right side of image, higher X).
    27-30  nose bridge (top → tip).
    31-35  nose base   (right alar 31 → tip 33 → left alar 35).
    36-41  right eye   (outer corner 36, top 37/38, inner 39, bot 40/41).
    42-47  left  eye   (outer 45..., symmetric layout — see _LEFT_EYE).
    48-59  mouth outer ring (48 = right corner, 54 = left corner).
    60-67  mouth inner ring (60 = right corner, 64 = left corner).

Sign of "outward" displacement on a brow / mouth corner depends on
which side of the face it's on — handled per-row in the basis table.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public axis list — order matters; this defines the column order of B.
# ----------------------------------------------------------------------
_AXIS_NAMES: Tuple[str, ...] = (
    "brow_inner_raise",   # AU1
    "brow_outer_raise",   # AU2
    "brow_furrow",        # AU4
    "eye_close_L",        # AU45-L
    "eye_close_R",        # AU45-R
    "nose_wrinkle",       # AU9
    "cheek_raise",        # AU6
    "smile",              # AU12
    "frown",              # AU15
    "mouth_open",         # AU25
    "jaw_drop",           # AU26
    "lip_pucker",         # AU18
)
_N_AXES = len(_AXIS_NAMES)
_N_LM   = 68


# ----------------------------------------------------------------------
# Basis table.  Each entry is (axis_name, {landmark_idx: (dx, dy)}).
# Units = face-bbox-normalised XY at intensity = 1.0.
# Signs: +X = image right (subject's left for a frontal face),
#        +Y = image down. Positive intensity = the canonical FACS direction.
# ----------------------------------------------------------------------
_BASIS_TABLE: Dict[str, Dict[int, Tuple[float, float]]] = {
    # --- AU1 inner brow raiser ------------------------------------
    "brow_inner_raise": {
        20: (0.0, -0.040), 21: (0.0, -0.080),
        22: (0.0, -0.080), 23: (0.0, -0.040),
    },
    # --- AU2 outer brow raiser ------------------------------------
    "brow_outer_raise": {
        17: (0.0, -0.070), 18: (0.0, -0.050),
        25: (0.0, -0.050), 26: (0.0, -0.070),
    },
    # --- AU4 brow furrow (brows down + inward) --------------------
    "brow_furrow": {
        17: (+0.005, +0.015), 18: (+0.010, +0.020),
        19: (+0.015, +0.025), 20: (+0.020, +0.030),
        21: (+0.025, +0.035),
        22: (-0.025, +0.035), 23: (-0.020, +0.030),
        24: (-0.015, +0.025), 25: (-0.010, +0.020),
        26: (-0.005, +0.015),
    },
    # --- AU45 left eye blink --------------------------------------
    "eye_close_L": {
        43: (0.0, +0.025), 44: (0.0, +0.025),
        46: (0.0, -0.005), 47: (0.0, -0.005),
    },
    # --- AU45 right eye blink -------------------------------------
    "eye_close_R": {
        37: (0.0, +0.025), 38: (0.0, +0.025),
        40: (0.0, -0.005), 41: (0.0, -0.005),
    },
    # --- AU9 nose wrinkler ----------------------------------------
    "nose_wrinkle": {
        30: (0.0, -0.015), 33: (0.0, -0.020),
        31: (0.0, -0.020), 35: (0.0, -0.020),
    },
    # --- AU6 cheek raise (lower lids up + slight corner-up) -------
    "cheek_raise": {
        40: (0.0, -0.010), 41: (0.0, -0.010),
        46: (0.0, -0.010), 47: (0.0, -0.010),
        48: (-0.003, -0.005), 54: (+0.003, -0.005),
    },
    # --- AU12 smile (corners up + out, mid-lip stretches up) ------
    "smile": {
        48: (-0.025, -0.030), 49: (-0.015, -0.020), 50: (-0.005, -0.010),
        52: (+0.005, -0.010), 53: (+0.015, -0.020), 54: (+0.025, -0.030),
        # Inner mouth ring slightly lifted too.
        60: (-0.015, -0.018), 64: (+0.015, -0.018),
    },
    # --- AU15 frown (corners down) --------------------------------
    "frown": {
        48: (-0.005, +0.030), 49: (-0.002, +0.020), 50: (0.000, +0.010),
        52: (0.000, +0.010), 53: (+0.002, +0.020), 54: (+0.005, +0.030),
    },
    # --- AU25 lips part (lower lip drops, upper lip mild) ---------
    "mouth_open": {
        56: (0.0, +0.020), 57: (0.0, +0.025), 58: (0.0, +0.020),
        65: (0.0, +0.020), 66: (0.0, +0.025), 67: (0.0, +0.020),
        # Mild upper-lip lift so the lips actually separate.
        50: (0.0, -0.004), 51: (0.0, -0.006), 52: (0.0, -0.004),
        61: (0.0, -0.004), 62: (0.0, -0.006), 63: (0.0, -0.004),
    },
    # --- AU26 jaw drop (chin drops + lower mouth follows) ---------
    "jaw_drop": {
        6:  (0.0, +0.035), 7:  (0.0, +0.050),
        8:  (0.0, +0.060),
        9:  (0.0, +0.050), 10: (0.0, +0.035),
        # Lower lip drops with the jaw.
        56: (0.0, +0.030), 57: (0.0, +0.040), 58: (0.0, +0.030),
        65: (0.0, +0.030), 66: (0.0, +0.040), 67: (0.0, +0.030),
    },
    # --- AU18 lip pucker (corners pull in, mid-lips push out) -----
    "lip_pucker": {
        48: (+0.020, 0.0), 54: (-0.020, 0.0),
        49: (+0.010, 0.0), 53: (-0.010, 0.0),
        51: (0.0, -0.005), 57: (0.0, +0.005),
        62: (0.0, -0.003), 66: (0.0, +0.003),
    },
}


def _build_basis() -> np.ndarray:
    """Materialise the basis as (K, 68, 2) ndarray. Validated at import."""
    B = np.zeros((_N_AXES, _N_LM, 2), dtype=np.float32)
    for k, name in enumerate(_AXIS_NAMES):
        entries = _BASIS_TABLE.get(name)
        if entries is None:
            raise RuntimeError(f"missing basis entries for axis '{name}'")
        for lm_idx, (dx, dy) in entries.items():
            if not (0 <= lm_idx < _N_LM):
                raise RuntimeError(
                    f"axis '{name}': landmark index {lm_idx} out of range [0,{_N_LM})"
                )
            B[k, lm_idx, 0] = dx
            B[k, lm_idx, 1] = dy
    return B


_BASIS = _build_basis()  # validated at import — fails loudly on typo


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _read_face_normalised(meta: dict) -> np.ndarray | None:
    arr = meta.get("keypoints_face")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < _N_LM or arr.shape[1] < 2:
        return None
    return arr[:_N_LM, :2].copy()


def _write_face_normalised(meta: dict, xy_norm: np.ndarray) -> None:
    src = np.asarray(meta["keypoints_face"], dtype=np.float32).copy()
    src[:_N_LM, :2] = xy_norm
    meta["keypoints_face"] = src


def _face_bbox_size(xy_norm: np.ndarray) -> Tuple[float, float]:
    mins = xy_norm.min(axis=0)
    maxs = xy_norm.max(axis=0)
    w = float(maxs[0] - mins[0])
    h = float(maxs[1] - mins[1])
    return max(w, 1e-6), max(h, 1e-6)


def _parse_coeffs_json(blob: str | None, n_frames: int) -> Dict[int, np.ndarray]:
    """Return ``{frame_idx: (K,) ndarray}``. Empty / invalid → ``{}``."""
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("expression_coeffs_json ignored — not valid JSON: %s", e)
        return {}
    out: Dict[int, np.ndarray] = {}

    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}

    name_to_idx = {n: k for k, n in enumerate(_AXIS_NAMES)}
    for f_key, axis_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if f_idx < 0 or (n_frames > 0 and f_idx >= n_frames):
            continue
        if not isinstance(axis_map, dict):
            continue
        vec = np.zeros(_N_AXES, dtype=np.float32)
        for k, v in axis_map.items():
            ki = name_to_idx.get(str(k))
            if ki is None:
                continue
            try:
                vec[ki] = float(v)
            except (TypeError, ValueError):
                continue
        out[f_idx] = vec
    return out


def _expand_range_overrides(
    blob: str | None, n_frames: int,
) -> Dict[int, np.ndarray]:
    """Apply a "ranges" section that broadcasts coefficients over [start..end].

    Schema (additive over per-frame entries):
    ``{"ranges": [{"start": 0, "end": 47, "coeffs": {"smile": 0.4}}, ...]}``.
    Per-frame entries (from ``_parse_coeffs_json``) WIN over range entries
    on conflicting frames.
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return {}
    ranges = data.get("ranges") if isinstance(data, dict) else None
    if not isinstance(ranges, list):
        return {}

    name_to_idx = {n: k for k, n in enumerate(_AXIS_NAMES)}
    out: Dict[int, np.ndarray] = {}
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
        coeffs = entry.get("coeffs")
        if not isinstance(coeffs, dict):
            continue
        vec = np.zeros(_N_AXES, dtype=np.float32)
        for k, v in coeffs.items():
            ki = name_to_idx.get(str(k))
            if ki is None:
                continue
            try:
                vec[ki] = float(v)
            except (TypeError, ValueError):
                continue
        for f in range(s, e + 1):
            out[f] = out.get(f, np.zeros(_N_AXES, dtype=np.float32)) + vec
    return out


def _merge_overrides(
    per_frame: Dict[int, np.ndarray],
    from_ranges: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    merged = dict(from_ranges)
    merged.update(per_frame)
    return merged


def propagate_expression_keyframes(
    overrides: Dict[int, np.ndarray],
    n_frames: int,
    mode: str,
) -> Dict[int, np.ndarray]:
    """Spread sparse expression keyframes across the timeline.

    Works like ``_propagate_head_overrides`` in face_controller_3d.py
    but operates on ``{frame_idx: (K,) ndarray}`` instead of dicts.

    mode:
        ``"off"``             — return *overrides* unchanged.
        ``"hold_last"``       — step-function: gaps hold previous keyframe.
        ``"interpolate"``     — linear lerp between adjacent keyframes per axis.
        ``"broadcast_first"`` — first keyframe applied everywhere.
    """
    if mode == "off" or not overrides or n_frames <= 0:
        return overrides
    if mode not in ("hold_last", "interpolate", "broadcast_first"):
        return overrides

    sorted_frames = sorted(f for f in overrides if 0 <= f < n_frames)
    if not sorted_frames:
        return overrides

    out: Dict[int, np.ndarray] = {f: v.copy() for f, v in overrides.items()}

    if mode == "broadcast_first":
        v0 = overrides[sorted_frames[0]].copy()
        for f in range(n_frames):
            if f not in out:
                out[f] = v0.copy()
        return out

    # Per-axis interpolation / hold.
    for ax in range(_N_AXES):
        pins = [(f, float(overrides[f][ax])) for f in sorted_frames
                if f in overrides and abs(float(overrides[f][ax])) > 1e-7]
        if not pins:
            continue

        for f in range(n_frames):
            if f in overrides:
                continue
            left = right = None
            for p in pins:
                if p[0] <= f:
                    left = p
                elif right is None:
                    right = p
                    break
            if left is None and right is None:
                continue
            slot = out.setdefault(f, np.zeros(_N_AXES, dtype=np.float32))
            if left is None:
                slot[ax] = right[1]
            elif right is None or mode == "hold_last":
                slot[ax] = left[1]
            else:
                span = right[0] - left[0]
                t = (f - left[0]) / max(span, 1)
                slot[ax] = left[1] + (right[1] - left[1]) * t
    return out


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------
class WanExpression3DCoeffsV2:
    CATEGORY    = "WanAnimatePreprocessV2/extras"
    DESCRIPTION = (
        "Clean-room AU-based 3D-aware expression coefficient editor.\n"
        "Exposes 12 FACS-inspired expression dials (smile, brow furrow, "
        "blink, jaw drop, lip pucker, ...) that apply linear iBUG-68 "
        "landmark deltas per frame.\n\n"
        "JSON schema (`expression_coeffs_json`):\n"
        '{\n'
        '  "frames":  {"4": {"smile": 0.7, "eye_close_L": 1.0}, ...},\n'
        '  "ranges":  [{"start": 0, "end": 24, "coeffs": {"brow_furrow": 0.3}}]\n'
        '}\n\n'
        "Per-frame entries override range entries on conflicts.\n"
        "Coefficients are clipped to [-coeff_range, +coeff_range]."
    )
    RETURN_TYPES = ("POSEDATA", "STRING", "STRING")
    RETURN_NAMES = ("pose_data", "info", "coeff_time_series_json")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {
                    "tooltip": "Input pose bundle with iBUG-68 face landmarks.",
                }),
            },
            "optional": {
                "expression_coeffs_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Per-frame and/or range expression coefficient JSON. See node description.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05,
                    "tooltip": "Global multiplier on all coefficients. Negative inverts every dial.",
                }),
                "coeff_range": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Symmetric clamp on each coefficient AFTER strength is applied.",
                }),
                "use_metas": (["edited", "original"], {
                    "default": "edited",
                    "tooltip": "'edited' = stack on top of pose_metas (recommended). "
                               "'original' = base off pose_metas_original (drops prior edits).",
                }),
                "frame_start": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "If >=0, ignore entries outside [frame_start..frame_end]. -1 = no clamp.",
                }),
                "frame_end": ("INT", {
                    "default": -1, "min": -1, "max": 999999, "step": 1,
                    "tooltip": "If >=0, ignore entries outside [frame_start..frame_end]. -1 = no clamp.",
                }),
            },
        }

    @classmethod
    def axis_names(cls) -> Tuple[str, ...]:
        """Public accessor so JS / editor UI can ask for the canonical list."""
        return _AXIS_NAMES

    def run(self, pose_data,
            expression_coeffs_json: str = "",
            strength: float = 1.0,
            coeff_range: float = 1.5,
            use_metas: str = "edited",
            frame_start: int = -1,
            frame_end: int = -1):

        if not isinstance(pose_data, dict):
            raise ValueError("pose_data is not a POSEDATA bundle (expected dict).")

        # Deep-copy bundle so we never mutate the upstream object.
        bundle = deepcopy(pose_data)

        key = "pose_metas" if use_metas == "edited" else "pose_metas_original"
        if key not in bundle and "pose_metas" in bundle:
            key = "pose_metas"
        metas = bundle.get(key)
        if not isinstance(metas, list) or not metas:
            raise ValueError(f"pose_data has no usable '{key}' list of metas.")
        n_frames = len(metas)

        per_frame = _parse_coeffs_json(expression_coeffs_json, n_frames)
        from_ranges = _expand_range_overrides(expression_coeffs_json, n_frames)
        overrides = _merge_overrides(per_frame, from_ranges)

        # Optional outer frame clamp.
        if frame_start >= 0 or frame_end >= 0:
            lo = frame_start if frame_start >= 0 else 0
            hi = frame_end   if frame_end   >= 0 else n_frames - 1
            overrides = {f: v for f, v in overrides.items() if lo <= f <= hi}

        # Apply.
        affected = 0
        skipped_no_face = 0
        coeff_series: Dict[int, Dict[str, float]] = {}
        for f_idx, coeffs in sorted(overrides.items()):
            if f_idx >= n_frames:
                continue
            meta = metas[f_idx]
            xy_norm = _read_face_normalised(meta)
            if xy_norm is None:
                skipped_no_face += 1
                continue
            # Apply strength + clamp.
            c = np.clip(coeffs * float(strength), -float(coeff_range), float(coeff_range))
            # B is in face-bbox-normalised coords; rescale to image-normalised
            # by multiplying by the face bbox size.
            bw, bh = _face_bbox_size(xy_norm)
            delta_bbox = np.einsum("k,klc->lc", c, _BASIS)  # (68, 2)
            delta_img = np.empty_like(delta_bbox)
            delta_img[:, 0] = delta_bbox[:, 0] * bw
            delta_img[:, 1] = delta_bbox[:, 1] * bh
            new_xy = xy_norm + delta_img
            _write_face_normalised(meta, new_xy.astype(np.float32))
            affected += 1
            coeff_series[f_idx] = {
                _AXIS_NAMES[k]: float(c[k]) for k in range(_N_AXES) if abs(c[k]) > 1e-6
            }

        # Make sure pose_metas is the authoritative edited list.
        if key != "pose_metas":
            bundle["pose_metas"] = metas

        # Annotate bundle so the editor UI / downstream nodes can see the
        # current coefficient state.
        bundle.setdefault("expression_coeffs", {})
        bundle["expression_coeffs"]["axis_names"] = list(_AXIS_NAMES)
        bundle["expression_coeffs"]["per_frame"]  = {
            str(k): v for k, v in coeff_series.items()
        }
        bundle["expression_coeffs"]["strength"]    = float(strength)
        bundle["expression_coeffs"]["coeff_range"] = float(coeff_range)

        info = (
            f"WanExpression3DCoeffsV2: frames={n_frames} | "
            f"edited={affected} | skipped_no_face={skipped_no_face} | "
            f"axes={_N_AXES} | strength={strength:.2f} clamp=±{coeff_range:.2f} | "
            f"base={key}"
        )
        log.info(info)

        ts_json = json.dumps({
            "axis_names": list(_AXIS_NAMES),
            "frames": {str(k): v for k, v in coeff_series.items()},
        }, separators=(",", ":"))

        return (bundle, info, ts_json)
