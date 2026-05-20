# Copyright 2026 Code2Collapse / WanAnimatePreprocessV2.
# Licensed under the Apache License, Version 2.0.
#
# Third-party data / models reused by this node:
#   * iBUG-68 / COCO-WholeBody 68-point face landmark convention (CC-BY-4.0).
#   * AAPoseMeta container (MIT, from the upstream Wan-Animate codebase).
#
# This node does NOT bundle or call any face-recognition / gaze-estimation
# models — it operates purely on 2-D keypoints that already live in the
# POSEDATA bundle produced by Pose+Face Detection (V2).

"""WanFaceExpressionEditorUI — delta-propagation face expression editor.

Why this exists (vs. the older V2 "blend toward reference" node)
================================================================
The V2 node did a *per-frame LERP* of every selected landmark toward the
reference shape. That flattens natural animation variance — every frame
ends up with the same mouth shape, the same brow lift, etc. — which is
the wrong semantic for an *animate-this-driving-clip-but-with-a-different-
neutral-expression* use case.

This node ships the correct algorithm: **delta propagation in face-bbox-
normalised space**.

Algorithm
---------
Let ``f0`` be the first driving frame whose face is visible and ``ref``
the reference frame. Both are converted to iBUG-68 keypoints normalised
into their own face-bounding-box space (``norm = (xy - bbox.xy) / bbox.wh``,
giving values roughly in ``[0, 1]``).

  d_norm[i]      = ref_norm[i] − f0_norm[i]     for every selected landmark i

Then, for every frame ``t`` (including ``t = 0``)::

  new_norm_t[i]  = tgt_norm_t[i] + α_i · d_norm[i]
  new_xy_t[i]    = new_norm_t[i] · bbox_t.wh + bbox_t.xy

where ``α_i = strength · (1.5 if i ∈ eye_emph else 1.0)``, clamped to
[0, 1]. The reference's *delta* (not its absolute shape) is what gets
applied, so per-frame motion of unselected landmarks is preserved AND
selected landmarks keep their per-frame variation (they all just shift
by the same normalised offset).

The denominator ``bbox_t.wh`` is each frame's own face bbox — that
implements the user-mandated formula::

  kps_face[t] += d_norm · (bbox_w_frame_t, bbox_h_frame_t)

(``d_norm`` is dimensionless, so multiplying by bbox.wh restores pixel
units in frame ``t``'s own face coordinate system.)

iBUG-68 region slices (consistent with ``AAPoseMeta.kps_face``):

* jaw       : indices 5..11  (chin / lower jaw subset)
* brows     : indices 17..26 (both brows)
* eyes      : indices 36..47 (both eye outlines)
* eye_emph  : indices 37, 38, 43, 44 (upper eyelids — applied at 1.5×)
* mouth     : indices 48..67 (full mouth)

Optional JS-overlay hook
------------------------
``landmark_overrides_json`` accepts a JSON object of the shape::

    {
      "frames": {
        "<frame_idx>": { "<lm_idx>": [x_norm, y_norm], ... },
        ...
      }
    }

Coordinates are in *that frame's* bbox-normalised space. Any landmark
listed here OVERRIDES the delta result (it is written as-is after
denormalisation). This is the contract the upcoming JS canvas overlay
(todo #7) will produce.
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy

import numpy as np

log = logging.getLogger(__name__)


# Canonical max-gaze angle constants — must match gaze_blendshape.py so that
# the JS drag offset (which is bbox-normalised) maps to the SAME (yaw, pitch)
# radians the downstream pipeline already expects.
_GAZE_MAX_YAW_RAD   = math.radians(30.0)
_GAZE_MAX_PITCH_RAD = math.radians(25.0)


# ── iBUG-68 region slices ─────────────────────────────────────────────
_JAW_IDX   = list(range(5, 12))                  # 5..11 inclusive
_BROWS_IDX = list(range(17, 27))                 # 17..26
_EYES_IDX  = list(range(36, 48))                 # 36..47
_EYE_EMPH  = (37, 38, 43, 44)                    # upper-eyelid emphasis
_MOUTH_IDX = list(range(48, 68))                 # 48..67


# ── OpenPose-18 (BODY_18 / COCO) joint topology ──────────────────────
# Index -> joint name (used by the JS overlay for hover tooltips).
_POSE18_JOINT_NAMES = (
    "nose", "neck",
    "rshoulder", "relbow", "rwrist",
    "lshoulder", "lelbow", "lwrist",
    "rhip", "rknee", "rankle",
    "lhip", "lknee", "lankle",
    "reye", "leye", "rear", "lear",
)

# Skeleton edges: pairs of joint indices that the JS overlay draws as
# bone lines.  This is the canonical OpenPose-COCO 18-joint topology
# (matches what `keypoints_body` stores in the POSEDATA bundle).
_POSE18_EDGES: tuple[tuple[int, int], ...] = (
    (1, 2),  (1, 5),                # shoulders <- neck
    (2, 3),  (3, 4),                # right arm
    (5, 6),  (6, 7),                # left  arm
    (1, 8),  (8, 9),  (9, 10),      # right leg
    (1, 11), (11, 12), (12, 13),    # left  leg
    (1, 0),                         # neck <- nose
    (0, 14), (14, 16),              # right eye / ear
    (0, 15), (15, 17),              # left  eye / ear
)


# ── Body keypoint helpers (image-normalised [0,1] floats) ─────────────
def _get_body_kps(meta) -> np.ndarray | None:
    """Return (18, 2) image-normalised body keypoints, or None.

    ``keypoints_body`` in POSEDATA is stored as a python list of 18 items;
    each item is either ``None`` (joint missing in this frame) or a
    sequence ``[x_norm, y_norm, (confidence?)]`` already normalised to
    the source image's WxH.  We preserve missing joints by writing NaN
    so the JS overlay can hide them.
    """
    if not isinstance(meta, dict):
        return None
    arr = meta.get("keypoints_body")
    if not isinstance(arr, (list, tuple)) or len(arr) < 18:
        return None
    out = np.full((18, 2), np.nan, dtype=np.float32)
    for i in range(18):
        v = arr[i]
        if v is None:
            continue
        try:
            out[i, 0] = float(v[0])
            out[i, 1] = float(v[1])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _set_body_kps(meta: dict, xy_norm: np.ndarray) -> None:
    """Write back (18, 2) image-normalised body keypoints into ``meta``.

    Preserves the original 3rd component (confidence) when present, and
    keeps slots that are still NaN as None so downstream renderers can
    distinguish "edited" from "missing".
    """
    src = meta.get("keypoints_body")
    if not isinstance(src, list):
        return
    for i in range(min(18, len(src))):
        x, y = float(xy_norm[i, 0]), float(xy_norm[i, 1])
        if np.isnan(x) or np.isnan(y):
            # Leave the joint as it was — user did NOT add a brand-new joint
            # via override; missing joints stay missing.
            continue
        v = src[i]
        if v is None:
            src[i] = [x, y, 1.0]
        else:
            try:
                src[i] = [x, y, float(v[2])] if len(v) >= 3 else [x, y]
            except (TypeError, ValueError, IndexError):
                src[i] = [x, y]


def _bbox_xywh(kps: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned (x, y, w, h) of a (N, 2) keypoint array. Never zero-sized."""
    mn = kps.min(axis=0)
    mx = kps.max(axis=0)
    w = max(float(mx[0] - mn[0]), 1e-6)
    h = max(float(mx[1] - mn[1]), 1e-6)
    return float(mn[0]), float(mn[1]), w, h


def _normalize(kps: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox
    out = np.empty_like(kps, dtype=np.float32)
    out[:, 0] = (kps[:, 0] - x) / w
    out[:, 1] = (kps[:, 1] - y) / h
    return out


def _denormalize(kps_norm: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = bbox
    out = np.empty_like(kps_norm, dtype=np.float32)
    out[:, 0] = kps_norm[:, 0] * w + x
    out[:, 1] = kps_norm[:, 1] * h + y
    return out


def _selected_indices(affect_mouth: bool, affect_brows: bool,
                      affect_eyes: bool, affect_jaw: bool) -> list[int]:
    sel: list[int] = []
    if affect_mouth: sel.extend(_MOUTH_IDX)
    if affect_brows: sel.extend(_BROWS_IDX)
    if affect_eyes:  sel.extend(_EYES_IDX)
    if affect_jaw:   sel.extend(_JAW_IDX)
    # De-duplicate while preserving order.
    seen: set[int] = set()
    out: list[int] = []
    for i in sel:
        if i in seen: continue
        seen.add(i); out.append(i)
    return out


def _get_face_kps(meta) -> np.ndarray | None:
    """Return (68, 2) pixel-space face keypoints, or None if unavailable.

    Honors the humanapi-style normalised-by-frame storage used in
    ``POSEDATA["pose_metas_original"]``."""
    if not isinstance(meta, dict):
        return None
    arr = meta.get("keypoints_face")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 68:
        return None
    w = float(meta.get("width", 1.0))
    h = float(meta.get("height", 1.0))
    xy = arr[:68, :2].copy()
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy


def _set_face_kps(meta: dict, xy_pixels: np.ndarray) -> None:
    arr = np.asarray(meta["keypoints_face"], dtype=np.float32).copy()
    w = max(float(meta.get("width", 1.0)), 1e-6)
    h = max(float(meta.get("height", 1.0)), 1e-6)
    arr[:68, 0] = xy_pixels[:, 0] / w
    arr[:68, 1] = xy_pixels[:, 1] / h
    meta["keypoints_face"] = arr


def _parse_overrides(blob: str | None) -> dict[int, dict[int, tuple[float, float]]]:
    """Parse the JS-overlay payload. Returns ``{}`` on empty/invalid."""
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("landmark_overrides_json ignored — not valid JSON: %s", e)
        return {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}
    out: dict[int, dict[int, tuple[float, float]]] = {}
    for f_key, lm_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(lm_map, dict):
            continue
        per_frame: dict[int, tuple[float, float]] = {}
        for lm_key, xy in lm_map.items():
            try:
                lm_idx = int(lm_key)
            except (TypeError, ValueError):
                continue
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                continue
            try:
                per_frame[lm_idx] = (float(xy[0]), float(xy[1]))
            except (TypeError, ValueError):
                continue
        if per_frame:
            out[f_idx] = per_frame
    return out


def _parse_gaze_overrides(blob: str | None) -> dict[int, dict[str, tuple[float, float]]]:
    """Parse the JS gaze-overlay payload.

    Expected shape (mirror of the JS contract)::

        {
          "frames": {
            "<frame_idx>": {
              "l": [yaw_rad, pitch_rad],
              "r": [yaw_rad, pitch_rad]
            }, ...
          }
        }

    Returns ``{frame_idx: {"l": (yaw, pitch), "r": (yaw, pitch)}}``.
    Missing eyes / invalid entries are silently dropped.
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("gaze_overrides_json ignored — not valid JSON: %s", e)
        return {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}
    out: dict[int, dict[str, tuple[float, float]]] = {}
    for f_key, eye_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(eye_map, dict):
            continue
        per_frame: dict[str, tuple[float, float]] = {}
        for eye_key in ("l", "r"):
            v = eye_map.get(eye_key)
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                continue
            try:
                per_frame[eye_key] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue
        if per_frame:
            out[f_idx] = per_frame
    return out


def _apply_gaze_override_to_iris_entry(entry: dict, eye_key: str,
                                       yaw_rad: float, pitch_rad: float) -> None:
    """Patch a single per-frame iris_data entry's ``{eye}_gaze`` dict in place.

    Mirrors the field shape that downstream nodes (gaze_blendshape, the
    Wan-Animate pose renderer) already consume: ``yaw_rad``, ``pitch_rad``,
    plus a normalised 2-D direction (``dx``, ``dy``) and a magnitude in
    [0, 1] derived from the override angles.  Source tag is changed to
    ``"user_override"`` so the pipeline knows the value came from the UI.
    """
    eye_name = "left" if eye_key == "l" else "right"
    key = f"{eye_name}_gaze"
    g = entry.get(key)
    if not isinstance(g, dict):
        g = {}
    # 2-D direction encoded as unit-length displacement on a flat plane.
    # Positive yaw → look-right (dx > 0); positive pitch → look-up.  This
    # matches how the JS overlay derives the offset from the eye centroid
    # and how gaze_blendshape encodes the screen-space arrow.
    yaw_n   = max(-1.0, min(1.0, yaw_rad   / _GAZE_MAX_YAW_RAD))
    pitch_n = max(-1.0, min(1.0, pitch_rad / _GAZE_MAX_PITCH_RAD))
    nrm = float(math.hypot(yaw_n, pitch_n))
    if nrm > 1e-6:
        dx = yaw_n / nrm
        dy = -pitch_n / nrm        # image-Y is down → invert pitch
        mag = min(1.0, nrm)
    else:
        dx = dy = 0.0
        mag = 0.0
    g["yaw_rad"]         = float(yaw_rad)
    g["pitch_rad"]       = float(pitch_rad)
    g["dx"]              = round(dx, 4)
    g["dy"]              = round(dy, 4)
    g["magnitude_norm"]  = round(mag, 4)
    g["source"]          = "user_override"
    entry[key] = g


class WanFaceExpressionEditorUI:
    """Apply a reference-derived **delta** to selected face regions of every
    driving frame, in face-bbox-normalised iBUG-68 space.

    Unlike the older V2 LERP node, this preserves per-frame variation: it
    shifts all selected landmarks by the same normalised offset rather
    than collapsing them onto the reference's absolute shape.

    Inputs
    ------
    pose_data                : POSEDATA (driving sequence, required)
    reference_pose_data      : POSEDATA (single-frame reference, required)
    strength                 : FLOAT 0..1 (0 = pass-through, 1 = full delta)
    affect_mouth/brows/eyes/jaw : BOOLEAN
    landmark_overrides_json  : STRING (optional, JS-overlay hook — see module
                               docstring). Empty = ignored.

    Outputs
    -------
    pose_data                : POSEDATA (modified deep-copy)
    info                     : STRING   diagnostic summary
    overlay_meta             : STRING   JSON: per-frame bboxes + selected
                                        indices, for the JS canvas overlay.
    """

    CATEGORY = "Code2Collapse/WanAnimateV2/Face"
    DESCRIPTION = (
        "Apply a reference-derived delta (ref - driving_frame0) to selected "
        "face regions of every driving frame. Operates in face-bbox-"
        "normalised iBUG-68 space so head pose and scale are preserved AND "
        "per-frame animation variance is retained. Drop-in replacement for "
        "WanFaceExpressionEditorV2 (which did a flatter per-frame LERP). "
        "Optional landmark_overrides_json input is the hook point for the "
        "JS canvas overlay."
    )
    RETURN_TYPES = ("POSEDATA", "STRING", "STRING")
    RETURN_NAMES = ("pose_data", "info", "overlay_meta")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data":           ("POSEDATA", {"tooltip": "Driving sequence (multi-frame)."}),
                "reference_pose_data": ("POSEDATA", {"tooltip": "Single-frame reference whose expression delta is the target."}),
                "strength":            ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                                  "tooltip": "Per-landmark alpha multiplier (0=pass-through, 1=full delta)."}),
                "affect_mouth":        ("BOOLEAN", {"default": True}),
                "affect_brows":        ("BOOLEAN", {"default": True}),
                "affect_eyes":         ("BOOLEAN", {"default": False, "tooltip": "Upper eyelid landmarks 37/38/43/44 get 1.5× emphasis."}),
                "affect_jaw":          ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "landmark_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"<lm>":[x_norm,y_norm]}}} from the JS canvas overlay. Empty = ignored.',
                }),
                "pose_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"<joint>":[x_norm,y_norm]}}} from the JS pose canvas. Joint indices follow OpenPose-18 (BODY_18 / COCO). Coordinates are IMAGE-normalised (not face-bbox-normalised). Empty = ignored.',
                }),
                "gaze_overrides_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": 'JSON {"frames":{"<idx>":{"l":[yaw_rad,pitch_rad],"r":[yaw_rad,pitch_rad]}}} from the JS gaze handles. Angles use the ETH-XGaze-compatible convention: positive yaw=look-right, positive pitch=look-up. Empty = ignored.',
                }),
                "reference_image": ("IMAGE", {"tooltip": "Optional preview only — not used for detection."}),
            },
        }

    # ── core ──────────────────────────────────────────────────────────
    def run(self, pose_data, reference_pose_data, strength,
            affect_mouth, affect_brows, affect_eyes, affect_jaw,
            landmark_overrides_json: str = "",
            pose_overrides_json: str = "",
            gaze_overrides_json: str = "",
            reference_image=None):

        if not isinstance(pose_data, dict):
            raise TypeError("pose_data must be a POSEDATA dict from Pose+Face Detection (V2).")
        if not isinstance(reference_pose_data, dict):
            raise TypeError("reference_pose_data must be a POSEDATA dict.")

        target_metas = pose_data.get("pose_metas_original") or pose_data.get("pose_metas")
        ref_metas    = reference_pose_data.get("pose_metas_original") or reference_pose_data.get("pose_metas")
        if not target_metas:
            raise ValueError("pose_data has no 'pose_metas_original' / 'pose_metas'.")
        if not ref_metas:
            raise ValueError("reference_pose_data has no 'pose_metas_original' / 'pose_metas'.")

        sel_idx = _selected_indices(affect_mouth, affect_brows, affect_eyes, affect_jaw)
        overrides = _parse_overrides(landmark_overrides_json)
        pose_overrides = _parse_overrides(pose_overrides_json)
        gaze_overrides = _parse_gaze_overrides(gaze_overrides_json)
        s = float(np.clip(strength, 0.0, 1.0))

        # Reference: first frame with valid keypoints.
        ref_xy_px = None
        for rm in ref_metas:
            xy = _get_face_kps(rm)
            if xy is not None:
                ref_xy_px = xy
                break
        if ref_xy_px is None:
            raise ValueError("reference_pose_data contains no usable 'keypoints_face'.")
        ref_norm = _normalize(ref_xy_px, _bbox_xywh(ref_xy_px))

        # Driving frame 0 (first frame with valid keypoints).
        f0_norm = None
        for tm in target_metas:
            xy = _get_face_kps(tm)
            if xy is not None:
                f0_norm = _normalize(xy, _bbox_xywh(xy))
                break
        if f0_norm is None:
            log.warning("driving sequence has no face keypoints — pass-through.")
            _meta = json.dumps({"frames": [], "selected": sel_idx})
            return {
                "ui":     {"overlay_meta": [_meta]},
                "result": (pose_data,
                           "FaceExpressionEditorUI: driving sequence has no face keypoints — pass-through.",
                           _meta),
            }

        # Normalised-space delta (only relevant rows used downstream).
        d_norm = ref_norm - f0_norm  # shape (68, 2)

        # Per-landmark alpha (eye-emph rows scaled to 1.5×, clamped to [0,1]).
        alpha = np.full((68,), s, dtype=np.float32)
        if affect_eyes:
            for i in _EYE_EMPH:
                alpha[i] = min(s * 1.5, 1.0)

        new_metas: list = []
        overlay_frames: list[dict] = []
        body_frames: list[dict] = []
        n_edited = 0
        n_skipped = 0
        n_overridden = 0
        n_pose_overridden = 0
        n_pose_frames_edited = 0

        for f_idx, tm in enumerate(target_metas):
            new_tm = deepcopy(tm)
            xy = _get_face_kps(new_tm)
            if xy is None:
                n_skipped += 1
                # Face missing — but pose may still exist; apply pose
                # overrides anyway and emit a body overlay record.
                body_xy = _get_body_kps(new_tm)
                if body_xy is not None:
                    f_ov = pose_overrides.get(f_idx)
                    if f_ov:
                        for j_idx, (xn, yn) in f_ov.items():
                            if 0 <= j_idx < body_xy.shape[0]:
                                body_xy[j_idx, 0] = float(xn)
                                body_xy[j_idx, 1] = float(yn)
                                n_pose_overridden += 1
                        _set_body_kps(new_tm, body_xy)
                        n_pose_frames_edited += 1
                    body_list = []
                    for j in range(body_xy.shape[0]):
                        x, y = body_xy[j, 0], body_xy[j, 1]
                        if np.isnan(x) or np.isnan(y):
                            body_list.append(None)
                        else:
                            body_list.append([round(float(x), 5), round(float(y), 5)])
                    body_frames.append({
                        "i": f_idx, "ok": True,
                        "w": float(new_tm.get("width", 1.0)),
                        "h": float(new_tm.get("height", 1.0)),
                        "kps": body_list,
                    })
                else:
                    body_frames.append({"i": f_idx, "ok": False})
                new_metas.append(new_tm)
                overlay_frames.append({"i": f_idx, "ok": False})
                continue

            tgt_bbox = _bbox_xywh(xy)
            tgt_norm = _normalize(xy, tgt_bbox)

            # Apply delta to selected landmarks only.
            if sel_idx and s > 0.0:
                idx_arr = np.asarray(sel_idx, dtype=np.int64)
                tgt_norm[idx_arr] = tgt_norm[idx_arr] + alpha[idx_arr, None] * d_norm[idx_arr]

            # Apply per-frame JS-overlay overrides (in bbox-normalised space)
            # AFTER the delta so the user gets the final word.
            frame_over = overrides.get(f_idx)
            if frame_over:
                for lm_idx, (xn, yn) in frame_over.items():
                    if 0 <= lm_idx < tgt_norm.shape[0]:
                        tgt_norm[lm_idx, 0] = float(xn)
                        tgt_norm[lm_idx, 1] = float(yn)
                        n_overridden += 1

            new_xy = _denormalize(tgt_norm, tgt_bbox)
            _set_face_kps(new_tm, new_xy)
            n_edited += 1
            new_metas.append(new_tm)
            overlay_frames.append({
                "i": f_idx, "ok": True,
                "bbox": [round(tgt_bbox[0], 2), round(tgt_bbox[1], 2),
                         round(tgt_bbox[2], 2), round(tgt_bbox[3], 2)],
            })

            # ── Body / pose overrides (image-normalised coords) ───────
            # We always emit an overlay record for every target frame
            # (regardless of whether overrides are present) so the JS
            # pose canvas can render the current shape.  Overrides are
            # applied in-place to the *deep-copied* meta so the upstream
            # POSEDATA is untouched.
            body_xy = _get_body_kps(new_tm)
            body_ok = body_xy is not None
            if body_ok:
                f_ov = pose_overrides.get(f_idx)
                if f_ov:
                    for j_idx, (xn, yn) in f_ov.items():
                        if 0 <= j_idx < body_xy.shape[0]:
                            body_xy[j_idx, 0] = float(xn)
                            body_xy[j_idx, 1] = float(yn)
                            n_pose_overridden += 1
                    _set_body_kps(new_tm, body_xy)
                    n_pose_frames_edited += 1
                # Serialise for the JS overlay.  ``None`` for joints that
                # were missing in this frame so the renderer can hide
                # them; numeric pairs otherwise.
                body_list: list = []
                for j in range(body_xy.shape[0]):
                    x, y = body_xy[j, 0], body_xy[j, 1]
                    if np.isnan(x) or np.isnan(y):
                        body_list.append(None)
                    else:
                        body_list.append([round(float(x), 5), round(float(y), 5)])
                body_frames.append({
                    "i": f_idx, "ok": True,
                    "w": float(new_tm.get("width", 1.0)),
                    "h": float(new_tm.get("height", 1.0)),
                    "kps": body_list,
                })
            else:
                body_frames.append({"i": f_idx, "ok": False})

        # Output bundle — preserve every other field.
        new_bundle = dict(pose_data)
        if "pose_metas_original" in pose_data:
            new_bundle["pose_metas_original"] = new_metas
        else:
            new_bundle["pose_metas"] = new_metas

        # ── Gaze overrides (Slice 2) ─────────────────────────────────
        # iris_data lives at pose_data["iris_data"] as a per-frame list.
        # Each entry has ``{eye}_iris`` (pixel xy + bbox-relative info) and
        # ``{eye}_gaze`` (yaw_rad, pitch_rad, dx, dy, magnitude_norm, source).
        # We deep-copy that list, apply any user override on top of it, and
        # build the JS overlay's iris/gaze payload using the freshly-edited
        # face landmarks so the handle anchors to the correct eye centroid.
        src_iris = pose_data.get("iris_data") or []
        new_iris = deepcopy(src_iris) if isinstance(src_iris, list) else []
        n_gaze_overridden = 0
        gaze_overlay_frames: list[dict] = []

        def _eye_centroid_norm(face_norm_arr: np.ndarray | None, side: str) -> list[float] | None:
            """Mean of iBUG-68 eye landmarks in face-bbox-norm space."""
            if face_norm_arr is None: return None
            # iBUG-68: right eye = 36..41, left eye = 42..47 (subject-relative).
            rng = (36, 42) if side == "r" else (42, 48)
            sub = face_norm_arr[rng[0]:rng[1]]
            return [float(sub[:, 0].mean()), float(sub[:, 1].mean())]

        for f_idx, _new_tm in enumerate(new_metas):
            # Recompute the FINAL face-bbox-normalised landmarks for this
            # frame (post face-edit) so the JS handle is anchored exactly
            # where the user sees the eye on screen.
            face_norm_now: np.ndarray | None = None
            xy_now = _get_face_kps(_new_tm)
            if xy_now is not None:
                face_norm_now = _normalize(xy_now, _bbox_xywh(xy_now))

            # Apply override into iris_data if there's a slot for this frame.
            f_ov = gaze_overrides.get(f_idx)
            if f_ov and f_idx < len(new_iris) and isinstance(new_iris[f_idx], dict):
                for eye_key, (yaw_r, pitch_r) in f_ov.items():
                    _apply_gaze_override_to_iris_entry(
                        new_iris[f_idx], eye_key, yaw_r, pitch_r,
                    )
                    n_gaze_overridden += 1

            # Build per-frame overlay record (always — so the JS canvas
            # knows where to anchor handles even before any override).
            cur_gaze = new_iris[f_idx] if f_idx < len(new_iris) and isinstance(new_iris[f_idx], dict) else None
            def _gaze_to_pair(g):
                if not isinstance(g, dict): return None
                try:
                    return [round(float(g.get("yaw_rad", 0.0)), 5),
                            round(float(g.get("pitch_rad", 0.0)), 5)]
                except (TypeError, ValueError):
                    return None
            gaze_overlay_frames.append({
                "i": f_idx,
                "l_eye_norm": _eye_centroid_norm(face_norm_now, "l"),
                "r_eye_norm": _eye_centroid_norm(face_norm_now, "r"),
                "l_gaze": _gaze_to_pair((cur_gaze or {}).get("left_gaze")),
                "r_gaze": _gaze_to_pair((cur_gaze or {}).get("right_gaze")),
            })

        # Only write back iris_data if we actually have entries (preserve
        # the absent-field semantics for downstream nodes).
        if new_iris:
            new_bundle["iris_data"] = new_iris

        # Diagnostic numbers.
        d_mag = float(np.linalg.norm(d_norm[sel_idx], axis=1).mean()) if sel_idx else 0.0
        info = (
            f"FaceExpressionEditorUI: delta applied to {n_edited}/{n_edited + n_skipped} frames. "
            f"regions={'mouth ' if affect_mouth else ''}"
            f"{'brows ' if affect_brows else ''}"
            f"{'eyes ' if affect_eyes else ''}"
            f"{'jaw ' if affect_jaw else ''}"
            f"| strength={s:.2f} | indices={len(sel_idx)} | "
            f"mean_|d_norm|={d_mag:.4f} | overrides_applied={n_overridden} | "
            f"pose_frames_edited={n_pose_frames_edited} | "
            f"pose_joint_overrides={n_pose_overridden} | "
            f"gaze_overrides={n_gaze_overridden}"
        )
        log.info(info)

        overlay_meta = json.dumps({
            "selected": sel_idx,
            "eye_emph": list(_EYE_EMPH) if affect_eyes else [],
            "d_norm":   [[round(float(v), 6) for v in row] for row in d_norm.tolist()],
            "frames":   overlay_frames,
            "strength": s,
            # ── pose / body data for the JS pose canvas (Slice 1) ─────
            "pose": {
                "format":      "openpose_18",
                "joint_names": list(_POSE18_JOINT_NAMES),
                "edges":       [list(e) for e in _POSE18_EDGES],
                "frames":      body_frames,
            },
            # ── gaze data for the JS gaze handles (Slice 2) ───────────
            # Per-frame eye centroids in face-bbox-norm space + current
            # (yaw, pitch) in radians.  The JS overlay anchors a draggable
            # arrow tip at the centroid; the user drags it to set the
            # override.  Mapping uses ETH-XGaze-compatible angle limits.
            "gaze": {
                "max_yaw_rad":   _GAZE_MAX_YAW_RAD,
                "max_pitch_rad": _GAZE_MAX_PITCH_RAD,
                "frames":        gaze_overlay_frames,
            },
        })

        # Surface overlay_meta to the JS canvas overlay via the `ui` dict.
        # The overlay registers a `nodeType.prototype.onExecuted` handler that
        # reads `message.overlay_meta[0]` and renders draggable landmarks.
        return {
            "ui":     {"overlay_meta": [overlay_meta]},
            "result": (new_bundle, info, overlay_meta),
        }
