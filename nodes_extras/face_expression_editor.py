# Copyright 2026 Code2Collapse / WanAnimatePreprocessV2.
# Licensed under the Apache License, Version 2.0.
#
# Third-party data / models reused by this node:
#   * iBUG-68 / COCO-WholeBody 68-point face landmark convention
#     (CC-BY-4.0).
#   * AAPoseMeta container (MIT, from the upstream Wan-Animate codebase).
#
# This node does not bundle or call any face-recognition / gaze-estimation
# datasets — it operates purely on 2-D keypoints that already live in the
# POSEDATA bundle produced by Pose+Face Detection (V2).

"""WanFaceExpressionEditorV2 — blend a driving face shape toward a
reference expression.

Use case
========
The user already has::

    Pose+Face Detection (V2) ──► POSEDATA (driving sequence, N frames)
    Pose+Face Detection (V2) ──► POSEDATA (single reference frame
                                            with the *desired* mouth
                                            shape / brow lift / blink)

This node takes both bundles plus an optional reference IMAGE (only used
for size/visualisation; no extra detector is invoked) and produces a
modified POSEDATA where each frame's selected face regions
(``mouth``, ``brows``, ``eyes``, ``jaw``) are linearly interpolated
toward the reference's, in **face-bbox-normalised space** so head pose
and scale are preserved.

iBUG-68 face landmark slices used (consistent with
``pose_utils.pose2d_utils.AAPoseMeta.kps_face``):

* jaw       : indices 5..11  (chin / lower jaw subset)
* brows     : indices 17..26 (both brows)
* eyes      : indices 36..47 (both eye outlines)
* eye_emph  : indices 37, 38, 43, 44 (upper eyelids — applied at 1.5x)
* mouth     : indices 48..67 (full mouth)
"""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np

try:
    import torch  # noqa: F401  (POSEDATA frames may carry tensors)
except Exception:  # pragma: no cover
    torch = None

log = logging.getLogger(__name__)


# ── iBUG-68 region slices ─────────────────────────────────────────────
_JAW_IDX   = list(range(5, 12))                  # 5..11 inclusive
_BROWS_IDX = list(range(17, 27))                 # 17..26
_EYES_IDX  = list(range(36, 48))                 # 36..47
_EYE_EMPH  = [37, 38, 43, 44]                    # upper-eyelid emphasis
_MOUTH_IDX = list(range(48, 68))                 # 48..67


def _bbox_xywh(kps: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box (x, y, w, h) of a (N, 2) keypoint array."""
    if kps is None or len(kps) == 0:
        return 0.0, 0.0, 1.0, 1.0
    mn = kps.min(axis=0)
    mx = kps.max(axis=0)
    w = max(float(mx[0] - mn[0]), 1e-6)
    h = max(float(mx[1] - mn[1]), 1e-6)
    return float(mn[0]), float(mn[1]), w, h


def _normalize(kps: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    """Map (N, 2) pixel keypoints into the 0..1 bbox-local space."""
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
                      affect_eyes: bool, affect_jaw: bool) -> tuple[list[int], list[int]]:
    """Return (region indices, indices receiving 1.5x emphasis)."""
    sel: list[int] = []
    if affect_mouth: sel.extend(_MOUTH_IDX)
    if affect_brows: sel.extend(_BROWS_IDX)
    if affect_eyes:
        sel.extend(_EYES_IDX)
        emph = list(_EYE_EMPH)
    else:
        emph = []
    if affect_jaw:   sel.extend(_JAW_IDX)
    # de-duplicate while preserving order
    seen = set(); out = []
    for i in sel:
        if i in seen: continue
        seen.add(i); out.append(i)
    return out, emph


def _get_face_kps(meta) -> np.ndarray | None:
    """Extract the (N, 2) face keypoint array from a humanapi-style meta dict
    (the format used in ``POSEDATA["pose_metas_original"]``).

    Returns None if the meta has no face data."""
    if not isinstance(meta, dict):
        return None
    arr = meta.get("keypoints_face")
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 68:
        return None
    # The bundle stores keypoints normalised to [0, 1] of the source frame.
    w = float(meta.get("width", 1.0))
    h = float(meta.get("height", 1.0))
    xy = arr[:68, :2].copy()
    xy[:, 0] *= w
    xy[:, 1] *= h
    return xy


def _set_face_kps(meta: dict, xy_pixels: np.ndarray) -> None:
    """Write a (68, 2) pixel-space face keypoint array back into a humanapi
    meta dict in its native normalised representation."""
    arr = np.asarray(meta["keypoints_face"], dtype=np.float32).copy()
    w = max(float(meta.get("width", 1.0)), 1e-6)
    h = max(float(meta.get("height", 1.0)), 1e-6)
    arr[:68, 0] = xy_pixels[:, 0] / w
    arr[:68, 1] = xy_pixels[:, 1] / h
    meta["keypoints_face"] = arr


class WanFaceExpressionEditorV2:
    """Blend selected face regions of every frame toward a reference
    expression, preserving head pose and face scale.

    Inputs
    ------
    pose_data            : POSEDATA  (driving sequence, required)
    reference_pose_data  : POSEDATA  (single-frame reference, required)
    blend_strength       : FLOAT     0..1  (0=no change, 1=full reference)
    affect_mouth/brows/eyes/jaw : BOOLEAN

    Outputs
    -------
    pose_data            : POSEDATA  (modified in a deep copy)
    info                 : STRING    diagnostic summary
    """

    CATEGORY = "Code2Collapse/WanAnimateV2/Face"
    DESCRIPTION = (
        "Blend selected face regions (mouth/brows/eyes/jaw) of every frame "
        "in a driving POSEDATA toward a single-frame reference POSEDATA. "
        "Works in face-bbox-normalised iBUG-68 space so head pose and scale "
        "stay intact. blend_strength=0 ⇒ no change, 1 ⇒ full reference."
    )
    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "info")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data":           ("POSEDATA", {"tooltip": "Driving sequence (multi-frame)."}),
                "reference_pose_data": ("POSEDATA", {"tooltip": "Single-frame reference whose face shape is the target."}),
                "blend_strength":      ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "affect_mouth":        ("BOOLEAN", {"default": True}),
                "affect_brows":        ("BOOLEAN", {"default": True}),
                "affect_eyes":         ("BOOLEAN", {"default": False}),
                "affect_jaw":          ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional: reference image (display only — not used for detection)."}),
            },
        }

    def run(self, pose_data, reference_pose_data, blend_strength,
            affect_mouth, affect_brows, affect_eyes, affect_jaw,
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

        # Pick the first reference frame whose face keypoints are valid.
        ref_xy_px = None
        ref_bbox  = None
        for rm in ref_metas:
            xy = _get_face_kps(rm)
            if xy is None:
                continue
            ref_xy_px = xy
            ref_bbox  = _bbox_xywh(xy)
            break
        if ref_xy_px is None:
            raise ValueError("reference_pose_data contains no usable 'keypoints_face' in any frame.")
        ref_norm = _normalize(ref_xy_px, ref_bbox)

        sel_idx, emph_idx = _selected_indices(affect_mouth, affect_brows, affect_eyes, affect_jaw)
        if not sel_idx:
            return (pose_data, "FaceExpressionEditor: no region selected — pass-through.")

        s = float(np.clip(blend_strength, 0.0, 1.0))

        # Deep-copy so we don't mutate upstream caches.
        new_metas: list = []
        n_edited = 0
        n_skipped = 0
        for tm in target_metas:
            new_tm = deepcopy(tm)
            xy = _get_face_kps(new_tm)
            if xy is None:
                n_skipped += 1
                new_metas.append(new_tm)
                continue
            tgt_bbox = _bbox_xywh(xy)
            tgt_norm = _normalize(xy, tgt_bbox)

            for i in sel_idx:
                if i >= tgt_norm.shape[0] or i >= ref_norm.shape[0]:
                    continue
                alpha = s * (1.5 if i in emph_idx else 1.0)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                tgt_norm[i] = (1.0 - alpha) * tgt_norm[i] + alpha * ref_norm[i]

            new_xy = _denormalize(tgt_norm, tgt_bbox)
            _set_face_kps(new_tm, new_xy)
            n_edited += 1
            new_metas.append(new_tm)

        # Build the output bundle — preserve every other field.
        new_bundle = dict(pose_data)
        # Always overwrite *_original*; if a 'pose_metas' (AAPoseMeta list)
        # is present we leave it alone (downstream visualisation usually
        # rebuilds it from *_original*).
        if "pose_metas_original" in pose_data:
            new_bundle["pose_metas_original"] = new_metas
        else:
            new_bundle["pose_metas"] = new_metas

        info = (
            f"FaceExpressionEditor: blended {n_edited}/{n_edited + n_skipped} frames. "
            f"regions={'mouth ' if affect_mouth else ''}"
            f"{'brows ' if affect_brows else ''}"
            f"{'eyes ' if affect_eyes else ''}"
            f"{'jaw ' if affect_jaw else ''}"
            f"strength={s:.2f} | indices_blended={len(sel_idx)} "
            f"(emphasis={len(emph_idx)}) | ref_bbox_wh="
            f"({ref_bbox[2]:.1f}, {ref_bbox[3]:.1f})"
        )
        log.info(info)
        return (new_bundle, info)
