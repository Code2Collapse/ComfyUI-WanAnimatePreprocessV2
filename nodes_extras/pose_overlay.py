"""
pose_overlay.py — verification-overlay drawing helpers (NOT a node).

The comprehensive pose/face/gaze verification overlay used to be a separate
node (``WanPoseOverlayV2``). It is now folded directly into
``PoseAndFaceDetectionV2``'s built-in ``debug_image`` output, so there is no
second node to wire — the detection node's debug image already carries the
full body skeleton + face points + hands + iris/gaze.

This module keeps only the reusable drawing primitives. ``draw_body_skeleton_rgb``
is called from ``nodes.py``'s debug-overlay loop (which already draws the
face landmarks / eye contours / iris / gaze via ``draw_debug_overlay``); this
adds the OpenPose-18 body skeleton + hands on top.

Pure cv2/numpy, standalone — imports nothing from third_party.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# OpenPose-18 limbs (index pairs).
_OP18_LIMBS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),        # neck→shoulders→arms
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),  # neck→hips→legs
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),          # neck→nose→eyes→ears
]
# Per-limb colours, in RGB (the detection node's debug frame is RGB uint8).
_LIMB_COLORS_RGB = [
    (255, 100, 0), (255, 180, 0), (200, 255, 0), (80, 255, 0), (0, 255, 80), (0, 255, 200),
    (0, 200, 255), (0, 120, 255), (0, 0, 255), (120, 0, 255), (200, 0, 255), (255, 0, 200),
    (255, 0, 120), (255, 0, 255), (255, 0, 180), (180, 80, 255), (150, 150, 255),
]
_FACE_OP18 = {0: "nose", 14: "r_eye", 15: "l_eye", 16: "r_ear", 17: "l_ear"}


def _xy(pt):
    """Return (x, y, conf) from an OP18 entry, or None if missing/invalid."""
    if pt is None:
        return None
    a = np.asarray(pt, dtype=np.float32).reshape(-1)
    if a.shape[0] < 2 or not np.isfinite(a[:2]).all():
        return None
    c = float(a[2]) if a.shape[0] > 2 else 1.0
    return float(a[0]), float(a[1]), c


def _draw_ptset(cv2, frame, kps, W, H, color, radius):
    """Draw a point set (face / hand) as small dots; auto-detect normalized coords.

    ``color`` is interpreted in the frame's own channel order (RGB here).
    """
    if kps is None:
        return 0
    arr = np.asarray(kps, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return 0
    finite = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
    norm = finite.size > 0 and float(np.nanmax(np.abs(finite[:, :2]))) <= 1.5
    n = 0
    for row in arr:
        x, y = float(row[0]), float(row[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if norm:
            x, y = x * W, y * H
        if x <= 0 and y <= 0:
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), radius, color, -1, cv2.LINE_AA)
        n += 1
    return n


def draw_body_skeleton_rgb(cv2, vis_rgb, meta, W, H, line_width=2, point_radius=4,
                           draw_hands=True):
    """Draw the OP18 body skeleton (+ optional hands) onto an RGB uint8 frame, in place.

    Args:
        cv2:         the imported cv2 module (passed in to avoid a hard import here).
        vis_rgb:     (H, W, 3) uint8 RGB frame — modified in place.
        meta:        a pose meta dict (``pose_metas_original`` entry) carrying
                     ``keypoints_body`` (OP18) and optionally
                     ``keypoints_left_hand`` / ``keypoints_right_hand``.
        W, H:        pixel dimensions.
    Returns:
        1 if a body skeleton was drawn, else 0.
    """
    if not isinstance(meta, dict):
        return 0
    kb = meta.get("keypoints_body")
    pts = {}
    if kb is not None:
        arr = list(kb)
        # Auto-detect normalized (0..1) vs pixel coords.
        norm = True
        for p in arr:
            v = _xy(p)
            if v and (abs(v[0]) > 1.5 or abs(v[1]) > 1.5):
                norm = False
                break
        for i, p in enumerate(arr[:18]):
            v = _xy(p)
            if v is None:
                continue
            x, y, c = v
            if norm:
                x, y = x * W, y * H
            pts[i] = (int(round(x)), int(round(y)), c)

    drawn = 0
    if pts:
        for (a, b), col in zip(_OP18_LIMBS, _LIMB_COLORS_RGB):
            if a in pts and b in pts:
                cv2.line(vis_rgb, pts[a][:2], pts[b][:2], col, line_width, cv2.LINE_AA)
        for i, (x, y, _c) in pts.items():
            col = (255, 255, 255) if i in _FACE_OP18 else (255, 255, 0)  # RGB
            cv2.circle(vis_rgb, (x, y), point_radius, col, -1, cv2.LINE_AA)
        drawn = 1

    if draw_hands:
        _draw_ptset(cv2, vis_rgb, meta.get("keypoints_left_hand"), W, H, (120, 255, 0), 2)
        _draw_ptset(cv2, vis_rgb, meta.get("keypoints_right_hand"), W, H, (0, 255, 140), 2)
    return drawn
