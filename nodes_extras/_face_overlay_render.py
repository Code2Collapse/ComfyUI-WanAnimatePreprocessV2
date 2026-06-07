"""Overlay renderer for ``WanFaceController3DV2`` outputs (Plan P1.C).

Produces ComfyUI-compatible IMAGE tensors that show, on a dark canvas:

  • The 68 iBUG face landmarks (mauve dots).
  • The 18 OpenPose body keypoints (green dots + blue bones).
  • Gaze arrows for each eye (red/green) when iris data is present.
  • A small frame-index badge bottom-left.

Used to render ``preview_image`` (single frame) and ``overlay_video``
(per-frame batch). Coordinates in the meta dicts are image-normalised
([0..1] across the source image), so the renderer can paint at any
square ``preview_size`` without re-discovering the source resolution.

Zero training, pure rasterisation — uses only numpy + Pillow + torch
which are already required by ComfyUI itself.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math
import numpy as np
import torch
from PIL import Image, ImageDraw

# ----------------------------------------------------------------------
# Palette (matches Catppuccin tokens used in face_controller_3d.js)
# ----------------------------------------------------------------------
_PAL = {
    "bg":            (14, 14, 22),       # crust
    "face_dot":      (203, 166, 247),    # mauve
    "face_eye_emph": (250, 179, 135),    # peach
    "body_joint":    (166, 227, 161),    # green
    "body_bone":     (116, 199, 236),    # blue
    "gaze_l":        (243, 139, 168),    # red
    "gaze_r":        (148, 226, 213),    # teal
    "text":          (205, 214, 244),    # fg
    "badge_bg":      (26, 26, 35),       # surface0
}

_EYE_EMPH_IDX = (37, 38, 43, 44)


def _to_px(
    xy_norm: np.ndarray,
    out_w: int,
    out_h: int,
    src_w: float,
    src_h: float,
) -> np.ndarray:
    """Map normalised coords from a source aspect ratio to the square
    preview canvas with letterboxing, preserving aspect."""
    sw = max(float(src_w), 1e-6)
    sh = max(float(src_h), 1e-6)
    src_aspect = sw / sh
    dst_aspect = out_w / out_h
    if src_aspect >= dst_aspect:
        # Pillarbox vertically.
        scale = out_w / sw
        new_h = sh * scale
        ox = 0.0
        oy = (out_h - new_h) * 0.5
        sx = sw * scale
        sy = new_h
    else:
        scale = out_h / sh
        new_w = sw * scale
        ox = (out_w - new_w) * 0.5
        oy = 0.0
        sx = new_w
        sy = sh * scale
    out = np.empty_like(xy_norm)
    out[:, 0] = xy_norm[:, 0] * sx + ox
    out[:, 1] = xy_norm[:, 1] * sy + oy
    return out


def render_overlay_frame(
    meta: dict,
    iris_entry: Optional[dict],
    body_xy: Optional[np.ndarray],
    out_size: int,
    edges: Sequence[Tuple[int, int]],
    *,
    face_norm: Optional[np.ndarray] = None,
    frame_idx: int = 0,
    n_frames: int = 1,
    max_gaze_yaw_rad: float = math.radians(30.0),
    max_gaze_pitch_rad: float = math.radians(25.0),
) -> np.ndarray:
    """Render one overlay frame and return a uint8 HxWx3 numpy array.

    ``meta`` must carry ``width``/``height`` so we can preserve the
    source aspect ratio inside the square output.

    ``face_norm`` (68, 2) is the IMAGE-normalised face landmarks from
    ``_read_face_normalised(meta)`` — passed in so the caller can use a
    deep-copied / edited version without re-reading.
    """
    W = H = int(out_size)
    img = Image.new("RGB", (W, H), _PAL["bg"])
    drw = ImageDraw.Draw(img, "RGBA")

    from ._face_helpers import _meta_height, _meta_width

    src_w = max(_meta_width(meta), 1e-6)
    src_h = max(_meta_height(meta), 1e-6)

    # ── Body bones first (drawn behind dots) ────────────────────────
    if body_xy is not None and body_xy.shape[0] >= 18:
        # Filter NaN joints.
        ok_mask = ~(np.isnan(body_xy[:, 0]) | np.isnan(body_xy[:, 1]))
        body_filled = np.where(np.stack([ok_mask, ok_mask], 1),
                               body_xy, np.zeros_like(body_xy))
        body_px = _to_px(body_filled, W, H, src_w, src_h)
        for (a, b) in edges:
            if a >= body_xy.shape[0] or b >= body_xy.shape[0]:
                continue
            if not (ok_mask[a] and ok_mask[b]):
                continue
            x1, y1 = float(body_px[a, 0]), float(body_px[a, 1])
            x2, y2 = float(body_px[b, 0]), float(body_px[b, 1])
            drw.line([(x1, y1), (x2, y2)], fill=_PAL["body_bone"] + (220,),
                     width=max(2, W // 256))
        joint_r = max(2, W // 192)
        for j in range(body_xy.shape[0]):
            if not ok_mask[j]:
                continue
            x, y = float(body_px[j, 0]), float(body_px[j, 1])
            drw.ellipse([(x - joint_r, y - joint_r), (x + joint_r, y + joint_r)],
                        fill=_PAL["body_joint"])

    # ── Face landmarks ──────────────────────────────────────────────
    if face_norm is not None and face_norm.shape == (68, 2):
        face_px = _to_px(face_norm, W, H, src_w, src_h)
        dot_r = max(1, W // 384)
        for i in range(68):
            x, y = float(face_px[i, 0]), float(face_px[i, 1])
            colour = _PAL["face_eye_emph"] if i in _EYE_EMPH_IDX else _PAL["face_dot"]
            drw.ellipse([(x - dot_r, y - dot_r), (x + dot_r, y + dot_r)],
                        fill=colour)

        # ── Gaze arrows from iris_entry (yaw/pitch in radians) ─────
        if isinstance(iris_entry, dict):
            for side, key, colour in (
                ("l", "left_gaze",  _PAL["gaze_l"]),
                ("r", "right_gaze", _PAL["gaze_r"]),
            ):
                g = iris_entry.get(key)
                if not isinstance(g, dict):
                    continue
                try:
                    yaw   = float(g.get("yaw_rad", 0.0))
                    pitch = float(g.get("pitch_rad", 0.0))
                except (TypeError, ValueError):
                    continue
                # Eye centre in face-bbox-normalised → image px.
                eye_idx = range(42, 48) if side == "l" else range(36, 42)
                ex_norm = float(face_norm[list(eye_idx), 0].mean())
                ey_norm = float(face_norm[list(eye_idx), 1].mean())
                ex, ey = _to_px(
                    np.array([[ex_norm, ey_norm]], dtype=np.float32),
                    W, H, src_w, src_h,
                )[0]
                # Arrow length scales with face bbox width in px.
                fbx_min = float(face_norm[:, 0].min())
                fbx_max = float(face_norm[:, 0].max())
                bbox_px = max(8.0, (fbx_max - fbx_min) * (W if src_w >= src_h * (W/H) else (W * src_w / src_h)) * 0.30)
                # Clamp yaw/pitch into displayable range.
                u = float(np.clip(yaw   / max_gaze_yaw_rad,   -1.0, 1.0))
                v = float(np.clip(pitch / max_gaze_pitch_rad, -1.0, 1.0))
                dx = u * bbox_px
                dy = -v * bbox_px  # image y grows downward; pitch up = arrow up
                drw.line([(ex, ey), (ex + dx, ey + dy)],
                         fill=colour + (255,), width=max(2, W // 256))
                drw.ellipse([(ex - 2, ey - 2), (ex + 2, ey + 2)], fill=colour)

    # ── Frame badge bottom-left ─────────────────────────────────────
    badge = f"f {frame_idx + 1}/{n_frames}"
    tx, ty = 8, H - 18
    drw.rectangle([(tx - 3, ty - 1), (tx + 7 * len(badge), ty + 14)],
                  fill=_PAL["badge_bg"] + (200,))
    drw.text((tx, ty), badge, fill=_PAL["text"])

    return np.asarray(img, dtype=np.uint8)


def render_to_tensor(frame_u8: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 → float32 1xHxWx3 in [0, 1] (ComfyUI IMAGE)."""
    arr = frame_u8.astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def stack_to_batch(frames_u8: list) -> torch.Tensor:
    """List of uint8 HxWx3 → float32 NxHxWx3 in [0, 1]."""
    if not frames_u8:
        # ComfyUI expects at least 1 image; return a 1x8x8x3 black.
        return torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    arr = np.stack(frames_u8, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr)
