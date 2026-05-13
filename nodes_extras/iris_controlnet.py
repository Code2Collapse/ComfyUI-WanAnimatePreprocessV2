"""WanIrisControlNetV2 — Render an iris/gaze ControlNet conditioning image.

Consumes the iris_data JSON emitted by PoseAndFaceDetectionV2 (or any JSON
matching the schema below) and renders a per-frame conditioning image that
encodes:
    - Eye region masks (filled ellipses, anatomically placed)
    - Iris discs at exact pupil pixel coords
    - Gaze direction arrows (length proportional to magnitude)
    - Soft gaze-target heatmap (Gaussian centred where the user is looking)

This is a real visual conditioning image suitable as input to a ControlNet
(or T2I-Adapter) trained on eye/gaze control — same format and value range as
ComfyUI's other IMAGE-typed ControlNet inputs.

Input JSON schema (per frame; missing fields tolerated):
    {
      "frame": int,
      "right_pupil_xy": [x, y],
      "left_pupil_xy":  [x, y],
      "right_eye_bbox": [x1, y1, x2, y2],
      "left_eye_bbox":  [x1, y1, x2, y2],
      "gaze_yaw_rad":   float,
      "gaze_pitch_rad": float,
      "right_yaw_rad": float, "right_pitch_rad": float,
      "left_yaw_rad":  float, "left_pitch_rad":  float,
      "iris_confidence": float
    }

The node also accepts a `face_bboxes` BBOX list — if eye bboxes are absent
we estimate eye regions as fractions of the face bbox.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import torch


def _draw_gradient_disc(canvas: np.ndarray, cx: int, cy: int, r: int,
                        color_outer: tuple[float, float, float],
                        color_inner: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """Soft radial gradient disc — anti-aliased, blends in-place."""
    H, W = canvas.shape[:2]
    x0, x1 = max(0, cx - r - 1), min(W, cx + r + 2)
    y0, y1 = max(0, cy - r - 1), min(H, cy + r + 2)
    if x0 >= x1 or y0 >= y1 or r <= 0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = np.clip(1.0 - dist / max(r, 1), 0.0, 1.0)
    inner_w = mask ** 3
    outer_w = mask - inner_w
    for c in range(3):
        canvas[y0:y1, x0:x1, c] = (
            canvas[y0:y1, x0:x1, c] * (1 - mask)
            + color_inner[c] * inner_w
            + color_outer[c] * outer_w
        )


def _draw_gaussian(canvas: np.ndarray, cx: float, cy: float, sigma: float,
                   color: tuple[float, float, float], intensity: float = 1.0):
    H, W = canvas.shape[:2]
    r = int(max(1, sigma * 3))
    x0, x1 = max(0, int(cx) - r), min(W, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(H, int(cy) + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * max(sigma, 1) ** 2)) * intensity
    for c in range(3):
        canvas[y0:y1, x0:x1, c] = np.maximum(canvas[y0:y1, x0:x1, c], g * color[c])


def _draw_filled_ellipse(canvas: np.ndarray, cx: int, cy: int, rx: int, ry: int,
                         color: tuple[float, float, float], alpha: float = 1.0):
    import cv2
    overlay = canvas.copy()
    cv2.ellipse(overlay, (cx, cy), (max(1, rx), max(1, ry)), 0, 0, 360, color, -1, cv2.LINE_AA)
    np.copyto(canvas, canvas * (1 - alpha) + overlay * alpha)


def _draw_arrow(canvas: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                color: tuple[float, float, float], thickness: int = 2):
    import cv2
    cv2.arrowedLine(canvas, (int(x0), int(y0)), (int(x1), int(y1)),
                    color, thickness, cv2.LINE_AA, tipLength=0.3)


def _eye_bbox_from_face(face_bbox: tuple[float, float, float, float],
                       side: str) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = face_bbox
    w = x2 - x1
    h = y2 - y1
    eye_y = y1 + h * 0.38
    eye_h = h * 0.10
    if side == "right":
        ex1 = x1 + w * 0.18
        ex2 = x1 + w * 0.42
    else:
        ex1 = x1 + w * 0.58
        ex2 = x1 + w * 0.82
    return int(ex1), int(eye_y - eye_h / 2), int(ex2), int(eye_y + eye_h / 2)


class WanIrisControlNetV2:
    CATEGORY = "WanAnimatePreprocess_V2/Gaze"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("control_image", "iris_mask", "info")
    DESCRIPTION = "Render an iris/gaze ControlNet conditioning image from iris_data JSON. Eye masks, iris discs, gaze arrows, gaze-target heatmap."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iris_data_json": ("STRING", {"multiline": True, "default": "[]"}),
                "image_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "image_height": ("INT", {"default": 1024, "min": 64, "max": 8192}),
                "render_style": (["full", "iris_only", "heatmap_only", "mask_only"],
                                 {"default": "full"}),
                "iris_radius_px": ("INT", {"default": 6, "min": 1, "max": 80}),
                "arrow_scale_px": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 400.0,
                                              "step": 5.0,
                                              "tooltip": "Pixels of arrow per radian of gaze."}),
                "heatmap_sigma_px": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 400.0}),
                "background": (["black", "white", "neutral_grey"], {"default": "black"}),
            },
            "optional": {
                "face_bboxes": ("BBOX",),
                "reference_image": ("IMAGE", {"tooltip": "If given, use its (H,W,B) and overlay onto it at low alpha."}),
                "overlay_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def execute(self, iris_data_json, image_width, image_height, render_style,
                iris_radius_px, arrow_scale_px, heatmap_sigma_px, background,
                face_bboxes=None, reference_image=None, overlay_alpha=0.0):
        try:
            data = json.loads(iris_data_json) if iris_data_json.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"WanIrisControlNetV2: invalid iris_data_json: {e}")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = []

        # Determine canvas size
        if reference_image is not None and reference_image.numel() > 0:
            B, H, W, _ = reference_image.shape
            n_frames = max(B, len(data) or 1)
        else:
            H, W = int(image_height), int(image_width)
            n_frames = max(1, len(data))

        bg_lut = {"black": 0.0, "white": 1.0, "neutral_grey": 0.5}
        bg = bg_lut.get(background, 0.0)

        # face_bboxes optional list of [x1,y1,x2,y2] per frame
        bb_list: list[Optional[tuple[float, float, float, float]]] = []
        if face_bboxes is not None:
            try:
                if hasattr(face_bboxes, "__iter__"):
                    for b in face_bboxes:
                        if b is None:
                            bb_list.append(None)
                        else:
                            arr = list(b)
                            if len(arr) >= 4:
                                bb_list.append((float(arr[0]), float(arr[1]),
                                                float(arr[2]), float(arr[3])))
                            else:
                                bb_list.append(None)
            except Exception:
                pass

        out_frames = []
        out_masks = []
        for i in range(n_frames):
            if reference_image is not None and i < reference_image.shape[0]:
                base = reference_image[i].detach().cpu().numpy().astype(np.float32).copy()
                if overlay_alpha < 1.0:
                    base = base * overlay_alpha + bg * (1 - overlay_alpha)
            else:
                base = np.full((H, W, 3), bg, dtype=np.float32)
            mask = np.zeros((H, W), dtype=np.float32)

            entry = data[i] if i < len(data) else {}
            if not isinstance(entry, dict):
                entry = {}

            # Determine eye bboxes
            r_eye = entry.get("right_eye_bbox")
            l_eye = entry.get("left_eye_bbox")
            face_bbox = bb_list[i] if i < len(bb_list) else None
            if r_eye is None and face_bbox is not None:
                r_eye = list(_eye_bbox_from_face(face_bbox, "right"))
            if l_eye is None and face_bbox is not None:
                l_eye = list(_eye_bbox_from_face(face_bbox, "left"))

            # Render eye masks (white-ish ellipse for the sclera)
            if render_style in ("full", "mask_only"):
                for eb in (r_eye, l_eye):
                    if eb is None:
                        continue
                    cx = int((eb[0] + eb[2]) / 2)
                    cy = int((eb[1] + eb[3]) / 2)
                    rx = max(2, int((eb[2] - eb[0]) / 2))
                    ry = max(2, int((eb[3] - eb[1]) / 2))
                    _draw_filled_ellipse(base, cx, cy, rx, ry, (0.92, 0.92, 0.88), alpha=0.95)
                    # mask channel
                    import cv2
                    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

            # Iris discs
            r_p = entry.get("right_pupil_xy")
            l_p = entry.get("left_pupil_xy")
            if render_style in ("full", "iris_only"):
                if isinstance(r_p, (list, tuple)) and len(r_p) >= 2:
                    _draw_gradient_disc(base, int(r_p[0]), int(r_p[1]),
                                        int(iris_radius_px), (0.15, 0.35, 0.85))
                if isinstance(l_p, (list, tuple)) and len(l_p) >= 2:
                    _draw_gradient_disc(base, int(l_p[0]), int(l_p[1]),
                                        int(iris_radius_px), (0.15, 0.35, 0.85))

            # Gaze arrows
            if render_style == "full" and arrow_scale_px > 0:
                # Prefer per-eye yaw/pitch, fall back to single gaze
                def _eye_gaze(entry, side):
                    y = entry.get(f"{side}_yaw_rad", entry.get("gaze_yaw_rad"))
                    p = entry.get(f"{side}_pitch_rad", entry.get("gaze_pitch_rad"))
                    return y, p
                for pupil, side in ((r_p, "right"), (l_p, "left")):
                    if not (isinstance(pupil, (list, tuple)) and len(pupil) >= 2):
                        continue
                    yaw, pit = _eye_gaze(entry, side)
                    if yaw is None or pit is None:
                        continue
                    dx = math.sin(float(yaw)) * arrow_scale_px
                    dy = -math.sin(float(pit)) * arrow_scale_px
                    _draw_arrow(base, pupil[0], pupil[1],
                                pupil[0] + dx, pupil[1] + dy,
                                (1.0, 0.5, 0.1), thickness=2)

            # Gaze-target Gaussian heatmap
            if render_style in ("full", "heatmap_only"):
                yaw = entry.get("gaze_yaw_rad")
                pit = entry.get("gaze_pitch_rad")
                # use midpoint of two pupils as origin
                origins = [p for p in (r_p, l_p) if isinstance(p, (list, tuple)) and len(p) >= 2]
                if origins and yaw is not None and pit is not None:
                    ox = float(np.mean([o[0] for o in origins]))
                    oy = float(np.mean([o[1] for o in origins]))
                    tx = ox + math.sin(float(yaw)) * arrow_scale_px * 2.0
                    ty = oy - math.sin(float(pit)) * arrow_scale_px * 2.0
                    _draw_gaussian(base, tx, ty, heatmap_sigma_px,
                                   (0.95, 0.35, 0.05), intensity=0.75)

            out_frames.append(np.clip(base, 0, 1).astype(np.float32))
            out_masks.append(mask)

        out_img = torch.from_numpy(np.stack(out_frames, axis=0))
        out_mask = torch.from_numpy(np.stack(out_masks, axis=0))
        info = json.dumps({
            "frames": n_frames,
            "image_size": [W, H],
            "render_style": render_style,
            "iris_radius_px": int(iris_radius_px),
            "had_iris_data": bool(data),
        })
        return (out_img, out_mask, info)
