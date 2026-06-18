"""WanHeadPose6DoFV2 — 6-DoF head pose via cv2.solvePnP.

Uses an iBUG-68 landmark model. Six rigid correspondences are sufficient
for a stable PnP solution; we use 14 for accuracy:

    Index  Name                           Canonical 3D (mm, head frame)
    -----  -----------------------------  ----------------------------
     30    nose tip                       ( 0.000,  0.000,  0.000)
      8    chin                           ( 0.000, -7.500, -1.500)
     36    left  eye left corner          (-4.500,  3.500, -2.500)
     45    right eye right corner         ( 4.500,  3.500, -2.500)
     39    left  eye right corner         (-1.500,  3.500, -2.000)
     42    right eye left corner          ( 1.500,  3.500, -2.000)
     48    left  mouth corner             (-2.500, -3.000, -2.000)
     54    right mouth corner             ( 2.500, -3.000, -2.000)
      6    chin side left                 (-5.500, -6.500, -4.000)
     10    chin side right                ( 5.500, -6.500, -4.000)
     27    nose bridge top                ( 0.000,  1.500,  0.500)
      0    jaw right                      (-7.000,  0.500, -4.000)
     16    jaw left                       ( 7.000,  0.500, -4.000)
     33    nose bottom                    ( 0.000, -1.000,  0.500)

Camera matrix: focal = max(W, H); principal point = image centre.
Distortion assumed zero (rectified input).

Inputs:
  - landmarks_json : STRING -- list per frame, each frame is a list of [x,y]
                     in image pixel coords for iBUG-68 face landmarks, OR a
                     list of face dicts with "kps_face" key.
  - image          : IMAGE  -- used only to read (H, W); optional if
                     image_size_override supplied.

Outputs:
  - poses_json     : STRING -- per-frame {yaw_deg, pitch_deg, roll_deg,
                     tx, ty, tz, rvec, tvec, reprojection_err_px,
                     reliable: bool}
  - overlay        : IMAGE  -- input image with projected XYZ axes drawn
                     at the nose tip for visual sanity.
  - yaw            : FLOAT  -- last-frame yaw degrees (UI convenience).
  - pitch          : FLOAT
  - roll           : FLOAT
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs


# Canonical 3D model points (iBUG-68 indices, mm)
_LM_INDICES = [30, 8, 36, 45, 39, 42, 48, 54, 6, 10, 27, 0, 16, 33]
_MODEL_3D = np.array([
    [ 0.0,  0.0,   0.0],   # 30 nose tip
    [ 0.0, -7.5,  -1.5],   # 8  chin
    [-4.5,  3.5,  -2.5],   # 36 left eye left corner
    [ 4.5,  3.5,  -2.5],   # 45 right eye right corner
    [-1.5,  3.5,  -2.0],   # 39 left eye right corner
    [ 1.5,  3.5,  -2.0],   # 42 right eye left corner
    [-2.5, -3.0,  -2.0],   # 48 left mouth corner
    [ 2.5, -3.0,  -2.0],   # 54 right mouth corner
    [-5.5, -6.5,  -4.0],   # 6  chin side left
    [ 5.5, -6.5,  -4.0],   # 10 chin side right
    [ 0.0,  1.5,   0.5],   # 27 nose bridge top
    [-7.0,  0.5,  -4.0],   # 0  jaw right
    [ 7.0,  0.5,  -4.0],   # 16 jaw left
    [ 0.0, -1.0,   0.5],   # 33 nose bottom
], dtype=np.float64)


def _rvec_to_euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX convention: yaw (Y), pitch (X), roll (Z)."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    else:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(y), math.degrees(x), math.degrees(z)


def _solve_pnp_one_frame(landmarks_2d: np.ndarray, camera_matrix: np.ndarray):
    """landmarks_2d: (N, 2) pixel coords for the 14 indices in _LM_INDICES order."""
    import cv2
    if landmarks_2d.shape[0] != _MODEL_3D.shape[0]:
        return None
    object_pts = _MODEL_3D.astype(np.float64)
    image_pts = landmarks_2d.astype(np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(object_pts, image_pts, camera_matrix, dist,
                                      flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
    R, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rvec_to_euler_deg(R)
    # reprojection error
    proj, _ = cv2.projectPoints(object_pts, rvec, tvec, camera_matrix, dist)
    err = float(np.linalg.norm(proj.reshape(-1, 2) - image_pts, axis=1).mean())
    return {
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "roll_deg": float(roll),
        "rvec": rvec.flatten().tolist(),
        "tvec": tvec.flatten().tolist(),
        "tx": float(tvec.flatten()[0]),
        "ty": float(tvec.flatten()[1]),
        "tz": float(tvec.flatten()[2]),
        "reprojection_err_px": err,
        "reliable": err < 8.0,
    }


def _draw_axes(img_u8: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
               camera_matrix: np.ndarray, origin_2d: tuple[float, float],
               axis_len: float = 6.0) -> np.ndarray:
    import cv2
    pts = np.float64([[axis_len, 0, 0], [0, axis_len, 0], [0, 0, -axis_len], [0, 0, 0]])
    proj, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, np.zeros((4, 1)))
    proj = proj.reshape(-1, 2).astype(int)
    o = (int(origin_2d[0]), int(origin_2d[1]))
    out = img_u8.copy()
    cv2.line(out, o, tuple(proj[0]), (255, 0, 0), 2, cv2.LINE_AA)  # X red
    cv2.line(out, o, tuple(proj[1]), (0, 255, 0), 2, cv2.LINE_AA)  # Y green
    cv2.line(out, o, tuple(proj[2]), (0, 64, 255), 2, cv2.LINE_AA)  # Z blue (out of screen)
    return out


_MIN_LANDMARKS_REQUIRED = max(_LM_INDICES) + 1  # 68 for iBUG-68


def _extract_frame_landmarks(frame_entry, w: int, h: int) -> Optional[np.ndarray]:
    """Extract the 14 named landmark pixel coords from a frame entry."""
    if isinstance(frame_entry, dict):
        for key in ("face_landmarks", "kps_face", "landmarks", "kps"):
            if key in frame_entry:
                frame_entry = frame_entry[key]
                break
    if not isinstance(frame_entry, (list, tuple)) or not frame_entry:
        return None
    arr = np.asarray(frame_entry, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < _MIN_LANDMARKS_REQUIRED or arr.shape[1] < 2:
        return None
    pts = arr[_LM_INDICES, :2]
    if pts.max() <= 1.5 and pts.min() >= -0.5:
        pts = pts * np.array([[w, h]], dtype=np.float64)
    return pts


class WanHeadPose6DoFV2:
    CATEGORY = "WanAnimatePreprocess_V2/HeadPose"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "IMAGE", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("poses_json", "overlay", "yaw", "pitch", "roll")
    DESCRIPTION = "6-DoF head pose via cv2.solvePnP on iBUG-68 landmarks (14-point canonical model)."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "landmarks_json": ("STRING", {"multiline": True, "default": "[]",
                    "tooltip": "JSON: list of frames; each frame is a list of [x,y] (iBUG-68, length >= 68) or a dict with face_landmarks/kps_face."}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image_width_override": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_height_override": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "axis_length": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.5}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, landmarks_json, image=None, image_width_override=0,
                image_height_override=0, axis_length=6.0):
        with torch.inference_mode():
            return self._execute_impl(
                landmarks_json, image, image_width_override,
                image_height_override, axis_length,
            )

    def _execute_impl(self, landmarks_json, image=None, image_width_override=0,
                image_height_override=0, axis_length=6.0):
        import cv2

        try:
            frames = json.loads(landmarks_json) if landmarks_json.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"WanHeadPose6DoFV2: invalid landmarks_json: {e}")
        if not isinstance(frames, list):
            frames = [frames]

        if image is not None and image.numel() > 0:
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] != 3:
                raise ValueError(
                    f"WanHeadPose6DoFV2: image expected (B,H,W,3); got {tuple(image.shape)}"
                )
            B, H, W, _ = image.shape
        else:
            H = int(image_height_override) or 720
            W = int(image_width_override) or 1280
            B = max(1, len(frames))

        focal = float(max(W, H))
        camera_matrix = np.array([[focal, 0, W / 2.0],
                                  [0, focal, H / 2.0],
                                  [0, 0, 1]], dtype=np.float64)

        results = []
        overlay_frames = []
        first_valid = None
        last_valid = None
        for i in range(max(B, len(frames))):
            entry = frames[i] if i < len(frames) else None
            pts = _extract_frame_landmarks(entry, W, H) if entry is not None else None
            res = _solve_pnp_one_frame(pts, camera_matrix) if pts is not None else None
            if res is None:
                res = {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
                       "tx": 0.0, "ty": 0.0, "tz": 0.0,
                       "rvec": [0, 0, 0], "tvec": [0, 0, 0],
                       "reprojection_err_px": float("nan"), "reliable": False}
            else:
                if first_valid is None:
                    first_valid = res
                last_valid = res
            results.append(res)

            # Build overlay if we have an image
            if image is not None and i < B:
                img_u8 = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                if res["reliable"] and pts is not None:
                    rvec = np.array(res["rvec"], dtype=np.float64).reshape(3, 1)
                    tvec = np.array(res["tvec"], dtype=np.float64).reshape(3, 1)
                    img_u8 = _draw_axes(img_u8, rvec, tvec, camera_matrix,
                                        origin_2d=(float(pts[0, 0]), float(pts[0, 1])),
                                        axis_len=float(axis_length))
                    cv2.putText(img_u8,
                                f"yaw={res['yaw_deg']:+6.1f}  pitch={res['pitch_deg']:+6.1f}  roll={res['roll_deg']:+6.1f}",
                                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                overlay_frames.append(img_u8.astype(np.float32) / 255.0)

        if not overlay_frames:
            overlay_frames = [np.zeros((H, W, 3), dtype=np.float32)]
        overlay = torch.from_numpy(np.stack(overlay_frames, axis=0))

        latest = last_valid or first_valid or results[-1]
        poses_json = json.dumps({
            "image_size": [W, H],
            "focal_px": focal,
            "frames": results,
            "summary": {
                "frames_total": len(results),
                "frames_reliable": sum(1 for r in results if r["reliable"]),
                "mean_yaw_deg": float(np.mean([r["yaw_deg"] for r in results if r["reliable"]] or [0.0])),
                "mean_pitch_deg": float(np.mean([r["pitch_deg"] for r in results if r["reliable"]] or [0.0])),
                "mean_roll_deg": float(np.mean([r["roll_deg"] for r in results if r["reliable"]] or [0.0])),
            }
        })

        return (poses_json, overlay,
                float(latest["yaw_deg"]), float(latest["pitch_deg"]), float(latest["roll_deg"]))
