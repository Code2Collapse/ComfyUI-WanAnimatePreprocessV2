# Copyright 2025 kijai (Jukka Seppänen) — original ComfyUI-WanAnimatePreprocess
#               https://github.com/kijai/ComfyUI-WanAnimatePreprocess
#               Apache License 2.0
#
# Copyright 2025 steven850 — improved pose/face pipeline (CLAHE, temporal
#               smoothing, constant-size face box, blur preprocessing)
#               Contributed in issue #10 of ComfyUI-WanAnimatePreprocess:
#               https://github.com/kijai/ComfyUI-WanAnimatePreprocess/issues/10
#               Apache License 2.0 (contributed to an Apache-2.0 repo)
#
# Copyright 2025-2026 Code2Collapse (https://github.com/Code2Collapse)
#               Additional work: iris/pupil detection (gradient voting, Timm-Barth
#               inspired multi-strategy), MediaPipe FaceMesh integration,
#               protobuf-5.x compatibility fix, V2 extensions and enhancements
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ---- Modifications by Code2Collapse (2025-2026) relative to steven850/kijai base ----
# - Added MediaPipe FaceMesh 478-point landmark pipeline with iris/gaze tracking
# - Added protobuf >=5.x compatibility fix for mediapipe <=0.10.x
# - Added gradient-based pupil centre detection (Timm-Barth 2011 inspired)
# - Added multi-strategy iris fallback (contour moments + weighted centroid)
# - Added iris/gaze overlay to debug visualisation
# - Added lip openness ratio output
# - Renamed nodes to V2 namespace; added RETURN_TYPES for iris/gaze/lip outputs

import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import json
import logging
import math

from . import _interrupt_check as _IC
script_directory = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------
# Optional MediaPipe Face Mesh (graceful fallback)
# ---------------------------------------------------
# MediaPipe provides 478 facial landmarks (468 mesh + 10 iris when
# refine_landmarks=True). It dramatically improves iris/lip tracking
# fidelity compared to the 68-point ViTPose face output and the custom
# OpenCV pupil voter. If `mediapipe` is not installed, the pipeline
# transparently falls back to the legacy ViTPose + `_find_pupil_center`
# code path and keeps working.
try:
    import mediapipe as _mp
    _MP_AVAILABLE = True
except Exception as _mp_err:  # ImportError or runtime DLL issues
    _mp = None
    _MP_AVAILABLE = False
    logging.getLogger(__name__).info(
        "MediaPipe not available, falling back to ViTPose-only face pipeline (%s)",
        _mp_err,
    )

# Module-level FaceMesh handle (lazily constructed, reused across frames).
_MP_FACE_MESH = None


def _get_mp_face_mesh():
    """Lazily construct a single static-image FaceMesh with iris refinement.

    MANUAL bug-fix (Apr 2026): mediapipe <=0.10.x's solution_base.py reads
    ``FieldDescriptor.label`` which was removed in protobuf 5.x. That raises
    ``AttributeError: 'google._upb._message.FieldDescriptor' object has no
    attribute 'label'`` during ``FaceMesh()`` construction. We catch it,
    log a clear remediation, and permanently disable mediapipe for this
    process so the existing ViTPose-only fallback is used instead.
    """
    global _MP_FACE_MESH, _MP_AVAILABLE
    if not _MP_AVAILABLE:
        return None
    if _MP_FACE_MESH is None:
        try:
            _MP_FACE_MESH = _mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,   # enables 10 iris landmarks (468-477)
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        except AttributeError as exc:
            _MP_AVAILABLE = False
            _MP_FACE_MESH = None
            logging.getLogger(__name__).warning(
                "MediaPipe FaceMesh init failed (%s). This is a known "
                "mediapipe<=0.10.x vs protobuf>=5.x incompatibility. "
                "Fix: pip install --upgrade mediapipe (>=0.10.18) OR "
                "pip install \"protobuf<=3.20.3\". Falling back to "
                "ViTPose-only face pipeline for the rest of this session.",
                exc,
            )
            return None
        except Exception as exc:  # noqa: BLE001 - any runtime failure -> fallback
            _MP_AVAILABLE = False
            _MP_FACE_MESH = None
            logging.getLogger(__name__).warning(
                "MediaPipe FaceMesh init failed (%s). Falling back to "
                "ViTPose-only face pipeline for the rest of this session.",
                exc,
            )
            return None
    return _MP_FACE_MESH


# ---------------------------------------------------
# MediaPipe -> dlib 68 landmark mapping
# ---------------------------------------------------
# Wan 2.x face conditioning consumes the standard 68-point dlib layout
# (slotted into face_kps[1:69]; face_kps[0] is the body-anchored face
# centre coming from ViTPose). We slice the 478 MediaPipe FaceMesh
# vertices to reconstruct that exact ordering, so existing limbSeq /
# `draw_aapose_by_meta_new` visualisation and the Wan pose encoder keep
# working without modification.
#
# Layout (68 = 17+5+5+4+5+6+6+12+8):
#   0-16  jawline (right ear -> chin -> left ear)
#   17-21 right eyebrow
#   22-26 left eyebrow
#   27-30 nose bridge (top -> tip)
#   31-35 nose bottom (right nostril -> tip -> left nostril)
#   36-41 right eye  (outer, upper-outer, upper-inner, inner, lower-inner, lower-outer)
#   42-47 left eye
#   48-59 outer lip (12 pts, clockwise from right corner)
#   60-67 inner lip (8 pts, clockwise from right corner)
MP_TO_DLIB68 = [
    # Jaw 0-16
    127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
    377, 400, 378, 379, 365,
    # Right eyebrow 17-21
    70, 63, 105, 66, 107,
    # Left eyebrow 22-26
    336, 296, 334, 293, 300,
    # Nose bridge 27-30
    168, 6, 197, 195,
    # Nose bottom 31-35
    115, 220, 4, 440, 344,
    # Right eye 36-41
    33, 160, 158, 133, 153, 144,
    # Left eye 42-47
    362, 385, 387, 263, 373, 380,
    # Outer lip 48-59
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # Inner lip 60-67
    78, 81, 13, 311, 308, 402, 14, 178,
]
assert len(MP_TO_DLIB68) == 68, "MP -> dlib mapping must define exactly 68 indices"

# Iris landmarks (only present when refine_landmarks=True).
MP_RIGHT_IRIS_CENTER = 468
MP_LEFT_IRIS_CENTER = 473
MP_RIGHT_IRIS_RING = [469, 470, 471, 472]
MP_LEFT_IRIS_RING = [474, 475, 476, 477]

# Inner-lip indices used for "openness" (mouth aspect ratio).
# Vertical opening: top-inner (13) <-> bottom-inner (14).
# Horizontal width: right inner corner (78) <-> left inner corner (308).
MP_INNER_LIP_TOP = 13
MP_INNER_LIP_BOTTOM = 14
MP_INNER_LIP_RIGHT = 78
MP_INNER_LIP_LEFT = 308


def mediapipe_to_dlib_68(mp_landmarks_xy):
    """Slice the 478-point MediaPipe array down to the 68-point dlib layout.

    Args:
        mp_landmarks_xy: (478, 2) ndarray of (x, y) coordinates in any
            consistent space (normalised or pixel).

    Returns:
        (68, 2) ndarray in the same coordinate space, ordered exactly as
        dlib's 68-point shape predictor expects.
    """
    return mp_landmarks_xy[MP_TO_DLIB68].copy()


def _run_mediapipe_on_face_crop(face_crop_rgb_uint8, crop_origin_xy, crop_size_wh,
                                  full_w, full_h):
    """Run MediaPipe FaceMesh on a single face crop.

    Args:
        face_crop_rgb_uint8: (h, w, 3) uint8 RGB face crop.
        crop_origin_xy:      (x1, y1) origin of the crop in the full image.
        crop_size_wh:        (cw, ch) pixel size of the crop.
        full_w, full_h:      full image pixel dimensions.

    Returns:
        dict with:
            - 'kps68_norm'  (68, 3) [x/W, y/H, conf=1.0] in *full image* normalised space
            - 'right_iris_px', 'left_iris_px': (x, y) in full image pixel space
            - 'right_iris_radius_px', 'left_iris_radius_px': float
            - 'lip_openness_ratio': float (vertical inner-lip / inner-lip width)
        Or None if MediaPipe failed to detect a face.
    """
    fm = _get_mp_face_mesh()
    if fm is None or face_crop_rgb_uint8 is None or face_crop_rgb_uint8.size == 0:
        return None

    try:
        results = fm.process(face_crop_rgb_uint8)
    except Exception:
        return None
    if not results.multi_face_landmarks:
        return None

    lms = results.multi_face_landmarks[0].landmark
    if len(lms) < 478:
        # Iris landmarks missing - refine_landmarks must have been disabled
        # at build time (e.g. older mediapipe). Treat as failure so we use
        # the fallback path that includes iris voting.
        return None

    cx0, cy0 = crop_origin_xy
    cw, ch = crop_size_wh

    # MediaPipe gives normalised coords [0, 1] relative to the crop.
    pts_px = np.zeros((len(lms), 2), dtype=np.float32)
    for i, lm in enumerate(lms):
        pts_px[i, 0] = lm.x * cw + cx0
        pts_px[i, 1] = lm.y * ch + cy0

    # Build the 68-point array in *full image* normalised space, with
    # confidence forced to 1.0 (MediaPipe doesn't expose per-point conf).
    kps68_px = pts_px[MP_TO_DLIB68]
    kps68_norm = np.zeros((68, 3), dtype=np.float32)
    kps68_norm[:, 0] = kps68_px[:, 0] / max(full_w, 1)
    kps68_norm[:, 1] = kps68_px[:, 1] / max(full_h, 1)
    kps68_norm[:, 2] = 1.0

    # Iris centres (full image pixel space)
    r_iris = pts_px[MP_RIGHT_IRIS_CENTER]
    l_iris = pts_px[MP_LEFT_IRIS_CENTER]
    r_ring = pts_px[MP_RIGHT_IRIS_RING]
    l_ring = pts_px[MP_LEFT_IRIS_RING]
    r_radius = float(np.mean(np.linalg.norm(r_ring - r_iris[None, :], axis=1)))
    l_radius = float(np.mean(np.linalg.norm(l_ring - l_iris[None, :], axis=1)))

    # Lip openness ratio (inner-lip MAR)
    top = pts_px[MP_INNER_LIP_TOP]
    bot = pts_px[MP_INNER_LIP_BOTTOM]
    rgt = pts_px[MP_INNER_LIP_RIGHT]
    lft = pts_px[MP_INNER_LIP_LEFT]
    v = float(np.linalg.norm(top - bot))
    h = float(np.linalg.norm(rgt - lft))
    lip_ratio = float(v / h) if h > 1e-6 else 0.0

    return {
        'kps68_norm': kps68_norm,
        'right_iris_px': (float(r_iris[0]), float(r_iris[1])),
        'left_iris_px': (float(l_iris[0]), float(l_iris[1])),
        'right_iris_radius_px': r_radius,
        'left_iris_radius_px': l_radius,
        'lip_openness_ratio': lip_ratio,
    }


# ---------------------------------------------------
# Production gaze via FaceLandmarker Tasks API + blend shapes
# ---------------------------------------------------
# The Tasks API replaces the legacy `mp.solutions.face_mesh` glue and
# additionally returns 52 ARKit-compatible blend shapes per face. We use
# the eight `eyeLookIn/Out/Up/Down{Left,Right}` shapes to derive
# head-pose-corrected per-eye yaw/pitch in radians — i.e. real gaze
# angles, not 2D iris offsets. See `gaze_blendshape.py` for the math.
try:
    from . import gaze_blendshape as _gaze_bs  # type: ignore
    _GAZE_BS_IMPORTED = True
except Exception as _exc:  # noqa: BLE001
    _gaze_bs = None
    _GAZE_BS_IMPORTED = False
    logging.getLogger(__name__).info(
        "gaze_blendshape module unavailable (%s); blend-shape gaze disabled.",
        _exc,
    )


def _run_face_landmarker_on_face_crop(
    face_crop_rgb_uint8, crop_origin_xy, crop_size_wh, full_w, full_h,
):
    """Run MediaPipe FaceLandmarker on a face crop and pack into the
    same dict shape as :func:`_run_mediapipe_on_face_crop`, plus an
    extra ``gaze`` entry derived from the eye-look blend shapes.

    Returns ``None`` when the Tasks API is not available or no face was
    found in the crop. Caller may fall back to FaceMesh.
    """
    if not _GAZE_BS_IMPORTED or _gaze_bs is None:
        return None
    if face_crop_rgb_uint8 is None or face_crop_rgb_uint8.size == 0:
        return None
    res = _gaze_bs.run_face_landmarker(face_crop_rgb_uint8)
    if res is None:
        return None

    landmarks = res["landmarks_norm"]   # (478, 3) in [0,1] crop-space
    if landmarks.shape[0] < 478:
        return None

    cx0, cy0 = crop_origin_xy
    cw, ch = crop_size_wh

    pts_px = np.empty((landmarks.shape[0], 2), dtype=np.float32)
    pts_px[:, 0] = landmarks[:, 0] * cw + cx0
    pts_px[:, 1] = landmarks[:, 1] * ch + cy0

    kps68_px = pts_px[MP_TO_DLIB68]
    kps68_norm = np.zeros((68, 3), dtype=np.float32)
    kps68_norm[:, 0] = kps68_px[:, 0] / max(full_w, 1)
    kps68_norm[:, 1] = kps68_px[:, 1] / max(full_h, 1)
    kps68_norm[:, 2] = 1.0

    r_iris = pts_px[MP_RIGHT_IRIS_CENTER]
    l_iris = pts_px[MP_LEFT_IRIS_CENTER]
    r_ring = pts_px[MP_RIGHT_IRIS_RING]
    l_ring = pts_px[MP_LEFT_IRIS_RING]
    r_radius = float(np.mean(np.linalg.norm(r_ring - r_iris[None, :], axis=1)))
    l_radius = float(np.mean(np.linalg.norm(l_ring - l_iris[None, :], axis=1)))

    top = pts_px[MP_INNER_LIP_TOP]
    bot = pts_px[MP_INNER_LIP_BOTTOM]
    rgt = pts_px[MP_INNER_LIP_RIGHT]
    lft = pts_px[MP_INNER_LIP_LEFT]
    v = float(np.linalg.norm(top - bot))
    h = float(np.linalg.norm(rgt - lft))
    lip_ratio = float(v / h) if h > 1e-6 else 0.0

    gaze = _gaze_bs.blendshapes_to_gaze(res.get("blendshapes") or {})

    return {
        'kps68_norm': kps68_norm,
        'right_iris_px': (float(r_iris[0]), float(r_iris[1])),
        'left_iris_px': (float(l_iris[0]), float(l_iris[1])),
        'right_iris_radius_px': r_radius,
        'left_iris_radius_px': l_radius,
        'lip_openness_ratio': lip_ratio,
        # NEW: production gaze from blend shapes — head-pose corrected,
        # in radians per eye, plus a 2D dx/dy for legacy debug overlay.
        'gaze_blendshape': gaze,
        'blendshapes': res.get("blendshapes") or {},
        'face_transform': res.get("transform"),
        'source': 'face_landmarker',
    }


from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .models.onnx_models import ViTPose, Yolo
from .pose_utils.pose2d_utils import load_pose_metas_from_kp2ds_seq, crop, bbox_from_detector
from .utils import (
    get_face_bboxes,
    padding_resize,
    adjust_bbox_eye_upper_third,
    compute_eye_midpoint_from_face_kps,
    compute_frame_blur_score,
    compute_eye_region_brightness,
)
from .pose_utils.human_visualization import AAPoseMeta, draw_aapose_by_meta_new


# ---------------------------------------------------
# Image enhancement utilities
# ---------------------------------------------------
def preprocess_for_pose(img, use_clahe=True):
    """Optional CLAHE contrast enhancement for ViTPose inputs."""
    if not use_clahe:
        return img

    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img_uint8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return img_uint8.astype(np.float32) / 255.0


# ---------------------------------------------------
# Iris / pupil estimation (image-based)
# ---------------------------------------------------
# Eye contour landmark indices within the 69-point face array.
# face array = kp2ds[22:91]; index 0 is body, indices 1-68 are
# standard 68-face landmarks (standard N -> face[N+1]).
#
# Right eye (standard 36-41):
#   37=outer  38=upper_outer  39=upper_inner
#   40=inner  41=lower_inner  42=lower_outer
# Left eye (standard 42-47):
#   43=inner  44=upper_inner  45=upper_outer
#   46=outer  47=lower_outer  48=lower_inner
_RIGHT_EYE_IDX = [37, 38, 39, 40, 41, 42]
_LEFT_EYE_IDX  = [43, 44, 45, 46, 47, 48]
_EYE_CONTOUR_INDICES = [_RIGHT_EYE_IDX, _LEFT_EYE_IDX]


def _gradient_vote_pupil(roi_gray, mask, gc_local, eye_w, eye_h):
    """Gradient-based pupil centre detection (Timm-Barth 2011 inspired).

    Edge gradients around the circular pupil boundary point radially outward.
    By casting rays in the *negative* gradient direction from every strong-edge
    pixel we accumulate votes at the true centre.

    Returns (local_cx, local_cy, score) or None.
    """
    h, w = roi_gray.shape
    gx = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_gray.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    mag_masked = mag.copy()
    mag_masked[mask == 0] = 0
    vals = mag_masked[mask > 0]
    if len(vals) < 8 or vals.max() < 1:
        return None

    thresh = float(np.percentile(vals, 70))  # top 30 % of gradients
    strong = (mag > thresh) & (mask > 0)
    if np.count_nonzero(strong) < 8:
        return None

    ys, xs = np.where(strong)
    gx_s = gx[ys, xs]
    gy_s = gy[ys, xs]
    mag_s = mag[ys, xs]

    # Normalise & negate  (point *toward* centre)
    gx_n = -gx_s / (mag_s + 1e-10)
    gy_n = -gy_s / (mag_s + 1e-10)

    accumulator = np.zeros((h, w), np.float64)
    max_t = max(int(max(eye_w, eye_h) * 0.5), 5)

    for t in range(1, max_t + 1):
        px = (xs + gx_n * t + 0.5).astype(np.int32)
        py = (ys + gy_n * t + 0.5).astype(np.int32)
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        pxv, pyv, magv = px[valid], py[valid], mag_s[valid]
        in_mask = mask[pyv, pxv] > 0
        np.add.at(accumulator, (pyv[in_mask], pxv[in_mask]), magv[in_mask])

    # Weight by darkness (pupil region is darker than sclera)
    dark_w = (255.0 - roi_gray.astype(np.float64)) / 255.0
    accumulator *= (0.5 + 0.5 * dark_w)
    accumulator[mask == 0] = 0

    if accumulator.max() < 1:
        return None

    acc_smooth = cv2.GaussianBlur(accumulator, (5, 5), 1.0)
    _, max_val, _, max_loc = cv2.minMaxLoc(acc_smooth, mask=mask)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    mean_acc = float(np.mean(accumulator[mask > 0]) + 1e-6)
    score = min(1.0, max_val / (mean_acc * 5))
    return cx, cy, score


def _find_pupil_center(eye_pts_px, img_gray, W, H):
    """Locate the pupil/iris centre inside one eye.

    Pipeline
    --------
    1. Build a tight eye-region mask from 6 contour landmarks.
    2. Restrict search to the **upper 65 %** of the lid opening to avoid
       eyelid / eyelash shadow contamination.
    3. Apply CLAHE for better pupil-iris-sclera contrast.
    4. **Primary** – gradient-based centre voting (Timm-Barth inspired):
       robust to lighting, threshold-free.
    5. **Secondary** – multi-threshold contour moments with asymmetric
       vertical scoring.
    6. **Tertiary** – weighted dark-pixel centroid with upper-region bias.
    7. Fallback – geometric centre of the eye contour.
    """
    geo_center = np.mean(eye_pts_px, axis=0)

    # --- Eye Aspect Ratio (EAR) – skip closed eyes ---
    v1 = np.linalg.norm(eye_pts_px[1] - eye_pts_px[5])
    v2 = np.linalg.norm(eye_pts_px[2] - eye_pts_px[4])
    horiz = np.linalg.norm(eye_pts_px[0] - eye_pts_px[3])
    if horiz < 3:
        return float(geo_center[0]), float(geo_center[1]), 0.0
    ear = (v1 + v2) / (2.0 * horiz)
    if ear < 0.12:
        return float(geo_center[0]), float(geo_center[1]), 0.05

    # --- Tight padded ROI ---
    min_xy = np.min(eye_pts_px, axis=0)
    max_xy = np.max(eye_pts_px, axis=0)
    eye_w = max_xy[0] - min_xy[0]
    eye_h = max_xy[1] - min_xy[1]
    pad = max(int(eye_w * 0.15), 2)
    rx1 = max(0, int(min_xy[0]) - pad)
    ry1 = max(0, int(min_xy[1]) - pad)
    rx2 = min(W, int(max_xy[0]) + pad)
    ry2 = min(H, int(max_xy[1]) + pad)
    roi = img_gray[ry1:ry2, rx1:rx2]
    if roi.size < 20:
        return float(geo_center[0]), float(geo_center[1]), 0.1
    h_roi, w_roi = roi.shape

    # --- Eye contour mask ---
    pts_local = eye_pts_px.astype(np.int32).copy()
    pts_local[:, 0] -= rx1
    pts_local[:, 1] -= ry1
    mask_full = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.fillConvexPoly(mask_full, pts_local, 255)

    # --- Restrict to upper 65 % of lid opening ---
    eye_top_l = max(0, int(min_xy[1]) - ry1)
    eye_bot_l = min(h_roi, int(max_xy[1]) - ry1)
    cutoff = int(eye_top_l + 0.65 * (eye_bot_l - eye_top_l))
    mask = mask_full.copy()
    mask[cutoff:, :] = 0

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_inner = cv2.erode(mask, kern, iterations=2)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = cv2.erode(mask, kern, iterations=1)
    if np.count_nonzero(mask_inner) < 5:
        mask_inner = mask

    # --- CLAHE + gentle blur ---
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    roi_eq = clahe.apply(roi)
    roi_blur = cv2.GaussianBlur(roi_eq, (3, 3), 0.7)

    masked_pixels = roi_blur[mask_inner > 0]
    if len(masked_pixels) < 5:
        return float(geo_center[0]), float(geo_center[1]), 0.1

    gc_local = geo_center - np.array([rx1, ry1])
    mask_area = float(max(np.count_nonzero(mask_inner), 1))

    # ================================================================
    # Strategy 1 – gradient-based centre voting  (primary)
    # ================================================================
    grad = _gradient_vote_pupil(roi_blur, mask_inner, gc_local, eye_w, eye_h)
    if grad is not None:
        gcx, gcy, gscore = grad
        if gscore > 0.20:
            conf = float(np.clip(ear * 2.5 * gscore, 0.1, 1.0))
            return float(gcx) + rx1, float(gcy) + ry1, conf

    # ================================================================
    # Strategy 2 – multi-threshold contour moments  (secondary)
    # ================================================================
    best_cx, best_cy, best_score = None, None, -1.0
    for pct in (10, 20, 30, 40):
        thresh_val = int(np.percentile(masked_pixels, pct))
        binary = np.zeros_like(roi_blur)
        binary[(roi_blur <= thresh_val) & (mask_inner > 0)] = 255
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_OPEN, kern)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue
            M = cv2.moments(cnt)
            if M["m00"] < 1:
                continue
            cx_l = M["m10"] / M["m00"]
            cy_l = M["m01"] / M["m00"]
            ix, iy = int(cx_l), int(cy_l)
            if not (0 <= ix < w_roi and 0 <= iy < h_roi):
                continue
            if mask_full[iy, ix] == 0:
                continue

            # Circularity
            perim = cv2.arcLength(cnt, True)
            circ = 4 * np.pi * area / (perim ** 2 + 1e-6)
            circ_score = min(circ, 1.0)

            # Proximity – asymmetric vertical penalty
            dx = abs(cx_l - gc_local[0])
            dy = cy_l - gc_local[1]          # positive = below centre
            max_dx = max(eye_w * 0.45, 1)
            h_prox = max(0.0, 1.0 - dx / max_dx)
            max_dy_up = max(eye_h * 0.4, 1)
            max_dy_dn = max(eye_h * 0.25, 1)  # tighter below
            v_prox = max(0.0, 1.0 - abs(dy) / (max_dy_dn if dy > 0 else max_dy_up))
            prox_score = 0.5 * h_prox + 0.5 * v_prox

            # Size
            ratio = area / mask_area
            if ratio < 0.03 or ratio > 0.70:
                size_score = 0.1
            elif 0.08 <= ratio <= 0.45:
                size_score = 1.0
            else:
                size_score = 0.5

            score = circ_score * 0.25 + prox_score * 0.45 + size_score * 0.30
            if score > best_score:
                best_score = score
                best_cx = cx_l + rx1
                best_cy = cy_l + ry1

    if best_cx is not None and best_score > 0.25:
        conf = float(np.clip(ear * 2.5 * best_score, 0.1, 1.0))
        return float(best_cx), float(best_cy), conf

    # ================================================================
    # Strategy 3 – weighted dark-pixel centroid with upper-region bias
    # ================================================================
    thresh_val = int(np.percentile(masked_pixels, 25))
    dark = (roi_blur <= thresh_val) & (mask_inner > 0)
    ys, xs = np.where(dark)
    if len(xs) > 3:
        weights = (255.0 - roi_blur[dark]).astype(np.float64)
        vert_bias = np.clip(
            1.0 - (ys - gc_local[1]) / max(eye_h * 0.3, 1), 0.3, 1.5)
        weights *= vert_bias
        total = weights.sum()
        if total > 0:
            cx = float(np.sum(xs * weights) / total) + rx1
            cy = float(np.sum(ys * weights) / total) + ry1
            return cx, cy, float(np.clip(ear * 1.5, 0.1, 0.7))

    # --- Fallback: geometric centre ---
    return float(geo_center[0]), float(geo_center[1]), 0.1


def estimate_iris_positions(face_kps, image_np, img_width, img_height):
    """Estimate iris centres for both eyes using image-based pupil detection.

    Args:
        face_kps: (69, 3) normalised keypoints [x/W, y/H, conf]
        image_np: (H, W, 3) float32 RGB image [0, 1]
        img_width, img_height: pixel dimensions

    Returns:
        dict with right_iris, left_iris, right_gaze, left_gaze.
    """
    W, H = img_width, img_height
    kps_px = face_kps[:, :2].copy() * np.array([W, H])
    kps_conf = face_kps[:, 2].copy()

    img_u8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)

    results = {}
    for eye_name, eye_idx in [('right', _RIGHT_EYE_IDX),
                               ('left', _LEFT_EYE_IDX)]:
        pts = kps_px[eye_idx]      # (6, 2)
        confs = kps_conf[eye_idx]
        mc = float(np.mean(confs))
        geo = np.mean(pts, axis=0)

        if mc < 0.05:
            results[f'{eye_name}_iris'] = {
                'x': float(geo[0]), 'y': float(geo[1]), 'confidence': 0.0}
            results[f'{eye_name}_gaze'] = {'dx': 0.0, 'dy': 0.0}
            continue

        ix, iy, ic = _find_pupil_center(pts, img_gray, W, H)
        results[f'{eye_name}_iris'] = {'x': ix, 'y': iy, 'confidence': ic}

        dx = ix - float(geo[0])
        dy = iy - float(geo[1])
        norm = max(np.hypot(dx, dy), 1e-6)
        # Approximate yaw/pitch from 2D iris-offset (small-angle, scaled by
        # eye span). This lets the downstream gaze-lock + OneEuro paths
        # work even when MediaPipe blendshapes are unavailable.
        eye_span_x = float(np.ptp(pts[:, 0])) or 1.0
        eye_span_y = float(np.ptp(pts[:, 1])) or 1.0
        yaw_rad = np.clip(dx / (eye_span_x * 0.5), -1.2, 1.2) * 0.5  # ~30 deg max
        pitch_rad = np.clip(dy / (eye_span_y * 0.5), -1.2, 1.2) * 0.4  # ~25 deg max
        results[f'{eye_name}_gaze'] = {
            'dx': round(dx / norm, 4), 'dy': round(dy / norm, 4),
            'yaw_rad': float(yaw_rad), 'pitch_rad': float(pitch_rad),
            'magnitude': float(np.hypot(yaw_rad, pitch_rad)),
            'source': 'iris_offset_2d',
        }

    return results


# ---------------------------------------------------
# Debug visualisation overlay
# ---------------------------------------------------
# Colour palette for 68-landmark regions (face-array index ranges)
_LANDMARK_COLORS = [
    (1,  17, (255, 200, 0)),    # jawline
    (18, 22, (200, 255, 0)),    # right eyebrow
    (23, 27, (200, 255, 0)),    # left eyebrow
    (28, 36, (0, 0, 255)),      # nose
    (37, 42, (0, 255, 0)),      # right eye
    (43, 48, (0, 255, 0)),      # left eye
    (49, 60, (0, 255, 255)),    # outer mouth
    (61, 68, (0, 255, 200)),    # inner mouth
]


def draw_debug_overlay(frame_uint8, face_kps_norm, iris_data,
                       face_bbox, body_bbox, W, H):
    """Draw face landmarks, iris positions and bounding boxes for debugging.

    Args:
        frame_uint8:   (H, W, 3) uint8 RGB
        face_kps_norm: (69, 3) normalised keypoints
        iris_data:     dict from estimate_iris_positions
        face_bbox:     (x1, x2, y1, y2) or None
        body_bbox:     [x1, y1, x2, y2, ...] array or None
        W, H:          image pixel dimensions

    Returns:
        vis: (H, W, 3) uint8 RGB image with annotations
    """
    vis = frame_uint8.copy()
    kps_px = face_kps_norm[:, :2] * np.array([W, H])
    kps_conf = face_kps_norm[:, 2]

    # --- Face landmarks ---
    for idx in range(1, min(69, len(kps_px))):
        if kps_conf[idx] < 0.05:
            continue
        x, y = int(kps_px[idx, 0]), int(kps_px[idx, 1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        color = (180, 180, 180)
        for lo, hi, c in _LANDMARK_COLORS:
            if lo <= idx <= hi:
                color = c
                break
        cv2.circle(vis, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 2, color, -1, cv2.LINE_AA)

    # --- Eye contour polylines ---
    for eye_indices in _EYE_CONTOUR_INDICES:
        pts = []
        for i in eye_indices:
            if i < len(kps_px) and kps_conf[i] > 0.05:
                pts.append([int(kps_px[i, 0]), int(kps_px[i, 1])])
        if len(pts) >= 4:
            cv2.polylines(vis, [np.array(pts, np.int32)], True,
                          (0, 255, 0), 1, cv2.LINE_AA)

    # --- Iris markers + gaze arrows ---
    for eye_key, gaze_key in [('right_iris', 'right_gaze'),
                               ('left_iris', 'left_gaze')]:
        iris = iris_data.get(eye_key)
        gaze = iris_data.get(gaze_key)
        if iris is None or iris['confidence'] < 0.05:
            continue
        ix, iy = int(iris['x']), int(iris['y'])
        if 0 <= ix < W and 0 <= iy < H:
            cv2.drawMarker(vis, (ix, iy), (255, 0, 255),
                           cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
            cv2.circle(vis, (ix, iy), 5, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{iris['confidence']:.2f}",
                        (ix + 8, iy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (255, 0, 255), 1, cv2.LINE_AA)
        if gaze and (abs(gaze['dx']) > 1e-4 or abs(gaze['dy']) > 1e-4):
            arrow_len = 35
            ex = int(ix + gaze['dx'] * arrow_len)
            ey = int(iy + gaze['dy'] * arrow_len)
            cv2.arrowedLine(vis, (ix, iy), (ex, ey),
                            (0, 200, 255), 2, cv2.LINE_AA, tipLength=0.3)

    # --- Face bounding box (cyan) ---
    if face_bbox is not None:
        x1, x2, y1, y2 = face_bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, "FACE", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # --- Body bounding box (green) ---
    if body_bbox is not None:
        bb = np.asarray(body_bbox).flatten()
        if len(bb) >= 4:
            cv2.rectangle(vis, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "BODY", (int(bb[0]), int(bb[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    return vis


# ---------------------------------------------------
# ONNX model loader
# ---------------------------------------------------
class OnnxDetectionModelLoaderV2:
    DESCRIPTION = (
        "Load ONNX ViTPose + YOLO detection models for Wan 2.2 Animate "
        "preprocessing. Place model files in `ComfyUI/models/detection/`. "
        "Outputs a `POSEMODEL` bundle that the detection node consumes."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": (folder_paths.get_filename_list("detection"), {"tooltip": "ViTPose ONNX file (e.g. vitpose-h.onnx). Place in ComfyUI/models/detection/."}),
                "yolo_model":    (folder_paths.get_filename_list("detection"), {"tooltip": "YOLO person-detector ONNX file. Place in ComfyUI/models/detection/."}),
                "onnx_device":   (["CUDAExecutionProvider", "CPUExecutionProvider"], {"default": "CUDAExecutionProvider", "tooltip": "Execution provider for ONNX Runtime. CUDA is much faster; CPU is the safe fallback."}),
            },
        }

    RETURN_TYPES = ("POSEMODEL",)
    RETURN_NAMES = ("model", )
    OUTPUT_TOOLTIPS = ("ViTPose+YOLO model bundle. Connect to `model` on Pose and Face Detection (V2).",)
    FUNCTION = "loadmodel"
    CATEGORY = "WanAnimatePreprocess_V2"

    def loadmodel(self, vitpose_model, yolo_model, onnx_device):
        vitpose_model_path = folder_paths.get_full_path_or_raise("detection", vitpose_model)
        yolo_model_path = folder_paths.get_full_path_or_raise("detection", yolo_model)
        vitpose = ViTPose(vitpose_model_path, onnx_device)
        yolo = Yolo(yolo_model_path, onnx_device)
        return ({"vitpose": vitpose, "yolo": yolo},)


# ---------------------------------------------------
# Jitterless face-crop helpers
# ---------------------------------------------------
def _parse_keyframes_json(s, B):
    """Return a list of dicts {frame, cx, cy, size?} sorted by frame.

    Tolerates malformed input — bad entries are skipped with a warning.
    """
    if not s or not isinstance(s, str):
        return []
    try:
        raw = json.loads(s)
    except Exception as e:
        print(f"[PoseAndFaceDetectionV2] keyframes_json parse error: {e}; ignoring.")
        return []
    if not isinstance(raw, list):
        return []
    out = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            f = int(entry.get("frame", -1))
            cx = float(entry.get("cx"))
            cy = float(entry.get("cy"))
        except (TypeError, ValueError):
            continue
        if f < 0 or f >= B:
            continue
        kf = {"frame": f, "cx": cx, "cy": cy}
        if "size" in entry and entry["size"] is not None:
            try:
                kf["size"] = int(entry["size"])
            except (TypeError, ValueError):
                pass
        out.append(kf)
    out.sort(key=lambda e: e["frame"])
    return out


def _interp_keyframes(keyframes, B, default_cx, default_cy, default_size):
    """Densify sorted keyframes into per-frame (cx, cy, size) arrays.

    Frames before the first keyframe hold the first; after the last hold
    the last; in-between frames are linearly interpolated.
    Returns three np.ndarray of length B (or None if no keyframes).
    """
    if not keyframes:
        return None, None, None
    cx = np.full(B, default_cx, dtype=np.float32)
    cy = np.full(B, default_cy, dtype=np.float32)
    sz = np.full(B, default_size, dtype=np.float32)
    # spread cx/cy across all frames
    frames = np.array([k["frame"] for k in keyframes], dtype=np.int32)
    cxs = np.array([k["cx"] for k in keyframes], dtype=np.float32)
    cys = np.array([k["cy"] for k in keyframes], dtype=np.float32)
    sizes_known = np.array([k.get("size", -1) for k in keyframes], dtype=np.float32)
    xs = np.arange(B, dtype=np.float32)
    cx = np.interp(xs, frames, cxs).astype(np.float32)
    cy = np.interp(xs, frames, cys).astype(np.float32)
    if np.any(sizes_known > 0):
        # only interpolate sizes between keyframes that actually specify one
        mask = sizes_known > 0
        sz_known = sizes_known[mask]
        f_known = frames[mask].astype(np.float32)
        if f_known.size >= 2:
            sz = np.interp(xs, f_known, sz_known).astype(np.float32)
        elif f_known.size == 1:
            sz = np.full(B, float(sz_known[0]), dtype=np.float32)
    return cx, cy, sz


def _gaussian_window(window):
    """1D Gaussian kernel (length=window, odd) suitable for np.convolve."""
    window = max(3, int(window) | 1)  # force odd and >= 3
    sigma = max(0.5, window / 6.0)
    xs = np.arange(window) - (window - 1) / 2.0
    k = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _smooth_centers(centers_xy, method, *, ema_strength=0.6, image_diag=1.0,
                    one_euro_min_cutoff=1.0, one_euro_beta=0.05,
                    gaussian_window=7):
    """Apply a temporal filter to a (B, 2) array of (cx, cy)."""
    centers_xy = np.asarray(centers_xy, dtype=np.float32)
    if centers_xy.ndim != 2 or centers_xy.shape[0] < 2 or method == "none":
        return centers_xy.copy()

    if method == "ema":
        out = np.empty_like(centers_xy)
        out[0] = centers_xy[0]
        norm = max(1.0, image_diag)
        base = float(np.clip(ema_strength, 0.0, 1.0))
        for i in range(1, len(centers_xy)):
            curr = centers_xy[i]
            prev = out[i - 1]
            motion = float(np.mean(np.abs(curr - prev)) / norm)
            dyn = base * np.exp(-motion * 5.0)
            alpha = 1.0 - dyn
            out[i] = alpha * curr + (1.0 - alpha) * prev
        return out

    if method == "gaussian":
        k = _gaussian_window(gaussian_window)
        # reflect-pad so the ends don't darken
        pad = len(k) // 2
        padded = np.pad(centers_xy, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(centers_xy)
        out[:, 0] = np.convolve(padded[:, 0], k, mode="valid")
        out[:, 1] = np.convolve(padded[:, 1], k, mode="valid")
        return out

    # default: one_euro
    _OEF = None
    if _GAZE_BS_IMPORTED and _gaze_bs is not None:
        _OEF = getattr(_gaze_bs, "OneEuroFilter", None)
    if _OEF is None:
        # graceful fallback
        return _smooth_centers(centers_xy, "ema",
                               ema_strength=ema_strength,
                               image_diag=image_diag)
    fx = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    fy = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    out = np.empty_like(centers_xy)
    for i in range(len(centers_xy)):
        out[i, 0] = fx(float(centers_xy[i, 0]))
        out[i, 1] = fy(float(centers_xy[i, 1]))
    return out


def _smooth_1d(values, method, *, ema_strength=0.6, scale_norm=1.0,
               one_euro_min_cutoff=1.0, one_euro_beta=0.05,
               gaussian_window=7):
    """Temporal filter for a 1-D scalar series (e.g. per-frame crop sizes).

    Same methods as _smooth_centers: one_euro (default), ema, gaussian, none.
    ``scale_norm`` normalises the motion magnitude for the adaptive EMA.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 2 or method == "none":
        return values.copy()

    if method == "gaussian":
        k = _gaussian_window(gaussian_window)
        pad = len(k) // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        return np.convolve(padded, k, mode="valid").astype(np.float32)

    if method == "ema":
        out = np.empty_like(values)
        out[0] = values[0]
        norm = max(1.0, float(scale_norm))
        base = float(np.clip(ema_strength, 0.0, 1.0))
        for i in range(1, len(values)):
            motion = abs(float(values[i]) - float(out[i - 1])) / norm
            dyn = base * np.exp(-motion * 5.0)
            out[i] = (1.0 - dyn) * float(values[i]) + dyn * float(out[i - 1])
        return out

    # default: one_euro
    _OEF = None
    if _GAZE_BS_IMPORTED and _gaze_bs is not None:
        _OEF = getattr(_gaze_bs, "OneEuroFilter", None)
    if _OEF is None:
        return _smooth_1d(values, "ema", ema_strength=ema_strength,
                          scale_norm=scale_norm)
    f = _OEF(freq=30.0, min_cutoff=one_euro_min_cutoff, beta=one_euro_beta)
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = f(float(values[i]))
    return out


# ---------------------------------------------------
# Pose and Face Detection
# ---------------------------------------------------
class PoseAndFaceDetectionV2:
    DESCRIPTION = (
        "Run YOLO person detection + ViTPose 2D keypoints + (optional) MediaPipe "
        "FaceMesh on a video tensor. Produces the full pose/face/iris bundle "
        "required by Wan 2.2 Animate Character Replacement workflows."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model":  ("POSEMODEL", {"tooltip": "From ONNX Detection Model Loader (V2)."}),
                "images": ("IMAGE",     {"tooltip": "Video frames as an IMAGE batch (B,H,W,C float [0,1])."}),
                "width":  ("INT", {"default": 832, "min": 64, "max": 2048, "tooltip": "Target canvas width (px) used for retarget math. Match your Wan 2.2 latent size."}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "tooltip": "Target canvas height (px). Match your Wan 2.2 latent size."}),
                "detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO confidence threshold. Lower = more permissive person detection."}),
                "pose_threshold":      ("FLOAT", {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold. Below this a keypoint is treated as missing."}),
                # Enhancement options
                "use_clahe": ("BOOLEAN", {"default": True, "tooltip": "Apply CLAHE contrast enhancement for pose detection."}),
                "use_blur_for_pose": ("BOOLEAN", {"default": True, "tooltip": "Apply Gaussian blur internally for YOLO and ViTPose."}),
                "blur_radius": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1, "tooltip": "Gaussian blur kernel radius applied to the face mask edge to soften the boundary. Higher = wider feather. Kernel size = radius*2+1 px."}),
                "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1, "tooltip": "Gaussian blur sigma (standard deviation) for the face mask feather. Higher sigma = softer falloff. Tune together with blur_radius."}),
                # Face smoothing
                "use_face_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Smooth face bounding box center over time."}),
                "face_smoothing_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Higher = more smoothing"}),
                # Constant-size face box
                "use_constant_face_box": ("BOOLEAN", {"default": True, "tooltip": "Keep a constant pixel size face crop; position adapts."}),
                "face_box_size_px": ("INT", {"default": 224, "min": 64, "max": 1024, "step": 16, "tooltip": "Pixel size of the square face crop when constant mode is on."}),
                # Iris estimation
                "use_iris_smoothing": ("BOOLEAN", {"default": True, "tooltip": "Temporally smooth iris pixel positions across frames. Reduces per-frame jitter that Wan 2.2 Animate's face encoder picks up and reproduces as wobbly gaze."}),
                "iris_smoothing_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "EMA mix weight when iris_smoothing_method='ema'. Higher = more smoothing, more lag. Ignored for one_euro / none."}),
                "iris_smoothing_method": (["one_euro", "ema", "none"], {"default": "one_euro", "tooltip": "Iris pixel-position smoother. one_euro = adaptive low-pass (Casiez 2012, recommended). ema = legacy first-order; tweak via iris_smoothing_strength. none = raw per-frame positions."}),
                "iris_one_euro_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro min cutoff (Hz) for iris pixel coords. Lower = stronger jitter rejection on near-static eyes (small saccades preserved)."}),
                "iris_one_euro_beta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "One-euro speed coefficient for iris pixel coords. Higher = filter relaxes faster on quick eye movements; lower = stronger steady-state smoothing."}),
                # Cross-eye coupling (NEW: directly fixes 'eyeballs not in same direction').
                "gaze_lock_eyes": ("BOOLEAN", {"default": True, "tooltip": "Couple left & right eye gaze so they always look in the SAME direction. Both eyes' yaw/pitch are blended toward their per-frame average. Single most effective fix for the 'eyes pointing different directions' artefact in Wan 2.2 Animate output."}),
                "gaze_lock_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "How strongly to pull each eye toward the shared average. 0 = independent (legacy). 1 = perfectly conjugate (both eyes always parallel). 0.7 keeps a touch of natural convergence/divergence."}),
                # MediaPipe face mesh (high-fidelity iris/lip tracking, falls back to ViTPose if unavailable)
                "use_mediapipe_face": ("BOOLEAN", {"default": True, "tooltip": "Use MediaPipe FaceMesh (478 pts incl. iris/lips) to override face landmarks. Falls back to ViTPose pupil voting if MediaPipe is missing or fails on a frame."}),
                # Production gaze (ARKit blend shapes via FaceLandmarker Tasks API)
                "use_blendshape_gaze": ("BOOLEAN", {"default": True, "tooltip": "Use MediaPipe FaceLandmarker (Tasks API) blend shapes for production-grade per-eye yaw/pitch in radians. Head-pose-corrected by training. Auto-downloads face_landmarker.task (~3MB) on first run. Falls back to legacy 2D iris-offset gaze if disabled or unavailable."}),
                "gaze_one_euro_min_cutoff": ("FLOAT", {"default": 1.7, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro filter base cutoff frequency (Hz). Lower = more aggressive jitter rejection at the cost of slight lag. 1.7 is a good default for 24-30 fps gaze."}),
                "gaze_one_euro_beta": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "One-euro filter speed coefficient. Higher = filter relaxes faster on quick saccades, preserving responsiveness; lower = stronger smoothing during fast moves."}),
                "gaze_max_yaw_deg": ("FLOAT", {"default": 30.0, "min": 5.0, "max": 60.0, "step": 1.0, "tooltip": "Saturation yaw angle in degrees that corresponds to blend shape value 1.0. 30\u00b0 covers the comfortable physiological range; raise for more dramatic eye motion."}),
                "gaze_max_pitch_deg": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 60.0, "step": 1.0, "tooltip": "Saturation pitch angle in degrees that corresponds to blend shape value 1.0. 25\u00b0 covers the comfortable physiological range."}),
                # ---- Jitterless face crop (manual frame-0 anchor + keyframes) ----
                "crop_mode": (["default", "auto", "jitterless"], {"default": "default", "tooltip": "default = raw detected bbox per frame (NO smoothing / NO constant size — crop is effectively 'off'). auto = legacy smoothed + optional constant-size box. jitterless = lock crop SIZE from frame 0, smoothly track the CENTER, allow manual frame-0 + key-frame overrides."}),
                "frame0_cx": ("INT", {"default": -1, "min": -1, "max": 8192, "tooltip": "Frame 0 anchor center X in pixels. -1 = use detected face center on frame 0. Used only when crop_mode=jitterless."}),
                "frame0_cy": ("INT", {"default": -1, "min": -1, "max": 8192, "tooltip": "Frame 0 anchor center Y in pixels. -1 = use detected face center on frame 0."}),
                "frame0_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "Locked square crop size in pixels (used for the entire clip). 0 = fall back to face_box_size_px."}),
                "keyframes_json": ("STRING", {"default": "[]", "multiline": True, "tooltip": "JSON list of per-frame overrides: [{\"frame\":N, \"cx\":X, \"cy\":Y, \"size\":S?}, ...]. Frames between key-frames are linearly interpolated. size is optional; if omitted the locked size is kept."}),
                "smoothing_method": (["one_euro", "ema", "gaussian", "none"], {"default": "one_euro", "tooltip": "Center-trajectory filter. one_euro = jitterless adaptive low-pass (recommended). ema = legacy motion-adaptive EMA. gaussian = fixed-window 1D blur. none = raw."}),
                "crop_one_euro_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05, "tooltip": "One-euro min cutoff (Hz) for crop center. Lower = stronger jitter rejection."}),
                "crop_one_euro_beta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "One-euro speed coefficient for crop center. Higher = filter relaxes faster on quick motion."}),
                "crop_gaussian_window": ("INT", {"default": 7, "min": 3, "max": 51, "step": 2, "tooltip": "Window size (odd) for the Gaussian temporal blur of the crop center."}),
                # ---- Wan-Animate paper-driven gaze fixes (arXiv:2509.14055) ----
                "eye_align_mode": (["default", "eye_upper_third"], {"default": "default", "tooltip": "Wan-Animate paper recommendation #1: 'eye_upper_third' vertically shifts the face crop so eyes land at the upper third of the 512x512 face encoder input. The encoder reads holistic face appearance, so consistent eye placement directly improves gaze fidelity. 'default' keeps legacy bbox center."}),
                "eye_y_fraction": ("FLOAT", {"default": 0.30, "min": 0.10, "max": 0.60, "step": 0.01, "tooltip": "Target eye row as a fraction of crop height (0.30 = upper third). Only used when eye_align_mode = 'eye_upper_third'."}),
                "face_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1, "tooltip": "Wan-Animate paper recommendation #3 (paper section 4.3): CFG on the face conditioning input gives finer control over expression / gaze when finer reenactment is desired. This widget is a passthrough -- wire the FLOAT output 'face_cfg_scale' into your Wan-Animate sampler's face CFG input. 1.0 = CFG disabled (default, fastest). 2.0-4.0 = stronger expression adherence. >5.0 may over-saturate."}),
            },
            "optional": {
                "bbox_override": ("BBOX", {"tooltip": "Optional external BBOX for the frame-0 anchor. Highest priority; overrides frame0_cx/cy/size widgets."}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "IMAGE", "STRING", "BBOX", "BBOX", "STRING", "IMAGE", "STRING", "STRING", "FLOAT", "FACE_RESTORE_INFO", "FLOAT")
    RETURN_NAMES = ("pose_data", "face_images", "key_frame_body_points", "bboxes", "face_bboxes", "iris_data", "debug_image", "right_pupil_xy", "left_pupil_xy", "lip_openness_ratio", "restore_info", "face_cfg_scale")
    OUTPUT_TOOLTIPS = (
        "Per-frame pose+face+iris dict bundle. Feed into Draw ViT Pose (V2).",
        "Cropped face IMAGE batch suitable for face-id encoders.",
        "Key-frame body points as JSON string (debug).",
        "Per-frame body BBOX list.",
        "Per-frame face BBOX list.",
        "Iris/gaze JSON dump (debug).",
        "Annotated debug IMAGE batch (skeleton overlay).",
        "Right pupil pixel xy as JSON (per frame).",
        "Left pupil pixel xy as JSON (per frame).",
        "Mouth-open scalar list (0=closed, 1=wide).",
        "Per-frame {x1,y1,x2,y2,size,frame_shape} dict for paste-back nodes.",
        "CFG scale for the face conditioning input. Wire into the Wan-Animate sampler's face CFG. 1.0 = CFG off (paper default).",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    def process(
        self,
        model,
        images,
        width,
        height,
        detection_threshold,
        pose_threshold,
        use_clahe,
        use_blur_for_pose,
        blur_radius,
        blur_sigma,
        use_face_smoothing,
        face_smoothing_strength,
        use_constant_face_box,
        face_box_size_px,
        use_iris_smoothing,
        iris_smoothing_strength,
        iris_smoothing_method="one_euro",
        iris_one_euro_min_cutoff=1.0,
        iris_one_euro_beta=0.05,
        gaze_lock_eyes=True,
        gaze_lock_strength=0.7,
        use_mediapipe_face=True,
        use_blendshape_gaze=True,
        gaze_one_euro_min_cutoff=1.7,
        gaze_one_euro_beta=0.3,
        gaze_max_yaw_deg=30.0,
        gaze_max_pitch_deg=25.0,
        crop_mode="default",
        frame0_cx=-1,
        frame0_cy=-1,
        frame0_size=0,
        keyframes_json="[]",
        smoothing_method="one_euro",
        crop_one_euro_min_cutoff=1.0,
        crop_one_euro_beta=0.05,
        crop_gaussian_window=7,
        eye_align_mode="default",
        eye_y_fraction=0.30,
        face_cfg_scale=1.0,
        bbox_override=None,
    ):
        detector = model["yolo"]
        pose_model = model["vitpose"]

        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = detection_threshold

        B, H, W, C = images.shape
        shape = np.array([H, W])[None]
        images_np = images.detach().cpu().numpy() if hasattr(images, "detach") else images.cpu().numpy()

        # --- Prepare blurred version for detection & pose ---
        if use_blur_for_pose:
            ksize = int(blur_radius) * 2 + 1
            images_blurred = np.stack([
                cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
                for img in images_np
            ])
        else:
            images_blurred = images_np

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B * 2)
        progress = 0
        bboxes = []

        # --- YOLO detection (on blurred) ---
        for img in _IC.track(
            images_blurred, B, "WanAnimateV2: YOLO bbox detect",
        ):
            detections = detector(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0]
            if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
                bboxes.append(detections[0]["bbox"])
            else:
                bboxes.append(None)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        # --- Pose detection (on blurred) ---
        kp2ds = []
        for img, bbox in _IC.track(
            zip(images_blurred, bboxes), B,
            "WanAnimateV2: pose keypoint extract",
        ):
            if (
                bbox is None
                or len(bbox) < 5
                or bbox[4] <= 0
                or (bbox[2] - bbox[0]) < 10
                or (bbox[3] - bbox[1]) < 10
            ):
                bbox_use = np.array([0, 0, img.shape[1], img.shape[0], 1.0], dtype=np.float32)
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_crop = preprocess_for_pose(img_crop, use_clahe)
            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)

            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()
        kp2ds = np.concatenate(kp2ds, 0)

        # --- Confidence threshold for keypoints ---
        if pose_threshold > 0.0:
            kp2ds[..., 2] = np.where(kp2ds[..., 2] < pose_threshold, 0, kp2ds[..., 2])

        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)

        # --- Raw face bboxes (from blurred pose keypoints; values are in pixel space) ---
        raw_face_bboxes = []
        for meta in pose_metas:
            bbox_face = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(H, W))
            # Ensure ints and within bounds
            x1, x2, y1, y2 = map(int, bbox_face)
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))
            # Fallback if invalid
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, min(W, 128), min(H, 128)
            raw_face_bboxes.append((x1, x2, y1, y2))

        # --- Convert to centers and raw face sizes (for smoothing) ---
        raw_centers = []
        raw_face_sizes = []
        for (x1, x2, y1, y2) in raw_face_bboxes:
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            raw_centers.append(np.array([cx, cy], dtype=np.float32))
            raw_face_sizes.append(float(max(x2 - x1, y2 - y1)))

        crop_mode_str = str(crop_mode)
        jitterless = crop_mode_str == "jitterless"
        crop_off = crop_mode_str == "default"

        if crop_off:
            # ── default: raw detected bboxes, no smoothing, no constant-size ──
            face_bboxes = list(raw_face_bboxes)
        elif jitterless:
            # ── Jitterless crop pipeline ─────────────────────────────────
            # 1. Resolve the frame-0 anchor (size & center) with priority:
            #      bbox_override > frame0_cx/cy widgets > raw detection.
            #    The locked size is then used for the whole clip.
            anchor_cx = None
            anchor_cy = None
            anchor_size = None
            if bbox_override is not None:
                try:
                    bb = bbox_override
                    # Accept (x1,y1,x2,y2) or (x1,x2,y1,y2) — heuristic:
                    if isinstance(bb, (list, tuple)) and len(bb) > 0 and isinstance(bb[0], (list, tuple)):
                        bb = bb[0]
                    bb = [float(v) for v in (bb[:4] if hasattr(bb, "__len__") else [])]
                    if len(bb) == 4:
                        # detect ordering by checking which pair is closer
                        a, b, c, d = bb
                        # try (x1,y1,x2,y2)
                        x1o, y1o, x2o, y2o = sorted([a, c])[0], sorted([b, d])[0], sorted([a, c])[1], sorted([b, d])[1]
                        anchor_cx = 0.5 * (x1o + x2o)
                        anchor_cy = 0.5 * (y1o + y2o)
                        anchor_size = max(x2o - x1o, y2o - y1o)
                except Exception as e:
                    print(f"[PoseAndFaceDetectionV2] bbox_override parse failed: {e}; ignoring.")
            if (anchor_cx is None) and frame0_cx >= 0 and frame0_cy >= 0:
                anchor_cx = float(frame0_cx)
                anchor_cy = float(frame0_cy)
            if anchor_cx is None and len(raw_centers) > 0:
                anchor_cx = float(raw_centers[0][0])
                anchor_cy = float(raw_centers[0][1])
            if anchor_size is None or anchor_size <= 0:
                if frame0_size and int(frame0_size) > 0:
                    anchor_size = float(frame0_size)
                else:
                    anchor_size = float(face_box_size_px)
            anchor_size = float(np.clip(anchor_size, 8.0, max(W, H)))
            anchor_cx = 0.0 if anchor_cx is None else float(np.clip(anchor_cx, 0.0, W - 1))
            anchor_cy = 0.0 if anchor_cy is None else float(np.clip(anchor_cy, 0.0, H - 1))

            # 1b. Face-scale ratio.
            #    anchor_face_size_0: how many pixels wide the YOLO face bbox
            #    was on frame 0.  anchor_scale_ratio = anchor_size / that width.
            #    Per frame N:  target_crop_size_N = raw_face_size_N * ratio.
            #    This keeps the face occupying the same fraction of the output
            #    512×512 tile regardless of camera pan / zoom.
            anchor_face_size_0 = float(raw_face_sizes[0]) if raw_face_sizes else anchor_size
            if anchor_face_size_0 < 4.0:
                # No reliable detection on frame 0 — fall back to constant size.
                anchor_face_size_0 = anchor_size
            anchor_scale_ratio = anchor_size / max(anchor_face_size_0, 4.0)
            _missing_bbox = (0, 0, min(W, 128), min(H, 128))

            # 2. Build the per-frame target-center series.
            #    Start from raw detected centers (so the face is followed
            #    when the user adds no keyframes) and overwrite with
            #    interpolated keyframe centers wherever keyframes exist.
            kfs = _parse_keyframes_json(keyframes_json, B)
            # Always anchor frame 0 to (anchor_cx, anchor_cy, anchor_size)
            kfs = [k for k in kfs if k["frame"] != 0]
            kfs.insert(0, {"frame": 0, "cx": anchor_cx, "cy": anchor_cy, "size": int(anchor_size)})

            kf_cx, kf_cy, kf_sz = _interp_keyframes(
                kfs, B,
                default_cx=anchor_cx,
                default_cy=anchor_cy,
                default_size=anchor_size,
            )

            # If user supplied >1 keyframes (besides frame 0), trust them
            # fully for the center; otherwise blend with the smoothed raw
            # detection so the crop still tracks the face.
            user_added = len([k for k in kfs if k["frame"] != 0])
            target_centers = np.stack(raw_centers, axis=0).astype(np.float32) \
                if raw_centers else np.zeros((B, 2), dtype=np.float32)
            if user_added >= 1:
                # The user-controlled trajectory wins.
                target_centers[:, 0] = kf_cx
                target_centers[:, 1] = kf_cy
            else:
                # No extra keyframes — keep the detected centers but
                # snap frame 0 to the anchor (so the user's manual frame-0
                # override actually takes effect).
                target_centers[0, 0] = anchor_cx
                target_centers[0, 1] = anchor_cy

            # 2b. Build per-frame target crop sizes.
            #    If user keyframes supply explicit sizes, trust them.
            #    Otherwise scale each frame's crop proportionally to the
            #    detected face size so the face stays at constant apparent
            #    scale in the output (face-scale-preserving crop).
            if user_added >= 1 and kf_sz is not None:
                target_sizes = kf_sz.copy()
            else:
                target_sizes = np.array(
                    [float(s) * anchor_scale_ratio
                     if (s >= 4.0 and raw_face_bboxes[idx] != _missing_bbox)
                     else anchor_size
                     for idx, s in enumerate(raw_face_sizes)],
                    dtype=np.float32)
            target_sizes = np.clip(target_sizes, 8.0, float(min(W, H)))

            # 3. Smooth the trajectory (centers + crop sizes independently).
            image_diag = float((W * W + H * H) ** 0.5)
            smoothed_centers = _smooth_centers(
                target_centers,
                method=str(smoothing_method),
                ema_strength=face_smoothing_strength,
                image_diag=image_diag,
                one_euro_min_cutoff=crop_one_euro_min_cutoff,
                one_euro_beta=crop_one_euro_beta,
                gaussian_window=int(crop_gaussian_window),
            )
            smoothed_sizes = _smooth_1d(
                target_sizes,
                method=str(smoothing_method),
                ema_strength=face_smoothing_strength,
                scale_norm=anchor_size,
                one_euro_min_cutoff=crop_one_euro_min_cutoff,
                one_euro_beta=crop_one_euro_beta,
                gaussian_window=int(crop_gaussian_window),
            )

            # 4. Hold-last-known on missing detections (raw_face_bbox was
            #    the placeholder fallback (0,0,128,128) — treat that as
            #    "missing" and carry forward the previous smoothed values).
            for i in range(1, B):
                if raw_face_bboxes[i] == _missing_bbox and user_added == 0:
                    smoothed_centers[i] = smoothed_centers[i - 1]
                    smoothed_sizes[i] = smoothed_sizes[i - 1]

            # 5. Build face-scale-preserving square crops.
            #    Each frame uses its own smoothed crop size so that the face
            #    fills the same fraction of the 512×512 output every frame.
            face_bboxes = []
            for i in range(B):
                size_i = float(smoothed_sizes[i])
                size_i = float(np.clip(size_i, 8.0, min(W, H)))
                size_i_int = int(round(size_i))
                half = size_i / 2.0
                cx_i = float(smoothed_centers[i, 0])
                cy_i = float(smoothed_centers[i, 1])
                x1 = int(np.clip(cx_i - half, 0, W - size_i_int))
                y1 = int(np.clip(cy_i - half, 0, H - size_i_int))
                x2 = x1 + size_i_int
                y2 = y1 + size_i_int
                face_bboxes.append((x1, x2, y1, y2))
        else:
            # ── Legacy auto pipeline (unchanged) ──────────────────────────
            # --- Temporal smoothing for centers (motion-adaptive) ---
            if use_face_smoothing and len(raw_centers) > 1:
                base_strength = float(np.clip(face_smoothing_strength, 0.0, 1.0))
                smoothed_centers = [raw_centers[0].copy()]
                norm = max(1.0, (W + H) / 2.0)
                for i in range(1, len(raw_centers)):
                    curr = raw_centers[i]
                    prev = smoothed_centers[-1]
                    motion = float(np.mean(np.abs(curr - prev)) / norm)
                    # More motion -> less smoothing
                    k = 5.0
                    dynamic_strength = base_strength * np.exp(-motion * k)
                    alpha = 1.0 - dynamic_strength  # 1=no smoothing, 0=full smoothing
                    smoothed = alpha * curr + (1.0 - alpha) * prev
                    smoothed_centers.append(smoothed.astype(np.float32))
            else:
                smoothed_centers = raw_centers

            # --- Build final face bboxes from smoothed centers ---
            face_bboxes = []
            if use_constant_face_box:
                half = face_box_size_px / 2.0
                for c in smoothed_centers:
                    cx, cy = float(c[0]), float(c[1])
                    # Clamp so the square stays in bounds
                    x1 = int(np.clip(cx - half, 0, W - face_box_size_px))
                    y1 = int(np.clip(cy - half, 0, H - face_box_size_px))
                    x2 = x1 + int(face_box_size_px)
                    y2 = y1 + int(face_box_size_px)
                    face_bboxes.append((x1, x2, y1, y2))
            else:
                # If not constant size, just slightly pad the original (helps tilted heads)
                for (x1, x2, y1, y2), c in zip(raw_face_bboxes, smoothed_centers):
                    w = x2 - x1
                    h = y2 - y1
                    x_pad = int(w * 0.2)
                    y_pad = int(h * 0.3)
                    # Recenter to smoothed center but keep variable size
                    cx, cy = float(c[0]), float(c[1])
                    half_w = (w / 2.0) + x_pad
                    half_h = (h / 2.0) + y_pad
                    nx1 = int(np.clip(cx - half_w, 0, W - 1))
                    ny1 = int(np.clip(cy - half_h, 0, H - 1))
                    nx2 = int(np.clip(cx + half_w, 0, W))
                    ny2 = int(np.clip(cy + half_h, 0, H))
                    if nx2 <= nx1 or ny2 <= ny1:
                        nx1, ny1, nx2, ny2 = x1, y1, x2, y2  # fallback to raw
                    face_bboxes.append((nx1, nx2, ny1, ny2))

        # --- Wan-Animate paper recommendation #1: eye-centred crop -----
        # Vertically shift each per-frame face bbox so the eyes land at
        # `eye_y_fraction` of the crop height (0.30 = upper third).
        # This is a temporal-stable post-pass on top of whatever crop
        # mode the user picked: crop SIZE is preserved, only y1/y2 shift.
        # Rationale (paper Sec. 3.2 + 4.3): the face encoder reads the
        # holistic appearance from a fixed-size 512x512 crop. If the eyes
        # drift around within that crop frame-to-frame the encoder learns
        # spurious gaze cues and produces flicker / wrong gaze direction.
        if str(eye_align_mode) == "eye_upper_third":
            ey_frac = float(np.clip(eye_y_fraction, 0.05, 0.80))
            adjusted = []
            for idx, fb in enumerate(face_bboxes):
                eye_xy = compute_eye_midpoint_from_face_kps(
                    pose_metas[idx]['keypoints_face'], W, H
                )
                if eye_xy is None:
                    adjusted.append(fb)
                    continue
                adjusted.append(
                    adjust_bbox_eye_upper_third(fb, eye_xy, W, H, ey_frac)
                )
            face_bboxes = adjusted

        # --- Face crops from sharp original frames ---
        face_images = []
        for idx, (x1, x2, y1, y2) in enumerate(face_bboxes):
            face_image = images_np[idx][y1:y2, x1:x2]
            if face_image.size == 0:
                fallback_size = int(min(H, W) * 0.3)
                fx1 = (W - fallback_size) // 2
                fx2 = fx1 + fallback_size
                fy1 = int(H * 0.1)
                fy2 = fy1 + fallback_size
                face_image = images_np[idx][fy1:fy2, fx1:fx2]
                if face_image.size == 0:
                    face_image = np.zeros((fallback_size, fallback_size, C), dtype=images_np.dtype)
            face_image = cv2.resize(face_image, (512, 512))
            face_images.append(face_image)

        face_images_np = np.stack(face_images, 0)
        face_images_tensor = torch.from_numpy(face_images_np)

        retarget_pose_metas = [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas]

        # use first bbox for return (legacy)
        bbox0 = bboxes[0]
        bbox = np.array(bbox0).flatten() if bbox0 is not None else np.array([0, 0, 0, 0])
        bbox_ints = tuple(int(v) for v in bbox[:4]) if bbox.shape[0] >= 4 else (0, 0, 0, 0)

        # key frame points (unchanged)
        key_points_index = [0, 1, 2, 5, 8, 11, 10, 13]
        body_key_points = pose_metas[0]['keypoints_body']
        keypoints_body = np.array([body_key_points[i] for i in key_points_index if body_key_points[i] is not None])[:, :2]
        wh = np.array([[pose_metas[0]['width'], pose_metas[0]['height']]])
        points = (keypoints_body * wh).astype(np.int32)
        points_dict_list = [{"x": int(p[0]), "y": int(p[1])} for p in points]

        # --- Iris + gaze estimation ---
        # Preferred path: FaceLandmarker (Tasks API) — returns 478-pt mesh,
        # iris ring, and 52 ARKit blend shapes from which we derive
        # head-pose-corrected per-eye yaw/pitch (radians). Falls back to
        # legacy FaceMesh and finally to the OpenCV pupil voter.
        all_iris = []
        all_lip_ratios = []
        mp_enabled = bool(use_mediapipe_face) and _MP_AVAILABLE
        bs_enabled = bool(use_blendshape_gaze) and _GAZE_BS_IMPORTED and (
            _gaze_bs is not None and _gaze_bs.is_available()
        )
        max_yaw_rad = math.radians(float(gaze_max_yaw_deg))
        max_pitch_rad = math.radians(float(gaze_max_pitch_deg))
        mp_used_count = 0
        bs_used_count = 0
        for idx, meta in _IC.track(
            list(enumerate(pose_metas)), len(pose_metas),
            "WanAnimateV2: face/gaze per-frame",
        ):
            mp_result = None
            # IMPORTANT: MediaPipe FaceLandmarker expects the ENTIRE face
            # in its input. The YOLO/raw face bbox is often tight (eyes
            # outside the box), which makes the model produce landmarks
            # in wrong locations. We pad the *landmark* crop here, while
            # keeping `face_bboxes[idx]` as the user-configured output
            # crop (which downstream consumers and the cyan FACE rect use).
            rx1, rx2, ry1, ry2 = raw_face_bboxes[idx]
            rw, rh = rx2 - rx1, ry2 - ry1
            # Square + 40% padding around the raw face box so eyes /
            # forehead / chin always fit, then clamp to image bounds.
            side = max(rw, rh) * 1.4
            cx_r = 0.5 * (rx1 + rx2)
            cy_r = 0.5 * (ry1 + ry2)
            half = 0.5 * side
            mx1 = int(max(0, round(cx_r - half)))
            my1 = int(max(0, round(cy_r - half)))
            mx2 = int(min(W, round(cx_r + half)))
            my2 = int(min(H, round(cy_r + half)))
            mcw, mch = mx2 - mx1, my2 - my1
            crop_rgb = None
            if mcw > 8 and mch > 8:
                crop_rgb = (np.clip(images_np[idx][my1:my2, mx1:mx2], 0, 1) * 255).astype(np.uint8)
            # Output-crop coords (kept for parity with legacy code paths
            # that still expected `x1..y2` in scope below).
            x1, x2, y1, y2 = face_bboxes[idx]
            cw, ch = x2 - x1, y2 - y1

            # 1) Try FaceLandmarker Tasks API (with blend-shape gaze)
            if bs_enabled and crop_rgb is not None:
                mp_result = _run_face_landmarker_on_face_crop(
                    crop_rgb, (mx1, my1), (mcw, mch), W, H,
                )
                if mp_result is not None:
                    bs_used_count += 1
            # 2) Fall back to legacy FaceMesh (no blend shapes)
            if mp_result is None and mp_enabled and crop_rgb is not None:
                mp_result = _run_mediapipe_on_face_crop(
                    crop_rgb, (mx1, my1), (mcw, mch), W, H,
                )

            if mp_result is not None:
                mp_used_count += 1
                # Override face_kps[1:69] with MediaPipe-derived 68 landmarks
                # (face_kps[0] is the body-anchored face anchor from ViTPose;
                # leave it intact so Wan's pose encoder keeps its global hook).
                face_kps = meta['keypoints_face']
                if face_kps.shape[0] >= 69:
                    face_kps[1:69, :] = mp_result['kps68_norm']
                    meta['keypoints_face'] = face_kps

                rix, riy = mp_result['right_iris_px']
                lix, liy = mp_result['left_iris_px']
                iris_result = {
                    'right_iris': {'x': rix, 'y': riy, 'confidence': 1.0,
                                    'radius': mp_result['right_iris_radius_px']},
                    'left_iris':  {'x': lix, 'y': liy, 'confidence': 1.0,
                                    'radius': mp_result['left_iris_radius_px']},
                    'source': mp_result.get('source', 'face_mesh'),
                }
                gaze_bs = mp_result.get('gaze_blendshape')
                if gaze_bs is not None:
                    # Blend-shape path: rescale yaw/pitch to user-tuned max
                    # angles (defaults already factor in MAX_GAZE_*_RAD,
                    # so divide by them and remultiply by the new max).
                    base_yaw = _gaze_bs.MAX_GAZE_YAW_RAD if _gaze_bs else 1.0
                    base_pitch = _gaze_bs.MAX_GAZE_PITCH_RAD if _gaze_bs else 1.0
                    # Gaze-arrow screen-space convention (FIX, May 2026):
                    # The renderer in draw_debug_overlay() draws the arrow as
                    #   ex = ix + dx*L;  ey = iy + dy*L
                    # so dx/dy MUST be in image-pixel direction (right=+x,
                    # down=+y). The legacy 2D-offset path correctly derives
                    # dx/dy from the iris pixel offset to the eye centroid;
                    # we replicate the same screen-space derivation here so
                    # both gaze paths agree. Synthesising dx from
                    # ``-sin(yaw_rad)`` (anatomical-camera convention) made
                    # the arrow point opposite to the iris in the rendered
                    # debug view — that was the user-reported bug.
                    kps_px_for_gaze = meta['keypoints_face'][:, :2] * np.array([W, H])
                    iris_pix = {
                        'right': (float(rix), float(riy)),
                        'left':  (float(lix), float(liy)),
                    }
                    eye_idx_map = {
                        'right': _RIGHT_EYE_IDX,
                        'left':  _LEFT_EYE_IDX,
                    }
                    for eye_name in ('right', 'left'):
                        e = dict(gaze_bs[eye_name])
                        e['yaw_rad'] = float(e['yaw_rad']) / max(base_yaw, 1e-6) * max_yaw_rad
                        e['pitch_rad'] = float(e['pitch_rad']) / max(base_pitch, 1e-6) * max_pitch_rad
                        e['source'] = 'blendshape'
                        # Screen-space dx/dy = unit vector from eye centroid
                        # to detected iris pixel. Matches path A exactly and
                        # is automatically correct for both mirrored
                        # (selfie) and unmirrored camera inputs because the
                        # iris position is sampled from the same image
                        # frame as the eye centroid.
                        ipx, ipy = iris_pix[eye_name]
                        geo = np.mean(kps_px_for_gaze[eye_idx_map[eye_name]], axis=0)
                        ddx = ipx - float(geo[0])
                        ddy = ipy - float(geo[1])
                        nrm = float(math.hypot(ddx, ddy))
                        if nrm > 1e-6:
                            e['dx'] = round(ddx / nrm, 4)
                            e['dy'] = round(ddy / nrm, 4)
                        else:
                            e['dx'] = 0.0
                            e['dy'] = 0.0
                        e['magnitude'] = float(math.hypot(e['yaw_rad'], e['pitch_rad']))
                        iris_result[f'{eye_name}_gaze'] = e
                    iris_result['blendshapes'] = mp_result.get('blendshapes', {})
                else:
                    # Legacy fallback: 2D iris-offset gaze (kept for
                    # backward compatibility when blend shapes are off).
                    kps_px_local = meta['keypoints_face'][:, :2] * np.array([W, H])
                    for eye_name, iris_xy, eye_idx in (
                        ('right', (rix, riy), _RIGHT_EYE_IDX),
                        ('left',  (lix, liy), _LEFT_EYE_IDX),
                    ):
                        geo = np.mean(kps_px_local[eye_idx], axis=0)
                        dx = iris_xy[0] - float(geo[0])
                        dy = iris_xy[1] - float(geo[1])
                        norm = max(np.hypot(dx, dy), 1e-6)
                        iris_result[f'{eye_name}_gaze'] = {
                            'dx': round(dx / norm, 4),
                            'dy': round(dy / norm, 4),
                            'yaw_rad': 0.0,
                            'pitch_rad': 0.0,
                            'source': 'iris_offset_2d',
                        }
                all_iris.append(iris_result)
                all_lip_ratios.append(float(mp_result['lip_openness_ratio']))
            else:
                # Fallback: legacy ViTPose + image-based pupil voter
                iris_result = estimate_iris_positions(
                    meta['keypoints_face'], images_np[idx], W, H,
                )
                iris_result['source'] = 'pupil_voter'
                all_iris.append(iris_result)
                all_lip_ratios.append(0.0)

        if mp_enabled or bs_enabled:
            logging.getLogger(__name__).info(
                "Face mesh: %d/%d frames (%.1f%%); blend-shape gaze: %d/%d frames (%.1f%%)",
                mp_used_count, B, 100.0 * mp_used_count / max(B, 1),
                bs_used_count, B, 100.0 * bs_used_count / max(B, 1),
            )

        # --- Temporal smoothing ---
        # Iris pixel positions: choose method (one_euro recommended).
        if use_iris_smoothing and len(all_iris) > 1 and iris_smoothing_method != "none":
            if iris_smoothing_method == "one_euro":
                # Per-eye, per-axis One-Euro filter on the iris pixel
                # coordinates. Far better than EMA at separating jitter
                # from real saccades — exactly what the Wan 2.2 face
                # encoder needs to reproduce stable gaze.
                try:
                    from .gaze_blendshape import OneEuroFilter as _OEF
                    fps_est = 30.0
                    filt_kw = dict(
                        freq=fps_est,
                        min_cutoff=float(iris_one_euro_min_cutoff),
                        beta=float(iris_one_euro_beta),
                    )
                    filters = {
                        ("right_iris", "x"): _OEF(**filt_kw),
                        ("right_iris", "y"): _OEF(**filt_kw),
                        ("left_iris",  "x"): _OEF(**filt_kw),
                        ("left_iris",  "y"): _OEF(**filt_kw),
                    }
                    for fr in all_iris:
                        for eye_key in ("right_iris", "left_iris"):
                            iris = fr.get(eye_key)
                            if not isinstance(iris, dict):
                                continue
                            # Skip blink frames so the filter doesn't drift.
                            if float(iris.get("confidence", 1.0)) < 0.05:
                                continue
                            iris["x"] = filters[(eye_key, "x")](float(iris["x"]))
                            iris["y"] = filters[(eye_key, "y")](float(iris["y"]))
                except Exception as _exc:
                    logging.getLogger(__name__).warning(
                        "Iris one-euro smoothing failed (%s); falling back to EMA.", _exc,
                    )
                    iris_smoothing_method = "ema"

            if iris_smoothing_method == "ema":
                strength = float(np.clip(iris_smoothing_strength, 0.0, 1.0))
                for eye_key in ('right_iris', 'left_iris'):
                    prev_x = all_iris[0][eye_key]['x']
                    prev_y = all_iris[0][eye_key]['y']
                    for i in range(1, len(all_iris)):
                        cur = all_iris[i][eye_key]
                        alpha = 1.0 - strength
                        cur['x'] = alpha * cur['x'] + strength * prev_x
                        cur['y'] = alpha * cur['y'] + strength * prev_y
                        prev_x, prev_y = cur['x'], cur['y']

        # Gaze yaw/pitch: one-euro filter per eye (low-lag, kills jitter).
        if bs_used_count > 0 and _GAZE_BS_IMPORTED and _gaze_bs is not None:
            try:
                fps_est = 30.0
                smoother = _gaze_bs.GazeStreamSmoother(
                    fps=fps_est,
                    min_cutoff=float(gaze_one_euro_min_cutoff),
                    beta=float(gaze_one_euro_beta),
                )
                for fr in all_iris:
                    rg = fr.get('right_gaze')
                    lg = fr.get('left_gaze')
                    if not (isinstance(rg, dict) and isinstance(lg, dict)
                            and 'yaw_rad' in rg and 'yaw_rad' in lg):
                        continue
                    smoothed = smoother.step({
                        'left':  {'yaw_rad': float(lg['yaw_rad']),
                                  'pitch_rad': float(lg['pitch_rad'])},
                        'right': {'yaw_rad': float(rg['yaw_rad']),
                                  'pitch_rad': float(rg['pitch_rad'])},
                    })
                    for side, key in (('right', 'right_gaze'), ('left', 'left_gaze')):
                        e = fr[key]
                        e['yaw_rad'] = smoothed[side]['yaw_rad']
                        e['pitch_rad'] = smoothed[side]['pitch_rad']
                        e['dx'] = smoothed[side]['dx']
                        e['dy'] = smoothed[side]['dy']
                        e['magnitude'] = smoothed[side]['magnitude']
            except Exception as _exc:
                logging.getLogger(__name__).warning(
                    "Gaze one-euro smoothing failed (%s); using raw values.", _exc,
                )

        # --- Cross-eye gaze locking (NEW) ---------------------------------
        # The single most common Wan-Animate artefact is the two eyes
        # pointing in slightly different directions. Per-eye OneEuro
        # smoothing leaves them independent; here we pull each eye toward
        # the per-frame average (yaw, pitch) of the two eyes.
        # Re-derive dx/dy/magnitude after the blend so debug arrows match.
        if gaze_lock_eyes and len(all_iris) > 0:
            lock = float(np.clip(gaze_lock_strength, 0.0, 1.0))
            if lock > 0.0:
                for fr in all_iris:
                    rg = fr.get('right_gaze')
                    lg = fr.get('left_gaze')
                    if not (isinstance(rg, dict) and isinstance(lg, dict)
                            and 'yaw_rad' in rg and 'yaw_rad' in lg):
                        continue
                    avg_yaw = 0.5 * (float(rg['yaw_rad']) + float(lg['yaw_rad']))
                    avg_pitch = 0.5 * (float(rg['pitch_rad']) + float(lg['pitch_rad']))
                    for e in (rg, lg):
                        e['yaw_rad']   = (1.0 - lock) * float(e['yaw_rad'])   + lock * avg_yaw
                        e['pitch_rad'] = (1.0 - lock) * float(e['pitch_rad']) + lock * avg_pitch
                        dx = -math.sin(e['yaw_rad'])
                        dy = -math.sin(e['pitch_rad'])
                        n = math.hypot(dx, dy)
                        if n > 1e-6:
                            e['dx'] = round(dx / n, 4)
                            e['dy'] = round(dy / n, 4)
                        else:
                            e['dx'] = 0.0
                            e['dy'] = 0.0
                        e['magnitude'] = float(math.hypot(e['yaw_rad'], e['pitch_rad']))

        # Build per-frame iris output
        iris_output = []
        for idx, iris in enumerate(all_iris):
            iris_output.append({
                'frame': idx,
                'right_iris': iris.get('right_iris'),
                'left_iris': iris.get('left_iris'),
                'right_gaze': iris.get('right_gaze'),
                'left_gaze': iris.get('left_gaze'),
                'lip_openness_ratio': all_lip_ratios[idx] if idx < len(all_lip_ratios) else 0.0,
            })

        pose_data = {
            "pose_metas": retarget_pose_metas,
            "pose_metas_original": pose_metas,
            "iris_data": all_iris,
            "lip_openness_ratios": all_lip_ratios,
            # MANUAL bug-fix (Apr 2026): expose source frame dims + target
            # render dims so DrawViTPoseV2 can map iris pixel coords (which
            # live in the *original* frame coord system) into the retargeted
            # canvas using the same padding_resize transform that body
            # keypoints went through.
            "source_size": (int(H), int(W)),
            "target_size": (int(height), int(width)),
        }

        # --- Debug visualisation ---
        debug_frames = []
        for idx in _IC.track(
            range(B), B, "WanAnimateV2: per-frame finalize",
        ):
            frame = images_np[idx]
            if frame.dtype != np.uint8:
                frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame_u8 = frame.copy()
            vis = draw_debug_overlay(
                frame_u8, pose_metas[idx]['keypoints_face'],
                all_iris[idx], face_bboxes[idx], bboxes[idx], W, H,
            )
            debug_frames.append(vis)
        debug_np = np.stack(debug_frames, 0).astype(np.float32) / 255.0
        debug_tensor = torch.from_numpy(debug_np)

        # --- Aggregate per-frame eye/lip outputs ---
        right_pupil_seq = [
            [round(it['right_iris']['x'], 3), round(it['right_iris']['y'], 3)]
            for it in all_iris
        ]
        left_pupil_seq = [
            [round(it['left_iris']['x'], 3), round(it['left_iris']['y'], 3)]
            for it in all_iris
        ]
        mean_lip_openness = float(np.mean(all_lip_ratios)) if all_lip_ratios else 0.0

        # Per-frame paste-back metadata. Always emit so downstream nodes
        # can rely on it regardless of crop_mode.
        restore_info = {
            "frame_shape": [int(H), int(W)],
            "resized_to": [512, 512],
            "crop_mode": str(crop_mode),
            "crops": [
                {
                    "frame": int(i),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "size": [int(y2 - y1), int(x2 - x1)],
                }
                for i, (x1, x2, y1, y2) in enumerate(face_bboxes)
            ],
        }

        return (
            pose_data,
            face_images_tensor,
            json.dumps(points_dict_list),
            [bbox_ints],
            face_bboxes,
            json.dumps(iris_output),
            debug_tensor,
            json.dumps(right_pupil_seq),
            json.dumps(left_pupil_seq),
            mean_lip_openness,
            restore_info,
            float(face_cfg_scale),
        )


# ---------------------------------------------------
# Draw ViTPose
# ---------------------------------------------------
class DrawViTPoseV2:
    DESCRIPTION = (
        "Render the detected skeleton, face landmarks, iris pupils and gaze "
        "arrows onto a clean canvas at the target Wan 2.2 latent resolution. "
        "Outputs an IMAGE batch ready to drop into a Wan-Animate sampler."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_data":         ("POSEDATA", {"tooltip": "From Pose and Face Detection (V2)."}),
                "width":             ("INT",   {"default": 832, "min": 64, "max": 2048, "tooltip": "Render canvas width (px). Match the sampler latent size."}),
                "height":            ("INT",   {"default": 480, "min": 64, "max": 2048, "tooltip": "Render canvas height (px). Match the sampler latent size."}),
                "retarget_padding":  ("INT",   {"default": 16,  "min": 0,  "max": 512, "tooltip": "Padding (px) added around the body bbox when retargeting. Larger = more headroom for big motions."}),
                "body_stick_width":  ("INT",   {"default": -1,  "min": -1, "max": 20,  "tooltip": "Body skeleton stick width in px. -1 = auto from canvas size."}),
                "hand_stick_width":  ("INT",   {"default": -1,  "min": -1, "max": 20,  "tooltip": "Hand skeleton stick width in px. -1 = auto."}),
                "draw_head":         ("BOOLEAN", {"default": True, "tooltip": "Draw the head/face skeleton (eyes, nose, ears)."}),
                "pose_draw_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold for drawing."}),
            },
            # MANUAL bug-fix (Apr 2026): MediaPipe iris/gaze integration.
            # The Pose-and-Face-Detection node already produces per-frame
            # iris pixel coords + gaze vectors in pose_data["iris_data"];
            # these optional widgets let the rendered pose image carry
            # explicit pupil + gaze cues that the Wan 2.2 Animate sampler
            # consumes through cross-attention.  All defaults preserve the
            # legacy behaviour when the operator does not opt in.
            "optional": {
                "draw_iris": ("BOOLEAN", {"default": True,
                    "tooltip": "Draw iris/pupil markers from MediaPipe iris_data."}),
                "draw_gaze": ("BOOLEAN", {"default": True,
                    "tooltip": "Draw gaze direction arrows from iris_data."}),
                "iris_radius": ("INT", {"default": 4, "min": 1, "max": 20,
                    "tooltip": "Pupil circle radius in pixels."}),
                "gaze_arrow_len": ("INT", {"default": 30, "min": 4, "max": 200,
                    "tooltip": "Length of gaze direction arrow in pixels."}),
                "iris_min_confidence": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Skip iris frames whose detection confidence is below this."}),
                "iris_color": (["white", "magenta", "yellow", "green"], {"default": "white",
                    "tooltip": "Color of the drawn pupil; magenta gives strongest sampler signal."}),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("pose_images", )
    OUTPUT_TOOLTIPS = ("Rendered skeleton IMAGE batch. Feed into your Wan 2.2 Animate sampler.",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    @staticmethod
    def _padding_resize_transform(src_h, src_w, out_h, out_w):
        """Replicate utils.padding_resize math as a (scale, ox, oy) transform.

        Returns the per-pixel scale and (offset_x, offset_y) that map a
        source-coord (x, y) into the padded target canvas of size out_h*out_w.
        """
        if (src_h / max(src_w, 1)) > (out_h / max(out_w, 1)):
            new_w = int(out_h / src_h * src_w)
            scale = out_h / src_h
            ox = (out_w - new_w) // 2
            oy = 0
        else:
            new_h = int(out_w / src_w * src_h)
            scale = out_w / src_w
            ox = 0
            oy = (out_h - new_h) // 2
        return scale, ox, oy

    def _draw_iris_overlay(self, canvas, iris_dict, transform,
                            iris_radius, gaze_arrow_len, min_conf,
                            color_bgr, draw_iris, draw_gaze):
        if iris_dict is None:
            return
        scale, ox, oy = transform
        H, W = canvas.shape[:2]
        for eye_key, gaze_key in (("right_iris", "right_gaze"),
                                    ("left_iris", "left_gaze")):
            iris = iris_dict.get(eye_key)
            if not isinstance(iris, dict):
                continue
            try:
                conf = float(iris.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < min_conf:
                continue
            try:
                src_x = float(iris["x"]); src_y = float(iris["y"])
            except (KeyError, TypeError, ValueError):
                continue
            cx = int(round(src_x * scale + ox))
            cy = int(round(src_y * scale + oy))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if draw_iris:
                cv2.circle(canvas, (cx, cy), iris_radius, color_bgr, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), max(iris_radius + 2, 6),
                           (0, 0, 0), 1, cv2.LINE_AA)
            if draw_gaze:
                gaze = iris_dict.get(gaze_key)
                if isinstance(gaze, dict):
                    try:
                        dx = float(gaze.get("dx", 0.0))
                        dy = float(gaze.get("dy", 0.0))
                    except (TypeError, ValueError):
                        dx = dy = 0.0
                    if abs(dx) > 1e-4 or abs(dy) > 1e-4:
                        ex = int(round(cx + dx * gaze_arrow_len))
                        ey = int(round(cy + dy * gaze_arrow_len))
                        cv2.arrowedLine(canvas, (cx, cy), (ex, ey),
                                        color_bgr, 2, cv2.LINE_AA, tipLength=0.3)

    def process(self, pose_data, width, height, body_stick_width, hand_stick_width,
                draw_head, pose_draw_threshold, retarget_padding=64,
                draw_iris=True, draw_gaze=True,
                iris_radius=4, gaze_arrow_len=30,
                iris_min_confidence=0.05, iris_color="white"):
        pose_metas = pose_data["pose_metas"]
        draw_hand = hand_stick_width != 0

        # MANUAL bug-fix (Apr 2026): support optional iris drawing on top of
        # the rendered pose canvas.  iris_data is always rendered into the
        # *target* (width, height) coord system using the same padding-resize
        # transform that body keypoints went through.
        iris_data = pose_data.get("iris_data") or []
        src_size = pose_data.get("source_size")
        # RGB (cv2 expects BGR but we draw on a uint8 canvas that is later
        # converted to a float [0,1] tensor as RGB; OpenCV draws in BGR order
        # numerically, but since values are symmetric (white) or chosen to
        # match the eventual sampler signal we pick a single consistent
        # palette).  Here color tuples are (R, G, B) on the array directly.
        color_map = {
            "white":   (255, 255, 255),
            "magenta": (255, 0, 255),
            "yellow":  (255, 255, 0),
            "green":   (0, 255, 0),
        }
        iris_color_rgb = color_map.get(iris_color, (255, 255, 255))

        if src_size and len(src_size) == 2:
            transform = self._padding_resize_transform(
                int(src_size[0]), int(src_size[1]), int(height), int(width)
            )
        else:
            transform = None  # cannot retarget without source dims

        comfy_pbar = ProgressBar(len(pose_metas))
        progress = 0
        pose_images = []

        for idx, meta in _IC.track(
            list(enumerate(pose_metas)), len(pose_metas),
            "WanAnimateV2: draw pose images",
        ):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            pose_image = draw_aapose_by_meta_new(
                canvas,
                meta,
                draw_hand=draw_hand,
                draw_head=draw_head,
                body_stick_width=body_stick_width,
                hand_stick_width=hand_stick_width,
                threshold=pose_draw_threshold,
            )
            pose_image = padding_resize(pose_image, height, width)
            if transform is not None and idx < len(iris_data) and (draw_iris or draw_gaze):
                self._draw_iris_overlay(
                    pose_image, iris_data[idx], transform,
                    int(iris_radius), int(gaze_arrow_len),
                    float(iris_min_confidence), iris_color_rgb,
                    bool(draw_iris), bool(draw_gaze),
                )
            pose_images.append(pose_image)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_images_np = np.stack(pose_images, 0)
        pose_images_tensor = torch.from_numpy(pose_images_np).float() / 255.0
        return (pose_images_tensor, )


# ====================================================================
# Wan-Animate paper recommendation #4: face-quality gating.
# ====================================================================
class WanAnimateFaceQualityCheckV2:
    DESCRIPTION = (
        "Score each face crop on (a) Laplacian-variance sharpness and "
        "(b) eye-region brightness, then optionally repair bad frames by "
        "copying the previous good frame or by simple sharpening. Bad "
        "face conditioning frames cause the Wan-Animate face encoder to "
        "produce drifting / wrong-direction gaze (paper Sec. 4.3). "
        "Connect this BETWEEN Pose-and-Face-Detection (V2)'s `face_images` "
        "output and your downstream face-id encoder."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_images":           ("IMAGE", {"tooltip": "Per-frame 512x512 face crops (output of Pose and Face Detection V2)."}),
                "blur_threshold":        ("FLOAT", {"default": 50.0, "min": 0.0, "max": 5000.0, "step": 1.0, "tooltip": "Laplacian-variance threshold below which a frame is flagged as blurry. Typical sharp 512x512 frames score 100-1000; <50 indicates motion blur or out-of-focus."}),
                "min_eye_brightness":    ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Minimum mean luma of the eye-region strip (rows 30%-55%). Below this, eyes are likely closed or the frame is too dark for the encoder to read gaze."}),
                "auto_repair_bad_frames": ("BOOLEAN", {"default": True, "tooltip": "If true, repair frames flagged as bad. If false, just report stats."}),
                "repair_strategy":       (["copy_previous_good", "unsharp_mask", "skip"], {"default": "copy_previous_good", "tooltip": "copy_previous_good: replace with last good frame. unsharp_mask: deconvolve-style sharpening. skip: leave untouched but report."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("face_images_repaired", "good_frame_ratio", "report_json")
    OUTPUT_TOOLTIPS = (
        "Repaired face IMAGE batch (same shape as input).",
        "Fraction of frames that passed BOTH thresholds (0..1).",
        "JSON report: per-frame blur score, eye brightness, verdict, repair action.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    def _unsharp(self, frame_np):
        # Frame is float32 [0,1].
        u8 = (np.clip(frame_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (0, 0), sigmaX=1.5)
        sharp = cv2.addWeighted(u8, 1.5, blurred, -0.5, 0)
        return np.clip(sharp.astype(np.float32) / 255.0, 0.0, 1.0)

    def process(self, face_images, blur_threshold, min_eye_brightness,
                auto_repair_bad_frames, repair_strategy):
        if hasattr(face_images, "detach"):
            arr = face_images.detach().cpu().numpy()
        else:
            arr = np.asarray(face_images)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"WanAnimateFaceQualityCheckV2: expected (B,H,W,3); got {arr.shape}"
            )
        B = arr.shape[0]
        report = []
        good_count = 0
        repaired = arr.copy().astype(np.float32)
        last_good_idx = -1

        for i in range(B):
            frame = repaired[i]
            blur = compute_frame_blur_score(frame)
            eye_lum = compute_eye_region_brightness(frame)
            blur_ok = blur >= float(blur_threshold)
            lum_ok = eye_lum >= float(min_eye_brightness)
            ok = blur_ok and lum_ok
            action = "none"
            if ok:
                good_count += 1
                last_good_idx = i
            elif auto_repair_bad_frames:
                if repair_strategy == "copy_previous_good" and last_good_idx >= 0:
                    repaired[i] = repaired[last_good_idx]
                    action = f"copied_from_frame_{last_good_idx}"
                elif repair_strategy == "unsharp_mask":
                    repaired[i] = self._unsharp(frame)
                    action = "unsharp_mask"
                else:
                    action = "skipped_no_prior_good_frame"
            report.append({
                "frame": int(i),
                "blur_score": round(blur, 2),
                "eye_brightness": round(eye_lum, 4),
                "blur_ok": bool(blur_ok),
                "brightness_ok": bool(lum_ok),
                "verdict": "ok" if ok else "bad",
                "action": action,
            })

        ratio = float(good_count) / float(max(1, B))
        report_json = json.dumps({
            "good_frame_ratio": round(ratio, 4),
            "blur_threshold": float(blur_threshold),
            "min_eye_brightness": float(min_eye_brightness),
            "frames": report,
        })
        return (torch.from_numpy(repaired.astype(np.float32)), ratio, report_json)


# ====================================================================
# Standalone Depth + Pose + Canny composer
# ====================================================================
class DepthPoseCannyCombinedV2:
    DESCRIPTION = (
        "Self-contained ControlNet preprocessor producing depth, pose, canny, "
        "normal, layout-combined preview, AND a weighted blended map.\n\n"
        "DEPTH backends (set via `depth_backend`):\n"
        "  - auto       : prefer external_depth_map -> any wired loader -> built_in_midas\n"
        "  - external   : require external_depth_map IMAGE input\n"
        "  - built_in_midas : MiDaS small via torch.hub (downloads ~80MB to torch hub cache, no extra node pack needed)\n"
        "  - damodel_v2     : kijai/ComfyUI-DepthAnythingV2 (models/depthanything/)\n"
        "  - da3            : PozzettiAndrea/ComfyUI-DepthAnythingV3 (models/depthanything3/) - delegates to V3 pack\n"
        "  - depthcrafter   : akatz-ai/ComfyUI-DepthCrafter-Nodes (models/depthcrafter/)\n"
        "  - depth_pro      : spacepxl/ComfyUI-Depth-Pro (models/depth/ml-depth-pro/)\n\n"
        "POSE source priority: external_pose_map > posemodel.\n\n"
        "NORMAL map: Sobel-from-depth (Lambertian-style RGB). No extra model.\n\n"
        "BLEND modes (research-backed, Wikipedia/W3C Compositing 1.0):\n"
        "  - none           : returns the depth_map\n"
        "  - weighted_avg   : per-channel sum normalised by total weight (perceptually balanced)\n"
        "  - screen         : 1 - prod(1 - layer_i*w_i)  (avoids highlight clipping, good for stacking depth+canny gradients)\n"
        "  - linear_dodge   : min(1, sum(layer_i*w_i))  (additive; sharpens edges; preferred for pose+canny per Fooocus/SDXL controlnet community)\n"
        "  - max            : per-pixel maximum across weighted layers (preserves strongest cue per pixel)\n"
        "  - multiply       : prod(layer_i^w_i)  (darkening; emphasises overlap)\n"
        "  - overlay        : combined multiply/screen S-curve on weighted_avg base\n"
        "  - channel_split  : R=depth, G=canny, B=pose (Fun-Control / IP-Adapter style multi-condition packing)\n\n"
        "OUTPUTS: depth_map, pose_map, canny_map, normal_map, combined_map (layout), blended_map (per blend_mode)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":             ("IMAGE", {"tooltip": "Input video frames (B,H,W,3) float32 [0,1]."}),
                "width":              ("INT", {"default": 832, "min": 64, "max": 4096, "tooltip": "Output canvas width."}),
                "height":             ("INT", {"default": 480, "min": 64, "max": 4096, "tooltip": "Output canvas height."}),
                "enable_depth":       ("BOOLEAN", {"default": True, "tooltip": "Run the depth pass. Requires at least ONE depth source wired."}),
                "enable_pose":        ("BOOLEAN", {"default": True, "tooltip": "Run the pose pass."}),
                "enable_canny":       ("BOOLEAN", {"default": True, "tooltip": "Run the canny pass."}),
                "canny_threshold1":   ("INT", {"default": 100, "min": 0, "max": 500, "tooltip": "Canny lower hysteresis threshold."}),
                "canny_threshold2":   ("INT", {"default": 200, "min": 0, "max": 500, "tooltip": "Canny upper hysteresis threshold."}),
                "canny_aperture":     ([3, 5, 7], {"default": 3, "tooltip": "Sobel aperture for Canny (odd: 3/5/7)."}),
                "depth_colorize":     ("BOOLEAN", {"default": False, "tooltip": "If true, colorize grayscale depth with INFERNO colormap. Skipped when external_depth_map is already RGB."}),
                "depth_invert":       ("BOOLEAN", {"default": False, "tooltip": "Invert depth (1 - depth). Use when source produces 'far = bright' but you want 'near = bright' (typical ControlNet expectation)."}),
                "pose_detection_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO confidence threshold (only used when posemodel is wired)."}),
                "pose_draw_threshold":      ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-keypoint score threshold for drawing the skeleton."}),
                "combined_layout":    (["horizontal_3", "vertical_3", "grid_2x2", "depth_only", "pose_only", "canny_only"], {"default": "horizontal_3", "tooltip": "Layout for the combined output. grid_2x2 = depth | pose // canny | original."}),
                # ---- Task 2: self-contained additions (appended at end so saved workflows keep their positional values) ----
                "depth_backend":      (["auto", "external", "built_in_midas", "damodel_v2", "da3", "depthcrafter", "depth_pro"], {"default": "auto", "tooltip": "Which depth backend to use. 'auto' tries: external_depth_map -> any wired loader -> built_in_midas. 'built_in_midas' makes the node fully self-contained (downloads MiDaS small via torch.hub on first use, ~80MB)."}),
                "enable_normal":      ("BOOLEAN", {"default": True, "tooltip": "Compute Sobel-from-depth NORMAL map. No model required (uses depth pass output)."}),
                "normal_strength":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "Scales the Sobel gradients before normalisation. Higher = stronger normal contrast."}),
                "blend_mode":         (["none", "weighted_avg", "screen", "linear_dodge", "max", "multiply", "overlay", "channel_split"], {"default": "weighted_avg", "tooltip": "How to combine depth+pose+canny+normal into blended_map. linear_dodge=additive (sharp), screen=highlight-safe, channel_split=Fun-Control (R=depth/G=canny/B=pose)."}),
                "depth_weight":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of depth in blended_map."}),
                "pose_weight":        ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of pose in blended_map."}),
                "canny_weight":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of canny in blended_map."}),
                "normal_weight":      ("FLOAT", {"default": 0.5, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Weight of normal map in blended_map."}),
            },
            "optional": {
                "external_depth_map":  ("IMAGE", {"tooltip": "Pre-computed depth IMAGE batch from ANY upstream node. Highest priority."}),
                "damodel_v2":          ("DAMODEL", {"tooltip": "DepthAnything V2 model bundle from kijai/ComfyUI-DepthAnythingV2 (DownloadAndLoadDepthAnythingV2Model). Models: ComfyUI/models/depthanything/."}),
                "da3_model":           ("DA3MODEL", {"tooltip": "DepthAnything V3 config bundle from PozzettiAndrea/ComfyUI-DepthAnythingV3. Use the V3 pack's Inference node and feed its IMAGE output into external_depth_map. Models: ComfyUI/models/depthanything3/."}),
                "depthcrafter_model":  ("DEPTHCRAFTER_MODEL", {"tooltip": "DepthCrafter bundle from akatz-ai/ComfyUI-DepthCrafter-Nodes. Temporally consistent video depth. Models: ComfyUI/models/depthcrafter/."}),
                "depth_pro_model":     ("DEPTH_PRO_MODEL", {"tooltip": "Depth-Pro bundle from spacepxl/ComfyUI-Depth-Pro. Metric depth. Models: ComfyUI/models/depth/ml-depth-pro/."}),
                "posemodel":           ("POSEMODEL", {"tooltip": "From ONNX Detection Model Loader (V2) or animal-pose loader. Used if enable_pose=True AND no external_pose_map wired."}),
                "external_pose_map":   ("IMAGE", {"tooltip": "Pre-rendered pose map from any upstream node (e.g. Fannovel16/comfyui_controlnet_aux DWPose / OpenPose / AnimalPose). Highest priority for pose."}),
                "depthcrafter_steps":      ("INT", {"default": 5, "min": 1, "max": 100, "tooltip": "DepthCrafter only: diffusion inference steps."}),
                "depthcrafter_guidance":   ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "tooltip": "DepthCrafter only: classifier-free guidance."}),
                "depthcrafter_window":     ("INT", {"default": 110, "min": 1, "max": 200, "tooltip": "DepthCrafter only: temporal window size."}),
                "depthcrafter_overlap":    ("INT", {"default": 25, "min": 0, "max": 100, "tooltip": "DepthCrafter only: window overlap."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("depth_map", "pose_map", "canny_map", "normal_map", "combined_map", "blended_map")
    OUTPUT_TOOLTIPS = (
        "Per-frame depth IMAGE batch (3-channel, height x width).",
        "Per-frame pose IMAGE batch (3-channel, on black canvas).",
        "Per-frame canny edge IMAGE batch (3-channel grayscale).",
        "Per-frame normal map (RGB-encoded surface normals from Sobel-of-depth).",
        "Side-by-side combined preview per `combined_layout`.",
        "Weighted blend of {depth, pose, canny, normal} per `blend_mode` and per-channel weights.",
    )
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess_V2"

    # ---------- helpers ----------
    @staticmethod
    def _to_np(images):
        if hasattr(images, "detach"):
            return images.detach().cpu().numpy().astype(np.float32)
        return np.asarray(images, dtype=np.float32)

    @staticmethod
    def _resize_batch(arr, target_w, target_h):
        # arr: (B,H,W,3) float32 [0,1]
        if arr.shape[1] == target_h and arr.shape[2] == target_w:
            return arr
        out = np.zeros((arr.shape[0], target_h, target_w, arr.shape[3]), dtype=np.float32)
        for i in range(arr.shape[0]):
            out[i] = cv2.resize(arr[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return out

    def _depth_pass(self, images_np, target_w, target_h, colorize, invert,
                    external_depth_map, damodel_v2, da3_model,
                    depthcrafter_model, depth_pro_model,
                    dc_steps, dc_guidance, dc_window, dc_overlap,
                    depth_backend="auto"):
        """Run depth using the selected backend.

        depth_backend:
          - auto       : external_depth_map -> any wired loader -> built_in_midas (fully self-contained fallback)
          - external   : requires external_depth_map
          - built_in_midas : torch.hub MiDaS small (downloads on first use)
          - damodel_v2 / da3 / depthcrafter / depth_pro : require the matching loader wired
        Always returns (B, target_h, target_w, 3) float32 in [0, 1].
        """
        depth_2d = None
        depth_rgb = None

        backend = (depth_backend or "auto").lower()

        def _from_external():
            ext = self._to_np(external_depth_map)
            if ext.ndim != 4 or ext.shape[-1] != 3:
                raise ValueError(
                    f"external_depth_map must be IMAGE (B,H,W,3); got {ext.shape}"
                )
            return self._resize_batch(ext, target_w, target_h)

        if backend == "external":
            if external_depth_map is None:
                raise RuntimeError("depth_backend='external' but external_depth_map is not wired.")
            depth_rgb = _from_external()
        elif backend == "built_in_midas":
            depth_2d = self._infer_midas_small(images_np)
        elif backend == "damodel_v2":
            if damodel_v2 is None:
                raise RuntimeError("depth_backend='damodel_v2' but damodel_v2 is not wired.")
            depth_2d = self._infer_damodel_v2(damodel_v2, images_np)
        elif backend == "da3":
            if da3_model is None:
                raise RuntimeError("depth_backend='da3' but da3_model is not wired.")
            depth_2d = self._infer_da3model(da3_model, images_np)
        elif backend == "depthcrafter":
            if depthcrafter_model is None:
                raise RuntimeError("depth_backend='depthcrafter' but depthcrafter_model is not wired.")
            depth_2d = self._infer_depthcrafter(
                depthcrafter_model, images_np,
                int(dc_steps), float(dc_guidance), int(dc_window), int(dc_overlap),
            )
        elif backend == "depth_pro":
            if depth_pro_model is None:
                raise RuntimeError("depth_backend='depth_pro' but depth_pro_model is not wired.")
            depth_2d = self._infer_depth_pro(depth_pro_model, images_np)
        else:
            # auto: external -> any loader -> built_in_midas
            if external_depth_map is not None:
                depth_rgb = _from_external()
            elif damodel_v2 is not None:
                depth_2d = self._infer_damodel_v2(damodel_v2, images_np)
            elif da3_model is not None:
                depth_2d = self._infer_da3model(da3_model, images_np)
            elif depthcrafter_model is not None:
                depth_2d = self._infer_depthcrafter(
                    depthcrafter_model, images_np,
                    int(dc_steps), float(dc_guidance), int(dc_window), int(dc_overlap),
                )
            elif depth_pro_model is not None:
                depth_2d = self._infer_depth_pro(depth_pro_model, images_np)
            else:
                # Self-contained fallback
                depth_2d = self._infer_midas_small(images_np)

        if depth_2d is not None:
            if depth_2d.shape[1:] != (target_h, target_w):
                resized = np.zeros((depth_2d.shape[0], target_h, target_w), dtype=np.float32)
                for i in range(depth_2d.shape[0]):
                    resized[i] = cv2.resize(
                        depth_2d[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                depth_2d = resized
            if invert:
                depth_2d = 1.0 - depth_2d
            if colorize:
                out = np.zeros((depth_2d.shape[0], target_h, target_w, 3), dtype=np.float32)
                for i in range(depth_2d.shape[0]):
                    u8 = (np.clip(depth_2d[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                    col = cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)
                    col_rgb = cv2.cvtColor(col, cv2.COLOR_BGR2RGB)
                    out[i] = col_rgb.astype(np.float32) / 255.0
                return out
            return np.repeat(depth_2d[..., None], 3, axis=-1).astype(np.float32)

        if invert:
            depth_rgb = 1.0 - depth_rgb
        return depth_rgb.astype(np.float32)

    # ---------- per-backend inference adapters ----------
    @staticmethod
    def _normalize_per_frame(depth_np):
        """Per-frame min-max normalize (B,H,W) -> [0,1]."""
        out = np.zeros_like(depth_np, dtype=np.float32)
        for i in range(depth_np.shape[0]):
            f = depth_np[i].astype(np.float32)
            fmin, fmax = float(f.min()), float(f.max())
            if fmax - fmin > 1e-6:
                out[i] = (f - fmin) / (fmax - fmin)
        return out

    def _infer_damodel_v2(self, damodel, images_np):
        """Mirror kijai/ComfyUI-DepthAnythingV2 inference loop."""
        import torch.nn.functional as F
        from torchvision.transforms import Normalize
        try:
            import comfy.model_management as mm
        except ImportError:
            mm = None

        device = mm.get_torch_device() if mm else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        offload_device = mm.unet_offload_device() if mm else torch.device("cpu")
        model = damodel["model"]
        dtype = damodel.get("dtype", torch.float32)
        is_metric = damodel.get("is_metric", False)

        images_t = torch.from_numpy(images_np).float()
        B, H, W, _ = images_t.shape
        images_t = images_t.permute(0, 3, 1, 2)
        new_W = W - (W % 14)
        new_H = H - (H % 14)
        if new_W != W or new_H != H:
            images_t = F.interpolate(images_t, size=(new_H, new_W), mode="bilinear")
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_t = normalize(images_t)

        model.to(device)
        autocast_ok = (dtype != torch.float32) and (device.type == "cuda")
        out = []
        with torch.inference_mode():
            if autocast_ok:
                with torch.autocast("cuda", dtype=dtype):
                    for img in images_t:
                        d = model(img.unsqueeze(0).to(device))
                        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                        out.append(d.detach().float().cpu())
            else:
                for img in images_t:
                    d = model(img.unsqueeze(0).to(device))
                    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    out.append(d.detach().float().cpu())
        try:
            model.to(offload_device)
            if mm:
                mm.soft_empty_cache()
        except Exception:
            pass
        depth = torch.cat(out, dim=0).numpy()
        if depth.ndim == 4:
            depth = depth.squeeze(1)
        if is_metric:
            depth = 1.0 - depth
        if depth.shape[1:] != (H, W):
            resized = np.zeros((B, H, W), dtype=np.float32)
            for i in range(B):
                resized[i] = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
            depth = resized
        return depth.astype(np.float32)

    def _infer_da3model(self, da3_config, images_np):
        """DA3 inference is tightly coupled to V3 internals; route via V3 pack."""
        raise RuntimeError(
            "DA3MODEL is a JSON config bundle. Please run the V3 pack's own "
            "inference node (DepthAnythingV3 Inference) on the V3 pack, then "
            "feed its IMAGE output into `external_depth_map` here. We do not "
            "duplicate V3 inference internals (they are version-dependent)."
        )

    def _infer_depthcrafter(self, dc_model, images_np, steps, guidance, window, overlap):
        """Mirror akatz-ai/ComfyUI-DepthCrafter-Nodes inference logic."""
        device = dc_model.get("device") if isinstance(dc_model, dict) else None
        pipe = dc_model["pipe"] if isinstance(dc_model, dict) else dc_model
        if device is None:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        images_t = torch.from_numpy(images_np).float()
        B, H, W, _ = images_t.shape
        new_W = max(64, (round(W / 64) * 64) or 64)
        new_H = max(64, (round(H / 64) * 64) or 64)
        if new_W != W or new_H != H:
            x = images_t.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(
                x, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
            images_t = x.permute(0, 2, 3, 1)
        x = images_t.permute(0, 3, 1, 2).to(device=device, dtype=torch.float16)
        x = torch.clamp(x, 0.0, 1.0)
        with torch.inference_mode():
            result = pipe(
                x,
                height=new_H,
                width=new_W,
                output_type="pt",
                guidance_scale=float(guidance),
                num_inference_steps=int(steps),
                window_size=int(window),
                overlap=int(overlap),
                track_time=False,
            )
        res = result.frames[0]
        depth = res.detach().float().cpu().numpy()
        if depth.ndim == 4:
            depth = depth.squeeze(-1) if depth.shape[-1] == 1 else depth.mean(axis=-1)
        depth = self._normalize_per_frame(depth)
        if depth.shape[1:] != (H, W):
            resized = np.zeros((depth.shape[0], H, W), dtype=np.float32)
            for i in range(depth.shape[0]):
                resized[i] = cv2.resize(depth[i], (W, H), interpolation=cv2.INTER_LINEAR)
            depth = resized
        return depth.astype(np.float32)

    def _infer_depth_pro(self, dp_model, images_np):
        """Mirror spacepxl/ComfyUI-Depth-Pro inference (relative depth)."""
        from torchvision.transforms import Normalize
        model = dp_model["model"]
        device = dp_model.get("device")
        dtype = dp_model.get("dtype", torch.float32)
        if device is None:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rgb = torch.from_numpy(images_np).float().movedim(-1, 1)
        transform = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        depth_list = []
        with torch.inference_mode():
            for i in range(rgb.size(0)):
                rgb_i = rgb[i].unsqueeze(0).to(device, dtype=dtype)
                rgb_i = transform(rgb_i)
                pred = model.infer(rgb_i, f_px=None)
                d = pred["depth"].detach().float().cpu().numpy()
                depth_list.append(d)
        depth = np.stack(depth_list, axis=0)
        while depth.ndim > 3:
            depth = depth.squeeze(1) if depth.shape[1] == 1 else depth.mean(axis=1)
        # Depth-Pro returns metric depth (meters). Convert to relative [0,1].
        depth = 1.0 / (1.0 + depth)
        depth = self._normalize_per_frame(depth)
        return depth.astype(np.float32)

    # ---------- built-in MiDaS (self-contained depth) ----------
    _midas_cache = {"model": None, "transform": None, "device": None}

    @classmethod
    def _infer_midas_small(cls, images_np):
        """MiDaS small via torch.hub. Self-contained depth fallback.

        First call downloads ~80MB to torch hub cache (HOME/.cache/torch/hub/).
        Subsequent calls reuse the cached model. Returns (B,H,W) float32 [0,1].
        """
        try:
            try:
                import comfy.model_management as mm
                device = mm.get_torch_device()
            except Exception:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if cls._midas_cache["model"] is None or cls._midas_cache["device"] != device:
                midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
                midas.to(device).eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
                cls._midas_cache["model"] = midas
                cls._midas_cache["transform"] = transforms.small_transform
                cls._midas_cache["device"] = device

            midas = cls._midas_cache["model"]
            transform = cls._midas_cache["transform"]

            B, H, W, _ = images_np.shape
            out = np.zeros((B, H, W), dtype=np.float32)
            with torch.inference_mode():
                for i in range(B):
                    u8 = (np.clip(images_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
                    inp = transform(u8).to(device)
                    pred = midas(inp)
                    pred = torch.nn.functional.interpolate(
                        pred.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False
                    ).squeeze(1)
                    d = pred[0].detach().float().cpu().numpy()
                    dmin, dmax = float(d.min()), float(d.max())
                    if dmax - dmin > 1e-6:
                        d = (d - dmin) / (dmax - dmin)
                    else:
                        d = np.zeros_like(d)
                    out[i] = d.astype(np.float32)
            return out
        except Exception as e:
            raise RuntimeError(
                "DepthPoseCannyCombinedV2 built_in_midas backend failed. "
                "Cause: " + str(e) + "\n"
                "Possible fixes: (a) ensure internet is reachable for first-time torch.hub download, "
                "(b) install timm via pip (MiDaS small needs it), "
                "(c) wire an external_depth_map instead and set depth_backend='external'."
            )

    # ---------- normal map (Sobel-from-depth, no extra model) ----------
    @staticmethod
    def _normal_from_depth(depth_2d, strength=1.0):
        """Compute per-frame RGB normal map from grayscale depth.

        depth_2d: (B,H,W) float32 [0,1]. Returns (B,H,W,3) float32 [0,1].
        Encoding: R = (nx+1)/2, G = (ny+1)/2, B = (nz+1)/2 (standard tangent-space).
        """
        B, H, W = depth_2d.shape
        out = np.zeros((B, H, W, 3), dtype=np.float32)
        for i in range(B):
            d = depth_2d[i].astype(np.float32)
            # Scharr is more accurate than Sobel for small kernels
            gx = cv2.Scharr(d, cv2.CV_32F, 1, 0) * float(strength)
            gy = cv2.Scharr(d, cv2.CV_32F, 0, 1) * float(strength)
            # Normal vector: (-dz/dx, -dz/dy, 1) then normalise
            nx = -gx
            ny = -gy
            nz = np.ones_like(nx)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
            nx /= norm
            ny /= norm
            nz /= norm
            # Encode to [0,1]
            out[i, ..., 0] = np.clip((nx + 1.0) * 0.5, 0.0, 1.0)
            out[i, ..., 1] = np.clip((ny + 1.0) * 0.5, 0.0, 1.0)
            out[i, ..., 2] = np.clip((nz + 1.0) * 0.5, 0.0, 1.0)
        return out

    def _normal_pass(self, depth_out, target_w, target_h, strength):
        """Convert depth_out (B,H,W,3) RGB depth -> (B,H,W,3) normal map."""
        # depth_out is RGB but for normals we need a scalar field — use channel mean.
        depth_2d = depth_out.mean(axis=-1).astype(np.float32)
        # Light blur to smooth normals (depth is noisy at edges)
        for i in range(depth_2d.shape[0]):
            depth_2d[i] = cv2.GaussianBlur(depth_2d[i], (0, 0), sigmaX=1.2)
        n = self._normal_from_depth(depth_2d, strength=float(strength))
        if n.shape[1] != target_h or n.shape[2] != target_w:
            out = np.zeros((n.shape[0], target_h, target_w, 3), dtype=np.float32)
            for i in range(n.shape[0]):
                out[i] = cv2.resize(n[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return out
        return n

    # ---------- blend (research-backed, Wikipedia/W3C Compositing 1.0) ----------
    @staticmethod
    def _blend_pass(depth, pose, canny, normal, mode, w_depth, w_pose, w_canny, w_normal):
        """Combine per-channel maps. All inputs (B,H,W,3) float32 [0,1]."""
        d = (depth * float(w_depth))
        p = (pose * float(w_pose))
        c = (canny * float(w_canny))
        n = (normal * float(w_normal))
        eps = 1e-6
        m = (mode or "weighted_avg").lower()

        if m == "none":
            return depth.astype(np.float32)
        if m == "channel_split":
            # Fun-Control style: R=depth(gray), G=canny(gray), B=pose(gray)
            out = np.zeros_like(depth, dtype=np.float32)
            out[..., 0] = np.clip(depth.mean(axis=-1) * float(w_depth), 0.0, 1.0)
            out[..., 1] = np.clip(canny.mean(axis=-1) * float(w_canny), 0.0, 1.0)
            out[..., 2] = np.clip(pose.mean(axis=-1) * float(w_pose), 0.0, 1.0)
            return out
        if m == "linear_dodge":
            return np.clip(d + p + c + n, 0.0, 1.0).astype(np.float32)
        if m == "max":
            return np.maximum.reduce([
                np.clip(d, 0.0, 1.0),
                np.clip(p, 0.0, 1.0),
                np.clip(c, 0.0, 1.0),
                np.clip(n, 0.0, 1.0),
            ]).astype(np.float32)
        if m == "screen":
            # 1 - prod(1 - x_i)
            inv = (1.0 - np.clip(d, 0.0, 1.0)) \
                * (1.0 - np.clip(p, 0.0, 1.0)) \
                * (1.0 - np.clip(c, 0.0, 1.0)) \
                * (1.0 - np.clip(n, 0.0, 1.0))
            return np.clip(1.0 - inv, 0.0, 1.0).astype(np.float32)
        if m == "multiply":
            # Use weights as exponents: x^w
            r = (np.clip(depth, eps, 1.0) ** float(w_depth)) \
              * (np.clip(pose,  eps, 1.0) ** float(w_pose)) \
              * (np.clip(canny, eps, 1.0) ** float(w_canny)) \
              * (np.clip(normal,eps, 1.0) ** float(w_normal))
            return np.clip(r, 0.0, 1.0).astype(np.float32)
        if m == "overlay":
            # Base = weighted_avg of (depth, pose, canny); top = normal
            base_w = max(eps, float(w_depth) + float(w_pose) + float(w_canny))
            base = np.clip((d + p + c) / base_w, 0.0, 1.0)
            top = np.clip(normal, 0.0, 1.0)
            lo = 2.0 * base * top
            hi = 1.0 - 2.0 * (1.0 - base) * (1.0 - top)
            r = np.where(base < 0.5, lo, hi)
            # Mix in normal weight as opacity
            alpha = float(np.clip(w_normal, 0.0, 1.0))
            r = (1.0 - alpha) * base + alpha * r
            return np.clip(r, 0.0, 1.0).astype(np.float32)
        # default: weighted_avg
        total = max(eps, float(w_depth) + float(w_pose) + float(w_canny) + float(w_normal))
        return np.clip((d + p + c + n) / total, 0.0, 1.0).astype(np.float32)

    def _canny_pass(self, images_np, target_w, target_h, t1, t2, aperture):
        B = images_np.shape[0]
        out = np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        for i in range(B):
            u8 = (np.clip(images_np[i], 0.0, 1.0) * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, int(t1), int(t2), apertureSize=int(aperture))
            edges_rgb = np.repeat(edges[..., None], 3, axis=-1)
            if edges_rgb.shape[:2] != (target_h, target_w):
                edges_rgb = cv2.resize(
                    edges_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST
                )
            out[i] = edges_rgb.astype(np.float32) / 255.0
        return out

    def _pose_pass(self, images_np, posemodel, target_w, target_h,
                   detection_threshold, draw_threshold, external_pose_map):
        B, H, W, _ = images_np.shape
        # External pose map takes priority.
        if external_pose_map is not None:
            ext_np = self._to_np(external_pose_map)
            return self._resize_batch(ext_np, target_w, target_h)

        if posemodel is None:
            # Return a black canvas — caller already validated enable_pose.
            return np.zeros((B, target_h, target_w, 3), dtype=np.float32)

        # Render pose using YOLO + ViTPose, then draw onto target canvas.
        detector = posemodel["yolo"]
        pose_model = posemodel["vitpose"]
        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = float(detection_threshold)

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution = (256, 192)
        rescale = 1.25
        shape = np.array([H, W])[None]

        pose_canvases = []
        for img in _IC.track(
            images_np, B, "DepthPoseCannyCombined: pose render"
        ):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape
            )[0]
            if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
                bbox = detections[0]["bbox"]
            else:
                bbox = None
            if bbox is None or len(bbox) < 5 or bbox[4] <= 0:
                bbox_use = np.array([0, 0, W, H, 1.0], dtype=np.float32)
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
            img_norm = (img_crop - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
            kp2ds = pose_model(
                img_norm[None], np.array(center)[None], np.array(scale)[None]
            )

            metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
            meta = metas[0]
            aa = AAPoseMeta.from_humanapi_meta(meta)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            try:
                draw_aapose_by_meta_new(
                    canvas, aa,
                    body_stick_width=-1, hand_stick_width=-1,
                    draw_head=True,
                    pose_draw_threshold=float(draw_threshold),
                )
            except TypeError:
                # Older signature without pose_draw_threshold
                draw_aapose_by_meta_new(
                    canvas, aa,
                    body_stick_width=-1, hand_stick_width=-1,
                    draw_head=True,
                )
            pose_canvases.append(canvas.astype(np.float32) / 255.0)

        try:
            detector.cleanup()
        except Exception:
            pass
        try:
            pose_model.cleanup()
        except Exception:
            pass

        return np.stack(pose_canvases, 0)

    def _compose(self, depth, pose, canny, original, layout, target_w, target_h):
        B = original.shape[0]
        zeros = np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        d = depth if depth is not None else zeros
        p = pose if pose is not None else zeros
        c = canny if canny is not None else zeros

        if layout == "depth_only":
            return d
        if layout == "pose_only":
            return p
        if layout == "canny_only":
            return c
        if layout == "horizontal_3":
            return np.concatenate([d, p, c], axis=2)
        if layout == "vertical_3":
            return np.concatenate([d, p, c], axis=1)
        if layout == "grid_2x2":
            top = np.concatenate([d, p], axis=2)
            bot = np.concatenate([c, original], axis=2)
            return np.concatenate([top, bot], axis=1)
        return d

    def process(
        self, images, width, height,
        enable_depth, enable_pose, enable_canny,
        canny_threshold1, canny_threshold2, canny_aperture,
        depth_colorize, depth_invert,
        pose_detection_threshold, pose_draw_threshold,
        combined_layout,
        depth_backend="auto", enable_normal=True, normal_strength=1.0,
        blend_mode="weighted_avg",
        depth_weight=1.0, pose_weight=1.0, canny_weight=1.0, normal_weight=0.5,
        external_depth_map=None,
        damodel_v2=None, da3_model=None,
        depthcrafter_model=None, depth_pro_model=None,
        posemodel=None, external_pose_map=None,
        depthcrafter_steps=5, depthcrafter_guidance=1.0,
        depthcrafter_window=110, depthcrafter_overlap=25,
    ):
        images_np = self._to_np(images)
        if images_np.ndim != 4 or images_np.shape[-1] != 3:
            raise ValueError(
                f"DepthPoseCannyCombinedV2: expected (B,H,W,3); got {images_np.shape}"
            )
        B = images_np.shape[0]
        target_w, target_h = int(width), int(height)
        original_resized = self._resize_batch(images_np, target_w, target_h)

        depth_out = (
            self._depth_pass(
                images_np, target_w, target_h,
                bool(depth_colorize), bool(depth_invert),
                external_depth_map, damodel_v2, da3_model,
                depthcrafter_model, depth_pro_model,
                depthcrafter_steps, depthcrafter_guidance,
                depthcrafter_window, depthcrafter_overlap,
                depth_backend=str(depth_backend),
            ) if enable_depth else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        pose_out = (
            self._pose_pass(
                images_np, posemodel, target_w, target_h,
                pose_detection_threshold, pose_draw_threshold, external_pose_map,
            ) if enable_pose else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        canny_out = (
            self._canny_pass(
                images_np, target_w, target_h,
                canny_threshold1, canny_threshold2, canny_aperture,
            ) if enable_canny else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )
        normal_out = (
            self._normal_pass(depth_out, target_w, target_h, float(normal_strength))
            if (enable_normal and enable_depth) else
            np.zeros((B, target_h, target_w, 3), dtype=np.float32)
        )

        combined = self._compose(
            depth_out, pose_out, canny_out, original_resized,
            combined_layout, target_w, target_h,
        )

        blended = self._blend_pass(
            depth_out, pose_out, canny_out, normal_out,
            str(blend_mode),
            float(depth_weight), float(pose_weight),
            float(canny_weight), float(normal_weight),
        )

        return (
            torch.from_numpy(depth_out.astype(np.float32)),
            torch.from_numpy(pose_out.astype(np.float32)),
            torch.from_numpy(canny_out.astype(np.float32)),
            torch.from_numpy(normal_out.astype(np.float32)),
            torch.from_numpy(combined.astype(np.float32)),
            torch.from_numpy(blended.astype(np.float32)),
        )


try:
    from .nodes_extras import (
        EXTRA_NODE_CLASS_MAPPINGS as _EXTRA_NODE_CLASS_MAPPINGS,
        EXTRA_NODE_DISPLAY_NAME_MAPPINGS as _EXTRA_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as _e:  # pragma: no cover
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "WanAnimatePreprocessV2: failed to import nodes_extras: %s", _e
    )
    _EXTRA_NODE_CLASS_MAPPINGS = {}
    _EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {}


NODE_CLASS_MAPPINGS = {
    "OnnxDetectionModelLoaderV2": OnnxDetectionModelLoaderV2,
    "PoseAndFaceDetectionV2": PoseAndFaceDetectionV2,
    "DrawViTPoseV2": DrawViTPoseV2,
    "WanAnimateFaceQualityCheckV2": WanAnimateFaceQualityCheckV2,
    "DepthPoseCannyCombinedV2": DepthPoseCannyCombinedV2,
    # Self-contained alias (Task 2): same class, friendlier name highlighting bundled MiDaS + Normal + Blend modes
    "SelfContainedControlNetPreprocessorV2": DepthPoseCannyCombinedV2,
    **_EXTRA_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnnxDetectionModelLoaderV2": "ONNX Detection Model Loader (V2)",
    "PoseAndFaceDetectionV2": "Pose and Face Detection (V2)",
    "DrawViTPoseV2": "Draw ViT Pose (V2)",
    "WanAnimateFaceQualityCheckV2": "Wan-Animate Face Quality Check (V2)",
    "DepthPoseCannyCombinedV2": "Depth + Pose + Canny Combined (V2)",
    "SelfContainedControlNetPreprocessorV2": "Self-Contained ControlNet Preprocessor (V2)",
    **_EXTRA_NODE_DISPLAY_NAME_MAPPINGS,
}
