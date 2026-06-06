"""WanPoseDetectViTPoseV2 — standalone YOLO + ViTPose detection node.

Most workflows run the full Wan-Animate preprocessor up-front, but
there are three common cases where users want to run pose detection
**without** the rest of the preprocessor:

  *   They already have edited pose metas and want to re-detect on a
      single reference frame.
  *   They want to feed an external image (e.g. a still photo or a
      different video) into the pose-editor UI and inspect / edit the
      detected skeleton.
  *   They want the convert-format node (Slice 4) to operate on a
      freshly detected skeleton instead of one produced by the main
      pipeline.

This node reuses the same ``POSEMODEL`` bundle that
``OnnxDetectionModelLoaderV2`` produces, so no extra model file needs
to be added.  Output is a POSEDATA bundle with the same shape the
editor expects (``pose_metas_original`` + ``pose_metas`` + empty
``iris_data``).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

import cv2
import numpy as np

try:
    from comfy.utils import ProgressBar                              # type: ignore
except Exception:                                                    # tests
    class ProgressBar:                                               # type: ignore
        def __init__(self, n): self.n = n
        def update_absolute(self, k): pass

from ..pose_utils.pose2d_utils import (
    load_pose_metas_from_kp2ds_seq,
    bbox_from_detector,
    crop,
)

log = logging.getLogger(__name__)

_IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_for_pose(img: np.ndarray, use_clahe: bool) -> np.ndarray:
    """CLAHE-equalise an RGB float image in [0,1] (matches nodes.py)."""
    if not use_clahe:
        return img
    img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = cl.apply(l)
    img_u8 = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return img_u8.astype(np.float32) / 255.0


def _images_to_numpy(images) -> np.ndarray:
    if hasattr(images, "detach"):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr.astype(np.float32)


class WanPoseDetectViTPoseV2:
    CATEGORY    = "WanAnimatePreprocessV2/extras"
    DESCRIPTION = (
        "Standalone YOLO + ViTPose detection. Takes an IMAGE batch and a "
        "POSEMODEL bundle (from OnnxDetectionModelLoaderV2) and emits a "
        "POSEDATA bundle compatible with the V2 editor and downstream "
        "conditioning nodes. No face-cropping / gaze pipeline — keypoints "
        "only."
    )
    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "info")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "RGB image stack to detect poses on. Float [0,1], shape (B,H,W,3).",
                }),
                "model": ("POSEMODEL", {
                    "tooltip": "ViTPose+YOLO bundle from OnnxDetectionModelLoaderV2.",
                }),
            },
            "optional": {
                "detection_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "YOLO person-detection confidence threshold.",
                }),
                "pose_threshold": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-keypoint confidence threshold; below this the keypoint's confidence is forced to 0.",
                }),
                "use_clahe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "CLAHE contrast enhancement on the 256x192 pose crop. Matches the main preprocessor default.",
                }),
                "use_blur_for_pose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply a Gaussian blur to the images before YOLO+ViTPose (anti-aliases noisy frames).",
                }),
                "blur_radius": ("INT",   {"default": 2, "min": 0, "max": 32, "step": 1}),
                "blur_sigma":  ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.1}),
                "rescale": ("FLOAT", {
                    "default": 1.25, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Bbox padding factor before crop for ViTPose. 1.25 = match the main preprocessor.",
                }),
                "fallback_to_full_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If YOLO finds no person in a frame, run ViTPose on the entire frame instead of skipping it.",
                }),
            },
        }

    def run(self, images, model,
            detection_threshold: float = 0.3,
            pose_threshold: float = 0.3,
            use_clahe: bool = True,
            use_blur_for_pose: bool = False,
            blur_radius: int = 2,
            blur_sigma: float = 1.5,
            rescale: float = 1.25,
            fallback_to_full_frame: bool = True):

        detector   = model["yolo"]
        pose_model = model["vitpose"]
        if hasattr(detector, "threshold_conf"):
            detector.threshold_conf = float(detection_threshold)

        imgs = _images_to_numpy(images)
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError("`images` must be a float IMAGE tensor (B,H,W,3) in [0,1].")
        B, H, W, _ = imgs.shape
        shape = np.array([H, W])[None]

        # Blur (optional).
        if use_blur_for_pose and blur_radius > 0:
            ksize = int(blur_radius) * 2 + 1
            imgs_b = np.stack([
                cv2.GaussianBlur(im, (ksize, ksize), float(blur_sigma)) for im in imgs
            ])
        else:
            imgs_b = imgs

        pbar = ProgressBar(B * 2)
        progress = 0

        # 1) YOLO person detection.
        bboxes: list = []
        n_no_person = 0
        for i in range(B):
            inp = cv2.resize(imgs_b[i], (640, 640)).transpose(2, 0, 1)[None]
            try:
                det = detector(inp, shape)[0]
            except Exception as e:
                log.warning("yolo failure frame %d: %s", i, e)
                det = None
            if isinstance(det, list) and det and isinstance(det[0], dict):
                bboxes.append(det[0]["bbox"])
            else:
                bboxes.append(None)
                n_no_person += 1
            progress += 1
            if progress % 10 == 0:
                pbar.update_absolute(progress)
        if hasattr(detector, "cleanup"):
            try: detector.cleanup()
            except Exception: pass

        # 2) ViTPose keypoint extraction.
        kp2ds: list[np.ndarray] = []
        input_resolution = (256, 192)
        n_fullframe = 0
        for i in range(B):
            bbox = bboxes[i]
            img  = imgs_b[i]
            valid_bbox = (
                bbox is not None and len(bbox) >= 5 and bbox[4] > 0
                and (bbox[2] - bbox[0]) >= 10 and (bbox[3] - bbox[1]) >= 10
            )
            if not valid_bbox:
                if not fallback_to_full_frame:
                    # Skip: ViTPose still needs a tensor, so synthesise an
                    # all-zero (1, 133, 3) array to keep frame indices aligned.
                    kp2ds.append(np.zeros((1, 133, 3), dtype=np.float32))
                    progress += 1
                    if progress % 10 == 0:
                        pbar.update_absolute(progress)
                    continue
                bbox_use = np.array([0, 0, img.shape[1], img.shape[0], 1.0], dtype=np.float32)
                n_fullframe += 1
            else:
                bbox_use = bbox

            center, scale = bbox_from_detector(bbox_use, input_resolution, rescale=rescale)
            img_crop = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]
            img_crop = _preprocess_for_pose(img_crop, use_clahe)
            img_norm = (img_crop - _IMG_NORM_MEAN) / _IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)
            kp = pose_model(
                img_norm[None],
                np.array(center)[None],
                np.array(scale)[None],
            )
            kp2ds.append(kp)
            progress += 1
            if progress % 10 == 0:
                pbar.update_absolute(progress)
        if hasattr(pose_model, "cleanup"):
            try: pose_model.cleanup()
            except Exception: pass

        kp2ds_arr = np.concatenate(kp2ds, 0) if kp2ds else np.zeros((0, 133, 3), dtype=np.float32)

        # 3) Confidence threshold.
        if pose_threshold > 0.0 and kp2ds_arr.size:
            kp2ds_arr[..., 2] = np.where(
                kp2ds_arr[..., 2] < pose_threshold, 0.0, kp2ds_arr[..., 2]
            )

        # 4) Bundle.
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds_arr, width=W, height=H)

        bundle = {
            "pose_metas_original": [deepcopy(m) for m in pose_metas],
            "pose_metas":          pose_metas,
            "iris_data":           [{} for _ in pose_metas],
            "width":               W,
            "height":              H,
            "n_frames":            len(pose_metas),
            "source":              "WanPoseDetectViTPoseV2",
        }

        info = (
            f"WanPoseDetectViTPoseV2: detected {len(pose_metas)} frames | "
            f"no_person={n_no_person} | full_frame_fallback={n_fullframe} | "
            f"det_thr={detection_threshold:.2f} pose_thr={pose_threshold:.2f} "
            f"clahe={'y' if use_clahe else 'n'} blur={'y' if use_blur_for_pose else 'n'}"
        )
        log.info(info)
        return (bundle, info)
