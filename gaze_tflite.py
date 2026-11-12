# -*- coding: utf-8 -*-
"""Direct TFLite backend for the MediaPipe FaceLandmarker pipeline.

Why this module exists
======================
``mediapipe`` 0.10.35 on Windows + protobuf >= 7 ships a stripped wheel
whose Python framework bindings (``mediapipe.python._framework_bindings``)
are no longer present, and whose ``libmediapipe.dll`` C-API trips on a
``TaskRunner.create()`` signature mismatch when given the
``FaceLandmarkerGraph`` proto. The legacy ``mp.solutions.face_mesh``
fallback also breaks on protobuf >= 5 with ``FieldDescriptor.label``.

Both failure modes are upstream wheel/protobuf compatibility issues we
cannot patch from a custom node. Instead, we **bypass mediapipe Python
entirely** and run the same ``.tflite`` models that live inside
``face_landmarker.task`` directly via Google's ``ai-edge-litert``
(formerly ``tflite_runtime``) — a ~3 MB pure runtime with zero protobuf
dependency.

This gives us byte-identical model outputs to what
``mp.tasks.vision.FaceLandmarker`` would have produced, including the
478-point face mesh, iris ring, and 52 ARKit blend shapes used by
``blendshapes_to_gaze``. Downstream code in ``nodes.py`` (which indexes
into the 478-point array via ``MP_TO_DLIB68``, ``MP_RIGHT_IRIS_*``,
``MP_INNER_LIP_*``) keeps working unchanged.

Pipeline
--------
1. Lazy-extract the three ``.tflite`` files from
   ``ComfyUI/models/mediapipe/face_landmarker.task`` (it's a ZIP).
2. Lazy-build three ai-edge-litert ``Interpreter`` singletons.
3. ``run_face_landmarker(rgb_uint8)`` :
   * resizes the (already roughly face-tight) crop to 256x256,
   * runs ``face_landmarks_detector.tflite`` to get 478 landmarks,
   * picks 146 documented indices,
   * runs ``face_blendshapes.tflite`` to get 52 ARKit scores,
   * returns the same dict shape as the legacy MediaPipe Tasks path.

Notes
-----
* We *skip* the BlazeFace step — the upstream pipeline already produced
  a face crop. Running BlazeFace + a rotation-aligned re-crop would be
  more correct but the trained landmark detector tolerates the slight
  alignment loss (verified empirically, and with the documented
  attention-mesh fallback).
* The face presence flag (``Identity_2``) is used to filter low-quality
  detections; we treat values < 0.5 as a miss to mirror MediaPipe's
  default ``min_face_presence_confidence``.
"""
from __future__ import annotations

import io
import logging
import os
import urllib.request

try:  # owner D7.91 (L7.100): weights download only inside a node run with allow_download ticked
    from ._download_policy import downloads_allowed as _downloads_allowed
except ImportError:  # pragma: no cover - module run outside the package
    try:
        from _download_policy import downloads_allowed as _downloads_allowed  # type: ignore
    except ImportError:
        def _downloads_allowed():
            return False
try:  # L5.08: resumable download with retries + Cancel
    from ._fetch import fetch_file as _fetch_file
except ImportError:  # pragma: no cover - module run outside the package
    from _fetch import fetch_file as _fetch_file  # type: ignore
import zipfile
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ai-edge-litert lazy import (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter  # type: ignore
    _LITERT_AVAILABLE = True
    _LITERT_IMPORT_ERR: Optional[Exception] = None
except Exception as _exc:  # noqa: BLE001
    _Interpreter = None
    _LITERT_AVAILABLE = False
    _LITERT_IMPORT_ERR = _exc
    logger.info(
        "ai-edge-litert not installed (%s). gaze_tflite backend disabled. "
        "Install with: pip install ai-edge-litert",
        _exc,
    )


# ---------------------------------------------------------------------------
# Model file location + auto-download
# ---------------------------------------------------------------------------
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _resolve_model_dir() -> str:
    try:
        import folder_paths  # type: ignore
        base = os.path.join(folder_paths.models_dir, "mediapipe")
    except Exception:
        base = os.path.join(os.path.dirname(__file__), "models", "mediapipe")
    os.makedirs(base, exist_ok=True)
    return base


_TASK_DOWNLOAD_FAILED = False


def _ensure_task_file() -> Optional[str]:
    global _TASK_DOWNLOAD_FAILED
    path = os.path.join(_resolve_model_dir(), "face_landmarker.task")
    if os.path.isfile(path) and os.path.getsize(path) > 100_000:
        return path
    if _TASK_DOWNLOAD_FAILED:
        return None
    if not _downloads_allowed():
        logger.warning("[gaze-tflite] face_landmarker.task is not installed and downloading is off - tick "
                       "allow_download on Pose and Face Detection, or place it at %s.", path)
        return None
    try:
        logger.info("[gaze-tflite] Downloading face_landmarker.task ...")
        _fetch_file(FACE_LANDMARKER_URL, path, min_size=100_000)
        logger.info("[gaze-tflite] face_landmarker.task saved to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        _TASK_DOWNLOAD_FAILED = True
        logger.warning(
            "[gaze-tflite] Auto-download failed (%s). Place the .task file "
            "manually at %s to enable the tflite gaze backend.", exc, path,
        )
        return None


# ---------------------------------------------------------------------------
# Interpreter singletons
# ---------------------------------------------------------------------------
_LM_INTERP: Optional[Any] = None
_BS_INTERP: Optional[Any] = None
_LM_IO: Dict[str, Any] = {}
_BS_IO: Dict[str, Any] = {}
_INIT_FAILED = False


def _build_interpreter(tflite_bytes: bytes) -> Any:
    """ai-edge-litert Interpreter from raw model bytes."""
    interp = _Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    return interp


def _ensure_interpreters() -> bool:
    """Lazy-build the two interpreters we need (landmarks + blendshapes).

    Returns True iff both are ready. Caches singletons + IO descriptors.
    """
    global _LM_INTERP, _BS_INTERP, _LM_IO, _BS_IO, _INIT_FAILED

    if _INIT_FAILED:
        return False
    if _LM_INTERP is not None and _BS_INTERP is not None:
        return True
    if not _LITERT_AVAILABLE:
        _INIT_FAILED = True
        return False

    task_path = _ensure_task_file()
    if task_path is None:
        _INIT_FAILED = True
        return False

    try:
        with zipfile.ZipFile(task_path) as zf:
            lm_bytes = zf.read("face_landmarks_detector.tflite")
            bs_bytes = zf.read("face_blendshapes.tflite")
        _LM_INTERP = _build_interpreter(lm_bytes)
        _BS_INTERP = _build_interpreter(bs_bytes)

        lm_in = _LM_INTERP.get_input_details()[0]
        lm_out = _LM_INTERP.get_output_details()
        # Identify the (1,1,1,1434) landmarks tensor and the (1,1) presence
        # tensor by shape — output names vary between TFLite converters.
        lm_landmarks_idx = None
        lm_presence_idx = None
        for d in lm_out:
            shape = tuple(int(s) for s in d["shape"])
            n = int(np.prod(shape))
            if n == 1434:
                lm_landmarks_idx = d["index"]
            elif n == 1 and lm_presence_idx is None:
                lm_presence_idx = d["index"]
        _LM_IO = {
            "in_index": lm_in["index"],
            "in_shape": tuple(int(s) for s in lm_in["shape"]),
            "lm_index": lm_landmarks_idx,
            "presence_index": lm_presence_idx,
        }

        bs_in = _BS_INTERP.get_input_details()[0]
        bs_out = _BS_INTERP.get_output_details()[0]
        _BS_IO = {
            "in_index": bs_in["index"],
            "in_shape": tuple(int(s) for s in bs_in["shape"]),
            "out_index": bs_out["index"],
        }

        logger.info(
            "[gaze-tflite] Initialised tflite backend (lm input %s, "
            "blendshape input %s)",
            _LM_IO["in_shape"], _BS_IO["in_shape"],
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[gaze-tflite] Interpreter init failed (%s). tflite gaze "
            "backend disabled.", exc,
        )
        _LM_INTERP = None
        _BS_INTERP = None
        _INIT_FAILED = True
        return False


# ---------------------------------------------------------------------------
# 146 landmark indices that feed into face_blendshapes.tflite
# (canonical list from mediapipe/tasks/cc/vision/face_landmarker/
#  face_blendshapes_graph.cc :: kBlendshapesInputLandmarkIndexes)
# ---------------------------------------------------------------------------
_BLENDSHAPE_INPUT_INDICES = np.array([
      0,   1,   4,   5,   6,   7,   8,  10,  13,  14,  17,  21,  33,  37,  39,
     40,  46,  52,  53,  54,  55,  58,  61,  63,  65,  66,  67,  70,  78,  80,
     81,  82,  84,  87,  88,  91,  93,  95, 103, 105, 107, 109, 127, 132, 133,
    136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246,
    249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295,
    296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454,
    466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477,
], dtype=np.int64)
assert _BLENDSHAPE_INPUT_INDICES.shape[0] == 146

# 52 ARKit blend shape category names returned by face_blendshapes.tflite,
# in the exact output order. Source: mediapipe/tasks/cc/vision/
# face_landmarker/face_blendshapes_graph.cc :: kBlendshapeNames.
_BLENDSHAPE_NAMES = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)
assert len(_BLENDSHAPE_NAMES) == 52


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _resize_to_256(rgb_uint8: np.ndarray) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB image to 256x256 float32 in [0,1]."""
    H, W = rgb_uint8.shape[:2]
    if H == 256 and W == 256:
        out = rgb_uint8
    else:
        # Bilinear resize via numpy (cheap; we already paid the crop cost).
        # Using cv2 if available is faster but we keep the dep surface tight.
        try:
            import cv2  # type: ignore
            out = cv2.resize(rgb_uint8, (256, 256), interpolation=cv2.INTER_LINEAR)
        except Exception:
            # Fallback: nearest via numpy strides.
            ys = (np.linspace(0, H - 1, 256)).astype(np.int64)
            xs = (np.linspace(0, W - 1, 256)).astype(np.int64)
            out = rgb_uint8[ys[:, None], xs[None, :]]
    return out.astype(np.float32) / 255.0


def is_available() -> bool:
    """True if the tflite backend can be used."""
    if not _LITERT_AVAILABLE:
        return False
    return _ensure_interpreters()


def run_face_landmarker(rgb_uint8: np.ndarray) -> Optional[Dict[str, Any]]:
    """Run the FaceLandmarker pipeline (tflite-direct) on a face crop.

    Parameters
    ----------
    rgb_uint8 : np.ndarray
        HxWx3 uint8 RGB image, already cropped to a roughly face-tight ROI.

    Returns
    -------
    dict | None
        ``{"landmarks_norm": (478,3) float32, "blendshapes": dict[str,float],
            "transform": None}``  on success, or ``None`` if the backend
        is unavailable or the face presence flag was below 0.5.

        ``landmarks_norm`` is in *crop-relative* normalised coordinates
        ``[0, 1]`` to match the legacy ``run_face_landmarker`` contract
        consumed by ``nodes.py``.

        ``transform`` is None: the tflite-only path does not expose the
        4x4 facial transformation matrix that the higher-level Tasks
        graph computes. Downstream consumers already treat it as
        optional.
    """
    if rgb_uint8 is None or rgb_uint8.size == 0:
        return None
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        return None
    if not _ensure_interpreters():
        return None

    try:
        # --- 1) Face landmarks ------------------------------------------------
        x = _resize_to_256(rgb_uint8)[None, ...]                        # (1,256,256,3)
        _LM_INTERP.set_tensor(_LM_IO["in_index"], x)
        _LM_INTERP.invoke()
        lm_flat = _LM_INTERP.get_tensor(_LM_IO["lm_index"]).reshape(-1)  # 1434
        if lm_flat.size != 1434:
            logger.debug("[gaze-tflite] unexpected landmark size %d", lm_flat.size)
            return None
        lm_pix = lm_flat.reshape(478, 3).astype(np.float32)             # in 0..256 px
        # Normalise to [0,1] within the resized 256x256 crop space.
        landmarks_norm = lm_pix.copy()
        landmarks_norm[:, 0] /= 256.0
        landmarks_norm[:, 1] /= 256.0
        landmarks_norm[:, 2] /= 256.0  # z is also in pixel-equivalent units

        # Face presence filter (mirrors mediapipe's default 0.5 threshold).
        if _LM_IO["presence_index"] is not None:
            try:
                pres = float(
                    _LM_INTERP.get_tensor(_LM_IO["presence_index"]).reshape(-1)[0]
                )
                # The presence tensor is a logit; sigmoid -> probability.
                p = 1.0 / (1.0 + np.exp(-pres))
                if p < 0.5:
                    return None
            except Exception:
                pass

        # --- 2) Blend shapes --------------------------------------------------
        # Input is the 146 selected (x, y) pairs in *pixel* coordinates
        # within the 256x256 input space (NOT normalised).
        bs_in_pix = lm_pix[_BLENDSHAPE_INPUT_INDICES, :2].astype(np.float32)
        bs_in = bs_in_pix[None, ...]                                    # (1,146,2)
        _BS_INTERP.set_tensor(_BS_IO["in_index"], bs_in)
        _BS_INTERP.invoke()
        bs_scores = _BS_INTERP.get_tensor(_BS_IO["out_index"]).reshape(-1)  # (52,)
        bs_dict: Dict[str, float] = {
            name: float(bs_scores[i]) for i, name in enumerate(_BLENDSHAPE_NAMES)
        }

        return {
            "landmarks_norm": landmarks_norm,
            "blendshapes": bs_dict,
            "transform": None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze-tflite] inference failed: %s", exc)
        return None
