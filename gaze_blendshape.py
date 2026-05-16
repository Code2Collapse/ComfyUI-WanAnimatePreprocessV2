# -*- coding: utf-8 -*-
"""Production gaze tracking via MediaPipe Tasks API ``FaceLandmarker``.

This module replaces the legacy ``mp.solutions.face_mesh`` glue with the
modern ``mediapipe.tasks.python.vision.FaceLandmarker`` API, which exposes
the 52 ARKit-compatible blend shapes alongside the 478-point face mesh.

Why blend shapes for gaze
=========================
The previous integration computed gaze as the 2D unit vector from the
mean of six eyelid landmarks to the iris pixel, then normalised it.
This is fundamentally incorrect because (a) the eyelid centre is NOT the
eyeball centre — the eyeball sits ~12 mm behind the eyelid plane —
(b) the 2D offset mixes head rotation with eye rotation, and (c) the
unit-vector normalisation discards magnitude so subtle saccades and a
stable forward gaze look identical to downstream consumers.

MediaPipe's ``FaceLandmarker`` returns four pairs of eye-look blend
shapes per eye (``eyeLookIn / Out / Up / Down`` × ``Left / Right``).
These are trained on millions of frames and are head-pose-corrected by
construction (the same signal Apple ARKit / Snapchat use). Each value
is in [0, 1] where 1 indicates physiologically maximal rotation in that
direction. Combining the antagonistic pairs gives signed yaw / pitch in
[-1, 1], which we scale to radians via empirically calibrated max-rotation
constants.

Pipeline
--------
1. Lazy-load ``face_landmarker.task`` from
   ``ComfyUI/models/mediapipe/face_landmarker.task``.
   Auto-download on first run if the file is missing.
2. ``run_face_landmarker(rgb_uint8)`` returns landmarks + iris + blend
   shapes + facial transformation matrix per detected face.
3. ``blendshapes_to_gaze(blendshapes)`` converts the per-eye blend shape
   coefficients into ``(yaw_rad, pitch_rad)`` plus 2D ``(dx, dy)`` for
   backward compatibility with the legacy iris arrow drawing.
4. :class:`OneEuroFilter` smooths the yaw / pitch streams with low lag
   and aggressive jitter rejection (Casiez 2012, the de-facto industry
   standard for interactive tracking).
"""
from __future__ import annotations

import logging
import math
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tasks API import (graceful fallback when mediapipe is missing or too old)
# ---------------------------------------------------------------------------
try:
    import mediapipe as _mp  # type: ignore
    from mediapipe.tasks import python as _mp_tasks  # type: ignore
    from mediapipe.tasks.python import vision as _mp_vision  # type: ignore
    _TASKS_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001
    _mp = None
    _mp_tasks = None
    _mp_vision = None
    _TASKS_AVAILABLE = False
    logger.info(
        "MediaPipe Tasks API not available (%s). FaceLandmarker disabled.",
        _exc,
    )

# Direct-tflite backend (bypasses mediapipe Python wrapper). Used as the
# primary path on installs where the mediapipe wheel is stripped or its
# C-API trips on a TaskRunner signature mismatch (mediapipe 0.10.35 +
# protobuf >= 7), and as a graceful fallback otherwise.
try:
    from . import gaze_tflite as _gaze_tflite  # type: ignore
except Exception:  # noqa: BLE001
    try:
        import gaze_tflite as _gaze_tflite  # type: ignore
    except Exception as _tflite_exc:  # noqa: BLE001
        _gaze_tflite = None
        logger.info(
            "gaze_tflite backend unavailable (%s). Falling back to "
            "mediapipe Tasks API only.", _tflite_exc,
        )

# Public Google-hosted model URL (~3 MB, MIT-style terms).
# float16 v1, current as of May 2026.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Module-level singletons (lazy).
_LANDMARKER: Optional[Any] = None
_LANDMARKER_PATH: Optional[str] = None
_DOWNLOAD_FAILED = False


# ---------------------------------------------------------------------------
# Blend shape index map (ARKit order, same order MediaPipe returns)
# ---------------------------------------------------------------------------
# The 52-entry blend shape category list returned by FaceLandmarker is in
# a stable, documented order. We only need eye-look shapes plus blink, but
# we expose a couple more so downstream consumers can build expression
# control without re-running the model.
EYE_BLENDSHAPE_NAMES = (
    "eyeLookInLeft",   "eyeLookInRight",
    "eyeLookOutLeft",  "eyeLookOutRight",
    "eyeLookUpLeft",   "eyeLookUpRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeBlinkLeft",    "eyeBlinkRight",
    "eyeSquintLeft",   "eyeSquintRight",
    "eyeWideLeft",     "eyeWideRight",
)

# Empirically calibrated saturation values for yaw/pitch in radians.
# 30° yaw and 25° pitch is the comfortable physiological range; pushing
# blend shape == 1.0 to a higher angle would over-rotate the iris when
# transcoded for Wan 2.2 Animate.
MAX_GAZE_YAW_RAD = math.radians(30.0)
MAX_GAZE_PITCH_RAD = math.radians(25.0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_model_dir() -> str:
    """Locate ``ComfyUI/models/mediapipe`` via folder_paths.

    Falls back to a local ``models/mediapipe`` directory inside this pack
    if folder_paths cannot be imported (e.g. unit tests outside ComfyUI).
    """
    try:
        import folder_paths  # type: ignore
        base = os.path.join(folder_paths.models_dir, "mediapipe")
    except Exception:
        base = os.path.join(os.path.dirname(__file__), "models", "mediapipe")
    os.makedirs(base, exist_ok=True)
    return base


def _ensure_model_file() -> Optional[str]:
    """Return absolute path to ``face_landmarker.task`` (download if missing)."""
    global _DOWNLOAD_FAILED
    path = os.path.join(_resolve_model_dir(), "face_landmarker.task")
    if os.path.isfile(path) and os.path.getsize(path) > 100_000:
        return path
    if _DOWNLOAD_FAILED:
        return None
    try:
        logger.info(
            "[gaze] Downloading face_landmarker.task from %s ...",
            FACE_LANDMARKER_URL,
        )
        tmp = path + ".part"
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
        os.replace(tmp, path)
        logger.info("[gaze] FaceLandmarker model saved to %s", path)
        return path
    except Exception as exc:  # noqa: BLE001
        _DOWNLOAD_FAILED = True
        logger.warning(
            "[gaze] FaceLandmarker model download failed (%s). "
            "Place face_landmarker.task at %s manually to enable "
            "blend-shape gaze.",
            exc, path,
        )
        return None


def get_face_landmarker() -> Optional[Any]:
    """Lazily build (and cache) the FaceLandmarker singleton."""
    global _LANDMARKER, _LANDMARKER_PATH
    if not _TASKS_AVAILABLE:
        return None
    if _LANDMARKER is not None:
        return _LANDMARKER
    model_path = _ensure_model_file()
    if model_path is None:
        return None
    try:
        base_options = _mp_tasks.BaseOptions(model_asset_path=model_path)
        options = _mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=_mp_vision.RunningMode.IMAGE,
        )
        _LANDMARKER = _mp_vision.FaceLandmarker.create_from_options(options)
        _LANDMARKER_PATH = model_path
        logger.info("[gaze] FaceLandmarker initialised from %s", model_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[gaze] FaceLandmarker init failed (%s).", exc)
        _LANDMARKER = None
    return _LANDMARKER


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_face_landmarker(rgb_uint8: np.ndarray) -> Optional[Dict[str, Any]]:
    """Run FaceLandmarker on a single RGB uint8 image.

    Returns ``None`` if the model is unavailable or no face was detected.
    Otherwise returns a dict::

        {
            "landmarks_norm": (478, 3) float32 [x, y, z] in [0, 1] (z relative),
            "blendshapes":     {category_name: float in [0, 1], ...},
            "transform":       (4, 4) float32 facial transformation matrix
                               (face -> camera, OpenGL convention),
        }
    """
    if rgb_uint8 is None or rgb_uint8.size == 0:
        return None
    # Prefer the tflite-direct backend when available: it bypasses the
    # mediapipe Python wrapper entirely and works on stripped wheels /
    # protobuf-7 environments where the Tasks API silently breaks.
    if _gaze_tflite is not None and _gaze_tflite.is_available():
        res = _gaze_tflite.run_face_landmarker(rgb_uint8)
        if res is not None:
            return res
        # tflite backend reported no face -> let the Tasks API try too,
        # but only if it's actually usable in this env.
    landmarker = get_face_landmarker()
    if landmarker is None:
        return None
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        return None

    try:
        mp_image = _mp.Image(
            image_format=_mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_uint8)
        )
        result = landmarker.detect(mp_image)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze] FaceLandmarker.detect failed: %s", exc)
        return None

    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    landmarks = np.empty((len(lms), 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        landmarks[i, 0] = lm.x
        landmarks[i, 1] = lm.y
        landmarks[i, 2] = lm.z

    bs_dict: Dict[str, float] = {}
    if result.face_blendshapes:
        for cat in result.face_blendshapes[0]:
            bs_dict[cat.category_name] = float(cat.score)

    transform = None
    if result.facial_transformation_matrixes:
        transform = np.asarray(
            result.facial_transformation_matrixes[0], dtype=np.float32
        )

    return {
        "landmarks_norm": landmarks,
        "blendshapes": bs_dict,
        "transform": transform,
    }


# ---------------------------------------------------------------------------
# Blend shape -> gaze conversion
# ---------------------------------------------------------------------------
def blendshapes_to_gaze(
    blendshapes: Dict[str, float],
    max_yaw_rad: float = MAX_GAZE_YAW_RAD,
    max_pitch_rad: float = MAX_GAZE_PITCH_RAD,
) -> Dict[str, Any]:
    """Convert eye-look blend shapes to per-eye gaze angles.

    Sign conventions
    ----------------
    * ``yaw_rad``   positive = subject looking to *their* right
                    (= viewer's left in a non-mirrored capture).
    * ``pitch_rad`` positive = looking up.

    For the LEFT eye (subject's left, image right side):
        yaw   = (eyeLookInLeft  - eyeLookOutLeft) * MAX_YAW
                (looking "in" toward nose == looking right in subject space)
        pitch = (eyeLookUpLeft  - eyeLookDownLeft) * MAX_PITCH
    For the RIGHT eye (subject's right, image left side):
        yaw   = (eyeLookOutRight - eyeLookInRight) * MAX_YAW
                ("out" of right eye == subject's right)
        pitch = (eyeLookUpRight  - eyeLookDownRight) * MAX_PITCH
    """
    bs = blendshapes or {}

    def g(name: str) -> float:
        return float(bs.get(name, 0.0))

    yaw_l = (g("eyeLookInLeft") - g("eyeLookOutLeft")) * max_yaw_rad
    pitch_l = (g("eyeLookUpLeft") - g("eyeLookDownLeft")) * max_pitch_rad
    yaw_r = (g("eyeLookOutRight") - g("eyeLookInRight")) * max_yaw_rad
    pitch_r = (g("eyeLookUpRight") - g("eyeLookDownRight")) * max_pitch_rad

    blink_l = g("eyeBlinkLeft")
    blink_r = g("eyeBlinkRight")

    # 2D image-space arrow (kept so the legacy debug-overlay drawing path
    # works unchanged). dx > 0 == iris drifts toward subject's right
    # (image left), so we project the world-space yaw with a negative sign
    # to match the expected screen convention used by the existing
    # draw_debug_overlay() arrow code.
    def to_dxdy(yaw: float, pitch: float) -> Tuple[float, float]:
        dx = -math.sin(yaw)
        dy = -math.sin(pitch)  # image y axis points down
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return 0.0, 0.0
        return round(dx / n, 4), round(dy / n, 4)

    dxl, dyl = to_dxdy(yaw_l, pitch_l)
    dxr, dyr = to_dxdy(yaw_r, pitch_r)

    return {
        "left":  {"yaw_rad": yaw_l, "pitch_rad": pitch_l,
                   "blink": blink_l, "dx": dxl, "dy": dyl,
                   "magnitude": float(math.hypot(yaw_l, pitch_l))},
        "right": {"yaw_rad": yaw_r, "pitch_rad": pitch_r,
                   "blink": blink_r, "dx": dxr, "dy": dyr,
                   "magnitude": float(math.hypot(yaw_r, pitch_r))},
    }


# ---------------------------------------------------------------------------
# One-euro filter (Casiez et al., CHI 2012)
# ---------------------------------------------------------------------------
class OneEuroFilter:
    """Adaptive low-pass filter for noisy interactive signals.

    Reference: https://gery.casiez.net/1euro/

    Cuts jitter aggressively at low velocities and stays low-lag during
    fast motion. Recommended defaults (min_cutoff=1.0, beta=0.0) are
    overly aggressive for gaze; we use ``min_cutoff=1.7, beta=0.3`` which
    keeps small saccades visible while killing the static jitter that
    eats up Wan 2.2 Animate's eye conditioning.
    """

    def __init__(
        self,
        freq: float = 30.0,
        min_cutoff: float = 1.7,
        beta: float = 0.3,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(freq, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        x = float(x)
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t if t is not None else 0.0
            return x
        if t is not None and self._t_prev is not None:
            dt = t - self._t_prev
            if dt > 1e-6:
                self.freq = 1.0 / dt
            self._t_prev = t
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class GazeStreamSmoother:
    """Per-eye yaw/pitch one-euro filters bundled together."""

    def __init__(self, fps: float = 30.0,
                 min_cutoff: float = 1.7, beta: float = 0.3) -> None:
        kw = dict(freq=fps, min_cutoff=min_cutoff, beta=beta)
        self._filters = {
            ("left",  "yaw"):   OneEuroFilter(**kw),
            ("left",  "pitch"): OneEuroFilter(**kw),
            ("right", "yaw"):   OneEuroFilter(**kw),
            ("right", "pitch"): OneEuroFilter(**kw),
        }
        self._t = 0.0
        self._dt = 1.0 / max(float(fps), 1.0)

    def step(self, gaze: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for eye in ("left", "right"):
            entry = dict(gaze[eye])
            entry["yaw_rad"] = self._filters[(eye, "yaw")](
                entry["yaw_rad"], self._t,
            )
            entry["pitch_rad"] = self._filters[(eye, "pitch")](
                entry["pitch_rad"], self._t,
            )
            # Bug-fix (May 2026): do NOT regenerate dx/dy from -sin(yaw).
            # The iris-pixel-offset dx/dy assigned upstream in
            # PoseAndFaceDetectionV2 (nodes.py around line 1620) is the
            # screen-space ground truth and is what the renderer assumes.
            # Overwriting it from smoothed yaw caused the gaze arrow to
            # point opposite to the iris in debug images (user-reported).
            # We still smooth yaw_rad / pitch_rad (used by downstream
            # conditioning) but leave dx/dy intact.
            entry["magnitude"] = float(
                math.hypot(entry["yaw_rad"], entry["pitch_rad"]),
            )
            out[eye] = entry
        self._t += self._dt
        return out


# ---------------------------------------------------------------------------
# Convenience wrapper used by nodes.py
# ---------------------------------------------------------------------------
def is_available() -> bool:
    """True if any FaceLandmarker pipeline (tflite or Tasks API) is usable."""
    if _gaze_tflite is not None and _gaze_tflite.is_available():
        return True
    return _TASKS_AVAILABLE and (get_face_landmarker() is not None)
