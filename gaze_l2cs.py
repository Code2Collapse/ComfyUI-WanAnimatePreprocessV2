# -*- coding: utf-8 -*-
"""Stage-2 gaze upgrade: L2CS-Net inference.

L2CS-Net (Abdelrahman et al., 2022, MIT license) is a CNN that
estimates 3-D gaze (yaw, pitch) from a single RGB face crop. It is
trained on Gaze360 (extreme head poses, in-the-wild) and MPIIGaze
(near-frontal portraits). Both variants are MIT-licensed, weight files
hosted at HF ``tianfxc/l2cs``.

This module is OPTIONAL — the L2CS python package is not a hard
dependency of the node pack. When unavailable, the engine combo silently
falls back to ``blendshape_head_corrected``.

Public API
----------
* :func:`get_pipeline(variant='gaze360'|'mpiigaze')` -> cached pipeline
* :func:`infer_frame(pipeline, rgb_uint8, face_bbox=None)` ->
  ``(yaw_rad, pitch_rad, confidence)``

Sign convention matches the rest of the pack:
* ``yaw_rad`` > 0  -> subject looking to their *right*
* ``pitch_rad`` > 0 -> looking *up*

Install
-------
    pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main

Weights auto-download from HF on first call to ``get_pipeline``.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import numpy as np

try:
    import folder_paths  # type: ignore
    _FOLDER_PATHS_OK = True
except Exception:  # noqa: BLE001
    folder_paths = None  # type: ignore
    _FOLDER_PATHS_OK = False

logger = logging.getLogger(__name__)

# (HF repo, weight filename) per variant.
_WEIGHTS_HF = {
    "gaze360":  ("tianfxc/l2cs", "L2CSNet_gaze360.pkl"),
    "mpiigaze": ("tianfxc/l2cs", "L2CSNet_mpiigaze.pkl"),
}

_PIPELINE_CACHE: dict = {}


def _gaze_dir() -> str:
    """Resolve ``ComfyUI/models/gaze/`` (or a sensible fallback)."""
    if _FOLDER_PATHS_OK and hasattr(folder_paths, "models_dir"):
        base = folder_paths.models_dir
    else:
        base = os.path.join(os.path.dirname(__file__), "models")
    d = os.path.join(base, "gaze")
    os.makedirs(d, exist_ok=True)
    return d


def get_pipeline(variant: str = "gaze360"):
    """Lazy-construct an ``l2cs.Pipeline`` and cache it.

    Downloads weights from HF on first use. Raises ``RuntimeError`` if
    the ``l2cs`` package is not installed.
    """
    if variant not in _WEIGHTS_HF:
        raise ValueError(f"Unknown L2CS variant {variant!r}; "
                         f"expected one of {list(_WEIGHTS_HF)}.")
    if variant in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[variant]

    try:
        from l2cs import Pipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "l2cs python package not installed. Run:\n"
            "  pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main"
        ) from exc

    import torch  # type: ignore

    repo, fname = _WEIGHTS_HF[variant]
    weight_path = os.path.join(_gaze_dir(), fname)
    if not os.path.exists(weight_path):
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "huggingface_hub not available; cannot auto-download L2CS "
                f"weights. Place {fname} manually at {weight_path}."
            ) from exc
        logger.info("[gaze_l2cs] downloading %s/%s -> %s", repo, fname, weight_path)
        weight_path = hf_hub_download(
            repo_id=repo, filename=fname, local_dir=_gaze_dir(),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[gaze_l2cs] init Pipeline(variant=%s, device=%s, weights=%s)",
                variant, device, weight_path)
    pipeline = Pipeline(weights=weight_path, arch="ResNet50", device=device)
    _PIPELINE_CACHE[variant] = pipeline
    return pipeline


def infer_frame(
    pipeline,
    rgb_uint8: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Run L2CS on a single RGB frame and return ``(yaw, pitch, conf)``.

    The L2CS-Net ``Pipeline.step(frame)`` API runs a YOLOv8 face
    detector and then estimates gaze for each detected face. We return
    the primary (largest-bbox) face. Output ``yaw``/``pitch`` are in
    radians. Returns ``None`` if no face was detected.

    Parameters
    ----------
    pipeline
        From :func:`get_pipeline`.
    rgb_uint8
        ``(H, W, 3)`` uint8 image (RGB).
    face_bbox
        Optional ``(x1, y1, x2, y2)`` in ``rgb_uint8`` coords. Unused
        right now (L2CS pipeline detects faces itself), kept for
        future per-face routing.

    Returns
    -------
    ``(yaw_rad, pitch_rad, confidence)`` or ``None``.
    """
    if rgb_uint8 is None or rgb_uint8.size == 0:
        return None
    try:
        # L2CS step() returns a GazeResultContainer with arrays of
        # per-face yaw/pitch/score/bbox.
        results = pipeline.step(rgb_uint8)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_l2cs] pipeline.step failed: %s", exc)
        return None
    if results is None:
        return None
    yaws = getattr(results, "pitch", None)  # placeholder check; real attrs below
    yaws = getattr(results, "yaw", None)
    pitches = getattr(results, "pitch", None)
    if yaws is None or pitches is None or len(yaws) == 0:
        return None

    # Pick the primary face: largest bbox if available, else first.
    bboxes = getattr(results, "bboxes", None)
    idx = 0
    if bboxes is not None and len(bboxes) == len(yaws) and len(bboxes) > 1:
        try:
            areas = [max(0, (b[2] - b[0])) * max(0, (b[3] - b[1])) for b in bboxes]
            idx = int(np.argmax(areas))
        except Exception:  # noqa: BLE001
            idx = 0

    try:
        yaw = float(yaws[idx])
        pitch = float(pitches[idx])
    except Exception:  # noqa: BLE001
        return None

    # L2CS outputs are already in radians. Sign convention matches our
    # spec: yaw>0 = subject's right, pitch>0 = up — verified against
    # the upstream renderer in `l2cs.render`.
    scores = getattr(results, "scores", None)
    conf = 1.0
    if scores is not None and len(scores) > idx:
        try:
            conf = float(scores[idx])
        except Exception:  # noqa: BLE001
            conf = 1.0
    return yaw, pitch, conf


def is_available() -> bool:
    """Return True iff the ``l2cs`` python package can be imported."""
    try:
        import l2cs  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
