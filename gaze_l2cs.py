# -*- coding: utf-8 -*-
"""Stage-2 gaze upgrade: L2CS-Net inference.

L2CS-Net (Abdelrahman et al., 2022, MIT license) is a CNN that
estimates 3-D gaze (yaw, pitch) from a single RGB face crop. It is
trained on Gaze360 (extreme head poses, in-the-wild) and MPIIGaze
(near-frontal portraits). Both variants are MIT-licensed, weight files
hosted at HF ``tianfxc/l2cs``.

The L2CS-Net inference code ships directly in this repo
(``nodes_extras/l2csnet``) — no ``pip install l2cs`` and no clone needed.
The model runs on the face bounding box this pack already detects
(MediaPipe), so the upstream RetinaFace detector is not required. If torch
is somehow unavailable the engine combo falls back to
``blendshape_head_corrected``.

Public API
----------
* :func:`get_pipeline(variant='gaze360'|'mpiigaze')` -> cached pipeline
* :func:`infer_frame(pipeline, rgb_uint8, face_bbox=None)` ->
  ``(yaw_rad, pitch_rad, confidence)``

Sign convention matches the rest of the pack:
* ``yaw_rad`` > 0  -> subject looking to their *right*
* ``pitch_rad`` > 0 -> looking *up*

Weights auto-download from HF on first call to ``get_pipeline``; place
``L2CSNet_gaze360.pkl`` manually at ``ComfyUI/models/gaze/`` if offline.
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


class _L2CSGaze:
    """In-repo L2CS gaze predictor over a PRE-DETECTED face crop.

    Drop-in replacement for ``l2cs.Pipeline`` minus the bundled RetinaFace:
    this pack already supplies a face bbox (MediaPipe), so only the gaze
    regressor + the exact upstream preprocess/decode are needed.
    """

    def __init__(self, model, device, num_bins: int):
        import torch  # type: ignore
        self.model = model
        self.device = device
        self.idx_tensor = torch.arange(
            num_bins, dtype=torch.float32, device=device,
        )

    def predict_crop(self, face_rgb_u8: np.ndarray) -> Tuple[float, float]:
        """``face_rgb_u8``: HxWx3 uint8 RGB crop -> ``(yaw_rad, pitch_rad)``."""
        import torch  # type: ignore
        from .nodes_extras.l2csnet import prep_input_numpy, decode_gaze
        inp = prep_input_numpy(face_rgb_u8, self.device)
        with torch.no_grad():
            head_a, head_b = self.model(inp)   # (fc_yaw_gaze, fc_pitch_gaze)
        yaw_rad, pitch_rad = decode_gaze(head_a, head_b, self.idx_tensor)
        return float(yaw_rad[0]), float(pitch_rad[0])


def get_pipeline(variant: str = "gaze360"):
    """Lazy-construct the in-repo L2CS gaze predictor and cache it.

    Downloads weights from HF on first use. Raises ``RuntimeError`` if torch /
    the in-repo model cannot be imported.
    """
    if variant not in _WEIGHTS_HF:
        raise ValueError(f"Unknown L2CS variant {variant!r}; "
                         f"expected one of {list(_WEIGHTS_HF)}.")
    if variant in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[variant]

    import torch  # type: ignore
    try:
        from .nodes_extras.l2csnet import getArch, NUM_BINS  # type: ignore
    except Exception as exc:  # noqa: BLE001
        msg = (
            "[gaze_l2cs] FAILED to import the in-repo L2CS model "
            f"(torch/torchvision missing?): {exc!r}"
        )
        print(msg, flush=True)
        logger.error(msg)
        raise RuntimeError(msg) from exc

    repo, fname = _WEIGHTS_HF[variant]
    weight_path = os.path.join(_gaze_dir(), fname)
    if not os.path.exists(weight_path):
        print(
            f"[gaze_l2cs] Weights NOT found at {weight_path} -- "
            f"downloading from huggingface.co/{repo}/{fname} ...",
            flush=True,
        )
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as exc:  # noqa: BLE001
            msg = (
                "[gaze_l2cs] FAILED: huggingface_hub not available; cannot "
                f"auto-download L2CS weights. Place {fname} manually at "
                f"{weight_path}."
            )
            print(msg, flush=True)
            logger.error(msg)
            raise RuntimeError(msg) from exc
        try:
            weight_path = hf_hub_download(
                repo_id=repo, filename=fname, local_dir=_gaze_dir(),
            )
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"[gaze_l2cs] DOWNLOAD FAILED for {repo}/{fname}: {exc!r}. "
                f"Check internet / HF_TOKEN / disk space. Falling back to "
                f"blendshape_head_corrected engine."
            )
            print(msg, flush=True)
            logger.error(msg)
            raise RuntimeError(msg) from exc
        try:
            size_mb = os.path.getsize(weight_path) / (1024 * 1024)
        except Exception:  # noqa: BLE001
            size_mb = -1.0
        print(
            f"[gaze_l2cs] Downloaded OK ({size_mb:.1f} MB) -> {weight_path}",
            flush=True,
        )
    else:
        print(f"[gaze_l2cs] Using cached weights: {weight_path}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[gaze_l2cs] Initialising in-repo L2CS-Net "
        f"(variant={variant}, device={device})",
        flush=True,
    )
    try:
        model = getArch("ResNet50", NUM_BINS)
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        # The released .pkl is a plain state_dict; tolerate a few wrappers.
        if isinstance(state, torch.nn.Module):
            state = state.state_dict()
        elif isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        pipeline = _L2CSGaze(model, device, NUM_BINS)
    except Exception as exc:  # noqa: BLE001
        msg = (
            f"[gaze_l2cs] MODEL INIT FAILED ({exc!r}); falling back to "
            f"blendshape_head_corrected."
        )
        print(msg, flush=True)
        logger.error(msg)
        raise
    _PIPELINE_CACHE[variant] = pipeline
    return pipeline


def infer_frame(
    pipeline,
    rgb_uint8: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[Tuple[float, float, float]]:
    """Run in-repo L2CS on a single RGB frame -> ``(yaw, pitch, conf)``.

    Uses the face bounding box this pack already detected (MediaPipe): crop
    it (with a small margin), resize to 224x224 and run the gaze regressor.
    No RetinaFace, no full-frame detection. Returns ``None`` if there is no
    usable face crop.

    Parameters
    ----------
    pipeline
        From :func:`get_pipeline` (a :class:`_L2CSGaze`).
    rgb_uint8
        ``(H, W, 3)`` uint8 image in RGB order (ComfyUI convention).
    face_bbox
        ``(x1, y1, x2, y2)`` in ``rgb_uint8`` pixel coords. Required.

    Returns
    -------
    ``(yaw_rad, pitch_rad, confidence)`` or ``None``.

    Sign convention (unchanged, matches l2cs.utils.draw_gaze):
        yaw  > 0 -> subject's RIGHT gaze;  pitch > 0 -> looking UP.
    """
    if rgb_uint8 is None or getattr(rgb_uint8, "size", 0) == 0:
        return None
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        return None
    if face_bbox is None:
        # This pack always supplies a bbox; without a detector we can't crop.
        return None
    H, W = rgb_uint8.shape[:2]
    try:
        x1, y1, x2, y2 = (float(v) for v in face_bbox[:4])
    except Exception:  # noqa: BLE001
        return None
    bw, bh = x2 - x1, y2 - y1
    if bw <= 1 or bh <= 1:
        return None
    # Small context margin around the face (gaze benefits from a little context).
    mx, my = bw * 0.10, bh * 0.10
    ix1 = max(0, int(round(x1 - mx)))
    iy1 = max(0, int(round(y1 - my)))
    ix2 = min(W, int(round(x2 + mx)))
    iy2 = min(H, int(round(y2 + my)))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    crop = rgb_uint8[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    # Resize to 224x224 (upstream Pipeline.step does the same; the transform
    # then up-samples to 448 before normalising).
    try:
        import cv2  # type: ignore
        crop224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    except Exception:  # noqa: BLE001
        try:
            from PIL import Image  # type: ignore
            crop224 = np.asarray(
                Image.fromarray(crop).resize((224, 224)), dtype=np.uint8,
            )
        except Exception:  # noqa: BLE001
            return None
    crop224 = np.ascontiguousarray(crop224.astype(np.uint8))
    try:
        yaw_rad, pitch_rad = pipeline.predict_crop(crop224)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_l2cs] predict_crop failed: %s", exc)
        return None
    return float(yaw_rad), float(pitch_rad), 1.0


def is_available() -> bool:
    """True iff the in-repo L2CS model imports (needs torch + torchvision)."""
    try:
        from .nodes_extras.l2csnet import getArch  # type: ignore  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
