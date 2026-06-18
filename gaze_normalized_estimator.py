# -*- coding: utf-8 -*-
"""Pose-normalized ResNet50 gaze estimator.

A clean implementation of the standard ResNet50 + 2-D regression head
architecture used by appearance-based gaze research since
Zhang et al. (2017, "It's Written All Over Your Face") and refined by
their 2020 ECCV paper for cross-dataset gaze estimation.

The architecture is:

    ResNet50 backbone (ImageNet layout, 2048-d feature vector)
        |
        +-- Linear(2048, 2) -> (pitch_rad, yaw_rad) in normalized space.

Input
-----
* 224x224 RGB face crop produced by :mod:`gaze_pose_norm` (i.e. the
  pose-normalized canonical view). The image must be normalized using
  ImageNet mean/std (the de-facto standard for ResNet50-pretrained
  features) before being passed to :func:`infer_normalized`.

Output
------
``(yaw_rad, pitch_rad)`` in the **normalized camera frame**. The
caller MUST de-rotate these using ``gaze_pose_norm.denormalize_gaze``
to bring them back to the original camera frame.

Checkpoint
----------
This estimator loads a community-released ResNet50 gaze checkpoint
trained on a large-scale gaze dataset. Place the checkpoint at one of:

    <ComfyUI>/models/gaze/pose_normalized_resnet50.pth.tar
    <ComfyUI>/models/gaze/pose_normalized_resnet50.pth
    <ComfyUI>/models/gaze/epoch_24_ckpt.pth.tar

The checkpoint is expected to be a state-dict (or a wrapper dict with
key ``'model_state'``) containing weights for ``gaze_network.*`` (the
backbone) and ``gaze_fc.0.*`` (the regression head). Mismatched keys
are tolerated via ``strict=False`` so older checkpoints (where the
1000-class ImageNet ``fc`` is also stored but unused) load cleanly.

LICENSE NOTE FOR USERS
----------------------
The weight files distributed by the original research group are under a
**non-commercial** licence (CC BY-NC-SA 4.0). This file itself is
Apache-2.0 / MIT and contains NO third-party code. You are responsible
for verifying that your use of the weight file you place at the path
above complies with its licence terms. The author of this node-pack
does not redistribute the weights; only the inference scaffold is
provided here.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for torch / torchvision.
_TORCH_OK: Optional[bool] = None
_torch = None  # type: ignore
_tv = None  # type: ignore

# Per-process pipeline cache.
_MODEL_CACHE: dict = {}

# Standard ImageNet preprocessing for ResNet50-pretrained backbones.
_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_torch() -> bool:
    """Import torch + torchvision lazily; cache the import status."""
    global _TORCH_OK, _torch, _tv
    if _TORCH_OK is not None:
        return _TORCH_OK
    try:
        import torch  # type: ignore
        import torchvision  # type: ignore  # noqa: F401
        _torch = torch
        import torchvision as tv  # type: ignore
        _tv = tv
        _TORCH_OK = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_normalized_estimator] torch import failed: %s", exc)
        _TORCH_OK = False
    return _TORCH_OK


def is_available() -> bool:
    """Return True iff torch+torchvision can be imported.

    A weight file is also required for inference; see
    :func:`locate_checkpoint`.
    """
    return _ensure_torch()


def _build_gaze_network():
    """Build the (uninitialized) Module matching the checkpoint layout.

    Module structure:

        gaze_network: torchvision ResNet50 backbone (returns 2048-d
            features; the ImageNet fc is kept in the state dict but
            bypassed during forward).
        gaze_fc: ``nn.Sequential(nn.Linear(2048, 2))`` regression head.

    The state-dict key prefixes (``gaze_network.*`` and ``gaze_fc.0.*``)
    are chosen to match the conventional community checkpoint released
    alongside the original research code. Using a different prefix
    layout would still load if the checkpoint dict is renamed at load
    time, but the convention here matches what is in the wild.
    """
    assert _ensure_torch()
    torch = _torch
    tv = _tv

    class _Backbone(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            # weights=None: we'll overwrite with the gaze checkpoint.
            # Keep the imagenet-style 1000-way fc so state-dict keys for
            # `fc.weight`/`fc.bias` exist (older gaze checkpoints store
            # them, just unused). Bypassed in forward.
            try:
                self.net = tv.models.resnet50(weights=None)
            except TypeError:
                # Older torchvision (<0.13) signature.
                self.net = tv.models.resnet50(pretrained=False)

        def forward(self, x):  # type: ignore[override]
            n = self.net
            x = n.conv1(x)
            x = n.bn1(x)
            x = n.relu(x)
            x = n.maxpool(x)
            x = n.layer1(x)
            x = n.layer2(x)
            x = n.layer3(x)
            x = n.layer4(x)
            x = n.avgpool(x)
            x = torch.flatten(x, 1)
            return x  # (B, 2048), pre-fc features.

    class _GazeNet(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.gaze_network = _Backbone()
            self.gaze_fc = torch.nn.Sequential(torch.nn.Linear(2048, 2))

        def forward(self, x):  # type: ignore[override]
            feat = self.gaze_network(x)
            return self.gaze_fc(feat)  # (B, 2) -> (pitch, yaw) in radians.

    return _GazeNet()


def _remap_state_dict(state_dict: dict) -> dict:
    """Rewrite state-dict keys so the ResNet50 backbone params live
    under ``gaze_network.net.*``.

    Community checkpoints store backbone keys as ``gaze_network.conv1.weight``,
    ``gaze_network.layer1.0.conv1.weight``, etc. -- i.e. directly under
    ``gaze_network``. Our wrapper adds an extra ``.net`` level so we
    can subclass torchvision's resnet50 module. Map the prefixes here.

    Also tolerates checkpoints saved from a ``DataParallel`` wrapper
    (``module.gaze_network.*``).
    """
    out: dict = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("gaze_network.") and not nk.startswith("gaze_network.net."):
            nk = "gaze_network.net." + nk[len("gaze_network."):]
        out[nk] = v
    return out


def locate_checkpoint() -> Optional[str]:
    """Search ComfyUI ``models/gaze/`` for a usable checkpoint file.

    Returns the absolute path of the first match, or ``None`` if no
    candidate is found. The caller is responsible for ensuring the
    checkpoint they placed at this path is licence-compatible with
    their use case.
    """
    candidates = [
        "pose_normalized_resnet50.pth.tar",
        "pose_normalized_resnet50.pth",
        "pose_normalized_resnet50.pt",
        "epoch_24_ckpt.pth.tar",  # original community filename
    ]
    # Preferred: use folder_paths if ComfyUI is on sys.path.
    try:
        import folder_paths  # type: ignore
        roots = list(folder_paths.get_folder_paths("gaze"))  # type: ignore
    except Exception:  # noqa: BLE001
        roots = []
    # Fallback: scan the standard ComfyUI install layout relative to CWD.
    cwd = os.getcwd()
    extra_roots = [
        os.path.join(cwd, "models", "gaze"),
        os.path.normpath(os.path.join(cwd, "..", "models", "gaze")),
        # Portable layout: ComfyUI_windows_portable\ComfyUI\models\gaze
        os.path.normpath(os.path.join(cwd, "ComfyUI", "models", "gaze")),
    ]
    for root in list(roots) + extra_roots:
        if not root or not os.path.isdir(root):
            continue
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None


def _register_gaze_folder() -> None:
    """Best-effort: register ``gaze`` under ComfyUI's folder registry so
    ``folder_paths.get_folder_paths('gaze')`` picks up
    ``ComfyUI/models/gaze/`` automatically. No-op outside ComfyUI."""
    try:
        import folder_paths  # type: ignore
        roots = list(folder_paths.get_folder_paths("gaze"))
        if roots:
            return
        # Walk up from this file to find the ComfyUI root then point at
        # models/gaze. This mirrors how core nodes are bootstrapped.
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(6):
            cand = os.path.join(here, "models", "gaze")
            if os.path.isdir(cand):
                folder_paths.add_model_folder_path("gaze", cand, is_default=True)  # type: ignore
                return
            here_parent = os.path.dirname(here)
            if here_parent == here:
                break
            here = here_parent
    except Exception:  # noqa: BLE001
        pass


def get_model(checkpoint_path: Optional[str] = None):
    """Return a ready-to-infer ``(model, device)`` pair, cached per checkpoint.

    Parameters
    ----------
    checkpoint_path
        Absolute path to the ``.pth.tar`` / ``.pth`` weight file. When
        ``None`` we call :func:`locate_checkpoint` to discover one in
        the standard ComfyUI ``models/gaze/`` location.

    Returns
    -------
    ``(model, device)`` tuple, or ``None`` on failure (missing torch,
    missing checkpoint, or unloadable state-dict).
    """
    if not _ensure_torch():
        return None
    torch = _torch

    if checkpoint_path is None:
        _register_gaze_folder()
        checkpoint_path = locate_checkpoint()
    if checkpoint_path is None or not os.path.isfile(checkpoint_path):
        logger.warning(
            "[gaze_normalized_estimator] No checkpoint found. Place a "
            "ResNet50 gaze weight file at "
            "<ComfyUI>/models/gaze/pose_normalized_resnet50.pth.tar to "
            "enable the pose_normalized_resnet50 engine."
        )
        return None

    cached = _MODEL_CACHE.get(checkpoint_path)
    if cached is not None:
        return cached

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "[gaze_normalized_estimator] Failed to load checkpoint %s: %s",
            checkpoint_path, exc,
        )
        return None
    # Accept both raw state-dicts and wrapper dicts.
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        logger.error(
            "[gaze_normalized_estimator] Unexpected checkpoint type: %r",
            type(ckpt),
        )
        return None
    state_dict = _remap_state_dict(state_dict)

    model = _build_gaze_network()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        # The bypassed imagenet fc may show up as missing if the
        # checkpoint omitted it -- benign.
        non_fc_missing = [k for k in missing if not k.startswith("gaze_network.net.fc.")]
        if non_fc_missing:
            logger.warning(
                "[gaze_normalized_estimator] Missing %d keys after remap; "
                "first 5: %s",
                len(non_fc_missing), non_fc_missing[:5],
            )
    if unexpected:
        logger.debug(
            "[gaze_normalized_estimator] %d unexpected keys ignored; "
            "first 5: %s", len(unexpected), unexpected[:5],
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _MODEL_CACHE[checkpoint_path] = (model, device)
    return model, device


def infer_normalized(
    model_device: tuple,
    warped_rgb_u8: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Run the ResNet50 gaze head on a pose-normalized 224x224 RGB crop.

    Parameters
    ----------
    model_device
        Tuple ``(model, device)`` returned by :func:`get_model`.
    warped_rgb_u8
        ``(224, 224, 3)`` uint8 RGB image, the output of
        :func:`gaze_pose_norm.normalize_face_for_gaze`. Sizes other
        than 224x224 will be resized; non-RGB inputs will produce
        garbage gaze.

    Returns
    -------
    ``(yaw_rad, pitch_rad, confidence)`` in the **normalized camera
    frame**, or ``None`` on failure. The confidence is always 1.0 here
    because the network has no logit; the field exists for API parity
    with :mod:`gaze_l2cs`.
    """
    if model_device is None or warped_rgb_u8 is None or warped_rgb_u8.size == 0:
        return None
    if not _ensure_torch():
        return None
    torch = _torch
    model, device = model_device
    img = warped_rgb_u8
    if img.shape[:2] != (224, 224):
        try:
            import cv2  # type: ignore
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:  # noqa: BLE001
            return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    try:
        x = torch.from_numpy(arr).unsqueeze(0).float().to(device)  # (1,3,224,224)
        with torch.no_grad():
            y = model(x)
        out = y.detach().cpu().numpy().reshape(-1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_normalized_estimator] forward failed: %s", exc)
        return None
    if out.size < 2 or not np.all(np.isfinite(out)):
        return None
    # Network output order is (pitch, yaw) in radians per convention.
    pitch_rad = float(out[0])
    yaw_rad = float(out[1])
    return yaw_rad, pitch_rad, 1.0


__all__ = [
    "is_available",
    "get_model",
    "infer_normalized",
    "locate_checkpoint",
]
