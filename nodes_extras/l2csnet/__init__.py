# In-repo L2CS-Net inference (gaze estimation).
# Source: https://github.com/edavalosanaya/L2CS-Net (MIT) — model + the
# preprocessing/decoding from l2cs/utils.py + l2cs/pipeline.py, reproduced
# here so the L2CS gaze engine ships built-in (no pip package, no clone).
#
# What is included: the L2CS ResNet model, getArch(), the exact ImageNet
# transform, and the 90-bin softmax->angle decode. What is NOT included: the
# upstream RetinaFace detector (this pack provides its own face bbox).
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from .model import L2CS  # noqa: F401

__all__ = ["L2CS", "getArch", "prep_input_numpy", "decode_gaze", "NUM_BINS"]

NUM_BINS = 90

# Exact upstream transform (l2cs/utils.py): face crop -> PIL -> Resize(448) ->
# tensor -> ImageNet normalize.
_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def getArch(arch: str, bins: int) -> L2CS:
    """Build an L2CS model for the requested ResNet backbone (upstream parity)."""
    if arch == "ResNet18":
        return L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], bins)
    if arch == "ResNet34":
        return L2CS(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], bins)
    if arch == "ResNet101":
        return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], bins)
    if arch == "ResNet152":
        return L2CS(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], bins)
    if arch != "ResNet50":
        print(f"[l2csnet] Unknown arch {arch!r}; defaulting to ResNet50.")
    return L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], bins)


def prep_input_numpy(img: np.ndarray, device) -> torch.Tensor:
    """Preprocess an RGB uint8 face crop (HxWx3) or batch (NxHxWx3)."""
    if img.ndim == 4:
        img = torch.stack([_TRANSFORM(im) for im in img])
    else:
        img = _TRANSFORM(img)
    img = img.to(device)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    return img


def decode_gaze(head_a: torch.Tensor, head_b: torch.Tensor,
                idx_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Decode the two raw L2CS head outputs to continuous (yaw, pitch) RADIANS.

    ``head_a`` / ``head_b`` are the model's forward() outputs in order, i.e.
    ``(fc_yaw_gaze, fc_pitch_gaze)`` — head_a is ALWAYS the true yaw head,
    head_b is ALWAYS the true pitch head (see model.py forward()).

    BUG FIX (verified against github.com/Ahmednull/L2CS-Net upstream,
    cloned to third_party/L2CS-Net for direct comparison): the previous
    version of this function replicated upstream pipeline.py's confusingly
    -named intermediate variables (``gaze_pitch, gaze_yaw = self.model(img)``
    -- upstream names head_a's output "gaze_pitch" even though it is the
    true-yaw head) but then ALSO swapped the final RETURN order to
    ``(yaw_rad, pitch_rad) = (decode(head_b), decode(head_a))``. Upstream's
    own predict_gaze() does NOT do that: despite the confusing internal
    names, its literal `return pitch_predicted, yaw_predicted` returns
    ``(decode(head_a), decode(head_b))`` -- the SAME order as the model's
    own heads. Verified end-to-end against l2cs/vis.py's draw_gaze(), the
    only place upstream's convention has an unambiguous, testable meaning
    (screen-space arrow direction): draw_gaze receives (results.pitch,
    results.yaw) = (decode(head_a), decode(head_b)) and computes
    ``dx = -sin(pitchyaw[0])*cos(pitchyaw[1])``, i.e. it needs
    pitchyaw[0]=true_yaw and pitchyaw[1]=true_pitch -- confirming
    decode(head_a)=true_yaw, decode(head_b)=true_pitch, in that order.
    The previous code's swapped return meant every caller received pitch
    data labeled yaw and vice versa -- a 90-degree-confused gaze estimate
    that reads as "confidently wrong", not just noisy. Decode each head
    DIRECTLY (no relabeling detour -- it never changed the computation,
    only obscured which value ended up in which return slot).
    Expectation over 90 bins, *4-180 -> degrees (4-deg bins, -180..176),
    then deg->rad.
    """
    yaw_p = torch.softmax(head_a, dim=1)
    pitch_p = torch.softmax(head_b, dim=1)
    yaw_deg = torch.sum(yaw_p * idx_tensor, dim=1) * 4 - 180
    pitch_deg = torch.sum(pitch_p * idx_tensor, dim=1) * 4 - 180
    yaw_rad = yaw_deg.cpu().detach().numpy() * np.pi / 180.0
    pitch_rad = pitch_deg.cpu().detach().numpy() * np.pi / 180.0
    return yaw_rad, pitch_rad
