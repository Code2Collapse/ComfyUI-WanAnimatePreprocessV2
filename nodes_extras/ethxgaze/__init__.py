# ETH-XGaze gaze network — source: https://github.com/xucong-zhang/ETH-XGaze
# (ECCV 2020). Inference code added directly into this repo so the `ethxgaze`
# gaze engine works without a separate clone. Only the gaze_network (ResNet-50)
# + face_model.txt are included; the demo's dlib detector/landmark .dat files are
# NOT needed (this pack does its own face detection). The pretrained checkpoint
# (epoch_24_ckpt.pth.tar) is a research weight — place it in ComfyUI/models/ethxgaze/.
from .model import gaze_network  # noqa: F401
