"""WanAnimatePreprocessV2 — extension nodes.

The face-expression / 3-DoF head-pose / gaze / FACS-coefficient capabilities
that used to live in 5 separate nodes are now consolidated into a single
unified node — ``WanFaceController3DV2`` — which exposes everything through
one input panel and an in-canvas viewer (draggable iBUG-68 face landmarks,
OpenPose-18 body joints, ETH-XGaze gaze handles, per-frame timeline).

The legacy classes still live on disk under ``nodes_extras/`` so that
``face_controller_3d.py`` can import their helper functions, but they are
no longer registered with ComfyUI — only the unified node appears in the
node picker.

Currently registered:
  - WanIrisControlNetV2          iris-gaze ControlNet conditioning image
  - WanSHLightingTransferV2      spherical-harmonics lighting fit + transfer
  - WanQualityScorerJitterV2     temporal jitter / visibility / stability
  - WanPoseFormatConvertV2       OP18 → BODY-25 / COCO-17 / MP-33 conversion
  - WanPoseDetectViTPoseV2       YOLO + ViTPose detector
  - WanFaceController3DV2        unified face / expression / pose / gaze
"""

from .iris_controlnet import WanIrisControlNetV2
from .sh_lighting import WanSHLightingTransferV2
from .quality_scorer_jitter import WanQualityScorerJitterV2
from .pose_format_convert import WanPoseFormatConvertV2
from .pose_detect_vitpose import WanPoseDetectViTPoseV2
from .face_controller_3d import WanFaceController3DV2

# Phase 1.B — live preview route for the Face Director real-time editor.
# Registers POST /c2c/fc3d_preview against ComfyUI's aiohttp server.
# Failure is non-fatal: the node still works, only the live gizmo loses
# its server-truth sync.
try:
    from . import _face_preview_server as _fps
    _fps.try_register_routes_deferred()
except Exception as _e:                                                  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).info(
        "fc3d_preview route registration skipped: %s", _e,
    )

# ETH-XGaze post-processor (optional; only loads if torch + checkpoint available).
try:
    from .gaze_ethxgaze import WanGazeETHXGazeV2
    _ETHXGAZE_OK = True
except Exception as _e:                                                  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).info(
        "WanGazeETHXGazeV2 not registered: %s", _e,
    )
    WanGazeETHXGazeV2 = None                                             # type: ignore
    _ETHXGAZE_OK = False


EXTRA_NODE_CLASS_MAPPINGS = {
    "WanIrisControlNetV2":      WanIrisControlNetV2,
    "WanSHLightingTransferV2":  WanSHLightingTransferV2,
    "WanQualityScorerJitterV2": WanQualityScorerJitterV2,
    "WanPoseFormatConvertV2":   WanPoseFormatConvertV2,
    "WanPoseDetectViTPoseV2":   WanPoseDetectViTPoseV2,
    "WanFaceController3DV2":    WanFaceController3DV2,
}
if _ETHXGAZE_OK:
    EXTRA_NODE_CLASS_MAPPINGS["WanGazeETHXGazeV2"] = WanGazeETHXGazeV2

EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {
    "WanIrisControlNetV2":      "Wan Iris ControlNet Conditioning (V2)",
    "WanSHLightingTransferV2":  "Wan SH Lighting Transfer (V2)",
    "WanQualityScorerJitterV2": "Wan Quality Scorer — Temporal Jitter (V2)",
    "WanPoseFormatConvertV2":   "Wan Pose Format Convert — OP18 → BODY-25 / COCO-17 / MP-33 (V2)",
    "WanPoseDetectViTPoseV2":   "Wan Pose Detect — YOLO + ViTPose (V2)",
    "WanFaceController3DV2":    "Wan Face Controller 3D",
}
if _ETHXGAZE_OK:
    EXTRA_NODE_DISPLAY_NAME_MAPPINGS["WanGazeETHXGazeV2"] = (
        "Wan Gaze — ETH-XGaze Post-Processor (V2)"
    )
