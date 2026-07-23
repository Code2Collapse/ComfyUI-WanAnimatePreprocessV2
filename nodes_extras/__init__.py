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
# WanExpressionCoefficientsV2 is NOT superseded by the WanFaceController3DV2
# consolidation: that node's coeff_time_series_json logs the user's MANUALLY
# APPLIED editor overrides (an authoring/audit trail), whereas this node
# MEASURES real ARKit-52 blendshapes detected from footage via MediaPipe
# (iris_data_json in, coeffs_json out) — usable on ANY video, including a
# Wan-Animate generated output, which is exactly the "AU-extraction tooling"
# a closed-loop fidelity critic (Wan-Animate spec Section 3.1) needs. It was
# never wired into EXTRA_NODE_CLASS_MAPPINGS below, so it was unreachable —
# see WanExpressionCriticV2 in expression_critic.py, which consumes its output.
from .expression_coeffs import WanExpressionCoefficientsV2
from .expression_critic import WanExpressionCriticV2

# Phase 1.B — live preview route for the Face Director real-time editor.
# Registers POST /c2c/fc3d_preview against ComfyUI's aiohttp server.
# Failure is non-fatal: the node still works, only the live gizmo loses
# its server-truth sync.
import os as _os
if not _os.environ.get("FC3D_SKIP_ROUTE_REG"):
    try:
        from . import _face_preview_server as _fps
        _fps.try_register_routes_deferred()
    except Exception as _e:                                              # noqa: BLE001
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


# NLF 3D-pose refinement (optional; needs torch — guarded so a failure never
# breaks the rest of the pack).
try:
    from .pose3d_nlf import WanPose3DRefineNLFV2
    _POSE3D_OK = True
except Exception as _e:                                                  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).info("WanPose3DRefineNLFV2 not registered: %s", _e)
    WanPose3DRefineNLFV2 = None                                          # type: ignore
    _POSE3D_OK = False

# NOTE: the standalone WanPoseOverlayV2 node was removed — its full
# body+face+hands+gaze verification overlay is now folded directly into
# PoseAndFaceDetectionV2's built-in `debug_image` output (no second node to
# wire). pose_overlay.py is kept as a drawing-helper module.

EXTRA_NODE_CLASS_MAPPINGS = {
    "WanIrisControlNetV2":      WanIrisControlNetV2,
    "WanSHLightingTransferV2":  WanSHLightingTransferV2,
    "WanQualityScorerJitterV2": WanQualityScorerJitterV2,
    "WanPoseFormatConvertV2":   WanPoseFormatConvertV2,
    "WanPoseDetectViTPoseV2":   WanPoseDetectViTPoseV2,
    "WanFaceController3DV2":    WanFaceController3DV2,
    "WanExpressionCoefficientsV2": WanExpressionCoefficientsV2,
    "WanExpressionCriticV2":       WanExpressionCriticV2,
}
if _ETHXGAZE_OK:
    EXTRA_NODE_CLASS_MAPPINGS["WanGazeETHXGazeV2"] = WanGazeETHXGazeV2
if _POSE3D_OK:
    EXTRA_NODE_CLASS_MAPPINGS["WanPose3DRefineNLFV2"] = WanPose3DRefineNLFV2

EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {
    "WanIrisControlNetV2":      "Wan Iris ControlNet Conditioning (V2)",
    "WanSHLightingTransferV2":  "Wan SH Lighting Transfer (V2)",
    "WanQualityScorerJitterV2": "Wan Quality Scorer — Temporal Jitter (V2)",
    "WanPoseFormatConvertV2":   "Wan Pose Format Convert — OP18 → BODY-25 / COCO-17 / MP-33 (V2)",
    "WanPoseDetectViTPoseV2":   "Wan Pose Detect — YOLO + ViTPose (V2)",
    "WanFaceController3DV2":    "Wan Face Controller 3D",
    "WanExpressionCoefficientsV2": "Wan Expression Coefficients — ARKit-52 Export (V2)",
    "WanExpressionCriticV2":       "Wan Expression Critic — Source vs Generated AU Error (V2)",
}
if _ETHXGAZE_OK:
    EXTRA_NODE_DISPLAY_NAME_MAPPINGS["WanGazeETHXGazeV2"] = (
        "Wan Gaze — ETH-XGaze Post-Processor (V2)"
    )
if _POSE3D_OK:
    EXTRA_NODE_DISPLAY_NAME_MAPPINGS["WanPose3DRefineNLFV2"] = (
        "Wan Pose 3D Refine — NLF (V2)"
    )
