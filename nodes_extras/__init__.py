"""WanAnimatePreprocessV2 — extension nodes.

Adds the missing capabilities from the upgrade catalog:
  - WanHeadPose6DoFV2              (b) 6DoF head pose via PnP
  - WanIrisControlNetV2            (c) iris-gaze ControlNet conditioning image
  - WanSHLightingTransferV2        (e) Spherical-harmonics lighting fit + transfer
  - WanExpressionCoefficientsV2    (f) Structured ARKit/FACS expression coeffs
  - WanQualityScorerJitterV2       (g) Temporal jitter + visibility + stability

All nodes are real, working implementations — no stubs.
"""

from .head_pose_6dof import WanHeadPose6DoFV2
from .iris_controlnet import WanIrisControlNetV2
from .sh_lighting import WanSHLightingTransferV2
from .expression_coeffs import WanExpressionCoefficientsV2
from .quality_scorer_jitter import WanQualityScorerJitterV2


EXTRA_NODE_CLASS_MAPPINGS = {
    "WanHeadPose6DoFV2": WanHeadPose6DoFV2,
    "WanIrisControlNetV2": WanIrisControlNetV2,
    "WanSHLightingTransferV2": WanSHLightingTransferV2,
    "WanExpressionCoefficientsV2": WanExpressionCoefficientsV2,
    "WanQualityScorerJitterV2": WanQualityScorerJitterV2,
}

EXTRA_NODE_DISPLAY_NAME_MAPPINGS = {
    "WanHeadPose6DoFV2": "Wan Head Pose 6DoF — solvePnP (V2)",
    "WanIrisControlNetV2": "Wan Iris ControlNet Conditioning (V2)",
    "WanSHLightingTransferV2": "Wan SH Lighting Transfer (V2)",
    "WanExpressionCoefficientsV2": "Wan Expression Coefficients (V2)",
    "WanQualityScorerJitterV2": "Wan Quality Scorer — Temporal Jitter (V2)",
}
