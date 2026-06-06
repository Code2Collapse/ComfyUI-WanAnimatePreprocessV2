# Copyright 2025-2026 Code2Collapse (https://github.com/Code2Collapse)
# Licensed under the Apache License, Version 2.0

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

# Temporal smoothing utilities (Kalman / EMA) for landmark sequences.
# Only numpy is required; the import is non-fatal so missing deps
# never block the rest of the node pack from loading.
try:
    from .nodes_extras.temporal_smoother import (  # noqa: F401
        KalmanSmoother1D,
        TemporalSmootherEMA,
        smooth_landmarks,
    )
    _TEMPORAL_SMOOTHER_OK = True
except Exception as _e:  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "WanAnimatePreprocessV2: temporal smoother unavailable: %s", _e,
    )
    _TEMPORAL_SMOOTHER_OK = False

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]