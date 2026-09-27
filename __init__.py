# Copyright 2025-2026 Code2Collapse (https://github.com/Code2Collapse)
# Licensed under the Apache License, Version 2.0

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./js"

# ViTPose / YOLO loaders use ONNX files under models/detection/, but ComfyUI's
# default folder_paths entry only whitelists .bin/.pt/etc.  Without .onnx the
# combo shows vitpose_h_wholebody_data.bin (a corrupt 2.5 GB blob) and hides the
# real vitpose_h_wholebody_model.onnx + yolov10m.onnx pair.
try:
    import folder_paths as _fp

    _det_entry = _fp.folder_names_and_paths.get("detection")
    if _det_entry:
        _det_dirs, _det_ext = _det_entry
        if ".onnx" not in _det_ext:
            _fp.folder_names_and_paths["detection"] = (_det_dirs, _det_ext | {".onnx"})
            _fp.filename_list_cache.pop("detection", None)
except Exception as _e:  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).debug("detection .onnx extension patch skipped: %s", _e)

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

# ── One menu root for every Code2Collapse pack ─────────────────────────────
# Every node lands under "🐺 C2C/<pack>/<family>" in the Add Node menu and the
# node library (see _c2c_menu.py). Node ids are untouched, so saved workflows
# are unaffected. Guarded: a menu placement must never cost the pack its nodes.
try:
    from ._c2c_menu import rebrand_v1 as _c2c_menu_rebrand

    _c2c_menu_rebrand(
        NODE_CLASS_MAPPINGS, "\U0001F9CD Wan Animate Preprocess",
        strip=("WanAnimatePreprocess", "WanAnimatePreprocessV2", "WanAnimatePreprocess_V2"),
        rename={"extras": "Extras", "KANIBUS": "Kanibus"},
    )
except Exception as _c2c_menu_exc:  # noqa: BLE001
    import logging as _c2c_menu_log

    _c2c_menu_log.getLogger(__name__).warning("C2C menu root not applied: %s", _c2c_menu_exc)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]