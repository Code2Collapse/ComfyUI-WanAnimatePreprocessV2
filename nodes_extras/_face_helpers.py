# Copyright 2026 Code2Collapse / WanAnimatePreprocessV2.
# Licensed under the Apache License, Version 2.0.

"""Neutral helpers shared by face / pose / gaze nodes.

These were previously private to ``face_expression_editor_ui`` but are
imported by ``face_controller_3d`` and ``gaze_ethxgaze``. Lifting them
out of the UI module makes the import graph acyclic and removes the
implicit dependency on an unregistered node class.

Public surface (re-exported by ``face_expression_editor_ui`` for backward
compatibility):

* :data:`_GAZE_MAX_YAW_RAD`
* :data:`_GAZE_MAX_PITCH_RAD`
* :func:`_get_body_kps`
* :func:`_set_body_kps`
* :func:`_parse_overrides`
* :func:`_parse_gaze_overrides`
* :func:`_apply_gaze_override_to_iris_entry`
"""

from __future__ import annotations

import json
import logging
import math
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)


# Canonical max-gaze angle constants — must match gaze_blendshape.py so that
# the JS drag offset (which is bbox-normalised) maps to the SAME (yaw, pitch)
# radians the downstream pipeline already expects.
_GAZE_MAX_YAW_RAD = math.radians(30.0)
_GAZE_MAX_PITCH_RAD = math.radians(25.0)


# ── Body keypoint helpers (image-normalised [0,1] floats) ─────────────
def _get_body_kps(meta) -> np.ndarray | None:
    """Return (18, 2) image-normalised body keypoints, or None.

    ``keypoints_body`` in POSEDATA is stored as a python list of 18 items;
    each item is either ``None`` (joint missing) or ``[x_norm, y_norm,
    (confidence?)]``. Missing joints are preserved as NaN so the JS
    overlay can hide them.
    """
    if not isinstance(meta, dict):
        return None
    arr = meta.get("keypoints_body")
    if not isinstance(arr, (list, tuple)) or len(arr) < 18:
        return None
    out = np.full((18, 2), np.nan, dtype=np.float32)
    for i in range(18):
        v = arr[i]
        if v is None:
            continue
        try:
            out[i, 0] = float(v[0])
            out[i, 1] = float(v[1])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _set_body_kps(meta: dict, xy_norm: np.ndarray) -> None:
    """Write back (18, 2) image-normalised body keypoints into ``meta``.

    Preserves the original confidence (3rd component) when present, and
    keeps NaN-slots as None so downstream renderers can distinguish
    "edited" from "missing".
    """
    src = meta.get("keypoints_body")
    if not isinstance(src, list):
        return
    for i in range(min(18, len(src))):
        x, y = float(xy_norm[i, 0]), float(xy_norm[i, 1])
        if np.isnan(x) or np.isnan(y):
            continue
        v = src[i]
        if v is None:
            src[i] = [x, y, 1.0]
        else:
            try:
                src[i] = [x, y, float(v[2])] if len(v) >= 3 else [x, y]
            except (TypeError, ValueError, IndexError):
                src[i] = [x, y]


# ── Per-frame override JSON parsers (from JS overlays) ────────────────
def _parse_overrides(blob: str | None) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Parse the JS-overlay payload. Returns ``{}`` on empty/invalid.

    Expected shape: ``{"frames":{"<frame_idx>":{"<lm_idx>":[x,y]}}}``.
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("landmark_overrides_json ignored - not valid JSON: %s", e)
        return {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for f_key, lm_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(lm_map, dict):
            continue
        per_frame: Dict[int, Tuple[float, float]] = {}
        for lm_key, xy in lm_map.items():
            try:
                lm_idx = int(lm_key)
            except (TypeError, ValueError):
                continue
            if not (isinstance(xy, (list, tuple)) and len(xy) == 2):
                continue
            try:
                per_frame[lm_idx] = (float(xy[0]), float(xy[1]))
            except (TypeError, ValueError):
                continue
        if per_frame:
            out[f_idx] = per_frame
    return out


def _parse_gaze_overrides(blob: str | None) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """Parse the JS gaze-overlay payload.

    Expected shape::

        {"frames":{"<frame_idx>":{"l":[yaw,pitch],"r":[yaw,pitch]}}}

    Returns ``{frame_idx: {"l": (yaw, pitch), "r": (yaw, pitch)}}``.
    Missing eyes / invalid entries are silently dropped.
    """
    if not blob or not blob.strip():
        return {}
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        log.warning("gaze_overrides_json ignored - not valid JSON: %s", e)
        return {}
    frames = data.get("frames") if isinstance(data, dict) else None
    if not isinstance(frames, dict):
        return {}
    out: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for f_key, eye_map in frames.items():
        try:
            f_idx = int(f_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(eye_map, dict):
            continue
        per_frame: Dict[str, Tuple[float, float]] = {}
        for eye_key in ("l", "r"):
            v = eye_map.get(eye_key)
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                continue
            try:
                per_frame[eye_key] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue
        if per_frame:
            out[f_idx] = per_frame
    return out


def _apply_gaze_override_to_iris_entry(entry: dict, eye_key: str,
                                       yaw_rad: float, pitch_rad: float) -> None:
    """Patch a single per-frame ``iris_data`` entry's ``{eye}_gaze`` dict.

    Mirrors the field shape consumed by gaze_blendshape and the
    Wan-Animate pose renderer: ``yaw_rad``, ``pitch_rad``, plus a
    normalised 2-D direction (``dx``, ``dy``) and a magnitude in [0, 1].
    ``source`` is set to ``"user_override"`` so the pipeline knows the
    value came from the UI.
    """
    eye_name = "left" if eye_key == "l" else "right"
    key = f"{eye_name}_gaze"
    g = entry.get(key)
    if not isinstance(g, dict):
        g = {}
    yaw_n = max(-1.0, min(1.0, yaw_rad / _GAZE_MAX_YAW_RAD))
    pitch_n = max(-1.0, min(1.0, pitch_rad / _GAZE_MAX_PITCH_RAD))
    nrm = float(math.hypot(yaw_n, pitch_n))
    if nrm > 1e-6:
        dx = yaw_n / nrm
        dy = -pitch_n / nrm  # image-Y is down -> invert pitch
        mag = min(1.0, nrm)
    else:
        dx = dy = 0.0
        mag = 0.0
    g["yaw_rad"] = float(yaw_rad)
    g["pitch_rad"] = float(pitch_rad)
    g["dx"] = round(dx, 4)
    g["dy"] = round(dy, 4)
    g["magnitude_norm"] = round(mag, 4)
    g["source"] = "user_override"
    entry[key] = g


__all__ = [
    "_GAZE_MAX_YAW_RAD",
    "_GAZE_MAX_PITCH_RAD",
    "_get_body_kps",
    "_set_body_kps",
    "_parse_overrides",
    "_parse_gaze_overrides",
    "_apply_gaze_override_to_iris_entry",
]
