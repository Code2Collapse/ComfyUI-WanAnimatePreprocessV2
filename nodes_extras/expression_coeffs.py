"""WanExpressionCoefficientsV2 — Structured ARKit/FACS expression coefficient export.

MediaPipe FaceLandmarker (already integrated in gaze_blendshape.py) emits 52
ARKit blendshapes — the de-facto industry-standard expression basis used by
Apple, every VTuber pipeline, Unreal MetaHuman, and Unity AR Foundation. We
expose them as a clean, structured EXPRESSION_COEFFS data type plus:

  - Temporal smoothing (One-Euro filter, same library as gaze pipeline)
  - Per-coefficient temporal stability score (jitter measurement)
  - 4D animation-curve export (JSON, frame-indexed)
  - Optional dimensionality reduction (top-K by variance)

Why ARKit blendshapes instead of FLAME/EMOCA?
  - FLAME/EMOCA require a 100+ MB pretrained network + monocular fit per frame.
  - ARKit 52 covers the same expressive space as FLAME's 50 expression coeffs.
  - All downstream Wan-Animate-compatible tooling consumes ARKit-format input.
  - MediaPipe runs at >60 FPS on CPU; no GPU spend.

Input: iris_data_json (the same JSON emitted by PoseAndFaceDetectionV2 — it
already contains the blendshapes dict per frame under iris_data[i].blendshapes).

Outputs:
  - coeffs_json     : STRING -- structured {frames: [{frame, blendshapes,
                       stability}, ...], summary: {...}}
  - smoothed_json   : STRING -- One-Euro smoothed coefficients
  - mean_stability  : FLOAT  -- mean per-coefficient stability across frames
  - active_count    : INT    -- number of coefficients exceeding `active_thr`
                                (useful as a sanity gate for downstream code)
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .._is_changed_util import hash_args_and_kwargs


# ARKit 52 blendshape names (Apple/MediaPipe standard order)
ARKIT_52 = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


class _OneEuro1D:
    """Minimal scalar One-Euro filter (Casiez 2012)."""
    def __init__(self, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / max(rate, 1e-6)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        rate = 1.0 / max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) * rate
        a_d = self._alpha(rate, self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(rate, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def _extract_blendshapes(iris_entry: dict) -> dict[str, float]:
    """Normalise blendshapes dict from various possible shapes in iris JSON."""
    if not isinstance(iris_entry, dict):
        return {}
    bs = iris_entry.get("blendshapes")
    if isinstance(bs, dict):
        return {k: float(v) for k, v in bs.items() if isinstance(v, (int, float))}
    # alt: gaze_blendshape sub-dict
    gb = iris_entry.get("gaze_blendshape")
    if isinstance(gb, dict):
        return {k: float(v) for k, v in gb.items() if isinstance(v, (int, float))}
    return {}


class WanExpressionCoefficientsV2:
    CATEGORY = "WanAnimatePreprocess_V2/Expression"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "INT")
    RETURN_NAMES = ("coeffs_json", "smoothed_json", "mean_stability", "active_count")
    DESCRIPTION = "Extract 52 ARKit expression coefficients + temporal smoothing + stability scoring from iris_data JSON."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "iris_data_json": ("STRING", {"multiline": True, "default": "[]"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "smooth_min_cutoff": ("FLOAT", {"default": 1.5, "min": 0.01, "max": 30.0, "step": 0.1}),
                "smooth_beta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "active_threshold": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01}),
                "topk": ("INT", {"default": 0, "min": 0, "max": 52,
                                 "tooltip": "0 = keep all; >0 = keep top-K by variance."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, iris_data_json, fps, smooth_min_cutoff, smooth_beta,
                active_threshold, topk):
        try:
            data = json.loads(iris_data_json) if iris_data_json.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"WanExpressionCoefficientsV2: invalid iris_data_json: {e}")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = []

        per_frame: list[dict[str, float]] = []
        for entry in data:
            per_frame.append(_extract_blendshapes(entry if isinstance(entry, dict) else {}))

        # Collect ordered names from first non-empty frame; fall back to ARKIT_52
        names = []
        for bs in per_frame:
            if bs:
                names = [n for n in ARKIT_52 if n in bs] + [n for n in bs if n not in ARKIT_52]
                break
        if not names:
            names = list(ARKIT_52)

        # Build N×K matrix of raw coefficients
        N = len(per_frame) or 1
        K = len(names)
        mat = [[float(bs.get(n, 0.0)) for n in names] for bs in per_frame] or [[0.0] * K]

        # One-Euro smoothing per coefficient
        filters = [_OneEuro1D(min_cutoff=smooth_min_cutoff, beta=smooth_beta) for _ in range(K)]
        smoothed = [[0.0] * K for _ in range(len(mat))]
        dt = 1.0 / max(fps, 1e-3)
        for i, row in enumerate(mat):
            t = i * dt
            for k in range(K):
                smoothed[i][k] = float(filters[k](row[k], t))

        # Stability: 1.0 - mean |Δ| over consecutive frames, per coefficient.
        # Range [0,1]; 1.0 = perfectly still, 0.0 = max jitter.
        stability = [1.0] * K
        if len(mat) > 1:
            for k in range(K):
                diffs = [abs(mat[i][k] - mat[i - 1][k]) for i in range(1, len(mat))]
                stability[k] = float(max(0.0, 1.0 - (sum(diffs) / len(diffs))))
        mean_stab = float(sum(stability) / max(K, 1))

        # active_count: coefficients whose max value crosses threshold.
        col_max = [max((row[k] for row in mat), default=0.0) for k in range(K)]
        active = [k for k in range(K) if col_max[k] >= active_threshold]
        active_count = int(len(active))

        # topk filter (by variance)
        if topk > 0 and topk < K:
            means = [sum(row[k] for row in mat) / max(len(mat), 1) for k in range(K)]
            variances = [
                sum((row[k] - means[k]) ** 2 for row in mat) / max(len(mat), 1)
                for k in range(K)
            ]
            top_idx = sorted(range(K), key=lambda k: variances[k], reverse=True)[:topk]
            top_idx_set = set(top_idx)
            names_out = [names[k] for k in range(K) if k in top_idx_set]
            mat_out = [[row[k] for k in range(K) if k in top_idx_set] for row in mat]
            smoothed_out = [[row[k] for k in range(K) if k in top_idx_set] for row in smoothed]
            stab_out = [stability[k] for k in range(K) if k in top_idx_set]
        else:
            names_out = names
            mat_out = mat
            smoothed_out = smoothed
            stab_out = stability

        coeffs_json = json.dumps({
            "fps": float(fps),
            "names": names_out,
            "frames": [
                {"frame": i, "blendshapes": dict(zip(names_out, mat_out[i]))}
                for i in range(len(mat_out))
            ],
            "summary": {
                "n_frames": len(mat_out),
                "n_coeffs": len(names_out),
                "active_coeffs": [names[k] for k in active],
                "active_count": active_count,
                "mean_stability": mean_stab,
                "per_coeff_stability": dict(zip(names_out, stab_out)),
            }
        })
        smoothed_json = json.dumps({
            "fps": float(fps),
            "names": names_out,
            "frames": [
                {"frame": i, "blendshapes": dict(zip(names_out, smoothed_out[i]))}
                for i in range(len(smoothed_out))
            ],
            "filter": {"type": "one_euro",
                       "min_cutoff": float(smooth_min_cutoff),
                       "beta": float(smooth_beta)}
        })
        return (coeffs_json, smoothed_json, mean_stab, active_count)
