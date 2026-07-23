"""WanExpressionCriticV2 — Wan-Animate spec Section 3.1: the closed-loop
fidelity critic.

Everything in the Wan-Animate fidelity spec up to this point is open-loop:
apply a fix or a setting, generate, eyeball the result, guess whether it
helped. This node closes the loop instead.

Given TWO ``coeffs_json`` blobs (the exact output of
``WanExpressionCoefficientsV2``, run once on the SOURCE driving video and
once on the GENERATED Wan-Animate output — same node, same schema, just two
different videos through the same PoseAndFaceDetectionV2 -> Expression
Coefficients pipeline), this computes a per-frame, per-AU numeric error
signal instead of "eyeballing frames":

  * per-AU mean absolute error across the whole clip (which blendshapes does
    the encoder measurably flatten?) — feeds spec 3.2's adaptive, non-uniform
    au_amplify bias (push harder on the worst-tracked AUs).
  * per-frame overall error curve, bucketed into segments matching Wan-
    Animate's ~78-frame splice window — feeds spec 3.4/3.5 (AU-bracketing /
    surgical segment regeneration: identify exactly which segments have the
    worst AU-tracking error, regenerate ONLY those).

This node does not run a sampler and does not require GPU — it is pure
Python/JSON comparison, so it is safe to run on any machine that already has
the two coefficient exports.
"""

from __future__ import annotations

import json
import statistics

from .._is_changed_util import hash_args_and_kwargs


def _parse_coeffs(coeffs_json: str) -> tuple[list[str], list[dict]]:
    """Return (names, frames) from a WanExpressionCoefficientsV2 coeffs_json
    blob. Tolerant of empty/malformed input — returns ([], []) rather than
    raising, so a missing/failed upstream export degrades to an empty (and
    therefore maximally-honest, zero-confidence) critic report."""
    try:
        data = json.loads(coeffs_json) if coeffs_json and coeffs_json.strip() else {}
    except json.JSONDecodeError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    names = data.get("names") or []
    frames = data.get("frames") or []
    if not isinstance(names, list) or not isinstance(frames, list):
        return [], []
    return names, frames


class WanExpressionCriticV2:
    CATEGORY = "WanAnimatePreprocess_V2/Expression"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("critic_report_json", "worst_aus_csv", "overall_mae")
    DESCRIPTION = (
        "Wan-Animate spec 3.1 closed-loop critic. Compares a SOURCE driving "
        "video's measured ARKit-52 blendshapes against a GENERATED Wan-Animate "
        "output's measured blendshapes (both from WanExpressionCoefficientsV2), "
        "frame-aligned, and reports per-AU + per-segment mean absolute error — "
        "a numeric signal for which expressions the face encoder measurably "
        "flattens, instead of eyeballing frames."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_coeffs_json": ("STRING", {"multiline": True, "default": "{}",
                    "tooltip": "WanExpressionCoefficientsV2.coeffs_json run on the SOURCE driving video."}),
                "generated_coeffs_json": ("STRING", {"multiline": True, "default": "{}",
                    "tooltip": "WanExpressionCoefficientsV2.coeffs_json run on the GENERATED Wan-Animate output."}),
                "segment_length": ("INT", {"default": 77, "min": 1, "max": 100000,
                    "tooltip": "Frames per segment for the worst-segment breakdown — match WanVideoAnimateEmbeds.frame_window_size (default 77) so segments here line up with Wan-Animate's own splice boundaries (spec 2.5/3.5)."}),
                "top_k_aus": ("INT", {"default": 10, "min": 1, "max": 52,
                    "tooltip": "How many worst-tracked AUs to report, worst-first."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, source_coeffs_json, generated_coeffs_json, segment_length, top_k_aus):
        src_names, src_frames = _parse_coeffs(source_coeffs_json)
        gen_names, gen_frames = _parse_coeffs(generated_coeffs_json)

        # Union of AU names seen on either side — an AU absent from one side
        # is treated as 0.0 there (a real, reportable discrepancy, not hidden).
        names = sorted(set(src_names) | set(gen_names))
        n = min(len(src_frames), len(gen_frames))
        truncated = len(src_frames) != len(gen_frames)

        per_au_abs_err = {name: [] for name in names}
        per_frame_err: list[float] = []

        for i in range(n):
            src_bs = (src_frames[i] or {}).get("blendshapes", {}) if isinstance(src_frames[i], dict) else {}
            gen_bs = (gen_frames[i] or {}).get("blendshapes", {}) if isinstance(gen_frames[i], dict) else {}
            frame_errs = []
            for name in names:
                e = abs(float(src_bs.get(name, 0.0)) - float(gen_bs.get(name, 0.0)))
                per_au_abs_err[name].append(e)
                frame_errs.append(e)
            per_frame_err.append(float(sum(frame_errs) / len(frame_errs)) if frame_errs else 0.0)

        per_au_mae = {
            name: (float(sum(errs) / len(errs)) if errs else 0.0)
            for name, errs in per_au_abs_err.items()
        }
        overall_mae = float(statistics.fmean(per_au_mae.values())) if per_au_mae else 0.0

        worst_aus = sorted(per_au_mae.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_k_aus))]

        # Segment breakdown (spec 3.5: which segments have the worst tracking).
        seg_len = max(1, int(segment_length))
        segments = []
        for s0 in range(0, n, seg_len):
            s1 = min(n, s0 + seg_len)
            seg_errs = per_frame_err[s0:s1]
            segments.append({
                "start_frame": s0,
                "end_frame": s1 - 1,
                "n_frames": s1 - s0,
                "mean_error": float(sum(seg_errs) / len(seg_errs)) if seg_errs else 0.0,
            })
        segments_worst_first = sorted(segments, key=lambda s: s["mean_error"], reverse=True)

        report = {
            "n_frames_compared": n,
            "frame_count_mismatch": truncated,
            "source_n_frames": len(src_frames),
            "generated_n_frames": len(gen_frames),
            "overall_mae": overall_mae,
            "per_au_mae": per_au_mae,
            "worst_aus": [{"name": name, "mae": mae} for name, mae in worst_aus],
            "per_frame_error": per_frame_err,
            "segments": segments,
            "segments_worst_first": segments_worst_first,
        }
        if truncated:
            report["note"] = (
                f"source has {len(src_frames)} frames, generated has {len(gen_frames)} — "
                f"compared the first {n} (frame-index truncation, no re-alignment attempted)."
            )

        worst_aus_csv = "\n".join(f"{name},{mae:.4f}" for name, mae in worst_aus)
        return (json.dumps(report), worst_aus_csv, overall_mae)


NODE_CLASS_MAPPINGS = {
    "WanExpressionCriticV2": WanExpressionCriticV2,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanExpressionCriticV2": "Wan Expression Critic — Source vs Generated AU Error (V2)",
}
