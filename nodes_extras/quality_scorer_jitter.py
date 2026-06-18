"""WanQualityScorerJitterV2 — Per-frame quality + jitter metrics for Wan-Animate input.

Aggregates objective per-frame quality signals so users (and downstream
auto-skip / auto-retry logic) can decide whether a clip is usable for
identity-preserving animation:

  - Pose landmark velocity (avg per-keypoint pixel motion between frames)
  - Pose visibility ratio (fraction of keypoints above conf threshold)
  - Face landmark velocity (face-only motion)
  - Expression jitter (blendshape variance over rolling window)
  - Overall aggregate quality score in [0,1] (higher = better)

This is a STATELESS analysis node — does not need any model loaded. Pure
numpy over JSON inputs.

Pose JSON schema:
    [
      {"frame": i,
       "people": [
          { "kps_body": [[x,y,conf], ...],
            "kps_face": [[x,y], ...],
            "kps_lhand": [[x,y,conf], ...],
            "kps_rhand": [[x,y,conf], ...]}
       ]}, ...
    ]
or simply a list of pose dicts (one person, no "people" wrapper).
"""

from __future__ import annotations

import json
import math
from typing import Optional

from .._is_changed_util import hash_args_and_kwargs(kps):
    if not isinstance(kps, list):
        return []
    out = []
    for k in kps:
        if isinstance(k, (list, tuple)) and len(k) >= 2:
            out.append((float(k[0]), float(k[1]),
                        float(k[2]) if len(k) >= 3 else 1.0))
    return out


def _frame_person(frame_entry):
    """Return list of person dicts for the frame."""
    if isinstance(frame_entry, dict):
        if "people" in frame_entry and isinstance(frame_entry["people"], list):
            return frame_entry["people"]
        return [frame_entry]
    return []


def _avg_velocity(prev_kps, curr_kps):
    """Average pixel distance between corresponding keypoints."""
    if not prev_kps or not curr_kps:
        return float("nan")
    n = min(len(prev_kps), len(curr_kps))
    if n == 0:
        return float("nan")
    s = 0.0
    cnt = 0
    for i in range(n):
        p = prev_kps[i]
        c = curr_kps[i]
        cp = p[2] if len(p) >= 3 else 1.0
        cc = c[2] if len(c) >= 3 else 1.0
        if cp <= 0 or cc <= 0:
            continue
        dx = c[0] - p[0]
        dy = c[1] - p[1]
        s += math.sqrt(dx * dx + dy * dy)
        cnt += 1
    return s / cnt if cnt > 0 else float("nan")


def _visibility_ratio(kps, threshold):
    if not kps:
        return 0.0
    valid = sum(1 for k in kps if (len(k) < 3 or k[2] >= threshold))
    return valid / max(len(kps), 1)


def _rolling_variance(seq, window):
    """Population variance over each rolling window."""
    out = []
    for i in range(len(seq)):
        lo = max(0, i - window + 1)
        sub = seq[lo:i + 1]
        if len(sub) < 2:
            out.append(0.0)
            continue
        m = sum(sub) / len(sub)
        out.append(sum((x - m) ** 2 for x in sub) / len(sub))
    return out


class WanQualityScorerJitterV2:
    CATEGORY = "WanAnimatePreprocess_V2/Quality"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("metrics_json", "quality_score",
                    "mean_body_velocity", "mean_face_velocity",
                    "mean_expression_jitter", "bad_frame_count")
    DESCRIPTION = "Per-frame quality and jitter metrics from pose + (optional) expression JSON. Outputs aggregate quality score in [0,1]."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data_json": ("STRING", {"multiline": True, "default": "[]"}),
                "image_diagonal_px": ("FLOAT", {"default": 1500.0, "min": 64.0, "max": 16384.0,
                                                  "step": 1.0,
                                                  "tooltip": "Used to normalise pixel velocities."}),
                "confidence_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_velocity_px": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 1000.0, "step": 1.0,
                                              "tooltip": "Body kp px velocity that would yield score 0."}),
                "expression_window": ("INT", {"default": 8, "min": 2, "max": 60}),
                "bad_velocity_thr_px": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "bad_visibility_thr": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "expression_coeffs_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def execute(self, pose_data_json, image_diagonal_px, confidence_threshold,
                max_velocity_px, expression_window, bad_velocity_thr_px,
                bad_visibility_thr, expression_coeffs_json=""):

        try:
            pose_data = json.loads(pose_data_json) if pose_data_json.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"WanQualityScorerJitterV2: invalid pose_data_json: {e}")
        if not isinstance(pose_data, list):
            pose_data = [pose_data]

        expr_frames = None
        if expression_coeffs_json and expression_coeffs_json.strip():
            try:
                expr = json.loads(expression_coeffs_json)
                if isinstance(expr, dict) and isinstance(expr.get("frames"), list):
                    expr_frames = expr["frames"]
                    expr_names = expr.get("names") or []
                elif isinstance(expr, list):
                    expr_frames = expr
                    expr_names = []
                else:
                    expr_frames = None
            except json.JSONDecodeError:
                expr_frames = None

        per_frame_metrics = []
        prev_body = prev_face = None

        body_vels = []
        face_vels = []
        vis_ratios = []

        for i, fr in enumerate(pose_data):
            persons = _frame_person(fr)
            p = persons[0] if persons else {}
            body = _kps_xy(p.get("kps_body"))
            face = _kps_xy(p.get("kps_face"))
            lhand = _kps_xy(p.get("kps_lhand"))
            rhand = _kps_xy(p.get("kps_rhand"))

            vb = _avg_velocity(prev_body, body) if prev_body else float("nan")
            vf = _avg_velocity(prev_face, face) if prev_face else float("nan")
            vis_body = _visibility_ratio(body, confidence_threshold)
            vis_hands = (_visibility_ratio(lhand, confidence_threshold)
                         + _visibility_ratio(rhand, confidence_threshold)) / 2.0

            body_vels.append(vb)
            face_vels.append(vf)
            vis_ratios.append(vis_body)

            per_frame_metrics.append({
                "frame": i,
                "body_velocity_px": vb if vb == vb else None,
                "face_velocity_px": vf if vf == vf else None,
                "body_visibility": vis_body,
                "hand_visibility": vis_hands,
                "n_body_kps": len(body),
                "n_face_kps": len(face),
            })
            prev_body, prev_face = body, face

        # Expression jitter
        expr_jitter = []
        if expr_frames:
            # Build matrix of blendshape values (frames × coefficients)
            all_names = set()
            for f in expr_frames:
                bs = f.get("blendshapes") if isinstance(f, dict) else None
                if isinstance(bs, dict):
                    all_names.update(bs.keys())
            names = sorted(all_names)
            mat = []
            for f in expr_frames:
                bs = f.get("blendshapes") if isinstance(f, dict) else {}
                mat.append([float(bs.get(n, 0.0)) for n in names])
            # Per-coefficient rolling variance, then mean across coeffs per frame
            if names and mat:
                K = len(names)
                per_coeff_rv = []
                for k in range(K):
                    seq = [row[k] for row in mat]
                    per_coeff_rv.append(_rolling_variance(seq, int(expression_window)))
                T = len(mat)
                for t in range(T):
                    expr_jitter.append(sum(per_coeff_rv[k][t] for k in range(K)) / max(K, 1))
                # write into per_frame_metrics
                for i in range(min(T, len(per_frame_metrics))):
                    per_frame_metrics[i]["expression_jitter"] = expr_jitter[i]

        # Aggregates
        def _mean(xs):
            xs = [x for x in xs if isinstance(x, float) and x == x]
            return float(sum(xs) / len(xs)) if xs else float("nan")

        # Normalize velocities by image diagonal for resolution-independence.
        diag = max(float(image_diagonal_px), 64.0)
        body_vels_norm = [v / diag if v == v else float("nan") for v in body_vels]
        face_vels_norm = [v / diag if v == v else float("nan") for v in face_vels]
        for i, m in enumerate(per_frame_metrics):
            m["body_velocity_norm"] = body_vels_norm[i] if body_vels_norm[i] == body_vels_norm[i] else None
            m["face_velocity_norm"] = face_vels_norm[i] if face_vels_norm[i] == face_vels_norm[i] else None

        m_body = _mean(body_vels)
        m_face = _mean(face_vels)
        m_body_norm = _mean(body_vels_norm)
        m_face_norm = _mean(face_vels_norm)
        m_expr = _mean(expr_jitter) if expr_jitter else 0.0
        m_vis = _mean(vis_ratios)

        # Per-frame bad flag: use normalized velocity for resolution-independent threshold.
        bad_vel_norm = bad_velocity_thr_px / diag
        bad = 0
        for i, m in enumerate(per_frame_metrics):
            bv_n = body_vels_norm[i] if i < len(body_vels_norm) else float("nan")
            if bv_n == bv_n and bv_n > bad_vel_norm:
                m["bad"] = True
                bad += 1
                continue
            if m["body_visibility"] < bad_visibility_thr:
                m["bad"] = True
                bad += 1
                continue
            ej = m.get("expression_jitter")
            if ej is not None and ej > 0.1:
                m["bad"] = True
                bad += 1
                continue
            m["bad"] = False

        # Aggregate score using normalized velocity.
        max_vel_norm = max_velocity_px / diag
        vel_score = 1.0 if not (m_body_norm == m_body_norm) else max(0.0, 1.0 - m_body_norm / max(max_vel_norm, 1e-6))
        vis_score = m_vis if (m_vis == m_vis) else 1.0
        # Expression jitter typical range ~0..0.02; map 0->1, 0.05->0
        expr_score = max(0.0, 1.0 - (m_expr / 0.05)) if expr_jitter else 1.0
        # Face velocity penalty (high face motion = identity drift risk).
        face_vel_score = 1.0 if not (m_face_norm == m_face_norm) else max(0.0, 1.0 - m_face_norm / max(max_vel_norm * 0.5, 1e-6))
        # Weighted: visibility most important, face velocity matters for identity work.
        quality = float(0.4 * vis_score + 0.25 * vel_score + 0.20 * face_vel_score + 0.15 * expr_score)
        quality = max(0.0, min(1.0, quality))

        out_json = json.dumps({
            "frames": per_frame_metrics,
            "summary": {
                "mean_body_velocity_px": m_body if m_body == m_body else None,
                "mean_face_velocity_px": m_face if m_face == m_face else None,
                "mean_body_velocity_norm": m_body_norm if m_body_norm == m_body_norm else None,
                "mean_face_velocity_norm": m_face_norm if m_face_norm == m_face_norm else None,
                "mean_visibility_body": m_vis if m_vis == m_vis else None,
                "mean_expression_jitter": m_expr,
                "bad_frame_count": bad,
                "n_frames": len(per_frame_metrics),
                "quality_score": quality,
                "weights": {"visibility": 0.4, "body_velocity": 0.25,
                            "face_velocity": 0.20, "expression": 0.15},
                "thresholds": {
                    "confidence": float(confidence_threshold),
                    "max_velocity_px": float(max_velocity_px),
                    "bad_velocity_px": float(bad_velocity_thr_px),
                    "bad_visibility": float(bad_visibility_thr),
                },
                "image_diagonal_px": float(image_diagonal_px),
            }
        })
        return (out_json,
                float(quality),
                float(m_body) if m_body == m_body else 0.0,
                float(m_face) if m_face == m_face else 0.0,
                float(m_expr),
                int(bad))
