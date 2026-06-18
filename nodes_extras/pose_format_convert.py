"""WanPoseFormatConvertV2 — re-emit pose_data in another skeleton format.

The V2 preprocessor stores body keypoints in OpenPose-18 (BODY_18 / COCO)
format inside each meta's ``keypoints_body``.  Downstream consumers often
expect one of:

  * OpenPose **BODY-25** (the modern OpenPose default, with mid-hip and
    six toe/heel landmarks).
  * **COCO-17** (the topology produced by most modern pose detectors,
    used by HRNet / ViTPose-COCO / RTMPose / Sapiens).
  * **MediaPipe-33** (Google MediaPipe Pose Landmarker — 33 landmarks
    with eyes, ears, mouth, hands, and feet).

This node converts the keypoints WITHOUT re-detecting them: the
geometric mapping is direct because OpenPose-18 is a strict subset of
both BODY-25 and MediaPipe-33, and is rearrangeable into COCO-17.
Landmarks the source can't supply (toes, mouth corners, inner-eye
points, …) are written as ``None`` so downstream nodes can decide
whether to discard or impute them.

Outputs
-------
1.  ``pose_data`` — same POSEDATA bundle, but every meta carries an
    additional ``keypoints_body_<format>`` key alongside the original
    ``keypoints_body``.  A bundle-level ``body_format`` string records
    the conversion that was performed.
2.  ``pose_keypoint`` — a ComfyUI ``POSE_KEYPOINT`` list (one dict per
    frame, OpenPose JSON convention) so the result plugs straight into
    ``Openpose Pose`` ControlNet preprocessors that expect the BODY-18
    or BODY-25 layout.
3.  ``info`` — human-readable summary.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Optional

import numpy as np

from .._is_changed_util import hash_args_and_kwargs

log = logging.getLogger(__name__)


# ── OpenPose-18 index helpers ────────────────────────────────────────
_OP18 = {
    "nose": 0, "neck": 1,
    "rsho": 2, "relb": 3, "rwri": 4,
    "lsho": 5, "lelb": 6, "lwri": 7,
    "rhip": 8, "rkne": 9, "rank": 10,
    "lhip": 11, "lkne": 12, "lank": 13,
    "reye": 14, "leye": 15,
    "rear": 16, "lear": 17,
}


# ── Mapping tables ───────────────────────────────────────────────────
# Each list entry is either ``int`` (= OP18 source index) or ``None``
# (= no source available; the resulting joint is left missing).
_OP18_TO_BODY25: list[Optional[int]] = [
    _OP18["nose"], _OP18["neck"],
    _OP18["rsho"], _OP18["relb"], _OP18["rwri"],
    _OP18["lsho"], _OP18["lelb"], _OP18["lwri"],
    None,                                       # 8 = midhip (computed)
    _OP18["rhip"], _OP18["rkne"], _OP18["rank"],
    _OP18["lhip"], _OP18["lkne"], _OP18["lank"],
    _OP18["reye"], _OP18["leye"],
    _OP18["rear"], _OP18["lear"],
    None, None, None,                           # 19-21 = L_bigtoe / L_smltoe / L_heel
    None, None, None,                           # 22-24 = R_bigtoe / R_smltoe / R_heel
]
_BODY25_MIDHIP_IDX  = 8
_BODY25_RHIP_SRC    = _OP18["rhip"]
_BODY25_LHIP_SRC    = _OP18["lhip"]

# COCO-17 (standard):
#   0=nose 1=l_eye 2=r_eye 3=l_ear 4=r_ear
#   5=l_sho 6=r_sho 7=l_elb 8=r_elb 9=l_wri 10=r_wri
#   11=l_hip 12=r_hip 13=l_kne 14=r_kne 15=l_ank 16=r_ank
_OP18_TO_COCO17: list[Optional[int]] = [
    _OP18["nose"],
    _OP18["leye"], _OP18["reye"],
    _OP18["lear"], _OP18["rear"],
    _OP18["lsho"], _OP18["rsho"],
    _OP18["lelb"], _OP18["relb"],
    _OP18["lwri"], _OP18["rwri"],
    _OP18["lhip"], _OP18["rhip"],
    _OP18["lkne"], _OP18["rkne"],
    _OP18["lank"], _OP18["rank"],
]

# MediaPipe Pose Landmarker (33 landmarks).  We only fill the joints
# that have a clean equivalent in OpenPose-18; mouth/iris/inner-eye/
# foot detail is left None.  See:
#   https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_OP18_TO_MP33: list[Optional[int]] = [None] * 33
_OP18_TO_MP33[0]  = _OP18["nose"]
_OP18_TO_MP33[2]  = _OP18["leye"]
_OP18_TO_MP33[5]  = _OP18["reye"]
_OP18_TO_MP33[7]  = _OP18["lear"]
_OP18_TO_MP33[8]  = _OP18["rear"]
_OP18_TO_MP33[11] = _OP18["lsho"]
_OP18_TO_MP33[12] = _OP18["rsho"]
_OP18_TO_MP33[13] = _OP18["lelb"]
_OP18_TO_MP33[14] = _OP18["relb"]
_OP18_TO_MP33[15] = _OP18["lwri"]
_OP18_TO_MP33[16] = _OP18["rwri"]
_OP18_TO_MP33[23] = _OP18["lhip"]
_OP18_TO_MP33[24] = _OP18["rhip"]
_OP18_TO_MP33[25] = _OP18["lkne"]
_OP18_TO_MP33[26] = _OP18["rkne"]
_OP18_TO_MP33[27] = _OP18["lank"]
_OP18_TO_MP33[28] = _OP18["rank"]


_FORMAT_TARGETS = {
    "body_25":      (25, _OP18_TO_BODY25),
    "coco_17":      (17, _OP18_TO_COCO17),
    "mediapipe_33": (33, _OP18_TO_MP33),
    "openpose_18":  (18, list(range(18))),  # passthrough
}


# ── Conversion helpers ───────────────────────────────────────────────
def _read_op18_xyc(src) -> np.ndarray:
    """Normalise the OP18 ``keypoints_body`` list/array to an (18, 3)
    float array of ``[x, y, conf]`` rows.  Missing joints become NaN."""
    out = np.full((18, 3), np.nan, dtype=np.float32)
    if src is None:
        return out
    if isinstance(src, np.ndarray) and src.ndim == 2 and src.shape[0] >= 18:
        m = min(18, src.shape[0])
        out[:m, 0] = src[:m, 0]
        out[:m, 1] = src[:m, 1]
        if src.shape[1] >= 3:
            out[:m, 2] = src[:m, 2]
        else:
            out[:m, 2] = 1.0
        return out
    if isinstance(src, (list, tuple)):
        for i in range(min(18, len(src))):
            v = src[i]
            if v is None:
                continue
            try:
                out[i, 0] = float(v[0])
                out[i, 1] = float(v[1])
                out[i, 2] = float(v[2]) if len(v) > 2 else 1.0
            except (TypeError, ValueError, IndexError):
                continue
    return out


def _convert_one(op18_xyc: np.ndarray, target: str) -> list:
    """Build the destination joint list for one frame.  Each entry is
    ``None`` for a missing joint or ``[x, y, conf]`` for a filled one.
    All coordinates stay in OP18's image-normalised [0, 1] space."""
    n_dst, table = _FORMAT_TARGETS[target]
    out: list = [None] * n_dst
    for j_dst, j_src in enumerate(table):
        if j_src is None:
            continue
        row = op18_xyc[j_src]
        if not (np.isfinite(row[0]) and np.isfinite(row[1])):
            continue
        out[j_dst] = [float(row[0]), float(row[1]), float(row[2]) if np.isfinite(row[2]) else 1.0]
    # Special-case: BODY-25 mid-hip = midpoint of OP18 rhip + lhip.
    if target == "body_25":
        rh = op18_xyc[_BODY25_RHIP_SRC]
        lh = op18_xyc[_BODY25_LHIP_SRC]
        if np.isfinite(rh[0]) and np.isfinite(lh[0]):
            mx = 0.5 * (float(rh[0]) + float(lh[0]))
            my = 0.5 * (float(rh[1]) + float(lh[1]))
            mc = 0.5 * (float(rh[2] if np.isfinite(rh[2]) else 1.0)
                      + float(lh[2] if np.isfinite(lh[2]) else 1.0))
            out[_BODY25_MIDHIP_IDX] = [mx, my, mc]
    return out


def _flatten_for_openpose_json(joints_xyc: list, W: int, H: int) -> list:
    """Turn a per-joint list (``None`` or ``[xN, yN, conf]``, OP18-normalised
    coords) into the flat ``[x, y, conf, ...]`` array used by OpenPose JSON.
    Missing joints become ``0, 0, 0`` per OpenPose convention.  Output is in
    PIXEL space because that's what every ComfyUI OpenPose consumer expects."""
    flat: list[float] = []
    for v in joints_xyc:
        if v is None:
            flat.extend([0.0, 0.0, 0.0])
        else:
            flat.extend([float(v[0]) * W, float(v[1]) * H, float(v[2])])
    return flat


def _face_pixels_for_openpose(meta: dict, W: int, H: int) -> list:
    """Convert face keypoints (iBUG-68, normalised [0,1]) to OpenPose
    pixel-space flat list.  OpenPose JSON wants 70 triples; iBUG-68 gives
    68, so we pad with two zero entries for compatibility."""
    arr = meta.get("keypoints_face")
    if arr is None:
        return [0.0] * (70 * 3)
    try:
        a = np.asarray(arr, dtype=np.float32)
    except Exception:
        return [0.0] * (70 * 3)
    if a.ndim != 2 or a.shape[0] < 1:
        return [0.0] * (70 * 3)
    flat: list[float] = []
    for i in range(min(68, a.shape[0])):
        x = float(a[i, 0]) * W; y = float(a[i, 1]) * H
        c = float(a[i, 2]) if a.shape[1] > 2 and np.isfinite(a[i, 2]) else 1.0
        flat.extend([x, y, c])
    while len(flat) < 70 * 3:
        flat.extend([0.0, 0.0, 0.0])
    return flat[: 70 * 3]


def _hand_pixels_for_openpose(meta: dict, side: str, W: int, H: int) -> list:
    """Convert hand keypoints (21 joints, normalised [0,1]) to OpenPose
    pixel-space flat list."""
    key = "keypoints_left_hand" if side == "left" else "keypoints_right_hand"
    arr = meta.get(key)
    if arr is None:
        return [0.0] * (21 * 3)
    try:
        a = np.asarray(arr, dtype=np.float32)
    except Exception:
        return [0.0] * (21 * 3)
    if a.ndim != 2 or a.shape[0] < 1:
        return [0.0] * (21 * 3)
    flat: list[float] = []
    for i in range(min(21, a.shape[0])):
        x = float(a[i, 0]) * W; y = float(a[i, 1]) * H
        c = float(a[i, 2]) if a.shape[1] > 2 and np.isfinite(a[i, 2]) else 1.0
        flat.extend([x, y, c])
    while len(flat) < 21 * 3:
        flat.extend([0.0, 0.0, 0.0])
    return flat[: 21 * 3]


# ── Node ─────────────────────────────────────────────────────────────
class WanPoseFormatConvertV2:
    CATEGORY    = "WanAnimatePreprocessV2/extras"
    DESCRIPTION = (
        "Convert POSEDATA's OpenPose-18 body keypoints to another skeleton "
        "format (BODY-25, COCO-17, or MediaPipe-33). Also emits a standard "
        "ComfyUI POSE_KEYPOINT JSON so the result plugs directly into "
        "OpenPose ControlNet preprocessors."
    )
    RETURN_TYPES = ("POSEDATA", "POSE_KEYPOINT", "STRING")
    RETURN_NAMES = ("pose_data", "pose_keypoint", "info")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {
                    "tooltip": "POSEDATA bundle from the V2 preprocessor (OpenPose-18 body).",
                }),
                "target_format": (
                    ["body_25", "coco_17", "mediapipe_33", "openpose_18"],
                    {"default": "body_25",
                     "tooltip": "Destination skeleton topology. openpose_18 = passthrough."},
                ),
            },
            "optional": {
                "use_metas": (["original", "edited"], {
                    "default": "edited",
                    "tooltip": "Whether to read from pose_metas (edited) or pose_metas_original.",
                }),
                "emit_face":      ("BOOLEAN", {"default": True}),
                "emit_hands":     ("BOOLEAN", {"default": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    def run(self, pose_data,
            target_format: str,
            use_metas: str = "edited",
            emit_face: bool = True,
            emit_hands: bool = True):
        if target_format not in _FORMAT_TARGETS:
            raise ValueError(f"Unknown target_format {target_format!r}")

        out_bundle = dict(pose_data)
        primary_key = "pose_metas" if use_metas == "edited" else "pose_metas_original"
        target_metas = pose_data.get(primary_key) or pose_data.get("pose_metas_original") or pose_data.get("pose_metas")
        if not target_metas:
            raise ValueError("pose_data has no 'pose_metas' / 'pose_metas_original'.")

        n_dst, _ = _FORMAT_TARGETS[target_format]
        new_metas = []
        n_filled = 0
        n_missing = 0
        pose_keypoint_list: list[dict] = []

        for f_idx, meta in enumerate(target_metas):
            new_meta = deepcopy(meta) if isinstance(meta, dict) else {}
            op18 = _read_op18_xyc(new_meta.get("keypoints_body"))
            converted = _convert_one(op18, target_format)
            new_meta[f"keypoints_body_{target_format}"] = converted
            new_meta["body_format"] = target_format
            for v in converted:
                if v is None:
                    n_missing += 1
                else:
                    n_filled += 1
            new_metas.append(new_meta)

            # Build the OpenPose-style POSE_KEYPOINT entry (always emitted —
            # for COCO-17 we still flatten so the user can repurpose it).
            W = int(new_meta.get("width") or 0)
            H = int(new_meta.get("height") or 0)
            if W <= 0 or H <= 0:
                W = W or 1024
                H = H or 1024
            person: dict = {
                "pose_keypoints_2d": _flatten_for_openpose_json(converted, W, H),
            }
            if emit_face:
                person["face_keypoints_2d"] = _face_pixels_for_openpose(new_meta, W, H)
            if emit_hands:
                person["hand_left_keypoints_2d"]  = _hand_pixels_for_openpose(new_meta, "left",  W, H)
                person["hand_right_keypoints_2d"] = _hand_pixels_for_openpose(new_meta, "right", W, H)
            pose_keypoint_list.append({
                "people": [person],
                "canvas_width":  W,
                "canvas_height": H,
            })

        # Mirror into the same key the editor preview consumes.  Replace
        # the active meta list so downstream consumers that look at
        # ``keypoints_body`` see the canonical OP18 still; new key holds
        # the converted topology.  ``pose_data`` stays a superset.
        if primary_key == "pose_metas":
            out_bundle["pose_metas"] = new_metas
        else:
            out_bundle["pose_metas_original"] = new_metas
        out_bundle["body_format"] = target_format
        out_bundle["body_format_n_joints"] = n_dst

        n_frames = len(new_metas)
        ratio = n_filled / max(1, n_filled + n_missing)
        info = (
            f"WanPoseFormatConvertV2: {n_frames} frames -> "
            f"{target_format} ({n_dst} joints/frame) | "
            f"filled={n_filled} missing={n_missing} ({ratio*100:.1f}% coverage) | "
            f"face={'y' if emit_face else 'n'} hands={'y' if emit_hands else 'n'} | "
            f"read={primary_key}"
        )
        log.info(info)
        return (out_bundle, pose_keypoint_list, info)
