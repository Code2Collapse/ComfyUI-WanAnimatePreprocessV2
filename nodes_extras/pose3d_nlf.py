"""
pose3d_nlf.py — NLF 3D-pose refinement for Wan 2.2 Animate.

Refines a ``POSEDATA`` bundle (OpenPose-18 body from ViTPose) using **Neural
Localizer Fields** (NLF, NeurIPS'24, isarandi/nlf): runs NLF on the frames to get
3D-consistent body joints, maps the SMPL-X body joints to OpenPose-18, and blends
the temporally-stable 2D projection into ``keypoints_body`` — fixing the
per-frame jitter / occlusion failure / temporal inconsistency that raw ViTPose
suffers (documented in the Wan-Animate paper). The output is the SAME POSEDATA
schema, so it drops straight into the existing Wan-Animate pose path.

Why NLF (not SMPLest-X): NLF is the 3D backbone used by **SCAIL** (CVPR'26
"3D-Consistent Pose Representations"); it outputs 3D directly, runs on 8 GB, and
does NOT need license-gated SMPL-X model files for inference (the TorchScript
model ships its own). SCAIL explicitly rejects SMPL for "identity leakage."

The TorchScript model auto-downloads to ``ComfyUI/models/nlf`` on first use.

Credits: NLF — Sárándi & Pons-Moll, NeurIPS 2024 (CC BY-NC 4.0, research use).
This node is a standalone ComfyUI wrapper; it imports nothing from third_party.

NOTE / TODO (finalize with a live NLF run): only the BODY joints
(neck…ankles) are 3D-refined here, mapped from SMPL-X canonical body indices
which are stable. nose/eyes/ears keep their ViTPose values until the exact
SMPL-X→openpose face-landmark indices are confirmed against live NLF output.
"""

from __future__ import annotations

import logging
import math
import os

import numpy as np

log = logging.getLogger(__name__)

# OpenPose-18 ← SMPL-X canonical body joint indices (confident subset).
# OP18:  0 nose 1 neck 2 rsho 3 relb 4 rwri 5 lsho 6 lelb 7 lwri
#        8 rhip 9 rkne 10 rank 11 lhip 12 lkne 13 lank 14 reye 15 leye 16 rear 17 lear
# SMPL-X body: 1 l_hip 2 r_hip 4 l_knee 5 r_knee 7 l_ankle 8 r_ankle
#              12 neck 16 l_shoulder 17 r_shoulder 18 l_elbow 19 r_elbow 20 l_wrist 21 r_wrist
_OP18_FROM_SMPLX = {
    1: 12,            # neck
    2: 17, 3: 19, 4: 21,   # right shoulder/elbow/wrist
    5: 16, 6: 18, 7: 20,   # left shoulder/elbow/wrist
    8: 2, 9: 5, 10: 8,     # right hip/knee/ankle
    11: 1, 12: 4, 13: 7,   # left hip/knee/ankle
}
# OP18 joints we 3D-refine (the rest keep ViTPose).
_REFINED_OP18 = sorted(_OP18_FROM_SMPLX.keys())

# Default NLF TorchScript (auto-download). Overridable via the widget.
_NLF_FILENAME = "nlf_l_multi.torchscript"
_NLF_URLS = [
    "https://github.com/isarandi/nlf/releases/download/v0.3.2/nlf_l_multi_0.3.2.torchscript",
    "https://github.com/isarandi/nlf/releases/download/v0.2.0/nlf_l_multi_0.2.0.torchscript",
]


def _models_dir():
    try:
        import folder_paths

        d = os.path.join(folder_paths.models_dir, "nlf")
    except Exception:  # noqa: BLE001
        d = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "nlf")
    os.makedirs(d, exist_ok=True)
    return d


def _list_nlf_models():
    d = _models_dir()
    out = [f for f in os.listdir(d) if f.endswith(".torchscript")]
    return out or [_NLF_FILENAME]


def _ensure_nlf(name):
    """Return a path to the NLF torchscript, downloading it if missing."""
    path = os.path.join(_models_dir(), name)
    if os.path.exists(path):
        return path
    import urllib.request

    for url in _NLF_URLS:
        try:
            log.info("[pose3d_nlf] downloading NLF model: %s", url)
            urllib.request.urlretrieve(url, path)
            if os.path.getsize(path) > 1_000_000:
                return path
        except Exception as exc:  # noqa: BLE001
            log.warning("[pose3d_nlf] download failed (%s): %s", url, exc)
    raise FileNotFoundError(
        f"NLF model '{name}' not found in models/nlf and auto-download failed. "
        f"Place a TorchScript NLF model there manually (see github.com/isarandi/nlf)."
    )


_NLF_CACHE = {}


def _load_nlf(name, device):
    import torch
    # NLF's TorchScript calls torchvision::nms — importing torchvision registers
    # those C++ ops, otherwise jit.load fails with "Unknown builtin op:
    # torchvision::nms". Touch ops.nms to force the extension to load.
    import torchvision
    try:
        _ = torchvision.ops.nms
    except Exception:  # noqa: BLE001
        pass

    key = f"{name}|{device}"
    if key in _NLF_CACHE:
        return _NLF_CACHE[key]
    path = _ensure_nlf(name)
    model = torch.jit.load(path, map_location=device).eval()
    _NLF_CACHE[key] = model
    log.info("[pose3d_nlf] NLF loaded on %s: %s", device, path)
    return model


class _OneEuro:
    """Minimal 1-euro filter for 2D keypoint streams."""

    def __init__(self, min_cutoff=1.0, beta=0.1, fps=30.0):
        self.mc = min_cutoff
        self.beta = beta
        self.dt = 1.0 / max(fps, 1e-3)
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) / self.dt
        a_d = self._alpha(1.0, self.dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mc + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class WanPose3DRefineNLFV2:
    """3D pose refinement (NLF) for Wan 2.2 Animate.

    Takes the ViTPose POSEDATA + frames, runs NLF for 3D-consistent body joints,
    and blends the temporally-stable 2D projection into the OpenPose-18 body of
    each frame's primary subject. Re-emits the same POSEDATA bundle.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),
                "images": ("IMAGE",),
                "nlf_model": (_list_nlf_models(),),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "blend_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                             "tooltip": "How strongly NLF's 3D body overrides ViTPose (1=full NLF)."}),
                "temporal_smoothing": ("BOOLEAN", {"default": True}),
                "smoothing_min_cutoff": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 10.0, "step": 0.05}),
                "smoothing_beta": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5.0, "step": 0.01}),
                "occlusion_fill": ("BOOLEAN", {"default": True,
                                               "tooltip": "Where ViTPose confidence is low, trust NLF's 3D projection."}),
            },
        }

    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "info")
    FUNCTION = "refine"
    CATEGORY = "WanAnimatePreprocess/Pose"
    DESCRIPTION = "Refine ViTPose POSEDATA with NLF 3D pose (jitter/occlusion/temporal) — Wan-Animate ready."

    def refine(self, pose_data, images, nlf_model, device, blend_strength,
               temporal_smoothing, smoothing_min_cutoff, smoothing_beta, occlusion_fill):
        from copy import deepcopy

        out = deepcopy(pose_data)
        try:
            import torch

            dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"
            model = _load_nlf(nlf_model, dev)

            # images: ComfyUI IMAGE [F,H,W,3] float 0..1 → uint8 NCHW per frame
            imgs = images
            if not torch.is_tensor(imgs):
                imgs = torch.as_tensor(np.asarray(imgs))
            if imgs.dim() == 3:
                imgs = imgs.unsqueeze(0)
            F, H, W, _ = imgs.shape

            metas = _get_metas(out)
            if not metas:
                return (out, "pose3d_nlf: no per-frame metas in POSEDATA; passthrough.")

            euro_x = _OneEuro(smoothing_min_cutoff, smoothing_beta) if temporal_smoothing else None
            euro_y = _OneEuro(smoothing_min_cutoff, smoothing_beta) if temporal_smoothing else None
            refined_frames = 0

            for fi in range(min(F, len(metas))):
                meta = metas[fi]
                kb = _get_kb(meta)
                if kb is None:
                    continue
                frame_u8 = (imgs[fi].clamp(0, 1) * 255).to(torch.uint8).permute(2, 0, 1).contiguous().unsqueeze(0).to(dev)
                with torch.inference_mode():
                    pred = model.detect_smpl_batched(frame_u8, model_name="smplx")
                j2d, unc = _primary_person(pred)
                if j2d is None:
                    continue

                normalized = bool(np.nanmax(np.abs(kb[:, :2])) <= 1.5)
                refined = kb.copy()
                # Uncertainty gate: skip NLF joints whose 3D uncertainty is an
                # outlier (prevents a bad joint from shooting a limb off-screen).
                unc_thr = float("inf")
                if unc is not None and unc.size:
                    unc_thr = float(np.median(unc) + 2.0 * np.std(unc))
                for op, sx in _OP18_FROM_SMPLX.items():
                    if op >= refined.shape[0] or sx >= j2d.shape[0]:
                        continue
                    px, py = float(j2d[sx, 0]), float(j2d[sx, 1])
                    # SKIP off-frame / uncertain joints entirely. (Do NOT clamp —
                    # clamping pins an off-frame joint to (0,0) and draws a limb to
                    # the corner, the artefact seen in testing.)
                    if px < 0 or px > W or py < 0 or py > H:
                        continue
                    if unc is not None and sx < unc.size and float(unc[sx]) > unc_thr:
                        continue
                    if normalized:
                        px, py = px / max(W, 1), py / max(H, 1)
                    vit_conf = float(refined[op, 2]) if refined.shape[1] > 2 else 1.0
                    w = blend_strength
                    if occlusion_fill and vit_conf < 0.3:
                        w = max(w, 0.9)
                    refined[op, 0] = (1 - w) * refined[op, 0] + w * px
                    refined[op, 1] = (1 - w) * refined[op, 1] + w * py

                if euro_x is not None:
                    refined[_REFINED_OP18, 0] = euro_x(refined[_REFINED_OP18, 0])
                    refined[_REFINED_OP18, 1] = euro_y(refined[_REFINED_OP18, 1])

                _set_kb(meta, refined)
                refined_frames += 1

            info = (f"pose3d_nlf: refined {refined_frames}/{F} frames on {dev} "
                    f"(blend={blend_strength}, smooth={temporal_smoothing}, occl_fill={occlusion_fill}). "
                    f"Body joints 3D-stabilized via NLF; head/face kept from ViTPose.")
            log.info("[pose3d_nlf] %s", info)
            return (out, info)
        except Exception as exc:  # noqa: BLE001
            msg = f"pose3d_nlf: NLF refine unavailable ({exc}); returning original POSEDATA unchanged."
            log.warning("[pose3d_nlf] %s", msg)
            return (out, msg)

    @classmethod
    def IS_CHANGED(cls, pose_data, images, nlf_model, device, blend_strength,
                   temporal_smoothing, smoothing_min_cutoff, smoothing_beta, occlusion_fill):
        import hashlib

        try:
            h = hashlib.md5(np.asarray(images).tobytes()).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            h = "x"
        return f"p3dnlf-{nlf_model}-{device}-{blend_strength}-{temporal_smoothing}-{smoothing_min_cutoff}-{smoothing_beta}-{occlusion_fill}-{h}"


# --------------------------------------------------------------------------- #
# POSEDATA accessors — tolerant of the bundle's exact container shape.
# --------------------------------------------------------------------------- #
def _get_metas(pose_data):
    if isinstance(pose_data, dict):
        # PoseAndFaceDetectionV2 stores the RAW dict metas (with OP18
        # keypoints_body) under 'pose_metas_original'; 'pose_metas' holds
        # retargeted AAPoseMeta objects (not plain dicts) — prefer the originals.
        for k in ("pose_metas_original", "metas", "frames"):
            if isinstance(pose_data.get(k), list):
                return pose_data[k]
    if isinstance(pose_data, list):
        return pose_data
    return []


def _get_kb(meta):
    """OP18 keypoints_body → (N,3) float array; None entries become [0,0,0]."""
    if not isinstance(meta, dict):
        return None
    kb = meta.get("keypoints_body")
    if kb is None:
        return None
    rows = []
    for p in kb:
        if p is None:
            rows.append([0.0, 0.0, 0.0])
            continue
        a = np.asarray(p, dtype=np.float32).reshape(-1)
        if a.shape[0] >= 3:
            rows.append([float(a[0]), float(a[1]), float(a[2])])
        elif a.shape[0] == 2:
            rows.append([float(a[0]), float(a[1]), 1.0])
        else:
            rows.append([0.0, 0.0, 0.0])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 14:
        return None
    return arr


def _set_kb(meta, arr):
    meta["keypoints_body"] = arr


def _primary_person(pred):
    """Return (joints2d[J,2], uncertainty[J]) for the largest detected person."""
    try:
        j2d_list = pred["joints2d"]
        if isinstance(j2d_list, (list, tuple)):
            j2d = j2d_list[0]
        else:
            j2d = j2d_list
        import torch

        if isinstance(j2d, torch.Tensor):
            j2d = j2d.detach().cpu().numpy()
        j2d = np.asarray(j2d, dtype=np.float32)
        if j2d.ndim == 3:  # [people, J, 2]
            if j2d.shape[0] == 0:
                return None, None
            # pick the person with the largest 2D bbox span
            spans = [(p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()) for p in j2d]
            j2d = j2d[int(np.argmax(spans))]
        unc = None
        u = pred.get("joint_uncertainties")
        if u is not None:
            if isinstance(u, (list, tuple)):
                u = u[0]
            if isinstance(u, torch.Tensor):
                u = u.detach().cpu().numpy()
            unc = np.asarray(u, dtype=np.float32).reshape(-1)
        return j2d, unc
    except Exception:  # noqa: BLE001
        return None, None
