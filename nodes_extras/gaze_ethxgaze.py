"""WanGazeETHXGazeV2 — replace iris_data gaze with ETH-XGaze inference.

Reference: Zhang et al., ECCV 2020. https://github.com/xucong-zhang/ETH-XGaze

This node consumes a Wan-Animate POSEDATA bundle (which already carries
iBUG-68 ``keypoints_face`` per frame plus an existing MediaPipe-based
``iris_data`` list) and replaces the per-frame ``left_gaze`` / ``right_gaze``
dicts with predictions from the ETH-XGaze ResNet-50 gaze regressor.

Why
---
MediaPipe iris triangulation gives a usable but jittery vector that
collapses on extreme yaw / partial occlusion. ETH-XGaze is the current
academic SOTA for in-the-wild gaze estimation (±2.5° accuracy on the
benchmark), so the resulting iris_data is a strict improvement for the
Wan-Animate gaze conditioning + the UI editor's "current gaze" arrow.

Source of weights
-----------------
The pretrained checkpoint is NOT shipped with the repo. The node looks
for ``epoch_24_ckpt.pth.tar`` in the following locations, in order:

1.  ``ComfyUI/models/ethxgaze/`` (preferred — registered as a
    ``folder_paths`` key so users can drop the file there).
2.  ``third_party/ETH-XGaze/ckpt/``.
3.  Any path the user supplies via the ``checkpoint_path_override`` widget.

Download: https://drive.google.com/file/d/1Jy6rgmUYK-_nGcvldcF4qIK46t6mNk0c
(see the ETH-XGaze repo README for the latest link).

Output
------
The node deep-copies the input POSEDATA and writes new ``left_gaze`` /
``right_gaze`` entries (shape compatible with the existing pipeline:
``yaw_rad``, ``pitch_rad``, ``dx``, ``dy``, ``magnitude_norm``,
``source``).  ETH-XGaze predicts a single face-level gaze vector — we
copy it to both eyes since the model was trained on a normalised face
crop, not per-eye.  Source field is ``"ethxgaze"``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from .._is_changed_util import hash_args_and_kwargs

try:
    import folder_paths  # type: ignore
except Exception:                              # ComfyUI not in path (tests)
    folder_paths = None

log = logging.getLogger(__name__)


# ── third_party / model-path resolution ──────────────────────────────
def _third_party_root() -> Path:
    """``D:\\PROJECT\\Custom_Nodes\\third_party\\ETH-XGaze`` (or wherever
    the repo was cloned). We walk up from this file to ``Custom_Nodes``."""
    here = Path(__file__).resolve()
    # nodes_extras/ -> ComfyUI-WanAnimatePreprocessV2/ -> Custom_Nodes/
    custom_nodes = here.parents[2]
    return custom_nodes / "third_party" / "ETH-XGaze"


def _ensure_eth_xgaze_on_path() -> None:
    p = str(_third_party_root())
    if p not in sys.path:
        sys.path.insert(0, p)


_GAZE_FOLDER_KEY = "ethxgaze"


def _register_gaze_folder() -> None:
    """Register ``ComfyUI/models/ethxgaze/`` with folder_paths so users
    can drop the checkpoint into the standard models tree."""
    if folder_paths is None:
        return
    try:
        # Already registered?
        if _GAZE_FOLDER_KEY in folder_paths.folder_names_and_paths:
            return
        models_dir = Path(folder_paths.models_dir) / "ethxgaze"
        models_dir.mkdir(parents=True, exist_ok=True)
        folder_paths.add_model_folder_path(
            _GAZE_FOLDER_KEY, str(models_dir), is_default=True,
        )
    except Exception as e:
        log.debug("ETH-XGaze folder registration skipped: %s", e)


_register_gaze_folder()


def _list_candidate_ckpts() -> list[Path]:
    """Return all .pth / .pth.tar candidates in the ComfyUI models dir
    and the third_party/ckpt dir, newest first."""
    out: list[Path] = []
    if folder_paths is not None:
        try:
            for root in folder_paths.get_folder_paths(_GAZE_FOLDER_KEY):
                p = Path(root)
                if not p.exists():
                    continue
                for f in p.iterdir():
                    if f.suffix in (".pth", ".tar") or f.name.endswith(".pth.tar"):
                        out.append(f)
        except Exception:
            pass
    tp_ckpt = _third_party_root() / "ckpt"
    if tp_ckpt.exists():
        for f in tp_ckpt.iterdir():
            if f.suffix in (".pth", ".tar") or f.name.endswith(".pth.tar"):
                out.append(f)
    # dedupe + sort by mtime
    seen, uniq = set(), []
    for f in out:
        rp = str(f.resolve())
        if rp in seen:
            continue
        seen.add(rp); uniq.append(f)
    uniq.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return uniq


def _resolve_checkpoint(override: str = "") -> Path:
    if override and override.strip():
        p = Path(override.strip())
        if p.is_file():
            return p
        log.warning("ethxgaze: override path %s not found, falling back", p)
    cands = _list_candidate_ckpts()
    if cands:
        return cands[0]
    raise FileNotFoundError(
        "ETH-XGaze checkpoint not found. Drop `epoch_24_ckpt.pth.tar` into "
        f"`ComfyUI/models/{_GAZE_FOLDER_KEY}/` or `third_party/ETH-XGaze/ckpt/`. "
        "Get it from https://github.com/xucong-zhang/ETH-XGaze (README link)."
    )


def _list_ckpt_choices() -> list[str]:
    cands = _list_candidate_ckpts()
    if not cands:
        return ["<none — drop epoch_24_ckpt.pth.tar in models/ethxgaze/>"]
    # Use relative-ish names so the combo widget is readable.
    return [str(p) for p in cands]


# ── face_model.txt — 68×3 generic 3D landmarks (mm) ──────────────────
def _load_face_model_68() -> Optional[np.ndarray]:
    fm = _third_party_root() / "face_model.txt"
    if not fm.is_file():
        return None
    try:
        arr = np.loadtxt(str(fm))
        if arr.shape == (68, 3):
            return arr.astype(np.float64)
    except Exception as e:
        log.warning("ethxgaze: failed to load face_model.txt: %s", e)
    return None


# Subset of face_model rows used for solvePnP (mirrors demo.py exactly).
# face_model_load[20,23,26,29,15,19] correspond to:
#   right-eye outer (36), right-eye inner (39), left-eye inner (42),
#   left-eye outer (45), nose-left  (31), nose-right (35) in iBUG-68.
_FACE_MODEL_SUBSET_IDX = [20, 23, 26, 29, 15, 19]
_IBUG68_PNP_IDX        = [36, 39, 42, 45, 31, 35]


# ── ETH-XGaze inference helpers ──────────────────────────────────────
def _build_camera_matrix(W: int, H: int) -> np.ndarray:
    """Fallback intrinsics when no per-clip calibration is available.
    Mirrors the in-pipeline assumption used by head_pose_6dof.py."""
    focal = max(W, H) * 1.2
    return np.array([
        [focal, 0.0,   W / 2.0],
        [0.0,   focal, H / 2.0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)


def _estimate_head_pose(landmarks_2d_6: np.ndarray,
                        face_model_6: np.ndarray,
                        cam: np.ndarray,
                        dist: np.ndarray):
    import cv2
    object_pts = face_model_6.reshape(6, 1, 3).astype(np.float64)
    image_pts  = landmarks_2d_6.reshape(6, 1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        object_pts, image_pts, cam, dist, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok:
        return None
    ok2, rvec, tvec = cv2.solvePnP(
        object_pts, image_pts, cam, dist, rvec, tvec, True,
    )
    if not ok2:
        return None
    return rvec, tvec


def _normalize_face(img_bgr: np.ndarray,
                    face_model_6: np.ndarray,
                    landmarks_2d_6: np.ndarray,
                    rvec: np.ndarray, tvec: np.ndarray,
                    cam: np.ndarray) -> Optional[np.ndarray]:
    """Mirror of demo.normalizeData_face — returns 224x224 BGR crop."""
    import cv2
    focal_norm     = 960
    distance_norm  = 600
    roi_size       = (224, 224)

    hR = cv2.Rodrigues(rvec)[0]
    ht = tvec.reshape((3, 1))
    Fc = np.dot(hR, face_model_6.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center    = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center    = np.mean(np.concatenate((two_eye_center, nose_center), axis=1),
                             axis=1).reshape((3, 1))
    distance = np.linalg.norm(face_center)
    if distance < 1e-6:
        return None
    z_scale = distance_norm / distance
    cam_norm = np.array([
        [focal_norm, 0,          roi_size[0] / 2],
        [0,          focal_norm, roi_size[1] / 2],
        [0,          0,          1.0],
    ], dtype=np.float64)
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ], dtype=np.float64)
    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx); n = np.linalg.norm(down)
    if n < 1e-6:
        return None
    down /= n
    right = np.cross(down, forward); n = np.linalg.norm(right)
    if n < 1e-6:
        return None
    right /= n
    R = np.c_[right, down, forward].T
    try:
        cam_inv = np.linalg.inv(cam)
    except np.linalg.LinAlgError:
        return None
    W = np.dot(np.dot(cam_norm, S), np.dot(R, cam_inv))
    return cv2.warpPerspective(img_bgr, W, roi_size)


# ImageNet stats (matches ETH-XGaze demo.py).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_input_tensor(face_bgr_224: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = face_bgr_224[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).contiguous()
    return t.to(device)


_MODEL_CACHE: dict[str, "torch.nn.Module"] = {}


def _load_gaze_model(ckpt_path: Path, device: torch.device) -> "torch.nn.Module":
    key = f"{ckpt_path.resolve()}|{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    _ensure_eth_xgaze_on_path()
    # Local import — only after sys.path is patched.
    from model import gaze_network                                     # type: ignore
    net = gaze_network()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    net.load_state_dict(sd, strict=True)
    net.to(device).eval()
    _MODEL_CACHE[key] = net
    return net


def _images_input_to_uint8(images) -> Optional[np.ndarray]:
    """Convert a ComfyUI IMAGE tensor (B,H,W,3 float 0..1) to a list-shaped
    uint8 ndarray BGR (B,H,W,3) — same convention as cv2 expects."""
    if images is None:
        return None
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        return None
    arr = np.clip(arr, 0.0, 1.0)
    rgb_u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return rgb_u8[..., [2, 1, 0]]                # RGB -> BGR


def _patch_gaze_entry(entry: dict, eye_name: str,
                      yaw_rad: float, pitch_rad: float) -> None:
    """Mirror of face_expression_editor_ui._apply_gaze_override_to_iris_entry
    but with ``source='ethxgaze'``."""
    key = f"{eye_name}_gaze"
    g = entry.get(key) if isinstance(entry.get(key), dict) else {}
    # Direction in screen space — image-Y is down so flip pitch.
    dx = math.sin(yaw_rad) * math.cos(pitch_rad)
    dy = -math.sin(pitch_rad)
    mag = float(math.hypot(dx, dy))
    if mag > 1e-6:
        dx /= mag; dy /= mag
    mag_norm = min(1.0, abs(yaw_rad) / math.radians(30.0)
                       + abs(pitch_rad) / math.radians(25.0))
    g["yaw_rad"]        = float(yaw_rad)
    g["pitch_rad"]      = float(pitch_rad)
    g["dx"]             = round(float(dx), 4)
    g["dy"]             = round(float(dy), 4)
    g["magnitude_norm"] = round(mag_norm, 4)
    g["source"]         = "ethxgaze"
    entry[key] = g


# ── Node ─────────────────────────────────────────────────────────────
class WanGazeETHXGazeV2:
    CATEGORY    = "WanAnimatePreprocessV2/extras"
    DESCRIPTION = (
        "Replace pose_data['iris_data'] gaze vectors with predictions from "
        "the ETH-XGaze ResNet-50 model (ECCV 2020). Requires the pretrained "
        "checkpoint epoch_24_ckpt.pth.tar in ComfyUI/models/ethxgaze/."
    )
    RETURN_TYPES = ("POSEDATA", "STRING")
    RETURN_NAMES = ("pose_data", "info")
    FUNCTION     = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA", {
                    "tooltip": "POSEDATA bundle from the V2 preprocessor (with iBUG-68 keypoints_face).",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Same RGB image stack the POSEDATA was computed from. Used for the 224x224 face normalisation.",
                }),
                "checkpoint": (_list_ckpt_choices(), {
                    "tooltip": "ETH-XGaze pretrained weights. Auto-discovered from models/ethxgaze/ and third_party/ETH-XGaze/ckpt/.",
                }),
            },
            "optional": {
                "checkpoint_path_override": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to override the dropdown selection. Empty = use the dropdown.",
                }),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "blend": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = keep original iris_data gaze, 1 = full ETH-XGaze. Useful to smooth-blend in the new model.",
                }),
                "batch_size": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Number of normalised face crops fed through gaze_network per forward pass.",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash_args_and_kwargs(**kwargs)

    # ── core ──
    def run(self, pose_data, images, checkpoint,
            checkpoint_path_override: str = "",
            device: str = "auto",
            blend: float = 1.0,
            batch_size: int = 8):
        if isinstance(images, torch.Tensor):
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(
                    f"WanGazeETHXGazeV2: images expected (B,H,W,3); got {tuple(images.shape)}"
                )
        with torch.inference_mode():
            return self._run_impl(
                pose_data, images, checkpoint,
                checkpoint_path_override, device, blend, batch_size,
            )

    def _run_impl(self, pose_data, images, checkpoint,
            checkpoint_path_override: str = "",
            device: str = "auto",
            blend: float = 1.0,
            batch_size: int = 8):
        import cv2

        target_metas = pose_data.get("pose_metas_original") or pose_data.get("pose_metas")
        if not target_metas:
            raise ValueError("pose_data has no 'pose_metas_original' / 'pose_metas'.")

        face_model_68 = _load_face_model_68()
        if face_model_68 is None:
            raise FileNotFoundError(
                "third_party/ETH-XGaze/face_model.txt is missing — clone the repo or copy the file."
            )
        face_model_6 = face_model_68[_FACE_MODEL_SUBSET_IDX, :]

        # Pick checkpoint: explicit override > combo selection > auto.
        path_candidate = checkpoint_path_override.strip() or (
            "" if str(checkpoint).startswith("<none") else str(checkpoint)
        )
        ckpt_path = _resolve_checkpoint(path_candidate)

        # Device.
        if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        net = _load_gaze_model(ckpt_path, dev)

        imgs_bgr = _images_input_to_uint8(images)
        if imgs_bgr is None:
            raise ValueError("`images` must be an IMAGE tensor (B,H,W,3 float in [0,1]).")
        n_imgs = imgs_bgr.shape[0]
        n_frames = len(target_metas)
        if n_imgs < n_frames:
            log.warning("ethxgaze: only %d images for %d frames — re-using last image for the tail.", n_imgs, n_frames)

        # Make sure iris_data is a per-frame list of dicts in the bundle
        # (some upstream paths emit it shorter than the pose_metas list).
        src_iris = pose_data.get("iris_data") or []
        new_iris: list[dict] = []
        for i in range(n_frames):
            v = src_iris[i] if i < len(src_iris) and isinstance(src_iris[i], dict) else {}
            new_iris.append(deepcopy(v))

        dist = np.zeros((4, 1), dtype=np.float64)
        b = max(1, int(batch_size))

        # 1) Build all valid face crops + remember which frame each belongs to.
        crops:   list[np.ndarray] = []
        crop_fi: list[int]        = []
        n_no_face = 0
        n_pnp_fail = 0
        n_warp_fail = 0
        for f_idx, meta in enumerate(target_metas):
            kps = meta.get("keypoints_face")
            if kps is None:
                n_no_face += 1; continue
            kps_arr = np.asarray(kps, dtype=np.float64)
            if kps_arr.ndim != 2 or kps_arr.shape[0] < 68:
                n_no_face += 1; continue
            img_idx = min(f_idx, n_imgs - 1)
            img_bgr = imgs_bgr[img_idx]
            H, W = img_bgr.shape[:2]
            cam = _build_camera_matrix(W, H)
            # iBUG-68 PnP subset — upstream delivers NORMALISED [0,1] coords;
            # cv2.solvePnP needs pixel coords, so scale by image dimensions.
            lms6 = kps_arr[_IBUG68_PNP_IDX, :2].copy()
            lms6[:, 0] *= W
            lms6[:, 1] *= H
            pose = _estimate_head_pose(lms6, face_model_6, cam, dist)
            if pose is None:
                n_pnp_fail += 1; continue
            rvec, tvec = pose
            crop = _normalize_face(img_bgr, face_model_6, lms6, rvec, tvec, cam)
            if crop is None:
                n_warp_fail += 1; continue
            crops.append(crop); crop_fi.append(f_idx)

        # 2) Batched inference.
        n_pred = 0
        for i0 in range(0, len(crops), b):
            batch = crops[i0:i0 + b]
            if not batch:
                break
            ts = [_to_input_tensor(c, dev) for c in batch]
            x = torch.cat(ts, dim=0)
            y = net(x).detach().cpu().numpy()
            for k, (pitch_yaw) in enumerate(y):
                pitch_rad = float(pitch_yaw[0])
                yaw_rad   = float(pitch_yaw[1])
                fi = crop_fi[i0 + k]
                entry = new_iris[fi]
                # Blend with existing gaze if the user wants partial replacement.
                if blend < 1.0:
                    prev_l = entry.get("left_gaze")  or {}
                    prev_r = entry.get("right_gaze") or {}
                    py = (1.0 - blend) * float(prev_l.get("yaw_rad",   0.0)) + blend * yaw_rad
                    pp = (1.0 - blend) * float(prev_l.get("pitch_rad", 0.0)) + blend * pitch_rad
                    _patch_gaze_entry(entry, "left",  py, pp)
                    ry = (1.0 - blend) * float(prev_r.get("yaw_rad",   0.0)) + blend * yaw_rad
                    rp = (1.0 - blend) * float(prev_r.get("pitch_rad", 0.0)) + blend * pitch_rad
                    _patch_gaze_entry(entry, "right", ry, rp)
                else:
                    _patch_gaze_entry(entry, "left",  yaw_rad, pitch_rad)
                    _patch_gaze_entry(entry, "right", yaw_rad, pitch_rad)
                n_pred += 1

        out_bundle = dict(pose_data)
        out_bundle["iris_data"] = new_iris

        info = (
            f"WanGazeETHXGazeV2: predicted={n_pred}/{n_frames} frames | "
            f"no_face={n_no_face} | pnp_fail={n_pnp_fail} | warp_fail={n_warp_fail} | "
            f"device={dev.type} | ckpt={ckpt_path.name} | blend={blend:.2f}"
        )
        log.info(info)
        return (out_bundle, info)
