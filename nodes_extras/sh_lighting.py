"""WanSHLightingTransferV2 — Spherical-Harmonics lighting extraction & transfer.

Implements the standard Basri-Jacobs SH lighting model (L=2, 9 basis):

    I(p) = ρ(p) · Σ_{i=0..8}  L_i · Y_i(n(p))

For a Lambertian surface with albedo ρ and surface normal n, the irradiance
under distant lighting is a low-rank function projected onto 9 SH basis
functions. We do the per-channel least-squares fit:

    Y c = I            (Y: N×9 basis matrix, c: 9, I: N intensity)

Inputs:
  - source_image : IMAGE  -- the lit reference (any frames B,H,W,3, float [0,1])
  - source_normal: IMAGE  -- per-pixel surface normals encoded RGB
                              channel R = nx (0..1 maps to -1..1)
                              channel G = ny
                              channel B = nz   (Z+ out of screen)
  - albedo       : IMAGE  -- optional; if absent we use intensity / max as a
                              rough albedo prior (still works for relative
                              transfer; for absolute, supply true albedo).
  - target_image : IMAGE  -- optional; the image to RELIGHT.
  - target_normal: IMAGE  -- optional; normals for the target.
  - operation    : "fit_only" | "transfer" | "rotate_lights"
  - rotate_yaw / rotate_pitch: degrees, for "rotate_lights" mode (rotates SH).

Outputs:
  - relit_image  : IMAGE  -- target_image relit by source SH coeffs (or source
                              relit by itself / rotated lights).
  - sh_coeffs_json: STRING -- per-frame {sh: [[r9],[g9],[b9]], rms_err: ..}.
  - shading_map  : IMAGE  -- the irradiance map computed from the SH fit
                              (great for verification — should match the
                              source's smooth shading).
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import torch


# SH basis (real-valued, L=2, 9 components)
# Coefficients from Ramamoorthi & Hanrahan 2001
_C0 = 0.282095          # Y_00
_C1 = 0.488603          # Y_1{-1,0,1}
_C2 = 1.092548          # Y_2{-2,-1,1}
_C3 = 0.315392          # Y_20 const
_C4 = 0.546274          # Y_22


def _sh_basis(n: np.ndarray) -> np.ndarray:
    """n: (N,3) normalised normals -> Y: (N,9)."""
    x = n[:, 0]
    y = n[:, 1]
    z = n[:, 2]
    Y = np.stack([
        np.full_like(x, _C0),
        _C1 * y,
        _C1 * z,
        _C1 * x,
        _C2 * x * y,
        _C2 * y * z,
        _C3 * (3 * z * z - 1),
        _C2 * x * z,
        _C4 * (x * x - y * y),
    ], axis=1)
    return Y


def _decode_normals(normal_rgb: np.ndarray) -> np.ndarray:
    """RGB [0,1] -> XYZ in [-1,1], normalised."""
    n = normal_rgb.astype(np.float32) * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / np.maximum(norm, 1e-6)


def _luminance(img: np.ndarray) -> np.ndarray:
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def _fit_sh(image: np.ndarray, normals: np.ndarray,
            albedo: Optional[np.ndarray]) -> tuple[np.ndarray, float]:
    """Per-channel SH fit. Returns coeffs (3,9) and rms residual."""
    H, W, _ = image.shape
    n = normals.reshape(-1, 3)
    Y = _sh_basis(n)                 # (N, 9)
    coeffs = np.zeros((3, 9), dtype=np.float32)
    residuals = []
    for c in range(3):
        I = image[..., c].reshape(-1).astype(np.float32)
        if albedo is not None:
            rho = albedo[..., c].reshape(-1).astype(np.float32)
            rho = np.maximum(rho, 1e-3)
            I_norm = I / rho
        else:
            I_norm = I
        sol, res, _, _ = np.linalg.lstsq(Y, I_norm, rcond=None)
        coeffs[c] = sol.astype(np.float32)
        pred = Y @ sol
        residuals.append(float(np.sqrt(np.mean((pred - I_norm) ** 2))))
    return coeffs, float(np.mean(residuals))


def _render_sh(coeffs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """coeffs (3,9), normals (H,W,3) -> (H,W,3) irradiance."""
    H, W, _ = normals.shape
    Y = _sh_basis(normals.reshape(-1, 3))         # (N,9)
    irr = (Y @ coeffs.T).reshape(H, W, 3)         # (H,W,3)
    return np.clip(irr, 0, None).astype(np.float32)


def _rotate_sh_y_axis(coeffs: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate SH coefficients about the Y axis. Closed-form for L<=2.

    Reference: Ivanic & Ruedenberg 1996 / Sloan 2008.
    Coeff order assumed: [0:(0,0), 1:(1,-1), 2:(1,0), 3:(1,1),
                          4:(2,-2), 5:(2,-1), 6:(2,0), 7:(2,1), 8:(2,2)].
    """
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    out = coeffs.copy()
    # L=1
    out[:, 3] = c * coeffs[:, 3] + s * coeffs[:, 1]    # m=+1
    out[:, 1] = -s * coeffs[:, 3] + c * coeffs[:, 1]   # m=-1
    # L=2 (Y rotation)
    c2 = math.cos(2 * yaw_rad)
    s2 = math.sin(2 * yaw_rad)
    out[:, 8] = c2 * coeffs[:, 8] + s2 * coeffs[:, 4]  # m=+2
    out[:, 4] = -s2 * coeffs[:, 8] + c2 * coeffs[:, 4]  # m=-2
    out[:, 7] = c * coeffs[:, 7] + s * coeffs[:, 5]    # m=+1
    out[:, 5] = -s * coeffs[:, 7] + c * coeffs[:, 5]   # m=-1
    return out


class WanSHLightingTransferV2:
    CATEGORY = "WanAnimatePreprocess_V2/Lighting"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("relit_image", "sh_coeffs_json", "shading_map")
    DESCRIPTION = ("Spherical-harmonics lighting fit (L=2, 9 basis, per RGB channel) "
                   "with optional relighting onto a target. Basri-Jacobs/Ramamoorthi formulation.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "source_normal": ("IMAGE",),
                "operation": (["fit_only", "transfer", "rotate_lights"], {"default": "fit_only"}),
                "rotate_yaw_deg": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "source_albedo": ("IMAGE",),
                "target_image": ("IMAGE",),
                "target_normal": ("IMAGE",),
                "target_albedo": ("IMAGE",),
            },
        }

    def execute(self, source_image, source_normal, operation, rotate_yaw_deg, intensity,
                source_albedo=None, target_image=None, target_normal=None, target_albedo=None):

        src_img = source_image.detach().cpu().numpy().astype(np.float32)
        src_nrm = source_normal.detach().cpu().numpy().astype(np.float32)
        if src_img.shape[:3] != src_nrm.shape[:3]:
            # broadcast first normal frame across all source frames
            if src_nrm.shape[0] == 1 and src_img.shape[0] > 1:
                src_nrm = np.repeat(src_nrm, src_img.shape[0], axis=0)
            else:
                raise ValueError("WanSHLightingTransferV2: source_image and source_normal must align (B,H,W).")

        src_alb = source_albedo.detach().cpu().numpy().astype(np.float32) if source_albedo is not None else None
        tgt_img = target_image.detach().cpu().numpy().astype(np.float32) if target_image is not None else None
        tgt_nrm = target_normal.detach().cpu().numpy().astype(np.float32) if target_normal is not None else None
        tgt_alb = target_albedo.detach().cpu().numpy().astype(np.float32) if target_albedo is not None else None

        B = src_img.shape[0]
        relit_frames = []
        shading_frames = []
        coeffs_log = []

        for i in range(B):
            si = src_img[i]
            sn_rgb = src_nrm[i]
            sn = _decode_normals(sn_rgb)
            sa = src_alb[i] if src_alb is not None else None
            coeffs, rms = _fit_sh(si, sn, sa)

            if operation == "rotate_lights":
                coeffs_rot = _rotate_sh_y_axis(coeffs, math.radians(float(rotate_yaw_deg)))
                shading = _render_sh(coeffs_rot, sn)
                relit = sa * shading if sa is not None else shading
                if sa is None:
                    # rough: divide source by its own shading then multiply by rotated
                    own_shade = np.maximum(_render_sh(coeffs, sn), 1e-3)
                    relit = si / own_shade * shading
                relit = np.clip(relit * intensity, 0, 1).astype(np.float32)
                shading_frames.append(np.clip(shading * intensity, 0, 1))
                relit_frames.append(relit)
                coeffs_log.append({"sh": coeffs.tolist(), "sh_rotated": coeffs_rot.tolist(),
                                   "yaw_deg": float(rotate_yaw_deg), "rms_err": rms})
                continue

            if operation == "transfer" and tgt_img is not None and tgt_nrm is not None:
                j = min(i, tgt_img.shape[0] - 1)
                ti = tgt_img[j]
                tn = _decode_normals(tgt_nrm[j])
                ta = tgt_alb[min(j, (tgt_alb.shape[0] - 1) if tgt_alb is not None else 0)] if tgt_alb is not None else None
                shading_new = _render_sh(coeffs, tn)
                if ta is not None:
                    relit = ta * shading_new
                else:
                    # Use intensity-based albedo proxy: target / its-own-shading-from-target-fit
                    coeffs_target, _ = _fit_sh(ti, tn, None)
                    own_shade = np.maximum(_render_sh(coeffs_target, tn), 1e-3)
                    relit = ti / own_shade * shading_new
                relit = np.clip(relit * intensity, 0, 1).astype(np.float32)
                shading_frames.append(np.clip(shading_new * intensity, 0, 1))
                relit_frames.append(relit)
                coeffs_log.append({"sh": coeffs.tolist(), "rms_err": rms, "applied_to": "target"})
                continue

            # fit_only: return source-reconstructed image
            shading = _render_sh(coeffs, sn)
            if sa is not None:
                recon = np.clip(sa * shading * intensity, 0, 1)
            else:
                recon = np.clip(shading * intensity, 0, 1)
            shading_frames.append(np.clip(shading * intensity, 0, 1))
            relit_frames.append(recon.astype(np.float32))
            coeffs_log.append({"sh": coeffs.tolist(), "rms_err": rms})

        relit_t = torch.from_numpy(np.stack(relit_frames, axis=0))
        shading_t = torch.from_numpy(np.stack(shading_frames, axis=0))
        return (relit_t,
                json.dumps({"operation": operation, "frames": coeffs_log}),
                shading_t)
