# -*- coding: utf-8 -*-
"""Nuke-equivalent resampling filters for the face tile.

WHY
---
The face crop has to reach 512x512 for Wan-Animate's motion encoder, and the
filter used for that resample decides how much of the result is real. cv2's
INTER_LANCZOS4 — the previous hardcoded choice — is a windowed sinc whose
kernel has NEGATIVE lobes, so it overshoots and undershoots at hard edges.
That is ringing: invented structure the encoder cannot distinguish from real
texture, and the reason a crop taken to 512 and sampled back down does not
match the original sampled down directly.

Nuke exposes this choice for exactly this reason. Every filter here is a
BC-spline (Mitchell-Netravali) or a windowed sinc, matching Nuke's names so a
compositor can reason about it with the vocabulary they already use.

THE MATHS
---------
All the cubic filters are one family. Mitchell and Netravali parameterised
piecewise-cubic reconstruction with two numbers, B and C:

    k(x) = 1/6 * {
        (12 - 9B - 6C)|x|^3 + (-18 + 12B + 6C)|x|^2 + (6 - 2B)      |x| < 1
        (-B - 6C)|x|^3 + (6B + 30C)|x|^2 + (-12B - 48C)|x| + (8B+24C) 1 <= |x| < 2
        0                                                              otherwise
    }

B controls blur, C controls "cardinal" sharpening. The whole named zoo is just
points in that (B, C) plane.

RINGING, WHICH IS THE POINT
---------------------------
A filter rings when its kernel goes NEGATIVE. Verified against the
Mitchell-Netravali paper and ImageMagick's filter reference:

    no negative lobes  -> Parzen (B-spline), cubic/Hermite   : cannot ring
    mild negative lobes-> Mitchell (1/3, 1/3)                : the recommended
                                                               blur/ring balance
    strong negatives   -> Keys, Simon, Rifman, Lanczos, sinc : sharp, rings

Mitchell and Netravali's own conclusion was that "the best filter was a Keys
family filter using B,C values of 1/3, 1/3" — i.e. `mitchell`. That is why it
is this module's default rather than the sharpest option.

For DOWNSIZING toward a small target, ringing is worse than softness: an
overshoot becomes a hard bright/dark fringe that survives into the latent.
Prefer mitchell, or parzen if the plate is already noisy.
"""
from __future__ import annotations

import numpy as np

#: (B, C) for every Nuke-named cubic. Values verified against the
#: Mitchell-Netravali paper and ImageMagick's filter reference where those name
#: them; the Keys-family C values for simon/rifman follow the standard VFX
#: convention (Keys family, increasing C = increasing sharpness) and are marked
#: as such rather than presented as primary-sourced.
BC_FILTERS = {
    "cubic":    (0.0,      0.0),      # Hermite. No negative lobes.
    "keys":     (0.0,      0.5),      # Catmull-Rom. Sharp, rings.
    "simon":    (0.0,      0.75),     # convention: sharper Keys
    "rifman":   (0.0,      1.0),      # convention: sharpest Keys, most ringing
    "mitchell": (1.0 / 3,  1.0 / 3),  # the paper's recommended balance
    "parzen":   (1.0,      0.0),      # cubic B-spline. Softest, cannot ring.
    "notch":    (1.5,     -0.25),     # strongly suppresses postaliasing
}

#: Windowed / truncated sincs, by lobe count.
SINC_FILTERS = {"lanczos4": 4, "lanczos6": 6, "sinc4": 4}

RESIZE_FILTERS = ("impulse", "cubic", "keys", "simon", "rifman", "mitchell",
                  "parzen", "notch", "lanczos4", "lanczos6", "sinc4")

#: Filters whose kernel dips below zero, i.e. which can ring. Computed, not
#: asserted — see kernel_rings().
def _bc_kernel(x: np.ndarray, B: float, C: float) -> np.ndarray:
    ax = np.abs(x).astype(np.float64)
    k = np.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    k[m1] = ((12 - 9 * B - 6 * C) * ax[m1] ** 3
             + (-18 + 12 * B + 6 * C) * ax[m1] ** 2
             + (6 - 2 * B))
    k[m2] = ((-B - 6 * C) * ax[m2] ** 3
             + (6 * B + 30 * C) * ax[m2] ** 2
             + (-12 * B - 48 * C) * ax[m2]
             + (8 * B + 24 * C))
    return k / 6.0


def _sinc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(np.pi * x[nz]) / (np.pi * x[nz])
    return out


def _sinc_kernel(x: np.ndarray, lobes: int, windowed: bool) -> np.ndarray:
    ax = np.abs(x).astype(np.float64)
    k = np.zeros_like(ax)
    m = ax < lobes
    if windowed:                     # Lanczos = sinc * sinc(x/lobes)
        k[m] = _sinc(ax[m]) * _sinc(ax[m] / lobes)
    else:                            # sincN = hard-truncated sinc
        k[m] = _sinc(ax[m])
    return k


def kernel(name: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a named filter's kernel at sample offsets x."""
    n = str(name).lower()
    if n in BC_FILTERS:
        return _bc_kernel(x, *BC_FILTERS[n])
    if n in SINC_FILTERS:
        return _sinc_kernel(x, SINC_FILTERS[n], windowed=n.startswith("lanczos"))
    raise ValueError(f"unknown resize filter {name!r}; expected one of {RESIZE_FILTERS}")


def kernel_support(name: str) -> int:
    n = str(name).lower()
    if n in BC_FILTERS:
        return 2
    if n in SINC_FILTERS:
        return SINC_FILTERS[n]
    raise ValueError(f"unknown resize filter {name!r}")


def kernel_rings(name: str) -> bool:
    """Does this filter's kernel go negative? Measured, not asserted."""
    if str(name).lower() == "impulse":
        return False
    x = np.linspace(-kernel_support(name), kernel_support(name), 2001)
    return bool(kernel(name, x).min() < -1e-6)


def resize(img: np.ndarray, out_w: int, out_h: int, name: str = "mitchell") -> np.ndarray:
    """Separable resample with the named filter. Float in, float out, no clip.

    Deliberately NOT clipped: the tile can carry linear/HDR values, and a
    filter with negative lobes legitimately produces values outside the input
    range. Clamping here would hide the very overshoot the caller is choosing
    a filter to control.
    """
    n = str(name).lower()
    a = np.asarray(img)
    was_u8 = a.dtype == np.uint8
    f = a.astype(np.float32) / 255.0 if was_u8 else a.astype(np.float32)
    if f.ndim == 2:
        f = f[..., None]
    in_h, in_w = f.shape[:2]

    if n == "impulse":               # nearest neighbour
        yi = np.clip((np.arange(out_h) + 0.5) * in_h / out_h, 0, in_h - 1).astype(np.int32)
        xi = np.clip((np.arange(out_w) + 0.5) * in_w / out_w, 0, in_w - 1).astype(np.int32)
        out = f[yi][:, xi]
    else:
        sup = kernel_support(n)
        out = f
        for axis, (src, dst) in enumerate(((in_h, out_h), (in_w, out_w))):
            scale = dst / float(src)
            # Downscaling widens the kernel in SOURCE space — this is what makes
            # a downsize properly band-limited instead of aliased. Skipping it
            # is the classic "my downscale is crunchy" bug.
            fscale = 1.0 / scale if scale < 1.0 else 1.0
            radius = sup * fscale
            centres = (np.arange(dst) + 0.5) / scale - 0.5
            left = np.floor(centres - radius).astype(np.int32)
            n_taps = int(np.ceil(2 * radius)) + 2
            idx = left[:, None] + np.arange(n_taps)[None, :]
            w = kernel(n, (idx - centres[:, None]) / fscale)
            wsum = w.sum(1, keepdims=True)
            w = np.divide(w, wsum, out=np.zeros_like(w), where=np.abs(wsum) > 1e-12)
            idx = np.clip(idx, 0, src - 1)
            moved = np.moveaxis(out, axis, 0)
            gathered = moved[idx]                       # (dst, taps, ...)
            acc = np.einsum("dt,dt...->d...", w.astype(np.float32), gathered)
            out = np.moveaxis(acc, 0, axis)
    if out.shape[-1] == 1 and a.ndim == 2:
        out = out[..., 0]
    return (np.clip(out, 0, 1) * 255.0).astype(np.uint8) if was_u8 else out
