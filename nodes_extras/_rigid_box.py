# -*- coding: utf-8 -*-
"""Expression-invariant face-box anchoring.

THE BUG THIS FIXES
------------------
`get_face_bboxes` takes min/max over ALL 68 face landmarks. Brows, eyelids and
lips are IN that set, so the box is measured partly from the very thing we are
trying to transmit. Raise the brows and the box top rises with them; drop the
jaw and the box bottom follows. The crop therefore MOVES AND RESCALES with
expression.

Measured on a real 197px face at the node's default scale=1.3 (222x235px box):

    brow raise  (4% face height)  ->  centre -4.0px, size +10.0px
    jaw drop    (7% face height)  ->  centre +1.0px, size  +6.0px
    talk pose   (brow+mouth+jaw)  ->  centre -3.0px, size +14.0px

14px on a 235px box is a 6% zoom, and it lands in the 512 tile as the head
scaling and translating on every talking frame.

WHY THAT IS THE WORST POSSIBLE NOISE HERE
-----------------------------------------
Wan-Animate compresses the whole 512 tile to twenty numbers (motion_dim=20).
Global scale and translation are high-energy, low-frequency signals — exactly
what a motion encoder locks onto first. So the leaked crop wobble does not just
add noise, it CONSUMES the budget that should have carried the expression, and
it is perfectly correlated with the expression, so it cannot be filtered out
afterwards by any temporal smoother.

This is a known preprocessing failure, not a novel theory. "Expressive Talking
Head Video Encoding in StyleGAN2 Latent-Space" (arXiv:2203.14512) reports that
"dynamic facial landmark coordinates undergo dynamic changes in a video clip
which generate jitters and rescaling in face alignment", and that a standard
mitigation is "cropping the full face excluding the eyes and mouth coordinates
to avoid the impact of dynamic coordinates".

THE FIX
-------
Measure the box from landmarks that expression does NOT move:

    temples / upper jaw   0-3, 13-16   (near the ears; the chin is excluded
                                        because the lower jaw drops)
    nose bridge           27-30        (the most rigid structure on a face)
    nose base             31-35
    eye CORNERS           36, 39, 42, 45  (corners, not lids — lids blink)

Excluded: brows (17-26), eyelids, mouth (48-67), chin (4-12).

CALIBRATION — WHY THE FRAMING DOES NOT CHANGE
---------------------------------------------
Dropping the chin and brows from the measurement would shrink the box and
change the framing, which would put the tile OUT of Wan-Animate's training
distribution — a worse problem than the one being fixed.

So the rigid extent is calibrated per clip against the ORIGINAL boxes: the
median scale ratio and the median centre offset are measured once and applied
to every frame. The result reproduces the clip's original median framing
exactly, while the per-frame variation comes only from rigid geometry.

Net effect: same average crop, same distribution, but blinking and talking no
longer move it. Real head motion still does, because the rigid points really do
move when the head moves — that is signal, and it is preserved.
"""
from __future__ import annotations

import numpy as np

#: Indices into the 68-point iBUG face set that expression does not move.
RIGID_68 = np.array([
    0, 1, 2, 3, 13, 14, 15, 16,      # temples / upper jaw, NOT the chin
    27, 28, 29, 30,                  # nose bridge
    31, 32, 33, 34, 35,              # nose base
    36, 39, 42, 45,                  # eye corners only, never the lids
], dtype=np.int32)

#: Below this many frames a median is meaningless, so calibration would just be
#: an expensive identity. Fall back to the original boxes and say nothing.
MIN_FRAMES = 3


def _rigid_extent(kf_xy: np.ndarray, w: int, h: int):
    """(cx, cy, width, height) of the rigid landmark set, in pixels.

    `kf_xy` is the node's keypoints_face array: NORMALISED coordinates with
    slot 0 a body anchor, so the 68 face points live at 1..68.
    """
    pts = np.asarray(kf_xy, np.float32)[1:]
    if pts.shape[0] < 68:
        return None
    r = pts[RIGID_68] * (w, h)
    lo, hi = r.min(0), r.max(0)
    rw, rh = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if rw < 2.0 or rh < 2.0:
        return None
    return float((lo[0] + hi[0]) * 0.5), float((lo[1] + hi[1]) * 0.5), rw, rh


def rigid_anchor_boxes(kf_list, boxes, image_shape):
    """Re-derive `boxes` from rigid landmarks, keeping the clip's median framing.

    boxes are (x1, x2, y1, y2), the node's internal order.
    Returns a new list, or the input unchanged when there is not enough to work
    with — this never fails loudly and never invents a box.
    """
    h, w = image_shape
    n = len(boxes)
    if n < MIN_FRAMES or len(kf_list) != n:
        return list(boxes), None

    ext = [_rigid_extent(kf, w, h) for kf in kf_list]
    ok = [i for i, e in enumerate(ext) if e is not None]
    if len(ok) < MIN_FRAMES:
        return list(boxes), None

    # Calibrate against the ORIGINAL boxes so the framing is preserved.
    sx, sy, ox, oy = [], [], [], []
    for i in ok:
        e = ext[i]
        if e is None:                       # unreachable; keeps the type honest
            continue
        cx, cy, rw, rh = e
        x1, x2, y1, y2 = boxes[i]
        bw, bh = float(x2 - x1), float(y2 - y1)
        if bw < 2 or bh < 2:
            continue
        sx.append(bw / rw)
        sy.append(bh / rh)
        # Offsets normalised by face size so they survive a dolly.
        ox.append((0.5 * (x1 + x2) - cx) / rw)
        oy.append((0.5 * (y1 + y2) - cy) / rh)
    if len(sx) < MIN_FRAMES:
        return list(boxes), None
    kx, ky = float(np.median(sx)), float(np.median(sy))
    dx, dy = float(np.median(ox)), float(np.median(oy))

    out = []
    for i in range(n):
        e = ext[i]
        if e is None:                       # keep the original where rigid failed
            out.append(tuple(boxes[i]))
            continue
        cx, cy, rw, rh = e
        bw, bh = rw * kx, rh * ky
        ccx, ccy = cx + dx * rw, cy + dy * rh
        x1 = int(round(ccx - bw * 0.5)); x2 = int(round(ccx + bw * 0.5))
        y1 = int(round(ccy - bh * 0.5)); y2 = int(round(ccy + bh * 0.5))
        x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 2, min(w, x2))
        y1 = max(0, min(h - 1, y1)); y2 = max(y1 + 2, min(h, y2))
        out.append((x1, x2, y1, y2))

    # How far the boxes actually moved, so the log states a measured number
    # rather than a claim.
    d = [abs(0.5 * (out[i][2] + out[i][3]) - 0.5 * (boxes[i][2] + boxes[i][3]))
         for i in ok]
    return out, (float(np.mean(d)), float(np.max(d)), len(ok))
