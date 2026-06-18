"""
_face_warp.py — landmark-driven dense pixel warp (Spec-A Pillar-2, backend).

WanFaceController3DV2 edits face landmarks (head pose / FACS / reference blend)
and so far emitted only POSEDATA + coefficients — the actual pixels were left for
Wan to render downstream. This module adds the missing "render in-node" half: it
warps the SOURCE frame's pixels from the NEUTRAL (detected) landmarks to the
EDITED landmarks, so the node can emit a warped IMAGE preview/output directly.

Algorithm (pure numpy + OpenCV, CPU, no model):
  1. Augment the 68 iBUG landmarks with 8 frame-border anchor points so the warp
     covers the whole frame (not just the face hull) and the background stays put.
  2. Delaunay-triangulate the NEUTRAL (source) point set once.
  3. For each triangle, compute the affine map source-tri -> dest-tri and
     warpAffine that triangle's pixels, compositing through a triangle mask.
This is the standard Bowyer/Delaunay piecewise-affine face morph used by face
swappers; it is fast (~few ms/frame at video res) and dependency-light.

If OpenCV is unavailable the caller should skip the warped output gracefully —
`warp_available()` reports this so the node never hard-fails on a missing dep.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_CV2 = False


def warp_available() -> bool:
    """True if the OpenCV backend needed for warping is importable."""
    return _HAVE_CV2


def _border_points(w: int, h: int) -> np.ndarray:
    """8 anchor points around the frame edge so the warp covers the full image."""
    xs = [0, (w - 1) / 2.0, w - 1]
    ys = [0, (h - 1) / 2.0, h - 1]
    pts = []
    for y in ys:
        for x in xs:
            if x == (w - 1) / 2.0 and y == (h - 1) / 2.0:
                continue  # skip centre
            pts.append([x, y])
    return np.asarray(pts, dtype=np.float32)


def _delaunay_triangle_indices(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Return (M,3) int array of triangle vertex indices into `points`."""
    # Subdiv2D needs an integer-bounded rect that contains every point.
    rect = (0, 0, int(w) + 1, int(h) + 1)
    subdiv = cv2.Subdiv2D(rect)
    # Map rounded (x,y) -> index. Subdiv stores its own coords, so we look the
    # triangle vertices back up against our point list by nearest match.
    pts = points.astype(np.float32)
    for p in pts:
        subdiv.insert((float(np.clip(p[0], 0, w)), float(np.clip(p[1], 0, h))))
    tri_list = subdiv.getTriangleList()
    tris = []
    # Build a quick lookup: rounded coord -> index.
    lut = {}
    for i, p in enumerate(pts):
        lut[(round(float(p[0]), 1), round(float(p[1]), 1))] = i

    def _idx(x, y):
        key = (round(float(x), 1), round(float(y), 1))
        if key in lut:
            return lut[key]
        # nearest fallback (subdiv may nudge coords)
        d = np.sum((pts - np.array([x, y], dtype=np.float32)) ** 2, axis=1)
        return int(np.argmin(d))

    for t in tri_list:
        x1, y1, x2, y2, x3, y3 = t
        # drop triangles with vertices outside the frame (Subdiv adds super-tri)
        if min(x1, x2, x3) < -1 or min(y1, y2, y3) < -1:
            continue
        if max(x1, x2, x3) > w + 1 or max(y1, y2, y3) > h + 1:
            continue
        a, b, c = _idx(x1, y1), _idx(x2, y2), _idx(x3, y3)
        if a != b and b != c and a != c:
            tris.append((a, b, c))
    return np.asarray(sorted(set(tris)), dtype=np.int32)


def _warp_triangle(src_img, dst_img, t_src, t_dst):
    """Affine-warp one triangle from src_img into dst_img (in place)."""
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return
    t1 = [(p[0] - x1, p[1] - y1) for p in t_src]
    t2 = [(p[0] - x2, p[1] - y2) for p in t_dst]
    src_patch = src_img[y1:y1 + h1, x1:x1 + w1]
    if src_patch.size == 0:
        return
    M = cv2.getAffineTransform(np.float32(t1), np.float32(t2))
    warped = cv2.warpAffine(
        src_patch, M, (w2, h2),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = np.zeros((h2, w2, 1), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2), (1.0,), cv2.LINE_AA, 0)
    region = dst_img[y2:y2 + h2, x2:x2 + w2]
    if region.shape[:2] != warped.shape[:2]:
        return
    dst_img[y2:y2 + h2, x2:x2 + w2] = region * (1.0 - mask) + warped * mask


def warp_face(image: np.ndarray, src_lms: np.ndarray, dst_lms: np.ndarray) -> np.ndarray:
    """
    Warp `image` (H,W,3 float32 0..1) so pixels move from `src_lms` to `dst_lms`.

    src_lms / dst_lms: (68,2) float arrays in PIXEL coordinates. Returns a new
    (H,W,3) float32 image; on any failure returns a copy of the input unchanged.
    """
    if not _HAVE_CV2:
        return image.copy()
    img = np.ascontiguousarray(image.astype(np.float32))
    h, w = img.shape[:2]
    src = np.asarray(src_lms, dtype=np.float32).reshape(-1, 2)
    dst = np.asarray(dst_lms, dtype=np.float32).reshape(-1, 2)
    if src.shape != dst.shape or src.shape[0] < 3:
        return img.copy()
    # clamp into frame so triangulation/bounding rects stay valid
    src[:, 0] = np.clip(src[:, 0], 0, w - 1)
    src[:, 1] = np.clip(src[:, 1], 0, h - 1)
    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)
    border = _border_points(w, h)
    src_all = np.vstack([src, border])
    dst_all = np.vstack([dst, border])
    try:
        tris = _delaunay_triangle_indices(src_all, w, h)
    except Exception:
        return img.copy()
    if tris.size == 0:
        return img.copy()
    out = img.copy()
    for (a, b, c) in tris:
        t_src = [tuple(src_all[a]), tuple(src_all[b]), tuple(src_all[c])]
        t_dst = [tuple(dst_all[a]), tuple(dst_all[b]), tuple(dst_all[c])]
        try:
            _warp_triangle(img, out, t_src, t_dst)
        except Exception:
            continue
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Synthetic proof: gradient image + 68 random-but-structured landmarks,
    # push the mouth region down, confirm pixels actually moved there and the
    # border stayed fixed.
    import sys
    if not _HAVE_CV2:
        print("cv2 not available — cannot self-test"); sys.exit(1)
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.stack([xx / W, yy / H, np.ones_like(xx) * 0.5], axis=-1).astype(np.float32)
    rng = np.random.default_rng(0)
    src = rng.uniform(40, 216, size=(68, 2)).astype(np.float32)
    # iBUG mouth indices 48..67 — shove them +18px down
    dst = src.copy()
    dst[48:68, 1] += 18.0
    out = warp_face(base, src, dst)
    assert out.shape == base.shape and out.dtype == np.float32, "shape/dtype"
    moved = float(np.abs(out - base).mean())
    border_delta = float(np.abs(out[0, :] - base[0, :]).mean())  # top row anchored
    print(f"mean|delta|={moved:.5f}  top-border|delta|={border_delta:.6f}")
    assert moved > 1e-4, "warp produced no change"
    assert border_delta < 1e-3, "border should stay anchored"
    print("SELF-TEST PASS: pixels warped in face region, border anchored.")
