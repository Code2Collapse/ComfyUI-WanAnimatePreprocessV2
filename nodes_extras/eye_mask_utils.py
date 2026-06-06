"""Eye-region mask generation utilities.

Extracted from ``third_party/Kanibus/nodes/neural_pupil_tracker.py``
(``NeuralPupilTracker._create_eye_mask``) — adapted into a standalone
function that does **not** require MediaPipe or the Kanibus runtime.

Only **numpy** and **opencv-python** are needed.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def create_eye_mask(
    image_hw: tuple[int, int],
    eye_landmarks: np.ndarray,
    dilation: int = 5,
    feather: float = 0.0,
) -> np.ndarray:
    """Build a binary (0/255) mask around an eye region.

    Mirrors the convex-hull + dilation approach from the Kanibus
    ``NeuralPupilTracker`` but exposes feathering as an extra option.

    Parameters
    ----------
    image_hw : (int, int)
        ``(height, width)`` of the target frame.
    eye_landmarks : np.ndarray
        Landmark array with shape ``(N, 2+)`` — only the first two columns
        (x, y in pixel coords) are used.  Works with 4-point iris landmarks
        (MediaPipe 468–475) or full eye contour (16-point) sets.
    dilation : int
        Morphological dilation radius (pixels).  ``0`` to skip.
    feather : float
        Gaussian-blur sigma for soft-edge feathering.  ``0`` to skip.

    Returns
    -------
    np.ndarray
        Single-channel ``uint8`` mask of shape ``(H, W)``.
    """
    h, w = image_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = eye_landmarks[:, :2].astype(np.int32)
    if len(pts) < 3:
        for pt in pts:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), max(dilation, 3), 255, -1)
    else:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 255)

    if dilation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation, dilation),
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    if feather > 0:
        ksize = int(feather * 6) | 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)

    return mask


def create_eye_masks_from_landmarks(
    image_hw: tuple[int, int],
    landmarks: np.ndarray,
    *,
    left_iris_indices: Optional[list[int]] = None,
    right_iris_indices: Optional[list[int]] = None,
    left_contour_indices: Optional[list[int]] = None,
    right_contour_indices: Optional[list[int]] = None,
    dilation: int = 5,
    feather: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate left and right eye masks from a full-face landmark array.

    Default indices match the MediaPipe Face Mesh with iris refinement
    (468 + iris landmarks), the same convention used by the Kanibus
    ``NeuralPupilTracker``.  Override with ``left_contour_indices`` /
    ``right_contour_indices`` to use iBUG-68 eye contour points instead.

    Parameters
    ----------
    image_hw : (int, int)
        ``(height, width)`` of the target frame.
    landmarks : np.ndarray
        Full landmark array with shape ``(N, 2+)``.
    left_iris_indices / right_iris_indices
        4-point iris landmark indices (MediaPipe: ``[468,469,470,471]`` /
        ``[472,473,474,475]``).
    left_contour_indices / right_contour_indices
        Eye contour landmark indices.  If given, contour masks are generated
        instead of iris-only masks.
    dilation, feather
        Passed to :func:`create_eye_mask`.

    Returns
    -------
    (left_mask, right_mask) : tuple[np.ndarray, np.ndarray]
        Two ``uint8`` masks of shape ``(H, W)``.
    """
    if left_iris_indices is None:
        left_iris_indices = [468, 469, 470, 471]
    if right_iris_indices is None:
        right_iris_indices = [472, 473, 474, 475]

    left_idx = left_contour_indices or left_iris_indices
    right_idx = right_contour_indices or right_iris_indices

    max_idx = max(max(left_idx), max(right_idx))
    if max_idx >= len(landmarks):
        raise IndexError(
            f"Landmark array has {len(landmarks)} points but index "
            f"{max_idx} was requested — check your index lists."
        )

    left_mask = create_eye_mask(
        image_hw, landmarks[left_idx], dilation=dilation, feather=feather,
    )
    right_mask = create_eye_mask(
        image_hw, landmarks[right_idx], dilation=dilation, feather=feather,
    )
    return left_mask, right_mask
