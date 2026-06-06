# -*- coding: utf-8 -*-
"""Pose-normalized gaze input pipeline (clean-room implementation).

This module implements the analytical face-image normalization
methodology described in:

    Zhang et al. (2018). "Revisiting Data Normalization for
    Appearance-Based Gaze Estimation." Proc. ACM Symposium on Eye
    Tracking Research & Applications (ETRA).

The paper specifies an analytical preprocessing step that rotates and
scales a face image so the synthetic camera looks straight at the face
center from a fixed distance, with the head's X-axis kept horizontal in
the warped image. This removes head pose (roll + distance) as a
nuisance variable BEFORE feeding the face to any appearance-based gaze
estimator. The downstream network therefore only needs to learn gaze
relative to a *canonical* head pose, which is the single biggest
accuracy lever in modern gaze estimation pipelines.

LICENSE
-------
This file is Apache-2.0. It is a clean-room implementation derived
solely from the 2018 ETRA paper's published equations. No third-party
gaze-research code was copied or vendored. Mathematical algorithms
are not copyrightable; the implementation below is original.

What this buys you
------------------
* Major accuracy gain on tilted / off-axis heads.
* Robustness to camera distance variation (close-up portrait vs.
  full-body shot now produce equivalent gaze vectors).
* The roll component of head pose is REMOVED, so the rendered gaze
  arrow no longer wobbles when the subject tilts their head.

Public API
----------
* :func:`normalize_face_for_gaze(img_bgr, landmarks_478, image_size,
  cam=None, dist=None, focal_norm=960.0, distance_norm=600.0,
  roi_size=(224, 224))` -> :class:`NormalizedFace` or ``None``.
* :func:`denormalize_gaze(yaw_norm, pitch_norm, R_norm)` ->
  ``(yaw_cam, pitch_cam)`` — invert the rotation so the gaze vector
  is expressed in the original camera frame, ready for rendering with
  the renderer's screen-space convention.
* :class:`NormalizedFace`: dataclass with the warped image, the warp
  matrix ``W``, the rotation ``R_norm``, head pose ``(rvec, tvec)``,
  and the normalized landmark coordinates.

Sign convention
---------------
Matches the rest of this package and ``gaze_3d.py``:
* ``yaw`` > 0 -> subject looking to their *right*
* ``pitch`` > 0 -> looking *up*
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2_OK = False

# Re-use the canonical 6-point MediaPipe face model from gaze_3d so the
# PnP solve is identical and head poses are comparable.
try:
    from .gaze_3d import _MODEL_3D as _FACE_MODEL_3D  # type: ignore
    from .gaze_3d import MP_PNP_INDICES as _MP_INDICES  # type: ignore
except Exception:  # noqa: BLE001 - allow standalone import for unit tests
    _MP_INDICES = (1, 152, 33, 263, 57, 287)
    _FACE_MODEL_3D = np.asarray(
        [
            ( 0.0,    0.0,    0.0),
            ( 0.0,  -63.6,  -12.5),
            (-43.3,  32.7,  -26.0),
            ( 43.3,  32.7,  -26.0),
            (-28.9, -28.9,  -24.1),
            ( 28.9, -28.9,  -24.1),
        ],
        dtype=np.float64,
    )

logger = logging.getLogger(__name__)


@dataclass
class NormalizedFace:
    """Result of running :func:`normalize_face_for_gaze`."""

    image: np.ndarray
    """The warped face crop. dtype uint8, shape ``(roi_h, roi_w, 3)``,
    same BGR/RGB ordering as the input. Roll has been removed, the
    face center is at the principal point, and the apparent distance
    is ``distance_norm`` millimetres."""

    W: np.ndarray
    """``(3, 3)`` perspective warp used by ``cv2.warpPerspective``."""

    R_norm: np.ndarray
    """``(3, 3)`` rotation matrix from camera frame to normalized
    camera frame. To map a gaze vector measured in normalized space
    back to the camera frame multiply by ``R_norm.T``."""

    rvec: np.ndarray
    """``(3, 1)`` Rodrigues head rotation in the ORIGINAL camera frame
    (cv2.solvePnP output)."""

    tvec: np.ndarray
    """``(3, 1)`` head translation in the ORIGINAL camera frame, mm."""

    landmarks_norm: np.ndarray
    """``(6, 2)`` pixel coordinates of the 6 PnP landmarks AFTER warp.
    Useful for debugging: in a correctly normalized image the eye
    corners should sit on a horizontal line."""

    face_center_3d: np.ndarray
    """``(3,)`` 3D position of the face center in the original camera
    frame, mm."""


def _intrinsics_from_image(image_w: int, image_h: int) -> np.ndarray:
    """Synthetic pinhole intrinsics with focal == image width."""
    f = float(image_w)
    cx, cy = image_w * 0.5, image_h * 0.5
    return np.asarray(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _solve_head_pose(
    landmarks_px_6: np.ndarray,
    cam: np.ndarray,
    dist: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run PnP on the 6 canonical face landmarks.

    Returns ``(R_head, rvec, tvec)`` or ``None`` on failure.

    Strategy mirrors ETH-XGaze's ``demo.py``: try ``SOLVEPNP_EPNP`` for
    a fast initial estimate then refine with ``SOLVEPNP_ITERATIVE``.
    EPNP gives a numerically stable starting point even when the
    landmarks are nearly coplanar (which the 6-point set almost is for
    a frontal face), and the iterative pass tightens the residual.
    """
    if cv2 is None:
        return None
    try:
        pts = landmarks_px_6.astype(np.float64).reshape(-1, 1, 2)
        model = _FACE_MODEL_3D.reshape(-1, 1, 3)
        ok, rvec, tvec = cv2.solvePnP(
            model, pts, cam, dist, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok:
            return None
        # Refine with iterative method seeded by EPNP solution.
        try:
            ok2, rvec, tvec = cv2.solvePnP(
                model, pts, cam, dist, rvec, tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok2:
                # Keep the EPNP estimate if the refinement fails.
                pass
        except Exception:  # noqa: BLE001
            pass
        R, _ = cv2.Rodrigues(rvec)
        return R.astype(np.float64), rvec.astype(np.float64), tvec.astype(np.float64)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_pose_norm] solvePnP failed: %s", exc)
        return None


def normalize_face_for_gaze(
    img: np.ndarray,
    landmarks_478: np.ndarray,
    image_size: Optional[Tuple[int, int]] = None,
    cam: Optional[np.ndarray] = None,
    dist: Optional[np.ndarray] = None,
    focal_norm: float = 960.0,
    distance_norm: float = 600.0,
    roi_size: Tuple[int, int] = (224, 224),
) -> Optional[NormalizedFace]:
    """Apply pose-normalized data preprocessing to a face image.

    Parameters
    ----------
    img
        Input image, ``(H, W, 3)`` uint8. Either BGR or RGB — the warp
        is colour-agnostic, the caller must keep the same convention
        going into the downstream gaze network.
    landmarks_478
        MediaPipe FaceMesh landmarks. Either:
          * ``(N, 2)`` in pixel coordinates (full-frame), OR
          * ``(N, 3)`` with the third column being the normalized depth
            (ignored), OR
          * ``(N, 2)`` / ``(N, 3)`` in normalized [0,1] coords — the
            function will detect this by checking the maximum value and
            multiply by ``image_size`` if given. Pixel-space coords are
            assumed when ``max > 2.0``.
        ``N`` must be at least ``max(_MP_INDICES) + 1 = 288``.
    image_size
        ``(width, height)`` of the source image. Only required when
        ``landmarks_478`` is in normalized [0,1] space; ignored
        otherwise but defaults to ``img.shape[:2][::-1]`` when ``img``
        is supplied.
    cam, dist
        ``3x3`` intrinsic matrix and ``(4 or 5,)`` distortion vector. If
        ``cam`` is ``None`` we synthesise a pinhole with focal length
        equal to image width and principal point at image center; this
        is acceptable accuracy for relative head pose. If ``dist`` is
        ``None`` we assume zero distortion.
    focal_norm
        Synthetic camera focal length used for the normalized image, in
        the same units as ``cam``. Default 960 matches the original
        paper's face-normalization setting; use 1800 for eye-only
        normalization.
    distance_norm
        Synthetic camera-to-face distance in millimetres. Default 600
        matches the original paper's face-crop setting.
    roi_size
        Output crop size. The default (224, 224) is the standard input
        size for ResNet50-based gaze regressors and the L2CS-Net
        Gaze360 / MPIIGaze checkpoints.

    Returns
    -------
    A :class:`NormalizedFace` or ``None`` on any failure. The function
    is deliberately non-raising so callers can fall back to the
    un-normalized path without try/except boilerplate.
    """
    if not _CV2_OK or img is None or img.size == 0:
        return None
    if landmarks_478 is None or len(landmarks_478) <= max(_MP_INDICES):
        return None

    H_img, W_img = img.shape[:2]
    if image_size is None:
        image_size = (W_img, H_img)
    W_src, H_src = int(image_size[0]), int(image_size[1])
    if W_src <= 0 or H_src <= 0:
        return None

    # ---------- collect the 6 PnP landmarks in pixel space ----------
    pts = np.asarray(landmarks_478, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    pts_xy = pts[:, :2].copy()
    # Heuristic: normalized [0,1] vs. pixel.
    max_xy = float(np.nanmax(np.abs(pts_xy))) if pts_xy.size else 0.0
    if max_xy > 0.0 and max_xy <= 2.0:
        pts_xy[:, 0] *= W_src
        pts_xy[:, 1] *= H_src
    landmarks_px_6 = np.asarray(
        [pts_xy[i] for i in _MP_INDICES], dtype=np.float64
    )

    # ---------- camera intrinsics ----------
    if cam is None:
        cam = _intrinsics_from_image(W_src, H_src)
    cam = np.asarray(cam, dtype=np.float64)
    if dist is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1, 1)

    # ---------- PnP head pose in original camera frame ----------
    pose = _solve_head_pose(landmarks_px_6, cam, dist)
    if pose is None:
        return None
    hR, rvec, tvec = pose

    # ---------- 3D face center (avg of eye corners + mouth corners) ----------
    # In the canonical face model, indices 2..3 are eye outer corners and 4..5
    # are mouth corners. Project to camera frame as hR @ model + tvec.
    Fc = hR @ _FACE_MODEL_3D.T + tvec  # (3, 6)
    eye_center = np.mean(Fc[:, 2:4], axis=1).reshape(3, 1)
    mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape(3, 1)
    face_center = (eye_center + mouth_center) * 0.5  # (3, 1)
    distance = float(np.linalg.norm(face_center))
    if not math.isfinite(distance) or distance <= 1e-3:
        return None

    # ---------- build the normalization rotation R ----------
    # Per the 2018 ETRA paper's equations (3)-(6):
    #   forward = unit vector from camera to face_center (becomes new +Z)
    #   down    = forward x hR[:, 0], then normalised (new +Y, head's X
    #             stays horizontal in the rotated image so roll is removed)
    #   right   = down x forward, normalised (new +X)
    # R = [right; down; forward]^T  maps camera-frame vectors into
    # normalised-camera frame.
    forward = (face_center / distance).reshape(3)
    head_x = hR[:, 0]
    down = np.cross(forward, head_x)
    n_down = float(np.linalg.norm(down))
    if n_down < 1e-6:
        # Degenerate (head x-axis is parallel to the line of sight, i.e.
        # subject is looking straight down the camera axis with a 90deg
        # roll). Fall back to world up to keep the warp defined.
        down = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        n_down = 1.0
    down /= n_down
    right = np.cross(down, forward)
    n_right = float(np.linalg.norm(right))
    if n_right < 1e-6:
        return None
    right /= n_right
    R_norm = np.stack([right, down, forward], axis=0).astype(np.float64)

    # ---------- scaling so the face sits at distance_norm in the warped image ----------
    z_scale = distance_norm / distance
    S = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, z_scale]],
        dtype=np.float64,
    )

    # ---------- synthetic camera intrinsics for the normalized view ----------
    roi_w, roi_h = int(roi_size[0]), int(roi_size[1])
    cam_norm = np.asarray(
        [
            [focal_norm, 0.0, roi_w * 0.5],
            [0.0, focal_norm, roi_h * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ---------- compose the full warp ----------
    try:
        W_warp = cam_norm @ S @ R_norm @ np.linalg.inv(cam)
    except np.linalg.LinAlgError as exc:  # noqa: BLE001
        logger.debug("[gaze_pose_norm] inv(cam) failed: %s", exc)
        return None
    try:
        warped = cv2.warpPerspective(
            img, W_warp, (roi_w, roi_h), flags=cv2.INTER_LINEAR,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_pose_norm] warpPerspective failed: %s", exc)
        return None

    # ---------- propagate the 6 landmarks through the same warp ----------
    try:
        det_in = landmarks_px_6.reshape(-1, 1, 2)
        det_out = cv2.perspectiveTransform(det_in, W_warp).reshape(-1, 2)
    except Exception:  # noqa: BLE001
        det_out = landmarks_px_6.copy()

    return NormalizedFace(
        image=warped,
        W=W_warp.astype(np.float64),
        R_norm=R_norm,
        rvec=rvec,
        tvec=tvec,
        landmarks_norm=det_out.astype(np.float64),
        face_center_3d=face_center.reshape(3).astype(np.float64),
    )


def denormalize_gaze(
    yaw_norm: float,
    pitch_norm: float,
    R_norm: np.ndarray,
) -> Tuple[float, float]:
    """Map a (yaw, pitch) measured in normalized space back to camera frame.

    Inverts the rotation applied by :func:`normalize_face_for_gaze`. The
    gaze direction is treated as a unit vector with our package's sign
    convention:

        v_norm = (-sin(yaw) * cos(pitch),
                  -sin(pitch),
                  -cos(yaw) * cos(pitch))

    i.e. the OpenCV camera-frame forward-facing convention also used by
    L2CS-Net (``yaw>0`` rotates the vector toward subject's-right, which
    in camera coords is image-left, i.e. ``v_norm[0] < 0``). Returns
    the camera-frame (yaw, pitch) in the same convention.

    Returns
    -------
    ``(yaw_cam_rad, pitch_cam_rad)``
    """
    cp = math.cos(pitch_norm)
    sp = math.sin(pitch_norm)
    sy = math.sin(yaw_norm)
    cy = math.cos(yaw_norm)
    v_norm = np.asarray([-sy * cp, -sp, -cy * cp], dtype=np.float64)
    v_cam = R_norm.T @ v_norm  # inverse of the camera->normalized rotation
    # Re-extract yaw/pitch in the same convention.
    # v_cam = (-sin(yaw) cos(pitch), -sin(pitch), -cos(yaw) cos(pitch))
    # -> sin(pitch) = -v_cam[1]
    # -> tan(yaw)   = v_cam[0] / v_cam[2]   (signs cancel)
    sp_cam = float(np.clip(-v_cam[1], -1.0, 1.0))
    pitch_cam = math.asin(sp_cam)
    # Use the full 2-arg atan2 with the original signs preserved so the
    # quadrant is correct for back-facing gazes (which never occur in
    # practice but keep the math well-defined).
    yaw_cam = math.atan2(-v_cam[0], -v_cam[2])
    return yaw_cam, pitch_cam


def is_available() -> bool:
    """Return True iff OpenCV is importable. (numpy is a hard dep already.)"""
    return _CV2_OK


__all__ = [
    "NormalizedFace",
    "normalize_face_for_gaze",
    "denormalize_gaze",
    "is_available",
]
