# -*- coding: utf-8 -*-
"""Stage-1 gaze upgrade: head-pose correction (solvePnP) + 4-state Kalman.

The MediaPipe blend-shape gaze pipeline returns yaw/pitch *in head frame*
(i.e. eye-in-head rotation). To draw an accurate gaze arrow on the image
plane we must compose the eye rotation with the head rotation and re-
project onto the camera. Without that composition, a subject looking
forward but rotating their head left makes the rendered arrow stay
centred — visually wrong.

This module is pure, dependency-light (numpy + cv2 only) and is the
default when ``gaze_engine='blendshape_head_corrected'``.

Public API
----------
* :func:`estimate_head_pose(landmarks_norm, image_size)`
* :func:`world_gaze_from_eye_in_head(yaw_eye, pitch_eye, R_head)`
* :class:`AngleKalman2D`

Sign conventions match the rest of the package:
* ``yaw`` > 0  -> subject looking to their *right*
* ``pitch`` > 0 -> looking *up*
* image-space ``dx`` > 0 -> arrow tip drawn to the right of the iris
* image-space ``dy`` > 0 -> arrow tip drawn below the iris
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    _CV2_OK = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical 3D face model (millimetres) for 6 MediaPipe landmarks.
# Values from the standard MediaPipe canonical_face_model.obj, picked to
# give a numerically well-conditioned PnP problem.
# ---------------------------------------------------------------------------
# MP indices used:
#   1   nose tip
#   152 chin
#   33  right eye outer corner   (subject's right, image left)
#   263 left eye outer corner    (subject's left,  image right)
#   57  right mouth corner
#   287 left mouth corner
MP_PNP_INDICES: Tuple[int, ...] = (1, 152, 33, 263, 57, 287)
_MODEL_3D: np.ndarray = np.asarray(
    [
        ( 0.0,    0.0,    0.0),     # nose tip (origin)
        ( 0.0,   -63.6,  -12.5),    # chin
        (-43.3,   32.7,  -26.0),    # right eye outer
        ( 43.3,   32.7,  -26.0),    # left eye outer
        (-28.9,  -28.9,  -24.1),    # right mouth corner
        ( 28.9,  -28.9,  -24.1),    # left mouth corner
    ],
    dtype=np.float64,
)


def _intrinsics(image_w: int, image_h: int) -> np.ndarray:
    """Approximate pinhole intrinsics: focal = image width, principal point
    at image centre. Good enough for a relative-rotation PnP solve."""
    f = float(image_w)
    cx, cy = image_w * 0.5, image_h * 0.5
    return np.asarray(
        [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def estimate_head_pose(
    landmarks_norm: np.ndarray,
    image_size: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run solvePnP on the 6 canonical MediaPipe face landmarks.

    Parameters
    ----------
    landmarks_norm
        ``(N, 3)`` float32 array of MediaPipe face landmarks normalised to
        the model's own crop space (x in [0,1], y in [0,1]). N must be
        >= 288 so all PnP indices exist.
    image_size
        ``(width, height)`` in pixels of the FULL frame the landmarks
        were re-projected onto. The caller is expected to have already
        scaled ``landmarks_norm[:,:2]`` such that multiplying by
        ``(width, height)`` lands on the right pixel — see callers in
        ``nodes.py`` (``_run_face_landmarker_on_face_crop`` returns a
        ``kps68_norm`` already in *full frame* coords, while the raw
        ``landmarks_norm`` from MediaPipe is in *crop* coords; this
        helper expects FULL-FRAME normalised coords).

    Returns
    -------
    ``(R_head, t_head)`` or ``None`` on failure.
        ``R_head`` is a 3x3 rotation matrix mapping head-frame vectors to
        camera frame. ``t_head`` is the translation (mm) of the head
        origin in camera frame.
    """
    if not _CV2_OK or landmarks_norm is None:
        return None
    if landmarks_norm.shape[0] <= max(MP_PNP_INDICES):
        return None
    W, H = int(image_size[0]), int(image_size[1])
    if W <= 0 or H <= 0:
        return None

    pts_px = np.empty((len(MP_PNP_INDICES), 2), dtype=np.float64)
    for i, mp_idx in enumerate(MP_PNP_INDICES):
        pts_px[i, 0] = float(landmarks_norm[mp_idx, 0]) * W
        pts_px[i, 1] = float(landmarks_norm[mp_idx, 1]) * H

    K = _intrinsics(W, H)
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D, pts_px, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_3d] solvePnP failed: %s", exc)
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.astype(np.float64)


def estimate_head_pose_from_pixels(
    pts_px_478: np.ndarray,
    image_size: Tuple[int, int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Same as :func:`estimate_head_pose` but takes already-pixel-space
    MediaPipe landmarks (full-frame). Convenient when the caller has
    already mapped crop-space landmarks to full-frame pixels."""
    if not _CV2_OK or pts_px_478 is None:
        return None
    if pts_px_478.shape[0] <= max(MP_PNP_INDICES):
        return None
    W, H = int(image_size[0]), int(image_size[1])
    if W <= 0 or H <= 0:
        return None
    pts_px = np.asarray(
        [pts_px_478[i] for i in MP_PNP_INDICES], dtype=np.float64
    )
    K = _intrinsics(W, H)
    dist = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_3D, pts_px, K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[gaze_3d] solvePnP failed: %s", exc)
        return None
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float64), tvec.astype(np.float64)


def world_gaze_from_eye_in_head(
    yaw_eye_rad: float,
    pitch_eye_rad: float,
    R_head: Optional[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Compose an eye-in-head rotation with the head rotation and return
    both the world-frame yaw/pitch and the screen-space (dx, dy).

    The eye direction in head frame. MediaPipe canonical face model has
    +X to subject's LEFT, +Y UP, +Z OUT OF THE FACE (toward viewer when
    the subject is facing the camera). Our sign convention is yaw>0 =
    subject's right, pitch>0 = up. So:

        v_head = ( -sin(yaw), -sin(pitch), +cos(yaw)*cos(pitch) )

    Component-by-component: yaw>0 -> subject's right -> -X in model
    (since +X is subject's LEFT). pitch>0 -> up in head -> +Y in model,
    but we want pitch>0 to make dy<0 (arrow up in image, where +Y is
    down), and ``R_head`` is a camera-frame matrix where +Y is image
    DOWN, so flipping pitch sign here keeps everything consistent.

    Returns
    -------
    ``(dx_screen, dy_screen, yaw_world_rad, pitch_world_rad)``
        ``dx``>0 = arrow tip drawn to image right, ``dy``>0 = drawn
        below the iris (OpenCV image-y convention).
    """
    cy = math.cos(yaw_eye_rad)
    sy = math.sin(yaw_eye_rad)
    cp = math.cos(pitch_eye_rad)
    sp = math.sin(pitch_eye_rad)
    v_head = np.asarray([-sy * cp, -sp, cy * cp], dtype=np.float64)

    if R_head is None:
        v_cam = v_head
    else:
        v_cam = R_head @ v_head

    # Project onto image plane: positive image x = right, positive image
    # y = down (OpenCV convention). v_cam already lives in that frame.
    nrm = float(np.linalg.norm(v_cam[:2]))
    if nrm < 1e-6:
        dx, dy = 0.0, 0.0
    else:
        dx = float(v_cam[0] / nrm)
        dy = float(v_cam[1] / nrm)

    # World-frame yaw/pitch (useful for downstream consumers + Kalman).
    # For a forward-facing subject with R_head = identity-ish, v_cam[2]
    # is NEGATIVE (face-forward direction in camera frame points toward
    # the camera = -Z). yaw>0 = subject's right; subject's right
    # corresponds to image-left = -X in camera frame, so flip the sign
    # on v_cam[0].
    xz = math.hypot(v_cam[0], v_cam[2])
    yaw_world = math.atan2(-v_cam[0], -v_cam[2])
    pitch_world = -math.atan2(v_cam[1], xz) if xz > 1e-9 else 0.0
    return dx, dy, yaw_world, pitch_world


# ---------------------------------------------------------------------------
# 4-state Kalman filter on (yaw, pitch, yaw_rate, pitch_rate).
# Used to smooth the world-frame gaze stream across frames. Much better
# than One-Euro for predicting *through* saccades (it has a velocity
# model) without lagging.
# ---------------------------------------------------------------------------
class AngleKalman2D:
    """Constant-velocity Kalman for a 2D angle (yaw, pitch).

    State vector: ``[yaw, pitch, yaw_rate, pitch_rate]``.
    Process noise scales linearly with ``dt`` and the configurable
    ``process_std`` (rad/s). Measurement noise is the per-frame angle
    standard deviation ``meas_std`` (rad).
    """

    __slots__ = ("dt", "x", "P", "Q", "R", "_initialised")

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_std: float = 0.8,  # rad/s — typical saccade velocity scale
        meas_std: float = 0.05,    # rad   — ~3deg per-frame noise
    ) -> None:
        self.dt = float(dt)
        self.x = np.zeros((4, 1), dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 1.0
        q = float(process_std) ** 2
        # Discrete white-noise acceleration model.
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3 / 2.0
        dt4 = self.dt ** 4 / 4.0
        self.Q = np.asarray(
            [
                [dt4, 0.0, dt3, 0.0],
                [0.0, dt4, 0.0, dt3],
                [dt3, 0.0, dt2, 0.0],
                [0.0, dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        ) * q
        r = float(meas_std) ** 2
        self.R = np.asarray([[r, 0.0], [0.0, r]], dtype=np.float64)
        self._initialised = False

    def reset(self) -> None:
        self.x[:] = 0.0
        self.P = np.eye(4, dtype=np.float64) * 1.0
        self._initialised = False

    def step(self, yaw_meas: float, pitch_meas: float) -> Tuple[float, float]:
        """Predict + update on one frame. Returns smoothed (yaw, pitch)."""
        if not self._initialised:
            self.x[0, 0] = float(yaw_meas)
            self.x[1, 0] = float(pitch_meas)
            self._initialised = True
            return float(yaw_meas), float(pitch_meas)

        # Predict.
        dt = self.dt
        F = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Update.
        H = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64
        )
        z = np.asarray([[float(yaw_meas)], [float(pitch_meas)]], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float64) - K @ H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


__all__ = [
    "estimate_head_pose",
    "estimate_head_pose_from_pixels",
    "world_gaze_from_eye_in_head",
    "AngleKalman2D",
    "MP_PNP_INDICES",
]
