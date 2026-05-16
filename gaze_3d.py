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


def screen_dx_dy_from_camera_yaw_pitch(
    yaw_rad: float, pitch_rad: float,
) -> Tuple[float, float]:
    """Render a CAMERA-frame (yaw, pitch) directly to screen-space dx/dy.

    This is what L2CS-Net does in its own visualizer. The output gaze
    vector in camera frame (OpenCV: +X right, +Y down, +Z into scene)
    is:

        vx = -sin(yaw)*cos(pitch)
        vy = -sin(pitch)
        vz = -cos(yaw)*cos(pitch)

    so the in-plane projection is ``dx = vx, dy = vy``. Returns a unit
    vector (``hypot(dx, dy) == 1``) unless the gaze is exactly along
    the optical axis, in which case ``(0, 0)`` is returned.
    """
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    sy = math.sin(yaw_rad)
    dx = -sy * cp
    dy = -sp
    nrm = math.hypot(dx, dy)
    if nrm < 1e-9:
        return 0.0, 0.0
    return dx / nrm, dy / nrm


def world_gaze_from_eye_in_head(
    yaw_eye_rad: float,
    pitch_eye_rad: float,
    R_head: Optional[np.ndarray],
) -> Tuple[float, float, float, float]:
    """Compose an eye-in-head rotation with the head rotation and return
    both the world-frame yaw/pitch and the screen-space (dx, dy).

    MediaPipe canonical face model frame: +X to subject's LEFT, +Y UP,
    +Z OUT OF THE FACE (toward the viewer when the subject is facing
    the camera). Our sign convention is yaw>0 = subject's right and
    pitch>0 = up. In the MODEL frame the eye gaze vector is therefore:

        v_model = ( -sin(yaw)*cos(pitch),     # yaw>0 -> subject's right
                    +sin(pitch),              # pitch>0 -> +Y_model (up)
                    +cos(yaw)*cos(pitch) )    # forward -> +Z_model

    ``R_head`` from cv2.solvePnP maps model points to OpenCV CAMERA
    frame (+X right, +Y down, +Z into scene). For a forward-facing
    subject R_head ~ diag(1, -1, -1), which flips the Y and Z signs so
    the camera-frame vector becomes
    ``(-sin(yaw)*cos(pitch), -sin(pitch), -cos(yaw)*cos(pitch))``,
    matching L2CS-Net's direct rendering formula. When ``R_head`` is
    None we still want the SAME camera-frame output (so the function is
    consistent with the L2CS direct path), so we apply that flip
    explicitly.

    Returns
    -------
    ``(dx_screen, dy_screen, yaw_world_rad, pitch_world_rad)``
        ``dx``>0 = arrow tip drawn to image right, ``dy``>0 = drawn
        below the iris (OpenCV image-y convention).
        ``yaw_world > 0`` = subject's right (== screen dx < 0).
        ``pitch_world > 0`` = up (== screen dy < 0).
    """
    cy = math.cos(yaw_eye_rad)
    sy = math.sin(yaw_eye_rad)
    cp = math.cos(pitch_eye_rad)
    sp = math.sin(pitch_eye_rad)
    # MODEL frame: +Y up, +Z out of face. yaw>0 -> subject's right.
    v_model = np.asarray([-sy * cp, sp, cy * cp], dtype=np.float64)

    if R_head is None:
        # Default "facing camera" rotation: model +X -> cam +X (subject's
        # left = image right), model +Y -> cam -Y (up -> image up),
        # model +Z -> cam -Z (out of face -> toward camera).
        v_cam = np.asarray([v_model[0], -v_model[1], -v_model[2]], dtype=np.float64)
    else:
        v_cam = R_head @ v_model

    # Project onto image plane (OpenCV: +x right, +y down).
    nrm_xy = float(math.hypot(v_cam[0], v_cam[1]))
    if nrm_xy < 1e-6:
        dx, dy = 0.0, 0.0
    else:
        dx = float(v_cam[0] / nrm_xy)
        dy = float(v_cam[1] / nrm_xy)

    # World-frame yaw/pitch. Camera-frame forward gaze is -Z_cam, so a
    # subject looking straight at the camera has v_cam ~ (0,0,-1) and
    # yaw_world=pitch_world=0. yaw>0 (subject's right) gives
    # v_cam[0]<0, so atan2(-vx, -vz) returns +yaw.
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
    "screen_dx_dy_from_camera_yaw_pitch",
    "AngleKalman2D",
    "MP_PNP_INDICES",
]
