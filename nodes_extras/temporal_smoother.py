"""Temporal smoothing utilities for landmark sequences.

Extracted from Kanibus ``neural_pupil_tracker.py`` (KalmanFilter) and
``temporal_smoother.py`` (exponential-weight blending).  Adapted into
lightweight, dependency-minimal classes suitable for post-processing
any N-point landmark format (iBUG-68, MediaPipe-478, custom).

Only **numpy** is required.  No torch, no MediaPipe, no OpenCV.
"""

from __future__ import annotations

import copy
from typing import List, Literal, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# 1-D Kalman filter (constant-acceleration model)
# ---------------------------------------------------------------------------

class KalmanSmoother1D:
    """Scalar Kalman filter with a constant-acceleration motion model.

    State vector: ``[x, v, a]`` (position, velocity, acceleration).

    Adapted from the 6-DOF ``KalmanFilter`` in
    ``third_party/Kanibus/nodes/neural_pupil_tracker.py`` — reduced to 1-D
    so it can be applied independently to each axis of each landmark.

    Parameters
    ----------
    process_noise : tuple[float, float, float]
        Diagonal process-noise variances for (position, velocity, accel).
    measurement_noise : float
        Observation noise variance.
    dt : float
        Time-step between observations (seconds).  Default ``1/30`` (30 fps).
    """

    def __init__(
        self,
        process_noise: tuple[float, float, float] = (0.01, 0.1, 1.0),
        measurement_noise: float = 0.1,
        dt: float = 1.0 / 30.0,
    ) -> None:
        self._dt = dt

        # State [x, v, a]
        self.x = np.zeros(3, dtype=np.float64)

        # Covariance
        self.P = np.eye(3, dtype=np.float64) * 1000.0

        # Process noise
        self.Q = np.diag(np.array(process_noise, dtype=np.float64))

        # Measurement noise (scalar observation)
        self.R = np.array([[measurement_noise]], dtype=np.float64)

        # Transition matrix (constant-acceleration)
        d = self._dt
        self.F = np.array(
            [[1, d, 0.5 * d * d],
             [0, 1, d],
             [0, 0, 1]],
            dtype=np.float64,
        )

        # Observation matrix — we only observe position
        self.H = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    # -- public API ---------------------------------------------------------

    def predict(self) -> float:
        """Advance the state by one time-step.  Returns predicted position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0])

    def update(self, z: float) -> float:
        """Incorporate a new measurement *z*.  Returns corrected position."""
        z_arr = np.array([z], dtype=np.float64)
        y = z_arr - self.H @ self.x                       # innovation
        S = self.H @ self.P @ self.H.T + self.R           # innovation cov
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.x = self.x + (K @ y).ravel()
        self.P = self.P - K @ self.H @ self.P
        return float(self.x[0])

    def filter_value(self, z: float) -> float:
        """Convenience: predict → update in one call."""
        self.predict()
        return self.update(z)

    @property
    def position(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        return float(self.x[1])

    def reset(self) -> None:
        """Reset the filter to its initial (uninformed) state."""
        self.x[:] = 0.0
        self.P = np.eye(3, dtype=np.float64) * 1000.0


# ---------------------------------------------------------------------------
# Exponential moving average smoother
# ---------------------------------------------------------------------------

class TemporalSmootherEMA:
    """Per-axis exponential moving average for landmark smoothing.

    Inspired by the weighted-average blending in
    ``third_party/Kanibus/nodes/temporal_smoother.py``, but implemented as
    a simple EMA so it works frame-by-frame without buffering.

    Parameters
    ----------
    alpha : float
        Smoothing factor in ``(0, 1]``.  Smaller → smoother / more lag.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def filter_value(self, z: Union[float, np.ndarray]) -> np.ndarray:
        """Feed a new observation and return the smoothed value.

        *z* may be a scalar or an array of any shape; the internal state
        will match the shape of the first observation.
        """
        z = np.asarray(z, dtype=np.float64)
        if self._state is None:
            self._state = z.copy()
        else:
            self._state = self.alpha * z + (1.0 - self.alpha) * self._state
        return self._state.copy()

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def reset(self) -> None:
        self._state = None


# ---------------------------------------------------------------------------
# Convenience: smooth an entire landmark sequence at once
# ---------------------------------------------------------------------------

def smooth_landmarks(
    landmarks_sequence: Sequence[np.ndarray],
    method: Literal["kalman", "ema"] = "kalman",
    *,
    # Kalman params
    process_noise: tuple[float, float, float] = (0.01, 0.1, 1.0),
    measurement_noise: float = 0.1,
    dt: float = 1.0 / 30.0,
    # EMA params
    alpha: float = 0.4,
) -> List[np.ndarray]:
    """Smooth an entire sequence of landmark arrays.

    Parameters
    ----------
    landmarks_sequence : list[np.ndarray]
        Each element has shape ``(N, D)`` where *N* is the number of
        landmarks and *D* is the dimensionality (typically 2 or 3).
        Works with iBUG-68, MediaPipe-478, or any custom format.
    method : ``"kalman"`` | ``"ema"``
        Smoothing algorithm.
    process_noise, measurement_noise, dt
        Forwarded to :class:`KalmanSmoother1D` when *method* is ``"kalman"``.
    alpha
        Forwarded to :class:`TemporalSmootherEMA` when *method* is ``"ema"``.

    Returns
    -------
    list[np.ndarray]
        Smoothed landmark arrays with the same shapes as the inputs.
    """
    if not landmarks_sequence:
        return []

    first = np.asarray(landmarks_sequence[0], dtype=np.float64)
    n_points, n_dims = first.shape[0], first.shape[1] if first.ndim > 1 else 1
    flat = n_dims == 1 or first.ndim == 1

    if method == "kalman":
        filters = [
            [
                KalmanSmoother1D(
                    process_noise=process_noise,
                    measurement_noise=measurement_noise,
                    dt=dt,
                )
                for _ in range(n_dims)
            ]
            for _ in range(n_points)
        ]
        smoothed: List[np.ndarray] = []
        for lm in landmarks_sequence:
            arr = np.asarray(lm, dtype=np.float64)
            if flat:
                arr = arr.reshape(n_points, 1)
            out = np.empty_like(arr)
            for i in range(n_points):
                for j in range(n_dims):
                    out[i, j] = filters[i][j].filter_value(arr[i, j])
            if flat:
                out = out.ravel()
            smoothed.append(out)
        return smoothed

    elif method == "ema":
        smoother = TemporalSmootherEMA(alpha=alpha)
        return [smoother.filter_value(np.asarray(lm, dtype=np.float64))
                for lm in landmarks_sequence]

    else:
        raise ValueError(f"Unknown smoothing method: {method!r}")
