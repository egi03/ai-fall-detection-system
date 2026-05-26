"""
OneEuro filter for low-latency keypoint smoothing.

The OneEuro filter (Casiez et al., 2012) is an adaptive low-pass filter
designed for noisy real-time signals like pose keypoints. Cutoff
frequency increases with the derivative magnitude, so slow motion is
heavily smoothed while fast motion passes through with minimal lag —
exactly what we want for fall detection, where impact dynamics must
survive while idle jitter is removed.

Applied to (x, y) of each keypoint independently. Z and visibility are
passed through untouched (z is approximate depth from MediaPipe and not
worth smoothing; visibility is already a confidence score).

This module is dependency-free (pure NumPy) and disabled by default in
config. Enable via `pose.smoothing.enabled: true`.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class OneEuroFilter:
    """Scalar OneEuro filter.

    The filter applies a low-pass to the signal whose cutoff frequency
    is adapted from the derivative of the signal:

        cutoff = min_cutoff + beta * |derivative|

    High derivative (fast motion) → higher cutoff → less lag.
    Low derivative (idle jitter) → lower cutoff → more smoothing.

    Parameters
    ----------
    min_cutoff : float
        Cutoff frequency (Hz) at zero velocity. Lower = more idle smoothing.
    beta : float
        Speed coefficient. Higher = more responsive to fast motion.
    d_cutoff : float
        Cutoff frequency for the derivative low-pass. Usually fixed at 1.0.
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        self._min_cutoff = float(min_cutoff)
        self._beta = float(beta)
        self._d_cutoff = float(d_cutoff)
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        """Filter a single scalar sample.

        Parameters
        ----------
        x : float
            Raw sample value.
        dt : float
            Time delta since the previous sample (seconds). Must be > 0.

        Returns
        -------
        float
            Filtered sample.
        """
        if dt <= 0.0:
            dt = 1e-3
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self._d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        """Clear filter state so the next sample re-initializes the filter."""
        self._x_prev = None
        self._dx_prev = 0.0


class KeypointSmoother:
    """OneEuro smoothing applied to every (x, y) channel of a keypoint array.

    One independent filter per (joint, axis). Other channels (z, visibility)
    are returned unchanged. The smoother handles its own time bookkeeping
    so callers only pass the latest keypoints — `dt` is computed from
    `fps` or from a monotonic clock if `use_wall_clock=True`.

    Parameters
    ----------
    num_keypoints : int
        Number of joints (e.g. 33 for MediaPipe, 17 for COCO/RTMPose).
    fps : float
        Nominal frame rate. Used as fallback dt when not using wall-clock.
    min_cutoff : float
        OneEuro min_cutoff (Hz). Default 1.0 is a good starting point for
        pose at 15–30 FPS in normalized coordinates.
    beta : float
        OneEuro speed coefficient. Higher = more responsive to motion.
    d_cutoff : float
        OneEuro derivative cutoff (Hz).
    use_wall_clock : bool
        If True, compute dt from time.monotonic() between calls. Use False
        for deterministic / offline replay.
    """

    def __init__(
        self,
        num_keypoints: int,
        fps: float = 15.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        use_wall_clock: bool = False,
    ) -> None:
        if num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be > 0, got {num_keypoints}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self._num_keypoints = num_keypoints
        self._dt_default = 1.0 / float(fps)
        self._use_wall_clock = bool(use_wall_clock)
        self._last_t: Optional[float] = None
        self._filters = [
            [
                OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
                for _ in range(2)
            ]
            for _ in range(num_keypoints)
        ]

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply OneEuro to (x, y) of every keypoint.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape (N, C) with C >= 2. Column 0 is x, column 1 is y.
            Any extra columns (z, visibility) are passed through unchanged.

        Returns
        -------
        np.ndarray
            Same shape as input. A new array; the input is not mutated.
        """
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            raise ValueError(
                f"keypoints must be 2D with at least 2 columns, got {keypoints.shape}"
            )
        if keypoints.shape[0] != self._num_keypoints:
            raise ValueError(
                f"expected {self._num_keypoints} keypoints, got {keypoints.shape[0]}"
            )

        dt = self._compute_dt()
        out = keypoints.copy()
        for j in range(self._num_keypoints):
            out[j, 0] = self._filters[j][0](float(keypoints[j, 0]), dt)
            out[j, 1] = self._filters[j][1](float(keypoints[j, 1]), dt)
        return out

    def _compute_dt(self) -> float:
        if not self._use_wall_clock:
            return self._dt_default
        import time

        now = time.monotonic()
        if self._last_t is None:
            self._last_t = now
            return self._dt_default
        dt = now - self._last_t
        self._last_t = now
        if dt <= 0.0:
            return self._dt_default
        return dt

    def reset(self) -> None:
        """Reset every internal filter and the wall-clock timer."""
        for row in self._filters:
            for f in row:
                f.reset()
        self._last_t = None
