"""
Per-frame motion / stillness monitor for post-fall severity assessment.

After a fall is confirmed, the system needs to distinguish between
three outcomes that matter for the caregiver/operator:

  - SEVERE   : subject is motionless for a sustained period after impact
               (suggests loss of consciousness or serious injury)
  - MODERATE : subject is still moving on the floor but has not recovered
               (suggests a fall with residual mobility)
  - MINOR    : subject got back up shortly after impact

Frame counting alone (as in the original FSM) cannot tell these apart.
This monitor computes the mean per-frame L2 displacement of valid
keypoints, smooths it over a short rolling window, and exposes
``motion_score`` and ``is_still`` properties.

The monitor is pose-format agnostic: it averages over only those
joints whose visibility / confidence exceeds the threshold. It treats
input coordinates as already in the [0, 1] normalized range (as
MediaPipe outputs) or in pixel coords (as RTMPose/YOLO-Pose output);
the caller is responsible for picking ``motion_threshold`` accordingly.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np


class MotionMonitor:
    """Rolling per-keypoint motion estimator.

    Each call to :meth:`update` measures the mean L2 displacement of
    valid keypoints between the current frame and the previous frame.
    Displacement is averaged over a rolling history window to produce a
    stable ``motion_score`` that drives the ``is_still`` flag.

    Parameters
    ----------
    history_frames : int
        Length of the rolling window over which displacement is averaged.
    motion_threshold : float
        Rolling-mean displacement below which the subject is considered
        still. Units match the input keypoint coords (e.g. 0.003 for
        normalized MediaPipe coords; ~3 pixels for pixel-space backends).
    confidence_threshold : float
        Minimum visibility / confidence (column 3 of the keypoint array)
        required to include a joint in the displacement average.
    min_valid_joints : int
        If fewer than this many joints are valid in either the current
        or previous frame, the displacement for that step is treated as
        missing (the rolling history is not advanced).
    """

    def __init__(
        self,
        history_frames: int = 15,
        motion_threshold: float = 0.003,
        confidence_threshold: float = 0.5,
        min_valid_joints: int = 4,
    ) -> None:
        if history_frames <= 0:
            raise ValueError(
                f"history_frames must be > 0, got {history_frames}"
            )
        if motion_threshold < 0:
            raise ValueError(
                f"motion_threshold must be >= 0, got {motion_threshold}"
            )
        if min_valid_joints <= 0:
            raise ValueError(
                f"min_valid_joints must be > 0, got {min_valid_joints}"
            )

        self._history: deque = deque(maxlen=int(history_frames))
        self._motion_threshold = float(motion_threshold)
        self._confidence_threshold = float(confidence_threshold)
        self._min_valid_joints = int(min_valid_joints)
        self._prev_keypoints: Optional[np.ndarray] = None
        self._still_streak = 0

    def update(self, keypoints: Optional[np.ndarray]) -> Optional[float]:
        """Measure displacement between this frame and the previous one.

        Parameters
        ----------
        keypoints : np.ndarray or None
            Shape (N, C) with C >= 3. Column 3 (if present) is treated
            as confidence / visibility; otherwise all joints are
            considered valid. Pass None when no person is detected.

        Returns
        -------
        float or None
            The instantaneous displacement for this step, or None if
            the displacement could not be computed (first frame, or
            too few valid joints).
        """
        if keypoints is None:
            # Skipping resets the previous-frame reference but not the
            # rolling history — a brief detection dropout should not
            # zero out the stillness signal.
            self._prev_keypoints = None
            return None

        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            raise ValueError(
                f"keypoints must be 2D with >=2 columns, got {keypoints.shape}"
            )

        if self._prev_keypoints is None or self._prev_keypoints.shape != keypoints.shape:
            self._prev_keypoints = keypoints.copy()
            return None

        # Valid joints = visible in both frames
        if keypoints.shape[1] >= 4:
            cur_valid = keypoints[:, 3] >= self._confidence_threshold
            prev_valid = self._prev_keypoints[:, 3] >= self._confidence_threshold
            mask = cur_valid & prev_valid
        else:
            mask = np.ones(keypoints.shape[0], dtype=bool)

        if int(mask.sum()) < self._min_valid_joints:
            self._prev_keypoints = keypoints.copy()
            return None

        delta = keypoints[mask, :2] - self._prev_keypoints[mask, :2]
        per_joint = np.linalg.norm(delta, axis=1)
        displacement = float(per_joint.mean())

        self._history.append(displacement)
        self._prev_keypoints = keypoints.copy()

        if self.motion_score <= self._motion_threshold:
            self._still_streak += 1
        else:
            self._still_streak = 0

        return displacement

    @property
    def motion_score(self) -> float:
        """Mean displacement over the rolling history window.

        Returns 0.0 when no history is available yet.
        """
        if not self._history:
            return 0.0
        return float(np.mean(self._history))

    @property
    def is_still(self) -> bool:
        """True iff the smoothed motion score is below the threshold.

        Returns False until the history buffer is non-empty.
        """
        if not self._history:
            return False
        return self.motion_score <= self._motion_threshold

    @property
    def still_streak(self) -> int:
        """Number of consecutive recent frames classified as still."""
        return self._still_streak

    @property
    def history_full(self) -> bool:
        """True once the rolling history window has filled."""
        return len(self._history) >= self._history.maxlen

    def reset(self) -> None:
        """Clear all state."""
        self._history.clear()
        self._prev_keypoints = None
        self._still_streak = 0


def classify_severity(
    still_seconds: float,
    recovered: bool,
    severe_seconds: float = 5.0,
    minor_seconds: float = 1.5,
) -> str:
    """Map post-fall stillness duration / recovery to a severity label.

    Parameters
    ----------
    still_seconds : float
        Continuous duration the subject has been still since impact.
    recovered : bool
        Whether the subject became upright (FSM RECOVERED) before the
        severity decision was finalised.
    severe_seconds : float
        Stillness duration that promotes the event to SEVERE.
    minor_seconds : float
        Recovery within this duration downgrades the event to MINOR.

    Returns
    -------
    str
        One of ``"SEVERE"``, ``"MODERATE"``, ``"MINOR"``.
    """
    if recovered:
        return "MINOR" if still_seconds <= minor_seconds else "MODERATE"
    if still_seconds >= severe_seconds:
        return "SEVERE"
    return "MODERATE"
