"""
Alarm triggering logic with debouncing and finite state machine.

Implements a multi-state FSM to transition from raw model predictions
to confirmed fall alarms, preventing false positives from transient
postures. Includes EMA smoothing and persistence counters.

Reference: research/9.2 - FSM with states: NORMAL, IMPACT_DETECTED,
FALL_CONFIRMED, RECOVERED. Debouncing via persistence counters.
Reference: research/5.2 - Persistence counter and N-seconds rule.
"""

import time
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FallState(Enum):
    """Finite state machine states for fall detection.

    ``IMMINENT`` is a pre-impact warning state entered when the smoothed
    confidence crosses ``warning_threshold`` but the high-confidence
    persistence gate has not yet fired. It exists to give the UI an
    early signal ("yellow") before the actual alarm goes red.
    """

    NORMAL = 0
    IMPACT_DETECTED = 1
    FALL_CONFIRMED = 2
    RECOVERED = 3
    IMMINENT = 4


class AlarmDetector:
    """
    Processes sequential model predictions to trigger stable alarms.

    Uses exponential moving average smoothing, persistence counting,
    and post-fall stillness verification before confirming a fall.

    Parameters
    ----------
    confidence_threshold : float
        Minimum smoothed confidence to consider a fall candidate.
    persistence_frames : int
        Required consecutive fall-positive frames for impact detection.
    cooldown_seconds : float
        Minimum time between consecutive alarms.
    stillness_duration : float
        Required post-impact stillness duration to confirm fall.
    ema_alpha : float
        EMA smoothing factor (lower = more damping).
    fps : float
        Expected frames per second (for stillness timing).
    warning_threshold : float, optional
        Lower confidence gate for entering the IMMINENT state. Should
        be < ``confidence_threshold``. When None, defaults to
        ``max(0.0, confidence_threshold - 0.20)``. Set equal to
        ``confidence_threshold`` to disable the IMMINENT signal.
    """

    # Hold RECOVERED state for this many frames so the UI can display it
    _RECOVERED_HOLD_FRAMES = 30  # ~2 seconds at 15 FPS

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        persistence_frames: int = 10,
        cooldown_seconds: float = 30.0,
        stillness_duration: float = 5.0,
        ema_alpha: float = 0.3,
        fps: float = 15.0,
        warning_threshold: float = None,
    ) -> None:
        self._conf_threshold = confidence_threshold
        self._persistence_required = persistence_frames
        self._cooldown_seconds = cooldown_seconds
        self._stillness_frames = int(stillness_duration * fps)
        self._ema_alpha = ema_alpha
        self._recovered_hold = int(fps * 2)  # 2 seconds at configured FPS
        if warning_threshold is None:
            warning_threshold = max(0.0, confidence_threshold - 0.20)
        # Warning gate cannot exceed the confirmation gate.
        self._warning_threshold = min(warning_threshold, confidence_threshold)

        self._state = FallState.NORMAL
        self._smoothed_conf = 0.0
        self._persistence_counter = 0
        self._stillness_counter = 0
        self._recovered_counter = 0
        self._last_alarm_time = 0.0
        self._peak_confidence = 0.0

    def process_frame(
        self,
        raw_confidence: float,
        is_subject_upright: bool,
    ) -> FallState:
        """
        Process a single frame's prediction through the FSM.

        Parameters
        ----------
        raw_confidence : float
            Raw model fall probability [0.0, 1.0].
        is_subject_upright : bool
            Whether the subject's posture indicates upright position
            (derived from aspect ratio or torso angle).

        Returns
        -------
        FallState
            Current state of the alarm FSM.
        """
        # EMA smoothing
        self._smoothed_conf = (
            self._ema_alpha * raw_confidence
            + (1 - self._ema_alpha) * self._smoothed_conf
        )

        is_fall_candidate = self._smoothed_conf >= self._conf_threshold
        now = time.monotonic()

        if self._state in (FallState.NORMAL, FallState.IMMINENT):
            if is_fall_candidate:
                self._persistence_counter += 1
                if self._persistence_counter >= self._persistence_required:
                    # Check cooldown
                    if now - self._last_alarm_time >= self._cooldown_seconds:
                        self._state = FallState.IMPACT_DETECTED
                        self._stillness_counter = 0
                        self._peak_confidence = self._smoothed_conf
                        logger.info(
                            "Impact detected",
                            extra={"confidence": self._smoothed_conf},
                        )
                    else:
                        self._persistence_counter = 0
                        self._state = FallState.NORMAL
                else:
                    # High confidence but persistence not yet met → warn.
                    self._state = FallState.IMMINENT
            elif self._smoothed_conf >= self._warning_threshold:
                # Rising probability but below the confirmation gate.
                self._state = FallState.IMMINENT
            else:
                self._persistence_counter = 0
                self._state = FallState.NORMAL

        elif self._state == FallState.IMPACT_DETECTED:
            self._peak_confidence = max(
                self._peak_confidence, self._smoothed_conf
            )
            if is_subject_upright:
                # Subject recovered before confirmation
                self._state = FallState.NORMAL
                self._persistence_counter = 0
                self._stillness_counter = 0
                logger.info("Impact cleared — subject upright")
            else:
                self._stillness_counter += 1
                if self._stillness_counter >= self._stillness_frames:
                    self._state = FallState.FALL_CONFIRMED
                    self._last_alarm_time = now
                    logger.info(
                        "Fall confirmed",
                        extra={"peak_confidence": self._peak_confidence},
                    )

        elif self._state == FallState.FALL_CONFIRMED:
            if is_subject_upright:
                self._state = FallState.RECOVERED
                self._persistence_counter = 0
                self._stillness_counter = 0
                self._recovered_counter = 0
                logger.info("Subject recovered from fall")

        elif self._state == FallState.RECOVERED:
            # Hold RECOVERED for a visible duration before returning to NORMAL
            self._recovered_counter += 1
            if self._recovered_counter >= self._recovered_hold:
                self._state = FallState.NORMAL
                self._persistence_counter = 0
                self._smoothed_conf = 0.0

        return self._state

    def reset(self) -> None:
        """Reset the FSM to NORMAL state."""
        self._state = FallState.NORMAL
        self._smoothed_conf = 0.0
        self._persistence_counter = 0
        self._stillness_counter = 0
        self._recovered_counter = 0
        self._peak_confidence = 0.0

    @property
    def is_alarm_active(self) -> bool:
        """Whether a confirmed fall alarm is currently active."""
        return self._state == FallState.FALL_CONFIRMED

    @property
    def is_imminent(self) -> bool:
        """Whether the FSM is in the pre-impact warning state."""
        return self._state == FallState.IMMINENT

    @property
    def warning_threshold(self) -> float:
        """Lower confidence gate that triggers IMMINENT."""
        return self._warning_threshold

    @property
    def current_state(self) -> FallState:
        """Return the current FSM state."""
        return self._state

    @property
    def smoothed_confidence(self) -> float:
        """Return the current EMA-smoothed confidence."""
        return self._smoothed_conf

    @property
    def peak_confidence(self) -> float:
        """Return the peak confidence during current event."""
        return self._peak_confidence
