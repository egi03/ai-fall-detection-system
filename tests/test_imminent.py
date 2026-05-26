"""Tests for the IMMINENT pre-fall warning state."""

import pytest

from src.alarm.detector import AlarmDetector, FallState


def _ad(**overrides):
    defaults = dict(
        confidence_threshold=0.7,
        persistence_frames=3,
        cooldown_seconds=0.0,
        stillness_duration=0.2,
        ema_alpha=1.0,
        fps=10.0,
    )
    defaults.update(overrides)
    return AlarmDetector(**defaults)


class TestImminentState:
    def test_default_warning_threshold(self):
        ad = _ad(confidence_threshold=0.7)
        assert ad.warning_threshold == pytest.approx(0.5)

    def test_warning_threshold_clamped_to_confidence(self):
        # Pathological config: warning > confidence → clamp.
        ad = _ad(confidence_threshold=0.4, warning_threshold=0.9)
        assert ad.warning_threshold == pytest.approx(0.4)

    def test_low_confidence_stays_normal(self):
        ad = _ad()  # warning=0.5
        for _ in range(5):
            state = ad.process_frame(0.3, is_subject_upright=True)
        assert state == FallState.NORMAL
        assert not ad.is_imminent

    def test_rising_below_threshold_enters_imminent(self):
        ad = _ad()  # warning=0.5, conf=0.7
        state = ad.process_frame(0.55, is_subject_upright=True)
        assert state == FallState.IMMINENT
        assert ad.is_imminent

    def test_imminent_returns_to_normal_when_drops(self):
        ad = _ad()
        ad.process_frame(0.55, is_subject_upright=True)
        assert ad.is_imminent
        state = ad.process_frame(0.2, is_subject_upright=True)
        assert state == FallState.NORMAL

    def test_high_confidence_first_frame_is_imminent(self):
        # persistence=3, so first high-conf frame should be IMMINENT not IMPACT
        ad = _ad(persistence_frames=3)
        state = ad.process_frame(0.9, is_subject_upright=False)
        assert state == FallState.IMMINENT

    def test_persistence_advances_through_imminent(self):
        ad = _ad(persistence_frames=3, confidence_threshold=0.7)
        states = [
            ad.process_frame(0.9, is_subject_upright=False)
            for _ in range(3)
        ]
        # Frames 1-2: IMMINENT (counter < 3)
        assert states[0] == FallState.IMMINENT
        assert states[1] == FallState.IMMINENT
        # Frame 3: counter == 3 → IMPACT_DETECTED
        assert states[2] == FallState.IMPACT_DETECTED

    def test_persistence_one_skips_imminent(self):
        # With persistence=1, a single high-conf frame triggers IMPACT directly
        ad = _ad(persistence_frames=1, confidence_threshold=0.7)
        state = ad.process_frame(0.9, is_subject_upright=False)
        assert state == FallState.IMPACT_DETECTED

    def test_disabling_imminent_via_threshold_equality(self):
        # warning == confidence ⇒ no IMMINENT phase below confidence
        ad = _ad(
            confidence_threshold=0.7,
            warning_threshold=0.7,
            persistence_frames=2,
        )
        # Mid-range value: not >= 0.7 → NORMAL
        state = ad.process_frame(0.55, is_subject_upright=True)
        assert state == FallState.NORMAL

    def test_partial_persistence_then_drop_resets_counter(self):
        # IMMINENT entered, then conf drops → counter must reset so next spike
        # starts from zero (no carry-over from prior high frames).
        ad = _ad(persistence_frames=3, confidence_threshold=0.7)
        ad.process_frame(0.9, is_subject_upright=False)  # IMMINENT, ct=1
        ad.process_frame(0.9, is_subject_upright=False)  # IMMINENT, ct=2
        ad.process_frame(0.2, is_subject_upright=True)   # NORMAL, ct=0
        # Now need full persistence again
        states = [
            ad.process_frame(0.9, is_subject_upright=False) for _ in range(3)
        ]
        assert states[0] == FallState.IMMINENT
        assert states[1] == FallState.IMMINENT
        assert states[2] == FallState.IMPACT_DETECTED

    def test_reset_clears_imminent(self):
        ad = _ad()
        ad.process_frame(0.55, is_subject_upright=True)
        assert ad.is_imminent
        ad.reset()
        assert ad.current_state == FallState.NORMAL
        assert not ad.is_imminent
