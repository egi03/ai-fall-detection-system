"""Tests for the post-fall MotionMonitor and severity classifier."""

import numpy as np
import pytest

from src.alarm.motion_monitor import MotionMonitor, classify_severity


def _kps(n=17, coord=0.5, conf=1.0):
    """Build a keypoint array (N, 4) with constant coords and confidence."""
    arr = np.zeros((n, 4), dtype=np.float32)
    arr[:, 0] = coord
    arr[:, 1] = coord
    arr[:, 3] = conf
    return arr


class TestMotionMonitor:
    def test_first_frame_returns_none(self):
        m = MotionMonitor()
        assert m.update(_kps()) is None
        assert m.motion_score == 0.0

    def test_none_input_clears_prev(self):
        m = MotionMonitor()
        m.update(_kps(coord=0.4))
        # Drop a frame — next non-None call re-initializes prev
        assert m.update(None) is None
        assert m.update(_kps(coord=0.4)) is None  # still no displacement available

    def test_static_subject_is_still(self):
        m = MotionMonitor(history_frames=10, motion_threshold=0.001)
        for _ in range(12):
            m.update(_kps(coord=0.5))
        assert m.is_still
        assert m.motion_score < 0.001
        assert m.still_streak >= 10

    def test_moving_subject_not_still(self):
        m = MotionMonitor(history_frames=10, motion_threshold=0.001)
        for i in range(15):
            m.update(_kps(coord=0.5 + 0.05 * i))
        assert not m.is_still
        assert m.motion_score > 0.001
        assert m.still_streak == 0

    def test_motion_then_stillness_recovers_streak(self):
        m = MotionMonitor(history_frames=5, motion_threshold=0.002)
        for i in range(5):
            m.update(_kps(coord=0.5 + 0.05 * i))
        assert not m.is_still
        # Now hold still for enough frames to flush history
        for _ in range(10):
            m.update(_kps(coord=0.7))
        assert m.is_still
        assert m.still_streak >= 5

    def test_low_confidence_joints_excluded(self):
        m = MotionMonitor(history_frames=5, motion_threshold=0.002, min_valid_joints=4)
        # First frame all valid
        m.update(_kps(coord=0.5, conf=1.0))
        # Second frame: most joints invalid, but 4 visible static joints
        kps = _kps(coord=0.5, conf=0.1)
        kps[:4, 3] = 1.0
        assert m.update(kps) == pytest.approx(0.0, abs=1e-6)

    def test_too_few_valid_joints_returns_none(self):
        m = MotionMonitor(min_valid_joints=10)
        m.update(_kps(conf=1.0))
        kps = _kps(conf=0.1)  # all hidden
        assert m.update(kps) is None

    def test_motion_score_uses_l2_norm(self):
        m = MotionMonitor(history_frames=1, motion_threshold=0.0)
        m.update(_kps(coord=0.0))
        # Move every joint by (0.03, 0.04) → L2 = 0.05
        kps = _kps(coord=0.0)
        kps[:, 0] = 0.03
        kps[:, 1] = 0.04
        d = m.update(kps)
        assert d == pytest.approx(0.05, abs=1e-6)

    def test_history_full_property(self):
        m = MotionMonitor(history_frames=3, motion_threshold=0.01)
        m.update(_kps(coord=0.0))
        m.update(_kps(coord=0.01))
        assert not m.history_full
        m.update(_kps(coord=0.02))
        m.update(_kps(coord=0.03))
        assert m.history_full

    def test_reset_clears(self):
        m = MotionMonitor(history_frames=3)
        for c in (0.0, 0.1, 0.2):
            m.update(_kps(coord=c))
        m.reset()
        assert m.motion_score == 0.0
        assert m.still_streak == 0
        assert not m.history_full

    def test_invalid_keypoint_shape_raises(self):
        m = MotionMonitor()
        m.update(_kps())
        with pytest.raises(ValueError):
            m.update(np.zeros((17,), dtype=np.float32))

    def test_invalid_init_args(self):
        with pytest.raises(ValueError):
            MotionMonitor(history_frames=0)
        with pytest.raises(ValueError):
            MotionMonitor(motion_threshold=-0.1)
        with pytest.raises(ValueError):
            MotionMonitor(min_valid_joints=0)

    def test_keypoint_shape_change_resets_prev(self):
        # E.g. backend switches from MediaPipe (33) to COCO (17)
        m = MotionMonitor()
        m.update(_kps(n=33))
        # Different shape: should silently re-prime, not crash
        assert m.update(_kps(n=17)) is None
        assert m.update(_kps(n=17)) == pytest.approx(0.0, abs=1e-6)


class TestClassifySeverity:
    def test_recovered_quickly_is_minor(self):
        assert classify_severity(still_seconds=0.5, recovered=True) == "MINOR"

    def test_recovered_late_is_moderate(self):
        assert classify_severity(still_seconds=4.0, recovered=True) == "MODERATE"

    def test_long_stillness_no_recovery_is_severe(self):
        assert classify_severity(still_seconds=6.0, recovered=False) == "SEVERE"

    def test_short_stillness_no_recovery_is_moderate(self):
        assert classify_severity(still_seconds=2.0, recovered=False) == "MODERATE"

    def test_thresholds_respected(self):
        assert (
            classify_severity(
                still_seconds=3.0, recovered=False, severe_seconds=2.0
            )
            == "SEVERE"
        )
        assert (
            classify_severity(
                still_seconds=0.5, recovered=True, minor_seconds=0.4
            )
            == "MODERATE"
        )
