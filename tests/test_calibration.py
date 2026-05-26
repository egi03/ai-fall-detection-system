"""Tests for per-camera baseline calibration."""

import numpy as np
import pytest

from src.features.calibration import BaselineCalibrator


def _fv(values, num_features=15):
    """Build a feature vector with given (index, value) pairs."""
    out = np.zeros(num_features, dtype=np.float32)
    for idx, v in values:
        out[idx] = v
    return out


class TestBaselineCalibrator:
    def test_initial_state(self):
        cal = BaselineCalibrator(feature_indices=[3, 4], warmup_frames=10)
        assert not cal.is_ready
        assert cal.progress == 0.0
        assert cal.baseline_mean is None

    def test_progress_advances(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=10)
        for _ in range(5):
            cal.observe(_fv([(3, 0.1)]))
        assert cal.progress == pytest.approx(0.5)
        assert not cal.is_ready

    def test_ready_after_warmup(self):
        cal = BaselineCalibrator(feature_indices=[3, 4], warmup_frames=10)
        rng = np.random.default_rng(0)
        for _ in range(10):
            cal.observe(_fv([(3, rng.normal(0.0, 0.01)), (4, rng.normal(0.0, 0.02))]))
        assert cal.is_ready
        assert cal.progress == 1.0
        assert cal.baseline_mean is not None
        assert cal.baseline_mean.shape == (2,)

    def test_apply_noop_before_ready(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=10)
        f = _fv([(3, 0.5)])
        out = cal.apply(f)
        np.testing.assert_array_equal(out, f)

    def test_apply_suppresses_within_band(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=20, k=2.0)
        rng = np.random.default_rng(1)
        # baseline ~ N(0, 0.01)
        for _ in range(20):
            cal.observe(_fv([(3, rng.normal(0.0, 0.01))]))
        assert cal.is_ready
        # Test value within ~1σ of baseline mean is pulled to mean
        out = cal.apply(_fv([(3, 0.005)]))
        assert abs(out[3] - float(cal.baseline_mean[0])) < 1e-6

    def test_apply_preserves_large_signal(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=20, k=1.5)
        rng = np.random.default_rng(2)
        for _ in range(20):
            cal.observe(_fv([(3, rng.normal(0.0, 0.01))]))
        # Real fall: velocity ~0.5, well outside ±1.5σ band
        original = 0.5
        out = cal.apply(_fv([(3, original)]))
        # Soft thresholding shifts by ~band amount, preserves polarity and ~magnitude
        assert out[3] > 0.4  # still large
        assert out[3] < original  # but slightly reduced

    def test_apply_preserves_other_indices(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=10)
        for _ in range(10):
            cal.observe(_fv([(3, 0.01)]))
        f = _fv([(0, 30.0), (3, 0.0), (5, 0.7)])
        out = cal.apply(f)
        assert out[0] == 30.0
        assert out[5] == 0.7

    def test_skip_nan_samples_during_warmup(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=5)
        cal.observe(_fv([(3, np.nan)]))
        cal.observe(_fv([(3, np.nan)]))
        # Two NaN samples were skipped, so progress is still 0
        assert cal.progress == 0.0
        for _ in range(5):
            cal.observe(_fv([(3, 0.0)]))
        assert cal.is_ready

    def test_nan_input_at_inference_left_alone(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=5)
        for _ in range(5):
            cal.observe(_fv([(3, 0.0)]))
        f = _fv([(3, np.nan)])
        out = cal.apply(f)
        assert np.isnan(out[3])

    def test_input_not_mutated(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=5)
        for _ in range(5):
            cal.observe(_fv([(3, 0.01)]))
        f = _fv([(3, 0.5)])
        original = f.copy()
        cal.apply(f)
        np.testing.assert_array_equal(f, original)

    def test_min_std_clamp(self):
        cal = BaselineCalibrator(
            feature_indices=[3], warmup_frames=5, k=1.0, min_std=0.1
        )
        # Identical samples → raw std would be 0
        for _ in range(5):
            cal.observe(_fv([(3, 0.0)]))
        assert cal.is_ready
        # Effective std should be clamped to 0.1
        assert cal.baseline_std[0] == pytest.approx(0.1)

    def test_reset_clears(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=5)
        for _ in range(5):
            cal.observe(_fv([(3, 0.0)]))
        assert cal.is_ready
        cal.reset()
        assert not cal.is_ready
        assert cal.baseline_mean is None
        assert cal.progress == 0.0

    def test_observe_after_ready_is_noop(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=3)
        for v in [0.0, 0.0, 0.0]:
            cal.observe(_fv([(3, v)]))
        first_mean = cal.baseline_mean[0]
        # Try to "poison" with a huge sample after ready
        cal.observe(_fv([(3, 100.0)]))
        assert cal.baseline_mean[0] == pytest.approx(first_mean)

    def test_invalid_init_args(self):
        with pytest.raises(ValueError):
            BaselineCalibrator(feature_indices=[], warmup_frames=10)
        with pytest.raises(ValueError):
            BaselineCalibrator(feature_indices=[0], warmup_frames=0)
        with pytest.raises(ValueError):
            BaselineCalibrator(feature_indices=[0], warmup_frames=5, k=-1.0)

    def test_invalid_features_shape(self):
        cal = BaselineCalibrator(feature_indices=[0], warmup_frames=5)
        with pytest.raises(ValueError):
            cal.observe(np.zeros((2, 3)))

    def test_index_out_of_bounds(self):
        cal = BaselineCalibrator(feature_indices=[99], warmup_frames=3)
        with pytest.raises(ValueError):
            cal.observe(np.zeros(15, dtype=np.float32))

    def test_k_zero_is_strict_passthrough_outside_mean(self):
        cal = BaselineCalibrator(feature_indices=[3], warmup_frames=5, k=0.0)
        for _ in range(5):
            cal.observe(_fv([(3, 0.0)]))
        out = cal.apply(_fv([(3, 0.5)]))
        assert out[3] == pytest.approx(0.5)
