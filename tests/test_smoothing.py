"""Tests for OneEuro keypoint smoothing."""

import numpy as np
import pytest

from src.pose.smoothing import KeypointSmoother, OneEuroFilter


class TestOneEuroFilter:
    def test_first_sample_passes_through(self):
        f = OneEuroFilter()
        assert f(0.5, dt=1 / 30) == pytest.approx(0.5)

    def test_constant_signal_converges(self):
        f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        out = [f(1.0, dt=1 / 30) for _ in range(60)]
        assert out[-1] == pytest.approx(1.0, abs=1e-6)

    def test_smoothing_reduces_white_noise_variance(self):
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 1.0, size=500)
        f = OneEuroFilter(min_cutoff=0.5, beta=0.0)
        filtered = np.array([f(float(x), dt=1 / 30) for x in noise])
        # After warm-up, filtered variance must be much smaller than raw.
        assert filtered[50:].var() < noise[50:].var() * 0.25

    def test_high_velocity_passes_through(self):
        # Rapidly changing signal should not be over-smoothed (high beta lets it through)
        f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
        signal = np.linspace(0.0, 10.0, 100)
        filtered = np.array([f(float(x), dt=1 / 30) for x in signal])
        # Final filtered value tracks final input value closely
        assert abs(filtered[-1] - signal[-1]) < 0.5

    def test_reset_reinitializes(self):
        f = OneEuroFilter()
        f(1.0, dt=1 / 30)
        f(2.0, dt=1 / 30)
        f.reset()
        # After reset, next sample is returned unchanged
        assert f(5.0, dt=1 / 30) == pytest.approx(5.0)

    def test_zero_dt_does_not_explode(self):
        f = OneEuroFilter()
        f(1.0, dt=1 / 30)
        # Bad dt should be clamped, not divide-by-zero
        result = f(2.0, dt=0.0)
        assert np.isfinite(result)


class TestKeypointSmoother:
    def test_shape_preserved_mediapipe(self):
        smoother = KeypointSmoother(num_keypoints=33, fps=15.0)
        kps = np.random.RandomState(0).rand(33, 4).astype(np.float32)
        out = smoother.smooth(kps)
        assert out.shape == kps.shape
        assert out.dtype == kps.dtype

    def test_shape_preserved_coco(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        kps = np.random.RandomState(0).rand(17, 3).astype(np.float32)
        out = smoother.smooth(kps)
        assert out.shape == kps.shape

    def test_z_and_visibility_untouched(self):
        smoother = KeypointSmoother(num_keypoints=33, fps=15.0)
        kps = np.random.RandomState(1).rand(33, 4).astype(np.float32)
        out = smoother.smooth(kps)
        # Columns 2 (z) and 3 (visibility) must be identical
        np.testing.assert_array_equal(out[:, 2], kps[:, 2])
        np.testing.assert_array_equal(out[:, 3], kps[:, 3])

    def test_input_not_mutated(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        kps = np.full((17, 3), 0.5, dtype=np.float32)
        original = kps.copy()
        smoother.smooth(kps)
        np.testing.assert_array_equal(kps, original)

    def test_first_call_is_passthrough(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        kps = np.random.RandomState(2).rand(17, 4).astype(np.float32)
        out = smoother.smooth(kps)
        np.testing.assert_allclose(out[:, :2], kps[:, :2], atol=1e-6)

    def test_smoothing_reduces_jitter(self):
        # Generate a steady signal with small noise on every keypoint
        rng = np.random.default_rng(0)
        smoother = KeypointSmoother(num_keypoints=17, fps=30.0, min_cutoff=0.5)
        base = np.tile(np.array([[0.5, 0.5, 0.0, 1.0]], dtype=np.float32), (17, 1))
        raw_xs, smoothed_xs = [], []
        for _ in range(200):
            noisy = base.copy()
            noisy[:, 0] += rng.normal(0, 0.05, size=17).astype(np.float32)
            smoothed = smoother.smooth(noisy)
            raw_xs.append(noisy[0, 0])
            smoothed_xs.append(smoothed[0, 0])
        # Drop warm-up
        assert np.var(smoothed_xs[50:]) < np.var(raw_xs[50:]) * 0.5

    def test_reset_clears_state(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        kps = np.random.RandomState(3).rand(17, 4).astype(np.float32)
        smoother.smooth(kps)
        smoother.smooth(kps + 0.2)
        smoother.reset()
        # After reset, next call is pass-through again
        out = smoother.smooth(kps)
        np.testing.assert_allclose(out[:, :2], kps[:, :2], atol=1e-6)

    def test_invalid_keypoint_count_raises(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        with pytest.raises(ValueError):
            smoother.smooth(np.zeros((33, 4), dtype=np.float32))

    def test_invalid_shape_raises(self):
        smoother = KeypointSmoother(num_keypoints=17, fps=15.0)
        with pytest.raises(ValueError):
            smoother.smooth(np.zeros((17,), dtype=np.float32))

    def test_invalid_init_args(self):
        with pytest.raises(ValueError):
            KeypointSmoother(num_keypoints=0, fps=15.0)
        with pytest.raises(ValueError):
            KeypointSmoother(num_keypoints=17, fps=0.0)
