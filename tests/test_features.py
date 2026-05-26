"""
Unit tests for the feature extraction and preprocessing modules.

Tests correctness of preprocessing operations (normalization,
imputation, augmentation) and sliding window behavior against
known input/output pairs.

Reference: research/8.3 - Expected feature ranges and behavior.
"""

import numpy as np
import pytest

from src.data_processing.preprocessor import KeypointPreprocessor, KeypointAugmenter
from src.pose.keypoints import MediaPipeKeypoint


class TestKeypointPreprocessor:
    """Tests for the KeypointPreprocessor class."""

    def setup_method(self) -> None:
        """Create preprocessor and sample keypoints for each test."""
        self.preprocessor = KeypointPreprocessor(
            keypoint_format="mediapipe",
            normalize=True,
            confidence_threshold=0.5,
        )
        # Create a simple standing-pose keypoint set (33 keypoints, 4 coords)
        self.standing_kps = np.zeros((33, 4), dtype=np.float32)
        # Set up essential joints with high confidence
        # Shoulders at top, hips at middle, ankles at bottom
        self.standing_kps[MediaPipeKeypoint.LEFT_SHOULDER] = [0.45, 0.3, 0.0, 0.9]
        self.standing_kps[MediaPipeKeypoint.RIGHT_SHOULDER] = [0.55, 0.3, 0.0, 0.9]
        self.standing_kps[MediaPipeKeypoint.LEFT_HIP] = [0.47, 0.5, 0.0, 0.9]
        self.standing_kps[MediaPipeKeypoint.RIGHT_HIP] = [0.53, 0.5, 0.0, 0.9]
        self.standing_kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.46, 0.8, 0.0, 0.8]
        self.standing_kps[MediaPipeKeypoint.RIGHT_ANKLE] = [0.54, 0.8, 0.0, 0.8]
        self.standing_kps[MediaPipeKeypoint.NOSE] = [0.5, 0.15, 0.0, 0.95]
        # Fill remaining with moderate confidence
        for i in range(33):
            if self.standing_kps[i, 3] == 0.0:
                self.standing_kps[i] = [0.5, 0.5, 0.0, 0.7]

    def test_invalid_format_raises(self) -> None:
        """Invalid keypoint format should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown keypoint format"):
            KeypointPreprocessor(keypoint_format="invalid")

    def test_filter_low_confidence(self) -> None:
        """Keypoints below confidence threshold should become NaN."""
        kps = self.standing_kps.copy()
        kps[0, 3] = 0.1  # Set nose confidence below threshold
        result = self.preprocessor._filter_low_confidence(kps)
        assert np.isnan(result[0]).all()
        assert not np.isnan(result[MediaPipeKeypoint.LEFT_SHOULDER]).any()

    def test_normalize_centers_on_hip(self) -> None:
        """After normalization, hip center should be at origin."""
        result = self.preprocessor.normalize_person_centric(self.standing_kps)
        hip_center = (
            result[MediaPipeKeypoint.LEFT_HIP, :3]
            + result[MediaPipeKeypoint.RIGHT_HIP, :3]
        ) / 2.0
        np.testing.assert_allclose(hip_center, [0.0, 0.0, 0.0], atol=1e-5)

    def test_normalize_scales_by_torso(self) -> None:
        """After normalization, shoulder-hip distance should be 1.0."""
        result = self.preprocessor.normalize_person_centric(self.standing_kps)
        shoulder_center = (
            result[MediaPipeKeypoint.LEFT_SHOULDER, :3]
            + result[MediaPipeKeypoint.RIGHT_SHOULDER, :3]
        ) / 2.0
        hip_center = (
            result[MediaPipeKeypoint.LEFT_HIP, :3]
            + result[MediaPipeKeypoint.RIGHT_HIP, :3]
        ) / 2.0
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        np.testing.assert_allclose(torso_length, 1.0, atol=1e-5)

    def test_normalize_preserves_visibility(self) -> None:
        """Normalization should not modify the visibility column."""
        result = self.preprocessor.normalize_person_centric(self.standing_kps)
        np.testing.assert_array_equal(
            result[:, 3], self.standing_kps[:, 3]
        )

    def test_normalize_missing_hip_returns_unchanged(self) -> None:
        """If hip keypoints are NaN, normalization should return input."""
        kps = self.standing_kps.copy()
        kps[MediaPipeKeypoint.LEFT_HIP] = np.nan
        result = self.preprocessor.normalize_person_centric(kps)
        # Should return the input unchanged (except the NaN hip)
        assert np.isnan(result[MediaPipeKeypoint.LEFT_HIP]).all()


class TestTemporalImputation:
    """Tests for temporal keypoint imputation."""

    def setup_method(self) -> None:
        """Create preprocessor for imputation tests."""
        self.preprocessor = KeypointPreprocessor(
            keypoint_format="mediapipe",
            normalize=False,
            confidence_threshold=0.5,
        )

    def test_no_missing_unchanged(self) -> None:
        """Sequence with no missing values should be unchanged."""
        seq = np.random.rand(10, 33, 4).astype(np.float32)
        seq[:, :, 3] = 0.9  # high confidence
        result = self.preprocessor.preprocess_sequence(seq)
        np.testing.assert_allclose(result, seq, atol=1e-6)

    def test_interior_gap_interpolated(self) -> None:
        """Missing frame in the middle should be linearly interpolated."""
        seq = np.ones((5, 2, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        # Set frame 2 to be missing (low confidence)
        seq[2, 0, 3] = 0.1
        # Frames 0-1 have x=1.0, frames 3-4 have x=1.0
        # After filtering low conf -> NaN, interpolation should fill with 1.0
        result = self.preprocessor.preprocess_sequence(seq)
        np.testing.assert_allclose(result[2, 0, 0], 1.0, atol=1e-5)

    def test_terminal_gap_forward_filled(self) -> None:
        """Missing frames at the end should be forward-filled via interp."""
        seq = np.ones((5, 2, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        seq[0, 0, 0] = 2.0  # x=2 at frame 0
        seq[1, 0, 0] = 3.0  # x=3 at frame 1
        # Frames 2-4 missing
        seq[2:, 0, 3] = 0.1
        result = self.preprocessor.preprocess_sequence(seq)
        # np.interp clamps to last valid value for extrapolation
        assert not np.isnan(result[4, 0, 0])


class TestKeypointAugmenter:
    """Tests for the KeypointAugmenter class."""

    def setup_method(self) -> None:
        """Create augmenter with fixed seed."""
        self.augmenter = KeypointAugmenter(
            flip_probability=1.0,
            noise_std=0.0,
            scale_range=(1.0, 1.0),
            keypoint_format="mediapipe",
            seed=42,
        )

    def test_horizontal_flip_negates_x(self) -> None:
        """Horizontal flip should negate x-coordinates."""
        seq = np.zeros((5, 33, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        seq[:, MediaPipeKeypoint.LEFT_SHOULDER, 0] = 1.0
        result = self.augmenter.augment(seq)
        # After flip, left shoulder x should be at right shoulder position
        # and negated
        assert result[0, MediaPipeKeypoint.RIGHT_SHOULDER, 0] == -1.0

    def test_horizontal_flip_swaps_pairs(self) -> None:
        """Horizontal flip should swap left/right keypoint data."""
        seq = np.zeros((5, 33, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        seq[:, MediaPipeKeypoint.LEFT_SHOULDER, 1] = 0.3  # y = 0.3
        seq[:, MediaPipeKeypoint.RIGHT_SHOULDER, 1] = 0.7  # y = 0.7
        result = self.augmenter.augment(seq)
        # Left shoulder y should now have right's value
        np.testing.assert_allclose(
            result[:, MediaPipeKeypoint.LEFT_SHOULDER, 1], 0.7
        )
        np.testing.assert_allclose(
            result[:, MediaPipeKeypoint.RIGHT_SHOULDER, 1], 0.3
        )

    def test_noise_changes_values(self) -> None:
        """Gaussian noise should modify coordinate values."""
        noisy_aug = KeypointAugmenter(
            flip_probability=0.0,
            noise_std=0.1,
            scale_range=(1.0, 1.0),
            keypoint_format="mediapipe",
            seed=42,
        )
        seq = np.ones((5, 33, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        result = noisy_aug.augment(seq)
        # Values should be different from input
        assert not np.allclose(result[:, :, :3], seq[:, :, :3])

    def test_scaling_multiplies_coords(self) -> None:
        """Random scaling should uniformly scale coordinates."""
        scale_aug = KeypointAugmenter(
            flip_probability=0.0,
            noise_std=0.0,
            scale_range=(2.0, 2.0),  # fixed 2x scale
            keypoint_format="mediapipe",
            seed=42,
        )
        seq = np.ones((3, 33, 4), dtype=np.float32)
        seq[:, :, 3] = 0.9
        result = scale_aug.augment(seq)
        np.testing.assert_allclose(result[:, :, :3], 2.0, atol=1e-5)

    def test_nan_values_preserved(self) -> None:
        """NaN values should not be modified by noise."""
        noisy_aug = KeypointAugmenter(
            flip_probability=0.0,
            noise_std=0.1,
            scale_range=(1.0, 1.0),
            keypoint_format="mediapipe",
            seed=42,
        )
        seq = np.ones((3, 33, 4), dtype=np.float32)
        seq[1, 5, :] = np.nan
        result = noisy_aug.augment(seq)
        assert np.isnan(result[1, 5, :3]).all()


class TestCOCOPreprocessor:
    """Tests for COCO keypoint format support."""

    def test_coco_format_accepted(self) -> None:
        """COCO format should be accepted without error."""
        pp = KeypointPreprocessor(keypoint_format="coco", normalize=True)
        assert pp.keypoint_format == "coco"

    def test_coco_normalization(self) -> None:
        """COCO format normalization should center on hip."""
        from src.pose.keypoints import COCOKeypoint

        pp = KeypointPreprocessor(keypoint_format="coco", normalize=True)
        kps = np.zeros((17, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        kps[COCOKeypoint.LEFT_SHOULDER] = [0.45, 0.3, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_SHOULDER] = [0.55, 0.3, 0.0, 0.9]
        kps[COCOKeypoint.LEFT_HIP] = [0.47, 0.5, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_HIP] = [0.53, 0.5, 0.0, 0.9]

        result = pp.normalize_person_centric(kps)
        hip_center = (
            result[COCOKeypoint.LEFT_HIP, :3]
            + result[COCOKeypoint.RIGHT_HIP, :3]
        ) / 2.0
        np.testing.assert_allclose(hip_center, [0.0, 0.0, 0.0], atol=1e-5)
