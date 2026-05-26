"""
Unit tests for the FeatureExtractor class.

Tests all 15 feature computations with known-geometry keypoint
configurations (standing, lying, mid-fall).
"""

import numpy as np
import pytest

from src.features.extractor import FeatureExtractor
from src.pose.keypoints import MediaPipeKeypoint, COCOKeypoint


def _make_standing_kps() -> np.ndarray:
    """Create MediaPipe keypoints for a standing person.

    Coordinate system: normalized [0,1], y increases downward.
    Standing person: nose at top, ankles at bottom.
    """
    kps = np.zeros((33, 4), dtype=np.float32)
    kps[:, 3] = 0.9  # all visible

    kps[MediaPipeKeypoint.NOSE] = [0.5, 0.1, 0.0, 0.95]
    kps[MediaPipeKeypoint.LEFT_SHOULDER] = [0.45, 0.25, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_SHOULDER] = [0.55, 0.25, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_ELBOW] = [0.40, 0.40, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_ELBOW] = [0.60, 0.40, 0.0, 0.85]
    kps[MediaPipeKeypoint.LEFT_WRIST] = [0.42, 0.50, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_WRIST] = [0.58, 0.50, 0.0, 0.8]
    kps[MediaPipeKeypoint.LEFT_HIP] = [0.47, 0.50, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_HIP] = [0.53, 0.50, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_KNEE] = [0.46, 0.70, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_KNEE] = [0.54, 0.70, 0.0, 0.85]
    kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.45, 0.90, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_ANKLE] = [0.55, 0.90, 0.0, 0.8]
    return kps


def _make_lying_kps() -> np.ndarray:
    """Create MediaPipe keypoints for a person lying on the ground.

    Person is horizontal: nose and ankles at similar y, spread across x.
    """
    kps = np.zeros((33, 4), dtype=np.float32)
    kps[:, 3] = 0.9

    kps[MediaPipeKeypoint.NOSE] = [0.1, 0.8, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_SHOULDER] = [0.2, 0.78, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_SHOULDER] = [0.2, 0.82, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_ELBOW] = [0.30, 0.75, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_ELBOW] = [0.30, 0.85, 0.0, 0.85]
    kps[MediaPipeKeypoint.LEFT_WRIST] = [0.35, 0.73, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_WRIST] = [0.35, 0.87, 0.0, 0.8]
    kps[MediaPipeKeypoint.LEFT_HIP] = [0.50, 0.79, 0.0, 0.9]
    kps[MediaPipeKeypoint.RIGHT_HIP] = [0.50, 0.81, 0.0, 0.9]
    kps[MediaPipeKeypoint.LEFT_KNEE] = [0.70, 0.78, 0.0, 0.85]
    kps[MediaPipeKeypoint.RIGHT_KNEE] = [0.70, 0.82, 0.0, 0.85]
    kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.90, 0.79, 0.0, 0.8]
    kps[MediaPipeKeypoint.RIGHT_ANKLE] = [0.90, 0.81, 0.0, 0.8]
    return kps


class TestFeatureExtractorInit:
    """Test initialization and format validation."""

    def test_mediapipe_accepted(self) -> None:
        fe = FeatureExtractor(keypoint_format="mediapipe")
        assert fe.keypoint_format == "mediapipe"

    def test_coco_accepted(self) -> None:
        fe = FeatureExtractor(keypoint_format="coco")
        assert fe.keypoint_format == "coco"

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown keypoint format"):
            FeatureExtractor(keypoint_format="openpose")

    def test_num_features_constant(self) -> None:
        assert FeatureExtractor.NUM_FEATURES == 15


class TestTorsoInclination:
    """Test torso inclination angle computation."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_near_zero(self) -> None:
        """Standing person's torso should be nearly vertical (small angle)."""
        kps = _make_standing_kps()
        angle = self.fe.compute_torso_inclination(kps)
        assert 0.0 <= angle < 30.0  # nearly upright

    def test_lying_near_90(self) -> None:
        """Lying person's torso should be nearly horizontal (~90 deg)."""
        kps = _make_lying_kps()
        angle = self.fe.compute_torso_inclination(kps)
        assert 60.0 < angle <= 120.0

    def test_missing_shoulder_returns_nan(self) -> None:
        """Missing shoulder should return NaN."""
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.LEFT_SHOULDER, 3] = 0.1
        kps[MediaPipeKeypoint.RIGHT_SHOULDER, 3] = 0.1
        angle = self.fe.compute_torso_inclination(kps)
        assert np.isnan(angle)


class TestHipShoulderAngle:
    """Test hip-shoulder roll angle."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_near_90(self) -> None:
        """Standing torso should be ~90 from horizontal."""
        kps = _make_standing_kps()
        angle = self.fe.compute_hip_shoulder_angle(kps)
        assert 60.0 < angle <= 90.0

    def test_lying_near_0(self) -> None:
        """Lying torso should be near 0 from horizontal."""
        kps = _make_lying_kps()
        angle = self.fe.compute_hip_shoulder_angle(kps)
        assert 0.0 <= angle < 30.0


class TestBboxAspectRatio:
    """Test bounding box aspect ratio from keypoints."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_ar_less_than_1(self) -> None:
        """Standing person should have AR < 1 (taller than wide)."""
        kps = _make_standing_kps()
        ar = self.fe.compute_bbox_aspect_ratio(kps)
        assert ar < 1.0

    def test_lying_ar_greater_than_1(self) -> None:
        """Lying person should have AR > 1 (wider than tall)."""
        kps = _make_lying_kps()
        ar = self.fe.compute_bbox_aspect_ratio(kps)
        assert ar > 1.0

    def test_insufficient_keypoints_nan(self) -> None:
        """With fewer than 2 valid keypoints, should return NaN."""
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.1  # all below threshold
        ar = self.fe.compute_bbox_aspect_ratio(kps)
        assert np.isnan(ar)


class TestComVelocity:
    """Test center-of-mass vertical velocity."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_stationary_zero_velocity(self) -> None:
        """Same keypoints across frames should yield ~0 velocity."""
        kps = _make_standing_kps()
        vel = self.fe.compute_com_velocity(kps, kps, dt=1.0 / 15)
        assert abs(vel) < 1e-5

    def test_downward_motion_positive(self) -> None:
        """Hip moving downward (y increases) should give positive velocity."""
        kps1 = _make_standing_kps()
        kps2 = kps1.copy()
        # Move hips down by 0.1
        kps2[MediaPipeKeypoint.LEFT_HIP, 1] += 0.1
        kps2[MediaPipeKeypoint.RIGHT_HIP, 1] += 0.1
        vel = self.fe.compute_com_velocity(kps2, kps1, dt=1.0 / 15)
        assert vel > 0

    def test_zero_dt_returns_nan(self) -> None:
        """Zero time delta should return NaN."""
        kps = _make_standing_kps()
        vel = self.fe.compute_com_velocity(kps, kps, dt=0.0)
        assert np.isnan(vel)

    def test_missing_hip_returns_nan(self) -> None:
        """Missing hip keypoints should return NaN."""
        kps1 = _make_standing_kps()
        kps2 = kps1.copy()
        kps2[MediaPipeKeypoint.LEFT_HIP, 3] = 0.1
        kps2[MediaPipeKeypoint.RIGHT_HIP, 3] = 0.1
        vel = self.fe.compute_com_velocity(kps2, kps1, dt=1.0 / 15)
        assert np.isnan(vel)


class TestComAcceleration:
    """Test center-of-mass vertical acceleration."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_constant_velocity_zero_accel(self) -> None:
        accel = self.fe.compute_com_acceleration(5.0, 5.0, dt=1.0 / 15)
        assert abs(accel) < 1e-5

    def test_increasing_velocity_positive_accel(self) -> None:
        accel = self.fe.compute_com_acceleration(10.0, 5.0, dt=1.0 / 15)
        assert accel > 0

    def test_nan_velocity_returns_nan(self) -> None:
        accel = self.fe.compute_com_acceleration(float("nan"), 5.0, dt=0.1)
        assert np.isnan(accel)


class TestHeadToToeDistance:
    """Test head-to-toe vertical distance."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_large_distance(self) -> None:
        """Standing person should have large head-to-toe distance."""
        kps = _make_standing_kps()
        dist = self.fe.compute_head_to_toe_distance(kps)
        assert dist > 0.5  # nose at 0.1, ankles at 0.9

    def test_lying_small_distance(self) -> None:
        """Lying person should have small head-to-toe distance."""
        kps = _make_lying_kps()
        dist = self.fe.compute_head_to_toe_distance(kps)
        assert dist < 0.1  # nose and ankles at similar y

    def test_missing_nose_nan(self) -> None:
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.NOSE, 3] = 0.1
        assert np.isnan(self.fe.compute_head_to_toe_distance(kps))


class TestShoulderAnkleDistance:
    """Test shoulder-to-ankle Euclidean distance."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_positive_distance(self) -> None:
        kps = _make_standing_kps()
        dist = self.fe.compute_shoulder_ankle_distance(kps)
        assert dist > 0.3

    def test_missing_ankle_nan(self) -> None:
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.LEFT_ANKLE, 3] = 0.1
        kps[MediaPipeKeypoint.RIGHT_ANKLE, 3] = 0.1
        assert np.isnan(self.fe.compute_shoulder_ankle_distance(kps))


class TestJointAngle:
    """Test joint angle computation."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_straight_leg_180(self) -> None:
        """A perfectly straight leg should give ~180 degrees."""
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        # Collinear points: hip(0.5, 0.5), knee(0.5, 0.7), ankle(0.5, 0.9)
        kps[MediaPipeKeypoint.LEFT_HIP] = [0.5, 0.5, 0.0, 0.9]
        kps[MediaPipeKeypoint.LEFT_KNEE] = [0.5, 0.7, 0.0, 0.9]
        kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.5, 0.9, 0.0, 0.9]
        angle = self.fe.compute_joint_angle(
            kps,
            int(MediaPipeKeypoint.LEFT_HIP),
            int(MediaPipeKeypoint.LEFT_KNEE),
            int(MediaPipeKeypoint.LEFT_ANKLE),
        )
        assert abs(angle - 180.0) < 1.0

    def test_right_angle_90(self) -> None:
        """A 90-degree bend should give ~90 degrees."""
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        kps[MediaPipeKeypoint.LEFT_HIP] = [0.5, 0.5, 0.0, 0.9]
        kps[MediaPipeKeypoint.LEFT_KNEE] = [0.5, 0.7, 0.0, 0.9]
        kps[MediaPipeKeypoint.LEFT_ANKLE] = [0.7, 0.7, 0.0, 0.9]
        angle = self.fe.compute_joint_angle(
            kps,
            int(MediaPipeKeypoint.LEFT_HIP),
            int(MediaPipeKeypoint.LEFT_KNEE),
            int(MediaPipeKeypoint.LEFT_ANKLE),
        )
        assert abs(angle - 90.0) < 1.0

    def test_invalid_keypoint_nan(self) -> None:
        kps = np.zeros((33, 4), dtype=np.float32)
        kps[:, 3] = 0.1  # all invisible
        angle = self.fe.compute_joint_angle(kps, 23, 25, 27)
        assert np.isnan(angle)


class TestWristHipDistance:
    """Test wrist-to-hip distance (arm reflex indicator)."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_positive_distance(self) -> None:
        kps = _make_standing_kps()
        dist = self.fe.compute_wrist_hip_distance(kps)
        assert dist > 0

    def test_missing_both_wrists_nan(self) -> None:
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.LEFT_WRIST, 3] = 0.1
        kps[MediaPipeKeypoint.RIGHT_WRIST, 3] = 0.1
        assert np.isnan(self.fe.compute_wrist_hip_distance(kps))

    def test_one_wrist_still_works(self) -> None:
        """With only one valid wrist, should still compute distance."""
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.LEFT_WRIST, 3] = 0.1  # invalid
        dist = self.fe.compute_wrist_hip_distance(kps)
        assert not np.isnan(dist)
        assert dist > 0


class TestBodySpread:
    """Test body horizontal spread."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_standing_small_spread(self) -> None:
        kps = _make_standing_kps()
        spread = self.fe.compute_body_spread(kps)
        # Standing: x ranges from ~0.4 to ~0.6 (plus default 0.5 filler)
        assert 0 < spread < 1.0

    def test_lying_large_spread(self) -> None:
        kps = _make_lying_kps()
        spread = self.fe.compute_body_spread(kps)
        assert spread > 0.5


class TestVisibilityConfidence:
    """Test visibility confidence features."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_high_confidence(self) -> None:
        kps = _make_standing_kps()
        mean_vis, min_core = self.fe.compute_visibility_confidence(kps)
        assert mean_vis > 0.7
        assert min_core > 0.7

    def test_mixed_confidence(self) -> None:
        kps = _make_standing_kps()
        kps[MediaPipeKeypoint.LEFT_ANKLE, 3] = 0.3
        _, min_core = self.fe.compute_visibility_confidence(kps)
        assert min_core < 0.31  # float32 precision


class TestExtractFullVector:
    """Test the full extract() method."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_output_shape(self) -> None:
        kps = _make_standing_kps()
        features = self.fe.extract(kps)
        assert features.shape == (15,)

    def test_dtype_float32(self) -> None:
        kps = _make_standing_kps()
        features = self.fe.extract(kps)
        assert features.dtype == np.float32

    def test_no_prev_velocity_accel_nan(self) -> None:
        """Without previous keypoints, velocity and acceleration should be NaN."""
        kps = _make_standing_kps()
        features = self.fe.extract(kps)
        assert np.isnan(features[3])  # velocity
        assert np.isnan(features[4])  # acceleration

    def test_with_prev_keypoints_velocity_computed(self) -> None:
        """With previous keypoints, velocity should be computed."""
        kps1 = _make_standing_kps()
        kps2 = kps1.copy()
        kps2[MediaPipeKeypoint.LEFT_HIP, 1] += 0.05
        kps2[MediaPipeKeypoint.RIGHT_HIP, 1] += 0.05
        features = self.fe.extract(kps2, prev_keypoints=kps1)
        assert not np.isnan(features[3])

    def test_geometric_features_present(self) -> None:
        """All geometric features should be non-NaN for valid standing pose."""
        kps = _make_standing_kps()
        features = self.fe.extract(kps)
        # indices 0,1,2,5,6,7,8,9,10,11,12,13,14 should be valid
        for i in [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
            assert not np.isnan(features[i]), f"Feature {i} is NaN"


class TestExtractSequence:
    """Test the extract_sequence method."""

    def setup_method(self) -> None:
        self.fe = FeatureExtractor(keypoint_format="mediapipe")

    def test_output_shape(self) -> None:
        """Sequence output should be (T, 15)."""
        seq = np.stack([_make_standing_kps() for _ in range(10)])
        result = self.fe.extract_sequence(seq, dt=1.0 / 15)
        assert result.shape == (10, 15)

    def test_first_frame_velocity_nan(self) -> None:
        """First frame should have NaN velocity (no previous frame)."""
        seq = np.stack([_make_standing_kps() for _ in range(5)])
        result = self.fe.extract_sequence(seq)
        assert np.isnan(result[0, 3])

    def test_subsequent_velocity_computed(self) -> None:
        """Frames after the first should have computed velocity."""
        seq = np.stack([_make_standing_kps() for _ in range(5)])
        result = self.fe.extract_sequence(seq)
        assert not np.isnan(result[1, 3])

    def test_falling_sequence_features_change(self) -> None:
        """Features should change across a fall sequence."""
        standing = _make_standing_kps()
        lying = _make_lying_kps()
        # Interpolate a 10-frame fall
        seq = np.array([
            standing * (1 - t / 9) + lying * (t / 9)
            for t in range(10)
        ])
        result = self.fe.extract_sequence(seq)
        # Torso inclination should increase from start to end
        assert result[-1, 0] > result[0, 0]


class TestCOCOExtractor:
    """Test feature extraction with COCO keypoint format."""

    def test_coco_torso_inclination(self) -> None:
        fe = FeatureExtractor(keypoint_format="coco")
        kps = np.zeros((17, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        kps[COCOKeypoint.LEFT_SHOULDER] = [0.45, 0.25, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_SHOULDER] = [0.55, 0.25, 0.0, 0.9]
        kps[COCOKeypoint.LEFT_HIP] = [0.47, 0.50, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_HIP] = [0.53, 0.50, 0.0, 0.9]
        angle = fe.compute_torso_inclination(kps)
        assert 0.0 <= angle < 30.0

    def test_coco_extract_shape(self) -> None:
        fe = FeatureExtractor(keypoint_format="coco")
        kps = np.zeros((17, 4), dtype=np.float32)
        kps[:, 3] = 0.9
        kps[COCOKeypoint.NOSE] = [0.5, 0.1, 0.0, 0.9]
        kps[COCOKeypoint.LEFT_SHOULDER] = [0.45, 0.25, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_SHOULDER] = [0.55, 0.25, 0.0, 0.9]
        kps[COCOKeypoint.LEFT_HIP] = [0.47, 0.50, 0.0, 0.9]
        kps[COCOKeypoint.RIGHT_HIP] = [0.53, 0.50, 0.0, 0.9]
        kps[COCOKeypoint.LEFT_KNEE] = [0.46, 0.70, 0.0, 0.85]
        kps[COCOKeypoint.RIGHT_KNEE] = [0.54, 0.70, 0.0, 0.85]
        kps[COCOKeypoint.LEFT_ANKLE] = [0.45, 0.90, 0.0, 0.8]
        kps[COCOKeypoint.RIGHT_ANKLE] = [0.55, 0.90, 0.0, 0.8]
        kps[COCOKeypoint.LEFT_WRIST] = [0.42, 0.50, 0.0, 0.8]
        kps[COCOKeypoint.RIGHT_WRIST] = [0.58, 0.50, 0.0, 0.8]
        features = fe.extract(kps)
        assert features.shape == (15,)
