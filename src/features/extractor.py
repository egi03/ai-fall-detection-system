"""
Feature computation from pose keypoints for fall detection.

Computes all 15 features defined in the Research Synthesis:
geometric, kinematic, angular, reflexive, and confidence features.

Feature list (15 total):
  Geometric (3):    torso_inclination, hip_shoulder_angle, bbox_aspect_ratio
  Kinematic (2):    com_vertical_velocity, com_vertical_acceleration
  Distance (2):     head_to_toe_distance, shoulder_ankle_distance
  Angular (4):      left_knee_angle, right_knee_angle, left_hip_angle, right_hip_angle
  Reflexive (2):    wrist_hip_distance, body_horizontal_spread
  Confidence (2):   mean_visibility, min_core_visibility

Reference: research/3.2 - Core geometric, angular, and velocity features.
Reference: research/8.3 - Complete mathematical definitions and implementations.
"""

from typing import Optional, Tuple

import numpy as np

from src.pose.keypoints import MediaPipeKeypoint, COCOKeypoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keypoint index mappings per format
_INDEX_MAP = {
    "mediapipe": {
        "nose": int(MediaPipeKeypoint.NOSE),
        "left_shoulder": int(MediaPipeKeypoint.LEFT_SHOULDER),
        "right_shoulder": int(MediaPipeKeypoint.RIGHT_SHOULDER),
        "left_hip": int(MediaPipeKeypoint.LEFT_HIP),
        "right_hip": int(MediaPipeKeypoint.RIGHT_HIP),
        "left_knee": int(MediaPipeKeypoint.LEFT_KNEE),
        "right_knee": int(MediaPipeKeypoint.RIGHT_KNEE),
        "left_ankle": int(MediaPipeKeypoint.LEFT_ANKLE),
        "right_ankle": int(MediaPipeKeypoint.RIGHT_ANKLE),
        "left_wrist": int(MediaPipeKeypoint.LEFT_WRIST),
        "right_wrist": int(MediaPipeKeypoint.RIGHT_WRIST),
        "left_elbow": int(MediaPipeKeypoint.LEFT_ELBOW),
        "right_elbow": int(MediaPipeKeypoint.RIGHT_ELBOW),
    },
    "coco": {
        "nose": int(COCOKeypoint.NOSE),
        "left_shoulder": int(COCOKeypoint.LEFT_SHOULDER),
        "right_shoulder": int(COCOKeypoint.RIGHT_SHOULDER),
        "left_hip": int(COCOKeypoint.LEFT_HIP),
        "right_hip": int(COCOKeypoint.RIGHT_HIP),
        "left_knee": int(COCOKeypoint.LEFT_KNEE),
        "right_knee": int(COCOKeypoint.RIGHT_KNEE),
        "left_ankle": int(COCOKeypoint.LEFT_ANKLE),
        "right_ankle": int(COCOKeypoint.RIGHT_ANKLE),
        "left_wrist": int(COCOKeypoint.LEFT_WRIST),
        "right_wrist": int(COCOKeypoint.RIGHT_WRIST),
        "left_elbow": int(COCOKeypoint.LEFT_ELBOW),
        "right_elbow": int(COCOKeypoint.RIGHT_ELBOW),
    },
}

# Core keypoints for visibility computation (torso + limb joints)
_CORE_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


class FeatureExtractor:
    """
    Computes fall-detection features from a single frame's keypoints.

    All features are scale-invariant when person-centric normalization
    is applied upstream. Features return NaN when required keypoints
    have insufficient confidence.

    Parameters
    ----------
    keypoint_format : str
        Either 'mediapipe' (33 keypoints) or 'coco' (17 keypoints).
    confidence_threshold : float
        Minimum visibility score to consider a keypoint valid.
    """

    # DECISION: 15 features per research/3.2 and research/8.3 ablation guidelines.
    NUM_FEATURES = 15

    def __init__(
        self,
        keypoint_format: str = "mediapipe",
        confidence_threshold: float = 0.5,
    ) -> None:
        if keypoint_format not in _INDEX_MAP:
            raise ValueError(
                f"Unknown keypoint format: {keypoint_format}. "
                "Must be 'mediapipe' or 'coco'."
            )
        self._format = keypoint_format
        self._conf_threshold = confidence_threshold
        self._idx = _INDEX_MAP[keypoint_format]

    @property
    def keypoint_format(self) -> str:
        """Return the active keypoint format."""
        return self._format

    def _is_valid(self, keypoints: np.ndarray, idx: int) -> bool:
        """Check if a keypoint index is valid and above confidence threshold."""
        if idx >= len(keypoints):
            return False
        if keypoints.shape[1] >= 4:
            vis = keypoints[idx, 3]
        elif keypoints.shape[1] == 3:
            vis = keypoints[idx, 2]
        else:
            return True
        return not np.isnan(vis) and vis >= self._conf_threshold

    def _get_xy(self, keypoints: np.ndarray, idx: int) -> np.ndarray:
        """Get (x, y) coordinates for a keypoint."""
        return keypoints[idx, :2]

    def _get_xyz(self, keypoints: np.ndarray, idx: int) -> np.ndarray:
        """Get (x, y, z) coordinates for a keypoint."""
        return keypoints[idx, :3]

    def _midpoint_valid(
        self, keypoints: np.ndarray, name_a: str, name_b: str,
    ) -> Tuple[bool, np.ndarray]:
        """Compute midpoint of two named joints, check both valid."""
        idx_a = self._idx[name_a]
        idx_b = self._idx[name_b]
        if not (self._is_valid(keypoints, idx_a) and
                self._is_valid(keypoints, idx_b)):
            return False, np.array([np.nan, np.nan])
        mid = (self._get_xy(keypoints, idx_a) +
               self._get_xy(keypoints, idx_b)) / 2.0
        return True, mid

    def extract(
        self,
        keypoints: np.ndarray,
        prev_keypoints: Optional[np.ndarray] = None,
        dt: float = 1.0 / 15.0,
        prev_velocity: float = float("nan"),
    ) -> np.ndarray:
        """
        Extract the full feature vector from a single frame's keypoints.

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4) with columns (x, y, z, visibility).
        prev_keypoints : np.ndarray, optional
            Previous frame's keypoints for velocity/acceleration computation.
        dt : float
            Time delta between frames in seconds.
        prev_velocity : float
            Previous frame's CoM vertical velocity (for acceleration).

        Returns
        -------
        np.ndarray
            Feature vector of shape (NUM_FEATURES,).
        """
        features = np.full(self.NUM_FEATURES, np.nan, dtype=np.float32)

        # Feature 0: Torso inclination angle
        features[0] = self.compute_torso_inclination(keypoints)

        # Feature 1: Hip-shoulder angle (roll)
        features[1] = self.compute_hip_shoulder_angle(keypoints)

        # Feature 2: Bbox aspect ratio from keypoints
        features[2] = self.compute_bbox_aspect_ratio(keypoints)

        # Feature 3: CoM vertical velocity
        if prev_keypoints is not None:
            features[3] = self.compute_com_velocity(
                keypoints, prev_keypoints, dt
            )

        # Feature 4: CoM vertical acceleration
        if not np.isnan(features[3]) and not np.isnan(prev_velocity):
            features[4] = self.compute_com_acceleration(
                features[3], prev_velocity, dt
            )

        # Feature 5: Head-to-toe distance
        features[5] = self.compute_head_to_toe_distance(keypoints)

        # Feature 6: Shoulder-ankle distance
        features[6] = self.compute_shoulder_ankle_distance(keypoints)

        # Feature 7-8: Knee angles (left, right)
        features[7] = self.compute_joint_angle(
            keypoints,
            self._idx["left_hip"],
            self._idx["left_knee"],
            self._idx["left_ankle"],
        )
        features[8] = self.compute_joint_angle(
            keypoints,
            self._idx["right_hip"],
            self._idx["right_knee"],
            self._idx["right_ankle"],
        )

        # Feature 9-10: Hip angles (left, right)
        features[9] = self.compute_joint_angle(
            keypoints,
            self._idx["left_shoulder"],
            self._idx["left_hip"],
            self._idx["left_knee"],
        )
        features[10] = self.compute_joint_angle(
            keypoints,
            self._idx["right_shoulder"],
            self._idx["right_hip"],
            self._idx["right_knee"],
        )

        # Feature 11: Wrist-hip distance (arm reflex indicator)
        features[11] = self.compute_wrist_hip_distance(keypoints)

        # Feature 12: Body horizontal spread
        features[12] = self.compute_body_spread(keypoints)

        # Feature 13-14: Visibility confidence
        mean_vis, min_core_vis = self.compute_visibility_confidence(keypoints)
        features[13] = mean_vis
        features[14] = min_core_vis

        return features

    def compute_torso_inclination(self, keypoints: np.ndarray) -> float:
        """
        Compute torso inclination angle from vertical axis.

        Reference: research/3.2 - Angles > 60 degrees indicate fall.
        Uses: Shoulders (MP 11,12 / COCO 5,6), Hips (MP 23,24 / COCO 11,12).

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            Angle in degrees [0, 180]. NaN if keypoints insufficient.
        """
        sh_ok, shoulder_mid = self._midpoint_valid(
            keypoints, "left_shoulder", "right_shoulder"
        )
        hip_ok, hip_mid = self._midpoint_valid(
            keypoints, "left_hip", "right_hip"
        )
        if not (sh_ok and hip_ok):
            return float("nan")

        # Torso vector: hip -> shoulder (upward in image = negative y)
        torso_vec = shoulder_mid - hip_mid
        # Vertical axis (pointing up in normalized coords, but in image
        # coords y increases downward, so vertical "up" is (0, -1))
        vertical = np.array([0.0, -1.0])

        cos_angle = np.dot(torso_vec, vertical) / (
            np.linalg.norm(torso_vec) * np.linalg.norm(vertical) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def compute_hip_shoulder_angle(self, keypoints: np.ndarray) -> float:
        """
        Compute hip-to-shoulder vector angle relative to horizontal.

        Reference: research/8.3 - Detects lateral falls via roll angle.
        Uses: Shoulders (MP 11,12), Hips (MP 23,24).

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            Angle in degrees [0, 90]. NaN if keypoints insufficient.
        """
        sh_ok, shoulder_mid = self._midpoint_valid(
            keypoints, "left_shoulder", "right_shoulder"
        )
        hip_ok, hip_mid = self._midpoint_valid(
            keypoints, "left_hip", "right_hip"
        )
        if not (sh_ok and hip_ok):
            return float("nan")

        torso_vec = shoulder_mid - hip_mid
        # Angle relative to horizontal axis
        horizontal = np.array([1.0, 0.0])
        cos_angle = np.dot(torso_vec, horizontal) / (
            np.linalg.norm(torso_vec) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # Return absolute angle from horizontal [0, 90]
        angle = np.degrees(np.arccos(abs(cos_angle)))
        return float(angle)

    def compute_bbox_aspect_ratio(self, keypoints: np.ndarray) -> float:
        """
        Compute width/height ratio of keypoint bounding box.

        Reference: research/2.2, research/8.3 - AR > 1.0 = horizontal.
        Uses: All valid keypoints.

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            Aspect ratio (width/height). NaN if < 2 valid keypoints.
        """
        valid_mask = np.array([
            self._is_valid(keypoints, i) for i in range(len(keypoints))
        ])
        if valid_mask.sum() < 2:
            return float("nan")

        valid_xy = keypoints[valid_mask, :2]
        x_min, y_min = valid_xy.min(axis=0)
        x_max, y_max = valid_xy.max(axis=0)

        width = x_max - x_min
        height = y_max - y_min

        if height < 1e-8:
            return float("nan")

        return float(width / height)

    def compute_com_velocity(
        self,
        keypoints: np.ndarray,
        prev_keypoints: np.ndarray,
        dt: float,
    ) -> float:
        """
        Compute vertical velocity of center of mass (hip center).

        Reference: research/3.2 - Velocity > 0.09 m/s indicates fall.
        Uses: Hips (MP 23,24 / COCO 11,12).

        Parameters
        ----------
        keypoints : np.ndarray
            Current frame keypoints of shape (N, 4).
        prev_keypoints : np.ndarray
            Previous frame keypoints of shape (N, 4).
        dt : float
            Time delta in seconds.

        Returns
        -------
        float
            Vertical velocity (positive = downward). NaN if unavailable.
        """
        if dt <= 0:
            return float("nan")

        curr_ok, curr_hip = self._midpoint_valid(
            keypoints, "left_hip", "right_hip"
        )
        prev_ok, prev_hip = self._midpoint_valid(
            prev_keypoints, "left_hip", "right_hip"
        )
        if not (curr_ok and prev_ok):
            return float("nan")

        # In image coordinates, y increases downward, so positive dy = downward
        dy = curr_hip[1] - prev_hip[1]
        return float(dy / dt)

    def compute_com_acceleration(
        self,
        velocity: float,
        prev_velocity: float,
        dt: float,
    ) -> float:
        """
        Compute vertical acceleration of center of mass.

        Reference: research/3.2 - Threshold at 80% of g (7.84 m/s^2).

        Parameters
        ----------
        velocity : float
            Current frame velocity.
        prev_velocity : float
            Previous frame velocity.
        dt : float
            Time delta in seconds.

        Returns
        -------
        float
            Vertical acceleration. NaN if inputs are NaN.
        """
        if np.isnan(velocity) or np.isnan(prev_velocity) or dt <= 0:
            return float("nan")

        return float((velocity - prev_velocity) / dt)

    def compute_head_to_toe_distance(self, keypoints: np.ndarray) -> float:
        """
        Compute vertical distance from nose to average ankle position.

        Reference: research/8.3 - Height ratio < 0.5 indicates fall.
        Uses: Nose (MP 0), Ankles (MP 27,28 / COCO 15,16).

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            Vertical distance (unnormalized). NaN if keypoints insufficient.
        """
        nose_idx = self._idx["nose"]
        if not self._is_valid(keypoints, nose_idx):
            return float("nan")

        ankle_ok, ankle_mid = self._midpoint_valid(
            keypoints, "left_ankle", "right_ankle"
        )
        if not ankle_ok:
            return float("nan")

        nose_y = keypoints[nose_idx, 1]
        # Absolute vertical distance (image coords: ankle_y > nose_y = standing)
        return float(abs(ankle_mid[1] - nose_y))

    def compute_shoulder_ankle_distance(self, keypoints: np.ndarray) -> float:
        """
        Compute Euclidean distance from shoulder center to ankle center.

        Reference: research/8.3 - Robust substitute when head is occluded.
        Uses: Shoulders (MP 11,12), Ankles (MP 27,28).

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            L2 distance. NaN if keypoints insufficient.
        """
        sh_ok, shoulder_mid = self._midpoint_valid(
            keypoints, "left_shoulder", "right_shoulder"
        )
        ankle_ok, ankle_mid = self._midpoint_valid(
            keypoints, "left_ankle", "right_ankle"
        )
        if not (sh_ok and ankle_ok):
            return float("nan")

        return float(np.linalg.norm(shoulder_mid - ankle_mid))

    def compute_joint_angle(
        self,
        keypoints: np.ndarray,
        joint_a: int,
        joint_b: int,
        joint_c: int,
    ) -> float:
        """
        Compute the angle at joint_b formed by vectors BA and BC.

        Used for knee angles and hip angles.

        Reference: research/3.2 - Knee and hip angles for fine-grained ADL vs fall.

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).
        joint_a : int
            Index of the first endpoint.
        joint_b : int
            Index of the vertex joint.
        joint_c : int
            Index of the second endpoint.

        Returns
        -------
        float
            Angle in degrees [0, 180]. NaN if any keypoint is invalid.
        """
        if not all(self._is_valid(keypoints, j)
                   for j in (joint_a, joint_b, joint_c)):
            return float("nan")

        a = self._get_xy(keypoints, joint_a)
        b = self._get_xy(keypoints, joint_b)
        c = self._get_xy(keypoints, joint_c)

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return float("nan")

        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def compute_wrist_hip_distance(self, keypoints: np.ndarray) -> float:
        """
        Compute average wrist-to-hip distance (protective arm reflex).

        Reference: research/8.3 - Arms flail outward during falls,
        increasing this distance dramatically.
        Uses: Wrists (MP 15,16), Hips (MP 23,24).

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            Average L2 distance. NaN if keypoints insufficient.
        """
        hip_ok, hip_mid = self._midpoint_valid(
            keypoints, "left_hip", "right_hip"
        )
        if not hip_ok:
            return float("nan")

        lw_idx = self._idx["left_wrist"]
        rw_idx = self._idx["right_wrist"]

        distances = []
        if self._is_valid(keypoints, lw_idx):
            distances.append(
                np.linalg.norm(self._get_xy(keypoints, lw_idx) - hip_mid)
            )
        if self._is_valid(keypoints, rw_idx):
            distances.append(
                np.linalg.norm(self._get_xy(keypoints, rw_idx) - hip_mid)
            )

        if not distances:
            return float("nan")

        return float(np.mean(distances))

    def compute_body_spread(self, keypoints: np.ndarray) -> float:
        """
        Compute horizontal span of all valid keypoints.

        Reference: research/8.3 - Horizontal sprawl triples during lying.
        Uses: All valid keypoints.

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4).

        Returns
        -------
        float
            max_x - min_x of valid keypoints. NaN if < 2 valid.
        """
        valid_mask = np.array([
            self._is_valid(keypoints, i) for i in range(len(keypoints))
        ])
        if valid_mask.sum() < 2:
            return float("nan")

        x_vals = keypoints[valid_mask, 0]
        return float(x_vals.max() - x_vals.min())

    def compute_visibility_confidence(
        self, keypoints: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute mean and minimum core keypoint visibility.

        Reference: research/8.3 - Sudden visibility drop correlates
        with person collapsing out of view.

        Parameters
        ----------
        keypoints : np.ndarray
            Array of shape (N, 4) where column 3 is visibility.

        Returns
        -------
        tuple of (float, float)
            (mean_visibility, min_core_visibility).
        """
        vis_col = 3 if keypoints.shape[1] >= 4 else 2

        # Mean of all keypoints
        all_vis = keypoints[:, vis_col]
        valid_vis = all_vis[~np.isnan(all_vis)]
        mean_vis = float(np.mean(valid_vis)) if len(valid_vis) > 0 else float("nan")

        # Min of core keypoints
        core_indices = [self._idx[name] for name in _CORE_JOINTS
                        if self._idx[name] < len(keypoints)]
        core_vis = [keypoints[i, vis_col] for i in core_indices
                    if not np.isnan(keypoints[i, vis_col])]
        min_core = float(min(core_vis)) if core_vis else float("nan")

        return mean_vis, min_core

    def extract_sequence(
        self,
        keypoint_sequence: np.ndarray,
        dt: float = 1.0 / 15.0,
    ) -> np.ndarray:
        """
        Extract features from a full keypoint sequence.

        Iterates over frames computing per-frame features including
        temporal features (velocity, acceleration) using previous frames.

        Parameters
        ----------
        keypoint_sequence : np.ndarray
            Shape (T, N, 4) — T frames, N keypoints, 4 columns.
        dt : float
            Time delta between frames in seconds.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (T, NUM_FEATURES).
        """
        T = keypoint_sequence.shape[0]
        features = np.full(
            (T, self.NUM_FEATURES), np.nan, dtype=np.float32
        )

        prev_velocity = float("nan")

        for t in range(T):
            kps = keypoint_sequence[t]
            prev_kps = keypoint_sequence[t - 1] if t > 0 else None

            features[t] = self.extract(
                kps,
                prev_keypoints=prev_kps,
                dt=dt,
                prev_velocity=prev_velocity,
            )

            prev_velocity = features[t, 3]  # CoM velocity for next frame

        return features
