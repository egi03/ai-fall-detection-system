"""
Keypoint normalization, imputation, and augmentation.

Handles missing keypoints, applies person-centric normalization,
and provides data augmentation for keypoint sequences.

Reference: research/8.2 - Person-centric normalization is the gold standard.
Reference: research/8.3 - NaN handling, linear interpolation for short gaps,
forward-fill for terminal gaps. ASH scale-invariant normalization.
Reference: research/4.1 - Horizontal flip, Gaussian noise, random scaling.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pose.keypoints import COCOKeypoint, MediaPipeKeypoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Pairs of left/right keypoints for horizontal flip (MediaPipe)
_MEDIAPIPE_LR_PAIRS: List[Tuple[int, int]] = [
    (MediaPipeKeypoint.LEFT_EYE_INNER, MediaPipeKeypoint.RIGHT_EYE_INNER),
    (MediaPipeKeypoint.LEFT_EYE, MediaPipeKeypoint.RIGHT_EYE),
    (MediaPipeKeypoint.LEFT_EYE_OUTER, MediaPipeKeypoint.RIGHT_EYE_OUTER),
    (MediaPipeKeypoint.LEFT_EAR, MediaPipeKeypoint.RIGHT_EAR),
    (MediaPipeKeypoint.MOUTH_LEFT, MediaPipeKeypoint.MOUTH_RIGHT),
    (MediaPipeKeypoint.LEFT_SHOULDER, MediaPipeKeypoint.RIGHT_SHOULDER),
    (MediaPipeKeypoint.LEFT_ELBOW, MediaPipeKeypoint.RIGHT_ELBOW),
    (MediaPipeKeypoint.LEFT_WRIST, MediaPipeKeypoint.RIGHT_WRIST),
    (MediaPipeKeypoint.LEFT_PINKY, MediaPipeKeypoint.RIGHT_PINKY),
    (MediaPipeKeypoint.LEFT_INDEX, MediaPipeKeypoint.RIGHT_INDEX),
    (MediaPipeKeypoint.LEFT_THUMB, MediaPipeKeypoint.RIGHT_THUMB),
    (MediaPipeKeypoint.LEFT_HIP, MediaPipeKeypoint.RIGHT_HIP),
    (MediaPipeKeypoint.LEFT_KNEE, MediaPipeKeypoint.RIGHT_KNEE),
    (MediaPipeKeypoint.LEFT_ANKLE, MediaPipeKeypoint.RIGHT_ANKLE),
    (MediaPipeKeypoint.LEFT_HEEL, MediaPipeKeypoint.RIGHT_HEEL),
    (MediaPipeKeypoint.LEFT_FOOT_INDEX, MediaPipeKeypoint.RIGHT_FOOT_INDEX),
]

# Pairs of left/right keypoints for horizontal flip (COCO)
_COCO_LR_PAIRS: List[Tuple[int, int]] = [
    (COCOKeypoint.LEFT_EYE, COCOKeypoint.RIGHT_EYE),
    (COCOKeypoint.LEFT_EAR, COCOKeypoint.RIGHT_EAR),
    (COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.RIGHT_SHOULDER),
    (COCOKeypoint.LEFT_ELBOW, COCOKeypoint.RIGHT_ELBOW),
    (COCOKeypoint.LEFT_WRIST, COCOKeypoint.RIGHT_WRIST),
    (COCOKeypoint.LEFT_HIP, COCOKeypoint.RIGHT_HIP),
    (COCOKeypoint.LEFT_KNEE, COCOKeypoint.RIGHT_KNEE),
    (COCOKeypoint.LEFT_ANKLE, COCOKeypoint.RIGHT_ANKLE),
]

# Keypoint indices used for normalization
_NORM_INDICES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "mediapipe": {
        "left_hip": MediaPipeKeypoint.LEFT_HIP,
        "right_hip": MediaPipeKeypoint.RIGHT_HIP,
        "left_shoulder": MediaPipeKeypoint.LEFT_SHOULDER,
        "right_shoulder": MediaPipeKeypoint.RIGHT_SHOULDER,
    },
    "coco": {
        "left_hip": COCOKeypoint.LEFT_HIP,
        "right_hip": COCOKeypoint.RIGHT_HIP,
        "left_shoulder": COCOKeypoint.LEFT_SHOULDER,
        "right_shoulder": COCOKeypoint.RIGHT_SHOULDER,
    },
}

# Small epsilon to prevent division by zero
_EPSILON = 1e-6


class KeypointPreprocessor:
    """
    Preprocesses raw keypoint sequences for model input.

    Applies a three-step pipeline per frame:
    1. Filter low-confidence keypoints (replace with NaN)
    2. Impute missing values (interpolation / forward-fill)
    3. Person-centric normalization (hip-centered + ASH torso scaling)

    Parameters
    ----------
    keypoint_format : str
        Either 'mediapipe' (33 keypoints) or 'coco' (17 keypoints).
    normalize : bool
        Whether to apply person-centric normalization.
    confidence_threshold : float
        Keypoints below this visibility are treated as missing (NaN).
    """

    def __init__(
        self,
        keypoint_format: str = "mediapipe",
        normalize: bool = True,
        confidence_threshold: float = 0.5,
    ) -> None:
        if keypoint_format not in ("mediapipe", "coco"):
            raise ValueError(
                f"Unknown keypoint format: {keypoint_format}. "
                "Must be 'mediapipe' or 'coco'."
            )
        self._format = keypoint_format
        self._normalize = normalize
        self._conf_threshold = confidence_threshold
        self._indices = _NORM_INDICES[keypoint_format]

    @property
    def keypoint_format(self) -> str:
        """Return the keypoint format string."""
        return self._format

    def preprocess(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline to a single frame's keypoints.

        Steps: filter low confidence -> impute missing -> normalize.

        Parameters
        ----------
        keypoints : np.ndarray
            Raw keypoints of shape (N, 4) with (x, y, z, visibility).

        Returns
        -------
        np.ndarray
            Preprocessed keypoints of shape (N, 4).
        """
        result = self._filter_low_confidence(keypoints.copy())
        result = self.impute_missing(result)
        if self._normalize:
            result = self.normalize_person_centric(result)
        return result

    def preprocess_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing to a temporal sequence of keypoints.

        Uses temporal interpolation for missing values across frames.

        Parameters
        ----------
        sequence : np.ndarray
            Keypoint sequence of shape (T, N, 4) where T = num frames,
            N = num keypoints, 4 = (x, y, z, visibility).

        Returns
        -------
        np.ndarray
            Preprocessed sequence of shape (T, N, 4).
        """
        T, N, C = sequence.shape
        result = sequence.copy()

        # Step 1: Mask low-confidence keypoints as NaN across all frames
        for t in range(T):
            result[t] = self._filter_low_confidence(result[t])

        # Step 2: Temporal imputation — interpolate per keypoint, per coord
        result = self._temporal_impute(result)

        # Step 3: Person-centric normalization per frame
        if self._normalize:
            for t in range(T):
                result[t] = self.normalize_person_centric(result[t])

        return result

    def _filter_low_confidence(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Replace keypoints with visibility below threshold with NaN.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape (N, 4) with columns (x, y, z, visibility).

        Returns
        -------
        np.ndarray
            Keypoints with low-confidence entries set to NaN.
        """
        # Visibility is the last column (index 3)
        if keypoints.shape[1] >= 4:
            low_conf = keypoints[:, 3] < self._conf_threshold
        elif keypoints.shape[1] == 3:
            # (x, y, confidence) format — confidence is column 2
            low_conf = keypoints[:, 2] < self._conf_threshold
        else:
            return keypoints

        keypoints[low_conf, :] = np.nan
        return keypoints

    def _temporal_impute(self, sequence: np.ndarray) -> np.ndarray:
        """
        Impute missing keypoints across time using linear interpolation
        for interior gaps and forward/backward fill for terminal gaps.

        Reference: research/8.3 - Linear interpolation for short gaps,
        forward-fill for terminal gaps.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, N, C).

        Returns
        -------
        np.ndarray
            Imputed sequence.
        """
        T, N, C = sequence.shape

        for kp_idx in range(N):
            for coord_idx in range(C):
                values = sequence[:, kp_idx, coord_idx]
                valid_mask = ~np.isnan(values)

                if valid_mask.all():
                    continue
                if not valid_mask.any():
                    # Entire keypoint missing across all frames — leave as NaN
                    continue

                valid_indices = np.where(valid_mask)[0]
                invalid_indices = np.where(~valid_mask)[0]

                # Linear interpolation for interior gaps
                interpolated = np.interp(
                    invalid_indices,
                    valid_indices,
                    values[valid_indices],
                )
                sequence[invalid_indices, kp_idx, coord_idx] = interpolated

        return sequence

    def normalize_person_centric(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply person-centric (root-relative) normalization.

        DECISION: Using ASH normalization per research/8.3. Translates
        keypoints relative to hip center, scales by shoulder-hip distance.
        This isolates kinematics from camera perspective and subject size.

        Parameters
        ----------
        keypoints : np.ndarray
            Keypoints of shape (N, C) where C >= 3.

        Returns
        -------
        np.ndarray
            Normalized keypoints of shape (N, C).
        """
        idx = self._indices
        left_hip = keypoints[idx["left_hip"]]
        right_hip = keypoints[idx["right_hip"]]
        left_shoulder = keypoints[idx["left_shoulder"]]
        right_shoulder = keypoints[idx["right_shoulder"]]

        # Check if essential keypoints are available
        essential = np.stack([left_hip, right_hip, left_shoulder, right_shoulder])
        if np.isnan(essential[:, :3]).any():
            # Cannot normalize without hip/shoulder keypoints
            return keypoints

        # Hip center (root joint)
        hip_center = (left_hip[:3] + right_hip[:3]) / 2.0

        # Shoulder center
        shoulder_center = (left_shoulder[:3] + right_shoulder[:3]) / 2.0

        # ASH torso length (normalization scalar)
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        if torso_length < _EPSILON:
            return keypoints

        result = keypoints.copy()

        # Translate: center on hip midpoint (spatial coords only)
        num_spatial = min(3, keypoints.shape[1])
        result[:, :num_spatial] = result[:, :num_spatial] - hip_center[:num_spatial]

        # Scale: divide by torso length
        result[:, :num_spatial] = result[:, :num_spatial] / torso_length

        return result

    def impute_missing(
        self, keypoints: np.ndarray, prev_keypoints: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Impute missing (NaN) keypoints using forward fill from previous frame.

        For real-time single-frame processing where temporal interpolation
        is not possible.

        Parameters
        ----------
        keypoints : np.ndarray
            Keypoints with potential NaN values, shape (N, C).
        prev_keypoints : np.ndarray, optional
            Previous frame's keypoints for forward-fill.

        Returns
        -------
        np.ndarray
            Imputed keypoints of shape (N, C).
        """
        result = keypoints.copy()

        if prev_keypoints is not None:
            nan_mask = np.isnan(result).any(axis=1)
            valid_prev = ~np.isnan(prev_keypoints).any(axis=1)
            fill_mask = nan_mask & valid_prev
            result[fill_mask] = prev_keypoints[fill_mask]

        return result


class KeypointAugmenter:
    """
    Data augmentation for keypoint sequences.

    DECISION: Using horizontal flip, Gaussian noise, and random scaling
    per research/4.1. These augmentations force the LSTM to learn
    invariant physical laws rather than specific spatial patterns.

    Parameters
    ----------
    flip_probability : float
        Probability of applying horizontal flip.
    noise_std : float
        Standard deviation of Gaussian noise to add to coordinates.
    scale_range : tuple of float
        Min and max scale factors for random scaling.
    keypoint_format : str
        Either 'mediapipe' or 'coco' — determines which left/right
        pairs to swap during horizontal flip.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        flip_probability: float = 0.5,
        noise_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        keypoint_format: str = "mediapipe",
        seed: int = 42,
    ) -> None:
        self._flip_prob = flip_probability
        self._noise_std = noise_std
        self._scale_range = scale_range
        self._rng = np.random.RandomState(seed)

        if keypoint_format == "mediapipe":
            self._lr_pairs = _MEDIAPIPE_LR_PAIRS
        elif keypoint_format == "coco":
            self._lr_pairs = _COCO_LR_PAIRS
        else:
            raise ValueError(f"Unknown keypoint format: {keypoint_format}")

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to a keypoint sequence.

        Augmentations applied (each independently with their own probability):
        1. Horizontal flip (swap left/right keypoints, negate x)
        2. Gaussian noise on spatial coordinates
        3. Random uniform scaling of spatial coordinates

        Parameters
        ----------
        sequence : np.ndarray
            Sequence of shape (T, N, C) where T = frames,
            N = keypoints, C = coords (x, y, z, visibility).

        Returns
        -------
        np.ndarray
            Augmented sequence of the same shape.
        """
        result = sequence.copy()

        # 1. Horizontal flip
        if self._rng.random() < self._flip_prob:
            result = self._horizontal_flip(result)

        # 2. Gaussian noise on spatial coordinates
        if self._noise_std > 0:
            result = self._add_noise(result)

        # 3. Random scaling
        if self._scale_range != (1.0, 1.0):
            result = self._random_scale(result)

        return result

    def _horizontal_flip(self, sequence: np.ndarray) -> np.ndarray:
        """
        Flip keypoints horizontally by negating x-coordinates
        and swapping left/right keypoint pairs.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, N, C).

        Returns
        -------
        np.ndarray
            Flipped sequence.
        """
        result = sequence.copy()

        # Negate x-coordinate (column 0)
        result[:, :, 0] = -result[:, :, 0]

        # Swap left/right keypoint pairs
        for left_idx, right_idx in self._lr_pairs:
            temp = result[:, left_idx, :].copy()
            result[:, left_idx, :] = result[:, right_idx, :]
            result[:, right_idx, :] = temp

        return result

    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to spatial coordinates only.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, N, C).

        Returns
        -------
        np.ndarray
            Noisy sequence.
        """
        result = sequence.copy()
        num_spatial = min(3, result.shape[2])

        noise = self._rng.normal(0, self._noise_std, result[:, :, :num_spatial].shape)

        # Only add noise where values are not NaN
        valid = ~np.isnan(result[:, :, :num_spatial])
        result[:, :, :num_spatial] = np.where(
            valid,
            result[:, :, :num_spatial] + noise,
            result[:, :, :num_spatial],
        )
        return result

    def _random_scale(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply random uniform scaling to spatial coordinates.

        Parameters
        ----------
        sequence : np.ndarray
            Shape (T, N, C).

        Returns
        -------
        np.ndarray
            Scaled sequence.
        """
        result = sequence.copy()
        num_spatial = min(3, result.shape[2])

        scale = self._rng.uniform(self._scale_range[0], self._scale_range[1])
        result[:, :, :num_spatial] *= scale

        return result
