"""
PyTorch Dataset for fall detection temporal sequences.

Loads preprocessed keypoint sequences and their labels,
applying sliding window extraction and optional data augmentation.

Provides two main workflows:
1. FallDetectionDataset: Takes pre-windowed sequences
2. build_dataset_from_processed: Constructs windows from full
   preprocessed keypoint sequences with configurable stride

Reference: research/8.2 - Sliding window with configurable stride,
label assignment via positive frame ratio threshold.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_processing.preprocessor import KeypointAugmenter
from src.features.extractor import FeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Label constants
LABEL_ADL = 0
LABEL_FALL = 1


class FallDetectionDataset(Dataset):
    """
    PyTorch Dataset for loading windowed keypoint feature sequences.

    Parameters
    ----------
    sequences : np.ndarray
        Array of shape (N, window_size, num_features).
    labels : np.ndarray
        Binary labels of shape (N,). 1 = fall, 0 = non-fall.
    augmenter : KeypointAugmenter, optional
        Augmenter to apply during training. Pass None for val/test.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augmenter: Optional[KeypointAugmenter] = None,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError(
                f"Mismatch: {len(sequences)} sequences vs {len(labels)} labels"
            )
        if len(sequences) == 0:
            raise ValueError("Dataset must contain at least one sequence.")

        self._sequences = sequences.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self._augmenter = augmenter

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single (sequence, label) pair.

        If an augmenter is configured, it is applied to the sequence.
        Note: augmentation expects shape (T, N, C) where N = keypoints,
        but feature sequences are (T, F) where F = features. For
        feature-level sequences, only noise and scaling are applied.

        Parameters
        ----------
        idx : int
            Index of the sequence.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            Sequence tensor of shape (window_size, num_features),
            Label scalar tensor.
        """
        seq = self._sequences[idx].copy()
        label = self._labels[idx]

        if self._augmenter is not None:
            # For feature vectors (T, F), reshape to (T, F, 1) for augmenter
            # then squeeze back. Only noise and scaling will be effective.
            seq_3d = seq[:, :, np.newaxis]
            seq_3d = self._augmenter.augment(seq_3d)
            seq = seq_3d[:, :, 0]

        return (
            torch.from_numpy(seq),
            torch.tensor(label, dtype=torch.long),
        )

    @property
    def num_falls(self) -> int:
        """Number of fall sequences."""
        return int((self._labels == LABEL_FALL).sum())

    @property
    def num_adl(self) -> int:
        """Number of ADL sequences."""
        return int((self._labels == LABEL_ADL).sum())

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for balanced training.

        Reference: research/8.2 - Automatic class weight balancing
        to handle fall/ADL imbalance.

        Returns
        -------
        torch.Tensor
            Weight per class, shape (2,).
        """
        counts = np.bincount(self._labels, minlength=2)
        total = counts.sum()
        weights = total / (2.0 * counts.astype(np.float64))
        return torch.tensor(weights, dtype=torch.float32)


def extract_windows(
    keypoints: np.ndarray,
    label: int,
    window_size: int = 30,
    stride: int = 5,
    annotation: Optional[Dict] = None,
    positive_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Extract sliding windows from a full keypoint sequence.

    For fall sequences with frame-level annotations, the window label
    is determined by the ratio of fall frames within the window.

    Parameters
    ----------
    keypoints : np.ndarray
        Full keypoint sequence of shape (T, N, C).
    label : int
        Sequence-level label (0 = ADL, 1 = fall).
    window_size : int
        Number of frames per window.
    stride : int
        Step between consecutive windows.
    annotation : dict, optional
        Frame-level annotation with 'start_frame' and 'end_frame' keys.
    positive_threshold : float
        Minimum ratio of fall frames in a window to label it as fall.

    Returns
    -------
    tuple of (list of np.ndarray, list of int)
        Windows of shape (window_size, N, C) and their labels.
    """
    T = keypoints.shape[0]

    if T < window_size:
        # Pad short sequences by repeating the last frame
        pad_size = window_size - T
        padding = np.tile(keypoints[-1:], (pad_size, 1, 1))
        keypoints = np.concatenate([keypoints, padding], axis=0)
        T = window_size

    windows = []
    labels = []

    # Build frame-level label array
    frame_labels = np.full(T, label, dtype=np.int64)
    if annotation and label == LABEL_FALL:
        start = annotation.get("start_frame", 0)
        end = annotation.get("end_frame", T)
        # Only fall frames within the annotated range are positive
        frame_labels[:] = LABEL_ADL
        frame_labels[max(0, start) : min(T, end + 1)] = LABEL_FALL

    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size
        window = keypoints[start_idx:end_idx]

        # Determine window label
        window_labels = frame_labels[start_idx:end_idx]
        fall_ratio = window_labels.sum() / window_size

        if fall_ratio >= positive_threshold:
            window_label = LABEL_FALL
        else:
            window_label = LABEL_ADL

        windows.append(window)
        labels.append(window_label)

    return windows, labels


def build_dataset_from_processed(
    processed_dir: Path,
    subject_ids: List[str],
    window_size: int = 30,
    stride: int = 5,
    positive_threshold: float = 0.5,
    keypoint_format: str = "mediapipe",
    target_fps: float = 15.0,
    norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    apply_norm: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build windowed dataset arrays from processed keypoint files.

    Loads all processed sequences belonging to the specified subjects,
    computes 15 fall-detection features per frame using FeatureExtractor,
    extracts sliding windows, and concatenates into arrays suitable
    for FallDetectionDataset.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed keypoint .npy files and metadata.json.
    subject_ids : list of str
        Subject IDs to include (for train/val/test splitting).
    window_size : int
        Window size in frames.
    stride : int
        Stride between windows.
    positive_threshold : float
        Fall frame ratio threshold for window labeling.
    keypoint_format : str
        Keypoint format for feature extraction ('mediapipe' or 'coco').
    target_fps : float
        Target FPS for temporal feature computation (dt = 1/fps).
    norm_stats : tuple of (np.ndarray, np.ndarray), optional
        Pre-computed (mean, std) from training set. When provided, these
        stats are used instead of computing from this data.
    apply_norm : bool
        If True (default), apply z-score normalization before returning.
        Set to False when combining multiple datasets — caller normalizes
        the concatenated arrays using build_combined_dataset().

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, (np.ndarray, np.ndarray))
        Sequences of shape (N, window_size, 15), labels of shape (N,),
        and (mean, std) normalization stats.
    """
    processed_dir = Path(processed_dir)
    metadata_path = processed_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    extractor = FeatureExtractor(
        keypoint_format=keypoint_format,
        confidence_threshold=0.5,
    )
    dt = 1.0 / target_fps

    subject_set = set(subject_ids)
    all_windows = []
    all_labels = []

    # Use raw keypoints (pre-normalization) so that temporal features
    # (velocity, acceleration) are computed from absolute positions.
    # Person-centric normalization destroys CoM velocity (hip always at 0).
    # The FeatureExtractor handles scale invariance internally.
    kp_dir = processed_dir / "keypoints_raw"

    for seq_meta in metadata["sequences"]:
        if seq_meta["subject_id"] not in subject_set:
            continue

        seq_id = seq_meta["sequence_id"]
        label = seq_meta["label"]

        npy_path = kp_dir / f"{seq_id}.npy"
        if not npy_path.exists():
            logger.warning(f"Missing processed file: {npy_path}")
            continue

        keypoints = np.load(str(npy_path))

        # Compute 15 features per frame: (T, N, 4) -> (T, 15)
        feature_seq = extractor.extract_sequence(keypoints, dt=dt)

        # Replace any remaining NaN with 0 for model input
        feature_seq = np.nan_to_num(feature_seq, nan=0.0)

        annotation = {}
        if "start_frame" in seq_meta:
            annotation["start_frame"] = seq_meta.get("start_frame", 0)
            annotation["end_frame"] = seq_meta.get(
                "end_frame", keypoints.shape[0]
            )

        # extract_windows expects (T, N, C) but feature_seq is (T, 15)
        # Reshape to (T, 15, 1) for compatibility, then squeeze after
        feature_3d = feature_seq[:, :, np.newaxis]
        windows, labels = extract_windows(
            feature_3d,
            label,
            window_size=window_size,
            stride=stride,
            annotation=annotation if annotation else None,
            positive_threshold=positive_threshold,
        )

        all_windows.extend(windows)
        all_labels.extend(labels)

    if not all_windows:
        raise ValueError(
            f"No windows extracted for subjects {subject_ids}. "
            "Check that processed data exists."
        )

    # Squeeze the trailing dim: (window_size, 15, 1) -> (window_size, 15)
    sequences = []
    for w in all_windows:
        if w.ndim == 3:
            sequences.append(w.reshape(w.shape[0], -1))
        else:
            sequences.append(w)

    sequences_arr = np.array(sequences, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    # Per-feature z-score normalization.
    # If norm_stats=(mean, std) are provided (from training set), use them.
    # Otherwise compute from this data (training set call).
    # This ensures test data is normalized with training statistics.
    if norm_stats is not None:
        mean, std = norm_stats
    else:
        mean = sequences_arr.mean(axis=(0, 1), keepdims=True)  # (1, 1, 15)
        std = sequences_arr.std(axis=(0, 1), keepdims=True) + 1e-8

    if apply_norm:
        sequences_arr = (sequences_arr - mean) / std

    logger.info(
        f"Built dataset: {len(sequences_arr)} windows, "
        f"{(labels_arr == LABEL_FALL).sum()} falls, "
        f"{(labels_arr == LABEL_ADL).sum()} ADLs, "
        f"features_per_frame={FeatureExtractor.NUM_FEATURES}"
    )

    return sequences_arr, labels_arr, (mean, std)


def _person_centric_normalize(keypoints: np.ndarray) -> np.ndarray:
    """Recenter on hip midpoint and scale by torso length.

    Operates frame-by-frame on the 13-joint body subset emitted by
    :func:`build_keypoint_dataset_from_processed`. Channels expected
    along axis 2: ``(x, y, visibility)``.

    Parameters
    ----------
    keypoints : np.ndarray
        Shape ``(T, 13, 3)``.

    Returns
    -------
    np.ndarray
        Normalized keypoints with hip midpoint at origin and unit torso
        length. Visibility channel is untouched.
    """
    from src.models.stgcn import STGCN_NUM_JOINTS

    if keypoints.shape[1] != STGCN_NUM_JOINTS:
        raise ValueError(
            f"_person_centric_normalize expects {STGCN_NUM_JOINTS} joints; "
            f"got {keypoints.shape[1]}."
        )

    xy = keypoints[:, :, :2].astype(np.float32, copy=True)
    vis = keypoints[:, :, 2:3].astype(np.float32, copy=True)

    # Hip midpoint (subset indices 7=L_HIP, 8=R_HIP)
    hip = (xy[:, 7, :] + xy[:, 8, :]) * 0.5  # (T, 2)
    # Shoulder midpoint (subset indices 1=L_SH, 2=R_SH)
    shoulder = (xy[:, 1, :] + xy[:, 2, :]) * 0.5  # (T, 2)

    torso_len = np.linalg.norm(shoulder - hip, axis=1, keepdims=True)  # (T, 1)
    torso_len = np.where(torso_len < 1e-3, 1.0, torso_len)  # avoid /0

    xy_centered = xy - hip[:, None, :]
    xy_scaled = xy_centered / torso_len[:, None, :]
    return np.concatenate([xy_scaled, vis], axis=2).astype(np.float32)


# Pairs of (left_index, right_index) within the 13-joint body subset
# used by horizontal-flip augmentation. The nose stays in place.
STGCN_LR_FLIP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),    # shoulder
    (3, 4),    # elbow
    (5, 6),    # wrist
    (7, 8),    # hip
    (9, 10),   # knee
    (11, 12),  # ankle
]


def _append_motion_channels(seq: np.ndarray) -> np.ndarray:
    """Append per-frame deltas as additional channels.

    Given a window of shape ``(T, V, C0)`` returns ``(T, V, 2*C0)`` where
    the trailing ``C0`` channels are the frame-to-frame differences along
    the time axis (zero for ``t == 0``). For ST-GCN this gives the model
    explicit motion features without changing the spatial graph.
    """
    diffs = np.zeros_like(seq)
    diffs[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, diffs], axis=2)


def _subset_body_joints(keypoints: np.ndarray) -> np.ndarray:
    """Slice the full 33-landmark MediaPipe array to the 13-joint body subset.

    Drops the ``z`` channel — only ``(x, y, visibility)`` are needed for the
    spatial graph. The returned array has shape ``(T, 13, 3)``.
    """
    from src.models.stgcn import STGCN_BODY_JOINTS

    selected = keypoints[:, STGCN_BODY_JOINTS, :]  # (T, 13, 4)
    if selected.shape[2] >= 4:
        xy_vis = np.concatenate(
            [selected[:, :, :2], selected[:, :, 3:4]], axis=2,
        )
    else:
        xy_vis = selected[:, :, :3]
    return xy_vis.astype(np.float32)


def build_keypoint_dataset_from_processed(
    processed_dir: Path,
    subject_ids: List[str],
    window_size: int = 30,
    stride: int = 5,
    positive_threshold: float = 0.5,
    motion: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build windowed keypoint dataset for ST-GCN training.

    Loads raw MediaPipe keypoints, subsets to the 13-joint body graph,
    applies person-centric normalization (recenter on hips, scale by
    torso length), and extracts sliding windows.

    The returned tensor layout matches what :class:`src.models.stgcn.STGCN`
    expects: ``(N, T, V, C)`` with ``V = 13`` and ``C = 3``.

    Parameters
    ----------
    processed_dir : Path
        Root of the processed dataset (must contain ``metadata.json`` and
        ``keypoints_raw/``).
    subject_ids : list of str
        Subjects to include — typically the training side of a LOSO split.
    window_size : int
        Number of frames per window.
    stride : int
        Stride between consecutive windows.
    positive_threshold : float
        Fall-frame ratio above which a window is labelled as fall.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(sequences, labels)`` where ``sequences`` has shape
        ``(N, window_size, 13, 3)`` and ``labels`` has shape ``(N,)``.
    """
    processed_dir = Path(processed_dir)
    metadata_path = processed_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    kp_dir = processed_dir / "keypoints_raw"
    subject_set = set(subject_ids)
    all_windows: List[np.ndarray] = []
    all_labels: List[int] = []

    for seq_meta in metadata["sequences"]:
        if seq_meta["subject_id"] not in subject_set:
            continue

        seq_id = seq_meta["sequence_id"]
        label = seq_meta["label"]
        npy_path = kp_dir / f"{seq_id}.npy"
        if not npy_path.exists():
            logger.warning(f"Missing processed file: {npy_path}")
            continue

        keypoints = np.load(str(npy_path))  # (T, 33, 4)
        subset = _subset_body_joints(keypoints)  # (T, 13, 3)
        normalized = _person_centric_normalize(subset)
        normalized = np.nan_to_num(normalized, nan=0.0)
        if motion:
            normalized = _append_motion_channels(normalized)  # (T, 13, 6)

        annotation = {}
        if "start_frame" in seq_meta:
            annotation["start_frame"] = seq_meta.get("start_frame", 0)
            annotation["end_frame"] = seq_meta.get(
                "end_frame", normalized.shape[0]
            )

        windows, labels = extract_windows(
            normalized,
            label,
            window_size=window_size,
            stride=stride,
            annotation=annotation if annotation else None,
            positive_threshold=positive_threshold,
        )
        all_windows.extend(windows)
        all_labels.extend(labels)

    if not all_windows:
        raise ValueError(
            f"No keypoint windows extracted for subjects {subject_ids}."
        )

    sequences_arr = np.array(all_windows, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    logger.info(
        f"Built keypoint dataset: {len(sequences_arr)} windows, "
        f"{(labels_arr == LABEL_FALL).sum()} falls, "
        f"{(labels_arr == LABEL_ADL).sum()} ADLs, "
        f"shape={sequences_arr.shape}"
    )

    return sequences_arr, labels_arr


def build_combined_dataset(
    sources: List[Tuple[Path, List[str]]],
    window_size: int = 30,
    stride: int = 5,
    positive_threshold: float = 0.5,
    keypoint_format: str = "mediapipe",
    target_fps: float = 15.0,
    norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Build a combined windowed dataset from multiple processed directories.

    Extracts raw (unnormalized) windows from each source, concatenates them,
    then applies a single joint z-score normalization. This ensures features
    are on the same scale regardless of which dataset they came from.

    Used for combined URFD+Le2i training: each LOSO fold trains on
    URFD(n-1 subjects) + all Le2i, then tests on the held-out URFD subject
    normalized with the combined training stats.

    Parameters
    ----------
    sources : list of (processed_dir, subject_ids) tuples
        Each tuple specifies a processed data directory and the subject IDs
        to load from it. Pass all subject IDs from auxiliary datasets.
    window_size : int
        Window size in frames.
    stride : int
        Stride between consecutive windows.
    positive_threshold : float
        Fall frame ratio threshold for window labeling.
    keypoint_format : str
        Keypoint format ('mediapipe' or 'coco').
    target_fps : float
        Target FPS for temporal feature computation.
    norm_stats : tuple of (np.ndarray, np.ndarray), optional
        Pre-computed (mean, std). If provided, skips computation and applies
        directly (used for normalizing the test set with training stats).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, (np.ndarray, np.ndarray))
        Combined sequences (N, window_size, 15), labels (N,),
        and joint (mean, std) normalization stats.
    """
    all_seqs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for processed_dir, subject_ids in sources:
        seqs, labels, _ = build_dataset_from_processed(
            processed_dir=processed_dir,
            subject_ids=subject_ids,
            window_size=window_size,
            stride=stride,
            positive_threshold=positive_threshold,
            keypoint_format=keypoint_format,
            target_fps=target_fps,
            norm_stats=None,
            apply_norm=False,  # defer normalization until all sources are merged
        )
        all_seqs.append(seqs)
        all_labels.append(labels)

    combined_seqs = np.concatenate(all_seqs, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    # Compute joint normalization stats (or apply provided stats for test set)
    if norm_stats is not None:
        mean, std = norm_stats
    else:
        mean = combined_seqs.mean(axis=(0, 1), keepdims=True)
        std = combined_seqs.std(axis=(0, 1), keepdims=True) + 1e-8

    combined_seqs = (combined_seqs - mean) / std

    logger.info(
        f"Combined dataset: {len(combined_seqs)} windows from {len(sources)} source(s), "
        f"{(combined_labels == LABEL_FALL).sum()} falls, "
        f"{(combined_labels == LABEL_ADL).sum()} ADLs"
    )

    return combined_seqs, combined_labels, (mean, std)
