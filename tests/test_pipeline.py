"""
Integration tests for the sliding window, dataset building,
and data pipeline components.

Tests the sliding window buffer, window extraction from sequences,
and the PyTorch dataset construction.
"""

import numpy as np
import pytest

from src.features.window import SlidingWindow
from src.training.dataset import (
    FallDetectionDataset,
    extract_windows,
    LABEL_FALL,
    LABEL_ADL,
)


class TestSlidingWindow:
    """Tests for the sliding window buffer."""

    def test_not_ready_when_empty(self) -> None:
        """Window should not be ready before filling."""
        window = SlidingWindow(window_size=5, num_features=3)
        assert not window.is_ready
        assert window.current_size == 0

    def test_ready_after_filling(self) -> None:
        """Window should be ready after window_size pushes."""
        window = SlidingWindow(window_size=5, num_features=3)
        for i in range(5):
            window.push(np.array([float(i)] * 3))
        assert window.is_ready
        assert window.current_size == 5

    def test_not_ready_partial(self) -> None:
        """Window should not be ready with fewer than window_size frames."""
        window = SlidingWindow(window_size=5, num_features=3)
        for i in range(4):
            window.push(np.array([float(i)] * 3))
        assert not window.is_ready

    def test_oldest_frame_discarded(self) -> None:
        """Oldest frame should be automatically discarded on overflow."""
        window = SlidingWindow(window_size=3, num_features=2)
        window.push(np.array([1.0, 1.0]))
        window.push(np.array([2.0, 2.0]))
        window.push(np.array([3.0, 3.0]))
        # Push one more — oldest (1.0) should be discarded
        window.push(np.array([4.0, 4.0]))

        tensor = window.get_tensor()
        assert tensor is not None
        # First frame should be [2.0, 2.0], not [1.0, 1.0]
        np.testing.assert_allclose(tensor[0, 0, :], [2.0, 2.0])
        np.testing.assert_allclose(tensor[0, 2, :], [4.0, 4.0])

    def test_tensor_shape_correct(self) -> None:
        """Output tensor should have shape (1, window_size, num_features)."""
        window = SlidingWindow(window_size=30, num_features=15)
        for i in range(30):
            window.push(np.random.rand(15).astype(np.float32))

        tensor = window.get_tensor()
        assert tensor is not None
        assert tensor.shape == (1, 30, 15)
        assert tensor.dtype == np.float32

    def test_tensor_none_when_not_ready(self) -> None:
        """get_tensor should return None when window is not ready."""
        window = SlidingWindow(window_size=5, num_features=3)
        window.push(np.array([1.0, 1.0, 1.0]))
        assert window.get_tensor() is None

    def test_reset_clears_buffer(self) -> None:
        """Reset should empty the buffer."""
        window = SlidingWindow(window_size=3, num_features=2)
        window.push(np.array([1.0, 1.0]))
        window.push(np.array([2.0, 2.0]))
        window.reset()
        assert not window.is_ready
        assert window.current_size == 0

    def test_invalid_feature_size_raises(self) -> None:
        """Pushing wrong feature size should raise ValueError."""
        window = SlidingWindow(window_size=3, num_features=2)
        with pytest.raises(ValueError, match="Expected feature vector"):
            window.push(np.array([1.0, 2.0, 3.0]))

    def test_invalid_window_size_raises(self) -> None:
        """Window size < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindow(window_size=0, num_features=3)

    def test_get_array_no_batch_dim(self) -> None:
        """get_array should return without batch dimension."""
        window = SlidingWindow(window_size=3, num_features=2)
        for i in range(3):
            window.push(np.array([float(i), float(i)]))

        arr = window.get_array()
        assert arr is not None
        assert arr.shape == (3, 2)

    def test_data_independence(self) -> None:
        """Modifying pushed array should not affect window contents."""
        window = SlidingWindow(window_size=2, num_features=2)
        data = np.array([1.0, 2.0])
        window.push(data)
        data[0] = 999.0  # modify original
        window.push(np.array([3.0, 4.0]))

        tensor = window.get_tensor()
        assert tensor is not None
        np.testing.assert_allclose(tensor[0, 0, 0], 1.0)


class TestWindowExtraction:
    """Tests for extracting sliding windows from full sequences."""

    def test_basic_extraction(self) -> None:
        """Should extract correct number of windows."""
        # 10 frames, window=5, stride=5 -> 2 windows
        kps = np.random.rand(10, 33, 4).astype(np.float32)
        windows, labels = extract_windows(kps, label=LABEL_ADL, window_size=5, stride=5)
        assert len(windows) == 2
        assert all(l == LABEL_ADL for l in labels)

    def test_stride_overlap(self) -> None:
        """Stride < window_size should produce overlapping windows."""
        kps = np.random.rand(10, 33, 4).astype(np.float32)
        windows, labels = extract_windows(kps, label=LABEL_FALL, window_size=5, stride=2)
        # Windows: [0:5], [2:7], [4:9] -> 3 windows
        assert len(windows) == 3

    def test_short_sequence_padded(self) -> None:
        """Sequences shorter than window_size should be padded."""
        kps = np.random.rand(3, 33, 4).astype(np.float32)
        windows, labels = extract_windows(kps, label=LABEL_FALL, window_size=5, stride=5)
        assert len(windows) >= 1
        assert windows[0].shape[0] == 5

    def test_window_shape(self) -> None:
        """Each window should have correct shape."""
        kps = np.random.rand(30, 33, 4).astype(np.float32)
        windows, labels = extract_windows(kps, label=LABEL_ADL, window_size=10, stride=5)
        for w in windows:
            assert w.shape == (10, 33, 4)

    def test_frame_level_annotation(self) -> None:
        """Frame-level annotation should affect window labels."""
        kps = np.random.rand(20, 33, 4).astype(np.float32)
        # Fall occurs at frames 10-15
        annotation = {"start_frame": 10, "end_frame": 15}
        windows, labels = extract_windows(
            kps,
            label=LABEL_FALL,
            window_size=5,
            stride=5,
            annotation=annotation,
            positive_threshold=0.5,
        )
        # Window [0:5] -> all ADL frames -> ADL
        # Window [5:10] -> all ADL frames -> ADL
        # Window [10:15] -> all fall frames -> FALL
        # Window [15:20] -> 1 fall frame out of 5 -> ADL
        assert labels[0] == LABEL_ADL
        assert labels[2] == LABEL_FALL


class TestFallDetectionDataset:
    """Tests for the PyTorch Dataset wrapper."""

    def test_length(self) -> None:
        """Dataset length should match input array length."""
        seqs = np.random.rand(50, 30, 15).astype(np.float32)
        labels = np.zeros(50, dtype=np.int64)
        ds = FallDetectionDataset(seqs, labels)
        assert len(ds) == 50

    def test_getitem_shapes(self) -> None:
        """Items should have correct tensor shapes."""
        seqs = np.random.rand(10, 30, 15).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        ds = FallDetectionDataset(seqs, labels)

        seq_tensor, label_tensor = ds[0]
        assert seq_tensor.shape == (30, 15)
        assert label_tensor.shape == ()

    def test_class_weights(self) -> None:
        """Class weights should be inversely proportional to frequency."""
        seqs = np.random.rand(10, 30, 15).astype(np.float32)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        ds = FallDetectionDataset(seqs, labels)

        weights = ds.class_weights
        # Fall is minority (3/10), should have higher weight
        assert weights[1] > weights[0]

    def test_empty_dataset_raises(self) -> None:
        """Empty dataset should raise ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            FallDetectionDataset(
                np.array([]).reshape(0, 30, 15),
                np.array([], dtype=np.int64),
            )

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched sequence and label counts should raise ValueError."""
        with pytest.raises(ValueError, match="Mismatch"):
            FallDetectionDataset(
                np.random.rand(10, 30, 15).astype(np.float32),
                np.zeros(5, dtype=np.int64),
            )

    def test_fall_adl_counts(self) -> None:
        """num_falls and num_adl should match label distribution."""
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
        seqs = np.random.rand(6, 30, 15).astype(np.float32)
        ds = FallDetectionDataset(seqs, labels)
        assert ds.num_falls == 3
        assert ds.num_adl == 3
