"""
Sliding window management for temporal feature sequences.

Uses collections.deque for O(1) append/pop operations on the
real-time feature buffer. Converts to NumPy array only at
inference time to satisfy ONNX Runtime tensor requirements.

DECISION: Using deque(maxlen=W) per research/8.1. This gives O(1)
push operations and automatic oldest-frame discard, which is optimal
for real-time streaming.

Reference: research/8.1 - deque(maxlen=W) for sliding window,
cast to numpy only at inference time.
Reference: research/3.2 - Window size 30-90 frames, stride 3-15.
"""

from collections import deque
from typing import Optional

import numpy as np


class SlidingWindow:
    """
    Manages a fixed-size sliding window of feature vectors.

    Uses a deque with maxlen for automatic oldest-frame discard.
    The window content is only materialized as a contiguous NumPy
    array when get_tensor() is called, minimizing memory copies
    during real-time operation.

    Parameters
    ----------
    window_size : int
        Number of frames in the window.
    num_features : int
        Dimensionality of each frame's feature vector.
    """

    def __init__(self, window_size: int = 30, num_features: int = 15) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}")

        self._window_size = window_size
        self._num_features = num_features
        self._buffer: deque = deque(maxlen=window_size)

    @property
    def window_size(self) -> int:
        """Return the configured window size."""
        return self._window_size

    @property
    def num_features(self) -> int:
        """Return the expected feature dimensionality."""
        return self._num_features

    @property
    def current_size(self) -> int:
        """Return the current number of frames in the buffer."""
        return len(self._buffer)

    def push(self, feature_vector: np.ndarray) -> None:
        """
        Append a new feature vector to the window.

        Oldest frame is automatically discarded when window is full
        (handled by deque maxlen).

        Parameters
        ----------
        feature_vector : np.ndarray
            Feature vector of shape (num_features,).

        Raises
        ------
        ValueError
            If feature vector has incorrect dimensionality.
        """
        if feature_vector.shape != (self._num_features,):
            raise ValueError(
                f"Expected feature vector of shape ({self._num_features},), "
                f"got {feature_vector.shape}"
            )
        self._buffer.append(feature_vector.copy())

    @property
    def is_ready(self) -> bool:
        """
        Check if the window is fully populated.

        Returns
        -------
        bool
            True if the window contains exactly window_size frames.
        """
        return len(self._buffer) == self._window_size

    def get_tensor(self) -> Optional[np.ndarray]:
        """
        Convert the current window to a contiguous NumPy array.

        Only called at inference time. The batch dimension is added
        to satisfy model input requirements.

        Returns
        -------
        np.ndarray or None
            Array of shape (1, window_size, num_features) with batch dim.
            Returns None if window is not ready.
        """
        if not self.is_ready:
            return None

        # Stack deque contents into contiguous array
        tensor = np.array(list(self._buffer), dtype=np.float32)
        # Add batch dimension: (window_size, num_features) -> (1, window_size, num_features)
        return tensor[np.newaxis, :, :]

    def get_array(self) -> Optional[np.ndarray]:
        """
        Get the window content without batch dimension.

        Returns
        -------
        np.ndarray or None
            Array of shape (window_size, num_features).
            Returns None if window is not ready.
        """
        if not self.is_ready:
            return None
        return np.array(list(self._buffer), dtype=np.float32)

    def reset(self) -> None:
        """Clear the window buffer."""
        self._buffer.clear()
