"""Inference wrapper for the trained ST-GCN model.

Mirrors :class:`src.models.classifier.FallClassifier` but operates on
raw skeleton windows of shape ``(window_size, 13, 3)`` rather than the
15-D hand-crafted feature stream. Used by the demo when the
``--arch stgcn`` switch is set, and by evaluation scripts that want
identical-API predictors for the LSTM and ST-GCN baselines.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.stgcn import STGCN, STGCN_BODY_JOINTS, STGCN_NUM_JOINTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _append_motion(window_13: np.ndarray) -> np.ndarray:
    """Concatenate per-frame deltas to a ``(T, V, C)`` window.

    Mirrors :func:`src.training.dataset._append_motion_channels`. Used
    at inference time so a motion-trained ST-GCN sees the same 6-channel
    layout it was trained on.
    """
    diffs = np.zeros_like(window_13)
    diffs[1:] = window_13[1:] - window_13[:-1]
    return np.concatenate([window_13, diffs], axis=2)


def preprocess_mediapipe_window(window_33: np.ndarray) -> np.ndarray:
    """Convert a raw 33-joint MediaPipe window to ST-GCN input layout.

    Performs the same subset + person-centric normalization the training
    pipeline applies (see
    :func:`src.training.dataset.build_keypoint_dataset_from_processed`),
    so the live demo and the trained model agree on what an input window
    looks like.

    Parameters
    ----------
    window_33 : np.ndarray
        Shape ``(T, 33, 4)`` with columns ``(x, y, z, visibility)`` —
        either pixel coordinates or normalized coordinates; only relative
        geometry matters because of the hip-centering step.

    Returns
    -------
    np.ndarray
        Shape ``(T, 13, 3)`` ready to feed to :class:`STGCNClassifier`.
    """
    if window_33.ndim != 3 or window_33.shape[1] < 33:
        raise ValueError(
            f"Expected MediaPipe window of shape (T, 33, C); got {window_33.shape}"
        )

    # Subset to 13 joints and drop z
    subset = window_33[:, STGCN_BODY_JOINTS, :]
    xy = subset[:, :, :2].astype(np.float32, copy=True)
    vis_col = 3 if subset.shape[2] >= 4 else 2
    vis = subset[:, :, vis_col:vis_col + 1].astype(np.float32, copy=True)

    hip = (xy[:, 7, :] + xy[:, 8, :]) * 0.5
    shoulder = (xy[:, 1, :] + xy[:, 2, :]) * 0.5
    torso_len = np.linalg.norm(shoulder - hip, axis=1, keepdims=True)
    torso_len = np.where(torso_len < 1e-3, 1.0, torso_len)

    xy_norm = (xy - hip[:, None, :]) / torso_len[:, None, :]
    out = np.concatenate([xy_norm, vis], axis=2).astype(np.float32)
    return np.nan_to_num(out, nan=0.0)


class STGCNClassifier:
    """Real-time inference wrapper around a trained :class:`STGCN`.

    Parameters
    ----------
    model_path : Path
        Path to a ``.pth`` checkpoint produced by ``scripts/train_stgcn.py``.
    device : str
        Torch device. Defaults to CPU for parity with the demo loop.
    dropout : float
        Dropout rate. Overridden to 0 in eval mode but kept consistent
        with training for state-dict compatibility.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cpu",
        dropout: float = 0.2,
        in_channels: Optional[int] = None,
    ) -> None:
        self._device = torch.device(device)

        ckpt = torch.load(
            str(model_path), map_location=self._device, weights_only=False,
        )
        state = ckpt.get("model_state_dict", ckpt)

        # Auto-detect the trained input channel count from the data BN
        # buffer; STGCN uses BatchNorm1d(in_channels * num_joints).
        detected_in_channels = in_channels
        if detected_in_channels is None:
            bn_w = state.get("data_bn.weight")
            if bn_w is not None:
                detected_in_channels = int(bn_w.numel() // STGCN_NUM_JOINTS)
            else:
                detected_in_channels = 3
        self._in_channels = detected_in_channels
        self._motion = detected_in_channels >= 6

        self._model = STGCN(
            in_channels=detected_in_channels, num_classes=2, dropout=dropout,
        )
        self._model.load_state_dict(state)
        self._model.to(self._device).eval()
        logger.info(
            f"ST-GCN model loaded from {model_path} "
            f"(in_channels={detected_in_channels}, motion={self._motion})"
        )

    @property
    def model(self) -> torch.nn.Module:
        """Return the underlying ``nn.Module`` (for attribution etc.)."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Torch device the model lives on."""
        return self._device

    @property
    def in_channels(self) -> int:
        """Number of input channels the loaded checkpoint was trained on."""
        return self._in_channels

    @property
    def uses_motion(self) -> bool:
        """True when the loaded checkpoint expects motion-augmented input."""
        return self._motion

    def predict(self, window_13: np.ndarray) -> Tuple[int, float]:
        """Classify a pre-normalized ST-GCN window.

        Parameters
        ----------
        window_13 : np.ndarray
            Shape ``(T, 13, C)`` or ``(1, T, 13, C)`` with ``C`` matching
            :attr:`in_channels` of the loaded model.

        Returns
        -------
        tuple of (int, float)
            ``(predicted_class, P_fall)``.
        """
        if window_13.ndim == 3:
            window_13 = window_13[np.newaxis, :]
        if window_13.shape[2] != STGCN_NUM_JOINTS:
            raise ValueError(
                f"STGCNClassifier expects {STGCN_NUM_JOINTS} joints; "
                f"got window shape {window_13.shape}"
            )
        if window_13.shape[-1] != self._in_channels:
            raise ValueError(
                f"Window has {window_13.shape[-1]} channels but the trained "
                f"model expects {self._in_channels}."
            )

        tensor = torch.from_numpy(window_13.astype(np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        p_fall = float(probs[0, 1].item())
        return pred, p_fall

    def predict_from_mediapipe(self, window_33: np.ndarray) -> Tuple[int, float]:
        """Convenience: accept raw MediaPipe (33-joint) windows.

        Subsets to the 13 body joints, applies person-centric
        normalization, and (when :attr:`uses_motion` is True) appends
        per-frame deltas so the model sees the 6-channel layout it
        was trained on. Callers don't have to duplicate any of that.
        """
        window = preprocess_mediapipe_window(window_33)
        if self._motion:
            window = _append_motion(window)
        return self.predict(window)
