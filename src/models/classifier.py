"""
Inference wrapper for the trained fall detection model.

Loads a trained PyTorch model and provides a simple predict() interface
for real-time classification of feature sequences.  Supports both the
BiLSTM architecture and the Transformer-LSTM hybrid.

Reference: research/8.1 - PyTorch for inference; ONNX export available.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.models.lstm import FallDetectionLSTM
from src.models.transformer_lstm import FallDetectionTransformerLSTM
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Architecture name constants
ARCH_LSTM = "lstm"
ARCH_TRANSFORMER_LSTM = "transformer_lstm"


class FallClassifier:
    """
    Wraps the trained model for real-time inference.

    Supports two architectures:
      - "lstm": FallDetectionLSTM (BiLSTM)
      - "transformer_lstm": FallDetectionTransformerLSTM (Transformer-LSTM hybrid)

    Parameters
    ----------
    model_path : Path
        Path to the trained model file (.pth checkpoint).
    architecture : str
        Model architecture: "lstm" or "transformer_lstm".
    input_size : int
        Number of features per frame.
    hidden_size : int
        LSTM hidden size (must match trained model).
    num_layers : int
        LSTM layer count (must match trained model).
    num_classes : int
        Number of output classes.
    bidirectional : bool
        Whether the LSTM model is bidirectional (only for "lstm" arch).
    d_model : int
        Transformer embedding dimension (only for "transformer_lstm").
    nhead : int
        Number of attention heads (only for "transformer_lstm").
    num_encoder_layers : int
        Number of Transformer Encoder layers (only for "transformer_lstm").
    dim_feedforward : int
        Transformer feedforward dimension (only for "transformer_lstm").
    lstm_hidden_size : int
        LSTM hidden size in the hybrid (only for "transformer_lstm").
    lstm_num_layers : int
        LSTM layers in the hybrid (only for "transformer_lstm").
    device : str
        Inference device: 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model_path: Path,
        architecture: str = ARCH_LSTM,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        bidirectional: bool = False,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        device: str = "cpu",
    ) -> None:
        self._device = torch.device(device)
        self._architecture = architecture

        if architecture == ARCH_TRANSFORMER_LSTM:
            self._model = FallDetectionTransformerLSTM(
                input_size=input_size,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                num_classes=num_classes,
                dropout=0.0,  # No dropout at inference
            )
        else:
            self._model = FallDetectionLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=0.0,  # No dropout at inference
                bidirectional=bidirectional,
            )

        model_path = Path(model_path)
        checkpoint = torch.load(
            str(model_path), map_location=self._device, weights_only=False
        )

        # Support both raw state_dict and checkpoint dict
        if "model_state_dict" in checkpoint:
            self._model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.to(self._device)
        self._model.eval()
        logger.info(f"Model loaded from {model_path} (arch={architecture})")

    @property
    def model(self) -> torch.nn.Module:
        """Return the underlying ``nn.Module`` (for attribution / IG)."""
        return self._model

    @property
    def device(self) -> torch.device:
        """Return the torch device the model lives on."""
        return self._device

    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Classify a feature sequence as fall or non-fall.

        Parameters
        ----------
        sequence : np.ndarray
            Feature array of shape (1, window_size, num_features)
            or (window_size, num_features).

        Returns
        -------
        tuple of (int, float)
            (predicted_class, fall_probability).
            class 0 = non-fall, class 1 = fall.
            fall_probability is always P(fall), regardless of predicted class.
        """
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :]

        tensor = torch.from_numpy(sequence.astype(np.float32)).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)

        pred_class = int(probs.argmax(dim=1).item())
        fall_probability = float(probs[0, 1].item())

        return pred_class, fall_probability

    def predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Get class probabilities for a feature sequence.

        Parameters
        ----------
        sequence : np.ndarray
            Feature array of shape (1, window_size, num_features)
            or (window_size, num_features).

        Returns
        -------
        np.ndarray
            Probabilities of shape (num_classes,).
        """
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :]

        tensor = torch.from_numpy(sequence.astype(np.float32)).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)

        return probs[0].cpu().numpy()

    def export_onnx(
        self, output_path: Path, input_shape: Tuple[int, ...],
    ) -> None:
        """
        Export the PyTorch model to ONNX format.

        Parameters
        ----------
        output_path : Path
            Destination path for the .onnx file.
        input_shape : tuple
            Expected input tensor shape (batch, seq_len, features).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_shape).to(self._device)

        torch.onnx.export(
            self._model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
        )
        logger.info(f"ONNX model exported to {output_path}")
