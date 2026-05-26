"""
LSTM model definition for temporal fall classification.

Implements a configurable LSTM network that processes sliding windows
of pose features and outputs fall/non-fall classification.

Reference: research/4.1 - 128 hidden units, 2 layers, dropout 0.3
is the empirically established optimal configuration.
Reference: research/8.2 - PyTorch for training, ONNX for deployment.
"""

import torch
import torch.nn as nn


class FallDetectionLSTM(nn.Module):
    """
    LSTM-based sequence classifier for fall detection.

    Architecture: LSTM layers -> final hidden state -> dropout -> linear -> output

    Parameters
    ----------
    input_size : int
        Number of features per frame (e.g., 15).
    hidden_size : int
        Number of hidden units in each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes (2 for binary fall/non-fall).
    dropout : float
        Dropout rate applied between LSTM layers and before the output.
    bidirectional : bool
        Whether to use bidirectional LSTM.
        DECISION: False by default per research/4.1 - bidirectional
        adds latency without significant accuracy gain on small datasets.
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # LSTM with inter-layer dropout (only if num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Output projection
        directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size * directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # lstm_out: (batch, seq_len, hidden_size * directions)
        # h_n: (num_layers * directions, batch, hidden_size)
        lstm_out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # Concatenate final hidden states from both directions
            # h_n[-2] is forward, h_n[-1] is backward
            hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            # Take the last layer's hidden state
            hidden = h_n[-1]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits
