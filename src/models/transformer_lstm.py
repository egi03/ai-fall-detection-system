"""
Transformer-LSTM hybrid model for temporal fall classification.

Combines a Transformer Encoder (global self-attention over the temporal
window) with an LSTM decoder (sequential causal processing) for
fall detection from pose feature sequences.

The Transformer Encoder identifies which frames in the sliding window
are most informative (e.g., the impact moment), while the LSTM
captures sequential fall dynamics and progression.

Reference: research/4.1 - Hybrid Transformer-LSTM is the recommended
architecture, combining global attention with chronological causality.
Hyperparameters follow research/4.1 bounds: d_model 32-128, nhead 4-8,
encoder layers 1-4.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequence ordering.

    Injects temporal position information into the feature embeddings
    so the Transformer can distinguish frame order despite its
    permutation-invariant self-attention mechanism.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length supported.
    dropout : float
        Dropout applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Positionally-encoded tensor of same shape.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FallDetectionTransformerLSTM(nn.Module):
    """
    Hybrid Transformer-LSTM classifier for fall detection.

    Architecture:
        Input (B, T, F) -> Linear projection (F -> d_model)
        -> Positional Encoding -> Transformer Encoder (self-attention)
        -> LSTM (sequential processing) -> Dropout -> FC -> Logits

    The Transformer Encoder provides global temporal attention,
    allowing the model to identify critical frames (e.g., impact moment)
    across the entire window. The LSTM decoder processes the attended
    representations sequentially, capturing causal fall dynamics.

    Parameters
    ----------
    input_size : int
        Number of features per frame (e.g., 15).
    d_model : int
        Transformer embedding dimension.
        DECISION: 64 per research/4.1 bounds (32-128); conservative
        for small datasets to avoid overfitting.
    nhead : int
        Number of attention heads in the Transformer Encoder.
    num_encoder_layers : int
        Number of Transformer Encoder layers.
    lstm_hidden_size : int
        Hidden size for the LSTM decoder.
    lstm_num_layers : int
        Number of stacked LSTM layers.
    num_classes : int
        Number of output classes (2 for binary fall/non-fall).
    dropout : float
        Dropout rate applied throughout the model.
    """

    def __init__(
        self,
        input_size: int = 15,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        lstm_hidden_size: int = 64,
        lstm_num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes

        # Linear projection: feature space -> transformer embedding space
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding for temporal ordering
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, dropout=dropout,
        )

        # Transformer Encoder: global self-attention over temporal window
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        # LSTM decoder: sequential processing of attended representations
        # DECISION: Unidirectional per research/4.1 — the Transformer
        # already provides global context; bidirectional LSTM would add
        # redundant computation and latency.
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Classification head
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer-LSTM hybrid.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_size).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # Project features to transformer embedding dimension
        # (B, T, input_size) -> (B, T, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer Encoder: self-attention over temporal window
        # (B, T, d_model) -> (B, T, d_model)
        x = self.transformer_encoder(x)

        # LSTM decoder: sequential processing
        # (B, T, d_model) -> lstm_out: (B, T, lstm_hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Take final hidden state from last LSTM layer
        hidden = h_n[-1]  # (B, lstm_hidden_size)

        # Classification
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)  # (B, num_classes)

        return logits
