"""Spatial-Temporal Graph Convolutional Network for fall detection.

A faithful but compact ST-GCN baseline operating directly on the
skeleton graph rather than on hand-crafted features, so it can be
compared against the BiLSTM ensemble that the rest of the project
relies on.

References
----------
Yan, Xiong & Lin (2018). "Spatial Temporal Graph Convolutional Networks
for Skeleton-Based Action Recognition." AAAI 2018.
https://arxiv.org/abs/1801.07455

Graph topology
--------------
We use a 13-joint body subset extracted from the 33 MediaPipe BlazePose
landmarks. Face, hand and foot landmarks are dropped because fall
discrimination is dominated by torso / limb geometry. The mapping is
exposed as :data:`STGCN_BODY_JOINTS` so the data pipeline can subset
keypoints consistently.

Joint index (within the subset) and the MediaPipe landmark it comes from:

==  =================  =========================
ix  name               MediaPipeKeypoint value
==  =================  =========================
0   nose               NOSE                (0)
1   left_shoulder      LEFT_SHOULDER       (11)
2   right_shoulder     RIGHT_SHOULDER      (12)
3   left_elbow         LEFT_ELBOW          (13)
4   right_elbow        RIGHT_ELBOW         (14)
5   left_wrist         LEFT_WRIST          (15)
6   right_wrist        RIGHT_WRIST         (16)
7   left_hip           LEFT_HIP            (23)
8   right_hip          RIGHT_HIP           (24)
9   left_knee          LEFT_KNEE           (25)
10  right_knee         RIGHT_KNEE          (26)
11  left_ankle         LEFT_ANKLE          (27)
12  right_ankle        RIGHT_ANKLE         (28)
==  =================  =========================
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# MediaPipe landmark indices contributing each node of the body subset.
STGCN_BODY_JOINTS: List[int] = [
    0,   # nose
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]

STGCN_NUM_JOINTS: int = len(STGCN_BODY_JOINTS)

# Edges within the 13-joint body subset (index pairs into STGCN_BODY_JOINTS).
_STGCN_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2),                       # nose to shoulders
    (1, 2),                               # shoulder span
    (1, 3), (3, 5),                       # left arm
    (2, 4), (4, 6),                       # right arm
    (1, 7), (2, 8),                       # shoulders to hips
    (7, 8),                               # hip span
    (7, 9), (9, 11),                      # left leg
    (8, 10), (10, 12),                    # right leg
]


def build_normalized_adjacency() -> torch.Tensor:
    """Build the symmetrically normalized adjacency for the body subset.

    Computes :math:`\\tilde A = D^{-1/2} (A + I) D^{-1/2}` where :math:`A`
    is the binary adjacency of :data:`_STGCN_EDGES` (undirected) and
    :math:`D` is the degree matrix of :math:`A + I`. This is the
    standard "Kipf-Welling" normalization used inside each ST-GCN block.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(V, V)`` where ``V = STGCN_NUM_JOINTS``.
    """
    v = STGCN_NUM_JOINTS
    a = np.zeros((v, v), dtype=np.float32)
    for i, j in _STGCN_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    a = a + np.eye(v, dtype=np.float32)  # self-loops
    d = a.sum(axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    return torch.from_numpy(norm)


class STGCNBlock(nn.Module):
    """Single Spatial-Temporal Graph Conv block.

    A spatial GraphConv (one fixed normalized adjacency, à la Kipf-Welling)
    feeds a 1-D temporal convolution. Followed by batch norm, ReLU and an
    optional residual link.

    Parameters
    ----------
    in_channels : int
        Input feature channels per joint.
    out_channels : int
        Output channels.
    temporal_kernel_size : int
        Kernel size along the time axis. Must be odd; defaults to 9, which
        gives a receptive field of about 0.6 s at 15 FPS.
    temporal_stride : int
        Temporal stride for downsampling. ``1`` preserves the window length.
    dropout : float
        Dropout applied after the temporal conv.
    residual : bool
        If True, add an identity / projected shortcut from the input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 9,
        temporal_stride: int = 1,
        dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if temporal_kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be odd")

        # Spatial 1x1 conv: equivalent to a per-node linear projection,
        # but operating on the (N, C, T, V) tensor produced by the layout
        # used throughout this module.
        self.spatial = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        pad = (temporal_kernel_size - 1) // 2
        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(temporal_kernel_size, 1),
            stride=(temporal_stride, 1),
            padding=(pad, 0),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout)

        if not residual:
            self.residual = None
        elif in_channels == out_channels and temporal_stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=(temporal_stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, C_in, T, V)``.
        a_norm : torch.Tensor
            Normalized adjacency of shape ``(V, V)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(N, C_out, T_out, V)``.
        """
        res = self.residual(x) if self.residual is not None else 0.0

        # Spatial GCN: aggregate neighbours via einsum over the V dim,
        # then 1x1 conv projects to out_channels.
        x = torch.einsum("nctv,vw->nctw", x, a_norm)
        x = self.spatial(x)
        x = F.relu(self.bn1(x), inplace=True)

        # Temporal convolution
        x = self.temporal(x)
        x = self.bn2(x)
        x = self.dropout(x)

        x = x + res
        return F.relu(x, inplace=True)


class STGCN(nn.Module):
    """Compact ST-GCN classifier for skeleton-based fall detection.

    Three ST-GCN blocks with hidden widths [16, 32, 32], temporal kernel
    9, residual shortcuts, batch norm, dropout. Final classifier head
    is a global average pool over (T, V) followed by a linear layer.

    Parameters
    ----------
    in_channels : int
        Input feature channels per joint. Typically 3 — (x, y, visibility)
        after person-centric normalization.
    num_classes : int
        Output classes. 2 for fall vs non-fall.
    num_joints : int
        Number of vertices in the skeleton graph. Defaults to
        :data:`STGCN_NUM_JOINTS`.
    hidden_channels : tuple of int
        Channel widths of the three ST-GCN blocks.
    dropout : float
        Dropout applied inside each ST-GCN block and on the classifier head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        num_joints: int = STGCN_NUM_JOINTS,
        hidden_channels: Tuple[int, int, int] = (16, 32, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_joints != STGCN_NUM_JOINTS:
            raise ValueError(
                f"num_joints={num_joints} but STGCN_NUM_JOINTS="
                f"{STGCN_NUM_JOINTS}; adjacency is hard-coded for the body subset."
            )

        c1, c2, c3 = hidden_channels

        # Per-channel BN over the raw input keeps each coord scale-stable.
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.block1 = STGCNBlock(in_channels, c1, dropout=dropout, residual=False)
        self.block2 = STGCNBlock(c1, c2, dropout=dropout, residual=True)
        self.block3 = STGCNBlock(c2, c3, dropout=dropout, residual=True)

        self.classifier_dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(c3, num_classes)

        # Pre-compute and register the normalized adjacency once.
        self.register_buffer("a_norm", build_normalized_adjacency())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(N, T, V, C)``. The dataset loader emits
            sequences in this layout; we permute internally to
            ``(N, C, T, V)``.

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(N, num_classes)``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"STGCN expected 4-D input (N, T, V, C); got shape {tuple(x.shape)}"
            )

        n, t, v, c = x.shape
        # (N, T, V, C) -> (N, C, T, V) -> (N, C*V, T) for global BN -> back.
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)
        x_bn = x.permute(0, 1, 3, 2).contiguous().view(n, c * v, t)
        x_bn = self.data_bn(x_bn)
        x = x_bn.view(n, c, v, t).permute(0, 1, 3, 2).contiguous()  # back to (N,C,T,V)

        x = self.block1(x, self.a_norm)
        x = self.block2(x, self.a_norm)
        x = self.block3(x, self.a_norm)

        # Global average pool over (T, V)
        x = x.mean(dim=(2, 3))  # (N, C3)
        x = self.classifier_dropout(x)
        return self.fc(x)
