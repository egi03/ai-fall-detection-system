"""
Temporal Convolutional Network (TCN) baseline under LOSO on URFD.

A simple 1D-CNN with causal convolutions for fall detection on the same
(30, 15) sliding window features used by BiLSTM/Transformer-LSTM.

Architecture:
  - 3 causal conv1d blocks (channels: 15→64→64→64, kernel=3, dilation=1,2,4)
  - Each block: Conv1d → BatchNorm → ReLU → Dropout
  - Global average pooling → FC → Sigmoid

Usage:
    python scripts/tcn_baseline.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.splitter import SubjectSplitter
from src.training.dataset import build_dataset_from_processed

URFD_SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
DATA_DIR = Path("data/data/processed/urfd")
RESULTS_DIR = Path("results/tcn_baseline")
SEED = 42
DEVICE = "cpu"

# Hyperparameters
WINDOW_SIZE = 30
NUM_FEATURES = 15
TRAIN_STRIDE = 2
TEST_STRIDE = 5
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_BLOCKS = 3
DROPOUT = 0.5
LR = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 30
BATCH_SIZE = 32


class CausalConv1dBlock(nn.Module):
    """Single causal convolution block with padding, BN, ReLU, dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Residual connection if dimensions match
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal padding on left side
        out = nn.functional.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TCN(nn.Module):
    """Simple Temporal Convolutional Network for fall detection."""

    def __init__(self, in_features: int = 15, hidden: int = 64,
                 kernel: int = 3, num_blocks: int = 3, dropout: float = 0.5):
        super().__init__()
        channels = [in_features] + [hidden] * num_blocks
        dilations = [2 ** i for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            CausalConv1dBlock(channels[i], channels[i + 1], kernel,
                              dilations[i], dropout)
            for i in range(num_blocks)
        ])
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        # Global average pooling over time
        x = x.mean(dim=2)  # (batch, hidden)
        return torch.sigmoid(self.fc(x)).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "sensitivity": sens, "specificity": spec, "f1": f1}


def train_fold(train_seqs, train_labels, val_seqs, val_labels,
               test_seqs, test_labels, fold_idx):
    """Train TCN on one LOSO fold."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Compute class weights
    n_pos = train_labels.sum()
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    # Create datasets
    train_ds = TensorDataset(
        torch.tensor(train_seqs, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_seqs, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(test_seqs, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TCN(NUM_FEATURES, HIDDEN_CHANNELS, KERNEL_SIZE, NUM_BLOCKS, DROPOUT)
    if fold_idx == 0:
        print(f"  TCN parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.BCELoss(weight=None)  # Use manual weighting

    best_val_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = model(X)
            # Manual class weighting
            weights = torch.where(y == 1, pos_weight, torch.ones(1))
            loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_probs = []
        val_true = []
        with torch.no_grad():
            for X, y in val_loader:
                val_probs.extend(model(X).numpy())
                val_true.extend(y.numpy())

        try:
            val_auc = roc_auc_score(val_true, val_probs)
        except ValueError:
            val_auc = 0.0

        scheduler.step(-val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Load best model and evaluate on test
    model.load_state_dict(best_state)
    model.eval()
    test_probs = []
    test_true = []
    with torch.no_grad():
        for X, y in test_loader:
            test_probs.extend(model(X).numpy())
            test_true.extend(y.numpy())

    return np.array(test_probs), np.array(test_true)


def run_loso():
    """Run TCN LOSO cross-validation."""
    splitter = SubjectSplitter(URFD_SUBJECTS)
    folds = splitter.get_loso_folds()

    all_probs = []
    all_labels = []
    fold_metrics = []

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        n_train = max(1, int(len(train_subjects) * 0.8))
        train_actual = train_subjects[:n_train]
        val_subjects = train_subjects[n_train:]

        train_seqs, train_labels, norm_stats = build_dataset_from_processed(
            processed_dir=str(DATA_DIR),
            subject_ids=train_actual,
            window_size=WINDOW_SIZE,
            stride=TRAIN_STRIDE,
            positive_threshold=0.5,
        )

        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=str(DATA_DIR),
            subject_ids=val_subjects,
            window_size=WINDOW_SIZE,
            stride=TRAIN_STRIDE,
            positive_threshold=0.5,
            norm_stats=norm_stats,
        )

        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=str(DATA_DIR),
            subject_ids=[test_subject],
            window_size=WINDOW_SIZE,
            stride=TEST_STRIDE,
            positive_threshold=0.5,
            norm_stats=norm_stats,
        )

        print(f"\n  Fold {fold_idx} (test={test_subject}): "
              f"train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

        probs, labels = train_fold(
            train_seqs, train_labels, val_seqs, val_labels,
            test_seqs, test_labels, fold_idx
        )

        metrics = compute_metrics(labels, probs)
        fold_metrics.append(metrics)
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        print(f"    AUC={metrics['auc']:.3f}  "
              f"Sens={metrics['sensitivity']:.3f}  "
              f"Spec={metrics['specificity']:.3f}")

    # Pooled metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    pooled = compute_metrics(all_labels, all_probs)

    mean_auc = np.mean([m["auc"] for m in fold_metrics])
    std_auc = np.std([m["auc"] for m in fold_metrics])

    print(f"\n{'='*60}")
    print(f"  TCN RESULTS")
    print(f"{'='*60}")
    print(f"  Pooled AUC:  {pooled['auc']:.3f}")
    print(f"  Pooled Sens: {pooled['sensitivity']:.3f}")
    print(f"  Pooled Spec: {pooled['specificity']:.3f}")
    print(f"  Pooled F1:   {pooled['f1']:.3f}")
    print(f"  Mean fold AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"\n  For reference:")
    print(f"  BiLSTM:        AUC=0.888  Sens=0.806  Spec=0.842")
    print(f"  Transf.-LSTM:  AUC=0.877  Sens=0.768  Spec=0.838")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "TCN",
        "pooled_auc": pooled["auc"],
        "pooled_sensitivity": pooled["sensitivity"],
        "pooled_specificity": pooled["specificity"],
        "pooled_f1": pooled["f1"],
        "mean_fold_auc": mean_auc,
        "std_fold_auc": std_auc,
        "per_fold": fold_metrics,
        "params": "computed at runtime",
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    results = run_loso()
