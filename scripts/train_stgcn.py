"""Train the ST-GCN baseline on URFD with LOSO cross-validation.

This is the skeleton-graph counterpart to ``scripts/train.py``. Instead
of the 15-D hand-crafted feature window the ST-GCN consumes the raw
13-joint MediaPipe body subset directly. Adjacency and architecture
are defined in :mod:`src.models.stgcn`.

Per-fold predictions are concatenated to produce LOSO-level metrics
that are directly comparable with the BiLSTM ensemble already on disk.

Usage
-----
::

    python scripts/train_stgcn.py --epochs 60 --batch-size 32

Outputs land under ``results/stgcn_loso/`` and ``models/stgcn_loso/``.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.splitter import SubjectSplitter
from src.models.stgcn import STGCN
from src.training.dataset import (
    LABEL_FALL,
    STGCN_LR_FLIP_PAIRS,
    build_keypoint_dataset_from_processed,
)
from src.training.evaluate import Evaluator
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


class KeypointWindowDataset(Dataset):
    """Lightweight Dataset wrapping ``(N, T, V, C)`` arrays + labels.

    Supports two training-time augmentations applied per sample when
    ``augment=True``:

    * **Horizontal flip** (p=0.5): negate the x coordinate (channel 0)
      and swap left/right joint indices via
      :data:`src.training.dataset.STGCN_LR_FLIP_PAIRS`. When motion
      channels are present (``has_motion=True``, channels 3-5 are
      ``dx, dy, dvis``), ``dx`` is also negated.
    * **Gaussian coordinate noise** (std=0.02 in normalized torso units)
      on the xy channels (and on dx/dy if motion is enabled).

    Parameters
    ----------
    sequences : np.ndarray
        Float32 array of shape ``(N, T, V, C)``.
    labels : np.ndarray
        Int64 labels of shape ``(N,)``.
    augment : bool
        Enable training-time augmentation. Always False for val/test.
    has_motion : bool
        Whether the trailing 3 channels are motion deltas. Required when
        ``augment=True`` so flip correctly negates ``dx``.
    noise_std : float
        Std-dev of Gaussian noise applied to position channels.
    flip_prob : float
        Probability of applying a horizontal flip per sample.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        has_motion: bool = False,
        noise_std: float = 0.02,
        flip_prob: float = 0.5,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels length mismatch")
        self._sequences = sequences.astype(np.float32)
        self._labels = labels.astype(np.int64)
        self._augment = augment
        self._has_motion = has_motion
        self._noise_std = noise_std
        self._flip_prob = flip_prob
        self._rng = np.random.default_rng(seed=42)

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        seq = self._sequences[idx]
        if self._augment:
            seq = self._apply_augmentation(seq)
        return (
            torch.from_numpy(seq.astype(np.float32, copy=False)),
            torch.tensor(self._labels[idx], dtype=torch.long),
        )

    def _apply_augmentation(self, seq: np.ndarray) -> np.ndarray:
        """Return an augmented copy of ``seq`` of shape ``(T, V, C)``."""
        seq = seq.copy()

        # Horizontal flip
        if self._rng.random() < self._flip_prob:
            seq[..., 0] = -seq[..., 0]                   # negate x
            if self._has_motion and seq.shape[-1] >= 6:
                seq[..., 3] = -seq[..., 3]               # negate dx
            for left, right in STGCN_LR_FLIP_PAIRS:
                tmp = seq[:, left, :].copy()
                seq[:, left, :] = seq[:, right, :]
                seq[:, right, :] = tmp

        # Gaussian noise on position (and motion if present)
        if self._noise_std > 0:
            noise = self._rng.normal(
                0.0, self._noise_std, size=(seq.shape[0], seq.shape[1], 2),
            ).astype(np.float32)
            seq[..., 0:2] += noise
            if self._has_motion and seq.shape[-1] >= 5:
                motion_noise = self._rng.normal(
                    0.0, self._noise_std, size=(seq.shape[0], seq.shape[1], 2),
                ).astype(np.float32)
                seq[..., 3:5] += motion_noise

        return seq

    @property
    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for balanced loss."""
        counts = np.bincount(self._labels, minlength=2).astype(np.float64)
        total = counts.sum()
        return torch.tensor(total / (2.0 * counts), dtype=torch.float32)


def _collect_probs(
    model: torch.nn.Module, loader: DataLoader, device: torch.device,
) -> tuple:
    """Run inference and return ``(y_true, y_prob_fall, y_pred)``."""
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for seq, lbl in loader:
            seq = seq.to(device)
            logits = model(seq)
            p = F.softmax(logits, dim=1)[:, LABEL_FALL].cpu().numpy()
            probs.append(p)
            trues.append(lbl.numpy())
    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return y_true, y_prob, y_pred


def train_fold(
    processed_dir: Path,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: List[str],
    fold_dir: Path,
    window_size: int,
    stride: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    device: str,
    motion: bool = False,
    augment: bool = False,
) -> Dict:
    """Train + evaluate a single LOSO fold; returns per-fold dict."""

    train_seqs, train_lbls = build_keypoint_dataset_from_processed(
        processed_dir, train_subjects,
        window_size=window_size, stride=stride, motion=motion,
    )
    val_seqs, val_lbls = build_keypoint_dataset_from_processed(
        processed_dir, val_subjects,
        window_size=window_size, stride=stride, motion=motion,
    )
    test_seqs, test_lbls = build_keypoint_dataset_from_processed(
        processed_dir, test_subjects,
        window_size=window_size, stride=stride, motion=motion,
    )

    train_ds = KeypointWindowDataset(
        train_seqs, train_lbls, augment=augment, has_motion=motion,
    )
    val_ds = KeypointWindowDataset(val_seqs, val_lbls)
    test_ds = KeypointWindowDataset(test_seqs, test_lbls)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    in_channels = train_seqs.shape[-1]
    model = STGCN(in_channels=in_channels, num_classes=2, dropout=dropout)
    trainer = Trainer(
        model,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        epochs=epochs,
        early_stopping_patience=10,
        class_weights=train_ds.class_weights,
        device=device,
    )
    logger.info(
        f"Fold dir={fold_dir.name} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    t0 = time.monotonic()
    history = trainer.train(train_loader, val_loader)
    elapsed = time.monotonic() - t0

    fold_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(fold_dir / "best_model.pth")

    # Test-set predictions
    y_true, y_prob, y_pred = _collect_probs(
        trainer.model, test_loader, torch.device(device),
    )
    np.save(fold_dir / "y_true.npy", y_true)
    np.save(fold_dir / "y_prob.npy", y_prob)

    evaluator = Evaluator(output_dir=fold_dir)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)
    metrics["epochs_trained"] = len(history["train_loss"])
    metrics["train_seconds"] = round(elapsed, 1)
    metrics["test_subjects"] = test_subjects

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"Fold {test_subjects} sens={metrics['sensitivity']:.3f} "
        f"spec={metrics['specificity']:.3f} AUC={metrics['auc_roc']:.3f}"
    )
    return metrics


def main() -> None:
    """Run LOSO ST-GCN training on URFD and aggregate metrics."""
    parser = argparse.ArgumentParser(description="Train ST-GCN fall detector (LOSO)")
    parser.add_argument("--dataset", default="urfd")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--run-name", type=str, default="stgcn_loso")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--folds", type=int, default=0,
        help="If > 0, limit to the first N folds (smoke test).",
    )
    parser.add_argument(
        "--motion", action="store_true",
        help="Augment input with per-frame deltas (dx, dy, dvis) — 6 channels.",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable training-time horizontal flip + Gaussian coord noise.",
    )
    args = parser.parse_args()

    processed_dir = Path("data/data/processed") / args.dataset
    if not processed_dir.exists():
        logger.error(f"Processed data not found: {processed_dir}")
        sys.exit(1)

    output_root = Path("models") / args.run_name
    results_root = Path("results") / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        (processed_dir / "metadata.json").read_text(encoding="utf-8")
    )
    subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()
    if args.folds > 0:
        folds = folds[: args.folds]

    all_metrics = []
    all_y_true, all_y_prob = [], []

    for fold_idx, fold in enumerate(folds):
        test_subjects = fold["test"]
        train_pool = fold["train"]
        n_val = max(1, len(train_pool) // 5)
        val_subjects = train_pool[-n_val:]
        train_subjects = train_pool[:-n_val] or train_pool

        fold_dir = output_root / f"fold_{fold_idx}"
        metrics = train_fold(
            processed_dir=processed_dir,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
            fold_dir=fold_dir,
            window_size=args.window_size,
            stride=args.stride,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            device=args.device,
            motion=args.motion,
            augment=args.augment,
        )
        all_metrics.append(metrics)

        y_true = np.load(fold_dir / "y_true.npy")
        y_prob = np.load(fold_dir / "y_prob.npy")
        all_y_true.append(y_true)
        all_y_prob.append(y_prob)

    # Aggregate across folds
    y_true_cat = np.concatenate(all_y_true)
    y_prob_cat = np.concatenate(all_y_prob)
    y_pred_cat = (y_prob_cat >= 0.5).astype(np.int64)

    evaluator = Evaluator(output_dir=results_root)
    agg = evaluator.compute_metrics(y_true_cat, y_pred_cat, y_prob_cat)
    evaluator.plot_confusion_matrix(
        y_true_cat, y_pred_cat, results_root / "confusion_matrix.png",
    )
    evaluator.plot_roc_curve(
        y_true_cat, y_prob_cat, results_root / "roc_curve.png",
    )
    np.save(results_root / "all_y_true.npy", y_true_cat)
    np.save(results_root / "all_y_prob.npy", y_prob_cat)

    summary = {
        "aggregate": agg,
        "folds": all_metrics,
        "config": {
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "window_size": args.window_size,
            "stride": args.stride,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "motion": args.motion,
            "augment": args.augment,
        },
    }
    with open(results_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info(
        f"LOSO aggregate: sens={agg['sensitivity']:.3f} "
        f"spec={agg['specificity']:.3f} AUC={agg['auc_roc']:.3f}"
    )
    logger.info(f"Results saved to {results_root}")


if __name__ == "__main__":
    main()
