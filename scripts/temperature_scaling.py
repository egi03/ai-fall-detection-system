"""
Temperature scaling calibration experiment.

Learns a single temperature parameter T on validation predictions to
reduce model overconfidence. Reports ECE before and after calibration.

Usage:
    python scripts/temperature_scaling.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.splitter import SubjectSplitter
from src.training.dataset import build_dataset_from_processed
from src.models.lstm import FallDetectionLSTM

URFD_SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
DATA_DIR = Path("data/data/processed/urfd")
MODEL_DIR = Path("models/run5_stride2_aug")
RESULTS_DIR = Path("results/temperature_scaling")
SEED = 42
WINDOW_SIZE = 30
TEST_STRIDE = 5
NUM_BINS = 10


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10) -> dict:
    """Compute Expected Calibration Error and per-bin statistics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_stats = []
    total_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = mask | (y_prob == bins[i + 1])

        n = mask.sum()
        if n == 0:
            bin_stats.append({
                "bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
                "count": 0, "avg_pred": 0, "avg_true": 0, "gap": 0
            })
            continue

        avg_pred = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        gap = abs(avg_pred - avg_true)
        ece += (n / total_samples) * gap

        bin_stats.append({
            "bin": f"[{bins[i]:.1f}, {bins[i+1]:.1f})",
            "count": int(n),
            "avg_pred": float(avg_pred),
            "avg_true": float(avg_true),
            "gap": float(gap),
        })

    return {"ece": float(ece), "bins": bin_stats}


def temperature_scale(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    return 1.0 / (1.0 + np.exp(-logits / temperature))


def find_optimal_temperature(val_logits: np.ndarray,
                              val_labels: np.ndarray) -> float:
    """Find optimal temperature via grid search on validation NLL."""
    best_t = 1.0
    best_nll = float("inf")

    for t in np.arange(0.1, 5.0, 0.01):
        probs = temperature_scale(val_logits, t)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        nll = -np.mean(
            val_labels * np.log(probs) +
            (1 - val_labels) * np.log(1 - probs)
        )
        if nll < best_nll:
            best_nll = nll
            best_t = t

    return best_t


def run_calibration():
    """Run temperature scaling across LOSO folds."""
    splitter = SubjectSplitter(URFD_SUBJECTS)
    folds = splitter.get_loso_folds()

    all_probs_before = []
    all_probs_after = []
    all_labels = []
    temperatures = []

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        n_train = max(1, int(len(train_subjects) * 0.8))
        train_actual = train_subjects[:n_train]
        val_subjects = train_subjects[n_train:]

        # Load model
        model_path = MODEL_DIR / f"fold_{fold_idx}" / "best_model.pth"
        if not model_path.exists():
            print(f"  Fold {fold_idx}: model not found at {model_path}, skipping")
            continue

        norm_mean = np.load(MODEL_DIR / f"fold_{fold_idx}" / "norm_mean.npy")
        norm_std = np.load(MODEL_DIR / f"fold_{fold_idx}" / "norm_std.npy")
        norm_stats = (norm_mean, norm_std)

        # Load validation data for temperature tuning
        val_seqs, val_labels_arr, _ = build_dataset_from_processed(
            processed_dir=str(DATA_DIR),
            subject_ids=val_subjects,
            window_size=WINDOW_SIZE,
            stride=TEST_STRIDE,
            positive_threshold=0.5,
            norm_stats=norm_stats,
        )

        # Load test data
        test_seqs, test_labels_arr, _ = build_dataset_from_processed(
            processed_dir=str(DATA_DIR),
            subject_ids=[test_subject],
            window_size=WINDOW_SIZE,
            stride=TEST_STRIDE,
            positive_threshold=0.5,
            norm_stats=norm_stats,
        )

        # Load model
        model = FallDetectionLSTM(
            input_size=15, hidden_size=128, num_layers=2,
            dropout=0.5, bidirectional=True
        )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        # Model outputs (N, 2) logits for [non-fall, fall] classes
        # We use softmax prob of class 1 (fall) as the score
        with torch.no_grad():
            val_tensor = torch.tensor(val_seqs, dtype=torch.float32)
            val_out = model(val_tensor).numpy()  # (N, 2)
            # Use log-odds of fall class as logits
            val_logits = val_out[:, 1] - val_out[:, 0]
            val_probs = 1.0 / (1.0 + np.exp(-val_logits))

            test_tensor = torch.tensor(test_seqs, dtype=torch.float32)
            test_out = model(test_tensor).numpy()  # (N, 2)
            test_logits = test_out[:, 1] - test_out[:, 0]
            test_probs = 1.0 / (1.0 + np.exp(-test_logits))

        # Find optimal temperature on validation
        opt_t = find_optimal_temperature(val_logits, val_labels_arr)
        temperatures.append(opt_t)

        # Apply to test
        test_probs_calibrated = temperature_scale(test_logits, opt_t)

        all_probs_before.extend(test_probs.tolist())
        all_probs_after.extend(test_probs_calibrated.tolist())
        all_labels.extend(test_labels_arr.tolist())

        print(f"  Fold {fold_idx} (test={test_subject}): T={opt_t:.2f}")

    all_probs_before = np.array(all_probs_before)
    all_probs_after = np.array(all_probs_after)
    all_labels = np.array(all_labels)

    # Compute ECE before and after
    ece_before = compute_ece(all_labels, all_probs_before, NUM_BINS)
    ece_after = compute_ece(all_labels, all_probs_after, NUM_BINS)

    # Check AUC is preserved
    auc_before = roc_auc_score(all_labels, all_probs_before)
    auc_after = roc_auc_score(all_labels, all_probs_after)

    print(f"\n{'='*60}")
    print(f"  TEMPERATURE SCALING RESULTS")
    print(f"{'='*60}")
    print(f"  Mean temperature: {np.mean(temperatures):.2f} ± {np.std(temperatures):.2f}")
    print(f"  ECE before: {ece_before['ece']:.4f}")
    print(f"  ECE after:  {ece_after['ece']:.4f}")
    print(f"  ECE reduction: {(1 - ece_after['ece']/ece_before['ece'])*100:.1f}%")
    print(f"  AUC before: {auc_before:.3f}")
    print(f"  AUC after:  {auc_after:.3f} (should be identical)")

    print(f"\n  Per-bin calibration (after):")
    for b in ece_after["bins"]:
        if b["count"] > 0:
            print(f"    {b['bin']}: n={b['count']:>3}, "
                  f"pred={b['avg_pred']:.3f}, true={b['avg_true']:.3f}, "
                  f"gap={b['gap']:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "temperatures": temperatures,
        "mean_temperature": float(np.mean(temperatures)),
        "ece_before": ece_before["ece"],
        "ece_after": ece_after["ece"],
        "ece_reduction_pct": float((1 - ece_after["ece"] / ece_before["ece"]) * 100),
        "auc_before": auc_before,
        "auc_after": auc_after,
        "bins_before": ece_before["bins"],
        "bins_after": ece_after["bins"],
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_calibration()
