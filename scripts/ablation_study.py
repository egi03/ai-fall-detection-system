"""
Feature ablation study for fall detection.

Systematically removes each feature group and retrains full 5-fold LOSO
to measure the AUC impact. Uses Run 5 config (stride=2, augment) as baseline.

Feature groups (15 total features):
  Angles:     [0] torso_inclination, [1] hip_shoulder_angle,
              [7] left_knee_angle, [8] right_knee_angle,
              [9] left_hip_angle, [10] right_hip_angle
  Distances:  [5] head_to_toe_distance, [6] shoulder_ankle_distance,
              [11] wrist_hip_distance, [12] body_spread
  BBox:       [2] bbox_aspect_ratio
  Dynamics:   [3] com_velocity, [4] com_acceleration
  Visibility: [13] mean_visibility, [14] min_core_visibility

Ablation protocol per research/3.2:
  1. Baseline: all 15 features (should reproduce Run 5 AUC ~0.888)
  2. Remove one group at a time, retrain LOSO, compare AUC
  3. Individual removal for top-impact features

Reference: research/3.2, research/8.3.

Usage:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --skip-baseline   # if baseline already exists
    python scripts/ablation_study.py --groups-only      # skip individual ablations
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data_processing.preprocessor import KeypointAugmenter
from src.data_processing.splitter import SubjectSplitter
from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import FallDetectionDataset, build_dataset_from_processed
from src.training.evaluate import Evaluator
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Feature definitions ──────────────────────────────────────────────────────

FEATURE_NAMES = [
    "torso_inclination",      # 0
    "hip_shoulder_angle",     # 1
    "bbox_aspect_ratio",      # 2
    "com_velocity",           # 3
    "com_acceleration",       # 4
    "head_to_toe_distance",   # 5
    "shoulder_ankle_distance",# 6
    "left_knee_angle",        # 7
    "right_knee_angle",       # 8
    "left_hip_angle",         # 9
    "right_hip_angle",        # 10
    "wrist_hip_distance",     # 11
    "body_spread",            # 12
    "mean_visibility",        # 13
    "min_core_visibility",    # 14
]

# Feature groups for group-level ablation
FEATURE_GROUPS: Dict[str, List[int]] = {
    "angles": [0, 1, 7, 8, 9, 10],
    "distances": [5, 6, 11, 12],
    "bbox": [2],
    "dynamics": [3, 4],
    "visibility": [13, 14],
}

# Individual features to ablate (all 15 for completeness)
INDIVIDUAL_FEATURES = list(range(15))


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_features(
    sequences: np.ndarray, exclude_indices: List[int]
) -> np.ndarray:
    """
    Remove specified feature columns from windowed sequences.

    Parameters
    ----------
    sequences : np.ndarray
        Shape (N, window_size, num_features).
    exclude_indices : list of int
        Feature indices to remove.

    Returns
    -------
    np.ndarray
        Shape (N, window_size, num_features - len(exclude_indices)).
    """
    keep = [i for i in range(sequences.shape[2]) if i not in exclude_indices]
    return sequences[:, :, keep]


def run_ablation_loso(
    config: dict,
    exclude_indices: List[int],
    ablation_name: str,
    results_dir: Path,
) -> Dict:
    """
    Run 5-fold LOSO with specified features excluded.

    Uses Run 5 config: stride=2, augment, BiLSTM h=128, dropout=0.5.

    Returns
    -------
    dict
        Aggregated metrics including AUC, sensitivity, specificity, F1.
    """
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    hidden_size = cfg_model["hidden_size"]
    dropout = cfg_model["dropout"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)
    stride = 2  # Run 5 config
    window_size = cfg_feat["window_size"]

    num_input_features = FeatureExtractor.NUM_FEATURES - len(exclude_indices)

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        urfd_meta = json.load(f)
    urfd_subjects = sorted(set(s["subject_id"] for s in urfd_meta["sequences"]))

    splitter = SubjectSplitter(urfd_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    augmenter = KeypointAugmenter(
        flip_probability=0.0,
        noise_std=0.05,
        scale_range=(0.85, 1.15),
        seed=SEED,
    )

    all_preds, all_labels, all_probs = [], [], []
    fold_aucs = []

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]
        n_val = max(1, len(train_subjects) // 4)
        val_subjects = train_subjects[-n_val:]
        train_subjects_actual = train_subjects[:-n_val] or train_subjects

        # Build full-feature datasets
        train_seqs, train_labels, norm_stats = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=train_subjects_actual,
            window_size=window_size,
            stride=stride,
        )
        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=val_subjects,
            window_size=window_size,
            stride=stride,
            norm_stats=norm_stats,
        )
        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=[test_subject],
            window_size=window_size,
            stride=5,  # consistent test stride
            norm_stats=norm_stats,
        )

        # Remove excluded features AFTER normalization
        if exclude_indices:
            train_seqs = _select_features(train_seqs, exclude_indices)
            val_seqs = _select_features(val_seqs, exclude_indices)
            test_seqs = _select_features(test_seqs, exclude_indices)

        train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=cfg_train["batch_size"], shuffle=False)

        model = FallDetectionLSTM(
            input_size=num_input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        class_weights = train_ds.class_weights if cfg_train["class_weight_auto"] else None
        trainer = Trainer(
            model,
            learning_rate=cfg_train["learning_rate"],
            weight_decay=cfg_train.get("weight_decay", 0.0001),
            epochs=cfg_train["epochs"],
            early_stopping_patience=cfg_train["early_stopping_patience"],
            class_weights=class_weights,
            scheduler_patience=cfg_train.get("scheduler_patience", 5),
            scheduler_factor=cfg_train.get("scheduler_factor", 0.5),
        )

        trainer.train(train_loader, val_loader)

        # Inference
        model.eval()
        fold_preds, fold_probs = [], []
        with torch.no_grad():
            for x, _ in test_loader:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fold_preds.extend(logits.argmax(dim=1).numpy())
                fold_probs.extend(probs.numpy())

        fold_preds = np.array(fold_preds)
        fold_probs = np.array(fold_probs)

        evaluator = Evaluator(output_dir=results_dir)
        fold_metrics = evaluator.compute_metrics(test_labels, fold_preds, fold_probs)
        fold_aucs.append(fold_metrics["auc_roc"])

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(test_labels.tolist())
        all_probs.extend(fold_probs.tolist())

        logger.info(
            f"  {ablation_name} fold {fold_idx} ({test_subject}): "
            f"AUC={fold_metrics['auc_roc']:.3f}"
        )

    # Aggregate across all folds
    agg_preds = np.array(all_preds)
    agg_labels = np.array(all_labels)
    agg_probs = np.array(all_probs)

    evaluator = Evaluator(output_dir=results_dir)
    agg_metrics = evaluator.compute_metrics(agg_labels, agg_preds, agg_probs)
    agg_metrics["fold_aucs"] = fold_aucs
    agg_metrics["mean_fold_auc"] = float(np.mean(fold_aucs))
    agg_metrics["std_fold_auc"] = float(np.std(fold_aucs))

    return agg_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use if already computed)")
    parser.add_argument("--groups-only", action="store_true",
                        help="Only run group-level ablation (skip individual features)")
    args = parser.parse_args()

    config = _load_config()
    results_dir = Path(config["paths"]["results"]) / "ablation_study"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ── Baseline (all 15 features) ────────────────────────────────────────────
    if not args.skip_baseline:
        logger.info("\n=== BASELINE: All 15 features ===")
        t0 = time.time()
        baseline = run_ablation_loso(config, [], "baseline", results_dir)
        elapsed = time.time() - t0
        baseline["elapsed_seconds"] = round(elapsed, 1)
        all_results["baseline"] = baseline
        print(
            f"\nBaseline: AUC={baseline['auc_roc']:.4f}, "
            f"sens={baseline['sensitivity']:.3f}, spec={baseline['specificity']:.3f} "
            f"({elapsed:.0f}s)"
        )
        # Save intermediate results after each condition
        _save_results(all_results, results_dir)
    else:
        # Load existing baseline if available
        results_path = results_dir / "ablation_results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            logger.info(
                f"Loaded existing results with {len(all_results)} conditions "
                f"(baseline AUC={all_results.get('baseline', {}).get('auc_roc', 'N/A')})"
            )

    # ── Group-level ablation ──────────────────────────────────────────────────
    for group_name, indices in FEATURE_GROUPS.items():
        condition_name = f"remove_{group_name}"
        if condition_name in all_results:
            logger.info(f"Skipping {condition_name} (already computed)")
            continue

        removed_names = [FEATURE_NAMES[i] for i in indices]
        logger.info(
            f"\n=== ABLATION: Remove {group_name} ({len(indices)} features) ===\n"
            f"  Removing: {removed_names}\n"
            f"  Remaining: {FeatureExtractor.NUM_FEATURES - len(indices)} features"
        )

        t0 = time.time()
        result = run_ablation_loso(config, indices, condition_name, results_dir)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        result["removed_features"] = removed_names
        result["removed_indices"] = indices
        result["n_remaining"] = FeatureExtractor.NUM_FEATURES - len(indices)
        all_results[condition_name] = result

        baseline_auc = all_results.get("baseline", {}).get("auc_roc", 0.888)
        delta = result["auc_roc"] - baseline_auc
        print(
            f"\n{condition_name}: AUC={result['auc_roc']:.4f} "
            f"(delta={delta:+.4f}), "
            f"sens={result['sensitivity']:.3f}, spec={result['specificity']:.3f} "
            f"({elapsed:.0f}s)"
        )
        _save_results(all_results, results_dir)

    # ── Individual feature ablation ───────────────────────────────────────────
    if not args.groups_only:
        for feat_idx in INDIVIDUAL_FEATURES:
            condition_name = f"remove_{FEATURE_NAMES[feat_idx]}"
            if condition_name in all_results:
                logger.info(f"Skipping {condition_name} (already computed)")
                continue

            logger.info(
                f"\n=== ABLATION: Remove {FEATURE_NAMES[feat_idx]} (index {feat_idx}) ==="
            )

            t0 = time.time()
            result = run_ablation_loso(config, [feat_idx], condition_name, results_dir)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            result["removed_features"] = [FEATURE_NAMES[feat_idx]]
            result["removed_indices"] = [feat_idx]
            result["n_remaining"] = FeatureExtractor.NUM_FEATURES - 1
            all_results[condition_name] = result

            baseline_auc = all_results.get("baseline", {}).get("auc_roc", 0.888)
            delta = result["auc_roc"] - baseline_auc
            print(
                f"\n{condition_name}: AUC={result['auc_roc']:.4f} "
                f"(delta={delta:+.4f}), "
                f"sens={result['sensitivity']:.3f}, spec={result['specificity']:.3f} "
                f"({elapsed:.0f}s)"
            )
            _save_results(all_results, results_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    _print_final_summary(all_results)
    _generate_ablation_plot(all_results, results_dir)


def _save_results(results: dict, results_dir: Path) -> None:
    """Save ablation results to JSON (called after each condition for safety)."""
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also save a CSV summary
    csv_path = results_dir / "ablation_summary.csv"
    fields = ["condition", "auc_roc", "delta_auc", "sensitivity", "specificity",
              "f1", "n_remaining", "elapsed_seconds"]
    baseline_auc = results.get("baseline", {}).get("auc_roc", 0.888)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in results.items():
            w.writerow({
                "condition": name,
                "auc_roc": round(r.get("auc_roc", 0), 4),
                "delta_auc": round(r.get("auc_roc", 0) - baseline_auc, 4),
                "sensitivity": round(r.get("sensitivity", 0), 4),
                "specificity": round(r.get("specificity", 0), 4),
                "f1": round(r.get("f1", 0), 4),
                "n_remaining": r.get("n_remaining", 15),
                "elapsed_seconds": r.get("elapsed_seconds", 0),
            })


def _print_final_summary(results: dict) -> None:
    """Print a sorted summary table to stdout."""
    baseline_auc = results.get("baseline", {}).get("auc_roc", 0.888)

    print(f"\n{'='*70}")
    print("  FEATURE ABLATION STUDY — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Baseline AUC: {baseline_auc:.4f}")
    print(f"\n  {'Condition':<35} {'AUC':>7} {'Delta':>8} {'Sens':>6} {'Spec':>6}")
    print("  " + "-" * 65)

    # Sort by delta (most negative = most important)
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda x: x[1].get("auc_roc", 0) - baseline_auc,
    )

    for name, r in sorted_results:
        delta = r.get("auc_roc", 0) - baseline_auc
        impact = "***" if delta < -0.02 else "**" if delta < -0.01 else "*" if delta < -0.005 else ""
        print(
            f"  {name:<35} {r['auc_roc']:>7.4f} {delta:>+8.4f} "
            f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} {impact}"
        )

    print("\n  Impact: *** critical (>0.02 AUC), ** important (>0.01), * minor (>0.005)")


def _generate_ablation_plot(results: dict, results_dir: Path) -> None:
    """Generate a horizontal bar chart of AUC impact per ablation condition."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_auc = results.get("baseline", {}).get("auc_roc", 0.888)

    # Separate group and individual results
    group_items = [(k, v) for k, v in results.items()
                   if k.startswith("remove_") and k.replace("remove_", "") in FEATURE_GROUPS]
    indiv_items = [(k, v) for k, v in results.items()
                   if k.startswith("remove_") and k not in dict(group_items)]

    # Plot group ablation
    if group_items:
        group_items.sort(key=lambda x: x[1]["auc_roc"] - baseline_auc)
        names = [k.replace("remove_", "") for k, _ in group_items]
        deltas = [v["auc_roc"] - baseline_auc for _, v in group_items]

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#d32f2f" if d < -0.01 else "#ff9800" if d < -0.005 else "#4caf50" for d in deltas]
        bars = ax.barh(names, deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("AUC Change from Baseline", fontsize=11)
        ax.set_title(f"Feature Group Ablation (Baseline AUC={baseline_auc:.4f})", fontsize=13)
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for bar, delta in zip(bars, deltas):
            ax.text(
                delta + (0.001 if delta >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{delta:+.4f}",
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=9,
            )

        plt.tight_layout()
        fig.savefig(str(results_dir / "group_ablation.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved group ablation plot to {results_dir / 'group_ablation.png'}")

    # Plot individual feature ablation
    if indiv_items:
        indiv_items.sort(key=lambda x: x[1]["auc_roc"] - baseline_auc)
        names = [k.replace("remove_", "") for k, _ in indiv_items]
        deltas = [v["auc_roc"] - baseline_auc for _, v in indiv_items]

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ["#d32f2f" if d < -0.01 else "#ff9800" if d < -0.005 else "#4caf50" for d in deltas]
        bars = ax.barh(names, deltas, color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("AUC Change from Baseline", fontsize=11)
        ax.set_title(f"Individual Feature Ablation (Baseline AUC={baseline_auc:.4f})", fontsize=13)
        ax.grid(axis="x", alpha=0.3)

        for bar, delta in zip(bars, deltas):
            ax.text(
                delta + (0.001 if delta >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{delta:+.4f}",
                va="center", ha="left" if delta >= 0 else "right",
                fontsize=9,
            )

        plt.tight_layout()
        fig.savefig(str(results_dir / "individual_ablation.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved individual ablation plot to {results_dir / 'individual_ablation.png'}")


if __name__ == "__main__":
    main()
