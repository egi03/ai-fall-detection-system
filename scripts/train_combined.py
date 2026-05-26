"""
Combined URFD+Le2i training with LOSO cross-validation on URFD.

Strategy (Run 4):
  For each LOSO fold (5 folds, one URFD subject held out as test):
    - Training set  = URFD (4 subjects) + ALL Le2i sequences
    - Validation    = one URFD subject used as val (rotated from training 4)
    - Test          = held-out URFD subject
    - Normalization = joint z-score computed from combined training set,
                      same stats applied to test set

Motivation: URFD alone provides only ~500–750 training windows per fold.
Le2i adds ~6,574 windows (2,293 falls + 4,281 ADLs), effectively 9× more
training data while keeping the evaluation subject-independent on URFD.

Reference: research/1.2 Dataset Combination Strategies,
           research/8.2 LSTM Training Pipeline Best Practices.

Usage:
    python scripts/train_combined.py [--epochs 100] [--output-dir models/urfd_combined]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import (
    FallDetectionDataset,
    build_combined_dataset,
    build_dataset_from_processed,
)
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator
from src.data_processing.splitter import SubjectSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def _load_config() -> dict:
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_all_le2i_subjects(le2i_processed_dir: Path) -> list:
    """Return all subject IDs present in Le2i metadata."""
    metadata_path = le2i_processed_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return sorted(set(s["subject_id"] for s in metadata["sequences"]))


def train_combined_loso(
    urfd_dir: Path,
    le2i_dir: Path,
    config: dict,
    output_dir: Path,
) -> None:
    """
    Run LOSO cross-validation on URFD with Le2i as auxiliary training data.

    For each fold:
      - training = URFD(n-1 subjects) + all Le2i, jointly normalized
      - validation = one URFD subject (rotated from training 4), same norm stats
      - test = held-out URFD subject, same norm stats

    Parameters
    ----------
    urfd_dir : Path
        Processed URFD directory (keypoints_raw/ + metadata.json).
    le2i_dir : Path
        Processed Le2i directory (keypoints_raw/ + metadata.json).
    config : dict
        Loaded config.yaml.
    output_dir : Path
        Root directory to save fold models and results.
    """
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    # Load URFD subjects and build LOSO folds
    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        urfd_meta = json.load(f)
    urfd_subjects = sorted(set(s["subject_id"] for s in urfd_meta["sequences"]))
    le2i_subjects = _get_all_le2i_subjects(le2i_dir)

    splitter = SubjectSplitter(urfd_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    evaluator = Evaluator(output_dir=output_dir)
    all_preds = []
    all_labels_agg = []
    all_probs = []
    per_subject_results = {}

    logger.info(
        f"Combined LOSO: {len(folds)} folds, "
        f"URFD subjects={urfd_subjects}, "
        f"Le2i subjects={le2i_subjects}"
    )

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects_urfd = fold["train"]  # 4 URFD subjects

        # Split off one URFD subject for validation (last of the 4)
        # DECISION: Use last URFD training subject as val to keep val homogeneous
        # with test (both from URFD), ensuring val loss tracks URFD performance.
        n_val = max(1, len(train_subjects_urfd) // 4)
        val_subjects_urfd = train_subjects_urfd[-n_val:]
        train_subjects_urfd_actual = train_subjects_urfd[:-n_val]

        if not train_subjects_urfd_actual:
            # Edge case: if only 1 training subject, use it for both
            train_subjects_urfd_actual = train_subjects_urfd
            val_subjects_urfd = train_subjects_urfd[:1]

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"\n--- Fold {fold_idx} ---\n"
            f"  test  : {test_subject}\n"
            f"  val   : {val_subjects_urfd}\n"
            f"  train : URFD={train_subjects_urfd_actual} + Le2i(all {len(le2i_subjects)} rooms)\n"
        )

        # Build combined training set: URFD(n-2) + all Le2i, jointly normalized
        train_sources = [
            (urfd_dir, train_subjects_urfd_actual),
            (le2i_dir, le2i_subjects),
        ]
        train_seqs, train_labels, norm_stats = build_combined_dataset(
            sources=train_sources,
            window_size=cfg_feat["window_size"],
            stride=cfg_feat["window_stride"],
            positive_threshold=cfg_train["positive_label_threshold"],
        )

        # Validation: URFD val subject, normalized with training stats
        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=val_subjects_urfd,
            window_size=cfg_feat["window_size"],
            stride=cfg_feat["window_stride"],
            positive_threshold=cfg_train["positive_label_threshold"],
            norm_stats=norm_stats,
        )

        # Test: held-out URFD subject, normalized with training stats
        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=[test_subject],
            window_size=cfg_feat["window_size"],
            stride=cfg_feat["window_stride"],
            positive_threshold=cfg_train["positive_label_threshold"],
            norm_stats=norm_stats,
        )

        train_ds = FallDetectionDataset(train_seqs, train_labels)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        logger.info(
            f"  Windows — train: {len(train_ds)} "
            f"(falls={train_ds.num_falls}, adl={train_ds.num_adl}), "
            f"val: {len(val_ds)}, test: {len(test_ds)}"
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg_train["batch_size"], shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg_train["batch_size"], shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg_train["batch_size"], shuffle=False
        )

        model = FallDetectionLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            dropout=cfg_model["dropout"],
            bidirectional=cfg_model.get("bidirectional", False),
        )

        class_weights = (
            train_ds.class_weights if cfg_train["class_weight_auto"] else None
        )

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

        history = trainer.train(train_loader, val_loader)

        # Save model + normalization stats for inference
        trainer.save_checkpoint(fold_dir / "best_model.pth")
        np.save(str(fold_dir / "norm_mean.npy"), norm_stats[0])
        np.save(str(fold_dir / "norm_std.npy"), norm_stats[1])
        evaluator.plot_training_curves(history, fold_dir / "training_curves.png")

        # Evaluate on test set using the best model loaded from checkpoint
        best_model = FallDetectionLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            dropout=0.0,  # no dropout at inference
            bidirectional=cfg_model.get("bidirectional", False),
        )
        checkpoint = torch.load(
            str(fold_dir / "best_model.pth"), map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        best_model.load_state_dict(state)
        best_model.eval()

        fold_preds_list = []
        fold_probs_list = []
        with torch.no_grad():
            for sequences, _ in test_loader:
                logits = best_model(sequences)
                probs = torch.softmax(logits, dim=1)
                fold_preds_list.extend(logits.argmax(dim=1).numpy())
                fold_probs_list.extend(probs[:, 1].numpy())

        fold_preds = np.array(fold_preds_list)
        fold_probs = np.array(fold_probs_list)
        fold_metrics = evaluator.compute_metrics(test_labels, fold_preds, fold_probs)

        evaluator.save_results(fold_metrics, fold_dir / "metrics.json")
        evaluator.plot_confusion_matrix(
            test_labels, fold_preds, fold_dir / "confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            test_labels, fold_probs, fold_dir / "roc_curve.png"
        )

        all_preds.extend(fold_preds.tolist())
        all_labels_agg.extend(test_labels.tolist())  # test_labels is np.ndarray
        all_probs.extend(fold_probs.tolist())

        per_subject_results[test_subject] = fold_metrics
        logger.info(
            f"  Fold {fold_idx} ({test_subject}): "
            f"sens={fold_metrics['sensitivity']:.3f}, "
            f"spec={fold_metrics['specificity']:.3f}, "
            f"AUC={fold_metrics['auc_roc']:.3f}"
        )

    # Aggregate across all folds
    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels_agg)
    all_probs_arr = np.array(all_probs)

    agg_metrics = evaluator.compute_metrics(
        all_labels_arr, all_preds_arr, all_probs_arr
    )
    evaluator.save_results(agg_metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(
        all_labels_arr, all_preds_arr, output_dir / "confusion_matrix.png"
    )
    evaluator.plot_roc_curve(
        all_labels_arr, all_probs_arr, output_dir / "roc_curve.png"
    )

    # Per-subject CSV
    import csv
    csv_path = output_dir / "per_subject_results.csv"
    fieldnames = ["subject"] + list(next(iter(per_subject_results.values())).keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subj, metrics in per_subject_results.items():
            writer.writerow({"subject": subj, **metrics})

    logger.info(
        f"\n=== Combined LOSO Aggregated ===\n"
        f"  Sensitivity : {agg_metrics['sensitivity']:.4f}\n"
        f"  Specificity : {agg_metrics['specificity']:.4f}\n"
        f"  Accuracy    : {agg_metrics['accuracy']:.4f}\n"
        f"  F1          : {agg_metrics['f1']:.4f}\n"
        f"  AUC-ROC     : {agg_metrics['auc_roc']:.4f}\n"
        f"Results saved to {output_dir}"
    )


def main() -> None:
    """Entry point for combined URFD+Le2i LOSO training."""
    parser = argparse.ArgumentParser(
        description="Train LSTM fall detector on combined URFD+Le2i dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs from config"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/urfd_combined",
        help="Directory to save models and results (default: models/urfd_combined)",
    )
    args = parser.parse_args()

    config = _load_config()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    le2i_dir = Path(config["paths"]["data_processed"]) / "le2i"

    for d, name in [(urfd_dir, "URFD"), (le2i_dir, "Le2i")]:
        if not d.exists():
            logger.error(f"{name} processed data not found: {d}")
            logger.error("Run scripts/preprocess.py first.")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    # Mirror results alongside models
    results_dir = Path(config["paths"]["results"]) / output_dir.name
    results_dir.mkdir(parents=True, exist_ok=True)

    train_combined_loso(urfd_dir, le2i_dir, config, output_dir)

    # Copy aggregated metrics to results/ for comparison
    import shutil
    for fname in ["metrics.json", "confusion_matrix.png", "roc_curve.png",
                  "per_subject_results.csv"]:
        src = output_dir / fname
        if src.exists():
            shutil.copy2(src, results_dir / fname)
    logger.info(f"Results also copied to {results_dir}")


if __name__ == "__main__":
    main()
