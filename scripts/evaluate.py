"""
Standalone evaluation script.

Loads a trained model and runs the full evaluation suite:
metrics, confusion matrix, ROC curves, cross-dataset eval.

Usage:
    python scripts/evaluate.py --dataset urfd --loso
    python scripts/evaluate.py --dataset urfd --model models/urfd/best_model.pth

Reference: research/5.1, research/5.2.
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
    build_dataset_from_processed,
)
from src.training.evaluate import Evaluator
from src.data_processing.splitter import SubjectSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42


def _load_config() -> dict:
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(model_path: Path, config: dict) -> "FallDetectionLSTM":
    """Load a trained model from a checkpoint file."""
    cfg_model = config["model"]
    model = FallDetectionLSTM(
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=cfg_model["hidden_size"],
        num_layers=cfg_model["num_layers"],
        dropout=0.0,  # No dropout at inference
        bidirectional=cfg_model.get("bidirectional", False),
    )
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_model(
    model_path: Path,
    processed_dir: Path,
    test_subjects: list,
    config: dict,
    output_dir: Path,
    norm_stats: tuple = None,
) -> dict:
    """Evaluate a trained model on specified test subjects.

    Parameters
    ----------
    model_path : Path
        Path to the model checkpoint.
    processed_dir : Path
        Directory containing processed keypoint files.
    test_subjects : list
        Subject IDs to use as the test set.
    config : dict
        Loaded configuration dictionary.
    output_dir : Path
        Directory for saving evaluation artifacts.
    norm_stats : tuple, optional
        (mean, std) from the training fold. Required for correct evaluation.
        If None, normalizes from test data (incorrect for LOSO).
    """
    cfg_feat = config["features"]
    cfg_train = config["training"]

    test_seqs, test_labels, _ = build_dataset_from_processed(
        processed_dir,
        test_subjects,
        window_size=cfg_feat["window_size"],
        stride=1,  # Dense evaluation
        positive_threshold=cfg_train["positive_label_threshold"],
        norm_stats=norm_stats,
    )

    test_ds = FallDetectionDataset(test_seqs, test_labels)
    test_loader = DataLoader(
        test_ds, batch_size=cfg_train["batch_size"], shuffle=False
    )

    model = _load_model(model_path, config)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_probs.extend(probs[:, 1].numpy())
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    evaluator = Evaluator(output_dir=output_dir)
    metrics = evaluator.compute_metrics(y_true, y_pred, y_prob)

    fps = config["video"]["target_fps"]
    total_frames = len(y_true) * cfg_feat["window_stride"]
    total_hours = total_frames / fps / 3600
    metrics["false_alarm_rate"] = evaluator.compute_false_alarm_rate(
        y_true, y_pred, total_hours
    )
    metrics["total_test_windows"] = len(y_true)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    evaluator.plot_roc_curve(y_true, y_prob, output_dir / "roc_curve.png")

    logger.info(
        f"Evaluation: sensitivity={metrics['sensitivity']:.3f} "
        f"specificity={metrics['specificity']:.3f} "
        f"f1={metrics['f1']:.3f} auc={metrics['auc_roc']:.3f}"
    )

    return metrics, y_true, y_prob


def evaluate_loso(
    processed_dir: Path,
    models_dir: Path,
    config: dict,
    output_dir: Path,
) -> None:
    """Run LOSO evaluation using per-fold models and their saved norm stats.

    For each fold, loads the corresponding model and training normalization
    statistics, then evaluates on the held-out test subject.
    """
    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    evaluator = Evaluator(output_dir=output_dir)
    per_subject_results = {}
    all_y_true = []
    all_y_prob = []

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        fold_dir = models_dir / f"fold_{fold_idx}"
        fold_model = fold_dir / "best_model.pth"
        fold_mean = fold_dir / "norm_mean.npy"
        fold_std = fold_dir / "norm_std.npy"

        if not fold_model.exists():
            logger.warning(f"Model not found for fold {fold_idx}: {fold_model}")
            continue
        if not fold_mean.exists() or not fold_std.exists():
            logger.warning(f"Norm stats missing for fold {fold_idx}")
            norm_stats = None
        else:
            norm_stats = (np.load(str(fold_mean)), np.load(str(fold_std)))

        fold_output = output_dir / f"fold_{fold_idx}"
        logger.info(f"Evaluating fold {fold_idx}: test={test_subject}")

        try:
            metrics, y_true, y_prob = evaluate_model(
                fold_model,
                processed_dir,
                [test_subject],
                config,
                fold_output,
                norm_stats=norm_stats,
            )
            metrics["test_subject"] = test_subject
            per_subject_results[test_subject] = metrics
            all_y_true.extend(y_true.tolist())
            all_y_prob.extend(y_prob.tolist())
        except Exception as e:
            logger.error(f"Fold {fold_idx} evaluation failed: {e}")
            per_subject_results[test_subject] = {"error": str(e)}

    # Aggregate across all folds
    if all_y_true:
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        all_y_pred = (all_y_prob >= 0.5).astype(int)

        agg_metrics = evaluator.compute_metrics(all_y_true, all_y_pred, all_y_prob)

        fps = config["video"]["target_fps"]
        cfg_feat = config["features"]
        total_frames = len(all_y_true) * cfg_feat["window_stride"]
        total_hours = total_frames / fps / 3600
        agg_metrics["false_alarm_rate"] = evaluator.compute_false_alarm_rate(
            all_y_true, all_y_pred, total_hours
        )
        agg_metrics["total_test_windows"] = len(all_y_true)

        evaluator.save_results(agg_metrics, output_dir / "metrics.json")
        evaluator.plot_confusion_matrix(
            all_y_true, all_y_pred, output_dir / "confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            all_y_true, all_y_prob, output_dir / "roc_curve.png"
        )

        per_subject_results["_aggregated"] = agg_metrics
        evaluator.save_results(
            per_subject_results, output_dir / "per_subject_results.json"
        )

        _save_per_subject_csv(per_subject_results, output_dir / "per_subject_results.csv")

        print("\n=== LOSO Evaluation Results ===")
        header = f"{'Subject':<10} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}"
        print(header)
        print("-" * len(header))
        for subj in sorted(k for k in per_subject_results if not k.startswith("_")):
            r = per_subject_results[subj]
            if "error" in r:
                print(f"{subj:<10} ERROR: {r['error']}")
            else:
                print(
                    f"{subj:<10} "
                    f"{r['sensitivity']:>6.3f} "
                    f"{r['specificity']:>6.3f} "
                    f"{r['f1']:>6.3f} "
                    f"{r['auc_roc']:>6.3f}"
                )
        agg = agg_metrics
        print("-" * len(header))
        print(
            f"{'Aggregated':<10} "
            f"{agg['sensitivity']:>6.3f} "
            f"{agg['specificity']:>6.3f} "
            f"{agg['f1']:>6.3f} "
            f"{agg['auc_roc']:>6.3f}"
        )
        print(f"\nFalse alarm rate: {agg['false_alarm_rate']:.2f} per hour")
        print(f"Results saved to: {output_dir}")


def _save_per_subject_csv(results: dict, path: Path) -> None:
    """Save per-subject results to a CSV file."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["subject", "sensitivity", "specificity", "accuracy", "f1", "auc_roc",
              "pr_auc", "false_alarm_rate", "tp", "fp", "tn", "fn"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for subj, r in sorted(results.items()):
            if subj.startswith("_") or "error" in r:
                continue
            writer.writerow({
                "subject": subj,
                **{k: r.get(k, "") for k in fields[1:]},
            })
    logger.info(f"Per-subject CSV saved to {path}")


def main() -> None:
    """Execute the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate fall detection model")
    parser.add_argument("--dataset", type=str, default="urfd")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--loso", action="store_true", help="Run full LOSO evaluation")
    parser.add_argument(
        "--results-name", type=str, default=None,
        help="Results subdirectory name (default: {dataset}_loso_v3)"
    )
    args = parser.parse_args()

    config = _load_config()
    processed_dir = Path(config["paths"]["data_processed"]) / args.dataset
    models_dir = Path(config["paths"]["models"]) / args.dataset

    results_name = args.results_name or f"{args.dataset}_loso_v3"
    output_dir = Path(config["paths"]["results"]) / results_name

    if not processed_dir.exists():
        logger.error(f"Processed data not found: {processed_dir}")
        sys.exit(1)

    if args.loso:
        evaluate_loso(processed_dir, models_dir, config, output_dir)
    else:
        model_path = Path(args.model) if args.model else (models_dir / "best_model.pth")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)

        metadata_path = processed_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
        splitter = SubjectSplitter(subjects, seed=SEED)
        split = splitter.split_train_val_test()
        test_subjects = split["test"]

        metrics, y_true, y_prob = evaluate_model(
            model_path, processed_dir, test_subjects, config, output_dir
        )

        print("\n=== Evaluation Results ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
