"""
Training entry point for the LSTM fall detection model.

Loads preprocessed data, constructs DataLoaders, trains the model
with LOSO cross-validation, and saves checkpoints.

Usage:
    python scripts/train.py --dataset urfd [--loso] [--epochs 100]

Reference: research/8.2, research/5.1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import (
    FallDetectionDataset,
    build_dataset_from_processed,
)
from src.training.trainer import Trainer
from src.training.evaluate import Evaluator
from src.data_processing.splitter import SubjectSplitter
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def _load_config() -> dict:
    config_path = Path("config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_single_split(
    processed_dir: Path,
    train_subjects: list,
    val_subjects: list,
    config: dict,
    output_dir: Path,
) -> dict:
    """Train on a single train/val split. Returns training history."""
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    # Build datasets — val set uses train normalization stats
    train_seqs, train_labels, norm_stats = build_dataset_from_processed(
        processed_dir,
        train_subjects,
        window_size=cfg_feat["window_size"],
        stride=cfg_feat["window_stride"],
        positive_threshold=cfg_train["positive_label_threshold"],
    )

    val_seqs, val_labels, _ = build_dataset_from_processed(
        processed_dir,
        val_subjects,
        window_size=cfg_feat["window_size"],
        stride=cfg_feat["window_stride"],
        positive_threshold=cfg_train["positive_label_threshold"],
        norm_stats=norm_stats,
    )

    train_ds = FallDetectionDataset(train_seqs, train_labels)
    val_ds = FallDetectionDataset(val_seqs, val_labels)

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train["batch_size"], shuffle=False
    )

    # Model
    model = FallDetectionLSTM(
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=cfg_model["hidden_size"],
        num_layers=cfg_model["num_layers"],
        dropout=cfg_model["dropout"],
        bidirectional=cfg_model.get("bidirectional", False),
    )

    # Class weights
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

    logger.info(
        f"Training: {len(train_ds)} train, {len(val_ds)} val, "
        f"falls: {train_ds.num_falls}/{val_ds.num_falls}"
    )

    history = trainer.train(train_loader, val_loader)

    # Save model and normalization stats (needed to normalize test data consistently)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(output_dir / "best_model.pth")
    np.save(str(output_dir / "norm_mean.npy"), norm_stats[0])
    np.save(str(output_dir / "norm_std.npy"), norm_stats[1])

    return history, norm_stats


def train_loso(processed_dir: Path, config: dict, output_dir: Path) -> None:
    """Run full LOSO cross-validation training."""
    # Load metadata to get subjects
    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    evaluator = Evaluator(output_dir=output_dir)
    all_results = {}

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        # Use 20% of train subjects for validation
        n_val = max(1, len(train_subjects) // 5)
        val_subjects = train_subjects[-n_val:]
        train_subjects_actual = train_subjects[:-n_val]

        if not train_subjects_actual:
            train_subjects_actual = train_subjects
            val_subjects = train_subjects[:1]

        fold_dir = output_dir / f"fold_{fold_idx}"
        logger.info(f"LOSO Fold {fold_idx}: test={test_subject}")

        try:
            history, _ = train_single_split(
                processed_dir,
                train_subjects_actual,
                val_subjects,
                config,
                fold_dir,
            )

            evaluator.plot_training_curves(
                history, fold_dir / "training_curves.png"
            )

            all_results[test_subject] = {
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "epochs_trained": len(history["train_loss"]),
            }
        except ValueError as e:
            logger.warning(f"Fold {fold_idx} skipped: {e}")
            all_results[test_subject] = {"error": str(e)}

    evaluator.save_results(all_results, output_dir / "loso_results.json")
    logger.info(f"LOSO training complete. Results in {output_dir}")


def main() -> None:
    """Execute the training pipeline."""
    parser = argparse.ArgumentParser(description="Train LSTM fall detector")
    parser.add_argument(
        "--dataset", type=str, default="urfd",
        help="Dataset name (must be preprocessed)",
    )
    parser.add_argument("--loso", action="store_true", help="Run LOSO CV")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    config = _load_config()
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    processed_dir = Path(config["paths"]["data_processed"]) / args.dataset
    output_dir = Path(config["paths"]["models"]) / args.dataset

    if not processed_dir.exists():
        logger.error(f"Processed data not found: {processed_dir}")
        logger.error("Run scripts/preprocess.py first.")
        sys.exit(1)

    if args.loso:
        train_loso(processed_dir, config, output_dir)
    else:
        # Simple train/val split
        metadata_path = processed_dir / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
        splitter = SubjectSplitter(subjects, seed=SEED)
        split = splitter.split_train_val_test()

        train_subjects = split["train"] + split["val"]
        val_subjects = split["test"]

        history, _ = train_single_split(
            processed_dir, train_subjects, val_subjects, config, output_dir
        )

        evaluator = Evaluator(output_dir=output_dir)
        evaluator.plot_training_curves(
            history, output_dir / "training_curves.png"
        )
        logger.info(f"Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
