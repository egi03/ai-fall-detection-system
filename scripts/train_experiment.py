"""
Configurable URFD LOSO experiment runner (also supports full-dataset training).

Allows systematic comparison of:
  --stride       : sliding window stride (default 5; try 2 for more windows)
  --augment      : enable feature-level noise + scaling augmentation during training
  --hidden-size  : LSTM hidden units (default 128; try 64 to reduce overfitting)
  --dropout      : LSTM dropout (default 0.5)
  --threshold    : positive label threshold (default 0.5; try 0.3 for more fall windows)
  --aux-adl      : add Le2i ADL-only rooms as extra negative training examples
  --run-name     : name for saving results (e.g. run5_stride2_aug)
  --no-loso      : train on ALL subjects (for cross-dataset evaluation source models)
  --dataset      : which dataset to train on: urfd (default) or le2i
  --architecture : model architecture: lstm (BiLSTM) or transformer_lstm (Transformer-LSTM hybrid)

Usage:
    python scripts/train_experiment.py --run-name run5 --stride 2 --augment
    python scripts/train_experiment.py --run-name run6 --stride 2 --augment --hidden-size 64
    python scripts/train_experiment.py --run-name run7 --stride 2 --augment --aux-adl
    # Transformer-LSTM hybrid:
    python scripts/train_experiment.py --run-name tl_run1 --stride 2 --augment --architecture transformer_lstm
    # Full-dataset models for cross-dataset evaluation:
    python scripts/train_experiment.py --run-name cross_urfd_full_r5 --stride 2 --augment --no-loso
    python scripts/train_experiment.py --run-name cross_le2i_full_r5 --stride 2 --augment --no-loso --dataset le2i

Reference: research/8.2, research/1.2, research/4.1.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data_processing.preprocessor import KeypointAugmenter
from src.data_processing.splitter import SubjectSplitter
from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.models.transformer_lstm import FallDetectionTransformerLSTM
from src.training.dataset import (
    FallDetectionDataset,
    build_combined_dataset,
    build_dataset_from_processed,
)
from src.training.evaluate import Evaluator
from src.training.trainer import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Le2i ADL-only rooms — ceiling/side cameras, no fall videos, safe to use as extra negatives
LE2I_ADL_ONLY_ROOMS = ["Coffee_room_02", "Lecture_room", "Office"]


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(
    architecture: str,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    bidirectional: bool,
    transformer_cfg: dict,
) -> torch.nn.Module:
    """
    Create a model instance for the specified architecture.

    Parameters
    ----------
    architecture : str
        "lstm" or "transformer_lstm".
    input_size : int
        Number of input features.
    hidden_size : int
        Hidden size for LSTM architecture.
    num_layers : int
        Layer count for LSTM architecture.
    dropout : float
        Dropout rate.
    bidirectional : bool
        Whether to use bidirectional (LSTM only).
    transformer_cfg : dict
        Transformer-LSTM hyperparameters (d_model, nhead, etc.).

    Returns
    -------
    torch.nn.Module
        The constructed model.
    """
    if architecture == "transformer_lstm":
        return FallDetectionTransformerLSTM(
            input_size=input_size,
            d_model=transformer_cfg.get("d_model", 64),
            nhead=transformer_cfg.get("nhead", 4),
            num_encoder_layers=transformer_cfg.get("num_encoder_layers", 2),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 128),
            lstm_hidden_size=transformer_cfg.get("lstm_hidden_size", 64),
            lstm_num_layers=transformer_cfg.get("lstm_num_layers", 1),
            dropout=dropout,
        )
    else:
        return FallDetectionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )


def _infer_fold_model(
    fold_dir: Path,
    test_loader: DataLoader,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
    architecture: str = "lstm",
    transformer_cfg: Optional[dict] = None,
) -> tuple:
    """Load best checkpoint and run inference. Returns (preds, probs)."""
    model = _build_model(
        architecture=architecture,
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        bidirectional=bidirectional,
        transformer_cfg=transformer_cfg or {},
    )
    ckpt = torch.load(
        str(fold_dir / "best_model.pth"), map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    preds_list, probs_list = [], []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds_list.extend(logits.argmax(dim=1).numpy())
            probs_list.extend(probs.numpy())

    return np.array(preds_list), np.array(probs_list)


def run_loso(args: argparse.Namespace, config: dict) -> None:
    """
    Run LOSO cross-validation with the given experiment configuration.

    Supports all three datasets (urfd, le2i, up_fall) via --dataset flag.
    When --aux-adl is used the primary dataset must be urfd or up_fall;
    Le2i ADL-only rooms are added as extra negatives from the le2i processed dir.
    """
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    # Allow CLI overrides
    hidden_size = args.hidden_size if args.hidden_size is not None else cfg_model["hidden_size"]
    dropout = args.dropout if args.dropout is not None else cfg_model["dropout"]
    stride = args.stride
    pos_threshold = args.threshold
    bidirectional = cfg_model.get("bidirectional", False)
    architecture = args.architecture

    # Transformer-LSTM hyperparameters (from config or defaults)
    cfg_transformer = config.get("transformer_lstm", {})
    transformer_cfg = {
        "d_model": cfg_transformer.get("d_model", 64),
        "nhead": cfg_transformer.get("nhead", 4),
        "num_encoder_layers": cfg_transformer.get("num_encoder_layers", 2),
        "dim_feedforward": cfg_transformer.get("dim_feedforward", 128),
        "lstm_hidden_size": cfg_transformer.get("lstm_hidden_size", 64),
        "lstm_num_layers": cfg_transformer.get("lstm_num_layers", 1),
    }

    # DECISION: Use args.dataset to support urfd, le2i, and up_fall.
    # The primary data_dir is resolved from the dataset name.  Le2i is still
    # used as an auxiliary ADL source when --aux-adl is passed.
    dataset_name = args.dataset
    data_dir = Path(config["paths"]["data_processed"]) / dataset_name
    le2i_dir = Path(config["paths"]["data_processed"]) / "le2i"

    with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
        dataset_meta = json.load(f)
    all_subjects = sorted(
        set(s["subject_id"] for s in dataset_meta["sequences"])
    )

    splitter = SubjectSplitter(all_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    models_dir = Path(config["paths"]["models"]) / args.run_name
    results_dir = Path(config["paths"]["results"]) / args.run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(output_dir=results_dir)

    # Feature-level augmenter: noise + scaling only (no flip — not meaningful for feature vecs)
    augmenter = None
    if args.augment:
        augmenter = KeypointAugmenter(
            flip_probability=0.0,   # flip is not meaningful for feature vectors
            noise_std=0.05,         # Gaussian noise on normalized features
            scale_range=(0.85, 1.15),
            seed=SEED,
        )
        logger.info("Augmentation enabled: noise_std=0.05, scale=(0.85, 1.15)")

    all_preds, all_labels, all_probs = [], [], []
    per_subject_results = {}

    logger.info(
        f"\n=== Experiment: {args.run_name} ===\n"
        f"  architecture={architecture}, dataset={dataset_name}, stride={stride}, "
        f"hidden={hidden_size}, dropout={dropout}, augment={args.augment}, "
        f"aux_adl={args.aux_adl}, pos_threshold={pos_threshold}\n"
    )

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        n_val = max(1, len(train_subjects) // 4)
        val_subjects = train_subjects[-n_val:]
        train_subjects_actual = train_subjects[:-n_val] or train_subjects

        fold_dir = models_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build training set
        if args.aux_adl:
            # DECISION: Add Le2i ADL-only rooms as extra negatives.
            # These rooms have no fall videos so they only add negative examples,
            # avoiding the fall-domain-shift that broke Run 4.
            # We include Coffee_room_02, Lecture_room, Office.
            train_sources = [
                (data_dir, train_subjects_actual),
                (le2i_dir, LE2I_ADL_ONLY_ROOMS),
            ]
            train_seqs, train_labels, norm_stats = build_combined_dataset(
                sources=train_sources,
                window_size=cfg_feat["window_size"],
                stride=stride,
                positive_threshold=pos_threshold,
            )
        else:
            train_seqs, train_labels, norm_stats = build_dataset_from_processed(
                processed_dir=data_dir,
                subject_ids=train_subjects_actual,
                window_size=cfg_feat["window_size"],
                stride=stride,
                positive_threshold=pos_threshold,
            )

        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=data_dir,
            subject_ids=val_subjects,
            window_size=cfg_feat["window_size"],
            stride=stride,
            positive_threshold=pos_threshold,
            norm_stats=norm_stats,
        )
        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=data_dir,
            subject_ids=[test_subject],
            window_size=cfg_feat["window_size"],
            stride=5,  # always stride=5 for test (consistent with Run 3)
            positive_threshold=pos_threshold,
            norm_stats=norm_stats,
        )

        # Pass augmenter only to training set
        train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        logger.info(
            f"Fold {fold_idx} ({test_subject}): "
            f"train={len(train_ds)} (F={train_ds.num_falls},A={train_ds.num_adl}), "
            f"val={len(val_ds)}, test={len(test_ds)}"
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

        model = _build_model(
            architecture=architecture,
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=hidden_size,
            num_layers=cfg_model["num_layers"],
            dropout=dropout,
            bidirectional=bidirectional,
            transformer_cfg=transformer_cfg,
        )

        # Log parameter count
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold_idx == 0:
            logger.info(f"Model architecture: {architecture}, params: {n_params:,}")

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
        trainer.save_checkpoint(fold_dir / "best_model.pth")
        np.save(str(fold_dir / "norm_mean.npy"), norm_stats[0])
        np.save(str(fold_dir / "norm_std.npy"), norm_stats[1])
        evaluator.plot_training_curves(
            history, fold_dir / "training_curves.png"
        )

        fold_preds, fold_probs = _infer_fold_model(
            fold_dir, test_loader, hidden_size,
            cfg_model["num_layers"], bidirectional,
            architecture=architecture,
            transformer_cfg=transformer_cfg,
        )
        fold_metrics = evaluator.compute_metrics(
            test_labels, fold_preds, fold_probs
        )
        evaluator.save_results(fold_metrics, fold_dir / "metrics.json")
        evaluator.plot_confusion_matrix(
            test_labels, fold_preds, fold_dir / "confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            test_labels, fold_probs, fold_dir / "roc_curve.png"
        )

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(test_labels.tolist())
        all_probs.extend(fold_probs.tolist())
        per_subject_results[test_subject] = fold_metrics

        logger.info(
            f"  → sens={fold_metrics['sensitivity']:.3f}, "
            f"spec={fold_metrics['specificity']:.3f}, "
            f"AUC={fold_metrics['auc_roc']:.3f}"
        )

    # Aggregate
    agg_preds = np.array(all_preds)
    agg_labels = np.array(all_labels)
    agg_probs = np.array(all_probs)

    agg = evaluator.compute_metrics(agg_labels, agg_preds, agg_probs)
    evaluator.save_results(agg, results_dir / "metrics.json")
    evaluator.plot_confusion_matrix(
        agg_labels, agg_preds, results_dir / "confusion_matrix.png"
    )
    evaluator.plot_roc_curve(agg_labels, agg_probs, results_dir / "roc_curve.png")

    # Per-subject CSV
    import csv
    csv_fields = ["subject", "sensitivity", "specificity", "accuracy",
                  "f1", "auc_roc", "pr_auc", "tp", "fp", "tn", "fn"]
    with open(results_dir / "per_subject_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for subj, m in per_subject_results.items():
            w.writerow({"subject": subj, **{k: m.get(k, "") for k in csv_fields[1:]}})

    # Threshold sweep for this run
    from sklearn.metrics import f1_score, confusion_matrix as cm
    sweep_results = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = (agg_probs >= t).astype(int)
        tn, fp, fn, tp = cm(agg_labels, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(agg_labels, pred, zero_division=0)
        sweep_results.append({"threshold": round(float(t), 2), "sensitivity": sens,
                               "specificity": spec, "f1": f1})
    with open(results_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep_results, f, indent=2)

    _print_summary(args.run_name, agg, sweep_results, per_subject_results)


def _print_summary(
    run_name: str, agg: dict, sweep: list, per_subject: dict
) -> None:
    """Print a concise results summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")
    print(f"{'Subject':<10} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 36)
    for subj, m in sorted(per_subject.items()):
        print(
            f"{subj:<10} {m['sensitivity']:>6.3f} {m['specificity']:>6.3f} "
            f"{m['f1']:>6.3f} {m['auc_roc']:>6.3f}"
        )
    print("-" * 36)
    print(
        f"{'Aggregated':<10} {agg['sensitivity']:>6.3f} {agg['specificity']:>6.3f} "
        f"{agg['f1']:>6.3f} {agg['auc_roc']:>6.3f}"
    )
    print(f"\nThreshold sweep (sens / spec):")
    for r in sweep:
        marker = " ← both≥0.85" if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.85 else ""
        marker = marker or (" ← sens≥0.90" if r["sensitivity"] >= 0.90 else "")
        print(
            f"  t={r['threshold']:.2f}  sens={r['sensitivity']:.3f}  "
            f"spec={r['specificity']:.3f}  f1={r['f1']:.3f}{marker}"
        )


def run_full_dataset(args: argparse.Namespace, config: dict) -> None:
    """
    Train a single model on the FULL source dataset (no LOSO).

    Used to create source models for cross-dataset evaluation.
    Trains on all subjects except the last one (used as validation).
    Saves model + normalization stats to models/{run_name}/.

    DECISION: For cross-dataset evaluation we train on the full source dataset
    so the model sees the maximum possible training data (per PLAN_CROSS_DATASET_EVAL.md).
    """
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    hidden_size = args.hidden_size if args.hidden_size is not None else cfg_model["hidden_size"]
    dropout = args.dropout if args.dropout is not None else cfg_model["dropout"]
    stride = args.stride
    pos_threshold = args.threshold
    bidirectional = cfg_model.get("bidirectional", False)
    architecture = args.architecture

    # Transformer-LSTM hyperparameters
    cfg_transformer = config.get("transformer_lstm", {})
    transformer_cfg = {
        "d_model": cfg_transformer.get("d_model", 64),
        "nhead": cfg_transformer.get("nhead", 4),
        "num_encoder_layers": cfg_transformer.get("num_encoder_layers", 2),
        "dim_feedforward": cfg_transformer.get("dim_feedforward", 128),
        "lstm_hidden_size": cfg_transformer.get("lstm_hidden_size", 64),
        "lstm_num_layers": cfg_transformer.get("lstm_num_layers", 1),
    }

    dataset_name = args.dataset  # "urfd" or "le2i"
    data_dir = Path(config["paths"]["data_processed"]) / dataset_name

    with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    all_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    logger.info(
        f"\n=== Full-dataset training: {args.run_name} ===\n"
        f"  architecture={architecture}, dataset={dataset_name}, subjects={all_subjects}\n"
        f"  stride={stride}, hidden={hidden_size}, dropout={dropout}, "
        f"augment={args.augment}\n"
    )

    # Last subject → validation; rest → training
    val_subjects = [all_subjects[-1]]
    train_subjects = all_subjects[:-1]

    augmenter = None
    if args.augment:
        augmenter = KeypointAugmenter(
            flip_probability=0.0,
            noise_std=0.05,
            scale_range=(0.85, 1.15),
            seed=SEED,
        )
        logger.info("Augmentation enabled: noise_std=0.05, scale=(0.85, 1.15)")

    train_seqs, train_labels, norm_stats = build_dataset_from_processed(
        processed_dir=data_dir,
        subject_ids=train_subjects,
        window_size=cfg_feat["window_size"],
        stride=stride,
        positive_threshold=pos_threshold,
    )
    val_seqs, val_labels, _ = build_dataset_from_processed(
        processed_dir=data_dir,
        subject_ids=val_subjects,
        window_size=cfg_feat["window_size"],
        stride=stride,
        positive_threshold=pos_threshold,
        norm_stats=norm_stats,
    )

    train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
    val_ds = FallDetectionDataset(val_seqs, val_labels)

    logger.info(
        f"Train: {len(train_ds)} windows (F={train_ds.num_falls}, A={train_ds.num_adl}), "
        f"Val ({val_subjects[0]}): {len(val_ds)} windows"
    )

    train_loader = DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False)

    model = _build_model(
        architecture=architecture,
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=hidden_size,
        num_layers=cfg_model["num_layers"],
        dropout=dropout,
        bidirectional=bidirectional,
        transformer_cfg=transformer_cfg,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model architecture: {architecture}, params: {n_params:,}")

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

    model_dir = Path(config["paths"]["models"]) / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(model_dir / "best_model.pth")
    np.save(str(model_dir / "norm_mean.npy"), norm_stats[0])
    np.save(str(model_dir / "norm_std.npy"), norm_stats[1])

    # Save metadata about this run for cross_dataset_eval.py to reference
    run_meta = {
        "run_name": args.run_name,
        "architecture": architecture,
        "train_dataset": dataset_name,
        "train_subjects": train_subjects,
        "val_subject": val_subjects[0],
        "stride": stride,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "augment": args.augment,
        "window_size": cfg_feat["window_size"],
        "bidirectional": bidirectional,
        "norm_mean_shape": list(norm_stats[0].shape),
        "norm_std_shape": list(norm_stats[1].shape),
    }
    if architecture == "transformer_lstm":
        run_meta["transformer_cfg"] = transformer_cfg
    with open(model_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    logger.info(f"Saved model + norm stats to {model_dir}")
    print(f"\nDone. Model saved to: {model_dir}")
    print(f"  Trained on {dataset_name} subjects: {train_subjects}")
    print(f"  Validated on: {val_subjects[0]}")
    print(f"  Ready for cross-dataset evaluation with: python scripts/cross_dataset_eval.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fall detection experiment runner")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Name for this run (used for model/results dirs)")
    parser.add_argument("--stride", type=int, default=5,
                        help="Sliding window stride (default 5)")
    parser.add_argument("--augment", action="store_true",
                        help="Enable feature-level augmentation (noise + scaling)")
    parser.add_argument("--hidden-size", type=int, default=None,
                        help="LSTM hidden size override (default: from config)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout override (default: from config)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Positive label threshold (default 0.5)")
    parser.add_argument("--aux-adl", action="store_true",
                        help="Add Le2i ADL-only rooms as extra negative training data")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override max epochs from config")
    parser.add_argument("--no-loso", action="store_true",
                        help="Train on ALL subjects (for cross-dataset source models)")
    parser.add_argument("--dataset", type=str, default="urfd",
                        choices=["urfd", "le2i", "up_fall"],
                        help="Dataset to train on: urfd (default), le2i, or up_fall")
    parser.add_argument("--architecture", type=str, default="lstm",
                        choices=["lstm", "transformer_lstm"],
                        help="Model architecture: lstm (BiLSTM) or transformer_lstm (Transformer-LSTM hybrid)")
    args = parser.parse_args()

    config = _load_config()
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    if args.no_loso:
        run_full_dataset(args, config)
    else:
        run_loso(args, config)


if __name__ == "__main__":
    main()
