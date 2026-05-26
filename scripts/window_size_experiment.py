"""
Window size sensitivity experiment for fall detection.

Tests window sizes [15, 20, 30, 40, 50] frames using full 5-fold LOSO
cross-validation on URFD dataset with Run 5 configuration:
  - stride=2, augment=True, BiLSTM h=128, dropout=0.5, bidirectional=True

Saves per-window-size results and an aggregate summary to results/window_size_sensitivity/.

Usage:
    python scripts/window_size_experiment.py

Reference: research/4.1, research/8.2
"""

import csv
import json
import sys
import time
from pathlib import Path

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

# Run 5 configuration (best single model)
WINDOW_SIZES = [15, 20, 30, 40, 50]
STRIDE = 2
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5
BIDIRECTIONAL = True
POS_THRESHOLD = 0.5


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _infer_fold_model(
    fold_dir: Path,
    test_loader: DataLoader,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> tuple:
    """Load best checkpoint and run inference. Returns (preds, probs)."""
    model = FallDetectionLSTM(
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        bidirectional=bidirectional,
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


def run_single_window_size(
    window_size: int,
    config: dict,
    urfd_dir: Path,
    folds: list,
    base_results_dir: Path,
    base_models_dir: Path,
) -> dict:
    """Run full LOSO for one window size. Returns aggregate metrics + per-subject."""
    cfg_train = config["training"]
    run_name = f"ws{window_size}"

    models_dir = base_models_dir / run_name
    results_dir = base_results_dir / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(output_dir=results_dir)

    augmenter = KeypointAugmenter(
        flip_probability=0.0,
        noise_std=0.05,
        scale_range=(0.85, 1.15),
        seed=SEED,
    )

    all_preds, all_labels, all_probs = [], [], []
    per_subject_results = {}

    print(f"\n{'='*60}")
    print(f"  Window size = {window_size} frames ({window_size / 15.0:.1f}s at 15fps)")
    print(f"{'='*60}")

    for fold_idx, fold in enumerate(folds):
        # Reset seeds each fold for reproducibility
        np.random.seed(SEED + fold_idx)
        torch.manual_seed(SEED + fold_idx)

        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        n_val = max(1, len(train_subjects) // 4)
        val_subjects = train_subjects[-n_val:]
        train_subjects_actual = train_subjects[:-n_val] or train_subjects

        fold_dir = models_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Build datasets with this window size
        train_seqs, train_labels, norm_stats = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=train_subjects_actual,
            window_size=window_size,
            stride=STRIDE,
            positive_threshold=POS_THRESHOLD,
        )

        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=val_subjects,
            window_size=window_size,
            stride=STRIDE,
            positive_threshold=POS_THRESHOLD,
            norm_stats=norm_stats,
        )

        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir,
            subject_ids=[test_subject],
            window_size=window_size,
            stride=5,  # always stride=5 for test (consistent with Run 3/5)
            positive_threshold=POS_THRESHOLD,
            norm_stats=norm_stats,
        )

        train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        print(
            f"  Fold {fold_idx} ({test_subject}): "
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

        model = FallDetectionLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            bidirectional=BIDIRECTIONAL,
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

        trainer.train(train_loader, val_loader)
        trainer.save_checkpoint(fold_dir / "best_model.pth")
        np.save(str(fold_dir / "norm_mean.npy"), norm_stats[0])
        np.save(str(fold_dir / "norm_std.npy"), norm_stats[1])

        fold_preds, fold_probs = _infer_fold_model(
            fold_dir, test_loader, HIDDEN_SIZE, NUM_LAYERS, BIDIRECTIONAL,
        )
        fold_metrics = evaluator.compute_metrics(
            test_labels, fold_preds, fold_probs
        )
        evaluator.save_results(fold_metrics, fold_dir / "metrics.json")

        all_preds.extend(fold_preds.tolist())
        all_labels.extend(test_labels.tolist())
        all_probs.extend(fold_probs.tolist())
        per_subject_results[test_subject] = fold_metrics

        print(
            f"    -> sens={fold_metrics['sensitivity']:.3f}, "
            f"spec={fold_metrics['specificity']:.3f}, "
            f"AUC={fold_metrics['auc_roc']:.3f}"
        )

    # Aggregate across all folds
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
    csv_fields = [
        "subject", "sensitivity", "specificity", "accuracy",
        "f1", "auc_roc", "pr_auc", "tp", "fp", "tn", "fn",
    ]
    with open(results_dir / "per_subject_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for subj, m in per_subject_results.items():
            w.writerow(
                {"subject": subj, **{k: m.get(k, "") for k in csv_fields[1:]}}
            )

    # Print summary for this window size
    print(f"\n  {'Subject':<10} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}")
    print(f"  {'-'*36}")
    for subj, m in sorted(per_subject_results.items()):
        print(
            f"  {subj:<10} {m['sensitivity']:>6.3f} {m['specificity']:>6.3f} "
            f"{m['f1']:>6.3f} {m['auc_roc']:>6.3f}"
        )
    print(f"  {'-'*36}")
    print(
        f"  {'AGGREGATE':<10} {agg['sensitivity']:>6.3f} {agg['specificity']:>6.3f} "
        f"{agg['f1']:>6.3f} {agg['auc_roc']:>6.3f}"
    )

    return {
        "window_size": window_size,
        "window_seconds": round(window_size / 15.0, 2),
        "aggregate": agg,
        "per_subject": per_subject_results,
        "num_train_windows_total": len(all_preds),
    }


def main() -> None:
    config = _load_config()
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        urfd_meta = json.load(f)
    urfd_subjects = sorted(set(s["subject_id"] for s in urfd_meta["sequences"]))

    splitter = SubjectSplitter(urfd_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    base_results_dir = Path(config["paths"]["results"]) / "window_size_sensitivity"
    base_models_dir = Path(config["paths"]["models"]) / "window_size_sensitivity"
    base_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WINDOW SIZE SENSITIVITY EXPERIMENT")
    print(f"  Window sizes: {WINDOW_SIZES}")
    print(f"  Config: stride={STRIDE}, hidden={HIDDEN_SIZE}, dropout={DROPOUT}, bidir={BIDIRECTIONAL}")
    print(f"  Subjects: {urfd_subjects}")
    print(f"  Folds: {len(folds)}")
    print("=" * 60)

    all_results = []
    start_time = time.time()

    for ws in WINDOW_SIZES:
        ws_start = time.time()
        result = run_single_window_size(
            window_size=ws,
            config=config,
            urfd_dir=urfd_dir,
            folds=folds,
            base_results_dir=base_results_dir,
            base_models_dir=base_models_dir,
        )
        ws_elapsed = time.time() - ws_start
        result["elapsed_seconds"] = round(ws_elapsed, 1)
        all_results.append(result)
        print(f"\n  [Window {ws}] completed in {ws_elapsed:.0f}s")

    total_elapsed = time.time() - start_time

    # Save aggregate summary
    summary = {
        "experiment": "window_size_sensitivity",
        "config": {
            "stride": STRIDE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "bidirectional": BIDIRECTIONAL,
            "pos_threshold": POS_THRESHOLD,
            "seed": SEED,
            "augment": True,
        },
        "results": [],
    }

    for r in all_results:
        summary["results"].append({
            "window_size": r["window_size"],
            "window_seconds": r["window_seconds"],
            "sensitivity": r["aggregate"]["sensitivity"],
            "specificity": r["aggregate"]["specificity"],
            "f1": r["aggregate"]["f1"],
            "auc_roc": r["aggregate"]["auc_roc"],
            "accuracy": r["aggregate"]["accuracy"],
            "pr_auc": r["aggregate"].get("pr_auc", None),
            "elapsed_seconds": r["elapsed_seconds"],
            "per_subject": {
                subj: {
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "f1": m["f1"],
                    "auc_roc": m["auc_roc"],
                }
                for subj, m in r["per_subject"].items()
            },
        })

    with open(base_results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print final comparison table
    print(f"\n\n{'='*70}")
    print("  WINDOW SIZE SENSITIVITY - FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'WinSize':>7} {'Seconds':>7} {'Sens':>7} {'Spec':>7} {'F1':>7} {'AUC':>7} {'Time':>7}")
    print(f"  {'-'*49}")
    for r in all_results:
        a = r["aggregate"]
        print(
            f"  {r['window_size']:>7} {r['window_seconds']:>7.2f} "
            f"{a['sensitivity']:>7.3f} {a['specificity']:>7.3f} "
            f"{a['f1']:>7.3f} {a['auc_roc']:>7.3f} "
            f"{r['elapsed_seconds']:>6.0f}s"
        )
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  Results saved to: {base_results_dir}")


if __name__ == "__main__":
    main()
