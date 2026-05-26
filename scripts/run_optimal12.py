"""
Run full LOSO with the optimal 12-feature set identified by ablation.

Removes: wrist_hip_distance(11), com_acceleration(4), shoulder_ankle_distance(6)
Uses Run 5 config: stride=2, augment, BiLSTM h=128, dropout=0.5.

Saves per-fold models and results for potential ensemble with Run 5.
"""

import csv
import json
import sys
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

# Features to exclude (identified by ablation as noise)
EXCLUDE = [11, 4, 6]  # wrist_hip_distance, com_acceleration, shoulder_ankle_distance
N_FEATURES = FeatureExtractor.NUM_FEATURES - len(EXCLUDE)
RUN_NAME = "run10_optimal12"


def _select_features(seqs: np.ndarray) -> np.ndarray:
    keep = [i for i in range(seqs.shape[2]) if i not in EXCLUDE]
    return seqs[:, :, keep]


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    hidden_size = cfg_model["hidden_size"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)
    dropout = cfg_model["dropout"]

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    with open(urfd_dir / "metadata.json") as f:
        meta = json.load(f)
    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    models_dir = Path(config["paths"]["models"]) / RUN_NAME
    results_dir = Path(config["paths"]["results"]) / RUN_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(output_dir=results_dir)

    augmenter = KeypointAugmenter(flip_probability=0.0, noise_std=0.05,
                                   scale_range=(0.85, 1.15), seed=SEED)

    all_preds, all_labels, all_probs = [], [], []
    per_subject = {}

    logger.info(f"\n=== {RUN_NAME}: 12 features (exclude indices {EXCLUDE}) ===")

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]
        n_val = max(1, len(train_subjects) // 4)
        val_subjects = train_subjects[-n_val:]
        train_actual = train_subjects[:-n_val] or train_subjects

        fold_dir = models_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_seqs, train_labels, norm_stats = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=train_actual,
            window_size=cfg_feat["window_size"], stride=2)
        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=val_subjects,
            window_size=cfg_feat["window_size"], stride=2, norm_stats=norm_stats)
        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=[test_subject],
            window_size=cfg_feat["window_size"], stride=5, norm_stats=norm_stats)

        # Feature selection
        train_seqs = _select_features(train_seqs)
        val_seqs = _select_features(val_seqs)
        test_seqs = _select_features(test_seqs)

        train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        logger.info(
            f"Fold {fold_idx} ({test_subject}): "
            f"train={len(train_ds)} (F={train_ds.num_falls},A={train_ds.num_adl}), "
            f"val={len(val_ds)}, test={len(test_ds)}, features={N_FEATURES}"
        )

        model = FallDetectionLSTM(
            input_size=N_FEATURES, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

        class_weights = train_ds.class_weights if cfg_train["class_weight_auto"] else None
        trainer = Trainer(
            model, learning_rate=cfg_train["learning_rate"],
            weight_decay=cfg_train.get("weight_decay", 0.0001),
            epochs=cfg_train["epochs"],
            early_stopping_patience=cfg_train["early_stopping_patience"],
            class_weights=class_weights,
            scheduler_patience=cfg_train.get("scheduler_patience", 5),
            scheduler_factor=cfg_train.get("scheduler_factor", 0.5))

        history = trainer.train(
            DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True),
            DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False))

        trainer.save_checkpoint(fold_dir / "best_model.pth")
        np.save(str(fold_dir / "norm_mean.npy"), norm_stats[0])
        np.save(str(fold_dir / "norm_std.npy"), norm_stats[1])
        evaluator.plot_training_curves(history, fold_dir / "training_curves.png")

        # Inference
        model.eval()
        fp, fpr = [], []
        test_loader = DataLoader(test_ds, batch_size=cfg_train["batch_size"], shuffle=False)
        with torch.no_grad():
            for x, _ in test_loader:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fp.extend(logits.argmax(dim=1).numpy())
                fpr.extend(probs.numpy())

        fp = np.array(fp)
        fpr_arr = np.array(fpr)
        fold_metrics = evaluator.compute_metrics(test_labels, fp, fpr_arr)
        evaluator.save_results(fold_metrics, fold_dir / "metrics.json")
        evaluator.plot_confusion_matrix(test_labels, fp, fold_dir / "confusion_matrix.png")
        evaluator.plot_roc_curve(test_labels, fpr_arr, fold_dir / "roc_curve.png")

        all_preds.extend(fp.tolist())
        all_labels.extend(test_labels.tolist())
        all_probs.extend(fpr_arr.tolist())
        per_subject[test_subject] = fold_metrics

        logger.info(
            f"  -> sens={fold_metrics['sensitivity']:.3f}, "
            f"spec={fold_metrics['specificity']:.3f}, AUC={fold_metrics['auc_roc']:.3f}"
        )

    # Aggregate
    agg_labels = np.array(all_labels)
    agg_preds = np.array(all_preds)
    agg_probs = np.array(all_probs)

    agg = evaluator.compute_metrics(agg_labels, agg_preds, agg_probs)
    evaluator.save_results(agg, results_dir / "metrics.json")
    evaluator.plot_confusion_matrix(agg_labels, agg_preds, results_dir / "confusion_matrix.png")
    evaluator.plot_roc_curve(agg_labels, agg_probs, results_dir / "roc_curve.png")

    # Save probs for ensemble
    np.save(str(results_dir / "all_probs.npy"), agg_probs)
    np.save(str(results_dir / "all_labels.npy"), agg_labels)

    # Per-subject CSV
    csv_fields = ["subject", "sensitivity", "specificity", "f1", "auc_roc"]
    with open(results_dir / "per_subject_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for subj, m in sorted(per_subject.items()):
            w.writerow({k: round(m.get(k, 0), 4) if isinstance(m.get(k, 0), float) else m.get(k, subj)
                        for k in csv_fields} | {"subject": subj})

    # Threshold sweep
    from sklearn.metrics import f1_score, confusion_matrix as cm
    sweep = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = (agg_probs >= t).astype(int)
        tn, fp_, fn, tp = cm(agg_labels, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp_) if (tn + fp_) > 0 else 0
        f1 = f1_score(agg_labels, pred, zero_division=0)
        sweep.append({"threshold": round(float(t), 2), "sensitivity": round(sens, 4),
                       "specificity": round(spec, 4), "f1": round(f1, 4)})
    with open(results_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  {RUN_NAME}")
    print(f"{'='*60}")
    print(f"{'Subject':<10} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 36)
    for subj, m in sorted(per_subject.items()):
        print(f"{subj:<10} {m['sensitivity']:>6.3f} {m['specificity']:>6.3f} "
              f"{m['f1']:>6.3f} {m['auc_roc']:>6.3f}")
    print("-" * 36)
    print(f"{'Aggregated':<10} {agg['sensitivity']:>6.3f} {agg['specificity']:>6.3f} "
          f"{agg['f1']:>6.3f} {agg['auc_roc']:>6.3f}")

    print(f"\nThreshold sweep:")
    for r in sweep:
        marker = " <- both>=0.85" if r["sensitivity"] >= 0.85 and r["specificity"] >= 0.85 else ""
        marker = marker or (" <- sens>=0.90" if r["sensitivity"] >= 0.90 else "")
        print(f"  t={r['threshold']:.2f}  sens={r['sensitivity']:.3f}  "
              f"spec={r['specificity']:.3f}  f1={r['f1']:.3f}{marker}")

    # Ensemble with Run 5 probs
    run5_probs_path = Path("results/run5_stride2_aug")
    if (run5_probs_path / "all_probs.npy").exists():
        r5_probs = np.load(str(run5_probs_path / "all_probs.npy"))
        r5_labels = np.load(str(run5_probs_path / "all_labels.npy"))
        # Check alignment
        if len(r5_probs) == len(agg_probs):
            ens_probs = (r5_probs + agg_probs) / 2.0
            ens_preds = (ens_probs >= 0.5).astype(int)
            ens_metrics = evaluator.compute_metrics(agg_labels, ens_preds, ens_probs)
            print(f"\nEnsemble (Run5 + {RUN_NAME}):")
            print(f"  AUC={ens_metrics['auc_roc']:.4f}, sens={ens_metrics['sensitivity']:.3f}, "
                  f"spec={ens_metrics['specificity']:.3f}")
            np.save(str(results_dir / "ensemble_probs.npy"), ens_probs)
        else:
            print(f"\nCannot ensemble: Run5 has {len(r5_probs)} probs, "
                  f"{RUN_NAME} has {len(agg_probs)}")
    else:
        # Try to find Run 5 probs in another location
        for alt_path in [Path("results/urfd_loso_v3"), run5_probs_path]:
            for pname in ["all_probs.npy"]:
                p = alt_path / pname
                if p.exists():
                    logger.info(f"Found Run5 probs at {p}")


if __name__ == "__main__":
    main()
