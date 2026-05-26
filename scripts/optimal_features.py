"""
Validate optimal and minimal feature sets identified by ablation study.

Based on ablation_study.py results:
- 5 features HURT performance when included: wrist_hip_distance(11),
  com_acceleration(4), shoulder_ankle_distance(6), right_knee_angle(8),
  right_hip_angle(10)
- Top 5 most important: hip_shoulder_angle(1), left_knee_angle(7),
  mean_visibility(13), body_spread(12), head_to_toe_distance(5)

Experiments:
  1. Optimal-10: Remove 5 noisy features -> 10 features
  2. Minimal-5: Keep only top 5 critical features
  3. Optimal-12: Remove only the 3 worst (wrist_hip, com_accel, sh_ankle)
  4. BBox-removed: All 14 features minus bbox_aspect_ratio

Usage:
    python scripts/optimal_features.py
"""

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

FEATURE_NAMES = [
    "torso_inclination", "hip_shoulder_angle", "bbox_aspect_ratio",
    "com_velocity", "com_acceleration", "head_to_toe_distance",
    "shoulder_ankle_distance", "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle", "wrist_hip_distance",
    "body_spread", "mean_visibility", "min_core_visibility",
]

# Feature set configurations to test
CONFIGS = {
    "baseline_15": {
        "exclude": [],
        "desc": "All 15 features (baseline)"
    },
    "optimal_10": {
        "exclude": [11, 4, 6, 8, 10],  # wrist_hip, com_accel, sh_ankle, r_knee, r_hip
        "desc": "Remove 5 noisy features"
    },
    "optimal_12": {
        "exclude": [11, 4, 6],  # wrist_hip, com_accel, sh_ankle (top 3 worst)
        "desc": "Remove 3 worst features"
    },
    "minimal_5": {
        "exclude": [0, 2, 3, 4, 6, 8, 9, 10, 11, 14],  # keep: 1,5,7,12,13
        "desc": "Top 5 only: hip_sh_angle, head_toe, l_knee, body_spread, mean_vis"
    },
    "no_bbox_14": {
        "exclude": [2],  # bbox_aspect_ratio only
        "desc": "All 14 minus bbox"
    },
}


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_features(sequences: np.ndarray, exclude: List[int]) -> np.ndarray:
    keep = [i for i in range(sequences.shape[2]) if i not in exclude]
    return sequences[:, :, keep]


def run_loso_with_features(
    config: dict, exclude_indices: List[int], run_name: str
) -> Dict:
    """Run 5-fold LOSO with Run 5 config, excluding specified features."""
    cfg_train = config["training"]
    cfg_model = config["model"]
    cfg_feat = config["features"]

    hidden_size = cfg_model["hidden_size"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)
    dropout = cfg_model["dropout"]
    n_features = FeatureExtractor.NUM_FEATURES - len(exclude_indices)

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    with open(urfd_dir / "metadata.json") as f:
        meta = json.load(f)
    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    augmenter = KeypointAugmenter(flip_probability=0.0, noise_std=0.05,
                                   scale_range=(0.85, 1.15), seed=SEED)

    all_preds, all_labels, all_probs = [], [], []
    fold_aucs = []

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]
        n_val = max(1, len(train_subjects) // 4)
        val_subjects = train_subjects[-n_val:]
        train_actual = train_subjects[:-n_val] or train_subjects

        train_seqs, train_labels, norm_stats = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=train_actual,
            window_size=cfg_feat["window_size"], stride=2)
        val_seqs, val_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=val_subjects,
            window_size=cfg_feat["window_size"], stride=2, norm_stats=norm_stats)
        test_seqs, test_labels, _ = build_dataset_from_processed(
            processed_dir=urfd_dir, subject_ids=[test_subject],
            window_size=cfg_feat["window_size"], stride=5, norm_stats=norm_stats)

        if exclude_indices:
            train_seqs = _select_features(train_seqs, exclude_indices)
            val_seqs = _select_features(val_seqs, exclude_indices)
            test_seqs = _select_features(test_seqs, exclude_indices)

        train_ds = FallDetectionDataset(train_seqs, train_labels, augmenter=augmenter)
        val_ds = FallDetectionDataset(val_seqs, val_labels)
        test_ds = FallDetectionDataset(test_seqs, test_labels)

        model = FallDetectionLSTM(
            input_size=n_features, hidden_size=hidden_size,
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

        trainer.train(
            DataLoader(train_ds, batch_size=cfg_train["batch_size"], shuffle=True),
            DataLoader(val_ds, batch_size=cfg_train["batch_size"], shuffle=False))

        model.eval()
        fp, fpr = [], []
        with torch.no_grad():
            for x, _ in DataLoader(test_ds, batch_size=cfg_train["batch_size"], shuffle=False):
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fp.extend(logits.argmax(dim=1).numpy())
                fpr.extend(probs.numpy())

        evaluator = Evaluator(output_dir=Path("results/optimal_features"))
        fold_metrics = evaluator.compute_metrics(test_labels, np.array(fp), np.array(fpr))
        fold_aucs.append(fold_metrics["auc_roc"])

        all_preds.extend(fp)
        all_labels.extend(test_labels.tolist())
        all_probs.extend(fpr)

        logger.info(f"  {run_name} fold {fold_idx}: AUC={fold_metrics['auc_roc']:.3f}")

    agg = evaluator.compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    agg["fold_aucs"] = fold_aucs
    agg["mean_fold_auc"] = float(np.mean(fold_aucs))
    return agg


def main() -> None:
    config = _load_config()
    results_dir = Path(config["paths"]["results"]) / "optimal_features"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for name, cfg in CONFIGS.items():
        logger.info(f"\n=== {name}: {cfg['desc']} ===")
        t0 = time.time()
        result = run_loso_with_features(config, cfg["exclude"], name)
        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 1)
        result["desc"] = cfg["desc"]
        result["n_features"] = FeatureExtractor.NUM_FEATURES - len(cfg["exclude"])
        result["excluded"] = [FEATURE_NAMES[i] for i in cfg["exclude"]]
        all_results[name] = result

        print(
            f"\n{name} ({result['n_features']} feat): AUC={result['auc_roc']:.4f}, "
            f"sens={result['sensitivity']:.3f}, spec={result['specificity']:.3f} ({elapsed:.0f}s)"
        )

    # Save
    with open(results_dir / "optimal_features_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*70}")
    print("  OPTIMAL FEATURE SET COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Config':<18} {'#Feat':>5} {'AUC':>7} {'Sens':>6} {'Spec':>6} {'F1':>6}")
    print("  " + "-" * 52)
    for name, r in all_results.items():
        print(
            f"  {name:<18} {r['n_features']:>5} {r['auc_roc']:>7.4f} "
            f"{r['sensitivity']:>6.3f} {r['specificity']:>6.3f} {r['f1']:>6.3f}"
        )


if __name__ == "__main__":
    main()
