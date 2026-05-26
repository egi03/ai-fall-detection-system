"""
False alarm analysis: per-sequence breakdown of which ADL sequences cause false positives.

Uses Run 5 LOSO fold models to compute per-window predictions for each ADL sequence.
Identifies which sequences are most confused, analyzes feature distributions.

Reference: research/5.2.

Usage:
    python scripts/false_alarm_analysis.py
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.data_processing.splitter import SubjectSplitter
from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import FallDetectionDataset, build_dataset_from_processed, extract_windows
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_fold_model(
    fold_dir: Path, hidden_size: int, num_layers: int, bidirectional: bool
) -> FallDetectionLSTM:
    model = FallDetectionLSTM(
        input_size=FeatureExtractor.NUM_FEATURES,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        bidirectional=bidirectional,
    )
    ckpt = torch.load(str(fold_dir / "best_model.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def per_sequence_analysis(
    config: dict,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """
    Compute per-sequence false alarm rates using LOSO fold models.

    For each ADL sequence belonging to the held-out subject in each fold,
    run inference and compute: FP rate, max probability, mean probability.

    Also compute per-fall-sequence miss rates (false negatives).
    """
    cfg_model = config["model"]
    cfg_feat = config["features"]
    cfg_train = config["training"]

    hidden_size = cfg_model["hidden_size"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)
    window_size = cfg_feat["window_size"]

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    urfd_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    splitter = SubjectSplitter(urfd_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    output_dir.mkdir(parents=True, exist_ok=True)

    adl_rows = []
    fall_rows = []

    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)
    dt = 1.0 / 15.0  # target_fps

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        train_subjects = fold["train"]

        fold_dir = model_dir / f"fold_{fold_idx}"
        if not (fold_dir / "best_model.pth").exists():
            logger.warning(f"No model for fold {fold_idx}, skipping")
            continue

        # Load norm stats and model
        norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
        norm_std = np.load(str(fold_dir / "norm_std.npy"))
        model = _load_fold_model(fold_dir, hidden_size, num_layers, bidirectional)

        # Process each sequence for this test subject individually
        for seq_meta in meta["sequences"]:
            if seq_meta["subject_id"] != test_subject:
                continue

            seq_id = seq_meta["sequence_id"]
            label = seq_meta["label"]

            kp_path = urfd_dir / "keypoints_raw" / f"{seq_id}.npy"
            if not kp_path.exists():
                continue

            keypoints = np.load(str(kp_path))
            feature_seq = extractor.extract_sequence(keypoints, dt=dt)
            feature_seq = np.nan_to_num(feature_seq, nan=0.0)

            # Extract windows
            feature_3d = feature_seq[:, :, np.newaxis]
            windows, window_labels = extract_windows(
                feature_3d, label, window_size=window_size, stride=5
            )

            if not windows:
                continue

            # Reshape and normalize
            seqs = np.array([w.reshape(w.shape[0], -1) for w in windows], dtype=np.float32)
            seqs = (seqs - norm_mean) / norm_std

            # Inference
            ds = FallDetectionDataset(seqs, np.array(window_labels))
            loader = DataLoader(ds, batch_size=cfg_train["batch_size"], shuffle=False)

            all_probs = []
            all_preds = []
            with torch.no_grad():
                for x, _ in loader:
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    all_probs.extend(probs.numpy())
                    all_preds.extend(logits.argmax(dim=1).numpy())

            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)
            window_labels_arr = np.array(window_labels)

            n_windows = len(all_probs)
            mean_prob = float(all_probs.mean())
            max_prob = float(all_probs.max())
            min_prob = float(all_probs.min())

            if label == 0:  # ADL
                fp_count = int(all_preds.sum())
                fp_rate = fp_count / n_windows if n_windows > 0 else 0
                adl_rows.append({
                    "sequence_id": seq_id,
                    "subject": test_subject,
                    "n_windows": n_windows,
                    "n_frames": seq_meta["num_frames"],
                    "fp_count": fp_count,
                    "fp_rate": round(fp_rate, 4),
                    "mean_fall_prob": round(mean_prob, 4),
                    "max_fall_prob": round(max_prob, 4),
                    "min_fall_prob": round(min_prob, 4),
                })
            else:  # Fall
                fn_count = int((all_preds == 0).sum())
                fn_rate = fn_count / n_windows if n_windows > 0 else 0
                tp_count = int(all_preds.sum())
                fall_rows.append({
                    "sequence_id": seq_id,
                    "subject": test_subject,
                    "n_windows": n_windows,
                    "n_frames": seq_meta["num_frames"],
                    "tp_count": tp_count,
                    "fn_count": fn_count,
                    "miss_rate": round(fn_rate, 4),
                    "mean_fall_prob": round(mean_prob, 4),
                    "max_fall_prob": round(max_prob, 4),
                    "min_fall_prob": round(min_prob, 4),
                })

    # Sort ADLs by FP rate descending
    adl_rows.sort(key=lambda x: x["fp_rate"], reverse=True)
    fall_rows.sort(key=lambda x: x["miss_rate"], reverse=True)

    # Save ADL false alarm CSV
    adl_fields = ["sequence_id", "subject", "n_windows", "n_frames",
                  "fp_count", "fp_rate", "mean_fall_prob", "max_fall_prob", "min_fall_prob"]
    with open(output_dir / "adl_false_alarms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adl_fields)
        w.writeheader()
        w.writerows(adl_rows)

    # Save fall miss CSV
    fall_fields = ["sequence_id", "subject", "n_windows", "n_frames",
                   "tp_count", "fn_count", "miss_rate", "mean_fall_prob", "max_fall_prob", "min_fall_prob"]
    with open(output_dir / "fall_misses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fall_fields)
        w.writeheader()
        w.writerows(fall_rows)

    # Per-subject summary
    subj_adl_stats = {}
    for row in adl_rows:
        s = row["subject"]
        if s not in subj_adl_stats:
            subj_adl_stats[s] = {"total_windows": 0, "total_fp": 0, "n_sequences": 0}
        subj_adl_stats[s]["total_windows"] += row["n_windows"]
        subj_adl_stats[s]["total_fp"] += row["fp_count"]
        subj_adl_stats[s]["n_sequences"] += 1

    subj_fall_stats = {}
    for row in fall_rows:
        s = row["subject"]
        if s not in subj_fall_stats:
            subj_fall_stats[s] = {"total_windows": 0, "total_fn": 0, "n_sequences": 0}
        subj_fall_stats[s]["total_windows"] += row["n_windows"]
        subj_fall_stats[s]["total_fn"] += row["fn_count"]
        subj_fall_stats[s]["n_sequences"] += 1

    # Print summary
    print("\n" + "=" * 70)
    print("  FALSE ALARM ANALYSIS")
    print("=" * 70)

    print("\n  ADL Sequences — False Positive Rate (sorted worst first):")
    print(f"  {'Sequence':<25} {'Subject':>7} {'Windows':>7} {'FP':>4} {'FP Rate':>8} {'Max P':>7}")
    print("  " + "-" * 60)
    for row in adl_rows[:15]:  # top 15
        print(
            f"  {row['sequence_id']:<25} {row['subject']:>7} {row['n_windows']:>7} "
            f"{row['fp_count']:>4} {row['fp_rate']:>8.3f} {row['max_fall_prob']:>7.3f}"
        )

    print(f"\n  Per-subject ADL false alarm rates:")
    for s in sorted(subj_adl_stats):
        st = subj_adl_stats[s]
        rate = st["total_fp"] / st["total_windows"] if st["total_windows"] > 0 else 0
        print(f"    {s}: {st['total_fp']}/{st['total_windows']} windows = {rate:.3f} FP rate")

    print(f"\n  Fall Sequences — Miss Rate (sorted worst first):")
    print(f"  {'Sequence':<25} {'Subject':>7} {'Windows':>7} {'FN':>4} {'Miss':>8} {'Mean P':>7}")
    print("  " + "-" * 60)
    for row in fall_rows[:15]:
        print(
            f"  {row['sequence_id']:<25} {row['subject']:>7} {row['n_windows']:>7} "
            f"{row['fn_count']:>4} {row['miss_rate']:>8.3f} {row['mean_fall_prob']:>7.3f}"
        )

    print(f"\n  Per-subject fall miss rates:")
    for s in sorted(subj_fall_stats):
        st = subj_fall_stats[s]
        rate = st["total_fn"] / st["total_windows"] if st["total_windows"] > 0 else 0
        print(f"    {s}: {st['total_fn']}/{st['total_windows']} windows = {rate:.3f} miss rate")

    # Summary stats
    total_adl_windows = sum(r["n_windows"] for r in adl_rows)
    total_fp = sum(r["fp_count"] for r in adl_rows)
    total_fall_windows = sum(r["n_windows"] for r in fall_rows)
    total_fn = sum(r["fn_count"] for r in fall_rows)
    n_perfect_adl = sum(1 for r in adl_rows if r["fp_count"] == 0)
    n_perfect_fall = sum(1 for r in fall_rows if r["fn_count"] == 0)

    summary = {
        "total_adl_sequences": len(adl_rows),
        "total_fall_sequences": len(fall_rows),
        "total_adl_windows": total_adl_windows,
        "total_fall_windows": total_fall_windows,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "overall_fp_rate": round(total_fp / total_adl_windows, 4) if total_adl_windows > 0 else 0,
        "overall_miss_rate": round(total_fn / total_fall_windows, 4) if total_fall_windows > 0 else 0,
        "perfect_adl_sequences": n_perfect_adl,
        "perfect_fall_sequences": n_perfect_fall,
        "worst_adl_sequence": adl_rows[0]["sequence_id"] if adl_rows else "N/A",
        "worst_adl_fp_rate": adl_rows[0]["fp_rate"] if adl_rows else 0,
        "worst_fall_sequence": fall_rows[0]["sequence_id"] if fall_rows else "N/A",
        "worst_fall_miss_rate": fall_rows[0]["miss_rate"] if fall_rows else 0,
    }
    with open(output_dir / "false_alarm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary:")
    print(f"    Overall ADL FP rate: {summary['overall_fp_rate']:.3f} ({total_fp}/{total_adl_windows})")
    print(f"    Overall fall miss rate: {summary['overall_miss_rate']:.3f} ({total_fn}/{total_fall_windows})")
    print(f"    Perfect ADL sequences: {n_perfect_adl}/{len(adl_rows)}")
    print(f"    Perfect fall sequences: {n_perfect_fall}/{len(fall_rows)}")


def feature_distribution_analysis(
    config: dict,
    output_dir: Path,
) -> None:
    """
    Compare feature distributions between fall and ADL windows.
    Generates a per-feature violin/box plot and correlation matrix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg_feat = config["features"]
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    with open(urfd_dir / "metadata.json") as f:
        meta = json.load(f)
    all_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    # Build full dataset (unnormalized for distribution analysis)
    seqs, labels, _ = build_dataset_from_processed(
        processed_dir=urfd_dir,
        subject_ids=all_subjects,
        window_size=cfg_feat["window_size"],
        stride=2,
        apply_norm=False,
    )

    feature_names = [
        "torso_incl", "hip_sh_angle", "bbox_ar", "com_vel", "com_accel",
        "head_toe_d", "sh_ankle_d", "l_knee_a", "r_knee_a",
        "l_hip_a", "r_hip_a", "wrist_hip_d", "body_spread",
        "mean_vis", "min_vis",
    ]

    # Compute per-window mean features
    fall_mask = labels == 1
    adl_mask = labels == 0
    fall_means = seqs[fall_mask].mean(axis=1)  # (N_fall, 15)
    adl_means = seqs[adl_mask].mean(axis=1)    # (N_adl, 15)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Box plot comparison
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        data = [adl_means[:, i], fall_means[:, i]]
        bp = ax.boxplot(data, labels=["ADL", "Fall"], patch_artist=True,
                       widths=0.6, showfliers=False)
        bp["boxes"][0].set_facecolor("#4caf50")
        bp["boxes"][1].set_facecolor("#d32f2f")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)

    plt.suptitle("Feature Distributions: ADL vs Fall (per-window mean)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(output_dir / "feature_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Compute effect sizes (Cohen's d) for each feature
    effect_sizes = []
    for i in range(15):
        adl_m, adl_s = adl_means[:, i].mean(), adl_means[:, i].std()
        fall_m, fall_s = fall_means[:, i].mean(), fall_means[:, i].std()
        pooled_std = np.sqrt((adl_s**2 + fall_s**2) / 2) + 1e-8
        d = (fall_m - adl_m) / pooled_std
        effect_sizes.append({
            "feature": feature_names[i],
            "index": i,
            "adl_mean": round(float(adl_m), 4),
            "fall_mean": round(float(fall_m), 4),
            "cohens_d": round(float(d), 4),
            "abs_d": round(abs(float(d)), 4),
        })

    effect_sizes.sort(key=lambda x: x["abs_d"], reverse=True)
    with open(output_dir / "feature_effect_sizes.json", "w") as f:
        json.dump(effect_sizes, f, indent=2)

    # Effect size bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    names = [e["feature"] for e in effect_sizes]
    ds = [e["cohens_d"] for e in effect_sizes]
    colors = ["#d32f2f" if abs(d) > 0.5 else "#ff9800" if abs(d) > 0.2 else "#4caf50" for d in ds]
    ax.barh(names, ds, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (Fall - ADL)", fontsize=11)
    ax.set_title("Feature Effect Sizes: Fall vs ADL", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(output_dir / "feature_effect_sizes.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\n  Feature effect sizes (|Cohen's d|, sorted):")
    for e in effect_sizes:
        impact = "large" if e["abs_d"] > 0.8 else "medium" if e["abs_d"] > 0.5 else "small" if e["abs_d"] > 0.2 else "negligible"
        print(f"    {e['feature']:<15} d={e['cohens_d']:>+7.3f}  ({impact})")


def main() -> None:
    config = _load_config()

    # Use Run 5 models
    model_dir = Path(config["paths"]["models"]) / "run5_stride2_aug"
    output_dir = Path(config["paths"]["results"]) / "false_alarm_analysis"

    print("=== Phase 1: Per-Sequence False Alarm Analysis ===")
    per_sequence_analysis(config, model_dir, output_dir)

    print("\n=== Phase 2: Feature Distribution Analysis ===")
    feature_distribution_analysis(config, output_dir)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
