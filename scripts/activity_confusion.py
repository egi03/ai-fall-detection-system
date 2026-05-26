"""
Per-sequence activity confusion analysis for URFD dataset.

Loads Run 5 LOSO fold models and evaluates each test sequence individually
to identify which ADL sequences cause the most false alarms and which
fall sequences are hardest to detect.

Produces:
- Per-sequence classification table (mean probability, fall window fraction)
- Identification of hardest ADL and fall sequences
- Visualization of per-sequence difficulty sorted by probability

Reference: research/5.2 - False Alarm Analysis, research/10.3 - Error Analysis.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import extract_windows

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PROJECT_ROOT = Path(__file__).parent.parent
THRESHOLD = 0.5  # classification threshold


def load_config() -> dict:
    """Load project config."""
    with open(PROJECT_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_fold_model(fold_dir: Path) -> tuple:
    """Load model and normalization stats from a fold directory."""
    model = FallDetectionLSTM(
        input_size=15,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,  # no dropout at inference
        bidirectional=True,
    )
    ckpt = torch.load(
        str(fold_dir / "best_model.pth"), map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    mean = np.load(str(fold_dir / "norm_mean.npy"))
    std = np.load(str(fold_dir / "norm_std.npy"))

    return model, mean, std


def extract_features_for_sequence(
    npy_path: Path,
    label: int,
    annotation: dict,
    window_size: int = 30,
    stride: int = 2,
    positive_threshold: float = 0.5,
) -> tuple:
    """
    Load a single sequence, compute features, extract windows.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        feature_windows of shape (N, window_size, 15), window_labels of shape (N,).
        Returns (None, None) if no windows can be extracted.
    """
    keypoints = np.load(str(npy_path))

    extractor = FeatureExtractor(
        keypoint_format="mediapipe",
        confidence_threshold=0.5,
    )
    dt = 1.0 / 15.0
    feature_seq = extractor.extract_sequence(keypoints, dt=dt)
    feature_seq = np.nan_to_num(feature_seq, nan=0.0)

    # Reshape to (T, 15, 1) for extract_windows
    feature_3d = feature_seq[:, :, np.newaxis]
    windows, labels = extract_windows(
        feature_3d,
        label,
        window_size=window_size,
        stride=stride,
        annotation=annotation if annotation else None,
        positive_threshold=positive_threshold,
    )

    if not windows:
        return None, None

    # Squeeze trailing dim
    sequences = []
    for w in windows:
        if w.ndim == 3:
            sequences.append(w.reshape(w.shape[0], -1))
        else:
            sequences.append(w)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def predict_sequence(
    model: torch.nn.Module,
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple:
    """
    Run inference on windows from a single sequence.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        predictions (0/1) and fall probabilities for each window.
    """
    # Normalize
    windows_norm = (windows - mean) / std
    x = torch.from_numpy(windows_norm.astype(np.float32))

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()

    return preds, probs


def main() -> None:
    config = load_config()
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    models_dir = Path(config["paths"]["models"]) / "run5_stride2_aug"
    kp_dir = urfd_dir / "keypoints_raw"

    # Load metadata
    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Build subject -> fold mapping
    subjects = sorted(set(s["subject_id"] for s in metadata["sequences"]))
    # fold_i tests subject_i (S01..S05 -> fold_0..fold_4)
    subject_to_fold = {subj: i for i, subj in enumerate(subjects)}

    print("=" * 80)
    print("PER-SEQUENCE ACTIVITY CONFUSION ANALYSIS (Run 5, URFD LOSO)")
    print("=" * 80)

    # Results storage
    results = []

    # Process each sequence
    for seq_meta in metadata["sequences"]:
        seq_id = seq_meta["sequence_id"]
        label = seq_meta["label"]
        subject_id = seq_meta["subject_id"]
        num_frames = seq_meta["num_frames"]

        npy_path = kp_dir / f"{seq_id}.npy"
        if not npy_path.exists():
            print(f"  SKIP (missing): {seq_id}")
            continue

        # Get the fold where this subject is in the test set
        fold_idx = subject_to_fold[subject_id]
        fold_dir = models_dir / f"fold_{fold_idx}"

        # Load fold model
        model, mean, std = load_fold_model(fold_dir)

        # Build annotation dict
        annotation = {}
        if "start_frame" in seq_meta:
            annotation["start_frame"] = seq_meta.get("start_frame", 0)
            annotation["end_frame"] = seq_meta.get("end_frame", num_frames)

        # Extract features and windows
        windows, window_labels = extract_features_for_sequence(
            npy_path, label, annotation, window_size=30, stride=2
        )

        if windows is None:
            print(f"  SKIP (no windows): {seq_id}")
            continue

        # Predict
        preds, probs = predict_sequence(model, windows, mean, std)

        # Compute per-sequence stats
        mean_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        min_prob = float(np.min(probs))
        fall_window_fraction = float(np.mean(preds == 1))
        n_windows = len(windows)

        # Sequence-level classification: majority vote
        seq_pred = 1 if fall_window_fraction >= 0.5 else 0
        # Alternative: any window above threshold
        any_fall = 1 if np.any(probs >= THRESHOLD) else 0

        results.append({
            "sequence_id": seq_id,
            "label": label,
            "label_str": "FALL" if label == 1 else "ADL",
            "subject_id": subject_id,
            "fold": fold_idx,
            "num_frames": num_frames,
            "n_windows": n_windows,
            "mean_prob": mean_prob,
            "max_prob": max_prob,
            "min_prob": min_prob,
            "fall_window_frac": fall_window_fraction,
            "seq_pred_majority": seq_pred,
            "any_fall_window": any_fall,
        })

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    adl_results = [r for r in results if r["label"] == 0]
    fall_results = [r for r in results if r["label"] == 1]

    # Sort by difficulty
    # ADL: sorted by mean_prob descending (highest = most confusing)
    adl_sorted = sorted(adl_results, key=lambda r: r["mean_prob"], reverse=True)
    # Fall: sorted by mean_prob ascending (lowest = hardest to detect)
    fall_sorted = sorted(fall_results, key=lambda r: r["mean_prob"])

    print("\n")
    print("=" * 80)
    print("ADL SEQUENCES (sorted by mean fall probability - highest = most confusing)")
    print("=" * 80)
    print(f"{'Sequence':<25} {'Subj':>5} {'Frames':>6} {'Windows':>7} "
          f"{'MeanP':>7} {'MaxP':>7} {'FallFrac':>8} {'FalseAlarm':>10}")
    print("-" * 80)
    for r in adl_sorted:
        fa_str = "YES" if r["any_fall_window"] else "no"
        print(f"{r['sequence_id']:<25} {r['subject_id']:>5} {r['num_frames']:>6} "
              f"{r['n_windows']:>7} {r['mean_prob']:>7.3f} {r['max_prob']:>7.3f} "
              f"{r['fall_window_frac']:>8.3f} {fa_str:>10}")

    # ADL false alarm summary
    n_adl_false_alarm = sum(1 for r in adl_results if r["any_fall_window"])
    print(f"\nADL False Alarm Summary: {n_adl_false_alarm}/{len(adl_results)} "
          f"sequences have at least one window classified as fall "
          f"({100*n_adl_false_alarm/max(1,len(adl_results)):.1f}%)")

    print("\n")
    print("=" * 80)
    print("FALL SEQUENCES (sorted by mean fall probability - lowest = hardest to detect)")
    print("=" * 80)
    print(f"{'Sequence':<25} {'Subj':>5} {'Frames':>6} {'Windows':>7} "
          f"{'MeanP':>7} {'MaxP':>7} {'FallFrac':>8} {'Detected':>10}")
    print("-" * 80)
    for r in fall_sorted:
        det_str = "YES" if r["any_fall_window"] else "MISSED"
        print(f"{r['sequence_id']:<25} {r['subject_id']:>5} {r['num_frames']:>6} "
              f"{r['n_windows']:>7} {r['mean_prob']:>7.3f} {r['max_prob']:>7.3f} "
              f"{r['fall_window_frac']:>8.3f} {det_str:>10}")

    # Fall detection summary
    n_fall_detected = sum(1 for r in fall_results if r["any_fall_window"])
    n_fall_missed = len(fall_results) - n_fall_detected
    print(f"\nFall Detection Summary: {n_fall_detected}/{len(fall_results)} "
          f"sequences detected ({100*n_fall_detected/max(1,len(fall_results)):.1f}%), "
          f"{n_fall_missed} missed")

    # -----------------------------------------------------------------------
    # Per-subject breakdown
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("PER-SUBJECT BREAKDOWN")
    print("=" * 80)
    for subj in subjects:
        subj_adl = [r for r in adl_results if r["subject_id"] == subj]
        subj_fall = [r for r in fall_results if r["subject_id"] == subj]

        adl_fa = sum(1 for r in subj_adl if r["any_fall_window"])
        fall_det = sum(1 for r in subj_fall if r["any_fall_window"])

        adl_mean_p = np.mean([r["mean_prob"] for r in subj_adl]) if subj_adl else 0
        fall_mean_p = np.mean([r["mean_prob"] for r in subj_fall]) if subj_fall else 0

        print(f"  {subj} (fold_{subject_to_fold[subj]}):")
        if subj_adl:
            print(f"    ADL:  {len(subj_adl)} seqs, mean_prob={adl_mean_p:.3f}, "
                  f"false alarms={adl_fa}/{len(subj_adl)}")
        if subj_fall:
            print(f"    Fall: {len(subj_fall)} seqs, mean_prob={fall_mean_p:.3f}, "
                  f"detected={fall_det}/{len(subj_fall)}")

    # -----------------------------------------------------------------------
    # Camera view analysis (cam0 vs cam1)
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("CAMERA VIEW ANALYSIS (Fall sequences only)")
    print("=" * 80)
    cam0_falls = [r for r in fall_results if "cam0" in r["sequence_id"]]
    cam1_falls = [r for r in fall_results if "cam1" in r["sequence_id"]]

    if cam0_falls:
        cam0_mean = np.mean([r["mean_prob"] for r in cam0_falls])
        cam0_det = sum(1 for r in cam0_falls if r["any_fall_window"])
        print(f"  cam0: {len(cam0_falls)} seqs, mean_prob={cam0_mean:.3f}, "
              f"detected={cam0_det}/{len(cam0_falls)} "
              f"({100*cam0_det/len(cam0_falls):.1f}%)")

    if cam1_falls:
        cam1_mean = np.mean([r["mean_prob"] for r in cam1_falls])
        cam1_det = sum(1 for r in cam1_falls if r["any_fall_window"])
        print(f"  cam1: {len(cam1_falls)} seqs, mean_prob={cam1_mean:.3f}, "
              f"detected={cam1_det}/{len(cam1_falls)} "
              f"({100*cam1_det/len(cam1_falls):.1f}%)")

    # -----------------------------------------------------------------------
    # ADL index grouping analysis
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("ADL INDEX GROUP ANALYSIS")
    print("=" * 80)
    # Group ADLs by index ranges (1-8, 9-16, 17-24, 25-32, 33-40)
    adl_groups = {
        "adl-01..08 (S01)": (1, 8),
        "adl-09..16 (S02)": (9, 16),
        "adl-17..24 (S03)": (17, 24),
        "adl-25..32 (S04)": (25, 32),
        "adl-33..40 (S05)": (33, 40),
    }

    for group_name, (lo, hi) in adl_groups.items():
        group_seqs = []
        for r in adl_results:
            # Extract index number from sequence_id
            idx_str = r["sequence_id"].split("adl-")[1].split("_")[0]
            idx = int(idx_str)
            if lo <= idx <= hi:
                group_seqs.append(r)
        if group_seqs:
            gp_mean = np.mean([r["mean_prob"] for r in group_seqs])
            gp_fa = sum(1 for r in group_seqs if r["any_fall_window"])
            print(f"  {group_name}: {len(group_seqs)} seqs, "
                  f"mean_prob={gp_mean:.3f}, false_alarms={gp_fa}/{len(group_seqs)}")

    # -----------------------------------------------------------------------
    # Fall index grouping analysis
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("FALL INDEX GROUP ANALYSIS")
    print("=" * 80)
    fall_groups = {
        "fall-01..06 (S01)": (1, 6),
        "fall-07..12 (S02)": (7, 12),
        "fall-13..18 (S03)": (13, 18),
        "fall-19..24 (S04)": (19, 24),
        "fall-25..30 (S05)": (25, 30),
    }

    for group_name, (lo, hi) in fall_groups.items():
        group_seqs = []
        for r in fall_results:
            idx_str = r["sequence_id"].split("fall-")[1].split("_")[0]
            idx = int(idx_str)
            if lo <= idx <= hi:
                group_seqs.append(r)
        if group_seqs:
            gp_mean = np.mean([r["mean_prob"] for r in group_seqs])
            gp_det = sum(1 for r in group_seqs if r["any_fall_window"])
            print(f"  {group_name}: {len(group_seqs)} seqs, "
                  f"mean_prob={gp_mean:.3f}, detected={gp_det}/{len(group_seqs)}")

    # -----------------------------------------------------------------------
    # Top-5 hardest cases
    # -----------------------------------------------------------------------
    print("\n")
    print("=" * 80)
    print("TOP-5 MOST CONFUSING ADL SEQUENCES (highest false alarm risk)")
    print("=" * 80)
    for i, r in enumerate(adl_sorted[:5]):
        print(f"  {i+1}. {r['sequence_id']} (subj={r['subject_id']}): "
              f"mean_prob={r['mean_prob']:.3f}, max_prob={r['max_prob']:.3f}, "
              f"fall_frac={r['fall_window_frac']:.3f}")

    print("\n")
    print("=" * 80)
    print("TOP-5 HARDEST FALL SEQUENCES (lowest detection probability)")
    print("=" * 80)
    for i, r in enumerate(fall_sorted[:5]):
        print(f"  {i+1}. {r['sequence_id']} (subj={r['subject_id']}): "
              f"mean_prob={r['mean_prob']:.3f}, max_prob={r['max_prob']:.3f}, "
              f"fall_frac={r['fall_window_frac']:.3f}")

    # -----------------------------------------------------------------------
    # Visualization: per-sequence difficulty plot
    # -----------------------------------------------------------------------
    plot_dir = Path(config["paths"]["results"]) / "analysis_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # --- ADL subplot ---
    ax = axes[0]
    adl_names = [r["sequence_id"].replace("urfd_", "") for r in adl_sorted]
    adl_probs = [r["mean_prob"] for r in adl_sorted]
    adl_maxps = [r["max_prob"] for r in adl_sorted]
    colors_adl = ["#e74c3c" if r["any_fall_window"] else "#2ecc71" for r in adl_sorted]

    x_pos = np.arange(len(adl_names))
    bars = ax.bar(x_pos, adl_probs, color=colors_adl, alpha=0.7, label="Mean prob")
    ax.scatter(x_pos, adl_maxps, color="black", marker="v", s=20, zorder=5,
               label="Max prob")
    ax.axhline(y=THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({THRESHOLD})")
    ax.set_ylabel("Fall Probability", fontsize=12)
    ax.set_title("ADL Sequences - Sorted by Mean Fall Probability (Descending)\n"
                 "Red = has false alarm windows, Green = correctly classified",
                 fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(adl_names, rotation=90, fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # --- Fall subplot ---
    ax = axes[1]
    fall_names = [r["sequence_id"].replace("urfd_", "") for r in fall_sorted]
    fall_probs_mean = [r["mean_prob"] for r in fall_sorted]
    fall_maxps = [r["max_prob"] for r in fall_sorted]
    colors_fall = ["#e74c3c" if not r["any_fall_window"] else "#2ecc71"
                   for r in fall_sorted]

    x_pos = np.arange(len(fall_names))
    bars = ax.bar(x_pos, fall_probs_mean, color=colors_fall, alpha=0.7,
                  label="Mean prob")
    ax.scatter(x_pos, fall_maxps, color="black", marker="^", s=20, zorder=5,
               label="Max prob")
    ax.axhline(y=THRESHOLD, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold ({THRESHOLD})")
    ax.set_ylabel("Fall Probability", fontsize=12)
    ax.set_title("Fall Sequences - Sorted by Mean Fall Probability (Ascending)\n"
                 "Red = missed, Green = correctly detected",
                 fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(fall_names, rotation=90, fontsize=7)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = plot_dir / "per_sequence_difficulty.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {plot_path}")

    # -----------------------------------------------------------------------
    # Save results JSON
    # -----------------------------------------------------------------------
    results_json_path = plot_dir / "per_sequence_results.json"
    with open(str(results_json_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    print(f"Results JSON saved to: {results_json_path}")

    # -----------------------------------------------------------------------
    # Generate markdown log
    # -----------------------------------------------------------------------
    log_dir = PROJECT_ROOT / "DAILY LOGS" / "15.3"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "ACTIVITY_CONFUSION.md"

    with open(str(log_path), "w", encoding="utf-8") as f:
        f.write("# Per-Sequence Activity Confusion Analysis\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Model: Run 5 (BiLSTM, stride=2, augmented)\n")
        f.write(f"- Dataset: URFD, 5-fold LOSO\n")
        f.write(f"- Classification threshold: {THRESHOLD}\n")
        f.write(f"- Total sequences analyzed: {len(results)}\n")
        f.write(f"  - ADL: {len(adl_results)}, Fall: {len(fall_results)}\n\n")

        f.write("## ADL False Alarm Summary\n\n")
        f.write(f"- {n_adl_false_alarm}/{len(adl_results)} ADL sequences have "
                f"at least one window classified as fall "
                f"({100*n_adl_false_alarm/max(1,len(adl_results)):.1f}%)\n\n")

        f.write("### Top-5 Most Confusing ADL Sequences\n\n")
        f.write("| Rank | Sequence | Subject | Mean Prob | Max Prob | Fall Frac |\n")
        f.write("|------|----------|---------|-----------|----------|-----------|\n")
        for i, r in enumerate(adl_sorted[:5]):
            f.write(f"| {i+1} | {r['sequence_id']} | {r['subject_id']} | "
                    f"{r['mean_prob']:.3f} | {r['max_prob']:.3f} | "
                    f"{r['fall_window_frac']:.3f} |\n")

        f.write("\n## Fall Detection Summary\n\n")
        f.write(f"- {n_fall_detected}/{len(fall_results)} fall sequences detected "
                f"({100*n_fall_detected/max(1,len(fall_results)):.1f}%)\n")
        f.write(f"- {n_fall_missed} fall sequences completely missed\n\n")

        f.write("### Top-5 Hardest Fall Sequences\n\n")
        f.write("| Rank | Sequence | Subject | Mean Prob | Max Prob | Fall Frac |\n")
        f.write("|------|----------|---------|-----------|----------|-----------|\n")
        for i, r in enumerate(fall_sorted[:5]):
            f.write(f"| {i+1} | {r['sequence_id']} | {r['subject_id']} | "
                    f"{r['mean_prob']:.3f} | {r['max_prob']:.3f} | "
                    f"{r['fall_window_frac']:.3f} |\n")

        f.write("\n## Camera View Analysis\n\n")
        if cam0_falls:
            cam0_mean = np.mean([r["mean_prob"] for r in cam0_falls])
            cam0_det = sum(1 for r in cam0_falls if r["any_fall_window"])
            f.write(f"- cam0: {len(cam0_falls)} seqs, mean_prob={cam0_mean:.3f}, "
                    f"detected={cam0_det}/{len(cam0_falls)}\n")
        if cam1_falls:
            cam1_mean = np.mean([r["mean_prob"] for r in cam1_falls])
            cam1_det = sum(1 for r in cam1_falls if r["any_fall_window"])
            f.write(f"- cam1: {len(cam1_falls)} seqs, mean_prob={cam1_mean:.3f}, "
                    f"detected={cam1_det}/{len(cam1_falls)}\n")

        f.write("\n## Per-Subject Breakdown\n\n")
        for subj in subjects:
            subj_adl = [r for r in adl_results if r["subject_id"] == subj]
            subj_fall = [r for r in fall_results if r["subject_id"] == subj]
            adl_fa = sum(1 for r in subj_adl if r["any_fall_window"])
            fall_det = sum(1 for r in subj_fall if r["any_fall_window"])
            adl_mp = np.mean([r["mean_prob"] for r in subj_adl]) if subj_adl else 0
            fall_mp = np.mean([r["mean_prob"] for r in subj_fall]) if subj_fall else 0
            f.write(f"### {subj} (fold_{subject_to_fold[subj]})\n")
            if subj_adl:
                f.write(f"- ADL: {len(subj_adl)} seqs, mean_prob={adl_mp:.3f}, "
                        f"false alarms={adl_fa}/{len(subj_adl)}\n")
            if subj_fall:
                f.write(f"- Fall: {len(subj_fall)} seqs, mean_prob={fall_mp:.3f}, "
                        f"detected={fall_det}/{len(subj_fall)}\n")
            f.write("\n")

        f.write("## Plot\n\n")
        f.write("![Per-Sequence Difficulty](../../results/analysis_plots/"
                "per_sequence_difficulty.png)\n")

    print(f"Log saved to: {log_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
