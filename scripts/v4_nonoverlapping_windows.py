"""
Non-overlapping window AUC computation for BiLSTM and Transformer-LSTM.

Addresses the reviewer concern that overlapping windows inflate AUC:
"Overlapping windows with stride=2 create highly correlated samples
that may bias AUC upward. Please report AUC using non-overlapping
windows (stride=window_size) to confirm robustness."

Method: For each LOSO test fold, load the existing trained fold model
(no re-training). Run inference using stride=30 (fully non-overlapping,
since window_size=30). Aggregate per-window probabilities into a
per-sequence probability using max pooling, then compute sequence-level
AUC. Compare to v3 sequence-level AUC (stride=2, max-pool: 0.840/0.836)
to demonstrate that overlapping windows did not inflate results.

Usage:
    python scripts/v4_nonoverlapping_windows.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data_processing.splitter import SubjectSplitter
from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.models.transformer_lstm import FallDetectionTransformerLSTM
from src.training.dataset import FallDetectionDataset, extract_windows

SEED = 42
WINDOW_SIZE = 30
# Non-overlapping: stride equals window size → zero overlap
STRIDE_NONOVERLAPPING = 30
TARGET_FPS = 15.0

# Reference values from v3 sequence-level AUC (stride=2, max-pool)
V3_REFERENCE = {
    "BiLSTM": 0.840,
    "Transformer-LSTM": 0.836,
}


def load_bilstm(fold_dir: Path, cfg_model: dict, n_features: int) -> FallDetectionLSTM:
    """
    Load a trained BiLSTM model from a fold checkpoint.

    Parameters
    ----------
    fold_dir : Path
        Directory containing best_model.pth.
    cfg_model : dict
        Model hyperparameter dict from config.yaml.
    n_features : int
        Number of input features per frame.

    Returns
    -------
    FallDetectionLSTM
        Loaded model in eval mode.
    """
    model = FallDetectionLSTM(
        input_size=n_features,
        hidden_size=cfg_model["hidden_size"],
        num_layers=cfg_model["num_layers"],
        dropout=0.0,  # disable dropout at inference
        bidirectional=cfg_model.get("bidirectional", False),
    )
    ckpt = torch.load(
        str(fold_dir / "best_model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def load_tl(fold_dir: Path, cfg_tl: dict, n_features: int) -> FallDetectionTransformerLSTM:
    """
    Load a trained Transformer-LSTM model from a fold checkpoint.

    Parameters
    ----------
    fold_dir : Path
        Directory containing best_model.pth.
    cfg_tl : dict
        Transformer-LSTM hyperparameter dict from config.yaml.
    n_features : int
        Number of input features per frame.

    Returns
    -------
    FallDetectionTransformerLSTM
        Loaded model in eval mode.
    """
    model = FallDetectionTransformerLSTM(
        input_size=n_features,
        d_model=cfg_tl.get("d_model", 64),
        nhead=cfg_tl.get("nhead", 4),
        num_encoder_layers=cfg_tl.get("num_encoder_layers", 2),
        dim_feedforward=cfg_tl.get("dim_feedforward", 128),
        lstm_hidden_size=cfg_tl.get("lstm_hidden_size", 128),
        lstm_num_layers=cfg_tl.get("lstm_num_layers", 1),
        dropout=0.0,  # disable dropout at inference
    )
    ckpt = torch.load(
        str(fold_dir / "best_model.pth"),
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def infer_sequence(
    net: torch.nn.Module,
    keypoints: np.ndarray,
    label: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    extractor: FeatureExtractor,
    dt: float,
    stride: int,
) -> dict:
    """
    Run model inference on a single keypoint sequence.

    Extracts feature windows with the specified stride, normalises with
    training-set statistics, runs forward pass, and returns per-window
    probabilities aggregated to sequence level via max pooling.

    Parameters
    ----------
    net : torch.nn.Module
        Loaded model in eval mode.
    keypoints : np.ndarray
        Raw keypoint array of shape (T, N, C).
    label : int
        Ground-truth sequence label (0=ADL, 1=fall).
    norm_mean : np.ndarray
        Per-feature mean from the training fold.
    norm_std : np.ndarray
        Per-feature std from the training fold.
    extractor : FeatureExtractor
        Feature extractor instance.
    dt : float
        Frame time step (1 / target_fps).
    stride : int
        Window stride in frames.

    Returns
    -------
    dict
        Keys: label, n_windows, seq_prob_max, seq_prob_mean, window_probs.
        Returns None if fewer than one window can be extracted.
    """
    feature_seq = extractor.extract_sequence(keypoints, dt=dt)
    feature_seq = np.nan_to_num(feature_seq, nan=0.0)

    # extract_windows expects (T, N, C); feature_seq is (T, 15) → reshape
    feature_3d = feature_seq[:, :, np.newaxis]
    windows, window_labels = extract_windows(
        feature_3d,
        label,
        window_size=WINDOW_SIZE,
        stride=stride,
    )

    if not windows:
        return None

    seqs = np.array(
        [w.reshape(w.shape[0], -1) for w in windows],
        dtype=np.float32,
    )
    seqs = (seqs - norm_mean) / (norm_std + 1e-8)

    ds = FallDetectionDataset(seqs, np.array(window_labels, dtype=np.int64))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for x_batch, _ in loader:
            probs = torch.softmax(net(x_batch), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    return {
        "label": label,
        "n_windows": len(all_probs),
        "seq_prob_max": float(np.max(all_probs)),
        "seq_prob_mean": float(np.mean(all_probs)),
        "window_probs": all_probs,
    }


def run_nonoverlapping_auc(config: dict, output_dir: Path) -> dict:
    """
    Compute non-overlapping window AUC for BiLSTM and Transformer-LSTM.

    Loads existing LOSO fold models from run5_stride2_aug (BiLSTM) and
    tl_run5_d64_h128_drop03 (Transformer-LSTM). For each fold, infers on
    the held-out test subject using stride=30 (non-overlapping windows).
    Aggregates window probabilities to sequence level with max pooling
    and computes sequence-level AUC across all five folds.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    output_dir : Path
        Directory where results are saved.

    Returns
    -------
    dict
        Summary with per-fold and mean AUCs for both models.
    """
    cfg_model = config["model"]
    cfg_tl = config.get("transformer_lstm", {})
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    metadata_path = urfd_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    extractor = FeatureExtractor(
        keypoint_format="mediapipe",
        confidence_threshold=0.5,
    )
    n_features = extractor.NUM_FEATURES
    dt = 1.0 / TARGET_FPS

    # Model specs: (display_name, fold_model_dir, model_type)
    model_specs = [
        ("BiLSTM", Path("models/run5_stride2_aug"), "bilstm"),
        ("Transformer-LSTM", Path("models/tl_run5_d64_h128_drop03"), "tl"),
    ]

    summary = {}

    for model_name, model_dir, model_type in model_specs:
        print(f"\nProcessing {model_name} (stride={STRIDE_NONOVERLAPPING})...")

        all_seq_results = []
        skipped_folds = []

        for fold_idx, fold in enumerate(folds):
            test_subject = fold["test"][0]
            fold_dir = model_dir / f"fold_{fold_idx}"

            if not fold_dir.exists():
                print(f"  WARNING: Fold {fold_idx} dir missing ({fold_dir}), skipping")
                skipped_folds.append(fold_idx)
                continue

            norm_mean_path = fold_dir / "norm_mean.npy"
            norm_std_path = fold_dir / "norm_std.npy"
            ckpt_path = fold_dir / "best_model.pth"

            missing = [
                p for p in [norm_mean_path, norm_std_path, ckpt_path]
                if not p.exists()
            ]
            if missing:
                print(f"  WARNING: Fold {fold_idx} missing files: {missing}, skipping")
                skipped_folds.append(fold_idx)
                continue

            norm_mean = np.load(str(norm_mean_path))
            norm_std = np.load(str(norm_std_path))

            try:
                if model_type == "bilstm":
                    net = load_bilstm(fold_dir, cfg_model, n_features)
                else:
                    net = load_tl(fold_dir, cfg_tl, n_features)
            except Exception as exc:
                print(f"  WARNING: Fold {fold_idx} model load failed: {exc}, skipping")
                skipped_folds.append(fold_idx)
                continue

            fold_seq_count = 0
            for seq_meta in meta["sequences"]:
                if seq_meta["subject_id"] != test_subject:
                    continue

                seq_id = seq_meta["sequence_id"]
                label = seq_meta["label"]

                kp_path = urfd_dir / "keypoints_raw" / f"{seq_id}.npy"
                if not kp_path.exists():
                    continue

                keypoints = np.load(str(kp_path))

                result = infer_sequence(
                    net=net,
                    keypoints=keypoints,
                    label=label,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    extractor=extractor,
                    dt=dt,
                    stride=STRIDE_NONOVERLAPPING,
                )

                if result is None:
                    print(f"    WARNING: No windows for {seq_id} (too short), skipping")
                    continue

                all_seq_results.append({
                    "seq_id": seq_id,
                    "label": label,
                    "fold": fold_idx,
                    "subject": test_subject,
                    "n_windows": result["n_windows"],
                    "seq_prob_max": result["seq_prob_max"],
                    "seq_prob_mean": result["seq_prob_mean"],
                })
                fold_seq_count += 1

            print(
                f"  Fold {fold_idx} (subject={test_subject}): "
                f"{fold_seq_count} sequences, "
                f"{STRIDE_NONOVERLAPPING}-stride windows"
            )

        # Per-fold AUC
        fold_aucs = []
        per_fold_counts = {}
        for fold_idx in range(len(folds)):
            fold_res = [r for r in all_seq_results if r["fold"] == fold_idx]
            if not fold_res:
                fold_aucs.append(None)
                continue
            y = np.array([r["label"] for r in fold_res])
            p = np.array([r["seq_prob_max"] for r in fold_res])
            if len(np.unique(y)) < 2:
                print(f"  WARNING: Fold {fold_idx} has only one class — AUC undefined")
                fold_aucs.append(None)
                continue
            auc_val = roc_auc_score(y, p)
            fold_aucs.append(round(float(auc_val), 4))
            per_fold_counts[fold_idx] = {
                "n_seq": len(fold_res),
                "n_fall": int(y.sum()),
                "n_adl": int((y == 0).sum()),
            }

        valid_aucs = [a for a in fold_aucs if a is not None]
        if not valid_aucs:
            print(f"  ERROR: No valid folds for {model_name}")
            summary[model_name] = {"error": "no valid folds"}
            continue

        mean_auc = float(np.mean(valid_aucs))
        std_auc = float(np.std(valid_aucs))

        # Overall sequence-level AUC (all folds combined)
        all_labels = np.array([r["label"] for r in all_seq_results])
        all_probs = np.array([r["seq_prob_max"] for r in all_seq_results])
        overall_auc = None
        if len(np.unique(all_labels)) >= 2:
            overall_auc = round(float(roc_auc_score(all_labels, all_probs)), 4)

        ref_auc = V3_REFERENCE.get(model_name)
        delta = round(mean_auc - ref_auc, 4) if ref_auc is not None else None

        summary[model_name] = {
            "stride": STRIDE_NONOVERLAPPING,
            "window_size": WINDOW_SIZE,
            "per_fold_auc": fold_aucs,
            "mean_fold_auc": round(mean_auc, 4),
            "std_fold_auc": round(std_auc, 4),
            "overall_combined_auc": overall_auc,
            "n_sequences": len(all_seq_results),
            "n_fall": int(all_labels.sum()),
            "n_adl": int((all_labels == 0).sum()),
            "skipped_folds": skipped_folds,
            "v3_reference_seq_auc_stride2_maxpool": ref_auc,
            "delta_vs_v3": delta,
            "per_fold_counts": per_fold_counts,
        }

    return summary


def plot_roc_comparison(summary: dict, output_dir: Path) -> None:
    """
    Plot side-by-side ROC comparison: non-overlapping vs reference labels.

    Parameters
    ----------
    summary : dict
        Output from run_nonoverlapping_auc.
    output_dir : Path
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {
        "BiLSTM": "#2196F3",
        "Transformer-LSTM": "#FF5722",
    }

    for model_name, res in summary.items():
        if "error" in res:
            continue
        color = colors.get(model_name, "#333333")
        mean_auc = res["mean_fold_auc"]
        ref_auc = res.get("v3_reference_seq_auc_stride2_maxpool")

        label_str = f"{model_name}: stride=30 mean={mean_auc:.3f}"
        if ref_auc is not None:
            label_str += f" (vs stride=2: {ref_auc:.3f})"

        ax.bar(
            model_name,
            mean_auc,
            color=color,
            alpha=0.8,
            label=label_str,
            width=0.4,
        )
        if ref_auc is not None:
            ax.axhline(
                ref_auc,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.6,
            )

    ax.set_ylim([0.5, 1.0])
    ax.set_ylabel("Sequence-Level AUC (max-pool)")
    ax.set_title(
        "Non-Overlapping Windows (stride=30) vs Stride=2\n"
        "Dashed line = v3 stride=2 reference",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    out_png = output_dir / "nonoverlapping_auc_comparison.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {out_png}")


def main() -> None:
    """Entry point: load config, run analysis, save results, print summary."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "config/config.yaml not found. Run from the project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path("results/v4_analysis/nonoverlapping_windows")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("NON-OVERLAPPING WINDOW AUC (v4 reviewer response)")
    print(f"stride={STRIDE_NONOVERLAPPING} (== window_size={WINDOW_SIZE}, zero overlap)")
    print("=" * 65)

    summary = run_nonoverlapping_auc(config, output_dir)

    # Save JSON
    out_json = output_dir / "nonoverlapping_results.json"
    with open(str(out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Plot
    plot_roc_comparison(summary, output_dir)

    # Print clear summary
    print()
    print("=" * 65)
    print("NON-OVERLAPPING WINDOW AUC (stride=30)")
    print("=" * 65)
    for model_name, res in summary.items():
        if "error" in res:
            print(f"{model_name}: ERROR — {res['error']}")
            continue

        fold_aucs = res["per_fold_auc"]
        mean_auc = res["mean_fold_auc"]
        std_auc = res["std_fold_auc"]
        ref_auc = res.get("v3_reference_seq_auc_stride2_maxpool")
        delta = res.get("delta_vs_v3")
        n_seq = res["n_sequences"]
        n_fall = res["n_fall"]
        n_adl = res["n_adl"]

        fold_str = ", ".join(
            f"{a:.3f}" if a is not None else "N/A"
            for a in fold_aucs
        )
        print(f"\n{model_name}:")
        print(f"  Per-fold AUCs:  [{fold_str}]")
        print(f"  Mean AUC:       {mean_auc:.4f} ± {std_auc:.4f}")
        if ref_auc is not None and delta is not None:
            direction = "above" if delta >= 0 else "below"
            print(
                f"  vs stride=2:    {ref_auc:.4f}  "
                f"(delta = {delta:+.4f}, stride=30 is {direction})"
            )
        print(
            f"  Sequences:      {n_seq} total ({n_fall} fall, {n_adl} ADL)"
        )

    print()
    print("-" * 65)
    print("INTERPRETATION")
    print("-" * 65)
    for model_name, res in summary.items():
        if "error" in res:
            continue
        mean_auc = res["mean_fold_auc"]
        ref_auc = res.get("v3_reference_seq_auc_stride2_maxpool")
        if ref_auc is None:
            continue
        delta = res.get("delta_vs_v3", 0.0)
        if abs(delta) < 0.02:
            verdict = "CONFIRMED: non-overlapping AUC is within ±0.02 of stride=2"
        elif delta < 0:
            verdict = f"LOWER by {abs(delta):.4f} — stride=2 windows give slightly higher AUC"
        else:
            verdict = f"HIGHER by {delta:.4f} — non-overlapping windows actually improve AUC"
        print(f"  {model_name}: {verdict}")
    print("=" * 65)


if __name__ == "__main__":
    main()
