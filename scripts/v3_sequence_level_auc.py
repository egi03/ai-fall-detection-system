"""
Sequence-level AUC computation for BiLSTM and Transformer-LSTM.

Addresses reviewer comment: "Window-level AUCs and CIs might be inflated by
overlapping windows and temporal correlation. Per-sequence scoring for AUC or
aggregating windows within a sequence before ROC computation would reduce
dependence violations."

Method: For each LOSO test fold, run each model on all test sequences.
Aggregate per-window probabilities into a per-sequence probability using
max pooling (most conservative: detects if ANY window strongly fires).
Compute ROC/AUC at the sequence level.

Usage:
    python scripts/v3_sequence_level_auc.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

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
STRIDE = 2
TARGET_FPS = 15.0


def load_bilstm(fold_dir: Path, cfg_model: dict, n_features: int):
    model = FallDetectionLSTM(
        input_size=n_features,
        hidden_size=cfg_model["hidden_size"],
        num_layers=cfg_model["num_layers"],
        dropout=0.0,
        bidirectional=cfg_model.get("bidirectional", False),
    )
    ckpt = torch.load(str(fold_dir / "best_model.pth"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def load_tl(fold_dir: Path, n_features: int):
    model = FallDetectionTransformerLSTM(
        input_size=n_features,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=128,
        lstm_hidden_size=128,
        lstm_num_layers=1,
        dropout=0.0,
    )
    ckpt = torch.load(str(fold_dir / "best_model.pth"), map_location="cpu",
                      weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def run_sequence_level_auc(config: dict, output_dir: Path):
    cfg_model = config["model"]
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()
    extractor = FeatureExtractor(keypoint_format="mediapipe",
                                 confidence_threshold=0.5)
    n_features = extractor.NUM_FEATURES
    dt = 1.0 / TARGET_FPS

    model_specs = [
        ("BiLSTM", Path("models/run5_stride2_aug"), "bilstm"),
        ("Transformer-LSTM", Path("models/tl_run4_d64_drop03"), "tl"),
    ]

    all_model_results = {}

    for model_name, model_dir, model_type in model_specs:
        print(f"\nProcessing {model_name}...")
        all_seq_results = []

        for fold_idx, fold in enumerate(folds):
            test_subject = fold["test"][0]
            fold_dir = model_dir / f"fold_{fold_idx}"
            if not fold_dir.exists():
                print(f"  Fold {fold_idx} missing, skipping")
                continue

            norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
            norm_std = np.load(str(fold_dir / "norm_std.npy"))

            if model_type == "bilstm":
                net = load_bilstm(fold_dir, cfg_model, n_features)
            else:
                net = load_tl(fold_dir, n_features)

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

                feature_3d = feature_seq[:, :, np.newaxis]
                windows, window_labels = extract_windows(
                    feature_3d, label, window_size=WINDOW_SIZE, stride=STRIDE
                )
                if not windows:
                    continue

                seqs = np.array([w.reshape(w.shape[0], -1) for w in windows],
                                dtype=np.float32)
                seqs = (seqs - norm_mean) / (norm_std + 1e-8)

                ds = FallDetectionDataset(seqs, np.array(window_labels))
                loader = DataLoader(ds, batch_size=64, shuffle=False)

                all_probs = []
                with torch.no_grad():
                    for x, _ in loader:
                        p = torch.softmax(net(x), dim=1)[:, 1].cpu().numpy()
                        all_probs.extend(p.tolist())

                all_seq_results.append({
                    "seq_id": seq_id,
                    "label": label,
                    "fold": fold_idx,
                    "subject": test_subject,
                    "seq_prob_max": float(np.max(all_probs)),
                    "seq_prob_mean": float(np.mean(all_probs)),
                    "n_windows": len(all_probs),
                })

        all_model_results[model_name] = all_seq_results
        print(f"  {len(all_seq_results)} sequences processed")

    # Compute sequence-level AUCs
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {"BiLSTM": "#2196F3", "Transformer-LSTM": "#FF5722"}
    summary = {}

    for ax_idx, agg in enumerate(["max", "mean"]):
        ax = axes[ax_idx]
        agg_key = f"seq_prob_{agg}"
        ax.set_title(f"Sequence-Level ROC ({agg.capitalize()}-Pool)", fontsize=11)

        for model_name, results in all_model_results.items():
            labels_arr = np.array([r["label"] for r in results])
            probs_arr = np.array([r[agg_key] for r in results])

            if len(np.unique(labels_arr)) < 2:
                continue

            auc = roc_auc_score(labels_arr, probs_arr)
            fpr, tpr, _ = roc_curve(labels_arr, probs_arr)

            ax.plot(fpr, tpr, color=colors[model_name], lw=2,
                    label=f"{model_name} (AUC={auc:.3f})")

            if agg == "max":
                summary[model_name] = {
                    "seq_level_auc_max": round(auc, 4),
                    "n_sequences": len(results),
                    "n_fall": int(labels_arr.sum()),
                    "n_adl": int((labels_arr == 0).sum()),
                }
                print(f"  {model_name} seq-level AUC (max-pool): {auc:.4f}")
            if agg == "mean":
                if model_name in summary:
                    summary[model_name]["seq_level_auc_mean"] = round(auc, 4)
                print(f"  {model_name} seq-level AUC (mean-pool): {auc:.4f}")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    fig.suptitle("Sequence-Level ROC: Window Probability Aggregation\n"
                 "(URFD LOSO, reduces window-correlation bias)", fontsize=10)
    fig.tight_layout()
    out_png = output_dir / "sequence_level_roc.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nROC plot saved: {out_png}")

    # Per-fold sequence-level AUC
    for model_name, results in all_model_results.items():
        fold_aucs = []
        for fold_idx in range(5):
            fold_res = [r for r in results if r["fold"] == fold_idx]
            if not fold_res:
                continue
            y = np.array([r["label"] for r in fold_res])
            p = np.array([r["seq_prob_max"] for r in fold_res])
            if len(np.unique(y)) < 2:
                fold_aucs.append(None)
                continue
            fold_aucs.append(round(roc_auc_score(y, p), 4))

        valid_aucs = [x for x in fold_aucs if x is not None]
        if model_name in summary:
            summary[model_name]["per_fold_seq_auc"] = fold_aucs
            summary[model_name]["mean_fold_seq_auc"] = round(float(np.mean(valid_aucs)), 4)
            summary[model_name]["std_fold_seq_auc"] = round(float(np.std(valid_aucs)), 4)
        print(f"  {model_name} per-fold seq AUC: {fold_aucs}")
        print(f"    Mean: {np.mean(valid_aucs):.4f} +/- {np.std(valid_aucs):.4f}")

    out_json = output_dir / "sequence_level_auc.json"
    with open(str(out_json), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved: {out_json}")
    return summary


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path("results/v3_analysis/sequence_level_auc")
    print("=" * 60)
    print("SEQUENCE-LEVEL AUC COMPUTATION (v3 paper fix)")
    print("=" * 60)
    summary = run_sequence_level_auc(config, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
