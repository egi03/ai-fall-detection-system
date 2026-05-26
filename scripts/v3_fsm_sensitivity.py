"""
FSM parameter sensitivity analysis for the alarm system.

Addresses reviewer comment: "A sensitivity analysis over EMA alpha and
persistence thresholds would add robustness."

Runs a grid search over:
  - EMA alpha in {0.1, 0.2, 0.3, 0.4, 0.5}
  - persistence in {3, 5, 7, 10, 15}
  - threshold in {0.50, 0.60, 0.75}

For each combination, reports:
  - event-level sensitivity (fraction of fall sequences that trigger alarm)
  - event-level FP count (ADL sequences that trigger alarm)
  - 12x suppression ratio

Uses pre-loaded window probabilities (run5 BiLSTM).

Usage:
    python scripts/v3_fsm_sensitivity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import torch
from torch.utils.data import DataLoader

from src.data_processing.splitter import SubjectSplitter
from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.training.dataset import FallDetectionDataset, extract_windows

SEED = 42
WINDOW_SIZE = 30
STRIDE = 2
TARGET_FPS = 15.0


def simulate_fsm(probs: np.ndarray, threshold: float, persistence: int,
                 ema_alpha: float) -> dict:
    """Simulate EMA+persistence FSM. Returns dict with alarm_triggered."""
    ema_prob = 0.0
    consecutive = 0
    for p in probs:
        ema_prob = ema_alpha * p + (1 - ema_alpha) * ema_prob
        if ema_prob >= threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= persistence:
            return {"alarm_triggered": True}
    return {"alarm_triggered": False}


def load_sequence_probs(config: dict):
    """Load per-sequence window probability arrays using LOSO fold models."""
    cfg_model = config["model"]
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"

    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()
    extractor = FeatureExtractor(keypoint_format="mediapipe",
                                 confidence_threshold=0.5)
    model_dir = Path("models/run5_stride2_aug")
    dt = 1.0 / TARGET_FPS

    sequence_probs = []  # list of {seq_id, label, probs, subject}

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        fold_dir = model_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            continue

        norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
        norm_std = np.load(str(fold_dir / "norm_std.npy"))

        model = FallDetectionLSTM(
            input_size=extractor.NUM_FEATURES,
            hidden_size=cfg_model["hidden_size"],
            num_layers=cfg_model["num_layers"],
            dropout=0.0,
            bidirectional=cfg_model.get("bidirectional", False),
        )
        ckpt = torch.load(str(fold_dir / "best_model.pth"),
                          map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()

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
                    p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(p.tolist())

            sequence_probs.append({
                "seq_id": seq_id,
                "label": label,
                "subject": test_subject,
                "probs": np.array(all_probs),
            })

    print(f"Loaded {len(sequence_probs)} sequences")
    return sequence_probs


def run_grid_search(sequence_probs, output_dir: Path):
    """Run FSM parameter grid search and save results."""
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    persistences = [3, 5, 7, 10, 15]
    thresholds = [0.50, 0.60, 0.75]

    fall_seqs = [s for s in sequence_probs if s["label"] == 1]
    adl_seqs = [s for s in sequence_probs if s["label"] == 0]
    n_fall = len(fall_seqs)
    n_adl = len(adl_seqs)

    print(f"Fall sequences: {n_fall}, ADL sequences: {n_adl}")

    grid_results = []

    for threshold in thresholds:
        for alpha in alphas:
            for persistence in persistences:
                fall_detected = 0
                fp_count = 0

                for seq in fall_seqs:
                    r = simulate_fsm(seq["probs"], threshold, persistence, alpha)
                    if r["alarm_triggered"]:
                        fall_detected += 1

                for seq in adl_seqs:
                    r = simulate_fsm(seq["probs"], threshold, persistence, alpha)
                    if r["alarm_triggered"]:
                        fp_count += 1

                event_sens = fall_detected / max(n_fall, 1)
                event_far = fp_count  # number of FP events (not rate, since clips are short)
                grid_results.append({
                    "threshold": threshold,
                    "alpha": alpha,
                    "persistence": persistence,
                    "event_sensitivity": round(event_sens, 4),
                    "fp_count": fp_count,
                    "fall_detected": fall_detected,
                    "n_fall": n_fall,
                    "n_adl": n_adl,
                })

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "fsm_grid_search.json"
    with open(str(out_json), "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"Grid results saved: {out_json}")

    # Print default config result
    default = [r for r in grid_results
               if r["threshold"] == 0.75 and r["alpha"] == 0.3
               and r["persistence"] == 10][0]
    print(f"\nDefault config (α=0.3, persistence=10, t=0.75):")
    print(f"  Event sensitivity: {default['event_sensitivity']:.3f}")
    print(f"  FP count: {default['fp_count']}")

    # Create heatmaps: for each threshold, show sensitivity and FP as function of
    # alpha vs persistence
    fig, axes = plt.subplots(len(thresholds), 2,
                             figsize=(10, 4 * len(thresholds)))
    if len(thresholds) == 1:
        axes = axes[np.newaxis, :]

    for row, threshold in enumerate(thresholds):
        thresh_results = [r for r in grid_results if r["threshold"] == threshold]

        # Build matrices
        sens_matrix = np.zeros((len(alphas), len(persistences)))
        fp_matrix = np.zeros((len(alphas), len(persistences)))

        for r in thresh_results:
            ai = alphas.index(r["alpha"])
            pi = persistences.index(r["persistence"])
            sens_matrix[ai, pi] = r["event_sensitivity"]
            fp_matrix[ai, pi] = r["fp_count"]

        # Sensitivity heatmap
        ax = axes[row, 0]
        im = ax.imshow(sens_matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(persistences)))
        ax.set_xticklabels(persistences, fontsize=8)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels(alphas, fontsize=8)
        ax.set_xlabel("Persistence (windows)", fontsize=9)
        ax.set_ylabel("EMA α", fontsize=9)
        ax.set_title(f"Event Sensitivity  (threshold={threshold})", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(alphas)):
            for j in range(len(persistences)):
                ax.text(j, i, f"{sens_matrix[i,j]:.2f}", ha="center",
                        va="center", fontsize=7,
                        color="black" if sens_matrix[i, j] > 0.5 else "white")

        # Mark default config
        if threshold == 0.75:
            ai_def = alphas.index(0.3)
            pi_def = persistences.index(10)
            ax.add_patch(plt.Rectangle(
                (pi_def - 0.5, ai_def - 0.5), 1, 1,
                fill=False, edgecolor="blue", lw=2, label="Default"
            ))

        # FP count heatmap
        ax2 = axes[row, 1]
        max_fp = max(fp_matrix.max(), 1)
        im2 = ax2.imshow(fp_matrix, aspect="auto", cmap="RdYlGn_r",
                         vmin=0, vmax=max_fp)
        ax2.set_xticks(range(len(persistences)))
        ax2.set_xticklabels(persistences, fontsize=8)
        ax2.set_yticks(range(len(alphas)))
        ax2.set_yticklabels(alphas, fontsize=8)
        ax2.set_xlabel("Persistence (windows)", fontsize=9)
        ax2.set_ylabel("EMA α", fontsize=9)
        ax2.set_title(f"FP Alarm Count  (threshold={threshold})", fontsize=9)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        for i in range(len(alphas)):
            for j in range(len(persistences)):
                ax2.text(j, i, f"{int(fp_matrix[i,j])}", ha="center",
                         va="center", fontsize=7)

        if threshold == 0.75:
            ax2.add_patch(plt.Rectangle(
                (pi_def - 0.5, ai_def - 0.5), 1, 1,
                fill=False, edgecolor="blue", lw=2
            ))

    fig.suptitle("FSM Parameter Sensitivity Analysis (BiLSTM Run5, URFD LOSO)\n"
                 "Blue box = paper default (α=0.3, persistence=10)", fontsize=10)
    fig.tight_layout()
    out_png = output_dir / "fsm_sensitivity_heatmap.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {out_png}")

    # Print paper-ready summary of key configs
    print("\n" + "=" * 60)
    print("KEY PARAMETER CONFIGURATIONS:")
    print(f"{'α':>5} {'Pers':>5} {'Thr':>5} {'Sens':>6} {'FP':>4}")
    print("-" * 35)
    key_configs = [
        (0.3, 3, 0.75),
        (0.3, 5, 0.75),
        (0.3, 7, 0.75),
        (0.3, 10, 0.75),   # default
        (0.3, 15, 0.75),
        (0.3, 10, 0.50),
        (0.3, 10, 0.60),
    ]
    for alpha, pers, thr in key_configs:
        r = next(x for x in grid_results
                 if x["alpha"] == alpha and x["persistence"] == pers
                 and x["threshold"] == thr)
        marker = " *" if (alpha == 0.3 and pers == 10 and thr == 0.75) else ""
        print(f"{alpha:>5} {pers:>5} {thr:>5} {r['event_sensitivity']:>6.3f} "
              f"{r['fp_count']:>4}{marker}")

    return grid_results


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path("results/v3_analysis/fsm_sensitivity")

    print("=" * 60)
    print("FSM SENSITIVITY ANALYSIS (v3 paper fix)")
    print("=" * 60)
    print("Loading per-sequence window probabilities...")
    sequence_probs = load_sequence_probs(config)

    print("\nRunning grid search...")
    run_grid_search(sequence_probs, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
