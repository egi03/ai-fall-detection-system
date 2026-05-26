"""
Cross-dataset evaluation script for fall detection.

Evaluates a model trained on one dataset against a different test dataset.
Supports per-room breakdown for Le2i and per-subject breakdown for URFD.

Experiments implemented (per PLAN_CROSS_DATASET_EVAL.md):
  Experiment 1: URFD → Le2i  (train on URFD full, test on all Le2i rooms)
  Experiment 2: Le2i → URFD  (train on Le2i full, test on all URFD subjects)
  Experiment 3: Ensemble URFD→Le2i  (average probabilities from two models)

Usage:
    # Experiment 1: URFD → Le2i
    python scripts/cross_dataset_eval.py \\
        --model-dir models/cross_urfd_full_r5 \\
        --test-dataset le2i \\
        --output-dir results/cross_dataset/urfd_to_le2i_r5

    # Experiment 2: Le2i → URFD
    python scripts/cross_dataset_eval.py \\
        --model-dir models/cross_le2i_full_r5 \\
        --test-dataset urfd \\
        --output-dir results/cross_dataset/le2i_to_urfd_r5

    # Experiment 3: Ensemble (two model dirs)
    python scripts/cross_dataset_eval.py \\
        --model-dir models/cross_urfd_full_r5 \\
        --ensemble-model-dir models/cross_urfd_old \\
        --test-dataset le2i \\
        --output-dir results/cross_dataset/ensemble_urfd_to_le2i

Reference: PLAN_CROSS_DATASET_EVAL.md, research/5.1, research/5.2.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from src.features.extractor import FeatureExtractor
from src.models.lstm import FallDetectionLSTM
from src.models.transformer_lstm import FallDetectionTransformerLSTM
from src.training.dataset import FallDetectionDataset, build_dataset_from_processed
from src.training.evaluate import Evaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Le2i room groups for per-room breakdown
LE2I_ROOMS = [
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02",
    "Lecture_room",
    "Office",
]
# Rooms with only falls (sensitivity only — no ADLs to compute specificity)
LE2I_FALL_ONLY_ROOMS = {"Home_01", "Home_02"}
# Rooms with only ADLs (specificity only — no falls to compute sensitivity)
LE2I_ADL_ONLY_ROOMS = {"Coffee_room_02", "Lecture_room", "Office"}


def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(
    model_dir: Path,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> torch.nn.Module:
    """Load best_model.pth checkpoint from model_dir.

    Auto-detects architecture from run_meta.json if present.
    """
    # Check for run_meta.json to detect architecture
    meta_path = model_dir / "run_meta.json"
    architecture = "lstm"
    transformer_cfg = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        architecture = meta.get("architecture", "lstm")
        transformer_cfg = meta.get("transformer_cfg", {})

    if architecture == "transformer_lstm":
        model = FallDetectionTransformerLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            d_model=transformer_cfg.get("d_model", 64),
            nhead=transformer_cfg.get("nhead", 4),
            num_encoder_layers=transformer_cfg.get("num_encoder_layers", 2),
            dim_feedforward=transformer_cfg.get("dim_feedforward", 128),
            lstm_hidden_size=transformer_cfg.get("lstm_hidden_size", 128),
            lstm_num_layers=transformer_cfg.get("lstm_num_layers", 1),
            dropout=0.0,  # no dropout at inference
        )
    else:
        model = FallDetectionLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,  # no dropout at inference
            bidirectional=bidirectional,
        )
    ckpt = torch.load(
        str(model_dir / "best_model.pth"), map_location="cpu", weights_only=False
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    logger.info(f"Loaded model from {model_dir} (architecture={architecture})")
    return model


def _infer(
    model: FallDetectionLSTM,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (preds, probs)."""
    preds_list, probs_list = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds_list.extend(logits.argmax(dim=1).numpy())
            probs_list.extend(probs.numpy())
    return np.array(preds_list), np.array(probs_list)


def _load_norm_stats(
    model_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load normalization statistics saved during training."""
    mean = np.load(str(model_dir / "norm_mean.npy"))
    std = np.load(str(model_dir / "norm_std.npy"))
    return mean, std


def _threshold_sweep(
    labels: np.ndarray, probs: np.ndarray
) -> List[Dict]:
    """Compute sensitivity/specificity/F1 across thresholds 0.20–0.80."""
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix as cm

    results = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred = (probs >= t).astype(int)
        tn, fp, fn, tp = cm(labels, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(labels, pred, zero_division=0)
        results.append({
            "threshold": round(float(t), 2),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
        })
    return results


def evaluate_on_dataset(
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    test_dir: Path,
    subject_ids: List[str],
    window_size: int,
    batch_size: int,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load test sequences, normalize with training stats, run inference.

    Parameters
    ----------
    model : FallDetectionLSTM
        Loaded model in eval mode.
    norm_stats : (mean, std)
        Normalization statistics from the training dataset.
    test_dir : Path
        Processed data directory for the test dataset.
    subject_ids : list of str
        Subject IDs (or room names for Le2i) to evaluate.
    window_size : int
        Window size used during training.
    batch_size : int
        Inference batch size.
    stride : int
        Test stride (default 5 for consistent evaluation).

    Returns
    -------
    (labels, preds, probs)
    """
    seqs, labels, _ = build_dataset_from_processed(
        processed_dir=test_dir,
        subject_ids=subject_ids,
        window_size=window_size,
        stride=stride,
        positive_threshold=0.5,
        norm_stats=norm_stats,
    )
    ds = FallDetectionDataset(seqs, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, probs = _infer(model, loader)
    return labels, preds, probs


def run_urfd_to_le2i(
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    le2i_dir: Path,
    output_dir: Path,
    window_size: int,
    batch_size: int,
    evaluator: Evaluator,
    ensemble_model: Optional[FallDetectionLSTM] = None,
) -> None:
    """
    Experiment 1 (and 3): Evaluate URFD-trained model on all Le2i rooms.

    Produces:
    - Overall metrics (sens, spec, AUC, FAR)
    - Per-room metrics CSV
    - Confusion matrix, ROC curve, threshold sweep

    If ensemble_model is provided, averages probabilities before computing metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Overall evaluation across all Le2i rooms ---
    all_labels, all_preds, all_probs = evaluate_on_dataset(
        model, norm_stats, le2i_dir, LE2I_ROOMS, window_size, batch_size
    )
    if ensemble_model is not None:
        _, _, probs2 = evaluate_on_dataset(
            ensemble_model, norm_stats, le2i_dir, LE2I_ROOMS, window_size, batch_size
        )
        all_probs = (all_probs + probs2) / 2.0
        all_preds = (all_probs >= 0.5).astype(int)

    metrics = evaluator.compute_metrics(all_labels, all_preds, all_probs)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(all_labels, all_preds, output_dir / "confusion_matrix.png")
    evaluator.plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")
    sweep = _threshold_sweep(all_labels, all_probs)
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)
    np.save(str(output_dir / "all_probs.npy"), all_probs)
    np.save(str(output_dir / "all_labels.npy"), all_labels)

    # --- Per-room breakdown ---
    per_room_rows = []
    for room in LE2I_ROOMS:
        try:
            r_labels, r_preds, r_probs = evaluate_on_dataset(
                model, norm_stats, le2i_dir, [room], window_size, batch_size
            )
        except ValueError as e:
            logger.warning(f"Skipping room {room}: {e}")
            continue

        if ensemble_model is not None:
            try:
                _, _, r_probs2 = evaluate_on_dataset(
                    ensemble_model, norm_stats, le2i_dir, [room], window_size, batch_size
                )
                r_probs = (r_probs + r_probs2) / 2.0
                r_preds = (r_probs >= 0.5).astype(int)
            except ValueError:
                pass

        row: Dict = {"room": room, "n_windows": len(r_labels),
                     "n_fall_windows": int((r_labels == 1).sum()),
                     "n_adl_windows": int((r_labels == 0).sum())}

        if room in LE2I_FALL_ONLY_ROOMS:
            # Only falls — compute sensitivity only
            tp = int(((r_labels == 1) & (r_preds == 1)).sum())
            fn = int(((r_labels == 1) & (r_preds == 0)).sum())
            row["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan")
            row["specificity"] = "N/A"
            row["auc"] = "N/A"
            row["note"] = "falls_only"
        elif room in LE2I_ADL_ONLY_ROOMS:
            # Only ADLs — compute specificity (false alarm rate) only
            tn = int(((r_labels == 0) & (r_preds == 0)).sum())
            fp = int(((r_labels == 0) & (r_preds == 1)).sum())
            row["sensitivity"] = "N/A"
            row["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else float("nan")
            row["auc"] = "N/A"
            row["note"] = "adl_only"
        else:
            # Mixed room — full metrics
            r_metrics = evaluator.compute_metrics(r_labels, r_preds, r_probs)
            row["sensitivity"] = round(r_metrics["sensitivity"], 4)
            row["specificity"] = round(r_metrics["specificity"], 4)
            row["auc"] = round(r_metrics["auc_roc"], 4)
            row["note"] = "mixed"

        per_room_rows.append(row)
        logger.info(
            f"  {room}: sens={row['sensitivity']}, spec={row['specificity']}, "
            f"auc={row['auc']} ({row['note']})"
        )

    # Save per-room CSV
    csv_fields = ["room", "n_windows", "n_fall_windows", "n_adl_windows",
                  "sensitivity", "specificity", "auc", "note"]
    with open(output_dir / "per_room_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(per_room_rows)

    _print_cross_summary("URFD->Le2i", metrics, sweep, per_room_rows, group_col="room")


def run_le2i_to_urfd(
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    urfd_dir: Path,
    output_dir: Path,
    window_size: int,
    batch_size: int,
    evaluator: Evaluator,
) -> None:
    """
    Experiment 2: Evaluate Le2i-trained model on all URFD subjects.

    Produces:
    - Overall metrics
    - Per-subject metrics CSV
    - Confusion matrix, ROC curve, threshold sweep
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    urfd_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    all_labels, all_preds, all_probs = evaluate_on_dataset(
        model, norm_stats, urfd_dir, urfd_subjects, window_size, batch_size
    )

    metrics = evaluator.compute_metrics(all_labels, all_preds, all_probs)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(all_labels, all_preds, output_dir / "confusion_matrix.png")
    evaluator.plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")
    sweep = _threshold_sweep(all_labels, all_probs)
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)
    np.save(str(output_dir / "all_probs.npy"), all_probs)
    np.save(str(output_dir / "all_labels.npy"), all_labels)

    # Per-subject breakdown
    per_subject_rows = []
    for subj in urfd_subjects:
        try:
            s_labels, s_preds, s_probs = evaluate_on_dataset(
                model, norm_stats, urfd_dir, [subj], window_size, batch_size
            )
        except ValueError as e:
            logger.warning(f"Skipping subject {subj}: {e}")
            continue
        s_metrics = evaluator.compute_metrics(s_labels, s_preds, s_probs)
        per_subject_rows.append({
            "subject": subj,
            "sensitivity": round(s_metrics["sensitivity"], 4),
            "specificity": round(s_metrics["specificity"], 4),
            "f1": round(s_metrics["f1"], 4),
            "auc": round(s_metrics["auc_roc"], 4),
        })
        logger.info(
            f"  {subj}: sens={s_metrics['sensitivity']:.3f}, "
            f"spec={s_metrics['specificity']:.3f}, auc={s_metrics['auc_roc']:.3f}"
        )

    csv_fields = ["subject", "sensitivity", "specificity", "f1", "auc"]
    with open(output_dir / "per_subject_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(per_subject_rows)

    _print_cross_summary("Le2i->URFD", metrics, sweep, per_subject_rows, group_col="subject")


def generate_domain_gap_plot(
    output_dir: Path,
    loso_results_dir: Optional[Path],
    urfd_to_le2i_dir: Optional[Path],
    le2i_to_urfd_dir: Optional[Path],
) -> None:
    """
    Experiment 4: Overlay ROC curves for LOSO vs cross-dataset evaluations.

    Loads saved probability arrays and plots all curves on a single figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    def _add_roc(probs_path: Path, labels_path: Path, label: str, color: str) -> None:
        if not probs_path.exists() or not labels_path.exists():
            logger.warning(f"Missing ROC data: {probs_path}")
            return
        probs = np.load(str(probs_path))
        labels = np.load(str(labels_path))
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

    if loso_results_dir:
        # Try ensemble probs first; fall back to all_probs; labels may be all_true or all_labels
        probs_candidates = ["ensemble_r3r5_probs.npy", "all_probs.npy"]
        labels_candidates = ["ensemble_true.npy", "all_true.npy", "all_labels.npy"]
        p_path = next((loso_results_dir / p for p in probs_candidates
                       if (loso_results_dir / p).exists()), None)
        l_path = next((loso_results_dir / l for l in labels_candidates
                       if (loso_results_dir / l).exists()), None)
        if p_path and l_path:
            _add_roc(p_path, l_path, "URFD LOSO (within-dataset)", "steelblue")
    if urfd_to_le2i_dir:
        _add_roc(
            urfd_to_le2i_dir / "all_probs.npy",
            urfd_to_le2i_dir / "all_labels.npy",
            "URFD → Le2i (cross-dataset)", "darkorange",
        )
    if le2i_to_urfd_dir:
        _add_roc(
            le2i_to_urfd_dir / "all_probs.npy",
            le2i_to_urfd_dir / "all_labels.npy",
            "Le2i → URFD (cross-dataset)", "forestgreen",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: Within-Dataset vs Cross-Dataset", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = output_dir / "domain_gap_roc_overlay.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved domain gap ROC overlay to {out_path}")
    print(f"\nDomain gap ROC overlay saved to: {out_path}")


def _print_cross_summary(
    direction: str,
    metrics: dict,
    sweep: List[dict],
    breakdown_rows: List[dict],
    group_col: str,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Cross-dataset: {direction}")
    print(f"{'='*60}")
    print(
        f"  Overall — sens={metrics['sensitivity']:.3f}, "
        f"spec={metrics['specificity']:.3f}, "
        f"AUC={metrics['auc_roc']:.3f}, "
        f"F1={metrics['f1']:.3f}"
    )
    print(f"\n  Per-{group_col} breakdown:")
    print(f"  {group_col:<22} {'sens':>7} {'spec':>7} {'auc':>7}")
    print("  " + "-" * 42)
    for row in breakdown_rows:
        sens = f"{row['sensitivity']:.4f}" if isinstance(row["sensitivity"], float) else row["sensitivity"]
        spec = f"{row['specificity']:.4f}" if isinstance(row["specificity"], float) else row["specificity"]
        auc_ = f"{row['auc']:.4f}" if isinstance(row["auc"], float) else row["auc"]
        print(f"  {row[group_col]:<22} {sens:>7} {spec:>7} {auc_:>7}")
    print(f"\n  Threshold sweep (sens / spec):")
    for r in sweep:
        marker = " <- both>=0.70" if r["sensitivity"] >= 0.70 and r["specificity"] >= 0.70 else ""
        print(
            f"    t={r['threshold']:.2f}  sens={r['sensitivity']:.3f}  "
            f"spec={r['specificity']:.3f}  f1={r['f1']:.3f}{marker}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation for fall detection"
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory with best_model.pth and norm_mean/std.npy"
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True, choices=["urfd", "le2i"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to save results"
    )
    parser.add_argument(
        "--ensemble-model-dir", type=str, default=None,
        help="Optional second model dir for ensemble (URFD→Le2i only)"
    )
    parser.add_argument(
        "--domain-gap-plot", action="store_true",
        help="Generate ROC overlay plot comparing LOSO vs cross-dataset"
    )
    parser.add_argument(
        "--loso-results-dir", type=str, default=None,
        help="Path to saved LOSO probs/labels (for domain gap plot)"
    )
    parser.add_argument(
        "--urfd-to-le2i-dir", type=str, default=None,
        help="Path to URFD→Le2i results (for domain gap plot)"
    )
    parser.add_argument(
        "--le2i-to-urfd-dir", type=str, default=None,
        help="Path to Le2i→URFD results (for domain gap plot)"
    )
    args = parser.parse_args()

    config = _load_config()
    cfg_model = config["model"]
    cfg_feat = config["features"]
    cfg_train = config["training"]

    hidden_size = cfg_model["hidden_size"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)
    window_size = cfg_feat["window_size"]
    batch_size = cfg_train["batch_size"]

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    evaluator = Evaluator(output_dir=output_dir)

    # Try to read hidden_size from run_meta.json if available (overrides config)
    run_meta_path = model_dir / "run_meta.json"
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        hidden_size = run_meta.get("hidden_size", hidden_size)
        bidirectional = run_meta.get("bidirectional", bidirectional)
        window_size = run_meta.get("window_size", window_size)

    model = _load_model(model_dir, hidden_size, num_layers, bidirectional)
    norm_stats = _load_norm_stats(model_dir)

    ensemble_model = None
    if args.ensemble_model_dir:
        ensemble_dir = Path(args.ensemble_model_dir)
        ensemble_model = _load_model(ensemble_dir, hidden_size, num_layers, bidirectional)
        logger.info(f"Ensemble mode: averaging with {ensemble_dir}")

    data_processed = Path(config["paths"]["data_processed"])

    if args.domain_gap_plot:
        generate_domain_gap_plot(
            output_dir=output_dir,
            loso_results_dir=Path(args.loso_results_dir) if args.loso_results_dir else None,
            urfd_to_le2i_dir=Path(args.urfd_to_le2i_dir) if args.urfd_to_le2i_dir else None,
            le2i_to_urfd_dir=Path(args.le2i_to_urfd_dir) if args.le2i_to_urfd_dir else None,
        )
        return

    if args.test_dataset == "le2i":
        le2i_dir = data_processed / "le2i"
        run_urfd_to_le2i(
            model=model,
            norm_stats=norm_stats,
            le2i_dir=le2i_dir,
            output_dir=output_dir,
            window_size=window_size,
            batch_size=batch_size,
            evaluator=evaluator,
            ensemble_model=ensemble_model,
        )
    elif args.test_dataset == "urfd":
        urfd_dir = data_processed / "urfd"
        run_le2i_to_urfd(
            model=model,
            norm_stats=norm_stats,
            urfd_dir=urfd_dir,
            output_dir=output_dir,
            window_size=window_size,
            batch_size=batch_size,
            evaluator=evaluator,
        )


if __name__ == "__main__":
    main()
