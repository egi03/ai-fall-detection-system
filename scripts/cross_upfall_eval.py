"""
Cross-dataset evaluation involving the UP-Fall Detection Dataset.

Evaluates a model trained on one dataset (URFD or Le2i) against the UP-Fall
dataset, or evaluates a UP-Fall-trained model against URFD or Le2i.

Supported directions:
  urfd   → up_fall   (train on full URFD, test on all UP-Fall subjects)
  le2i   → up_fall   (train on full Le2i, test on all UP-Fall subjects)
  up_fall → urfd     (train on full UP-Fall, test on all URFD subjects)
  up_fall → le2i     (train on full UP-Fall, test on all Le2i rooms)

Usage:
    # URFD → UP-Fall
    python scripts/cross_upfall_eval.py \\
        --model-dir models/cross_urfd_full_r5 \\
        --test-dataset up_fall \\
        --output-dir results/cross_dataset/urfd_to_upfall

    # UP-Fall → URFD
    python scripts/cross_upfall_eval.py \\
        --model-dir models/cross_upfall_full \\
        --test-dataset urfd \\
        --output-dir results/cross_dataset/upfall_to_urfd

    # Generate domain gap ROC overlay
    python scripts/cross_upfall_eval.py \\
        --model-dir models/cross_urfd_full_r5 \\
        --test-dataset up_fall \\
        --output-dir results/cross_dataset/urfd_to_upfall \\
        --domain-gap-plot \\
        --loso-results-dir results/urfd_loso_v3 \\
        --source-loso-label "URFD LOSO (within-dataset)"

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

# Le2i rooms for per-room breakdown when testing on Le2i
LE2I_ROOMS = [
    "Coffee_room_01",
    "Coffee_room_02",
    "Home_01",
    "Home_02",
    "Lecture_room",
    "Office",
]
LE2I_FALL_ONLY_ROOMS = {"Home_01", "Home_02"}
LE2I_ADL_ONLY_ROOMS = {"Coffee_room_02", "Lecture_room", "Office"}


# ---------------------------------------------------------------------------
# Config / model loading helpers (mirrored from cross_dataset_eval.py)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_model(
    model_dir: Path,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> torch.nn.Module:
    """
    Load the best checkpoint from model_dir.

    Auto-detects architecture from run_meta.json if present.

    Parameters
    ----------
    model_dir : Path
        Directory containing best_model.pth.
    hidden_size : int
        LSTM hidden units.
    num_layers : int
        Number of LSTM layers.
    bidirectional : bool
        Whether the LSTM is bidirectional.

    Returns
    -------
    torch.nn.Module
        Model in eval mode with weights loaded.
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


def _load_norm_stats(model_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load feature normalization statistics saved during training.

    Parameters
    ----------
    model_dir : Path
        Directory containing norm_mean.npy and norm_std.npy.

    Returns
    -------
    (mean, std) : tuple of np.ndarray
    """
    mean = np.load(str(model_dir / "norm_mean.npy"))
    std = np.load(str(model_dir / "norm_std.npy"))
    return mean, std


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer(
    model: FallDetectionLSTM,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

    Parameters
    ----------
    model : FallDetectionLSTM
        Model in eval mode.
    loader : DataLoader
        DataLoader wrapping the test dataset.

    Returns
    -------
    (preds, probs) : tuple of np.ndarray
        Integer predictions and fall probabilities.
    """
    preds_list, probs_list = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds_list.extend(logits.argmax(dim=1).numpy())
            probs_list.extend(probs.numpy())
    return np.array(preds_list), np.array(probs_list)


def _evaluate_subjects(
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    data_dir: Path,
    subject_ids: List[str],
    window_size: int,
    batch_size: int,
    stride: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sequences for the given subjects, apply training norm_stats, infer.

    Parameters
    ----------
    model : FallDetectionLSTM
        Loaded model in eval mode.
    norm_stats : (mean, std)
        Normalization statistics from the source training run.
    data_dir : Path
        Processed data directory for the test dataset.
    subject_ids : list of str
        Subject IDs (or room names for Le2i) to include.
    window_size : int
        Sliding window length used during training.
    batch_size : int
        Inference batch size.
    stride : int
        Window stride (default 5 for consistent evaluation).

    Returns
    -------
    (labels, preds, probs) : tuple of np.ndarray
    """
    seqs, labels, _ = build_dataset_from_processed(
        processed_dir=data_dir,
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


def _threshold_sweep(
    labels: np.ndarray, probs: np.ndarray
) -> List[Dict]:
    """
    Compute sensitivity / specificity / F1 across thresholds 0.20-0.80.

    Parameters
    ----------
    labels : np.ndarray
        Ground-truth binary labels.
    probs : np.ndarray
        Predicted fall probabilities.

    Returns
    -------
    list of dict
        One entry per threshold with keys: threshold, sensitivity, specificity, f1.
    """
    from sklearn.metrics import confusion_matrix as cm, f1_score

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


# ---------------------------------------------------------------------------
# Per-subject evaluation (UP-Fall or URFD)
# ---------------------------------------------------------------------------

def _run_per_subject_eval(
    direction: str,
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    data_dir: Path,
    output_dir: Path,
    window_size: int,
    batch_size: int,
    evaluator: Evaluator,
) -> None:
    """
    Evaluate model on all subjects in data_dir with per-subject breakdown.

    Intended for both UP-Fall→URFD and URFD→UP-Fall directions.

    Parameters
    ----------
    direction : str
        Human-readable label for logging (e.g. 'URFD->UP-Fall').
    model : FallDetectionLSTM
        Loaded model in eval mode.
    norm_stats : (mean, std)
        Normalization statistics from the source training dataset.
    data_dir : Path
        Processed data directory for the test dataset.
    output_dir : Path
        Directory where results are saved.
    window_size : int
        Window size from training configuration.
    batch_size : int
        Inference batch size.
    evaluator : Evaluator
        Evaluator instance for metric computation and plot generation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all subjects from metadata
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {data_dir}. "
            "Run preprocessing first: python scripts/preprocess.py --dataset <name>"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    all_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))

    logger.info(
        f"Evaluating {direction}: {len(all_subjects)} test subjects in {data_dir.name}"
    )

    # Overall evaluation
    all_labels, all_preds, all_probs = _evaluate_subjects(
        model, norm_stats, data_dir, all_subjects, window_size, batch_size
    )

    metrics = evaluator.compute_metrics(all_labels, all_preds, all_probs)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(
        all_labels, all_preds, output_dir / "confusion_matrix.png"
    )
    evaluator.plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")

    sweep = _threshold_sweep(all_labels, all_probs)
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    np.save(str(output_dir / "all_probs.npy"), all_probs)
    np.save(str(output_dir / "all_labels.npy"), all_labels)

    # Per-subject breakdown
    per_subject_rows = []
    for subj in all_subjects:
        try:
            s_labels, s_preds, s_probs = _evaluate_subjects(
                model, norm_stats, data_dir, [subj], window_size, batch_size
            )
        except ValueError as exc:
            logger.warning(f"Skipping subject {subj}: {exc}")
            continue

        s_metrics = evaluator.compute_metrics(s_labels, s_preds, s_probs)
        per_subject_rows.append({
            "subject": subj,
            "n_windows": len(s_labels),
            "n_fall_windows": int((s_labels == 1).sum()),
            "n_adl_windows": int((s_labels == 0).sum()),
            "sensitivity": round(s_metrics["sensitivity"], 4),
            "specificity": round(s_metrics["specificity"], 4),
            "f1": round(s_metrics["f1"], 4),
            "auc": round(s_metrics["auc_roc"], 4),
        })
        logger.info(
            f"  {subj}: sens={s_metrics['sensitivity']:.3f}, "
            f"spec={s_metrics['specificity']:.3f}, "
            f"auc={s_metrics['auc_roc']:.3f}"
        )

    csv_fields = [
        "subject", "n_windows", "n_fall_windows", "n_adl_windows",
        "sensitivity", "specificity", "f1", "auc",
    ]
    with open(output_dir / "per_subject_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(per_subject_rows)

    _print_summary(direction, metrics, sweep, per_subject_rows, group_col="subject")


# ---------------------------------------------------------------------------
# UP-Fall → Le2i direction (per-room breakdown)
# ---------------------------------------------------------------------------

def _run_upfall_to_le2i(
    model: FallDetectionLSTM,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    le2i_dir: Path,
    output_dir: Path,
    window_size: int,
    batch_size: int,
    evaluator: Evaluator,
) -> None:
    """
    Evaluate an UP-Fall-trained model on all Le2i rooms with per-room breakdown.

    Parameters
    ----------
    model : FallDetectionLSTM
        Loaded model in eval mode.
    norm_stats : (mean, std)
        Normalization statistics from UP-Fall training.
    le2i_dir : Path
        Processed data directory for Le2i.
    output_dir : Path
        Directory where results are saved.
    window_size : int
        Window size from training configuration.
    batch_size : int
        Inference batch size.
    evaluator : Evaluator
        Evaluator instance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Evaluating UP-Fall -> Le2i across all rooms...")

    all_labels, all_preds, all_probs = _evaluate_subjects(
        model, norm_stats, le2i_dir, LE2I_ROOMS, window_size, batch_size
    )

    metrics = evaluator.compute_metrics(all_labels, all_preds, all_probs)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(
        all_labels, all_preds, output_dir / "confusion_matrix.png"
    )
    evaluator.plot_roc_curve(all_labels, all_probs, output_dir / "roc_curve.png")

    sweep = _threshold_sweep(all_labels, all_probs)
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)
    np.save(str(output_dir / "all_probs.npy"), all_probs)
    np.save(str(output_dir / "all_labels.npy"), all_labels)

    per_room_rows = []
    for room in LE2I_ROOMS:
        try:
            r_labels, r_preds, r_probs = _evaluate_subjects(
                model, norm_stats, le2i_dir, [room], window_size, batch_size
            )
        except ValueError as exc:
            logger.warning(f"Skipping room {room}: {exc}")
            continue

        row: Dict = {
            "room": room,
            "n_windows": len(r_labels),
            "n_fall_windows": int((r_labels == 1).sum()),
            "n_adl_windows": int((r_labels == 0).sum()),
        }

        if room in LE2I_FALL_ONLY_ROOMS:
            tp = int(((r_labels == 1) & (r_preds == 1)).sum())
            fn = int(((r_labels == 1) & (r_preds == 0)).sum())
            row["sensitivity"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else float("nan")
            row["specificity"] = "N/A"
            row["auc"] = "N/A"
            row["note"] = "falls_only"
        elif room in LE2I_ADL_ONLY_ROOMS:
            tn = int(((r_labels == 0) & (r_preds == 0)).sum())
            fp = int(((r_labels == 0) & (r_preds == 1)).sum())
            row["sensitivity"] = "N/A"
            row["specificity"] = round(tn / (tn + fp), 4) if (tn + fp) > 0 else float("nan")
            row["auc"] = "N/A"
            row["note"] = "adl_only"
        else:
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

    csv_fields = [
        "room", "n_windows", "n_fall_windows", "n_adl_windows",
        "sensitivity", "specificity", "auc", "note",
    ]
    with open(output_dir / "per_room_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(per_room_rows)

    _print_summary("UP-Fall->Le2i", metrics, sweep, per_room_rows, group_col="room")


# ---------------------------------------------------------------------------
# Domain gap ROC overlay
# ---------------------------------------------------------------------------

def _generate_domain_gap_plot(
    output_dir: Path,
    loso_results_dir: Optional[Path],
    cross_results_dir: Optional[Path],
    source_loso_label: str,
    cross_label: str,
) -> None:
    """
    Generate an ROC overlay comparing within-dataset LOSO vs cross-dataset eval.

    Parameters
    ----------
    output_dir : Path
        Directory where the plot is saved.
    loso_results_dir : Path or None
        Directory containing LOSO probability arrays (all_probs.npy /
        ensemble_r3r5_probs.npy and matching labels).
    cross_results_dir : Path or None
        Directory containing cross-dataset all_probs.npy / all_labels.npy.
    source_loso_label : str
        Legend label for the LOSO baseline curve.
    cross_label : str
        Legend label for the cross-dataset curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    def _add_roc(
        probs_path: Path, labels_path: Path, label: str, color: str
    ) -> None:
        if not probs_path.exists() or not labels_path.exists():
            logger.warning(f"Missing ROC data: {probs_path}")
            return
        probs = np.load(str(probs_path))
        labels = np.load(str(labels_path))
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

    if loso_results_dir is not None:
        probs_candidates = ["ensemble_r3r5_probs.npy", "all_probs.npy"]
        labels_candidates = ["ensemble_true.npy", "all_true.npy", "all_labels.npy"]
        p_path = next(
            (loso_results_dir / p for p in probs_candidates
             if (loso_results_dir / p).exists()),
            None,
        )
        l_path = next(
            (loso_results_dir / la for la in labels_candidates
             if (loso_results_dir / la).exists()),
            None,
        )
        if p_path and l_path:
            _add_roc(p_path, l_path, source_loso_label, "steelblue")

    if cross_results_dir is not None:
        _add_roc(
            cross_results_dir / "all_probs.npy",
            cross_results_dir / "all_labels.npy",
            cross_label,
            "darkorange",
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(
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
        sens = (
            f"{row['sensitivity']:.4f}"
            if isinstance(row["sensitivity"], float)
            else row["sensitivity"]
        )
        spec = (
            f"{row['specificity']:.4f}"
            if isinstance(row["specificity"], float)
            else row["specificity"]
        )
        auc_ = (
            f"{row['auc']:.4f}"
            if isinstance(row.get("auc"), float)
            else row.get("auc", "N/A")
        )
        print(f"  {row[group_col]:<22} {sens:>7} {spec:>7} {auc_:>7}")
    print("\n  Threshold sweep (sens / spec):")
    for r in sweep:
        marker = (
            " <- both>=0.70"
            if r["sensitivity"] >= 0.70 and r["specificity"] >= 0.70
            else ""
        )
        print(
            f"    t={r['threshold']:.2f}  sens={r['sensitivity']:.3f}  "
            f"spec={r['specificity']:.3f}  f1={r['f1']:.3f}{marker}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-dataset evaluation involving UP-Fall.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Directory with best_model.pth and norm_mean/std.npy"
    )
    parser.add_argument(
        "--test-dataset", type=str, required=True,
        choices=["urfd", "le2i", "up_fall"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to save results"
    )
    parser.add_argument(
        "--domain-gap-plot", action="store_true",
        help="Generate ROC overlay comparing LOSO vs cross-dataset"
    )
    parser.add_argument(
        "--loso-results-dir", type=str, default=None,
        help="Path to LOSO probs/labels dir (for domain gap plot)"
    )
    parser.add_argument(
        "--source-loso-label", type=str,
        default="LOSO within-dataset",
        help="Legend label for the LOSO baseline in the domain gap plot"
    )
    return parser


def main() -> None:
    """Entry point for the cross-dataset evaluation script."""
    parser = _build_parser()
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

    # Override config from run_meta.json if available (written by train_experiment.py)
    run_meta_path = model_dir / "run_meta.json"
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        hidden_size = run_meta.get("hidden_size", hidden_size)
        bidirectional = run_meta.get("bidirectional", bidirectional)
        window_size = run_meta.get("window_size", window_size)
        logger.info(
            f"Loaded run_meta.json: hidden={hidden_size}, "
            f"bidirectional={bidirectional}, window={window_size}"
        )

    model = _load_model(model_dir, hidden_size, num_layers, bidirectional)
    norm_stats = _load_norm_stats(model_dir)

    data_processed = Path(config["paths"]["data_processed"])
    test_dir = data_processed / args.test_dataset

    # Determine cross direction for labelling
    train_dataset = run_meta.get("train_dataset", "unknown") if run_meta_path.exists() else "unknown"
    direction = f"{train_dataset}->{args.test_dataset}"

    if args.domain_gap_plot:
        _generate_domain_gap_plot(
            output_dir=output_dir,
            loso_results_dir=Path(args.loso_results_dir) if args.loso_results_dir else None,
            cross_results_dir=output_dir,
            source_loso_label=args.source_loso_label,
            cross_label=direction,
        )
        return

    # Route to the appropriate evaluation function
    if args.test_dataset == "le2i":
        _run_upfall_to_le2i(
            model=model,
            norm_stats=norm_stats,
            le2i_dir=test_dir,
            output_dir=output_dir,
            window_size=window_size,
            batch_size=batch_size,
            evaluator=evaluator,
        )
    else:
        # up_fall or urfd: both use per-subject breakdown
        _run_per_subject_eval(
            direction=direction,
            model=model,
            norm_stats=norm_stats,
            data_dir=test_dir,
            output_dir=output_dir,
            window_size=window_size,
            batch_size=batch_size,
            evaluator=evaluator,
        )


if __name__ == "__main__":
    main()
