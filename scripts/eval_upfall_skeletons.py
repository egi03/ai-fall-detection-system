"""
Convert Zenodo UP-Fall 3D skeleton data and evaluate with pre-trained URFD model.

The Zenodo dataset (DOI: 10.5281/zenodo.12773013) contains 3D skeleton
keypoints extracted from the UP-Fall Detection Dataset for 5 subjects,
activities 1-5 (falls only), both camera views, with frame-level labels.

Pipeline:
  1. Parse Zenodo CSVs -> numpy arrays (T, 33, 4) with visibility=1.0
  2. Create processed data directory + metadata.json
  3. Load pre-trained URFD model (cross_urfd_full_r5)
  4. Extract features + windows using same pipeline as URFD
  5. Evaluate: per-fall-type, per-camera, per-subject sensitivity
  6. Save all results to results/upfall_cross_eval/

Usage:
    python scripts/eval_upfall_skeletons.py
    python scripts/eval_upfall_skeletons.py --model-dir models/cross_urfd_full_r5
    python scripts/eval_upfall_skeletons.py --camera 1  # Camera1 only

Reference: research/5.1, research/1.1
"""

import argparse
import csv
import json
import logging
import re
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
from src.training.dataset import (
    FallDetectionDataset,
    build_dataset_from_processed,
    extract_windows,
)
from src.training.evaluate import Evaluator
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# UP-Fall activity labels (1-5 are falls used in Zenodo data)
UP_FALL_ACTIVITIES = {
    1: "Falling forward (hands)",
    2: "Falling forward (knees)",
    3: "Falling backwards",
    4: "Falling sideward",
    5: "Falling sitting empty chair",
}

# Regex patterns for Zenodo file naming variants
FILENAME_PATTERNS = [
    # Standard: C1S1A1T1.csv
    re.compile(r"C(\d+)S(\d+)A(\d+)T(\d+)\.csv"),
    # Underscored: C2S1_A1_T2.csv
    re.compile(r"C(\d+)S(\d+)_A(\d+)_T(\d+)\.csv"),
    # Missing C prefix: 3S3A4T1.csv (cam=3)
    re.compile(r"(\d+)S(\d+)A(\d+)T(\d+)\.csv"),
    # Missing cam digit: CS3A4T2.csv (cam=0 fallback)
    re.compile(r"CS(\d+)A(\d+)T(\d+)\.csv"),
]


def _build_eval_model(
    model_dir: Path,
    hidden_size: int,
    num_layers: int,
    bidirectional: bool,
) -> torch.nn.Module:
    """Load model from model_dir, auto-detecting architecture from run_meta.json."""
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
            dropout=0.0,
        )
    else:
        model = FallDetectionLSTM(
            input_size=FeatureExtractor.NUM_FEATURES,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0,
            bidirectional=bidirectional,
        )
    ckpt = torch.load(str(model_dir / "best_model.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    logger.info(f"Loaded model from {model_dir} (architecture={architecture})")
    return model


def parse_filename(fname: str) -> Optional[Dict]:
    """
    Parse a Zenodo UP-Fall skeleton CSV filename.

    Returns dict with cam, subj, act, trial or None if unparseable.
    """
    for i, pat in enumerate(FILENAME_PATTERNS):
        m = pat.match(fname)
        if m:
            groups = m.groups()
            if i == 3:
                # CS3A4T2.csv -> cam unknown, subj=groups[0], act=groups[1], trial=groups[2]
                return {
                    "cam": 0,
                    "subj": int(groups[0]),
                    "act": int(groups[1]),
                    "trial": int(groups[2]),
                }
            else:
                return {
                    "cam": int(groups[0]),
                    "subj": int(groups[1]),
                    "act": int(groups[2]),
                    "trial": int(groups[3]),
                }
    return None


def convert_csv_to_npy(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a Zenodo skeleton CSV to numpy arrays.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with Joint{N}_{X,Y,Z} columns and LABEL column.

    Returns
    -------
    tuple of (keypoints, labels)
        keypoints: shape (T, 33, 4) — X, Y, Z, visibility=1.0
        labels: shape (T,) — frame-level labels (0=non-impact, 1=impact)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Extract joint coordinates
    n_joints = 33
    keypoints = np.zeros((len(df), n_joints, 4), dtype=np.float32)

    for j in range(n_joints):
        col_x = f"Joint{j+1}_X"
        col_y = f"Joint{j+1}_Y"
        col_z = f"Joint{j+1}_Z"
        if col_x in df.columns:
            keypoints[:, j, 0] = df[col_x].values
            keypoints[:, j, 1] = df[col_y].values
            keypoints[:, j, 2] = df[col_z].values
            keypoints[:, j, 3] = 1.0  # dummy visibility
        else:
            keypoints[:, j, :] = np.nan

    labels = df["LABEL"].values.astype(np.int64) if "LABEL" in df.columns else np.ones(len(df), dtype=np.int64)

    return keypoints, labels


def convert_all_zenodo(
    skeleton_dir: Path,
    output_dir: Path,
    camera_filter: Optional[int] = None,
) -> List[Dict]:
    """
    Convert all Zenodo CSVs to processed .npy format.

    Parameters
    ----------
    skeleton_dir : Path
        Directory containing extracted Zenodo CSVs.
    output_dir : Path
        Output directory for processed keypoints + metadata.
    camera_filter : int, optional
        If set, only include sequences from this camera.

    Returns
    -------
    list of dict
        Metadata records for all converted sequences.
    """
    kp_dir = output_dir / "keypoints_raw"
    kp_dir.mkdir(parents=True, exist_ok=True)

    records = []
    csv_files = sorted(skeleton_dir.rglob("*.csv"))

    for csv_path in csv_files:
        info = parse_filename(csv_path.name)
        if info is None:
            logger.warning(f"Skipping unparseable: {csv_path.name}")
            continue

        if camera_filter is not None and info["cam"] != camera_filter:
            continue

        # Activity must be 1-5 (falls)
        if info["act"] not in UP_FALL_ACTIVITIES:
            continue

        keypoints, frame_labels = convert_csv_to_npy(csv_path)

        seq_id = f"upfall_S{info['subj']:02d}_A{info['act']:02d}_T{info['trial']:02d}_C{info['cam']}"

        # Save raw keypoints
        npy_path = kp_dir / f"{seq_id}.npy"
        np.save(str(npy_path), keypoints)

        # Frame-level annotation: find fall start/end
        fall_indices = np.where(frame_labels == 1)[0]
        if len(fall_indices) > 0:
            start_frame = int(fall_indices[0])
            end_frame = int(fall_indices[-1])
        else:
            start_frame = 0
            end_frame = len(keypoints) - 1

        records.append({
            "sequence_id": seq_id,
            "label": 1,  # All sequences are falls
            "subject_id": f"S{info['subj']:02d}",
            "camera": f"cam{info['cam']}",
            "activity": info["act"],
            "activity_name": UP_FALL_ACTIVITIES[info["act"]],
            "trial": info["trial"],
            "num_frames": len(keypoints),
            "fall_frames": int(frame_labels.sum()),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

        logger.info(
            f"  Converted {seq_id}: {len(keypoints)} frames, "
            f"fall={frame_labels.sum()}/{len(frame_labels)}"
        )

    # Save metadata
    metadata = {
        "dataset": "up_fall_zenodo",
        "source": "Zenodo DOI:10.5281/zenodo.12773013",
        "note": "Falls only (A1-A5), no ADL sequences",
        "stats": {
            "total_sequences": len(records),
            "total_frames": sum(r["num_frames"] for r in records),
        },
        "sequences": records,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Converted {len(records)} sequences -> {output_dir}")
    return records


def evaluate_cross_dataset(
    model_dir: Path,
    upfall_dir: Path,
    output_dir: Path,
    config: dict,
    camera_filter: Optional[int] = None,
) -> Dict:
    """
    Evaluate a pre-trained URFD model on UP-Fall skeleton data.

    Parameters
    ----------
    model_dir : Path
        Directory with best_model.pth + norm stats.
    upfall_dir : Path
        Processed UP-Fall directory with keypoints_raw/ and metadata.json.
    output_dir : Path
        Directory to save evaluation results.
    config : dict
        Configuration dict.
    camera_filter : int, optional
        Evaluate only this camera view.

    Returns
    -------
    dict
        Aggregated evaluation metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model metadata
    run_meta_path = model_dir / "run_meta.json"
    if run_meta_path.exists():
        with open(run_meta_path, "r") as f:
            run_meta = json.load(f)
        hidden_size = run_meta.get("hidden_size", config["model"]["hidden_size"])
        bidirectional = run_meta.get("bidirectional", config["model"].get("bidirectional", False))
        window_size = run_meta.get("window_size", config["features"]["window_size"])
    else:
        hidden_size = config["model"]["hidden_size"]
        bidirectional = config["model"].get("bidirectional", False)
        window_size = config["features"]["window_size"]

    # Load normalization stats from source model
    norm_mean = np.load(str(model_dir / "norm_mean.npy"))
    norm_std = np.load(str(model_dir / "norm_std.npy"))
    norm_stats = (norm_mean, norm_std)

    # Load UP-Fall metadata
    with open(upfall_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    # Filter by camera if requested
    sequences = meta["sequences"]
    if camera_filter is not None:
        sequences = [s for s in sequences if s["camera"] == f"cam{camera_filter}"]

    all_subjects = sorted(set(s["subject_id"] for s in sequences))
    logger.info(
        f"Evaluating on UP-Fall: {len(sequences)} sequences, "
        f"subjects={all_subjects}, camera_filter={camera_filter}"
    )

    # Build dataset using our standard pipeline
    # All UP-Fall sequences are falls (label=1)
    upfall_seqs, upfall_labels, _ = build_dataset_from_processed(
        processed_dir=upfall_dir,
        subject_ids=all_subjects,
        window_size=window_size,
        stride=2,  # Same as Run5
        positive_threshold=0.5,
        norm_stats=norm_stats,
    )

    logger.info(
        f"UP-Fall windows: {len(upfall_seqs)} total, "
        f"{(upfall_labels == 1).sum()} fall, {(upfall_labels == 0).sum()} non-fall"
    )

    # Load model (auto-detect architecture from run_meta.json)
    model = _build_eval_model(model_dir, hidden_size, config["model"]["num_layers"], bidirectional)

    # Inference
    test_ds = FallDetectionDataset(upfall_seqs, upfall_labels)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds, all_probs, all_true = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            all_true.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true = np.array(all_true)

    # Compute metrics
    evaluator = Evaluator(output_dir=output_dir)
    metrics = evaluator.compute_metrics(all_true, all_preds, all_probs)
    evaluator.save_results(metrics, output_dir / "metrics.json")
    evaluator.plot_confusion_matrix(all_true, all_preds, output_dir / "confusion_matrix.png")
    evaluator.plot_roc_curve(all_true, all_probs, output_dir / "roc_curve.png")

    # Save raw predictions
    np.save(str(output_dir / "all_probs.npy"), all_probs)
    np.save(str(output_dir / "all_true.npy"), all_true)

    # Threshold sweep
    from sklearn.metrics import confusion_matrix as cm, f1_score
    sweep = []
    for t in np.arange(0.20, 0.81, 0.05):
        pred_t = (all_probs >= t).astype(int)
        tn, fp, fn, tp = cm(all_true, pred_t).ravel() if len(np.unique(all_true)) > 1 else (0, 0, (pred_t == 0).sum(), (pred_t == 1).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(all_true, pred_t, zero_division=0)
        sweep.append({"threshold": round(float(t), 2), "sensitivity": sens, "specificity": spec, "f1": f1})
    with open(output_dir / "threshold_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    return metrics, all_probs, all_true, sequences


def per_activity_analysis(
    upfall_dir: Path,
    model_dir: Path,
    output_dir: Path,
    config: dict,
) -> Dict:
    """
    Run per-fall-type and per-camera sensitivity analysis.

    For each activity (A1-A5) and each camera (C1, C2), evaluate the model
    and report sensitivity at multiple thresholds.
    """
    with open(upfall_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    norm_mean = np.load(str(model_dir / "norm_mean.npy"))
    norm_std = np.load(str(model_dir / "norm_std.npy"))

    run_meta_path = model_dir / "run_meta.json"
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        hidden_size = run_meta.get("hidden_size", config["model"]["hidden_size"])
        bidirectional = run_meta.get("bidirectional", False)
        window_size = run_meta.get("window_size", config["features"]["window_size"])
    else:
        hidden_size = config["model"]["hidden_size"]
        bidirectional = config["model"].get("bidirectional", False)
        window_size = config["features"]["window_size"]

    # Load model once (auto-detect architecture from run_meta.json)
    model = _build_eval_model(model_dir, hidden_size, config["model"]["num_layers"], bidirectional)

    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)
    dt = 1.0 / 15.0
    kp_dir = upfall_dir / "keypoints_raw"

    results_by_activity = {}
    results_by_camera = {}
    results_by_subject = {}
    per_sequence_results = []

    for seq_meta in meta["sequences"]:
        seq_id = seq_meta["sequence_id"]
        npy_path = kp_dir / f"{seq_id}.npy"
        if not npy_path.exists():
            continue

        keypoints = np.load(str(npy_path))
        feature_seq = extractor.extract_sequence(keypoints, dt=dt)
        feature_seq = np.nan_to_num(feature_seq, nan=0.0)

        # Extract windows
        annotation = {"start_frame": seq_meta.get("start_frame", 0),
                       "end_frame": seq_meta.get("end_frame", len(keypoints))}
        feature_3d = feature_seq[:, :, np.newaxis]
        windows, labels = extract_windows(
            feature_3d, label=1, window_size=window_size, stride=2,
            annotation=annotation, positive_threshold=0.5,
        )
        if not windows:
            continue

        seqs_arr = np.array([w.reshape(w.shape[0], -1) for w in windows], dtype=np.float32)
        labels_arr = np.array(labels, dtype=np.int64)

        # Normalize with URFD stats
        seqs_arr = (seqs_arr - norm_mean) / norm_std

        # Inference
        ds = FallDetectionDataset(seqs_arr, labels_arr)
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        probs_list = []
        with torch.no_grad():
            for x, _ in loader:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[:, 1]
                probs_list.extend(probs.numpy())

        probs_arr = np.array(probs_list)
        mean_prob = float(probs_arr.mean())
        max_prob = float(probs_arr.max())
        # Detection: at least one window above threshold
        detected_050 = int(max_prob >= 0.50)
        detected_030 = int(max_prob >= 0.30)

        act = seq_meta["activity"]
        cam = seq_meta["camera"]
        subj = seq_meta["subject_id"]

        per_sequence_results.append({
            "sequence_id": seq_id,
            "subject": subj,
            "activity": act,
            "activity_name": seq_meta["activity_name"],
            "camera": cam,
            "num_windows": len(windows),
            "fall_windows": int(labels_arr.sum()),
            "mean_prob": round(mean_prob, 4),
            "max_prob": round(max_prob, 4),
            "detected_t050": detected_050,
            "detected_t030": detected_030,
        })

        # Aggregate by activity
        if act not in results_by_activity:
            results_by_activity[act] = {"detected_050": 0, "detected_030": 0, "total": 0, "probs": []}
        results_by_activity[act]["total"] += 1
        results_by_activity[act]["detected_050"] += detected_050
        results_by_activity[act]["detected_030"] += detected_030
        results_by_activity[act]["probs"].append(mean_prob)

        # Aggregate by camera
        if cam not in results_by_camera:
            results_by_camera[cam] = {"detected_050": 0, "detected_030": 0, "total": 0, "probs": []}
        results_by_camera[cam]["total"] += 1
        results_by_camera[cam]["detected_050"] += detected_050
        results_by_camera[cam]["detected_030"] += detected_030
        results_by_camera[cam]["probs"].append(mean_prob)

        # Aggregate by subject
        if subj not in results_by_subject:
            results_by_subject[subj] = {"detected_050": 0, "detected_030": 0, "total": 0, "probs": []}
        results_by_subject[subj]["total"] += 1
        results_by_subject[subj]["detected_050"] += detected_050
        results_by_subject[subj]["detected_030"] += detected_030
        results_by_subject[subj]["probs"].append(mean_prob)

    # Save per-sequence results
    csv_path = output_dir / "per_sequence_results.csv"
    with open(csv_path, "w", newline="") as f:
        if per_sequence_results:
            w = csv.DictWriter(f, fieldnames=per_sequence_results[0].keys())
            w.writeheader()
            w.writerows(per_sequence_results)

    # Build summary
    summary = {
        "per_activity": {},
        "per_camera": {},
        "per_subject": {},
    }

    print(f"\n{'='*70}")
    print(f"  Per-Fall-Type Sensitivity (URFD model -> UP-Fall)")
    print(f"{'='*70}")
    print(f"{'Activity':<35} {'N':>4} {'Sens@0.5':>9} {'Sens@0.3':>9} {'Mean Prob':>10}")
    print("-" * 70)
    for act in sorted(results_by_activity.keys()):
        r = results_by_activity[act]
        sens_050 = r["detected_050"] / r["total"] if r["total"] > 0 else 0
        sens_030 = r["detected_030"] / r["total"] if r["total"] > 0 else 0
        mean_p = np.mean(r["probs"]) if r["probs"] else 0
        name = UP_FALL_ACTIVITIES.get(act, f"A{act}")
        summary["per_activity"][str(act)] = {
            "name": name, "total": r["total"],
            "sensitivity_050": round(sens_050, 3),
            "sensitivity_030": round(sens_030, 3),
            "mean_prob": round(float(mean_p), 4),
        }
        print(f"A{act}: {name:<30} {r['total']:>4} {sens_050:>9.3f} {sens_030:>9.3f} {mean_p:>10.4f}")

    print(f"\n{'='*70}")
    print(f"  Per-Camera Sensitivity")
    print(f"{'='*70}")
    for cam in sorted(results_by_camera.keys()):
        r = results_by_camera[cam]
        sens_050 = r["detected_050"] / r["total"] if r["total"] > 0 else 0
        sens_030 = r["detected_030"] / r["total"] if r["total"] > 0 else 0
        mean_p = np.mean(r["probs"]) if r["probs"] else 0
        summary["per_camera"][cam] = {
            "total": r["total"],
            "sensitivity_050": round(sens_050, 3),
            "sensitivity_030": round(sens_030, 3),
            "mean_prob": round(float(mean_p), 4),
        }
        print(f"  {cam}: N={r['total']:>3}  sens@0.5={sens_050:.3f}  sens@0.3={sens_030:.3f}  mean_prob={mean_p:.4f}")

    print(f"\n{'='*70}")
    print(f"  Per-Subject Sensitivity")
    print(f"{'='*70}")
    for subj in sorted(results_by_subject.keys()):
        r = results_by_subject[subj]
        sens_050 = r["detected_050"] / r["total"] if r["total"] > 0 else 0
        sens_030 = r["detected_030"] / r["total"] if r["total"] > 0 else 0
        mean_p = np.mean(r["probs"]) if r["probs"] else 0
        summary["per_subject"][subj] = {
            "total": r["total"],
            "sensitivity_050": round(sens_050, 3),
            "sensitivity_030": round(sens_030, 3),
            "mean_prob": round(float(mean_p), 4),
        }
        print(f"  {subj}: N={r['total']:>3}  sens@0.5={sens_050:.3f}  sens@0.3={sens_030:.3f}  mean_prob={mean_p:.4f}")

    # Overall
    total_seqs = sum(r["total"] for r in results_by_activity.values())
    total_det_050 = sum(r["detected_050"] for r in results_by_activity.values())
    total_det_030 = sum(r["detected_030"] for r in results_by_activity.values())
    overall_sens_050 = total_det_050 / total_seqs if total_seqs > 0 else 0
    overall_sens_030 = total_det_030 / total_seqs if total_seqs > 0 else 0
    summary["overall"] = {
        "total_sequences": total_seqs,
        "sensitivity_050": round(overall_sens_050, 3),
        "sensitivity_030": round(overall_sens_030, 3),
    }
    print(f"\n  OVERALL: {total_seqs} seqs, sens@0.5={overall_sens_050:.3f}, sens@0.3={overall_sens_030:.3f}")

    with open(output_dir / "detailed_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate URFD model on UP-Fall Zenodo skeleton data"
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "models" / "cross_urfd_full_r5",
        help="Pre-trained model directory",
    )
    parser.add_argument(
        "--skeleton-dir", type=Path,
        default=PROJECT_ROOT / "data" / "data" / "raw" / "up_fall_skeletons",
        help="Directory with Zenodo skeleton CSVs",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "results" / "upfall_cross_eval",
        help="Results output directory",
    )
    parser.add_argument(
        "--camera", type=int, default=None, choices=[1, 2],
        help="Evaluate only this camera (default: both)",
    )
    args = parser.parse_args()

    with open(PROJECT_ROOT / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Step 1: Convert Zenodo CSVs to processed format
    processed_dir = PROJECT_ROOT / "data" / "data" / "processed" / "up_fall_zenodo"
    logger.info("=" * 60)
    logger.info("Step 1: Converting Zenodo CSVs -> processed .npy")
    logger.info("=" * 60)
    records = convert_all_zenodo(
        skeleton_dir=args.skeleton_dir,
        output_dir=processed_dir,
        camera_filter=args.camera,
    )

    if not records:
        logger.error("No sequences converted. Check skeleton directory.")
        sys.exit(1)

    # Step 2: Aggregate evaluation (window-level metrics)
    logger.info("=" * 60)
    logger.info("Step 2: Window-level cross-dataset evaluation")
    logger.info("=" * 60)
    metrics, probs, true_labels, sequences = evaluate_cross_dataset(
        model_dir=args.model_dir,
        upfall_dir=processed_dir,
        output_dir=args.output_dir,
        config=config,
        camera_filter=args.camera,
    )

    print(f"\n{'='*60}")
    print(f"  Window-level metrics (URFD -> UP-Fall)")
    print(f"{'='*60}")
    for k in ["sensitivity", "specificity", "accuracy", "f1", "auc_roc"]:
        if k in metrics:
            print(f"  {k}: {metrics[k]:.4f}")

    # Step 3: Per-activity, per-camera, per-subject analysis
    logger.info("=" * 60)
    logger.info("Step 3: Detailed per-sequence analysis")
    logger.info("=" * 60)
    summary = per_activity_analysis(
        upfall_dir=processed_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config=config,
    )

    logger.info(f"\nAll results saved to: {args.output_dir}")
    print(f"\nDone! Results at: {args.output_dir}")


if __name__ == "__main__":
    main()
