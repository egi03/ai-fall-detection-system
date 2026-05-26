"""
Event-level alarm metrics: Simulate the EMA+persistence FSM on LOSO fold predictions.

Computes event-level false alarms per hour and time-to-detection (latency)
using the actual alarm FSM (EMA smoothing + 10-frame persistence).

Reference: paper_review/v1 weakness W4 - "event-level metrics with the proposed
alarm FSM are not reported (false alarms per hour, time-to-detection)."

Usage:
    python scripts/event_level_alarm_metrics.py
"""

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
from src.training.dataset import FallDetectionDataset, extract_windows
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── FSM constants ───────────────────────────────────────────────────────────
FSM_THRESHOLD = 0.75        # alarm confidence threshold
FSM_PERSISTENCE = 10        # frames needed to trigger alarm
FSM_EMA_ALPHA = 0.3         # smoothing
COOLDOWN_SECONDS = 30.0     # seconds between alarms
TARGET_FPS = 15.0
STRIDE = 2                  # frames between windows (matches run5)
WINDOW_SIZE = 30
SEED = 42


def simulate_fsm_on_sequence(
    probs: np.ndarray,
    label: int,
    stride: int = STRIDE,
    fps: float = TARGET_FPS,
) -> Dict:
    """
    Simulate EMA+persistence FSM on a sequence of window probabilities.

    Parameters
    ----------
    probs : array of shape (T,)
        Fall probability for each window in the sequence.
    label : int
        Ground-truth label (0=ADL, 1=fall).
    stride : int
        Frames between consecutive windows.
    fps : float
        Target FPS.

    Returns
    -------
    dict with keys: alarm_triggered, detection_latency_s, n_windows
    """
    seconds_per_window = stride / fps  # time between windows
    ema_prob = 0.0
    consecutive_above_thresh = 0
    alarm_triggered = False
    detection_window_idx = None

    for i, p in enumerate(probs):
        ema_prob = FSM_EMA_ALPHA * p + (1 - FSM_EMA_ALPHA) * ema_prob

        if ema_prob >= FSM_THRESHOLD:
            consecutive_above_thresh += 1
        else:
            consecutive_above_thresh = 0

        if consecutive_above_thresh >= FSM_PERSISTENCE:
            alarm_triggered = True
            detection_window_idx = i
            break

    result = {
        "alarm_triggered": alarm_triggered,
        "n_windows": len(probs),
        "label": label,
    }

    if alarm_triggered and label == 1:
        # Time from first window until alarm triggered (index of first qualifying window)
        # The alarm fires at window detection_window_idx (0-indexed)
        # The first window of the fall may start before the trigger point
        latency_windows = detection_window_idx - (FSM_PERSISTENCE - 1)  # approx onset
        detection_latency_s = detection_window_idx * seconds_per_window
        result["detection_latency_s"] = detection_latency_s

    return result


def load_fold_model(fold_dir: Path, hidden_size: int, num_layers: int, bidirectional: bool):
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


def run_event_level_analysis(
    config: dict,
    model_dir: Path,
    output_dir: Path,
) -> Dict:
    """Run per-sequence FSM simulation and compute event-level metrics."""
    cfg_model = config["model"]
    cfg_feat = config["features"]
    cfg_train = config["training"]

    hidden_size = cfg_model["hidden_size"]
    num_layers = cfg_model["num_layers"]
    bidirectional = cfg_model.get("bidirectional", False)

    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    with open(urfd_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    urfd_subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))
    splitter = SubjectSplitter(urfd_subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)
    dt = 1.0 / TARGET_FPS

    # Accumulators
    adl_alarm_events: List[Dict] = []   # event-level FP alarms
    fall_results: List[Dict] = []       # per-sequence fall detection

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        fold_dir = model_dir / f"fold_{fold_idx}"
        if not (fold_dir / "best_model.pth").exists():
            logger.warning(f"Fold {fold_idx} model missing, skipping")
            continue

        norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
        norm_std = np.load(str(fold_dir / "norm_std.npy"))
        model = load_fold_model(fold_dir, hidden_size, num_layers, bidirectional)

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

            seqs = np.array([w.reshape(w.shape[0], -1) for w in windows], dtype=np.float32)
            seqs = (seqs - norm_mean) / (norm_std + 1e-8)

            ds = FallDetectionDataset(seqs, np.array(window_labels))
            loader = DataLoader(ds, batch_size=cfg_train["batch_size"], shuffle=False)

            all_probs = []
            with torch.no_grad():
                for x, _ in loader:
                    probs = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())

            all_probs = np.array(all_probs)

            # Simulate FSM
            result = simulate_fsm_on_sequence(all_probs, label, stride=STRIDE, fps=TARGET_FPS)
            result["sequence_id"] = seq_id
            result["subject"] = test_subject
            result["n_windows"] = len(all_probs)

            # Duration in hours of this sequence
            seq_duration_s = len(all_probs) * STRIDE / TARGET_FPS
            result["duration_s"] = seq_duration_s

            if label == 0:
                adl_alarm_events.append(result)
            else:
                fall_results.append(result)

    # ── Compute event-level statistics ──────────────────────────────────────
    n_adl = len(adl_alarm_events)
    n_fall = len(fall_results)

    # Event-level FP: ADL sequences that trigger alarm
    fp_events = [r for r in adl_alarm_events if r["alarm_triggered"]]
    n_fp_events = len(fp_events)

    # Total ADL monitoring hours
    total_adl_hours = sum(r["duration_s"] for r in adl_alarm_events) / 3600.0

    # Event-level FAR
    event_far = n_fp_events / max(total_adl_hours, 1e-8)

    # Fall detection (sequence-level) using FSM
    fall_detected = [r for r in fall_results if r["alarm_triggered"]]
    n_fall_detected = len(fall_detected)
    event_sensitivity = n_fall_detected / max(n_fall, 1)

    # Detection latency
    latencies = [r.get("detection_latency_s") for r in fall_detected if "detection_latency_s" in r]
    mean_latency = float(np.mean(latencies)) if latencies else float("nan")
    median_latency = float(np.median(latencies)) if latencies else float("nan")
    p25_latency = float(np.percentile(latencies, 25)) if latencies else float("nan")
    p75_latency = float(np.percentile(latencies, 75)) if latencies else float("nan")

    # Per-subject event-level stats
    subjects = sorted(set(r["subject"] for r in adl_alarm_events))
    per_subject = {}
    for subj in subjects:
        subj_adl = [r for r in adl_alarm_events if r["subject"] == subj]
        subj_fall = [r for r in fall_results if r["subject"] == subj]
        subj_fp = [r for r in subj_adl if r["alarm_triggered"]]
        subj_hours = sum(r["duration_s"] for r in subj_adl) / 3600.0
        subj_det = [r for r in subj_fall if r["alarm_triggered"]]
        per_subject[subj] = {
            "n_adl_sequences": len(subj_adl),
            "n_fall_sequences": len(subj_fall),
            "event_fp_count": len(subj_fp),
            "event_sensitivity": len(subj_det) / max(len(subj_fall), 1),
            "adl_monitoring_hours": round(subj_hours, 4),
            "event_far": round(len(subj_fp) / max(subj_hours, 1e-8), 2),
            "fp_sequences": [r["sequence_id"] for r in subj_fp],
        }

    summary = {
        "n_adl_sequences": n_adl,
        "n_fall_sequences": n_fall,
        "total_adl_monitoring_hours": round(total_adl_hours, 4),
        "event_level_fp_count": n_fp_events,
        "event_level_far_per_hour": round(event_far, 2),
        "event_level_sensitivity": round(event_sensitivity, 4),
        "detection_latency_s": {
            "mean": round(mean_latency, 3),
            "median": round(median_latency, 3),
            "p25": round(p25_latency, 3),
            "p75": round(p75_latency, 3),
        },
        "fp_event_sequences": [r["sequence_id"] for r in fp_events],
        "per_subject": per_subject,
        "fsm_params": {
            "threshold": FSM_THRESHOLD,
            "persistence_frames": FSM_PERSISTENCE,
            "ema_alpha": FSM_EMA_ALPHA,
            "stride": STRIDE,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "event_level_alarm_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {out_file}")
    return summary


def main() -> None:
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_dir = Path("models/run5_stride2_aug")
    output_dir = Path("results/event_level_alarm_metrics")

    print("Running event-level alarm FSM simulation...")
    print(f"  Model: {model_dir}")
    print(f"  FSM: threshold={FSM_THRESHOLD}, persistence={FSM_PERSISTENCE}, EMA alpha={FSM_EMA_ALPHA}")
    print()

    summary = run_event_level_analysis(config, model_dir, output_dir)

    print("=" * 60)
    print("EVENT-LEVEL ALARM METRICS (URFD LOSO, Run 5 BiLSTM)")
    print("=" * 60)
    print(f"  ADL sequences:           {summary['n_adl_sequences']}")
    print(f"  Fall sequences:          {summary['n_fall_sequences']}")
    print(f"  Total ADL hours:         {summary['total_adl_monitoring_hours']:.4f} h")
    print()
    print(f"  Event-level FP count:    {summary['event_level_fp_count']}")
    print(f"  Event-level FAR:         {summary['event_level_far_per_hour']:.2f} / hour")
    print(f"  Event-level sensitivity: {summary['event_level_sensitivity']:.4f}")
    print()
    print(f"  Detection latency (s):   mean={summary['detection_latency_s']['mean']:.3f}, "
          f"median={summary['detection_latency_s']['median']:.3f}, "
          f"IQR=[{summary['detection_latency_s']['p25']:.3f}, {summary['detection_latency_s']['p75']:.3f}]")
    print()
    print("Per-subject:")
    for subj, stats in summary["per_subject"].items():
        print(f"  {subj}: {stats['event_fp_count']} FP events, "
              f"FAR={stats['event_far']:.1f}/h, "
              f"sens={stats['event_sensitivity']:.2f}")
    print()
    print(f"FP event sequences: {summary['fp_event_sequences']}")


if __name__ == "__main__":
    main()
