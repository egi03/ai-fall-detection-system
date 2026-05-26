"""
Concatenated ADL stream analysis for realistic false alarm rate estimation.

Addresses a limitation in event_level_alarm_metrics.json: the original analysis
resets the FSM between every clip, yielding only 0.0618 hours total ADL monitoring.
In real deployment the FSM never resets — it runs continuously across activity clips.

This script simulates that continuous monitoring scenario by:
  1. Loading all ADL/fall test sequences per LOSO fold (left-out subject).
  2. Sorting sequences by sequence_id for a deterministic, reproducible order.
  3. Concatenating window probabilities into ONE long probability stream per fold;
     the FSM state persists across clip boundaries.
  4. Running EMA(alpha=0.3) + persistence=10 + threshold=0.75 FSM on that stream,
     counting every alarm event that fires (with a 30-second / simulated cooldown).
  5. Computing FAR/hour = n_alarm_events / duration_hours from the total windows
     and stride/fps parameters.

The same procedure is applied to fall probability streams to compute concatenated
event-level sensitivity (how many of the 5 fold fall-streams trigger >= 1 alarm).

Results saved to: results/v4_analysis/concatenated_stream/concatenated_stream_results.json

Usage:
    python scripts/v4_concatenated_stream.py

Reference: research/5.2 - False alarm rate per hour is a primary metric for
continuous-monitoring fall detection systems. Short clip evaluation severely
underestimates FAR because FSM state resets between clips inflate the denominator
and prevent carry-over high-confidence regions from triggering alarms.
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

# ── FSM / model constants ────────────────────────────────────────────────────
FSM_THRESHOLD = 0.75         # alarm confidence threshold (paper default)
FSM_PERSISTENCE = 10         # consecutive EMA-above-threshold windows to fire
FSM_EMA_ALPHA = 0.3          # EMA smoothing factor
COOLDOWN_WINDOWS = 225       # 30 s cooldown at stride=2, fps=15: 30*15/2 = 225 windows
TARGET_FPS = 15.0
STRIDE = 2                   # frames between windows (matches run5)
WINDOW_SIZE = 30
SEED = 42
BATCH_SIZE = 64

# Model architecture (Run5 BiLSTM)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
INPUT_SIZE = 15

# Clip-by-clip FAR from event_level_alarm_metrics.json (for comparison printout)
CLIP_BY_CLIP_FAR = 113.31


# ── FSM simulation ───────────────────────────────────────────────────────────

def simulate_fsm_continuous(
    probs: np.ndarray,
    threshold: float = FSM_THRESHOLD,
    persistence: int = FSM_PERSISTENCE,
    ema_alpha: float = FSM_EMA_ALPHA,
    cooldown_windows: int = COOLDOWN_WINDOWS,
) -> Dict:
    """
    Simulate EMA+persistence FSM on a long concatenated probability stream.

    Unlike the per-clip FSM in event_level_alarm_metrics.py, this function:
      - Counts EVERY alarm event that fires (not just the first).
      - Enforces a cooldown of `cooldown_windows` windows between alarms so
        that a sustained high-probability region does not produce unbounded alarms.
      - Resets the EMA and persistence counter after each alarm fires (the FSM
        enters a cooldown period, matching the AlarmDetector behaviour).

    Parameters
    ----------
    probs : np.ndarray
        Fall probability for each consecutive window, shape (T,).
    threshold : float
        EMA probability threshold to count a window as fall-candidate.
    persistence : int
        Number of consecutive above-threshold windows required before alarm.
    ema_alpha : float
        EMA smoothing factor alpha.
    cooldown_windows : int
        Minimum number of windows between consecutive alarm events.

    Returns
    -------
    dict
        alarm_count : int
            Total number of alarm events fired.
        alarm_window_indices : list of int
            Window index at which each alarm fired.
        duration_windows : int
            Total number of windows processed.
    """
    ema_prob = 0.0
    consecutive = 0
    alarm_count = 0
    alarm_indices: List[int] = []
    cooldown_remaining = 0  # windows remaining in post-alarm cooldown

    for i, p in enumerate(probs):
        # Update EMA regardless of cooldown
        ema_prob = ema_alpha * p + (1 - ema_alpha) * ema_prob

        if cooldown_remaining > 0:
            # During cooldown: suppress detection, decay counter
            cooldown_remaining -= 1
            consecutive = 0
            # Note: EMA continues updating so the smoothed value is accurate
            # when the cooldown expires (matches AlarmDetector behaviour).
            continue

        if ema_prob >= threshold:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= persistence:
            # Alarm fires
            alarm_count += 1
            alarm_indices.append(i)
            # Reset FSM state and enter cooldown
            consecutive = 0
            ema_prob = 0.0
            cooldown_remaining = cooldown_windows

    return {
        "alarm_count": alarm_count,
        "alarm_window_indices": alarm_indices,
        "duration_windows": len(probs),
    }


def simulate_fsm_fall_stream(
    probs: np.ndarray,
    threshold: float = FSM_THRESHOLD,
    persistence: int = FSM_PERSISTENCE,
    ema_alpha: float = FSM_EMA_ALPHA,
) -> bool:
    """
    Return True if the concatenated fall probability stream triggers >= 1 alarm.

    Simpler than simulate_fsm_continuous: for fall streams we only need to know
    whether at least one alarm fires (sensitivity). No cooldown needed here.

    Parameters
    ----------
    probs : np.ndarray
        Fall probability stream, shape (T,).
    threshold, persistence, ema_alpha : FSM parameters.

    Returns
    -------
    bool
        True if at least one alarm event fires.
    """
    ema_prob = 0.0
    consecutive = 0
    for p in probs:
        ema_prob = ema_alpha * p + (1 - ema_alpha) * ema_prob
        if ema_prob >= threshold:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= persistence:
            return True
    return False


# ── Model inference helpers ──────────────────────────────────────────────────

def load_fold_model(fold_dir: Path) -> FallDetectionLSTM:
    """
    Load a BiLSTM fold model from disk.

    Parameters
    ----------
    fold_dir : Path
        Directory containing best_model.pth.

    Returns
    -------
    FallDetectionLSTM
        Model in eval mode on CPU.
    """
    model = FallDetectionLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=0.0,  # inference: no dropout
        bidirectional=BIDIRECTIONAL,
    )
    ckpt_path = fold_dir / "best_model.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def get_sequence_probs(
    keypoints: np.ndarray,
    label: int,
    extractor: FeatureExtractor,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    model: FallDetectionLSTM,
    dt: float,
) -> np.ndarray:
    """
    Extract features, build windows, and run model inference for one sequence.

    Parameters
    ----------
    keypoints : np.ndarray
        Raw keypoint array of shape (T, N, C).
    label : int
        Sequence label (used only for window extraction label assignment).
    extractor : FeatureExtractor
        Configured feature extractor.
    norm_mean, norm_std : np.ndarray
        Per-feature normalization stats from training fold.
    model : FallDetectionLSTM
        Trained model in eval mode.
    dt : float
        Time step in seconds (1 / TARGET_FPS).

    Returns
    -------
    np.ndarray
        Fall probabilities for each window, shape (n_windows,).
        Returns empty array if no windows can be extracted.
    """
    feature_seq = extractor.extract_sequence(keypoints, dt=dt)
    feature_seq = np.nan_to_num(feature_seq, nan=0.0)

    # Reshape to (T, 15, 1) for extract_windows compatibility
    feature_3d = feature_seq[:, :, np.newaxis]
    windows, window_labels = extract_windows(
        feature_3d, label, window_size=WINDOW_SIZE, stride=STRIDE
    )

    if not windows:
        return np.array([], dtype=np.float32)

    seqs = np.array(
        [w.reshape(w.shape[0], -1) for w in windows], dtype=np.float32
    )
    seqs = (seqs - norm_mean) / (norm_std + 1e-8)

    ds = FallDetectionDataset(seqs, np.array(window_labels, dtype=np.int64))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_probs: List[float] = []
    with torch.no_grad():
        for x, _ in loader:
            p = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(p.tolist())

    return np.array(all_probs, dtype=np.float32)


# ── Main analysis ────────────────────────────────────────────────────────────

def run_concatenated_stream_analysis(
    config: dict,
    model_base_dir: Path,
    output_dir: Path,
) -> Dict:
    """
    Perform the concatenated ADL/fall stream FSM analysis across all LOSO folds.

    For each fold (left-out subject):
      1. Collect ADL and fall sequences for the test subject.
      2. Sort by sequence_id (deterministic order).
      3. Concatenate window probabilities into one long stream per class.
      4. Run FSM simulation and count alarm events.
      5. Compute per-fold FAR and sensitivity.

    Aggregate across folds to produce overall FAR and sensitivity.

    Parameters
    ----------
    config : dict
        Parsed config.yaml.
    model_base_dir : Path
        Base directory containing fold_0/ … fold_N/ subdirectories.
    output_dir : Path
        Directory where results JSON will be saved.

    Returns
    -------
    dict
        Summary statistics for printing and saving.
    """
    urfd_dir = Path(config["paths"]["data_processed"]) / "urfd"
    metadata_path = urfd_dir / "metadata.json"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    subjects = sorted(set(s["subject_id"] for s in meta["sequences"]))
    splitter = SubjectSplitter(subjects, seed=SEED)
    folds = splitter.get_loso_folds()

    extractor = FeatureExtractor(keypoint_format="mediapipe", confidence_threshold=0.5)
    dt = 1.0 / TARGET_FPS
    seconds_per_window = STRIDE / TARGET_FPS  # 2/15 ≈ 0.1333 s

    # ── Per-fold results ─────────────────────────────────────────────────────
    per_fold_results: List[Dict] = []

    # Totals across all folds
    total_adl_windows = 0
    total_adl_alarm_events = 0
    total_fall_streams = 0
    fall_streams_triggered = 0

    print(f"  {'Fold':<6} {'Subject':<10} {'ADL win':>8} {'Alarms':>7} "
          f"{'Hours':>8} {'FAR/h':>8} {'Fall win':>9} {'Fall trig':>10}")
    print("  " + "-" * 75)

    for fold_idx, fold in enumerate(folds):
        test_subject = fold["test"][0]
        fold_dir = model_base_dir / f"fold_{fold_idx}"

        if not (fold_dir / "best_model.pth").exists():
            print(f"  Fold {fold_idx}: model missing, skipping")
            continue

        norm_mean = np.load(str(fold_dir / "norm_mean.npy"))
        norm_std = np.load(str(fold_dir / "norm_std.npy"))
        model = load_fold_model(fold_dir)

        # Collect all test sequences for this subject, sorted by sequence_id
        test_sequences = sorted(
            [s for s in meta["sequences"] if s["subject_id"] == test_subject],
            key=lambda s: s["sequence_id"],
        )

        # Separate into ADL and fall
        adl_sequences = [s for s in test_sequences if s["label"] == 0]
        fall_sequences = [s for s in test_sequences if s["label"] == 1]

        # ── Build concatenated ADL probability stream ─────────────────────────
        adl_prob_streams: List[np.ndarray] = []

        for seq_meta in adl_sequences:
            seq_id = seq_meta["sequence_id"]
            kp_path = urfd_dir / "keypoints_raw" / f"{seq_id}.npy"
            if not kp_path.exists():
                continue

            keypoints = np.load(str(kp_path))
            seq_probs = get_sequence_probs(
                keypoints, seq_meta["label"], extractor,
                norm_mean, norm_std, model, dt
            )
            if len(seq_probs) > 0:
                adl_prob_streams.append(seq_probs)

        if not adl_prob_streams:
            print(f"  Fold {fold_idx} ({test_subject}): no ADL sequences found")
            continue

        # Concatenate — FSM sees this as one continuous stream
        adl_stream = np.concatenate(adl_prob_streams, axis=0)

        adl_fsm_result = simulate_fsm_continuous(
            adl_stream,
            threshold=FSM_THRESHOLD,
            persistence=FSM_PERSISTENCE,
            ema_alpha=FSM_EMA_ALPHA,
            cooldown_windows=COOLDOWN_WINDOWS,
        )

        adl_windows = adl_fsm_result["duration_windows"]
        adl_alarms = adl_fsm_result["alarm_count"]
        adl_hours = adl_windows * seconds_per_window / 3600.0
        adl_far = adl_alarms / max(adl_hours, 1e-9)

        total_adl_windows += adl_windows
        total_adl_alarm_events += adl_alarms

        # ── Build concatenated fall probability stream ────────────────────────
        fall_prob_streams: List[np.ndarray] = []

        for seq_meta in fall_sequences:
            seq_id = seq_meta["sequence_id"]
            kp_path = urfd_dir / "keypoints_raw" / f"{seq_id}.npy"
            if not kp_path.exists():
                continue

            keypoints = np.load(str(kp_path))
            seq_probs = get_sequence_probs(
                keypoints, seq_meta["label"], extractor,
                norm_mean, norm_std, model, dt
            )
            if len(seq_probs) > 0:
                fall_prob_streams.append(seq_probs)

        fall_windows = 0
        fall_triggered = False
        if fall_prob_streams:
            fall_stream = np.concatenate(fall_prob_streams, axis=0)
            fall_windows = len(fall_stream)
            fall_triggered = simulate_fsm_fall_stream(
                fall_stream,
                threshold=FSM_THRESHOLD,
                persistence=FSM_PERSISTENCE,
                ema_alpha=FSM_EMA_ALPHA,
            )

        total_fall_streams += 1
        if fall_triggered:
            fall_streams_triggered += 1

        per_fold_results.append({
            "fold": fold_idx,
            "subject": test_subject,
            "adl_sequences": len(adl_sequences),
            "fall_sequences": len(fall_sequences),
            "adl_total_windows": adl_windows,
            "adl_alarm_count": adl_alarms,
            "adl_alarm_window_indices": adl_fsm_result["alarm_window_indices"],
            "adl_hours": round(adl_hours, 5),
            "adl_far_per_hour": round(adl_far, 2),
            "fall_total_windows": fall_windows,
            "fall_stream_triggered": fall_triggered,
        })

        print(f"  Fold {fold_idx} ({test_subject}): "
              f"{adl_windows:>8} {adl_alarms:>7} "
              f"{adl_hours:>8.4f} {adl_far:>8.1f} "
              f"{fall_windows:>9} {'YES' if fall_triggered else 'no':>10}")

    # ── Overall metrics ───────────────────────────────────────────────────────
    total_hours = total_adl_windows * seconds_per_window / 3600.0
    overall_far = total_adl_alarm_events / max(total_hours, 1e-9)
    overall_fall_sensitivity = fall_streams_triggered / max(total_fall_streams, 1)

    summary = {
        "method": "concatenated_stream",
        "description": (
            "FSM state persists across clips within each fold. "
            "Simulates continuous deployment where ADL clips are "
            "observed back-to-back without alarm system reset."
        ),
        "fsm_params": {
            "threshold": FSM_THRESHOLD,
            "persistence_frames": FSM_PERSISTENCE,
            "ema_alpha": FSM_EMA_ALPHA,
            "stride": STRIDE,
            "fps": TARGET_FPS,
            "cooldown_windows": COOLDOWN_WINDOWS,
            "cooldown_seconds": COOLDOWN_WINDOWS * STRIDE / TARGET_FPS,
        },
        "model": {
            "name": "BiLSTM run5_stride2_aug",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "bidirectional": BIDIRECTIONAL,
        },
        "total_adl_windows": total_adl_windows,
        "total_adl_hours": round(total_hours, 5),
        "total_adl_alarm_events": total_adl_alarm_events,
        "overall_adl_far_per_hour": round(overall_far, 2),
        "total_fall_streams": total_fall_streams,
        "fall_streams_triggered": fall_streams_triggered,
        "concatenated_fall_sensitivity": round(overall_fall_sensitivity, 4),
        "comparison": {
            "clip_by_clip_far_per_hour": CLIP_BY_CLIP_FAR,
            "clip_by_clip_adl_hours": 0.0618,
            "concatenated_adl_hours": round(total_hours, 5),
            "far_ratio_clip_over_concat": (
                round(CLIP_BY_CLIP_FAR / max(overall_far, 1e-9), 3)
                if overall_far > 0
                else None
            ),
        },
        "per_fold": per_fold_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "concatenated_stream_results.json"
    with open(str(out_path), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    """Entry point: load config, run analysis, print summary."""
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_dir = Path("models/run5_stride2_aug")
    output_dir = Path("results/v4_analysis/concatenated_stream")

    print("=" * 65)
    print("CONCATENATED ADL STREAM ANALYSIS (BiLSTM, run5)")
    print("=" * 65)
    print(f"  Model:         {model_dir}")
    print(f"  FSM params:    threshold={FSM_THRESHOLD}, persistence={FSM_PERSISTENCE}, "
          f"EMA alpha={FSM_EMA_ALPHA}")
    print(f"  Cooldown:      {COOLDOWN_WINDOWS} windows "
          f"({COOLDOWN_WINDOWS * STRIDE / TARGET_FPS:.0f} s)")
    print(f"  Stride:        {STRIDE} frames -> {STRIDE / TARGET_FPS:.4f} s/window")
    print()
    print("Per fold:")

    summary = run_concatenated_stream_analysis(config, model_dir, output_dir)

    print()
    print("=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)
    print(f"  Total ADL windows:          {summary['total_adl_windows']}")
    print(f"  Total ADL monitoring hours: {summary['total_adl_hours']:.5f} h "
          f"({summary['total_adl_hours'] * 60:.2f} min)")
    print(f"  Total ADL alarm events:     {summary['total_adl_alarm_events']}")
    print(f"  Overall FAR (concat):       {summary['overall_adl_far_per_hour']:.2f} / hour")
    print()
    print(f"  Concatenated fall sensitivity: {summary['concatenated_fall_sensitivity']:.4f} "
          f"({summary['fall_streams_triggered']}/{summary['total_fall_streams']} fold streams triggered)")
    print()
    print("Comparison:")
    print(f"  Clip-by-clip FAR:  {CLIP_BY_CLIP_FAR:.1f} / hour  "
          f"({summary['comparison']['clip_by_clip_adl_hours']:.4f} h total)")
    print(f"  Concatenated FAR:  {summary['overall_adl_far_per_hour']:.2f} / hour  "
          f"({summary['total_adl_hours']:.5f} h total)")

    far_ratio = summary["comparison"]["far_ratio_clip_over_concat"]
    if far_ratio is not None:
        print(f"  Clip/concat ratio: {far_ratio:.2f}x  "
              f"(clip-by-clip over-estimates FAR by this factor)")
    else:
        print("  No alarms fired in concatenated stream (FAR = 0.0 / hour)")

    print()
    print(f"  Results saved: {output_dir / 'concatenated_stream_results.json'}")
    print()

    # Per-fold detail
    print("Per-fold summary:")
    print(f"  {'Fold':<6} {'Subj':<6} {'Hours':>8} {'Alarms':>7} {'FAR/h':>8} "
          f"{'Fall trig':>10}")
    print("  " + "-" * 50)
    for fr in summary["per_fold"]:
        print(f"  Fold {fr['fold']} ({fr['subject']}): "
              f"{fr['adl_hours']:>8.4f} {fr['adl_alarm_count']:>7} "
              f"{fr['adl_far_per_hour']:>8.1f} "
              f"{'YES' if fr['fall_stream_triggered'] else 'no':>10}")


if __name__ == "__main__":
    main()
