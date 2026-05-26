"""
A/B comparison of pose backends (MediaPipe vs RTMPose) on the same video.

Runs both pose estimators frame-by-frame on a single video and records:
  - Per-frame detection success
  - Per-frame keypoint confidence (mean / min-core)
  - Per-frame inference latency
  - Per-frame extracted feature vector (15 features)
  - Per-frame classifier fall-probability (optional, --model)

Outputs (in --output-dir):
  - per_frame.csv          per-frame metrics for both backends
  - summary.json           aggregate stats (detection rate, FPS, mean conf)
  - side_by_side.mp4       optional side-by-side visualization (--save-video)
  - feature_distributions.png   optional plot (--save-plots)

Usage:
    python scripts/compare_pose_backends.py --source data/sample.mp4
    python scripts/compare_pose_backends.py --source 0 --max-frames 300 --save-video
    python scripts/compare_pose_backends.py --source clip.mp4 \\
        --model models/run5_stride2_aug --save-video --save-plots

Caveat about --model:
    The shipped LSTM was trained on MediaPipe-extracted features. Feeding it
    RTMPose-extracted features (COCO 17 anatomy) is not apples-to-apples — the
    raw probability values are not calibrated. The comparison still surfaces
    *relative* behavior (which backend triggers, which doesn't, ranking).
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import yaml

from src.features.extractor import FeatureExtractor
from src.features.window import SlidingWindow
from src.pose.estimator import PoseEstimator
from src.utils.logger import get_logger
from src.visualization.skeleton import SkeletonDrawer, COLOR_NORMAL

logger = get_logger(__name__)


# Backend configuration entries
_BACKENDS = {
    "mediapipe": {
        "kp_format": "mediapipe",
        "color": (0, 255, 0),       # green text overlay
    },
    "rtmpose": {
        "kp_format": "coco",
        "color": (0, 200, 255),     # cyan text overlay
    },
}


def _load_pose_cfg() -> dict:
    """Load pose config from config.yaml, with safe fallbacks."""
    cfg_path = Path("config/config.yaml")
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("pose", {})


def _build_estimator(backend: str, pose_cfg: dict) -> PoseEstimator:
    """Construct a PoseEstimator for a given backend."""
    return PoseEstimator(
        backend=backend,
        model_complexity=pose_cfg.get("model_complexity", 1),
        min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
        min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
        rtmpose_mode=pose_cfg.get("rtmpose_mode", "balanced"),
        rtmpose_device=pose_cfg.get("rtmpose_device", "cpu"),
    )


def _load_classifier(model_path: Path, num_features: int) -> object:
    """Load a single FallClassifier from a model dir or .pth file."""
    from src.models.classifier import FallClassifier

    cfg = {}
    cfg_path = Path("config/config.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    model_cfg = cfg.get("model", {})

    p = model_path
    if p.is_dir():
        fold0 = p / "fold_0" / "best_model.pth"
        direct = p / "best_model.pth"
        if fold0.exists():
            p = fold0
        elif direct.exists():
            p = direct
        else:
            raise FileNotFoundError(f"No checkpoint in {model_path}")

    return FallClassifier(
        model_path=p,
        architecture="lstm",
        input_size=num_features,
        hidden_size=model_cfg.get("hidden_size", 128),
        num_layers=model_cfg.get("num_layers", 2),
        bidirectional=model_cfg.get("bidirectional", True),
    )


def _draw_overlay(
    frame: np.ndarray,
    backend_name: str,
    detected: bool,
    latency_ms: float,
    mean_conf: float,
    fall_prob: Optional[float],
    color: tuple,
) -> None:
    """Draw a small status banner on the frame (in-place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(
        frame, backend_name.upper(), (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
    )
    status = "DET" if detected else "MISS"
    cv2.putText(
        frame,
        f"{status}  conf={mean_conf:.2f}  {latency_ms:.1f}ms",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    if fall_prob is not None:
        prob_color = (0, 0, 255) if fall_prob > 0.5 else (255, 255, 255)
        cv2.putText(
            frame, f"P(fall)={fall_prob:.2f}",
            (w - 200, 55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, prob_color, 1,
        )


def _summarize(rows: list, backend: str) -> dict:
    """Aggregate per-frame rows for a single backend."""
    sub = [r for r in rows if r["backend"] == backend]
    if not sub:
        return {}

    detected = [r for r in sub if r["detected"]]
    detection_rate = len(detected) / len(sub)
    mean_lat = float(np.mean([r["latency_ms"] for r in sub]))
    p95_lat = float(np.percentile([r["latency_ms"] for r in sub], 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    if detected:
        mean_conf = float(np.mean([r["mean_conf"] for r in detected]))
        min_conf = float(np.mean([r["min_core_conf"] for r in detected]))
    else:
        mean_conf = float("nan")
        min_conf = float("nan")

    fall_probs = [r["fall_prob"] for r in sub if r["fall_prob"] is not None]
    cls_stats = {}
    if fall_probs:
        arr = np.array(fall_probs)
        cls_stats = {
            "frames_classified": len(fall_probs),
            "mean_fall_prob": float(arr.mean()),
            "max_fall_prob": float(arr.max()),
            "frames_above_0p5": int((arr > 0.5).sum()),
            "frames_above_0p75": int((arr > 0.75).sum()),
        }

    return {
        "frames": len(sub),
        "detection_rate": detection_rate,
        "mean_latency_ms": mean_lat,
        "p95_latency_ms": p95_lat,
        "effective_fps": fps,
        "mean_keypoint_confidence": mean_conf,
        "mean_core_min_confidence": min_conf,
        "classifier": cls_stats,
    }


def main(
    source: str,
    output_dir: Path,
    model_path: Optional[Path] = None,
    max_frames: Optional[int] = None,
    save_video: bool = False,
    save_plots: bool = False,
    skip_mediapipe: bool = False,
    skip_rtmpose: bool = False,
    rtmpose_mode_override: Optional[str] = None,
) -> None:
    """Run the A/B comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_cfg = _load_pose_cfg()
    if rtmpose_mode_override is not None:
        pose_cfg["rtmpose_mode"] = rtmpose_mode_override

    enabled = []
    if not skip_mediapipe:
        enabled.append("mediapipe")
    if not skip_rtmpose:
        enabled.append("rtmpose")
    if not enabled:
        logger.error("Both backends skipped — nothing to do.")
        sys.exit(1)

    # Initialize estimators (lazy: only enabled ones)
    estimators = {}
    extractors = {}
    windows = {}
    drawers = {}
    for name in enabled:
        logger.info(f"Initializing backend: {name}")
        estimators[name] = _build_estimator(name, pose_cfg)
        kpf = _BACKENDS[name]["kp_format"]
        extractors[name] = FeatureExtractor(keypoint_format=kpf)
        windows[name] = SlidingWindow(
            window_size=30,
            num_features=FeatureExtractor.NUM_FEATURES,
        )
        drawers[name] = SkeletonDrawer(keypoint_format=kpf)

    # Optional classifier
    classifier = None
    if model_path is not None:
        classifier = _load_classifier(model_path, FeatureExtractor.NUM_FEATURES)
        logger.warning(
            "Using classifier across both backends. The shipped model was "
            "trained on MediaPipe features; RTMPose probabilities will be "
            "miscalibrated. Use the comparison qualitatively."
        )

    # Open video
    cap_source = int(source) if source.isdigit() else source
    if isinstance(cap_source, str) and not Path(cap_source).exists():
        logger.error(f"Source not found: {source}")
        sys.exit(1)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logger.error(f"Cannot open: {source}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    dt = 1.0 / src_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional side-by-side video writer
    writer = None
    if save_video:
        out_path = output_dir / "side_by_side.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        panel_count = len(enabled)
        writer = cv2.VideoWriter(
            str(out_path), fourcc, src_fps,
            (width * panel_count, height),
        )
        logger.info(f"Writing side-by-side video to {out_path}")

    rows: list = []
    prev_keypoints: dict = {name: None for name in enabled}
    prev_velocity: dict = {name: float("nan") for name in enabled}

    frame_idx = 0
    logger.info(
        f"Starting comparison: source={source} backends={enabled} "
        f"max_frames={max_frames}"
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            panels = []
            for name in enabled:
                est = estimators[name]
                ext = extractors[name]
                win = windows[name]
                drw = drawers[name]

                t0 = time.perf_counter()
                kps = est.estimate(frame)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                detected = kps is not None
                row = {
                    "frame_idx": frame_idx,
                    "backend": name,
                    "detected": detected,
                    "latency_ms": latency_ms,
                    "mean_conf": float("nan"),
                    "min_core_conf": float("nan"),
                    "fall_prob": None,
                }
                # Per-feature columns initialized to NaN
                for i in range(FeatureExtractor.NUM_FEATURES):
                    row[f"f{i:02d}"] = float("nan")

                panel = frame.copy()
                if detected:
                    h, w = frame.shape[:2]
                    pixel_kps = est.to_pixel_coords(kps, w, h)
                    drw.draw(panel, pixel_kps, color=COLOR_NORMAL)

                    feats = ext.extract(
                        kps,
                        prev_keypoints=prev_keypoints[name],
                        dt=dt,
                        prev_velocity=prev_velocity[name],
                    )
                    feats_clean = np.nan_to_num(feats, nan=0.0)
                    prev_velocity[name] = feats[3]
                    prev_keypoints[name] = kps

                    win.push(feats_clean)
                    for i, v in enumerate(feats):
                        row[f"f{i:02d}"] = float(v) if not np.isnan(v) else float("nan")

                    # Visibility stats
                    row["mean_conf"] = float(feats[13]) if not np.isnan(feats[13]) else float("nan")
                    row["min_core_conf"] = float(feats[14]) if not np.isnan(feats[14]) else float("nan")

                    if classifier is not None and win.is_ready:
                        seq = win.get_array()
                        _, fall_p = classifier.predict(seq)
                        row["fall_prob"] = float(fall_p)
                else:
                    prev_keypoints[name] = None
                    prev_velocity[name] = float("nan")

                _draw_overlay(
                    panel,
                    backend_name=name,
                    detected=detected,
                    latency_ms=latency_ms,
                    mean_conf=row["mean_conf"] if not np.isnan(row["mean_conf"]) else 0.0,
                    fall_prob=row["fall_prob"],
                    color=_BACKENDS[name]["color"],
                )
                panels.append(panel)
                rows.append(row)

            if writer is not None:
                composite = np.hstack(panels)
                writer.write(composite)

            frame_idx += 1
            if frame_idx % 50 == 0:
                logger.info(f"Processed {frame_idx} frames")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for est in estimators.values():
            est.release()

    if not rows:
        logger.error("No frames processed.")
        sys.exit(1)

    # Per-frame CSV
    import csv
    csv_path = output_dir / "per_frame.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info(f"Per-frame metrics: {csv_path}")

    # Summary JSON
    summary = {
        "source": str(source),
        "frames_processed": frame_idx,
        "source_fps": src_fps,
        "resolution": [width, height],
        "rtmpose_mode": pose_cfg.get("rtmpose_mode", "balanced"),
        "classifier_loaded": classifier is not None,
        "model_path": str(model_path) if model_path else None,
        "backends": {name: _summarize(rows, name) for name in enabled},
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    # Console report
    print("\n" + "=" * 70)
    print("POSE BACKEND COMPARISON")
    print("=" * 70)
    print(f"Source:              {source}")
    print(f"Frames processed:    {frame_idx}")
    print(f"Source FPS:          {src_fps:.1f}")
    print()
    for name in enabled:
        s = summary["backends"][name]
        if not s:
            continue
        print(f"[{name}]")
        print(f"  detection rate     = {s['detection_rate'] * 100:.1f}%")
        print(f"  mean latency       = {s['mean_latency_ms']:.1f} ms"
              f"  (p95 {s['p95_latency_ms']:.1f} ms)")
        print(f"  effective FPS      = {s['effective_fps']:.1f}")
        print(f"  mean keypoint conf = {s['mean_keypoint_confidence']:.3f}")
        print(f"  min-core conf      = {s['mean_core_min_confidence']:.3f}")
        if s["classifier"]:
            c = s["classifier"]
            print(f"  classifier:")
            print(f"    frames classified  = {c['frames_classified']}")
            print(f"    mean P(fall)       = {c['mean_fall_prob']:.3f}")
            print(f"    max P(fall)        = {c['max_fall_prob']:.3f}")
            print(f"    frames > 0.5       = {c['frames_above_0p5']}")
            print(f"    frames > 0.75      = {c['frames_above_0p75']}")
        print()

    if save_plots:
        _save_plots(rows, enabled, output_dir)


def _save_plots(rows: list, enabled: list, output_dir: Path) -> None:
    """Save per-feature distribution comparison plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots")
        return

    feature_names = [
        "torso_incl", "hip_sh_angle", "bbox_aspect", "com_vel", "com_acc",
        "head_toe", "sh_ankle", "L_knee", "R_knee", "L_hip", "R_hip",
        "wrist_hip", "body_spread", "mean_vis", "min_core_vis",
    ]

    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    by_backend: dict = defaultdict(list)
    for r in rows:
        if not r["detected"]:
            continue
        by_backend[r["backend"]].append(
            [r[f"f{i:02d}"] for i in range(FeatureExtractor.NUM_FEATURES)]
        )

    colors = {"mediapipe": "#1f77b4", "rtmpose": "#ff7f0e"}

    for i, name in enumerate(feature_names):
        ax = axes[i]
        for backend in enabled:
            if not by_backend[backend]:
                continue
            arr = np.array(by_backend[backend])[:, i]
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue
            ax.hist(
                arr, bins=30, alpha=0.5,
                label=backend, color=colors.get(backend, None),
            )
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out_path = output_dir / "feature_distributions.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Feature distributions: {out_path}")

    # Latency boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    data = []
    labels = []
    for backend in enabled:
        sub = [r["latency_ms"] for r in rows if r["backend"] == backend]
        if sub:
            data.append(sub)
            labels.append(backend)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Latency (ms / frame)")
    ax.set_title("Pose inference latency comparison")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_boxplot.png", dpi=120)
    plt.close()
    logger.info(f"Latency boxplot: {output_dir / 'latency_boxplot.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B compare MediaPipe vs RTMPose on a video."
    )
    parser.add_argument(
        "--source", required=True,
        help="Video file path or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/pose_comparison"),
        help="Where to write per_frame.csv, summary.json, etc.",
    )
    parser.add_argument(
        "--model", type=Path, default=None,
        help="Optional model dir (LOSO root) or .pth for classifier comparison.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Process at most N frames (useful for quick smoke runs).",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Write a side-by-side .mp4 visualization.",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save feature-distribution and latency plots.",
    )
    parser.add_argument(
        "--no-mediapipe", action="store_true",
        help="Skip MediaPipe (test RTMPose only).",
    )
    parser.add_argument(
        "--no-rtmpose", action="store_true",
        help="Skip RTMPose (test MediaPipe only).",
    )
    parser.add_argument(
        "--rtmpose-mode", choices=["lightweight", "balanced", "performance"],
        default=None,
        help="Override rtmpose_mode from config.",
    )
    args = parser.parse_args()

    main(
        source=args.source,
        output_dir=args.output_dir,
        model_path=args.model,
        max_frames=args.max_frames,
        save_video=args.save_video,
        save_plots=args.save_plots,
        skip_mediapipe=args.no_mediapipe,
        skip_rtmpose=args.no_rtmpose,
        rtmpose_mode_override=args.rtmpose_mode,
    )
