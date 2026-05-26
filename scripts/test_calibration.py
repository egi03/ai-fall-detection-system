"""
Exercise the per-camera BaselineCalibrator on a real video.

Runs the full pose -> feature extraction pipeline against a video and
shows what the calibrator learns from the warmup period plus how it
transforms the velocity / acceleration features during the rest of
the clip.

Best run on a clip that starts with the subject standing still
(typical webcam intro) before any action.

Usage:
    python scripts/test_calibration.py --source clip.mp4
    python scripts/test_calibration.py --source clip.mp4 --warmup-seconds 3.0 --k 1.5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.features.calibration import BaselineCalibrator
from src.features.extractor import FeatureExtractor
from src.pose.estimator import PoseEstimator


def main(source: str, warmup_seconds: float, k: float, fps: float, max_frames: int) -> int:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {source}")
        return 1

    pose = PoseEstimator(backend="mediapipe")
    extractor = FeatureExtractor(keypoint_format="mediapipe")
    warmup_frames = max(2, int(warmup_seconds * fps))
    calibrator = BaselineCalibrator(
        feature_indices=[3, 4], warmup_frames=warmup_frames, k=k
    )

    dt = 1.0 / fps
    raw_velocity, raw_accel = [], []
    cal_velocity, cal_accel = [], []
    suppressed = 0

    prev_kp = None
    prev_velocity = float("nan")
    frames = 0

    print(f"Running on {source} (warmup={warmup_frames} frames @ {fps:g} FPS, k={k}) ...")
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        kp = pose.estimate(frame)
        if kp is None:
            prev_kp = None
            prev_velocity = float("nan")
            continue

        feats = extractor.extract(
            kp, prev_keypoints=prev_kp, dt=dt, prev_velocity=prev_velocity
        )
        feats = np.nan_to_num(feats, nan=0.0)
        prev_velocity = feats[3]
        prev_kp = kp

        calibrator.observe(feats)
        applied = calibrator.apply(feats)

        raw_velocity.append(float(feats[3]))
        raw_accel.append(float(feats[4]))
        cal_velocity.append(float(applied[3]))
        cal_accel.append(float(applied[4]))

        if calibrator.is_ready and (
            applied[3] != feats[3] or applied[4] != feats[4]
        ):
            suppressed += 1

    cap.release()
    pose.release()

    if not calibrator.is_ready:
        print(
            f"ERROR: calibrator never reached warmup ({len(raw_velocity)} valid samples "
            f"out of {warmup_frames} required). Try a longer clip or shorter --warmup-seconds."
        )
        return 2

    mean = calibrator.baseline_mean
    std = calibrator.baseline_std
    raw_v = np.array(raw_velocity)
    raw_a = np.array(raw_accel)
    cal_v = np.array(cal_velocity)
    cal_a = np.array(cal_accel)

    print()
    print("--- Calibration baseline (learned from warmup) ----------")
    print(f"  velocity (idx 3) : mean={mean[0]:+.5f}   std={std[0]:.5f}")
    print(f"  accel    (idx 4) : mean={mean[1]:+.5f}   std={std[1]:.5f}")
    print(f"  suppression band : +/-{k} x std")
    print()
    print("--- Inference distribution (post-warmup frames) ---------")
    print(f"  RAW velocity     : mean={raw_v.mean():+.5f}   std={raw_v.std():.5f}   "
          f"max|.|={np.abs(raw_v).max():.5f}")
    print(f"  CAL velocity     : mean={cal_v.mean():+.5f}   std={cal_v.std():.5f}   "
          f"max|.|={np.abs(cal_v).max():.5f}")
    print(f"  RAW accel        : mean={raw_a.mean():+.5f}   std={raw_a.std():.5f}   "
          f"max|.|={np.abs(raw_a).max():.5f}")
    print(f"  CAL accel        : mean={cal_a.mean():+.5f}   std={cal_a.std():.5f}   "
          f"max|.|={np.abs(cal_a).max():.5f}")
    print()
    print(f"  frames suppressed/altered : {suppressed} / {frames}")
    print()
    if cal_v.std() < raw_v.std() and cal_a.std() < raw_a.std():
        print("  PASS: calibrator is narrowing the kinematic feature distribution.")
    else:
        print("  WARN: std did not decrease -- either the clip is very active "
              "during warmup, or k is too small.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test BaselineCalibrator on a video.")
    p.add_argument("--source", required=True)
    p.add_argument("--warmup-seconds", type=float, default=3.0)
    p.add_argument("--k", type=float, default=1.5)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max-frames", type=int, default=600)
    args = p.parse_args()
    sys.exit(main(args.source, args.warmup_seconds, args.k, args.fps, args.max_frames))
