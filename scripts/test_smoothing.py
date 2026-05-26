"""
Exercise the OneEuro keypoint smoother on a real video.

Runs the same pose stream twice -- once raw, once through the smoother --
and reports the standard deviation of per-frame keypoint displacements
for both. Lower std on the smoothed stream means the filter is doing
its job. Also reports peak displacement (which should survive smoothing
so falls aren't damped).

Usage:
    python scripts/test_smoothing.py --source clip.mp4
    python scripts/test_smoothing.py --source 0 --max-frames 200
    python scripts/test_smoothing.py --source clip.mp4 --min-cutoff 0.5 --beta 0.05
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.pose.estimator import PoseEstimator
from src.pose.smoothing import KeypointSmoother


def _displacement(prev: np.ndarray, curr: np.ndarray, conf_threshold: float = 0.5) -> float:
    if prev is None or curr is None or prev.shape != curr.shape:
        return float("nan")
    if curr.shape[1] >= 4:
        mask = (curr[:, 3] >= conf_threshold) & (prev[:, 3] >= conf_threshold)
    else:
        mask = np.ones(curr.shape[0], dtype=bool)
    if int(mask.sum()) < 4:
        return float("nan")
    delta = curr[mask, :2] - prev[mask, :2]
    return float(np.linalg.norm(delta, axis=1).mean())


def main(source: str, max_frames: int, min_cutoff: float, beta: float, backend: str) -> int:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {source}")
        return 1

    pose = PoseEstimator(backend=backend)
    num_kp = 33 if backend == "mediapipe" else 17
    smoother = KeypointSmoother(
        num_keypoints=num_kp, fps=15.0, min_cutoff=min_cutoff, beta=beta
    )

    raw_displacements = []
    smoothed_displacements = []
    prev_raw = None
    prev_smoothed = None
    frames = 0
    detected = 0
    t0 = time.monotonic()

    print(f"Running on {source} (backend={backend}, max_frames={max_frames}) ...")
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        kp = pose.estimate(frame)
        if kp is None:
            prev_raw = None
            prev_smoothed = None
            continue
        detected += 1

        smoothed_kp = smoother.smooth(kp)
        raw_displacements.append(_displacement(prev_raw, kp))
        smoothed_displacements.append(_displacement(prev_smoothed, smoothed_kp))
        prev_raw, prev_smoothed = kp, smoothed_kp

    cap.release()
    pose.release()
    elapsed = time.monotonic() - t0

    raw_arr = np.array([d for d in raw_displacements if np.isfinite(d)])
    sm_arr = np.array([d for d in smoothed_displacements if np.isfinite(d)])

    if raw_arr.size == 0:
        print("ERROR: no valid keypoint pairs to compare")
        return 2

    print()
    print("--- Results ---------------------------------------------")
    print(f"  frames processed     : {frames}")
    print(f"  frames with person   : {detected}")
    print(f"  comparable pairs     : {raw_arr.size}")
    print(f"  elapsed              : {elapsed:.2f}s ({frames / elapsed:.1f} FPS)")
    print()
    print(f"  RAW      mean disp.  : {raw_arr.mean():.5f}")
    print(f"  RAW      std  disp.  : {raw_arr.std():.5f}")
    print(f"  RAW      max  disp.  : {raw_arr.max():.5f}")
    print()
    print(f"  SMOOTHED mean disp.  : {sm_arr.mean():.5f}")
    print(f"  SMOOTHED std  disp.  : {sm_arr.std():.5f}")
    print(f"  SMOOTHED max  disp.  : {sm_arr.max():.5f}")
    print()
    jitter_reduction = (1 - sm_arr.std() / raw_arr.std()) * 100 if raw_arr.std() > 0 else 0
    peak_retention = (sm_arr.max() / raw_arr.max()) * 100 if raw_arr.max() > 0 else 0
    print(f"  > jitter reduction   : {jitter_reduction:.1f}%  (higher is better)")
    print(f"  > peak retention     : {peak_retention:.1f}%  (close to 100 = falls survive)")
    print()
    if jitter_reduction > 10.0:
        print("  PASS: smoother is reducing per-frame jitter.")
    else:
        print("  WARN: low jitter reduction -- try higher min_cutoff or lower beta.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test OneEuro smoothing on a video.")
    p.add_argument("--source", required=True, help="Video file or webcam index (0).")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--min-cutoff", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.007)
    p.add_argument("--backend", default="mediapipe", choices=["mediapipe", "yolo_pose", "rtmpose"])
    args = p.parse_args()
    sys.exit(main(args.source, args.max_frames, args.min_cutoff, args.beta, args.backend))
