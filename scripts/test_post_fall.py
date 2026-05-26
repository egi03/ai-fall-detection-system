"""
Run the MotionMonitor over a real video and print its frame-by-frame
output, then summarize how the classify_severity() helper would label
the event under different recovery assumptions.

Best used on a clip that contains both motion and stillness -- e.g. a
fall followed by lying still, or a person walking then sitting down.

Usage:
    python scripts/test_post_fall.py --source clip.mp4
    python scripts/test_post_fall.py --source clip.mp4 --motion-threshold 0.005
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.alarm.motion_monitor import MotionMonitor, classify_severity
from src.pose.estimator import PoseEstimator


def main(
    source: str,
    motion_threshold: float,
    history_frames: int,
    fps: float,
    max_frames: int,
    verbose: bool,
) -> int:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open source: {source}")
        return 1

    pose = PoseEstimator(backend="mediapipe")
    monitor = MotionMonitor(
        history_frames=history_frames,
        motion_threshold=motion_threshold,
    )

    dt = 1.0 / fps
    frames = 0
    longest_still_run = 0
    current_run = 0
    motion_scores = []

    print(f"Running on {source} (motion_threshold={motion_threshold}, "
          f"history={history_frames} frames) ...")
    if verbose:
        print(f"  {'frame':>5}  {'motion_score':>13}  {'is_still':>9}  {'streak':>6}")
        print("  " + "-" * 45)

    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        kp = pose.estimate(frame)
        monitor.update(kp)
        motion_scores.append(monitor.motion_score)

        if monitor.is_still:
            current_run += 1
            longest_still_run = max(longest_still_run, current_run)
        else:
            current_run = 0

        if verbose:
            print(
                f"  {frames:>5d}  {monitor.motion_score:>13.5f}  "
                f"{str(monitor.is_still):>9}  {monitor.still_streak:>6d}"
            )

    cap.release()
    pose.release()

    if not motion_scores:
        print("ERROR: no frames processed")
        return 2

    longest_still_seconds = longest_still_run * dt

    print()
    print("--- Motion summary --------------------------------------")
    print(f"  frames processed         : {frames}")
    print(f"  max motion_score         : {max(motion_scores):.5f}")
    print(f"  min motion_score         : {min(motion_scores):.5f}")
    print(f"  longest still run (fr.)  : {longest_still_run}")
    print(f"  longest still run (s)    : {longest_still_seconds:.2f}")
    print()
    print("--- Severity classification (hypothetical labels) -------")
    for rec_flag, label in [(False, "no recovery"), (True, "subject recovered")]:
        sev = classify_severity(
            still_seconds=longest_still_seconds, recovered=rec_flag
        )
        print(f"  if {label:<20} -> severity = {sev}")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Test MotionMonitor + severity classifier on a video.")
    p.add_argument("--source", required=True)
    p.add_argument("--motion-threshold", type=float, default=0.003)
    p.add_argument("--history-frames", type=int, default=15)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--max-frames", type=int, default=600)
    p.add_argument("--verbose", action="store_true", help="Print per-frame motion score table")
    args = p.parse_args()
    sys.exit(
        main(
            args.source,
            args.motion_threshold,
            args.history_frames,
            args.fps,
            args.max_frames,
            args.verbose,
        )
    )
