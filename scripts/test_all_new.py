"""
One-shot validator for every feature added in the recent batch.

Runs:
  1. pytest on the four new test modules (no video needed)
  2. test_imminent.py             (no video needed)
  3. test_smoothing.py            (skipped if no --source)
  4. test_calibration.py          (skipped if no --source)
  5. test_post_fall.py            (skipped if no --source)
  6. SQLite severity refinement spot-check (queries logs/events.db)

Each stage prints PASS / FAIL / SKIP. The script exits 0 only if every
non-skipped stage reports PASS.

Usage:
    python scripts/test_all_new.py
    python scripts/test_all_new.py --source clip.mp4
    python scripts/test_all_new.py --source clip.mp4 --max-frames 200
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(f" {title}")
    print("=" * 60)


def _run(cmd: list[str]) -> int:
    print(f"  $ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(ROOT))
    return int(res.returncode)


def stage_unit_tests() -> str:
    _section("[1/6] Unit tests for the new modules")
    rc = _run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_smoothing.py",
        "tests/test_calibration.py",
        "tests/test_imminent.py",
        "tests/test_motion_monitor.py",
    ])
    return "PASS" if rc == 0 else "FAIL"


def stage_imminent_demo() -> str:
    _section("[2/6] FSM scenarios for IMMINENT state")
    rc = _run([sys.executable, "scripts/test_imminent.py"])
    return "PASS" if rc == 0 else "FAIL"


def stage_smoothing(source: str | None, max_frames: int) -> str:
    _section("[3/6] OneEuro smoothing on a video")
    if source is None:
        print("  SKIP -- no --source provided")
        return "SKIP"
    rc = _run([
        sys.executable, "scripts/test_smoothing.py",
        "--source", source,
        "--max-frames", str(max_frames),
    ])
    return "PASS" if rc == 0 else "FAIL"


def stage_calibration(source: str | None, max_frames: int) -> str:
    _section("[4/6] Per-camera calibration on a video")
    if source is None:
        print("  SKIP -- no --source provided")
        return "SKIP"
    rc = _run([
        sys.executable, "scripts/test_calibration.py",
        "--source", source,
        "--max-frames", str(max_frames),
    ])
    return "PASS" if rc == 0 else "FAIL"


def stage_post_fall(source: str | None, max_frames: int) -> str:
    _section("[5/6] Post-fall motion monitor on a video")
    if source is None:
        print("  SKIP -- no --source provided")
        return "SKIP"
    rc = _run([
        sys.executable, "scripts/test_post_fall.py",
        "--source", source,
        "--max-frames", str(max_frames),
    ])
    return "PASS" if rc == 0 else "FAIL"


def stage_severity_db() -> str:
    _section("[6/6] Severity refinement column in logs/events.db")
    db = ROOT / "logs" / "events.db"
    if not db.exists():
        print(f"  SKIP -- {db} does not exist yet. Run the demo on a fall video first.")
        return "SKIP"
    from src.alarm.logger import EventLogger

    el = EventLogger(db)
    try:
        events = el.get_events(limit=5)
    finally:
        el.close()
    if not events:
        print("  SKIP -- no events in DB yet.")
        return "SKIP"

    print(f"  Latest {len(events)} event(s):")
    refined_count = 0
    for e in events:
        meta = json.loads(e["metadata"]) if e["metadata"] else {}
        has_still = "still_seconds" in meta
        is_refined = e["severity"] in {"SEVERE", "MODERATE", "MINOR"}
        refined_count += int(has_still and is_refined)
        print(
            f"    id={e['event_id']:>3}  severity={e['severity']:<9} "
            f"conf={e['confidence']:.3f}  "
            f"refined={'yes' if (has_still and is_refined) else 'no'}"
        )
    if refined_count == 0:
        print("  WARN -- no events show severity refinement. Either the demo "
              "was never run with severity enabled, or no fall was confirmed.")
        return "SKIP"
    print(f"  {refined_count} of {len(events)} event(s) carry refined severity + motion metadata.")
    return "PASS"


def main() -> int:
    p = argparse.ArgumentParser(description="Validate every recently added feature.")
    p.add_argument("--source", default=None, help="Video for stages 3-5 (optional).")
    p.add_argument("--max-frames", type=int, default=300)
    args = p.parse_args()

    results = [
        ("Unit tests",            stage_unit_tests()),
        ("IMMINENT FSM demo",     stage_imminent_demo()),
        ("OneEuro smoothing",     stage_smoothing(args.source, args.max_frames)),
        ("Per-camera calibration",stage_calibration(args.source, args.max_frames)),
        ("Post-fall motion",      stage_post_fall(args.source, args.max_frames)),
        ("Severity refinement DB",stage_severity_db()),
    ]

    _section("Summary")
    width = max(len(name) for name, _ in results) + 2
    for name, status in results:
        marker = {"PASS": "[OK]", "FAIL": "[XX]", "SKIP": "."}[status]
        print(f"  {marker} {name:<{width}} {status}")

    failed = [n for n, s in results if s == "FAIL"]
    skipped = [n for n, s in results if s == "SKIP"]
    print()
    print(f"  {len(results) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped.")
    if skipped and args.source is None:
        print("  Hint: pass --source <video.mp4> to run the video-dependent stages.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
