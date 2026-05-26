"""
Drive the AlarmDetector FSM with synthetic confidence sequences to
inspect the new IMMINENT pre-fall warning state.

No video required -- purely exercises the state machine. Prints a table
of (frame, confidence, smoothed_confidence, state) for each scenario.

Usage:
    python scripts/test_imminent.py
    python scripts/test_imminent.py --confidence-threshold 0.7 --persistence 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alarm.detector import AlarmDetector, FallState


SCENARIOS = {
    "Slow ramp up to a fall": [0.05] * 3 + [0.20, 0.35, 0.50, 0.62, 0.75, 0.88, 0.92, 0.92, 0.92],
    "False alarm (yellow, then cleared)": [0.05, 0.20, 0.50, 0.62, 0.65, 0.50, 0.35, 0.10, 0.05],
    "Sudden hard fall": [0.05, 0.05, 0.05, 0.95, 0.95, 0.95, 0.95, 0.95],
    "Flickering near the gate": [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6],
}


_STATE_TAG = {
    FallState.NORMAL:          "         NORMAL",
    FallState.IMMINENT:        "[yellow] IMMINENT",
    FallState.IMPACT_DETECTED: "[orange] IMPACT_DETECTED",
    FallState.FALL_CONFIRMED:  "[red]    FALL_CONFIRMED",
    FallState.RECOVERED:       "[cyan]   RECOVERED",
}


def _tag_state(state: FallState) -> str:
    return _STATE_TAG.get(state, state.name)


def run_scenario(name: str, sequence: list, conf_th: float, persistence: int) -> None:
    ad = AlarmDetector(
        confidence_threshold=conf_th,
        persistence_frames=persistence,
        ema_alpha=0.5,
        cooldown_seconds=0.0,
        fps=15.0,
    )
    print()
    print(f"=== {name} ===")
    print(
        f"   conf_threshold={conf_th}  warning_threshold={ad.warning_threshold:.2f}  "
        f"persistence={persistence}"
    )
    print(f"  {'frame':>5}  {'raw':>5}  {'smoothed':>8}  state")
    print("  " + "-" * 50)
    for i, c in enumerate(sequence):
        state = ad.process_frame(c, is_subject_upright=(c < 0.3))
        print(
            f"  {i:>5d}  {c:>5.2f}  {ad.smoothed_confidence:>8.3f}  {_tag_state(state)}"
        )


def main(conf_th: float, persistence: int) -> int:
    for name, seq in SCENARIOS.items():
        run_scenario(name, seq, conf_th, persistence)
    print()
    print("Legend: IMMINENT (yellow) -> IMPACT (orange) -> "
          "FALL_CONFIRMED (red) -> RECOVERED (cyan)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Drive AlarmDetector FSM through synthetic scenarios.")
    p.add_argument("--confidence-threshold", type=float, default=0.7)
    p.add_argument("--persistence", type=int, default=3)
    args = p.parse_args()
    sys.exit(main(args.confidence_threshold, args.persistence))
