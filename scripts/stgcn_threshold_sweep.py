"""Pick the best operating threshold for an ST-GCN LOSO run.

Reads ``results/<run>/all_y_true.npy`` and ``all_y_prob.npy`` produced
by ``scripts/train_stgcn.py`` and reports three useful operating
points: balanced accuracy, max F1, and the most sensitive threshold
still satisfying a chosen specificity floor (default 0.85).

The 0.5 default rarely lines up with the optimum for class-imbalanced
fall data, so a quick sweep typically buys a few percentage points of
sensitivity at no cost.

Usage
-----
::

    python scripts/stgcn_threshold_sweep.py --run stgcn_loso
    python scripts/stgcn_threshold_sweep.py --run stgcn_v2 --spec-floor 0.85
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_threshold(y_true: np.ndarray, y_prob: np.ndarray, t: float) -> dict:
    """Return sens/spec/F1 at a given decision threshold."""
    y_pred = (y_prob >= t).astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return {
        "threshold": float(t),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ST-GCN threshold sweep")
    parser.add_argument(
        "--run", required=True,
        help="Run name (folder under results/, e.g. stgcn_loso or stgcn_v2).",
    )
    parser.add_argument(
        "--spec-floor", type=float, default=0.85,
        help="Minimum specificity for the 'max sens at spec >= floor' point.",
    )
    parser.add_argument(
        "--num-thresholds", type=int, default=201,
        help="Number of thresholds sampled across [0, 1].",
    )
    args = parser.parse_args()

    results_dir = Path("results") / args.run
    y_true_path = results_dir / "all_y_true.npy"
    y_prob_path = results_dir / "all_y_prob.npy"
    if not (y_true_path.exists() and y_prob_path.exists()):
        print(
            f"Missing aggregated predictions in {results_dir}. "
            f"Run scripts/train_stgcn.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    auc = float(roc_auc_score(y_true, y_prob))

    thresholds = np.linspace(0.0, 1.0, args.num_thresholds)
    points = [evaluate_threshold(y_true, y_prob, t) for t in thresholds]

    # 1) Default threshold (for reference)
    default_point = evaluate_threshold(y_true, y_prob, 0.5)
    # 2) Max F1
    best_f1 = max(points, key=lambda p: p["f1"])
    # 3) Max balanced (avg of sens and spec)
    best_balanced = max(
        points, key=lambda p: 0.5 * (p["sensitivity"] + p["specificity"]),
    )
    # 4) Max sens subject to spec >= floor
    eligible = [p for p in points if p["specificity"] >= args.spec_floor]
    best_sens_at_spec = (
        max(eligible, key=lambda p: p["sensitivity"]) if eligible else None
    )

    def row(name: str, p: dict | None) -> str:
        if p is None:
            return f"{name:<24} (no threshold satisfies the constraint)"
        return (
            f"{name:<24} t={p['threshold']:.2f}  "
            f"sens={p['sensitivity']:.3f}  spec={p['specificity']:.3f}  "
            f"F1={p['f1']:.3f}  (TP={p['tp']}, FP={p['fp']}, FN={p['fn']})"
        )

    print(f"Run: {args.run}")
    print(f"AUC-ROC (threshold-independent): {auc:.4f}")
    print("-" * 70)
    print(row("Default (t=0.50)", default_point))
    print(row("Max F1", best_f1))
    print(row("Max balanced (sens+spec)", best_balanced))
    print(row(f"Max sens @ spec>={args.spec_floor}", best_sens_at_spec))
    print("-" * 70)

    # Persist so it lands next to the other artefacts
    output_path = results_dir / "threshold_sweep.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": args.run,
                "auc_roc": auc,
                "spec_floor": args.spec_floor,
                "default": default_point,
                "max_f1": best_f1,
                "max_balanced": best_balanced,
                "max_sens_at_spec_floor": best_sens_at_spec,
            },
            f,
            indent=2,
        )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
