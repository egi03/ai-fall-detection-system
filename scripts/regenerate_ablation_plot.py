"""
Regenerate individual_ablation.png with per-fold SD error bars.

Addresses reviewer weakness R5: ablation ΔAUC reported without variance.

Usage:
    python scripts/regenerate_ablation_plot.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    results_dir = Path("results/ablation_study")

    with open(results_dir / "ablation_results.json") as f:
        ablation = json.load(f)

    baseline_auc = ablation["baseline"]["auc_roc"]
    baseline_folds = np.array(ablation["baseline"]["fold_aucs"])

    # Collect individual (single-feature removal) conditions
    group_keys = {
        "remove_angles", "remove_distances", "remove_bbox",
        "remove_dynamics", "remove_visibility",
    }

    items = []
    for key, val in ablation.items():
        if key == "baseline" or key in group_keys:
            continue
        if "fold_aucs" not in val:
            continue
        cond_folds = np.array(val["fold_aucs"])
        paired_diffs = cond_folds - baseline_folds
        mean_delta = float(val["auc_roc"]) - baseline_auc
        sd_delta = np.std(paired_diffs, ddof=1)

        display_name = key.replace("remove_", "").replace("_", "\n", 1)
        items.append((display_name, mean_delta, sd_delta))

    # Sort by delta
    items.sort(key=lambda x: x[1])

    names = [x[0] for x in items]
    deltas = [x[1] for x in items]
    sds = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    # Sign-based colouring: any negative delta (= useful feature) is never green.
    colors = [
        "#d32f2f" if d < -0.01           # critical: removing hurts a lot
        else "#ff9800" if d < 0.0         # useful: removing hurts slightly
        else "#4caf50"                     # neutral / harmful: safe (or good) to drop
        for d in deltas
    ]
    bars = ax.barh(
        names, deltas, color=colors, edgecolor="black", linewidth=0.5,
        xerr=sds, error_kw={"ecolor": "black", "capsize": 3, "linewidth": 0.9},
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("AUC Change from Baseline (paired per-fold \u00b1 SD)", fontsize=11)
    ax.set_title(
        f"Individual Feature Ablation (Baseline AUC = {baseline_auc:.4f}, n=5 folds)",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.3)

    for bar, delta, sd in zip(bars, deltas, sds):
        offset = 0.001 if delta >= 0 else -0.001
        ax.text(
            delta + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.3f}",
            va="center",
            ha="left" if delta >= 0 else "right",
            fontsize=8,
        )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#d32f2f", edgecolor="black",
              label=r"Critical ($\Delta$AUC $< -0.01$) -- keep"),
        Patch(facecolor="#ff9800", edgecolor="black",
              label=r"Useful ($-0.01 \leq \Delta$AUC $< 0$) -- keep"),
        Patch(facecolor="#4caf50", edgecolor="black",
              label=r"Neutral/harmful ($\Delta$AUC $\geq 0$) -- safe to drop"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8.5, framealpha=0.95)

    plt.tight_layout()
    out_path = results_dir / "individual_ablation.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved updated ablation plot with error bars to: {out_path}")


if __name__ == "__main__":
    main()
