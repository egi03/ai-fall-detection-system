"""
Statistical comparison of BiLSTM vs Transformer-LSTM architectures.

Performs paired statistical tests on per-fold LOSO AUC values to determine
whether the performance difference between the two architectures is
statistically significant.

Tests performed:
  1. Wilcoxon signed-rank test (non-parametric, appropriate for n=5)
  2. Paired t-test (parametric, supplementary)
  3. Effect size (Cohen's d for paired samples)
  4. Bootstrap confidence interval for the mean difference

Outputs:
  - results/statistical_comparison/comparison_results.json
  - results/statistical_comparison/fold_comparison.png (bar chart)
  - results/statistical_comparison/paired_difference.png

Usage:
    python scripts/statistical_comparison.py
    python scripts/statistical_comparison.py --bilstm-dir models/run5_stride2_aug --tl-dir models/tl_run4_d64_drop03

Reference: research/5.1 - Statistical validation of performance claims.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
np.random.seed(SEED)


def load_per_fold_metrics(
    model_dir: Path,
    n_folds: int = 5,
) -> dict:
    """
    Load per-fold metrics from a model directory.

    Parameters
    ----------
    model_dir : Path
        Directory containing fold_0/, fold_1/, ... subdirectories.
    n_folds : int
        Number of LOSO folds.

    Returns
    -------
    dict
        Keys are metric names, values are lists of per-fold values.
    """
    metrics_per_fold = {
        "auc_roc": [],
        "sensitivity": [],
        "specificity": [],
        "f1": [],
        "accuracy": [],
    }

    for fold in range(n_folds):
        metrics_path = model_dir / f"fold_{fold}" / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics: {metrics_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        for key in metrics_per_fold:
            metrics_per_fold[key].append(m[key])

    return metrics_per_fold


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d for paired samples.

    d = mean(x - y) / std(x - y)
    Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    """
    diff = x - y
    return float(diff.mean() / (diff.std(ddof=1) + 1e-10))


def bootstrap_mean_diff(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
) -> dict:
    """
    Bootstrap confidence interval for the mean paired difference.

    Parameters
    ----------
    x, y : np.ndarray
        Paired observations.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95% CI).

    Returns
    -------
    dict
        mean_diff, ci_lower, ci_upper, p_value (proportion of bootstrap
        samples where the sign of the mean difference flips).
    """
    diff = x - y
    n = len(diff)
    boot_means = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        boot_means[b] = diff[idx].mean()

    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # Two-sided p-value: proportion of bootstrap means on opposite side of 0
    observed_mean = diff.mean()
    if observed_mean >= 0:
        p_value = float(2 * np.mean(boot_means <= 0))
    else:
        p_value = float(2 * np.mean(boot_means >= 0))

    return {
        "mean_diff": float(observed_mean),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_p_value": min(p_value, 1.0),
    }


def plot_comparison(
    bilstm_metrics: dict,
    tl_metrics: dict,
    results: dict,
    output_dir: Path,
) -> None:
    """Generate comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = mkdir_p(output_dir)

    bilstm_auc = np.array(bilstm_metrics["auc_roc"])
    tl_auc = np.array(tl_metrics["auc_roc"])
    n_folds = len(bilstm_auc)
    subjects = [f"S{i+1:02d}" for i in range(n_folds)]

    # =========================================================================
    # Figure 1: Per-fold AUC comparison (grouped bar chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(n_folds)
    width = 0.35

    bars1 = ax.bar(x - width/2, bilstm_auc, width, label="BiLSTM",
                    color="#2196F3", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, tl_auc, width, label="Transformer-LSTM",
                    color="#FF5722", alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Test Subject (LOSO Fold)", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Per-Fold AUC Comparison: BiLSTM vs Transformer-LSTM", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(fontsize=11)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, axis="y", alpha=0.3)

    # Add means as horizontal lines
    ax.axhline(y=bilstm_auc.mean(), color="#2196F3", linestyle="--", alpha=0.5)
    ax.axhline(y=tl_auc.mean(), color="#FF5722", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "fold_comparison_auc.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fold_comparison_auc.pdf", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 2: Paired difference plot
    # =========================================================================
    diff = bilstm_auc - tl_auc

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: difference per fold
    colors = ["green" if d > 0 else "red" for d in diff]
    axes[0].bar(x, diff, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].axhline(y=0, color="black", linewidth=1)
    axes[0].axhline(y=diff.mean(), color="purple", linestyle="--", alpha=0.7,
                     label=f"Mean diff: {diff.mean():+.4f}")
    axes[0].set_xlabel("Test Subject (LOSO Fold)", fontsize=12)
    axes[0].set_ylabel("AUC Difference (BiLSTM \u2212 TL)", fontsize=12)
    axes[0].set_title("Per-Fold AUC Difference", fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subjects)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Add value labels
    for i, d in enumerate(diff):
        axes[0].text(i, d + 0.001 * np.sign(d), f"{d:+.4f}",
                      ha="center", va="bottom" if d >= 0 else "top", fontsize=9)

    # Right: bootstrap CI
    boot = results["bootstrap_auc"]
    ci_lower = boot["ci_lower"]
    ci_upper = boot["ci_upper"]
    mean_diff = boot["mean_diff"]

    axes[1].barh(0, mean_diff, height=0.4, color="purple", alpha=0.6)
    axes[1].errorbar(mean_diff, 0, xerr=[[mean_diff - ci_lower], [ci_upper - mean_diff]],
                      fmt="o", color="black", capsize=8, capthick=2, markersize=8)
    axes[1].axvline(x=0, color="black", linewidth=1)
    axes[1].set_xlabel("AUC Difference (BiLSTM \u2212 TL)", fontsize=12)
    axes[1].set_title(f"Bootstrap 95% CI\np = {boot['bootstrap_p_value']:.3f}", fontsize=13)
    axes[1].set_yticks([])
    axes[1].grid(True, axis="x", alpha=0.3)

    # Annotate
    axes[1].text(0.05, 0.85, f"Mean: {mean_diff:+.4f}\n95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]",
                  transform=axes[1].transAxes, fontsize=10,
                  bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / "paired_difference_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "paired_difference_analysis.pdf", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 3: Multi-metric comparison radar/bar chart
    # =========================================================================
    metrics_to_compare = ["auc_roc", "sensitivity", "specificity", "f1"]
    metric_labels = ["AUC-ROC", "Sensitivity", "Specificity", "F1"]

    bilstm_means = [np.mean(bilstm_metrics[m]) for m in metrics_to_compare]
    tl_means = [np.mean(tl_metrics[m]) for m in metrics_to_compare]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    ax.bar(x - width/2, bilstm_means, width, label="BiLSTM", color="#2196F3", alpha=0.85)
    ax.bar(x + width/2, tl_means, width, label="Transformer-LSTM", color="#FF5722", alpha=0.85)

    for i in range(len(metrics_to_compare)):
        ax.text(x[i] - width/2, bilstm_means[i] + 0.005, f"{bilstm_means[i]:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(x[i] + width/2, tl_means[i] + 0.005, f"{tl_means[i]:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Multi-Metric Comparison (Mean Across 5 LOSO Folds)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.set_ylim(0.65, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "multi_metric_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "multi_metric_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved all comparison figures to {output_dir}")


def mkdir_p(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Statistical comparison of architectures")
    parser.add_argument(
        "--bilstm-dir", type=str, default="models/run5_stride2_aug",
        help="BiLSTM model directory",
    )
    parser.add_argument(
        "--tl-dir", type=str, default="models/tl_run4_d64_drop03",
        help="Transformer-LSTM model directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/statistical_comparison",
        help="Output directory",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap resamples",
    )
    args = parser.parse_args()

    output_dir = mkdir_p(Path(args.output_dir))

    # Load metrics
    bilstm_metrics = load_per_fold_metrics(Path(args.bilstm_dir))
    tl_metrics = load_per_fold_metrics(Path(args.tl_dir))

    bilstm_auc = np.array(bilstm_metrics["auc_roc"])
    tl_auc = np.array(tl_metrics["auc_roc"])

    print("=" * 60)
    print("STATISTICAL COMPARISON: BiLSTM vs Transformer-LSTM")
    print("=" * 60)
    print(f"\nPer-Fold AUC-ROC:")
    print(f"{'Fold':<8} {'BiLSTM':<10} {'TL':<10} {'Diff':>10}")
    print("-" * 40)
    for i in range(len(bilstm_auc)):
        diff = bilstm_auc[i] - tl_auc[i]
        print(f"S{i+1:02d}      {bilstm_auc[i]:.4f}     {tl_auc[i]:.4f}     {diff:+.4f}")
    print("-" * 40)
    print(f"Mean     {bilstm_auc.mean():.4f}     {tl_auc.mean():.4f}     {(bilstm_auc - tl_auc).mean():+.4f}")
    print(f"Std      {bilstm_auc.std():.4f}     {tl_auc.std():.4f}")

    # =========================================================================
    # Test 1: Wilcoxon signed-rank test
    # =========================================================================
    from scipy import stats

    # Wilcoxon requires non-zero differences. If all diffs are zero, p=1.
    diff = bilstm_auc - tl_auc
    non_zero_diff = diff[diff != 0]

    if len(non_zero_diff) >= 1:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
            bilstm_auc, tl_auc, alternative="two-sided"
        )
    else:
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    print(f"\n1. Wilcoxon Signed-Rank Test:")
    print(f"   Statistic: {wilcoxon_stat:.4f}")
    print(f"   p-value:   {wilcoxon_p:.4f}")
    print(f"   NOTE: With n=5 paired observations, the minimum achievable")
    print(f"         two-sided p-value is 0.0625 (2/2^5). The Wilcoxon test")
    print(f"         CANNOT reject H0 at alpha=0.05 with this sample size.")
    print(f"   Significant (p<0.05): {'YES' if wilcoxon_p < 0.05 else 'NO'}")

    # =========================================================================
    # Test 2: Paired t-test
    # =========================================================================
    t_stat, t_p = stats.ttest_rel(bilstm_auc, tl_auc)

    print(f"\n2. Paired t-Test:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {t_p:.4f}")
    print(f"   Significant (p<0.05): {'YES' if t_p < 0.05 else 'NO'}")

    # =========================================================================
    # Test 3: Cohen's d (effect size)
    # =========================================================================
    d = cohens_d_paired(bilstm_auc, tl_auc)
    if abs(d) < 0.2:
        effect_label = "negligible"
    elif abs(d) < 0.5:
        effect_label = "small"
    elif abs(d) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    print(f"\n3. Effect Size (Cohen's d):")
    print(f"   d = {d:+.4f} ({effect_label})")

    # =========================================================================
    # Test 4: Bootstrap CI
    # =========================================================================
    boot_auc = bootstrap_mean_diff(bilstm_auc, tl_auc, n_bootstrap=args.n_bootstrap)

    print(f"\n4. Bootstrap Analysis ({args.n_bootstrap} resamples):")
    print(f"   Mean AUC difference: {boot_auc['mean_diff']:+.4f}")
    print(f"   95% CI: [{boot_auc['ci_lower']:+.4f}, {boot_auc['ci_upper']:+.4f}]")
    print(f"   Bootstrap p-value: {boot_auc['bootstrap_p_value']:.4f}")
    contains_zero = boot_auc["ci_lower"] <= 0 <= boot_auc["ci_upper"]
    print(f"   CI contains zero: {'YES' if contains_zero else 'NO'}")

    # =========================================================================
    # Additional metrics comparison
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("MULTI-METRIC COMPARISON (Mean +/- Std across folds)")
    print(f"{'=' * 60}")
    for metric in ["auc_roc", "sensitivity", "specificity", "f1"]:
        b = np.array(bilstm_metrics[metric])
        t = np.array(tl_metrics[metric])
        _, p = stats.ttest_rel(b, t)
        print(f"  {metric:<15} BiLSTM: {b.mean():.4f}+/-{b.std():.4f}  "
              f"TL: {t.mean():.4f}+/-{t.std():.4f}  p={p:.4f}")

    # =========================================================================
    # Conclusion
    # =========================================================================
    print(f"\n{'=' * 60}")
    print("CONCLUSION")
    print(f"{'=' * 60}")
    if wilcoxon_p >= 0.05 and t_p >= 0.05:
        print("  The difference between BiLSTM and Transformer-LSTM is")
        print("  NOT statistically significant (p >= 0.05 on both tests).")
        print("  With n=5 folds, the architectures are statistically equivalent")
        print("  on within-dataset URFD LOSO evaluation.")
        conclusion = "not_significant"
    elif wilcoxon_p < 0.05 or t_p < 0.05:
        winner = "BiLSTM" if diff.mean() > 0 else "Transformer-LSTM"
        print(f"  {winner} is statistically significantly better (p < 0.05).")
        conclusion = f"{winner}_significant"
    else:
        conclusion = "mixed"

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "bilstm_model_dir": str(args.bilstm_dir),
        "tl_model_dir": str(args.tl_dir),
        "n_folds": len(bilstm_auc),
        "per_fold_auc": {
            "bilstm": bilstm_auc.tolist(),
            "transformer_lstm": tl_auc.tolist(),
            "difference": diff.tolist(),
        },
        "mean_auc": {
            "bilstm": float(bilstm_auc.mean()),
            "transformer_lstm": float(tl_auc.mean()),
            "mean_difference": float(diff.mean()),
        },
        "wilcoxon_test": {
            "statistic": float(wilcoxon_stat),
            "p_value": float(wilcoxon_p),
            "significant_at_005": bool(wilcoxon_p < 0.05),
        },
        "paired_t_test": {
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "significant_at_005": bool(t_p < 0.05),
        },
        "effect_size": {
            "cohens_d": float(d),
            "interpretation": effect_label,
        },
        "bootstrap_auc": boot_auc,
        "conclusion": conclusion,
    }

    with open(output_dir / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved comparison_results.json")

    # Generate plots
    plot_comparison(bilstm_metrics, tl_metrics, results, output_dir)


if __name__ == "__main__":
    main()
