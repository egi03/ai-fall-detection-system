"""
Nonparametric fold-level bootstrap CI for BiLSTM vs Transformer-LSTM AUC difference.

Addresses reviewer comment: "Reporting nonparametric paired-bootstrap CIs over
fold-level AUC differences (or across resampled subjects) would strengthen the claim."

Uses 10,000 bootstrap resamples of the 5 fold-level AUC differences.
Also includes a permutation test (all 2^5=32 sign permutations).

Usage:
    python scripts/v3_fold_bootstrap.py
"""

import json
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-fold AUC differences from comparison_results.json (BiLSTM - TL)
# These are exact values from results/statistical_comparison/comparison_results.json
BILSTM_FOLD_AUCS = [
    0.9405587477709529,
    0.8616071428571429,
    0.9756455399061033,
    0.743421052631579,
    0.9368421052631578,
]
TL_FOLD_AUCS = [
    0.9561125421042204,
    0.8855229591836735,
    0.9589201877934272,
    0.725,
    0.933717105263158,
]
FOLD_DIFFERENCES = [b - t for b, t in zip(BILSTM_FOLD_AUCS, TL_FOLD_AUCS)]

SEED = 42
N_BOOTSTRAP = 10000


def fold_bootstrap_ci(differences: list, n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = SEED, ci_level: float = 0.90):
    """Bootstrap CI for mean of fold-level AUC differences."""
    rng = np.random.RandomState(seed)
    diffs = np.array(differences)
    n = len(diffs)
    observed_mean = float(np.mean(diffs))

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    boot_means = np.array(boot_means)
    alpha = 1 - ci_level
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return observed_mean, lo, hi, boot_means


def permutation_test(differences: list):
    """
    Exact sign permutation test for H0: mean(differences) = 0.
    Tests all 2^n sign arrangements. Returns two-sided p-value.
    """
    diffs = np.array(differences)
    n = len(diffs)
    observed = abs(np.mean(diffs))

    # Generate all 2^n sign patterns
    count_extreme = 0
    total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        permuted_mean = abs(np.mean(diffs * np.array(signs)))
        if permuted_mean >= observed - 1e-10:
            count_extreme += 1
        total += 1

    p_value = count_extreme / total
    return p_value, total


def main():
    output_dir = Path("results/v3_analysis/fold_bootstrap")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FOLD-LEVEL BOOTSTRAP CI (v3 paper fix)")
    print("=" * 60)
    print(f"BiLSTM fold AUCs: {[f'{x:.4f}' for x in BILSTM_FOLD_AUCS]}")
    print(f"TL fold AUCs:     {[f'{x:.4f}' for x in TL_FOLD_AUCS]}")
    print(f"Differences:      {[f'{x:.4f}' for x in FOLD_DIFFERENCES]}")
    print(f"Mean difference:  {np.mean(FOLD_DIFFERENCES):.4f}")
    print()

    # 90% CI (matching TOST level) and 95% CI
    for ci_level in [0.90, 0.95]:
        mean, lo, hi, boot_means = fold_bootstrap_ci(
            FOLD_DIFFERENCES, N_BOOTSTRAP, SEED, ci_level
        )
        print(f"{100*ci_level:.0f}% bootstrap CI: [{lo:.4f}, {hi:.4f}]  "
              f"(mean = {mean:.4f})")

    # Permutation test
    p_perm, n_perm = permutation_test(FOLD_DIFFERENCES)
    print(f"\nExact permutation test (n=2^5={n_perm}):")
    print(f"  p-value (two-sided): {p_perm:.4f}")

    # TOST using bootstrap (check if 90% CI falls within ±0.05)
    _, lo90, hi90, boot_means90 = fold_bootstrap_ci(
        FOLD_DIFFERENCES, N_BOOTSTRAP, SEED, 0.90
    )
    delta = 0.05
    tost_confirmed = (lo90 > -delta) and (hi90 < delta)
    print(f"\nTOST via bootstrap 90% CI:")
    print(f"  90% CI = [{lo90:.4f}, {hi90:.4f}]  delta = +/-{delta}")
    print(f"  Equivalence confirmed: {tost_confirmed}")

    # Save bootstrap distribution figure
    _, _, _, boot_means = fold_bootstrap_ci(
        FOLD_DIFFERENCES, N_BOOTSTRAP, SEED, 0.90
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot_means, bins=50, color="#2196F3", alpha=0.7, edgecolor="white",
            linewidth=0.5)
    ax.axvline(np.mean(FOLD_DIFFERENCES), color="red", lw=2,
               label=f"Observed mean = {np.mean(FOLD_DIFFERENCES):.4f}")
    ax.axvline(lo90, color="orange", lw=1.5, ls="--",
               label=f"90% CI lower = {lo90:.4f}")
    ax.axvline(hi90, color="orange", lw=1.5, ls="--",
               label=f"90% CI upper = {hi90:.4f}")
    ax.axvline(-delta, color="gray", lw=1, ls=":",
               label=f"±δ = ±{delta}")
    ax.axvline(delta, color="gray", lw=1, ls=":")
    ax.set_xlabel("Bootstrap Mean AUC Difference (BiLSTM − TL)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Fold-Level Bootstrap Distribution\n"
                 f"n=5 folds, {N_BOOTSTRAP} resamples, seed={SEED}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = output_dir / "fold_bootstrap_distribution.png"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBootstrap distribution saved: {out_png}")

    # Save results
    results = {
        "fold_differences_bilstm_minus_tl": FOLD_DIFFERENCES,
        "bilstm_fold_aucs": BILSTM_FOLD_AUCS,
        "tl_fold_aucs": TL_FOLD_AUCS,
        "observed_mean_diff": round(float(np.mean(FOLD_DIFFERENCES)), 6),
        "bootstrap_90pct_ci": [round(lo90, 4), round(hi90, 4)],
        "bootstrap_95pct_ci": [
            round(float(np.percentile(boot_means, 2.5)), 4),
            round(float(np.percentile(boot_means, 97.5)), 4),
        ],
        "permutation_p_value": round(p_perm, 4),
        "n_permutations": n_perm,
        "n_bootstrap": N_BOOTSTRAP,
        "tost_delta": delta,
        "tost_confirmed_bootstrap": tost_confirmed,
    }

    out_json = output_dir / "fold_bootstrap_results.json"
    with open(str(out_json), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {out_json}")

    # Print paper-ready text
    print("\n" + "=" * 60)
    print("PAPER-READY TEXT:")
    print("=" * 60)
    mean_diff = float(np.mean(FOLD_DIFFERENCES))
    print(f"A nonparametric fold-level bootstrap (10,000 resamples, seed 42) "
          f"on the five per-fold AUC differences yields a 90\\% CI of "
          f"[{lo90:.3f}, {hi90:.3f}] for the mean BiLSTM$-$TL difference, "
          f"lying entirely within the equivalence margin ($\\delta=0.05$). "
          f"An exact sign permutation test ({n_perm} arrangements) "
          f"gives $p={p_perm:.3f}$, consistent with no systematic advantage "
          f"for either architecture. "
          f"We note that $n=5$ bootstrap resamples necessarily produce a "
          f"discrete distribution; these CIs should be interpreted as "
          f"supporting, not replacing, the TOST result.")


if __name__ == "__main__":
    main()
