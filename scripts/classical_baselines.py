"""
Classical ML baseline comparison using LOSO on URFD.

Trains Logistic Regression, Random Forest, and Gradient Boosting (XGBoost-style)
on the same LOSO splits and features as the BiLSTM/Transformer-LSTM models.

Two feature representations are tested:
  1. Flattened: (30, 15) -> (450,) raw temporal window
  2. Aggregated: per-feature statistics (mean, std, min, max, delta) -> (75,)

Reports AUC, sensitivity, specificity, and F1 per fold and overall.

Usage:
    python scripts/classical_baselines.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.splitter import SubjectSplitter
from src.training.dataset import build_dataset_from_processed


URFD_SUBJECTS = ["S01", "S02", "S03", "S04", "S05"]
DATA_DIR = Path("data/data/processed/urfd")
RESULTS_DIR = Path("results/classical_baselines")
WINDOW_SIZE = 30
TRAIN_STRIDE = 2
TEST_STRIDE = 5
SEED = 42


def aggregate_features(X: np.ndarray) -> np.ndarray:
    """
    Compute per-feature statistics from temporal windows.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 30, 15) — sliding windows.

    Returns
    -------
    np.ndarray
        Shape (N, 75) — 5 statistics per feature (mean, std, min, max, delta).
    """
    mean = X.mean(axis=1)        # (N, 15)
    std = X.std(axis=1)          # (N, 15)
    min_val = X.min(axis=1)      # (N, 15)
    max_val = X.max(axis=1)      # (N, 15)
    delta = X[:, -1, :] - X[:, 0, :]  # (N, 15) first-to-last change
    return np.concatenate([mean, std, min_val, max_val, delta], axis=1)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Compute AUC, sensitivity, specificity, F1 from probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "sensitivity": sens, "specificity": spec, "f1": f1}


def run_loso():
    """Run LOSO cross-validation with all classical baselines."""
    splitter = SubjectSplitter(URFD_SUBJECTS)
    folds = splitter.get_loso_folds()

    classifiers = {
        "LogReg": lambda: LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight="balanced", C=1.0
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=SEED,
            class_weight="balanced", n_jobs=-1
        ),
        "GBM": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=SEED, subsample=0.8
        ),
    }

    representations = {
        "flat": lambda X: X.reshape(X.shape[0], -1),
        "agg": aggregate_features,
    }

    all_results = {}

    for rep_name, rep_fn in representations.items():
        for clf_name, clf_factory in classifiers.items():
            key = f"{clf_name}_{rep_name}"
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")

            fold_metrics = []
            all_probs = []
            all_labels = []

            for fold_idx, fold in enumerate(folds):
                test_subject = fold["test"][0]
                train_subjects = fold["train"]

                # Split train into actual train + val (80/20)
                n_train = max(1, int(len(train_subjects) * 0.8))
                train_actual = train_subjects[:n_train]
                val_subjects = train_subjects[n_train:]

                # Load data using same pipeline as LSTM
                train_seqs, train_labels, norm_stats = build_dataset_from_processed(
                    processed_dir=str(DATA_DIR),
                    subject_ids=train_actual,
                    window_size=WINDOW_SIZE,
                    stride=TRAIN_STRIDE,
                    positive_threshold=0.5,
                )

                test_seqs, test_labels, _ = build_dataset_from_processed(
                    processed_dir=str(DATA_DIR),
                    subject_ids=[test_subject],
                    window_size=WINDOW_SIZE,
                    stride=TEST_STRIDE,
                    positive_threshold=0.5,
                    norm_stats=norm_stats,
                )

                # Apply representation
                X_train = rep_fn(train_seqs)
                X_test = rep_fn(test_seqs)
                y_train = train_labels
                y_test = test_labels

                # Additional scaling for LogReg
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train
                clf = clf_factory()
                clf.fit(X_train, y_train)

                # Predict probabilities
                y_prob = clf.predict_proba(X_test)[:, 1]

                metrics = compute_metrics(y_test, y_prob)
                fold_metrics.append(metrics)
                all_probs.extend(y_prob.tolist())
                all_labels.extend(y_test.tolist())

                print(f"  Fold {fold_idx} (test={test_subject}): "
                      f"AUC={metrics['auc']:.3f}  "
                      f"Sens={metrics['sensitivity']:.3f}  "
                      f"Spec={metrics['specificity']:.3f}")

            # Compute pooled metrics
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)
            pooled = compute_metrics(all_labels, all_probs)

            # Compute mean of fold-level metrics
            mean_auc = np.mean([m["auc"] for m in fold_metrics])
            std_auc = np.std([m["auc"] for m in fold_metrics])

            result = {
                "classifier": clf_name,
                "representation": rep_name,
                "pooled_auc": pooled["auc"],
                "pooled_sensitivity": pooled["sensitivity"],
                "pooled_specificity": pooled["specificity"],
                "pooled_f1": pooled["f1"],
                "mean_fold_auc": mean_auc,
                "std_fold_auc": std_auc,
                "per_fold": fold_metrics,
            }
            all_results[key] = result

            print(f"  POOLED: AUC={pooled['auc']:.3f}  "
                  f"Sens={pooled['sensitivity']:.3f}  "
                  f"Spec={pooled['specificity']:.3f}  "
                  f"F1={pooled['f1']:.3f}")
            print(f"  MEAN FOLD AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
    print(f"{'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for key, r in sorted(all_results.items(), key=lambda x: -x[1]["pooled_auc"]):
        print(f"{key:<20} {r['pooled_auc']:>6.3f} "
              f"{r['pooled_sensitivity']:>6.3f} "
              f"{r['pooled_specificity']:>6.3f} "
              f"{r['pooled_f1']:>6.3f}")

    # Compare with deep learning
    print(f"\n--- For reference (from prior experiments) ---")
    print(f"{'BiLSTM (Run5)':<20} {'0.888':>6} {'0.806':>6} {'0.842':>6} {'0.736':>6}")
    print(f"{'Transf.-LSTM':<20} {'0.877':>6} {'0.768':>6} {'0.838':>6} {'0.710':>6}")
    print(f"{'BiLSTM Ensemble':<20} {'0.897':>6} {'0.820':>6} {'0.838':>6} {'0.741':>6}")

    return all_results


if __name__ == "__main__":
    results = run_loso()
