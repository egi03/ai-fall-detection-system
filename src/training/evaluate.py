"""
Evaluation metrics computation for fall detection models.

Computes sensitivity, specificity, F1, AUC-ROC, PR-AUC,
false alarm rate, and generates evaluation artifacts.

Reference: research/5.1 - Sensitivity and specificity are primary
metrics; accuracy is misleading for imbalanced data.
Reference: research/5.2 - False alarm rate < 1/hour is mandatory.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Computes and saves all evaluation metrics for fall detection.

    Parameters
    ----------
    output_dir : Path
        Directory for saving evaluation artifacts (plots, CSVs).
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute all classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth binary labels.
        y_pred : np.ndarray
            Predicted binary labels.
        y_prob : np.ndarray
            Predicted probabilities for the positive class.

        Returns
        -------
        dict
            Metrics: sensitivity, specificity, accuracy, f1,
            precision, auc_roc, pr_auc.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
            average_precision_score,
            confusion_matrix,
        )

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_prob = np.asarray(y_prob)

        # Confusion matrix: [[TN, FP], [FN, TP]]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC-ROC (requires at least 2 classes present)
        try:
            auc_roc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc_roc = float("nan")

        # PR-AUC
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            pr_auc = float("nan")

        metrics = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(sensitivity),
            "auc_roc": auc_roc,
            "pr_auc": pr_auc,
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        return metrics

    def compute_false_alarm_rate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        total_hours: float,
    ) -> float:
        """
        Compute false alarms per hour.

        Reference: research/5.1 - FAR > 1/hour is unacceptable.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Predicted labels.
        total_hours : float
            Total monitoring duration in hours.

        Returns
        -------
        float
            False alarms per hour.
        """
        if total_hours <= 0:
            return float("nan")

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # False positives: predicted fall but actual non-fall
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        return fp / total_hours

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, path: Path
    ) -> None:
        """
        Generate and save a normalized confusion matrix plot.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_pred : np.ndarray
            Predicted labels.
        path : Path
            Output file path for the PNG image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Non-Fall", "Fall"],
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", values_format=".2f")
        ax.set_title("Normalized Confusion Matrix")
        plt.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to {path}")

    def plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, path: Path
    ) -> None:
        """
        Generate and save an ROC curve with AUC value.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels.
        y_prob : np.ndarray
            Predicted probabilities.
        path : Path
            Output file path for the PNG image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        try:
            auc_val = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_val = float("nan")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("ROC Curve")
        ax.legend()
        plt.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"ROC curve saved to {path}")

    def plot_training_curves(
        self, history: Dict[str, list], path: Path
    ) -> None:
        """
        Plot training and validation loss/accuracy curves.

        Parameters
        ----------
        history : dict
            Training history from Trainer.train().
        path : Path
            Output file path for the PNG image.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        epochs = range(1, len(history["train_loss"]) + 1)
        ax1.plot(epochs, history["train_loss"], label="Train")
        ax1.plot(epochs, history["val_loss"], label="Validation")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()

        ax2.plot(epochs, history["train_acc"], label="Train")
        ax2.plot(epochs, history["val_acc"], label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy Curves")
        ax2.legend()

        plt.tight_layout()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        logger.info(f"Training curves saved to {path}")

    def save_results(self, metrics: Dict[str, float], path: Path) -> None:
        """
        Save metrics to a JSON file.

        Parameters
        ----------
        metrics : dict
            Computed metrics dictionary.
        path : Path
            Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")
