"""
Subject-independent train/val/test splitting.

Ensures no data leakage by keeping all sequences from the same
subject in the same split. Implements LOSO cross-validation.

Reference: research/5.1 - LOSO is the gold standard; never mix
subjects across train/test splits (this is data leakage).
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubjectSplitter:
    """
    Creates subject-independent data splits.

    Parameters
    ----------
    subjects : list of str
        List of all subject identifiers in the dataset.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, subjects: List[str], seed: int = 42) -> None:
        if not subjects:
            raise ValueError("Subject list must not be empty.")
        self._subjects = sorted(set(subjects))
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def subjects(self) -> List[str]:
        """Return the sorted list of unique subjects."""
        return list(self._subjects)

    @property
    def num_subjects(self) -> int:
        """Return the number of unique subjects."""
        return len(self._subjects)

    def split_train_val_test(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[str]]:
        """
        Split subjects into train/val/test sets.

        Subjects are shuffled deterministically using the configured seed,
        then split by ratio. At least one subject is assigned to each
        non-empty split.

        Parameters
        ----------
        val_ratio : float
            Fraction of subjects for validation.
        test_ratio : float
            Fraction of subjects for testing.

        Returns
        -------
        dict
            Keys 'train', 'val', 'test' mapping to lists of subject IDs.

        Raises
        ------
        ValueError
            If ratios are invalid or there are too few subjects.
        """
        if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"Invalid split ratios: val={val_ratio}, test={test_ratio}. "
                "Sum must be < 1.0 and both non-negative."
            )

        n = self.num_subjects
        if n < 3:
            raise ValueError(
                f"Need at least 3 subjects for train/val/test split, got {n}."
            )

        shuffled = list(self._subjects)
        self._rng.shuffle(shuffled)

        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        if n_train < 1:
            raise ValueError(
                f"Not enough subjects for train split. n={n}, "
                f"n_val={n_val}, n_test={n_test}."
            )

        splits = {
            "train": sorted(shuffled[:n_train]),
            "val": sorted(shuffled[n_train : n_train + n_val]),
            "test": sorted(shuffled[n_train + n_val :]),
        }

        logger.info(
            "Subject split created",
            extra={
                "train_subjects": splits["train"],
                "val_subjects": splits["val"],
                "test_subjects": splits["test"],
            },
        )
        return splits

    def get_loso_folds(self) -> List[Dict[str, List[str]]]:
        """
        Generate Leave-One-Subject-Out cross-validation folds.

        Each fold uses exactly one subject for testing and the
        remaining subjects for training.

        Returns
        -------
        list of dict
            Each dict has keys 'train' and 'test', where 'test'
            contains exactly one subject.
        """
        folds = []
        for test_subject in self._subjects:
            fold = {
                "train": [s for s in self._subjects if s != test_subject],
                "test": [test_subject],
            }
            folds.append(fold)

        logger.info(
            "LOSO folds generated",
            extra={"num_folds": len(folds)},
        )
        return folds

    def save_splits(self, splits: Dict, path: Path) -> None:
        """
        Save split definitions to a JSON file.

        Parameters
        ----------
        splits : dict
            Split definition from split_train_val_test() or get_loso_folds().
        path : Path
            Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)

        logger.info("Splits saved", extra={"path": str(path)})

    @staticmethod
    def load_splits(path: Path) -> Dict:
        """
        Load split definitions from a JSON file.

        Parameters
        ----------
        path : Path
            Path to the JSON split file.

        Returns
        -------
        dict
            The loaded split definition.

        Raises
        ------
        FileNotFoundError
            If the split file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
