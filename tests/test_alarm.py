"""
Unit tests for the alarm detection and debouncing logic,
and the subject-independent data splitting logic.

Tests FSM state transitions, persistence counters, cooldown
behavior, LOSO fold generation, and split correctness.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data_processing.splitter import SubjectSplitter


class TestSubjectSplitter:
    """Tests for the SubjectSplitter class."""

    def test_empty_subjects_raises(self) -> None:
        """Empty subject list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            SubjectSplitter(subjects=[])

    def test_too_few_subjects_raises(self) -> None:
        """Fewer than 3 subjects should raise for train/val/test split."""
        splitter = SubjectSplitter(subjects=["S01", "S02"])
        with pytest.raises(ValueError, match="at least 3"):
            splitter.split_train_val_test()

    def test_split_no_overlap(self) -> None:
        """Train, val, and test sets must have no overlapping subjects."""
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        splitter = SubjectSplitter(subjects=subjects, seed=42)
        splits = splitter.split_train_val_test()

        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_split_covers_all_subjects(self) -> None:
        """All subjects must appear in exactly one split."""
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        splitter = SubjectSplitter(subjects=subjects, seed=42)
        splits = splitter.split_train_val_test()

        all_split = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert all_split == set(subjects)

    def test_split_deterministic(self) -> None:
        """Same seed should produce identical splits."""
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        splits1 = SubjectSplitter(subjects, seed=42).split_train_val_test()
        splits2 = SubjectSplitter(subjects, seed=42).split_train_val_test()

        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]
        assert splits1["test"] == splits2["test"]

    def test_different_seed_different_splits(self) -> None:
        """Different seeds should produce different splits."""
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        splits1 = SubjectSplitter(subjects, seed=42).split_train_val_test()
        splits2 = SubjectSplitter(subjects, seed=99).split_train_val_test()

        # Very likely to be different (not guaranteed but extremely probable)
        assert splits1["train"] != splits2["train"] or splits1["test"] != splits2["test"]

    def test_invalid_ratios_raises(self) -> None:
        """Invalid ratio combinations should raise ValueError."""
        splitter = SubjectSplitter(subjects=["S01", "S02", "S03"])
        with pytest.raises(ValueError, match="Invalid split ratios"):
            splitter.split_train_val_test(val_ratio=0.5, test_ratio=0.6)


class TestLOSOFolds:
    """Tests for Leave-One-Subject-Out cross-validation."""

    def test_loso_num_folds(self) -> None:
        """LOSO should produce one fold per subject."""
        subjects = ["S01", "S02", "S03", "S04", "S05"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()
        assert len(folds) == 5

    def test_loso_test_single_subject(self) -> None:
        """Each LOSO fold should have exactly one test subject."""
        subjects = ["S01", "S02", "S03", "S04", "S05"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()

        for fold in folds:
            assert len(fold["test"]) == 1

    def test_loso_train_has_remaining(self) -> None:
        """Each LOSO fold's train set should contain all other subjects."""
        subjects = ["S01", "S02", "S03", "S04", "S05"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()

        for fold in folds:
            assert len(fold["train"]) == 4
            assert fold["test"][0] not in fold["train"]

    def test_loso_every_subject_tested(self) -> None:
        """Every subject should be the test subject in exactly one fold."""
        subjects = ["S01", "S02", "S03", "S04", "S05"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()

        tested = [fold["test"][0] for fold in folds]
        assert sorted(tested) == sorted(subjects)

    def test_loso_no_data_leakage(self) -> None:
        """Test subject must never appear in train set."""
        subjects = ["S01", "S02", "S03"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()

        for fold in folds:
            test_subj = fold["test"][0]
            assert test_subj not in fold["train"]


class TestSplitPersistence:
    """Tests for saving and loading split definitions."""

    def test_save_and_load_roundtrip(self) -> None:
        """Saved splits should load identically."""
        subjects = ["S01", "S02", "S03", "S04", "S05"]
        splitter = SubjectSplitter(subjects=subjects, seed=42)
        splits = splitter.split_train_val_test()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "splits.json"
            splitter.save_splits(splits, path)

            loaded = SubjectSplitter.load_splits(path)
            assert loaded["train"] == splits["train"]
            assert loaded["val"] == splits["val"]
            assert loaded["test"] == splits["test"]

    def test_load_nonexistent_raises(self) -> None:
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SubjectSplitter.load_splits(Path("/nonexistent/splits.json"))

    def test_save_loso_folds(self) -> None:
        """LOSO folds should be saveable and loadable."""
        subjects = ["S01", "S02", "S03"]
        splitter = SubjectSplitter(subjects=subjects)
        folds = splitter.get_loso_folds()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "loso_folds.json"
            splitter.save_splits(folds, path)

            loaded = SubjectSplitter.load_splits(path)
            assert len(loaded) == 3
            assert loaded[0]["test"] == folds[0]["test"]
