"""
Unit tests for dataset loaders.

Tests the parsing logic for URFD, Le2i, and UP-Fall dataset
directory structures using temporary directories with mock data.
"""

import tempfile
from pathlib import Path

import pytest

from src.data_processing.loader import (
    DatasetLoader,
    SequenceInfo,
    LABEL_FALL,
    LABEL_ADL,
)


class TestDatasetLoaderRegistry:
    """Tests for the DatasetLoader registry and validation."""

    def test_supported_datasets(self) -> None:
        """All expected datasets should be registered."""
        supported = DatasetLoader.supported_datasets()
        assert "urfd" in supported
        assert "le2i" in supported
        assert "up_fall" in supported

    def test_unknown_dataset_raises(self) -> None:
        """Unknown dataset name should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown dataset"):
                DatasetLoader("nonexistent", Path(tmpdir))

    def test_missing_directory_raises(self) -> None:
        """Missing data directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            DatasetLoader("urfd", Path("/nonexistent/path"))


class TestURFDLoader:
    """Tests for URFD dataset parsing."""

    def _create_urfd_structure(self, tmpdir: Path) -> None:
        """Create mock URFD directory structure with frame dirs."""
        for i in range(1, 4):
            for cam in range(2):
                # Fall sequences
                fall_dir = tmpdir / f"fall-{i:02d}-cam{cam}-rgb"
                fall_dir.mkdir(parents=True)
                # Create a few dummy image files
                for f in range(5):
                    (fall_dir / f"frame_{f:04d}.png").touch()

                # ADL sequences
                adl_dir = tmpdir / f"adl-{i:02d}-cam{cam}-rgb"
                adl_dir.mkdir(parents=True)
                for f in range(5):
                    (adl_dir / f"frame_{f:04d}.png").touch()

    def test_urfd_finds_sequences(self) -> None:
        """URFD loader should find fall and ADL frame directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            assert len(sequences) > 0
            falls = [s for s in sequences if s.label == LABEL_FALL]
            adls = [s for s in sequences if s.label == LABEL_ADL]
            assert len(falls) > 0
            assert len(adls) > 0

    def test_urfd_correct_labels(self) -> None:
        """Fall directories should get LABEL_FALL, ADL should get LABEL_ADL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            for seq in sequences:
                if "fall" in seq.sequence_id:
                    assert seq.label == LABEL_FALL
                elif "adl" in seq.sequence_id:
                    assert seq.label == LABEL_ADL

    def test_urfd_camera_assignment(self) -> None:
        """Camera ID should be correctly parsed from directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            cam0_seqs = [s for s in sequences if s.camera == "cam0"]
            cam1_seqs = [s for s in sequences if s.camera == "cam1"]
            assert len(cam0_seqs) > 0
            assert len(cam1_seqs) > 0

    def test_urfd_subject_assignment(self) -> None:
        """Each sequence should have a subject ID assigned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            for seq in sequences:
                assert seq.subject_id.startswith("S")

    def test_urfd_dataset_field(self) -> None:
        """All sequences should have dataset='urfd'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            for seq in sequences:
                assert seq.dataset == "urfd"

    def test_urfd_is_frame_dir(self) -> None:
        """Frame directory sequences should have is_frame_dir=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            sequences = loader.load()

            for seq in sequences:
                assert seq.is_frame_dir is True

    def test_urfd_get_subject_ids(self) -> None:
        """get_subject_ids should return unique subject identifiers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            subject_ids = loader.get_subject_ids()

            assert len(subject_ids) > 0
            assert all(isinstance(s, str) for s in subject_ids)

    def test_urfd_filter_by_subject(self) -> None:
        """get_sequences_for_subjects should filter correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_urfd_structure(tmpdir)

            loader = DatasetLoader("urfd", tmpdir)
            all_seqs = loader.load()
            subject_ids = loader.get_subject_ids()

            if len(subject_ids) > 0:
                filtered = loader.get_sequences_for_subjects([subject_ids[0]])
                assert len(filtered) <= len(all_seqs)
                for seq in filtered:
                    assert seq.subject_id == subject_ids[0]

    def test_urfd_empty_dir_warns(self) -> None:
        """Empty URFD directory should produce no sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DatasetLoader("urfd", Path(tmpdir))
            sequences = loader.load()
            assert len(sequences) == 0


class TestLe2iLoader:
    """Tests for Le2i dataset parsing."""

    def _create_le2i_structure(self, tmpdir: Path) -> None:
        """Create mock Le2i directory structure."""
        for room in ["Coffee_room", "Home"]:
            video_dir = tmpdir / room / "Videos"
            video_dir.mkdir(parents=True)

            anno_dir = tmpdir / room / "Annotation_files"
            anno_dir.mkdir(parents=True)

            # Create mock video files
            (video_dir / "fall_video_01.avi").touch()
            (video_dir / "adl_video_01.avi").touch()

            # Create annotation (fall at frames 10-50)
            with open(anno_dir / "fall_video_01.txt", "w") as f:
                f.write("10\n50\n")

    def test_le2i_finds_sequences(self) -> None:
        """Le2i loader should find sequences in room directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_le2i_structure(tmpdir)

            loader = DatasetLoader("le2i", tmpdir)
            sequences = loader.load()

            assert len(sequences) > 0

    def test_le2i_fall_label_from_filename(self) -> None:
        """Videos with 'fall' in name should be labeled as falls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_le2i_structure(tmpdir)

            loader = DatasetLoader("le2i", tmpdir)
            sequences = loader.load()

            fall_seqs = [s for s in sequences if "fall" in s.sequence_id.lower()]
            for seq in fall_seqs:
                assert seq.label == LABEL_FALL

    def test_le2i_annotations_loaded(self) -> None:
        """Frame-level annotations should be parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_le2i_structure(tmpdir)

            loader = DatasetLoader("le2i", tmpdir)
            annotations = loader.get_annotations()

            # Should have some annotations
            assert len(annotations) > 0

    def test_le2i_subject_is_room(self) -> None:
        """Le2i subject_id should be the room name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_le2i_structure(tmpdir)

            loader = DatasetLoader("le2i", tmpdir)
            sequences = loader.load()

            subjects = set(s.subject_id for s in sequences)
            assert "Coffee_room" in subjects or "Home" in subjects


class TestUPFallLoader:
    """Tests for UP-Fall dataset parsing."""

    def _create_up_fall_flat(self, tmpdir: Path) -> None:
        """Create mock UP-Fall flat file structure."""
        # Activity 1-5 = fall, 6-11 = ADL
        for subj in [1, 2]:
            for act in [1, 3, 6, 8]:
                for trial in [1]:
                    name = f"Subject{subj}_Activity{act}_Trial{trial}_Camera1.avi"
                    (tmpdir / name).touch()

    def test_up_fall_flat_finds_sequences(self) -> None:
        """UP-Fall flat loader should find sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_up_fall_flat(tmpdir)

            loader = DatasetLoader("up_fall", tmpdir)
            sequences = loader.load()

            assert len(sequences) == 8  # 2 subjects * 4 activities

    def test_up_fall_correct_labels(self) -> None:
        """Activities 1-5 should be falls, 6-11 should be ADLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_up_fall_flat(tmpdir)

            loader = DatasetLoader("up_fall", tmpdir)
            sequences = loader.load()

            for seq in sequences:
                if "_A01_" in seq.sequence_id or "_A03_" in seq.sequence_id:
                    assert seq.label == LABEL_FALL
                elif "_A06_" in seq.sequence_id or "_A08_" in seq.sequence_id:
                    assert seq.label == LABEL_ADL

    def test_up_fall_subject_ids(self) -> None:
        """Subject IDs should be parsed from filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            self._create_up_fall_flat(tmpdir)

            loader = DatasetLoader("up_fall", tmpdir)
            subject_ids = loader.get_subject_ids()

            assert "S01" in subject_ids
            assert "S02" in subject_ids
