"""
Dataset-specific loaders for fall detection datasets.

Handles loading and parsing of different dataset formats (URFD, Le2i,
UP-Fall) into a unified internal representation. Each loader discovers
video sequences, parses labels, and maps sequences to subjects.

Reference: research/1.1 - Dataset survey with format specifications.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Label constants
LABEL_ADL = 0
LABEL_FALL = 1

# Video file extensions to search for
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".wmv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class SequenceInfo:
    """
    Metadata for a single video sequence in the dataset.

    Attributes
    ----------
    sequence_id : str
        Unique identifier for this sequence.
    video_path : Path
        Path to the video file or directory of frame images.
    label : int
        Class label: 0 = ADL (non-fall), 1 = fall.
    subject_id : str
        Subject identifier for subject-independent splitting.
    camera : str
        Camera identifier (e.g., 'cam0', 'cam1').
    dataset : str
        Name of the source dataset.
    is_frame_dir : bool
        True if video_path points to a directory of images.
    annotation : dict
        Additional annotation data (frame ranges, etc.).
    """

    sequence_id: str
    video_path: Path
    label: int
    subject_id: str
    camera: str = "cam0"
    dataset: str = ""
    is_frame_dir: bool = False
    annotation: dict = field(default_factory=dict)


class DatasetLoader:
    """
    Unified loader for fall detection datasets.

    Delegates to dataset-specific parsing logic based on the
    dataset_name parameter.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset: 'urfd', 'le2i', 'up_fall'.
    data_dir : Path
        Root directory containing the raw dataset.
    """

    # Registry of supported dataset parsers
    _PARSERS = {}

    def __init__(self, dataset_name: str, data_dir: Path) -> None:
        self._name = dataset_name.lower()
        self._data_dir = Path(data_dir)

        if not self._data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self._data_dir}"
            )

        if self._name not in self._PARSERS:
            raise ValueError(
                f"Unknown dataset: {self._name}. "
                f"Supported: {list(self._PARSERS.keys())}"
            )

        self._parser = self._PARSERS[self._name]
        self._sequences: Optional[List[SequenceInfo]] = None

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset parser function.

        Parameters
        ----------
        name : str
            Dataset identifier string.
        """
        def decorator(func):
            cls._PARSERS[name] = func
            return func
        return decorator

    @classmethod
    def supported_datasets(cls) -> List[str]:
        """Return list of supported dataset names."""
        return list(cls._PARSERS.keys())

    def load(self) -> List[SequenceInfo]:
        """
        Load the dataset and return all discovered sequences.

        Returns
        -------
        list of SequenceInfo
            All sequences found in the dataset directory.
        """
        if self._sequences is None:
            self._sequences = self._parser(self._data_dir)
            logger.info(
                "Dataset loaded",
                extra={
                    "dataset": self._name,
                    "num_sequences": len(self._sequences),
                    "num_falls": sum(
                        1 for s in self._sequences if s.label == LABEL_FALL
                    ),
                    "num_adl": sum(
                        1 for s in self._sequences if s.label == LABEL_ADL
                    ),
                    "subjects": sorted(
                        set(s.subject_id for s in self._sequences)
                    ),
                },
            )
        return self._sequences

    def get_video_paths(self) -> List[Path]:
        """
        Retrieve all video file paths in the dataset.

        Returns
        -------
        list of Path
            Sorted list of video file paths.
        """
        sequences = self.load()
        return sorted(s.video_path for s in sequences)

    def get_annotations(self) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Load frame-level annotations (fall start/end frames).

        Returns
        -------
        dict
            Mapping of sequence_id -> list of (start_frame, end_frame, label).
        """
        sequences = self.load()
        annotations = {}
        for seq in sequences:
            label_str = "fall" if seq.label == LABEL_FALL else "adl"
            start = seq.annotation.get("start_frame", 0)
            end = seq.annotation.get("end_frame", -1)
            annotations[seq.sequence_id] = [(start, end, label_str)]
        return annotations

    def get_subject_ids(self) -> List[str]:
        """
        Return sorted list of unique subject IDs.

        Returns
        -------
        list of str
            Unique subject identifiers.
        """
        sequences = self.load()
        return sorted(set(s.subject_id for s in sequences))

    def get_sequences_for_subjects(
        self, subject_ids: List[str]
    ) -> List[SequenceInfo]:
        """
        Filter sequences belonging to specific subjects.

        Parameters
        ----------
        subject_ids : list of str
            Subject IDs to include.

        Returns
        -------
        list of SequenceInfo
            Filtered sequences.
        """
        subject_set = set(subject_ids)
        sequences = self.load()
        return [s for s in sequences if s.subject_id in subject_set]


def _find_videos(directory: Path) -> List[Path]:
    """Find all video files in a directory (non-recursive)."""
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _find_frame_dirs(directory: Path) -> List[Path]:
    """Find directories containing image frames."""
    dirs = []
    for d in sorted(directory.iterdir()):
        if d.is_dir():
            # Check if directory contains image files
            has_images = any(
                f.suffix.lower() in IMAGE_EXTENSIONS
                for f in d.iterdir()
                if f.is_file()
            )
            if has_images:
                dirs.append(d)
    return dirs


def _count_frames_in_dir(directory: Path) -> int:
    """Count image files in a frame directory."""
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# URFD Dataset Loader
# ============================================================
# URFD structure:
#   data/raw/urfd/
#   ├── fall-01-cam0-rgb/  (frame images)
#   ├── fall-01-cam1-rgb/
#   ├── ...
#   ├── adl-01-cam0-rgb/
#   ├── ...
#   OR video files: fall-01-cam0-rgb.avi, etc.
#
# 5 subjects, 30 falls, 40 ADLs, 2 camera angles.
# Subject mapping: sequences are grouped by subject, 6 falls
# and 8 ADLs per subject approximately.
#
# Reference: research/1.1 - URFD dataset specifications.
# ============================================================

# Default URFD subject mapping based on literature convention
# Falls: 6 per subject, ADLs: 8 per subject
URFD_SUBJECT_MAP_FALL = {
    i: f"S{(i - 1) // 6 + 1:02d}" for i in range(1, 31)
}
URFD_SUBJECT_MAP_ADL = {
    i: f"S{(i - 1) // 8 + 1:02d}" for i in range(1, 41)
}


@DatasetLoader.register("urfd")
def _parse_urfd(data_dir: Path) -> List[SequenceInfo]:
    """
    Parse the UR Fall Detection Dataset directory.

    Supports both frame-directory and video-file formats.
    Uses default subject mapping (6 falls + 8 ADLs per subject).
    A custom mapping can be provided via a subject_map.json file
    in the dataset root.

    Parameters
    ----------
    data_dir : Path
        Root directory of the URFD dataset.

    Returns
    -------
    list of SequenceInfo
        Parsed sequence metadata.
    """
    sequences = []

    # Check for custom subject mapping
    custom_map_path = data_dir / "subject_map.json"
    fall_map = dict(URFD_SUBJECT_MAP_FALL)
    adl_map = dict(URFD_SUBJECT_MAP_ADL)
    if custom_map_path.exists():
        with open(custom_map_path, "r", encoding="utf-8") as f:
            custom = json.load(f)
        if "fall" in custom:
            fall_map = {int(k): v for k, v in custom["fall"].items()}
        if "adl" in custom:
            adl_map = {int(k): v for k, v in custom["adl"].items()}
        logger.info("Loaded custom URFD subject mapping")

    # Pattern: fall-XX-camY-rgb or adl-XX-camY-rgb (or without -rgb for MP4s)
    pattern = re.compile(
        r"(fall|adl)-(\d+)-cam(\d+)(?:-rgb)?", re.IGNORECASE
    )

    # Try frame directories first
    candidates = []
    for item in sorted(data_dir.iterdir()):
        match = pattern.match(item.name)
        if match is None:
            # Try matching without -rgb suffix for video files
            stem = item.stem if item.is_file() else item.name
            match = pattern.match(stem)
        if match is not None:
            candidates.append((item, match))

    for item_path, match in candidates:
        action_type = match.group(1).lower()
        seq_num = int(match.group(2))
        cam = int(match.group(3))

        is_frame_dir = item_path.is_dir()
        if not is_frame_dir and item_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        label = LABEL_FALL if action_type == "fall" else LABEL_ADL
        subj_map = fall_map if action_type == "fall" else adl_map
        subject_id = subj_map.get(seq_num, f"S{(seq_num - 1) // 6 + 1:02d}")

        seq_id = f"urfd_{action_type}-{seq_num:02d}_cam{cam}"

        sequences.append(SequenceInfo(
            sequence_id=seq_id,
            video_path=item_path,
            label=label,
            subject_id=subject_id,
            camera=f"cam{cam}",
            dataset="urfd",
            is_frame_dir=is_frame_dir,
        ))

    if not sequences:
        logger.warning(
            "No URFD sequences found. Expected directories/files matching "
            "'fall-XX-camY-rgb' or 'adl-XX-camY-rgb' pattern.",
            extra={"data_dir": str(data_dir)},
        )

    return sequences


# ============================================================
# Le2i Dataset Loader
# ============================================================
# Le2i structure varies; common layout:
#   data/raw/le2i/
#   ├── Coffee_room/
#   │   ├── Videos/
#   │   │   ├── video_01.avi
#   │   │   └── ...
#   │   └── Annotation_files/
#   │       ├── video_01.txt
#   │       └── ...
#   ├── Home/
#   ├── Office/
#   └── Lecture/
#
# Annotations: text files with start/end frame of fall event.
# Subject IDs are not explicitly provided; use room+video as proxy.
#
# Reference: research/1.1 - Le2i (FDD) dataset specifications.
# ============================================================

@DatasetLoader.register("le2i")
def _parse_le2i(data_dir: Path) -> List[SequenceInfo]:
    """
    Parse the Le2i Fall Detection Dataset directory.

    Searches for video files organized by room (Coffee_room, Home,
    Office, Lecture). Annotations are loaded from companion text files
    when available.

    Parameters
    ----------
    data_dir : Path
        Root directory of the Le2i dataset.

    Returns
    -------
    list of SequenceInfo
        Parsed sequence metadata.
    """
    sequences = []

    # Le2i rooms
    room_dirs = [d for d in sorted(data_dir.iterdir()) if d.is_dir()]

    for room_dir in room_dirs:
        room_name = room_dir.name

        # Find video directory — handle extra nesting from Kaggle download:
        # le2i/Coffee_room_01/Coffee_room_01/Videos/ (double-nested)
        # le2i/Lecture_room/Lecture room/  (inner name differs from outer)
        video_dir = room_dir / "Videos"
        anno_dir = room_dir / "Annotation_files"
        if not video_dir.exists():
            # Search one level deeper for any subdir containing Videos/ or video files
            inner_dirs = [d for d in room_dir.iterdir() if d.is_dir()]
            for inner in inner_dirs:
                if (inner / "Videos").exists():
                    video_dir = inner / "Videos"
                    anno_dir = inner / "Annotation_files"
                    break
                elif any(f.suffix.lower() in VIDEO_EXTENSIONS for f in inner.iterdir() if f.is_file()):
                    video_dir = inner
                    anno_dir = inner / "Annotation_files"
                    break
            else:
                video_dir = room_dir  # Videos files directly in room dir

        if not anno_dir.exists():
            anno_dir = room_dir / "Annotations"

        # Load annotations if available
        annotations = _load_le2i_annotations(anno_dir) if anno_dir.exists() else {}

        # Find all video files
        videos = _find_videos(video_dir)

        for video_path in videos:
            video_name = video_path.stem
            seq_id = f"le2i_{room_name}_{video_name}"

            # Determine label from annotations or filename
            anno = annotations.get(video_name, {})
            if anno.get("has_fall", False):
                label = LABEL_FALL
            elif "fall" in video_name.lower():
                label = LABEL_FALL
            else:
                label = LABEL_ADL

            # Le2i doesn't have explicit subjects; use room as proxy
            subject_id = room_name

            annotation_data = {}
            if "start_frame" in anno:
                annotation_data["start_frame"] = anno["start_frame"]
                annotation_data["end_frame"] = anno["end_frame"]

            sequences.append(SequenceInfo(
                sequence_id=seq_id,
                video_path=video_path,
                label=label,
                subject_id=subject_id,
                camera="cam0",
                dataset="le2i",
                annotation=annotation_data,
            ))

    if not sequences:
        logger.warning(
            "No Le2i sequences found. Expected room subdirectories "
            "(Coffee_room, Home, Office, Lecture) with video files.",
            extra={"data_dir": str(data_dir)},
        )

    return sequences


def _load_le2i_annotations(anno_dir: Path) -> Dict[str, Dict]:
    """
    Load Le2i annotation files.

    Each annotation file typically contains the start and end frame
    numbers of the fall event, one per line.

    Parameters
    ----------
    anno_dir : Path
        Directory containing annotation text files.

    Returns
    -------
    dict
        Mapping of video_name -> {has_fall, start_frame, end_frame}.
    """
    annotations = {}

    for anno_file in sorted(anno_dir.iterdir()):
        if anno_file.suffix.lower() not in (".txt", ".csv"):
            continue

        video_name = anno_file.stem
        try:
            with open(anno_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) >= 2:
                start_frame = int(lines[0])
                end_frame = int(lines[1])
                annotations[video_name] = {
                    "has_fall": True,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            elif len(lines) == 1 and lines[0] == "-1":
                annotations[video_name] = {"has_fall": False}
        except (ValueError, IndexError) as e:
            logger.warning(
                f"Failed to parse Le2i annotation: {anno_file.name}",
                extra={"error": str(e)},
            )

    return annotations


# ============================================================
# UP-Fall Dataset Loader
# ============================================================
# UP-Fall structure:
#   data/raw/up_fall/
#   ├── Subject1/
#   │   ├── Activity1/
#   │   │   ├── Trial1/
#   │   │   │   ├── Camera1/
#   │   │   │   └── Camera2/
#   OR flat structure with naming convention:
#   ├── SubjectXX_ActivityYY_TrialZZ_CameraWW.avi
#
# 17 subjects, activities 1-5 are falls, 6-11 are ADLs.
#
# Reference: research/1.1 - UP-Fall dataset specifications.
# ============================================================

# UP-Fall activity labels
UP_FALL_ACTIVITIES = {
    1: ("Falling forward using hands", LABEL_FALL),
    2: ("Falling forward using knees", LABEL_FALL),
    3: ("Falling backwards", LABEL_FALL),
    4: ("Falling sideward", LABEL_FALL),
    5: ("Falling sitting in empty chair", LABEL_FALL),
    6: ("Walking", LABEL_ADL),
    7: ("Standing", LABEL_ADL),
    8: ("Sitting", LABEL_ADL),
    9: ("Picking up an object", LABEL_ADL),
    10: ("Jumping", LABEL_ADL),
    11: ("Laying down", LABEL_ADL),
}


@DatasetLoader.register("up_fall")
def _parse_up_fall(data_dir: Path) -> List[SequenceInfo]:
    """
    Parse the UP-Fall Detection Dataset directory.

    Supports hierarchical (Subject/Activity/Trial/Camera) and flat
    filename-based structures.

    Parameters
    ----------
    data_dir : Path
        Root directory of the UP-Fall dataset.

    Returns
    -------
    list of SequenceInfo
        Parsed sequence metadata.
    """
    sequences = []

    # Try hierarchical structure first
    subject_dirs = [
        d for d in sorted(data_dir.iterdir())
        if d.is_dir() and re.match(r"Subject\d+", d.name, re.IGNORECASE)
    ]

    if subject_dirs:
        sequences = _parse_up_fall_hierarchical(data_dir, subject_dirs)
    else:
        # Try flat file structure
        sequences = _parse_up_fall_flat(data_dir)

    if not sequences:
        logger.warning(
            "No UP-Fall sequences found.",
            extra={"data_dir": str(data_dir)},
        )

    return sequences


def _parse_up_fall_hierarchical(
    data_dir: Path, subject_dirs: List[Path]
) -> List[SequenceInfo]:
    """
    Parse UP-Fall in hierarchical Subject/Activity/Trial structure.

    Handles three cases within each trial directory:

    Case A — Camera subdirectory with images directly inside (post-extraction):
        Trial1/Camera1/*.jpg  → is_frame_dir=True, video_path=Camera1/

    Case B — ZIP files not yet extracted:
        Trial1/Subject1Activity1Trial1Camera1.zip  → is_frame_dir=False, video_path=<zip>

    Case C — Camera subdirectory with nested frame subdirs (legacy structure):
        Trial1/Camera1/frames/*.jpg  → discovered by _find_frame_dirs()

    Extracted Camera directories take priority over their corresponding ZIP files.
    """
    sequences = []

    for subject_dir in subject_dirs:
        subject_match = re.search(r"(\d+)", subject_dir.name)
        if subject_match is None:
            continue
        subject_num = int(subject_match.group(1))
        subject_id = f"S{subject_num:02d}"

        for activity_dir in sorted(subject_dir.iterdir()):
            if not activity_dir.is_dir():
                continue
            activity_match = re.search(r"(\d+)", activity_dir.name)
            if activity_match is None:
                continue
            activity_num = int(activity_match.group(1))

            if activity_num not in UP_FALL_ACTIVITIES:
                continue
            _, label = UP_FALL_ACTIVITIES[activity_num]

            for trial_dir in sorted(activity_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                trial_match = re.search(r"(\d+)", trial_dir.name)
                trial_num = int(trial_match.group(1)) if trial_match else 1

                # Collect camera subdirectories and ZIP files
                camera_dirs = [
                    d for d in sorted(trial_dir.iterdir())
                    if d.is_dir()
                    and re.search(r"camera", d.name, re.IGNORECASE)
                ]
                camera_zips = [
                    f for f in sorted(trial_dir.iterdir())
                    if f.is_file()
                    and f.suffix.lower() == ".zip"
                    and re.search(r"camera", f.name, re.IGNORECASE)
                ]

                # Track camera numbers already covered by extracted dirs
                processed_cam_nums: set = set()

                # --- Case A / C: Camera subdirectories ---
                for cam_dir in camera_dirs:
                    cam_match = re.search(r"(\d+)", cam_dir.name)
                    cam_num = int(cam_match.group(1)) if cam_match else 0

                    # Case A: images directly in cam_dir (post-extraction)
                    has_direct_images = any(
                        f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                        for f in cam_dir.iterdir()
                    )
                    if has_direct_images:
                        processed_cam_nums.add(cam_num)
                        seq_id = (
                            f"up_fall_S{subject_num:02d}"
                            f"_A{activity_num:02d}"
                            f"_T{trial_num:02d}"
                            f"_C{cam_num}"
                        )
                        sequences.append(SequenceInfo(
                            sequence_id=seq_id,
                            video_path=cam_dir,
                            label=label,
                            subject_id=subject_id,
                            camera=f"cam{cam_num}",
                            dataset="up_fall",
                            is_frame_dir=True,
                        ))
                        continue

                    # Case C: nested frame subdirs inside cam_dir
                    videos = _find_videos(cam_dir)
                    frame_dirs = _find_frame_dirs(cam_dir)
                    targets = (
                        [(v, False) for v in videos]
                        + [(d, True) for d in frame_dirs]
                    )
                    if targets:
                        processed_cam_nums.add(cam_num)
                    for path, is_dir in targets:
                        seq_id = (
                            f"up_fall_S{subject_num:02d}"
                            f"_A{activity_num:02d}"
                            f"_T{trial_num:02d}"
                            f"_C{cam_num}"
                        )
                        sequences.append(SequenceInfo(
                            sequence_id=seq_id,
                            video_path=path,
                            label=label,
                            subject_id=subject_id,
                            camera=f"cam{cam_num}",
                            dataset="up_fall",
                            is_frame_dir=is_dir,
                        ))

                # --- Case B: ZIP files (only if camera not already extracted) ---
                for zip_file in camera_zips:
                    cam_match = re.search(r"Camera(\d+)", zip_file.name, re.IGNORECASE)
                    cam_num = int(cam_match.group(1)) if cam_match else 0
                    if cam_num in processed_cam_nums:
                        # Extracted directory takes priority — skip ZIP
                        continue
                    seq_id = (
                        f"up_fall_S{subject_num:02d}"
                        f"_A{activity_num:02d}"
                        f"_T{trial_num:02d}"
                        f"_C{cam_num}"
                    )
                    sequences.append(SequenceInfo(
                        sequence_id=seq_id,
                        video_path=zip_file,
                        label=label,
                        subject_id=subject_id,
                        camera=f"cam{cam_num}",
                        dataset="up_fall",
                        is_frame_dir=False,
                    ))

                # Fallback: if no camera dirs or ZIPs found, look for videos
                # directly in the trial directory
                if not camera_dirs and not camera_zips:
                    videos = _find_videos(trial_dir)
                    for video in videos:
                        seq_id = (
                            f"up_fall_S{subject_num:02d}"
                            f"_A{activity_num:02d}"
                            f"_T{trial_num:02d}"
                        )
                        sequences.append(SequenceInfo(
                            sequence_id=seq_id,
                            video_path=video,
                            label=label,
                            subject_id=subject_id,
                            camera="cam0",
                            dataset="up_fall",
                        ))

    return sequences


def _parse_up_fall_flat(data_dir: Path) -> List[SequenceInfo]:
    """Parse UP-Fall from flat filename structure."""
    sequences = []

    # Pattern: SubjectXX_ActivityYY_TrialZZ[_CameraWW].ext
    pattern = re.compile(
        r"Subject(\d+)[_\-]Activity(\d+)[_\-]Trial(\d+)"
        r"(?:[_\-]Camera(\d+))?",
        re.IGNORECASE,
    )

    for item in sorted(data_dir.iterdir()):
        name = item.stem if item.is_file() else item.name
        match = pattern.match(name)
        if match is None:
            continue

        if item.is_file() and item.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        subject_num = int(match.group(1))
        activity_num = int(match.group(2))
        trial_num = int(match.group(3))
        cam_num = int(match.group(4)) if match.group(4) else 0

        if activity_num not in UP_FALL_ACTIVITIES:
            continue
        _, label = UP_FALL_ACTIVITIES[activity_num]

        subject_id = f"S{subject_num:02d}"
        seq_id = (
            f"up_fall_S{subject_num:02d}"
            f"_A{activity_num:02d}"
            f"_T{trial_num:02d}"
            f"_C{cam_num}"
        )

        sequences.append(SequenceInfo(
            sequence_id=seq_id,
            video_path=item,
            label=label,
            subject_id=subject_id,
            camera=f"cam{cam_num}",
            dataset="up_fall",
            is_frame_dir=item.is_dir(),
        ))

    return sequences
