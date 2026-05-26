"""
Run the full preprocessing pipeline.

Extracts pose keypoints from video datasets using the configured
pose estimator, applies normalization, and saves processed keypoint
sequences as NumPy arrays for training.

Usage:
    python scripts/preprocess.py --dataset urfd
    python scripts/preprocess.py --dataset urfd --skip-extraction

Reference: research/8.2 - Person-centric normalization pipeline.
Reference: research/8.3 - Feature engineering preprocessing.
"""

import argparse
import json
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Setup logging before importing project modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess")

PROJECT_ROOT = Path(__file__).parent.parent


def extract_keypoints_from_video(
    video_path: Path,
    is_frame_dir: bool = False,
    target_fps: int = 15,
    crop_right_half: bool = False,
) -> Optional[np.ndarray]:
    """
    Extract pose keypoints from a video file or frame directory.

    Uses MediaPipe Pose to estimate 33 keypoints per frame.
    Resamples video to target FPS for consistency.

    Parameters
    ----------
    video_path : Path
        Path to the video file or directory of frame images.
    is_frame_dir : bool
        True if video_path is a directory of frame images.
    target_fps : int
        Target frames per second for resampling.

    Returns
    -------
    np.ndarray or None
        Keypoint array of shape (T, 33, 4) where 4 = (x, y, z, visibility).
        Returns None if extraction fails.
    """
    # ZIP files: extract to Camera{N}/ directory adjacent to ZIP, then process
    # as a frame directory.  ZIP naming convention:
    #   Subject1Activity1Trial1Camera1.zip  →  extract to  Trial1/Camera1/
    # DECISION: Extract on first encounter, reuse on subsequent runs to avoid
    # re-extracting large archives.
    if str(video_path).lower().endswith(".zip"):
        cam_match = re.search(r"Camera(\d+)", video_path.name, re.IGNORECASE)
        cam_num = cam_match.group(1) if cam_match else "0"
        extracted_dir = video_path.parent / f"Camera{cam_num}"

        needs_extraction = (
            not extracted_dir.exists()
            or not any(extracted_dir.iterdir())
        )
        if needs_extraction:
            logger.info(f"Extracting ZIP: {video_path.name} → {extracted_dir}")
            extracted_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(str(video_path), "r") as zf:
                    zf.extractall(str(extracted_dir))
            except zipfile.BadZipFile as exc:
                logger.error(f"Bad ZIP file {video_path}: {exc}")
                return None
        else:
            logger.info(f"Reusing existing extraction: {extracted_dir}")

        video_path = extracted_dir
        is_frame_dir = True

    try:
        import cv2
        import mediapipe as mp
    except ImportError as e:
        logger.error(
            f"Required dependency not installed: {e}. "
            "Install with: pip install opencv-python mediapipe"
        )
        return None

    # Resolve PoseLandmarker model file
    model_dir = PROJECT_ROOT / "models"
    model_path = model_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        logger.error(
            f"MediaPipe model not found: {model_path}. "
            "Run: python scripts/download_dataset.py --pose-model"
        )
        return None

    # Collect frames
    frames = []
    source_fps = target_fps

    if is_frame_dir:
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        frame_paths = sorted(
            p for p in video_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is not None:
                frames.append(img)
        # Assume source FPS matches target for frame directories
        source_fps = target_fps
    else:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None

        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = 30.0  # fallback

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        logger.warning(f"No frames extracted from: {video_path}")
        return None

    # URFD MP4s are side-by-side: left=depth, right=RGB. Crop to RGB half.
    if crop_right_half:
        w = frames[0].shape[1]
        frames = [f[:, w // 2:, :] for f in frames]

    # Resample to target FPS if needed
    if abs(source_fps - target_fps) > 1.0 and source_fps > 0:
        step = max(1, round(source_fps / target_fps))
        frames = frames[::step]

    # Extract keypoints with MediaPipe PoseLandmarker (Tasks API)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    keypoints_list = []
    timestamp_ms = 0

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )
        timestamp_ms += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            kps = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in landmarks
            ], dtype=np.float32)
        else:
            # No pose detected — fill with NaN
            kps = np.full((33, 4), np.nan, dtype=np.float32)

        keypoints_list.append(kps)

    landmarker.close()

    return np.array(keypoints_list, dtype=np.float32)


def save_processed_sequence(
    keypoints: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save a preprocessed keypoint sequence to disk.

    Parameters
    ----------
    keypoints : np.ndarray
        Keypoint array of shape (T, N, C).
    output_path : Path
        Path to save the .npy file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), keypoints)


def run_preprocessing(
    dataset_name: str,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    target_fps: int = 15,
    skip_extraction: bool = False,
    normalize: bool = True,
) -> Dict:
    """
    Run the full preprocessing pipeline for a dataset.

    Steps:
    1. Load dataset sequences via DatasetLoader
    2. Extract keypoints from each video (or load existing)
    3. Apply preprocessing (normalization, imputation)
    4. Save processed sequences and metadata

    Parameters
    ----------
    dataset_name : str
        Dataset to process: 'urfd', 'le2i', 'up_fall'.
    data_dir : Path, optional
        Raw data directory. Defaults to data/raw/{dataset_name}/.
    output_dir : Path, optional
        Output directory for processed data.
        Defaults to data/processed/{dataset_name}/.
    target_fps : int
        Target frames per second for resampling.
    skip_extraction : bool
        If True, skip keypoint extraction and load existing .npy files.
    normalize : bool
        Whether to apply person-centric normalization.

    Returns
    -------
    dict
        Processing summary with counts and statistics.
    """
    from src.data_processing.loader import DatasetLoader
    from src.data_processing.preprocessor import KeypointPreprocessor

    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "raw" / dataset_name
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "processed" / dataset_name

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    logger.info(f"Preprocessing dataset: {dataset_name}")
    logger.info(f"Raw data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset metadata
    loader = DatasetLoader(dataset_name, data_dir)
    sequences = loader.load()
    logger.info(f"Found {len(sequences)} sequences")

    # Initialize preprocessor
    preprocessor = KeypointPreprocessor(
        keypoint_format="mediapipe",
        normalize=normalize,
        confidence_threshold=0.5,
    )

    # Create output directories
    raw_kp_dir = output_dir / "keypoints_raw"
    processed_kp_dir = output_dir / "keypoints_processed"
    raw_kp_dir.mkdir(parents=True, exist_ok=True)
    processed_kp_dir.mkdir(parents=True, exist_ok=True)

    # Processing stats
    stats = {
        "dataset": dataset_name,
        "total_sequences": len(sequences),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_frames": 0,
        "nan_frames": 0,
    }

    metadata_records = []

    for i, seq in enumerate(sequences):
        logger.info(
            f"[{i + 1}/{len(sequences)}] Processing {seq.sequence_id}..."
        )

        raw_path = raw_kp_dir / f"{seq.sequence_id}.npy"
        processed_path = processed_kp_dir / f"{seq.sequence_id}.npy"

        # Step 1: Extract or load raw keypoints
        if skip_extraction and raw_path.exists():
            keypoints = np.load(str(raw_path))
            logger.info(f"  Loaded existing keypoints: {keypoints.shape}")
        elif skip_extraction and processed_path.exists():
            stats["skipped"] += 1
            logger.info("  Already processed, skipping.")
            continue
        else:
            # URFD MP4s contain side-by-side depth+RGB; crop to RGB half
            crop_right_half = dataset_name == "urfd" and not seq.is_frame_dir
            keypoints = extract_keypoints_from_video(
                seq.video_path,
                is_frame_dir=seq.is_frame_dir,
                target_fps=target_fps,
                crop_right_half=crop_right_half,
            )

            if keypoints is None:
                stats["failed"] += 1
                logger.warning(f"  FAILED: Could not extract keypoints")
                continue

            # Save raw keypoints
            save_processed_sequence(keypoints, raw_path)
            logger.info(f"  Raw keypoints: {keypoints.shape}")

        # Step 2: Preprocess (normalize + impute)
        processed = preprocessor.preprocess_sequence(keypoints)

        # Step 3: Save processed
        save_processed_sequence(processed, processed_path)

        # Stats
        num_frames = processed.shape[0]
        nan_count = np.isnan(processed).any(axis=(1, 2)).sum()
        stats["processed"] += 1
        stats["total_frames"] += num_frames
        stats["nan_frames"] += int(nan_count)

        # Record metadata
        metadata_records.append({
            "sequence_id": seq.sequence_id,
            "label": seq.label,
            "subject_id": seq.subject_id,
            "camera": seq.camera,
            "num_frames": num_frames,
            "nan_frames": int(nan_count),
            "raw_path": str(raw_path.relative_to(PROJECT_ROOT)),
            "processed_path": str(processed_path.relative_to(PROJECT_ROOT)),
        })

        logger.info(
            f"  Processed: {processed.shape}, "
            f"NaN frames: {nan_count}/{num_frames}"
        )

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "stats": stats,
                "sequences": metadata_records,
            },
            f,
            indent=2,
        )

    logger.info("=" * 50)
    logger.info(f"Preprocessing complete: {dataset_name}")
    logger.info(f"  Processed: {stats['processed']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Total frames: {stats['total_frames']}")
    logger.info(f"  NaN frames: {stats['nan_frames']}")
    logger.info(f"  Metadata saved: {metadata_path}")

    return stats


def main() -> None:
    """Execute the full preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess fall detection datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["urfd", "le2i", "up_fall"],
        help="Dataset to preprocess.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override raw data directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=15,
        help="Target FPS for resampling (default: 15).",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip keypoint extraction, load existing .npy files.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip person-centric normalization.",
    )

    args = parser.parse_args()

    start_time = time.time()

    run_preprocessing(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        skip_extraction=args.skip_extraction,
        normalize=not args.no_normalize,
    )

    elapsed = time.time() - start_time
    logger.info(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
