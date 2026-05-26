"""
Download and extract the UP-Fall Detection Dataset from Google Drive.

The UP-Fall dataset contains 17 subjects, 11 activities, 3 trials per activity,
and 2 camera views per trial.  Activities 1-5 are falls; activities 6-11 are ADLs.
Each camera view is stored as a ZIP archive of JPEG frames.

Dataset structure after download + extraction:
    data/data/raw/up_fall/
    └── Subject{N}/
        └── Activity{M}/
            └── Trial{K}/
                ├── Subject{N}Activity{M}Trial{K}Camera1.zip
                ├── Subject{N}Activity{M}Trial{K}Camera2.zip
                ├── Camera1/       (extracted frames)
                └── Camera2/       (extracted frames)

Usage:
    # Download and extract everything (all 17 subjects)
    python scripts/download_upfall.py

    # Download only subjects 1-5, cameras 1 and 2
    python scripts/download_upfall.py --subjects 1-5 --cameras 1,2

    # Extract only (ZIPs already downloaded manually)
    python scripts/download_upfall.py --extract-only

    # Download only, skip extraction
    python scripts/download_upfall.py --no-extract

    # Override destination
    python scripts/download_upfall.py --output-dir /data/up_fall

Google Drive folder:
    https://drive.google.com/drive/folders/1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58

Reference: research/1.1 - UP-Fall dataset specifications.
"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_upfall")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent

# Google Drive folder ID for the UP-Fall dataset
UPFALL_FOLDER_ID = "1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58"
UPFALL_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58"
)

UP_FALL_NUM_SUBJECTS = 17
UP_FALL_NUM_ACTIVITIES = 11
UP_FALL_NUM_TRIALS = 3

# Image extensions used in extracted frame directories
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def _parse_range_or_list(value: str) -> List[int]:
    """
    Parse a range string such as '1-5' or a comma-separated list '1,3,5'.

    Parameters
    ----------
    value : str
        Input string from the command line.

    Returns
    -------
    list of int
        Sorted list of integer values.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be parsed.
    """
    items = []
    for token in value.split(","):
        token = token.strip()
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
                items.extend(range(start, end + 1))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid range token: '{token}'"
                )
        else:
            try:
                items.append(int(token))
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid integer token: '{token}'"
                )
    return sorted(set(items))


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def extract_zip(
    zip_path: Path,
    output_dir: Path,
) -> bool:
    """
    Extract a ZIP archive to the specified directory.

    If the directory already exists and contains at least one image file,
    extraction is skipped (idempotent).

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    output_dir : Path
        Directory to extract frames into.

    Returns
    -------
    bool
        True if extraction was performed or already complete; False on error.
    """
    # Check if already extracted
    if output_dir.exists():
        has_images = any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            for f in output_dir.iterdir()
        )
        if has_images:
            logger.debug(f"Already extracted: {output_dir}")
            return True

    if not zip_path.exists():
        logger.error(f"ZIP not found, cannot extract: {zip_path}")
        return False

    logger.info(f"Extracting {zip_path.name} → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(output_dir))
        return True
    except zipfile.BadZipFile as exc:
        logger.error(f"Bad ZIP file {zip_path}: {exc}")
        return False
    except OSError as exc:
        logger.error(f"OS error extracting {zip_path}: {exc}")
        return False


def extract_all_zips(
    raw_dir: Path,
    subjects: List[int],
    activities: List[int],
    trials: List[int],
    cameras: List[int],
) -> Tuple[int, int]:
    """
    Extract all camera ZIP files found under raw_dir for the specified
    subject/activity/trial/camera combinations.

    Parameters
    ----------
    raw_dir : Path
        Root of the UP-Fall raw data directory.
    subjects : list of int
        Subject numbers to process.
    activities : list of int
        Activity numbers to process.
    trials : list of int
        Trial numbers to process.
    cameras : list of int
        Camera numbers to extract.

    Returns
    -------
    (extracted_count, failed_count) : tuple of int
        Counts of successfully extracted and failed archives.
    """
    extracted = 0
    failed = 0

    for subj in subjects:
        subj_dir = raw_dir / f"Subject{subj}"
        if not subj_dir.exists():
            logger.debug(f"Subject directory not found, skipping: {subj_dir}")
            continue

        for act in activities:
            act_dir = subj_dir / f"Activity{act}"
            if not act_dir.exists():
                continue

            for trial in trials:
                trial_dir = act_dir / f"Trial{trial}"
                if not trial_dir.exists():
                    continue

                for cam in cameras:
                    zip_name = (
                        f"Subject{subj}Activity{act}Trial{trial}Camera{cam}.zip"
                    )
                    zip_path = trial_dir / zip_name
                    cam_dir = trial_dir / f"Camera{cam}"

                    ok = extract_zip(zip_path, cam_dir)
                    if ok:
                        extracted += 1
                    else:
                        failed += 1

    return extracted, failed


# ---------------------------------------------------------------------------
# Download via gdown
# ---------------------------------------------------------------------------

def _attempt_gdown_folder_download(output_dir: Path) -> bool:
    """
    Attempt to download the entire UP-Fall folder via gdown.

    gdown downloads the full Google Drive folder tree.  The resulting
    directory layout should mirror the Google Drive structure.  After
    download we call extract_all_zips() to unpack the archives.

    Parameters
    ----------
    output_dir : Path
        Destination directory for the downloaded files.

    Returns
    -------
    bool
        True if download appeared to succeed (directory is non-empty).
    """
    try:
        import gdown  # type: ignore[import]
    except ImportError:
        logger.error(
            "gdown is not installed.  Install with: pip install gdown"
        )
        return False

    logger.info(
        f"Attempting gdown folder download (folder ID={UPFALL_FOLDER_ID})..."
    )
    logger.info(
        "Note: Google Drive may require you to confirm a virus-scan warning "
        "in your browser first.  If download stalls, see the manual "
        "instructions printed below."
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            id=UPFALL_FOLDER_ID,
            output=str(output_dir),
            quiet=False,
            remaining_ok=True,
        )
    except Exception as exc:  # gdown raises varied exceptions
        logger.error(f"gdown download failed: {exc}")
        return False

    # Verify something was downloaded
    has_content = any(output_dir.iterdir())
    if not has_content:
        logger.error("gdown completed but output directory is empty.")
        return False

    logger.info(f"gdown download complete.  Files in: {output_dir}")
    return True


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def _count_expected_zips(
    subjects: List[int],
    activities: List[int],
    trials: List[int],
    cameras: List[int],
) -> int:
    """Return total number of ZIP files expected for the given parameters."""
    return len(subjects) * len(activities) * len(trials) * len(cameras)


def _count_existing_zips(
    raw_dir: Path,
    subjects: List[int],
    activities: List[int],
    trials: List[int],
    cameras: List[int],
) -> int:
    """Count how many expected ZIP files are already present on disk."""
    count = 0
    for subj in subjects:
        for act in activities:
            for trial in trials:
                for cam in cameras:
                    zip_path = (
                        raw_dir
                        / f"Subject{subj}"
                        / f"Activity{act}"
                        / f"Trial{trial}"
                        / f"Subject{subj}Activity{act}Trial{trial}Camera{cam}.zip"
                    )
                    if zip_path.exists():
                        count += 1
    return count


# ---------------------------------------------------------------------------
# Main download orchestration
# ---------------------------------------------------------------------------

def download_upfall(
    output_dir: Path,
    subjects: List[int],
    activities: List[int],
    trials: List[int],
    cameras: List[int],
    no_extract: bool = False,
    extract_only: bool = False,
) -> None:
    """
    Download (and optionally extract) the UP-Fall dataset.

    The function first checks how many files are already present.  If all
    expected ZIPs exist it skips the download.  Otherwise it attempts an
    automatic gdown folder download and prints manual instructions on failure.

    Parameters
    ----------
    output_dir : Path
        Root directory for the UP-Fall raw data.
    subjects : list of int
        Subject numbers to download (1-17).
    activities : list of int
        Activity numbers to download (1-11).
    trials : list of int
        Trial numbers to download (1-3).
    cameras : list of int
        Camera numbers to download (1 or 2).
    no_extract : bool
        If True, skip ZIP extraction after download.
    extract_only : bool
        If True, skip download and only extract existing ZIPs.
    """
    _print_manual_instructions()

    output_dir.mkdir(parents=True, exist_ok=True)

    total_expected = _count_expected_zips(subjects, activities, trials, cameras)
    logger.info(
        f"UP-Fall download target: {len(subjects)} subjects, "
        f"{len(activities)} activities, {len(trials)} trials, "
        f"{len(cameras)} cameras → {total_expected} ZIP files expected"
    )
    logger.info(f"Output directory: {output_dir}")

    # -----------------------------------------------------------------------
    # Download phase
    # -----------------------------------------------------------------------
    if not extract_only:
        existing = _count_existing_zips(subjects, activities, trials, cameras, output_dir)
        logger.info(f"Already downloaded: {existing}/{total_expected} ZIPs")

        if existing >= total_expected:
            logger.info("All expected ZIP files already present.  Skipping download.")
        else:
            logger.info(
                f"Missing {total_expected - existing} ZIP(s). "
                "Attempting automatic download via gdown..."
            )
            success = _attempt_gdown_folder_download(output_dir)
            if not success:
                logger.error(
                    "Automatic download failed.  Please download manually using "
                    "the instructions printed above, then re-run with --extract-only."
                )
                sys.exit(1)

            # Re-count after download
            existing_after = _count_existing_zips(
                subjects, activities, trials, cameras, output_dir
            )
            logger.info(
                f"After download: {existing_after}/{total_expected} ZIPs present"
            )

    # -----------------------------------------------------------------------
    # Extraction phase
    # -----------------------------------------------------------------------
    if no_extract:
        logger.info("Skipping extraction (--no-extract was set).")
        return

    logger.info("Starting ZIP extraction...")
    extracted, failed = extract_all_zips(
        raw_dir=output_dir,
        subjects=subjects,
        activities=activities,
        trials=trials,
        cameras=cameras,
    )
    logger.info(
        f"Extraction complete: {extracted} extracted, {failed} failed"
    )
    if failed > 0:
        logger.warning(
            f"{failed} ZIP file(s) could not be extracted.  "
            "Check logs above for details."
        )


def _print_manual_instructions() -> None:
    """Print manual download instructions to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("UP-Fall Dataset Download")
    print(sep)
    print()
    print("Automatic download (requires: pip install gdown):")
    print("  python scripts/download_upfall.py")
    print()
    print("Manual download (if automatic fails):")
    print(f"  1. Open: {UPFALL_FOLDER_URL}")
    print(f"  2. Download all folders (Subject1-Subject17)")
    print(f"  3. Place extracted contents into:")
    print(f"       <project>/data/data/raw/up_fall/")
    print(f"     so you get:  up_fall/Subject1/Activity1/Trial1/<zip files>")
    print()
    print("Alternative: HAR-UP PyDrive downloader")
    print("  See: https://sites.google.com/up.edu.mx/har-up")
    print()
    print("After manual download, run extraction only:")
    print("  python scripts/download_upfall.py --extract-only")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and extract the UP-Fall Detection Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root directory for UP-Fall raw data "
            "(default: data/data/raw/up_fall relative to project root)"
        ),
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=f"1-{UP_FALL_NUM_SUBJECTS}",
        help=(
            "Subjects to download, as a range '1-17' or list '1,3,5' "
            f"(default: all 1-{UP_FALL_NUM_SUBJECTS})"
        ),
    )
    parser.add_argument(
        "--activities",
        type=str,
        default=f"1-{UP_FALL_NUM_ACTIVITIES}",
        help=(
            "Activities to download, as a range or list "
            f"(default: all 1-{UP_FALL_NUM_ACTIVITIES})"
        ),
    )
    parser.add_argument(
        "--trials",
        type=str,
        default=f"1-{UP_FALL_NUM_TRIALS}",
        help=(
            "Trials to download, as a range or list "
            f"(default: all 1-{UP_FALL_NUM_TRIALS})"
        ),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="1,2",
        help="Cameras to download as a comma-separated list (default: 1,2)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download ZIPs but skip extraction.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip download; only extract already-downloaded ZIPs.",
    )
    return parser


def main() -> None:
    """Entry point for the download script."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.no_extract and args.extract_only:
        parser.error("--no-extract and --extract-only are mutually exclusive.")

    subjects = _parse_range_or_list(args.subjects)
    activities = _parse_range_or_list(args.activities)
    trials = _parse_range_or_list(args.trials)
    cameras = _parse_range_or_list(args.cameras)

    # Validate ranges against dataset specification
    for s in subjects:
        if not (1 <= s <= UP_FALL_NUM_SUBJECTS):
            parser.error(f"Subject {s} out of range [1, {UP_FALL_NUM_SUBJECTS}]")
    for a in activities:
        if not (1 <= a <= UP_FALL_NUM_ACTIVITIES):
            parser.error(f"Activity {a} out of range [1, {UP_FALL_NUM_ACTIVITIES}]")
    for t in trials:
        if not (1 <= t <= UP_FALL_NUM_TRIALS):
            parser.error(f"Trial {t} out of range [1, {UP_FALL_NUM_TRIALS}]")
    for c in cameras:
        if c not in (1, 2):
            parser.error(f"Camera {c} not valid; UP-Fall has cameras 1 and 2 only.")

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = PROJECT_ROOT / "data" / "data" / "raw" / "up_fall"

    download_upfall(
        output_dir=output_dir,
        subjects=subjects,
        activities=activities,
        trials=trials,
        cameras=cameras,
        no_extract=args.no_extract,
        extract_only=args.extract_only,
    )


if __name__ == "__main__":
    main()
