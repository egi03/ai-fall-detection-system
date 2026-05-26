"""
Dataset download helper scripts.

Downloads and extracts supported fall detection datasets
to the data/raw/ directory. Provides verification of downloaded
files via checksums when available.

Reference: research/1.1 - Dataset URLs and access methods.

Note: Some datasets require manual download due to access restrictions.
This script handles automated downloads where possible and provides
clear instructions for manual downloads.
"""

import hashlib
import logging
import zipfile
from pathlib import Path
from typing import Optional

# Setup basic logging for standalone script usage
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_dataset")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a URL using urllib.

    Parameters
    ----------
    url : str
        URL to download from.
    output_path : Path
        Local path to save the file.

    Raises
    ------
    RuntimeError
        If download fails.
    """
    import urllib.request
    import urllib.error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading: {url}")
    logger.info(f"Destination: {output_path}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total else None

            downloaded = 0
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  Progress: {pct:.1f}%", end="", flush=True)

            if total:
                print()  # newline after progress

        logger.info(f"Download complete: {output_path.name}")

    except urllib.error.URLError as e:
        raise RuntimeError(f"Download failed: {e}") from e


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    """
    Extract a ZIP archive.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP file.
    output_dir : Path
        Directory to extract into.
    """
    logger.info(f"Extracting: {zip_path.name} -> {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)

    logger.info("Extraction complete")


def download_urfd(output_dir: Optional[Path] = None) -> None:
    """
    Download the UR Fall Detection Dataset.

    URFD is hosted at the University of Rzeszow. The dataset contains
    30 fall sequences and 40 ADL sequences from 5 subjects, captured
    with 2 cameras (parallel + ceiling-mounted).

    Reference: research/1.1 - URFD, CC BY-NC-SA 4.0 license.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save the downloaded files.
        Defaults to data/raw/urfd/.

    Notes
    -----
    The dataset moved from the old fenix.univ.rzeszow.pl server to
    the new fenix.ur.edu.pl server. Each sequence is a separate file.
    This function downloads only the RGB MP4 videos (smaller) and
    the CSV metadata files. If you also need depth or PNG frame ZIPs,
    download them manually from:
        https://fenix.ur.edu.pl/mkepski/ds/uf.html
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw" / "urfd"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # New base URL (moved from fenix.univ.rzeszow.pl to fenix.ur.edu.pl)
    base_url = "https://fenix.ur.edu.pl/mkepski/ds/data"
    page_url = "https://fenix.ur.edu.pl/mkepski/ds/uf.html"

    logger.info("=" * 60)
    logger.info("URFD Dataset Download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Downloading RGB MP4 videos and CSV metadata.")
    logger.info(f"Full dataset page: {page_url}")
    logger.info("")

    # Build list of all files to download:
    # Falls: sequences 01-30, cameras 0 and 1
    # ADLs:  sequences 01-40, camera 0 only (cam1 not available for ADLs)
    files_to_download: list[str] = []

    for i in range(1, 31):
        seq = f"{i:02d}"
        files_to_download += [
            f"fall-{seq}-cam0.mp4",
            f"fall-{seq}-cam1.mp4",
            f"fall-{seq}-data.csv",
            f"fall-{seq}-acc.csv",
        ]

    for i in range(1, 41):
        seq = f"{i:02d}"
        files_to_download += [
            f"adl-{seq}-cam0.mp4",
            f"adl-{seq}-data.csv",
            f"adl-{seq}-acc.csv",
        ]

    # Pre-extracted feature CSVs (useful for quick baseline)
    files_to_download += [
        "urfall-cam0-falls.csv",
        "urfall-cam0-adls.csv",
    ]

    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for filename in files_to_download:
        dest = output_dir / filename
        if dest.exists():
            skipped += 1
            continue

        url = f"{base_url}/{filename}"
        try:
            _download_file(url, dest)
            downloaded += 1
        except RuntimeError as e:
            logger.warning(f"Failed: {filename} — {e}")
            failed.append(filename)

    logger.info(
        f"URFD download complete: {downloaded} downloaded, "
        f"{skipped} already present, {len(failed)} failed."
    )
    if failed:
        logger.warning(f"Failed files ({len(failed)}): {failed[:5]}{'...' if len(failed) > 5 else ''}")
        logger.warning(f"Visit {page_url} to download missing files manually.")

    _verify_urfd(output_dir)


def _verify_urfd(data_dir: Path) -> None:
    """
    Verify URFD dataset integrity.

    Checks that the expected number of sequences exist.

    Parameters
    ----------
    data_dir : Path
        URFD dataset root directory.
    """
    import re

    pattern = re.compile(r"(fall|adl)-(\d+)-cam(\d+)", re.IGNORECASE)

    falls = set()
    adls = set()

    for item in data_dir.iterdir():
        name = item.stem if item.is_file() else item.name
        match = pattern.match(name)
        if match:
            action = match.group(1).lower()
            seq_num = int(match.group(2))
            if action == "fall":
                falls.add(seq_num)
            else:
                adls.add(seq_num)

    logger.info(f"URFD verification: {len(falls)} fall sequences, {len(adls)} ADL sequences")

    if len(falls) < 30:
        logger.warning(f"Expected 30 fall sequences, found {len(falls)}")
    if len(adls) < 40:
        logger.warning(f"Expected 40 ADL sequences, found {len(adls)}")

    if len(falls) >= 30 and len(adls) >= 40:
        logger.info("URFD dataset verification PASSED")


def download_le2i(output_dir: Optional[Path] = None) -> None:
    """
    Download the Le2i Fall Detection Dataset.

    Le2i (FDD) contains 221 videos across 4 scenes (Coffee room, Home,
    Office, Lecture room). The original CNRS server is gone; the dataset
    is now on Kaggle (~9.4 GB).

    Reference: research/1.1 - Le2i (FDD), GNU GPL license.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save the downloaded files.
        Defaults to data/raw/le2i/.

    Notes
    -----
    Download options (in order of ease):

    Option A — Kaggle CLI (recommended):
        1. Create a free account at https://www.kaggle.com
        2. Go to Account Settings → API → Create New Token
           This downloads kaggle.json — place it at ~/.kaggle/kaggle.json
        3. chmod 600 ~/.kaggle/kaggle.json
        4. pip install kaggle
        5. kaggle datasets download -d tuyenldvn/falldataset-imvia \\
               -p data/raw/le2i/ --unzip

    Option B — Kaggle browser download:
        1. Visit https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
        2. Click Download (~9.4 GB ZIP)
        3. Extract into data/raw/le2i/

    Expected directory structure after extraction:
        le2i/
        ├── Coffee_room/
        │   ├── Videos/          (.avi files)
        │   └── Annotation_files/ (.csv with fall frame numbers)
        ├── Home/
        │   ├── Videos/
        │   └── Annotation_files/
        ├── Office/
        │   └── Videos/          (no annotations)
        └── Lecture_room/
            └── Videos/          (no annotations)
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw" / "le2i"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kaggle_url = "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia"

    logger.info("=" * 60)
    logger.info("Le2i (FDD) Dataset Download")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")

    # Try Kaggle CLI if available
    import shutil as _shutil
    if _shutil.which("kaggle"):
        logger.info("Kaggle CLI detected — attempting automatic download...")
        import subprocess
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "tuyenldvn/falldataset-imvia",
                "-p", str(output_dir),
                "--unzip",
            ],
            capture_output=False,
        )
        if result.returncode == 0:
            logger.info("Le2i download via Kaggle CLI succeeded.")
            _verify_le2i(output_dir)
            return
        else:
            logger.warning("Kaggle CLI download failed. See instructions below.")
    else:
        logger.info("Kaggle CLI not found.")

    logger.info("")
    logger.info("Le2i requires MANUAL download. Choose one option:")
    logger.info("")
    logger.info("  OPTION A — Kaggle CLI (recommended):")
    logger.info("    pip install kaggle")
    logger.info("    # Place kaggle.json at ~/.kaggle/kaggle.json (from kaggle.com → Account → API)")
    logger.info("    kaggle datasets download -d tuyenldvn/falldataset-imvia \\")
    logger.info(f"        -p {output_dir} --unzip")
    logger.info("")
    logger.info("  OPTION B — Browser:")
    logger.info(f"    Visit: {kaggle_url}")
    logger.info("    Click Download (~9.4 GB) and extract into:")
    logger.info(f"    {output_dir}")
    logger.info("")

    # Verify if data already exists (manually placed)
    rooms = ["Coffee_room", "Home", "Office", "Lecture_room", "Lecture"]
    found_rooms = [r for r in rooms if (output_dir / r).is_dir()]

    if found_rooms:
        logger.info(f"Found room directories: {found_rooms}")
        _verify_le2i(output_dir)
    else:
        logger.warning(
            f"No Le2i data found in {output_dir}. Follow the instructions above."
        )


def _verify_le2i(data_dir: Path) -> None:
    """Verify Le2i dataset integrity."""
    total_videos = 0
    total_annotations = 0

    for room_dir in sorted(data_dir.iterdir()):
        if not room_dir.is_dir():
            continue

        video_dir = room_dir / "Videos"
        if not video_dir.exists():
            video_dir = room_dir

        anno_dir = room_dir / "Annotation_files"

        videos = [
            f for f in video_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".avi", ".mp4"}
        ] if video_dir.exists() else []

        annos = [
            f for f in anno_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        ] if anno_dir.exists() else []

        total_videos += len(videos)
        total_annotations += len(annos)
        logger.info(
            f"  {room_dir.name}: {len(videos)} videos, "
            f"{len(annos)} annotations"
        )

    logger.info(
        f"Le2i total: {total_videos} videos, "
        f"{total_annotations} annotations"
    )


def download_up_fall(output_dir: Optional[Path] = None) -> None:
    """
    Download the UP-Fall Detection Dataset (HAR-UP).

    UP-Fall contains data from 17 subjects performing 5 fall types
    and 6 ADL types, captured with 2 cameras.

    WARNING: The full dataset is ~812 GB. This function downloads
    the preliminary subset (4 subjects, small) or prints instructions
    for the full dataset.

    Reference: research/1.1 - UP-Fall, Open license.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save the downloaded files.
        Defaults to data/raw/up_fall/.

    Notes
    -----
    Access options:

    Option A — Preliminary dataset (4 subjects, small, no auth):
        Auto-downloaded by this script from Google Drive.

    Option B — Full dataset via Google Drive API (~812 GB):
        1. Visit https://sites.google.com/up.edu.mx/har-up/
        2. Clone https://github.com/jpnm561/HAR-UP
        3. Set up Google Drive API credentials (client_secrets.json)
        4. Run DataBaseDownload/Downloader_pydrive.py

    Option C — Pre-extracted 3D skeletons (Zenodo, 3.83 MB, recommended):
        This is the easiest option for prototyping. Contains joint
        coordinate CSVs (x, y, z per joint) with binary fall labels.
        Visit: https://zenodo.org/records/12773013
        Note: Only 5 subjects, not the full 17.
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "raw" / "up_fall"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("UP-Fall Dataset Download (HAR-UP)")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("NOTICE: Full UP-Fall dataset is ~812 GB.")
    logger.info("This script downloads the preliminary subset (4 subjects).")
    logger.info("")

    # Preliminary dataset — small Google Drive file, no auth required
    prelim_file_id = "1Y2MSUijPcB7--PcGoAKhGeqI8GxKK0Pm"
    prelim_zip = output_dir / "up_fall_preliminary.zip"

    if not prelim_zip.exists() and not any(output_dir.iterdir()):
        gdrive_url = (
            f"https://drive.google.com/uc?export=download&id={prelim_file_id}"
        )
        logger.info("Downloading preliminary UP-Fall dataset (4 subjects)...")
        try:
            _download_file(gdrive_url, prelim_zip)
            _extract_zip(prelim_zip, output_dir)
            logger.info("Preliminary UP-Fall dataset extracted successfully.")
        except RuntimeError as e:
            logger.warning(f"Automatic download failed: {e}")
            logger.info("Please download manually. See options in the docstring.")

    logger.info("")
    logger.info("For the full 17-subject dataset (~812 GB):")
    logger.info("  Official site: https://sites.google.com/up.edu.mx/har-up/")
    logger.info("  GitHub:        https://github.com/jpnm561/HAR-UP")
    logger.info("")
    logger.info("For quick prototyping (pre-extracted 3D skeletons, 3.83 MB):")
    logger.info("  Zenodo:        https://zenodo.org/records/12773013")
    logger.info("")

    # Check if data exists
    subject_dirs = [
        d for d in sorted(output_dir.iterdir())
        if d.is_dir() and "subject" in d.name.lower()
    ] if output_dir.exists() else []

    if subject_dirs:
        logger.info(f"Found {len(subject_dirs)} subject directories")
        _verify_up_fall(output_dir)
    else:
        logger.warning(
            f"No UP-Fall subject directories found in {output_dir}. "
            "See download instructions above."
        )


def _verify_up_fall(data_dir: Path) -> None:
    """Verify UP-Fall dataset integrity."""
    import re

    subjects = set()
    activities = set()
    total_sequences = 0

    for item in data_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in {".avi", ".mp4"}:
            total_sequences += 1

        if item.is_dir():
            s_match = re.match(r"Subject(\d+)", item.name, re.IGNORECASE)
            if s_match:
                subjects.add(int(s_match.group(1)))

            a_match = re.match(r"Activity(\d+)", item.name, re.IGNORECASE)
            if a_match:
                activities.add(int(a_match.group(1)))

    logger.info(
        f"UP-Fall: {len(subjects)} subjects, "
        f"{len(activities)} activities, "
        f"{total_sequences} video files"
    )


def main() -> None:
    """
    Run dataset download for all supported datasets.

    Usage:
        python scripts/download_dataset.py [dataset_name]

    If no dataset is specified, provides guidance for all datasets.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download fall detection datasets."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["urfd", "le2i", "up_fall", "all"],
        help="Dataset to download (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory.",
    )

    args = parser.parse_args()

    downloaders = {
        "urfd": download_urfd,
        "le2i": download_le2i,
        "up_fall": download_up_fall,
    }

    if args.dataset == "all":
        for name, func in downloaders.items():
            func(args.output_dir)
            print()
    else:
        downloaders[args.dataset](args.output_dir)


if __name__ == "__main__":
    main()
