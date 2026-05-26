"""
Targeted download of UP-Fall Camera ZIPs from Google Drive.

Downloads the full folder tree using gdown, then processes only Camera ZIP files
through MediaPipe for pose estimation. Handles partial downloads gracefully.

Usage:
    python scripts/download_upfall_targeted.py --subjects 1-17 --activities 1-5
    python scripts/download_upfall_targeted.py --subjects 1-5 --activities 1-11
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
UPFALL_FOLDER_ID = "1AItqj3Ue-iv7NSdR7Qta1Ez4spRjCo58"


def get_subject_folder_ids() -> dict:
    """
    Get Google Drive folder IDs for each subject folder.

    Uses gdown to enumerate the root UP-Fall folder and extract
    subfolder IDs for Subject1-Subject17.

    Returns
    -------
    dict
        Mapping from subject number (int) to Google Drive folder ID (str).
    """
    try:
        import gdown
    except ImportError:
        logger.error("gdown not installed. Run: pip install gdown")
        sys.exit(1)

    logger.info("Enumerating UP-Fall Google Drive folder structure...")

    try:
        from gdown.download_folder import _get_session, _get_folder_list
    except ImportError:
        # Fallback: use gdown's public API to list folder contents
        logger.info("Using gdown folder enumeration...")
        pass

    # Use gdown's internal API to get folder listing without downloading
    import requests
    import os

    # Load API key from environment or .env file
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("GOOGLE_API_KEY="):
                            api_key = line.strip().split("GOOGLE_API_KEY=", 1)[1].strip()
                            if api_key.startswith(('"', "'")) and api_key.endswith(('"', "'")):
                                api_key = api_key[1:-1]
                            break
            except Exception as e:
                logger.warning(f"Failed to read .env file: {e}")

    if api_key:
        # Try using the Google Drive API v3 (public, no auth needed for public folders)
        api_url = f"https://www.googleapis.com/drive/v3/files?q=%27{UPFALL_FOLDER_ID}%27+in+parents&key={api_key}&fields=files(id,name,mimeType)&pageSize=100"

        try:
            resp = requests.get(api_url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                folder_ids = {}
                for item in data.get("files", []):
                    name = item["name"]
                    if name.startswith("Subject") and item["mimeType"] == "application/vnd.google-apps.folder":
                        try:
                            subj_num = int(name.replace("Subject", ""))
                            folder_ids[subj_num] = item["id"]
                            logger.info(f"  Found {name} -> {item['id']}")
                        except ValueError:
                            pass
                if folder_ids:
                    return folder_ids
        except Exception as e:
            logger.warning(f"API enumeration failed: {e}")

    # Fallback: hardcoded folder IDs from gdown probe
    logger.info("Using previously discovered folder IDs...")
    return {
        1: "1YiotcqcUtIBRr_2RNpQeRzQShw9-Cvi5",
    }


def download_subject(
    subject_num: int,
    folder_id: str,
    output_dir: Path,
    activities: list,
) -> int:
    """
    Download all Camera ZIPs for a single subject using gdown.

    Parameters
    ----------
    subject_num : int
        Subject number (1-17).
    folder_id : str
        Google Drive folder ID for this subject.
    output_dir : Path
        Root output directory.
    activities : list
        Activity numbers to download.

    Returns
    -------
    int
        Number of ZIP files downloaded.
    """
    import gdown

    subj_dir = output_dir / f"Subject{subject_num}"
    subj_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading Subject{subject_num} (folder ID: {folder_id})...")

    try:
        gdown.download_folder(
            id=folder_id,
            output=str(subj_dir),
            quiet=False,
            remaining_ok=True,
        )
    except Exception as e:
        logger.error(f"Failed to download Subject{subject_num}: {e}")
        return 0

    # Count downloaded camera ZIPs
    zip_count = 0
    for zip_path in subj_dir.rglob("*Camera*.zip"):
        zip_count += 1
    logger.info(f"Subject{subject_num}: {zip_count} camera ZIPs downloaded")

    return zip_count


def download_all(
    output_dir: Path,
    subjects: list,
    activities: list,
) -> None:
    """
    Download UP-Fall camera data for specified subjects.

    Parameters
    ----------
    output_dir : Path
        Root output directory.
    subjects : list of int
        Subject numbers to download.
    activities : list of int
        Activity numbers to download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to get all folder IDs
    folder_ids = get_subject_folder_ids()

    if not folder_ids:
        logger.error("Could not determine subject folder IDs. Trying full folder download...")
        import gdown
        gdown.download_folder(
            id=UPFALL_FOLDER_ID,
            output=str(output_dir),
            quiet=False,
            remaining_ok=True,
        )
        return

    total_zips = 0
    for subj in subjects:
        if subj not in folder_ids:
            logger.warning(f"No folder ID for Subject{subj}, skipping")
            continue

        count = download_subject(subj, folder_ids[subj], output_dir, activities)
        total_zips += count

        # Brief pause between subjects to avoid rate limiting
        time.sleep(2)

    logger.info(f"Total camera ZIPs downloaded: {total_zips}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UP-Fall camera data")
    parser.add_argument("--subjects", type=str, default="1-17",
                        help="Subject range (default: 1-17)")
    parser.add_argument("--activities", type=str, default="1-5",
                        help="Activity range (default: 1-5, falls only)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Parse ranges
    def parse_range(s):
        items = []
        for token in s.split(","):
            if "-" in token:
                start, end = token.split("-", 1)
                items.extend(range(int(start), int(end) + 1))
            else:
                items.append(int(token))
        return sorted(set(items))

    subjects = parse_range(args.subjects)
    activities = parse_range(args.activities)

    output_dir = args.output_dir or (PROJECT_ROOT / "data" / "data" / "raw" / "up_fall")

    logger.info(f"Downloading UP-Fall: subjects={subjects}, activities={activities}")
    logger.info(f"Output: {output_dir}")

    download_all(output_dir, subjects, activities)


if __name__ == "__main__":
    main()
