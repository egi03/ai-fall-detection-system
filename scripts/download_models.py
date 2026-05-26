"""
Download pretrained model weights from GitHub Releases.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model run5       # best single model only
    python scripts/download_models.py --model urfd_run3  # Run 3 model (needed for ensemble)

Models available:
    run5       — Best single model (BiLSTM, URFD, stride=2, AUC=0.888)
    urfd_run3  — Run 3 model (BiLSTM, URFD, needed for ensemble, AUC=0.878)

Ensemble (best overall, AUC=0.897):
    python scripts/download_models.py           # downloads both
    python scripts/demo.py --model models/urfd --model2 models/run5_stride2_aug
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# ── Update these URLs after creating the GitHub Release ──────────────────────
GITHUB_RELEASE_BASE = (
    "https://github.com/egi03/ai-fall-detection-system/releases/download/v1.0.0"
)

MODELS = {
    "run5": {
        "url": f"{GITHUB_RELEASE_BASE}/run5_stride2_aug.zip",
        "dest": "models/run5_stride2_aug",
        "sha256": "1f6e654683896d1bfde3a9411e2f74de49508301790a7fa48e3e8333febf5281",
        "description": "Best single model — BiLSTM URFD stride=2 augmented (AUC=0.888)",
    },
    "urfd_run3": {
        "url": f"{GITHUB_RELEASE_BASE}/urfd_run3.zip",
        "dest": "models/urfd",
        "sha256": "35a95341e6a40b75a29cfa0088de93fc9060c46c7d16bef2377d523ab0222d22",
        "description": "Run 3 model — needed for ensemble (AUC=0.878)",
    },
}
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

MEDIAPIPE_MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size > 1024:
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        return "version https://git-lfs.github.com" in content
    except Exception:
        return False


def download_mediapipe_models() -> None:
    print("\n[MediaPipe Pose Models] Checking and downloading (.task files)...")
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in MEDIAPIPE_MODELS.items():
        dest_path = models_dir / filename
        if dest_path.exists() and not is_lfs_pointer(dest_path):
            print(f"  {filename} already exists and is a valid binary — skipping")
            continue

        if is_lfs_pointer(dest_path):
            print(f"  {filename} is a Git LFS pointer. Overwriting with the real binary...")
            try:
                dest_path.unlink()
            except Exception as e:
                print(f"  Warning: could not delete Git LFS pointer {filename}: {e}")

        print(f"  Downloading {filename}...")
        _download(url, dest_path)
        print(f"  Successfully downloaded {filename}.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest_path: Path) -> None:
    print(f"  Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            while chunk := response.read(65536):
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%", end="", flush=True)
        print()
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR: {e.code} {e.reason}")
        print("  Make sure the GitHub Release/storage URL exists and the URL is correct.")
        print(f"  Expected: {url}")
        sys.exit(1)


def has_lfs_pointers(directory: Path) -> bool:
    if not directory.exists():
        return False
    for path in directory.rglob("*"):
        if path.is_file() and is_lfs_pointer(path):
            return True
    return False


def download_model(key: str) -> None:
    info = MODELS[key]
    zip_path = ROOT / f"_tmp_{key}.zip"
    dest = ROOT / info["dest"]

    if dest.exists():
        if has_lfs_pointers(dest):
            print(f"  {dest} contains Git LFS pointers. Deleting and re-downloading...")
            shutil.rmtree(dest, ignore_errors=True)
        else:
            print(f"  {dest} already exists — skipping (delete to re-download)")
            return

    print(f"\n[{key}] {info['description']}")
    _download(info["url"], zip_path)

    if info["sha256"]:
        actual = _sha256(zip_path)
        if actual != info["sha256"]:
            zip_path.unlink()
            print(f"  ERROR: SHA-256 mismatch. Expected {info['sha256']}, got {actual}")
            sys.exit(1)
        print(f"  SHA-256 verified.")

    print(f"  Extracting to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest.parent)
    zip_path.unlink()
    print(f"  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pretrained model weights")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Which model to download (default: both)",
    )
    args = parser.parse_args()

    if "YOUR_USERNAME" in GITHUB_RELEASE_BASE:
        print("ERROR: Update GITHUB_RELEASE_BASE in scripts/download_models.py")
        print("       Replace YOUR_USERNAME with your GitHub username.")
        sys.exit(1)

    targets = [args.model] if args.model else list(MODELS.keys())
    for key in targets:
        download_model(key)

    # Always check and download MediaPipe pose task models as well
    download_mediapipe_models()

    print("\nAll models downloaded. Run the demo:")
    print("  python scripts/demo.py --source samples/fall_sample.mp4 --model models/run5_stride2_aug")
    print("  python scripts/demo.py  # live webcam")


if __name__ == "__main__":
    main()
