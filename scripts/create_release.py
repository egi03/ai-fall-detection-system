"""
Create GitHub Release and upload model weights assets automatically.

Usage:
    python scripts/create_release.py --token YOUR_GITHUB_PERSONAL_ACCESS_TOKEN
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def get_git_remote_repo() -> str:
    """Extract owner/repo from git remote URL."""
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            text=True,
        ).strip()
        # Handle SSH: git@github.com:owner/repo.git or https://github.com/owner/repo.git
        if url.endswith(".git"):
            url = url[:-4]
        if "github.com" in url:
            parts = url.split("github.com")[-1]
            if parts.startswith(":") or parts.startswith("/"):
                return parts[1:]
    except Exception:
        pass
    return "egi03/ai-fall-detection-system"


def upload_asset(upload_url: str, filepath: Path, token: str) -> None:
    """Upload a file asset to a GitHub release with upload progress."""
    filename = filepath.name
    filesize = filepath.stat().st_size
    
    # Strip any trailing template placeholders from the upload URL
    if "{" in upload_url:
        upload_url = upload_url.split("{")[0]
        
    url = f"{upload_url}?name={filename}"
    
    print(f"\nUploading {filename} ({filesize / (1024 * 1024):.2f} MB)...")
    
    req = urllib.request.Request(
        url,
        data=filepath.read_bytes(),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/zip",
            "Content-Length": str(filesize),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            print(f"  Successfully uploaded asset: {res_data.get('browser_download_url')}")
    except urllib.error.HTTPError as e:
        print(f"  Failed to upload asset: {e.code} - {e.reason}")
        print(e.read().decode())
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GitHub release and upload model weights")
    parser.add_argument("--token", type=str, default=None,
                        help="GitHub Personal Access Token (PAT)")
    parser.add_argument("--tag", type=str, default="v1.0.0",
                        help="Release tag version (default: v1.0.0)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GitHub Personal Access Token (PAT) is required.")
        print("Please set the GITHUB_TOKEN environment variable or pass it via --token.")
        sys.exit(1)

    repo = get_git_remote_repo()
    print(f"Target repository: {repo}")

    # Verify assets exist locally
    run5_path = PROJECT_ROOT / "run5_stride2_aug.zip"
    urfd_path = PROJECT_ROOT / "urfd_run3.zip"

    if not run5_path.exists():
        print(f"ERROR: Asset not found: {run5_path}")
        sys.exit(1)
    if not urfd_path.exists():
        print(f"ERROR: Asset not found: {urfd_path}")
        sys.exit(1)

    # Prepare release payload
    release_notes = (
        "This release bundles the pretrained model weights and pose estimator assets "
        "required to run the real-time fall detection demo out-of-the-box.\n\n"
        "### Release Contents:\n\n"
        "* `run5_stride2_aug.zip` : The best-performing single BiLSTM model weights (5-fold cross-validation, AUC 0.888).\n"
        "* `urfd_run3.zip` : The Run 3 BiLSTM weights (needed if you want to run the full two-model ensemble, AUC 0.897).\n\n"
        "### How to use:\n\n"
        "Cloners of this repository can download and extract these weights automatically to their correct directories by running:\n\n"
        "```bash\n"
        "python scripts/download_models.py\n"
        "```"
    )

    payload = {
        "tag_name": args.tag,
        "name": "Pretrained Model Weights for AI Fall Detection System",
        "body": release_notes,
        "draft": False,
        "prerelease": False
    }

    url = f"https://api.github.com/repos/{repo}/releases"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST"
    )

    print(f"Creating release {args.tag}...")
    try:
        with urllib.request.urlopen(req) as response:
            release = json.loads(response.read().decode())
            upload_url = release["upload_url"]
            html_url = release["html_url"]
            print(f"Created release successfully: {html_url}")
    except urllib.error.HTTPError as e:
        # Check if release already exists
        if e.code == 422:
            print(f"Release {args.tag} might already exist. Attempting to fetch it...")
            try:
                get_req = urllib.request.Request(
                    f"{url}/tags/{args.tag}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                with urllib.request.urlopen(get_req) as get_resp:
                    release = json.loads(get_resp.read().decode())
                    upload_url = release["upload_url"]
                    print(f"Found existing release: {release['html_url']}")
            except Exception as ex:
                print(f"ERROR fetching existing release: {ex}")
                sys.exit(1)
        else:
            print(f"ERROR creating release: {e.code} - {e.reason}")
            print(e.read().decode())
            sys.exit(1)

    # Upload the assets
    upload_asset(upload_url, run5_path, token)
    upload_asset(upload_url, urfd_path, token)

    print("\nAll assets uploaded successfully! Your GitHub Release is live.")


if __name__ == "__main__":
    main()
