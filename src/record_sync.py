"""
record_sync.py  –  keeps a copy of the live record (signals_db_path) in a private Hugging Face
dataset, for hosts whose disk is wiped on restart (free cloud hosting). Off unless the
HF_RECORD_REPO environment variable is set (token from HF_TOKEN or a `huggingface-cli login`).

Each copy is a new commit in the dataset, so its history also shows when every signal was saved.
The server restores the copy before opening the store; until that succeeds nothing is uploaded,
so a failed download can never replace the saved record with an empty one.

Usage (once, to seed the dataset with the record made on this Mac):
    HF_RECORD_REPO=<user>/amd-indicator-record python src/record_sync.py --seed
"""

import argparse
import hashlib
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
NAME = "signals.db"


def repo():
    return os.environ.get("HF_RECORD_REPO") or None


def _token():
    return os.environ.get("HF_TOKEN") or None


def restore(path: Path) -> str:
    """Download the saved record to `path`. Returns 'restored', 'none saved yet' or 'off'."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    if not repo():
        return "off"
    if path.exists() and path.stat().st_size:
        return "kept the copy on this disk"   # process restart without a wiped disk: it is newer
    try:
        got = hf_hub_download(repo(), NAME, repo_type="dataset", token=_token(),
                              local_dir=tempfile.mkdtemp())
    except EntryNotFoundError:
        return "none saved yet"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(got, path)
    return "restored"


def restore_until_done(path: Path, status: dict) -> None:
    """Retry until the saved record is known (restored or none saved); never start empty by mistake."""
    wait = 5
    while True:
        try:
            result = restore(path)
            status.update(message=f"Saved record: {result}", error=None)
            return
        except Exception as e:
            status["error"] = f"Restoring the saved record failed, retrying: {type(e).__name__}: {e}"
            traceback.print_exc()
            time.sleep(wait)
            wait = min(wait * 2, 120)


def _snapshot(path: Path) -> Path:
    """A consistent copy of the database, safe while the app is writing to it."""
    out = Path(tempfile.mkdtemp()) / NAME
    src, dst = sqlite3.connect(str(path)), sqlite3.connect(str(out))
    with dst:
        src.backup(dst)
    src.close(), dst.close()
    return out


def upload(path: Path, last: str = "") -> str:
    """Upload a copy of the record if it changed since `last` (a sha256); returns the new sha256."""
    from huggingface_hub import HfApi
    copy = _snapshot(path)
    digest = hashlib.sha256(copy.read_bytes()).hexdigest()
    if digest != last:
        HfApi(token=_token()).upload_file(path_or_fileobj=str(copy), path_in_repo=NAME, repo_id=repo(),
                                          repo_type="dataset",
                                          commit_message=f"live record {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
    return digest


def start(path: Path) -> None:
    """Upload the record every record_sync_seconds in a background thread."""
    if not repo():
        return

    def loop():
        last = ""
        while True:
            time.sleep(CFG["record_sync_seconds"])
            try:
                if path.exists():
                    last = upload(path, last)
            except Exception:
                traceback.print_exc()   # keep the app running; try again next time

    threading.Thread(target=loop, daemon=True).start()


def seed(path: Path) -> None:
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(repo(), repo_type="dataset", private=True, exist_ok=True)
    if api.file_exists(repo(), NAME, repo_type="dataset"):
        raise SystemExit(f"{repo()} already has {NAME}; not overwriting it.")
    upload(path)
    print(f"Uploaded {path.relative_to(ROOT)} to dataset {repo()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="store_true", help="upload this machine's record if none is saved")
    args = ap.parse_args()
    if not repo():
        raise SystemExit("Set HF_RECORD_REPO, e.g. HF_RECORD_REPO=<user>/amd-indicator-record")
    if args.seed:
        seed(ROOT / CFG["signals_db_path"])
