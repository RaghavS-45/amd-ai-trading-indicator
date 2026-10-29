"""
verify_checksum.py
Verifies the SHA-256 checksum of the raw data file matches the stored value.
Run before any training to confirm the dataset hasn't changed.

Usage:
    python src/verify_checksum.py
"""

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def verify():
    data_path = ROOT / CFG["raw_data_path"]
    checksum_path = ROOT / CFG["checksum_path"]

    if not data_path.exists():
        sys.exit(f"ERROR: Data file not found: {data_path}")
    if not checksum_path.exists():
        sys.exit(f"ERROR: Checksum file not found: {checksum_path}")

    stored = checksum_path.read_text().strip()
    actual = hashlib.sha256(data_path.read_bytes()).hexdigest()

    if stored == actual:
        print(f"✓  Checksum OK  ({actual[:16]}...)")
    else:
        sys.exit(
            f"✗  CHECKSUM MISMATCH!\n"
            f"   Stored : {stored}\n"
            f"   Actual : {actual}\n"
            "   The raw data file has changed. Re-download or investigate."
        )


if __name__ == "__main__":
    verify()
