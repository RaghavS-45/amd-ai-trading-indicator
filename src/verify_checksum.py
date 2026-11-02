"""
verify_checksum.py
Verifies the SHA-256 checksums of the raw data files (asset, context symbols, news).
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


def verify(symbol: str):
    data_path = ROOT / CFG["raw_data_template"].format(sym=symbol.lower())
    checksum_path = ROOT / CFG["checksum_template"].format(sym=symbol.lower())

    if not data_path.exists():
        sys.exit(f"ERROR: Data file not found: {data_path}")
    if not checksum_path.exists():
        sys.exit(f"ERROR: Checksum file not found: {checksum_path}")

    stored = checksum_path.read_text().strip()
    actual = hashlib.sha256(data_path.read_bytes()).hexdigest()

    if stored == actual:
        print(f"✓  {symbol:<5} checksum OK  ({actual[:16]}...)")
    else:
        sys.exit(
            f"✗  {symbol} CHECKSUM MISMATCH!\n"
            f"   Stored : {stored}\n"
            f"   Actual : {actual}\n"
            "   The raw data file has changed. Re-download or investigate."
        )


def verify_news():
    for key in ("news_raw_path", "news_scored_path"):
        data_path = ROOT / CFG[key]
        stored = (ROOT / CFG["news_checksum_template"].format(name=data_path.stem)).read_text().strip()
        actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if stored != actual:
            sys.exit(f"✗  {data_path.name} CHECKSUM MISMATCH! Re-download or investigate.")
        print(f"✓  {data_path.stem} checksum OK  ({actual[:16]}...)")


if __name__ == "__main__":
    for sym in [CFG["asset"]] + CFG["context_symbols"]:
        verify(sym)
    verify_news()
