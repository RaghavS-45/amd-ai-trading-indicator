"""
run_all.py  –  single entry point that regenerates all results.

Phases run in order:
  1. Verify checksum
  2. Build minute grid
  3. Build features + labels (saves to disk)
  4. Train baselines
  5. Train LightGBM with walk-forward validation
  6. Calibrate, pick threshold, run final test eval
  7. Save all report figures to report/

Usage:
    python run_all.py

This script is the authoritative way to reproduce all results.
Do NOT run individual phase scripts in a different order.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
scripts = [
    ROOT / "src" / "verify_checksum.py",
    ROOT / "src" / "build_grid.py",
    # Phases 3-6 will be added here as they are built
]


def run(script: Path):
    print(f"\n{'='*60}")
    print(f"  Running: {script.name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script)], check=False)
    if result.returncode != 0:
        sys.exit(f"FAILED: {script.name}  (exit code {result.returncode})")


if __name__ == "__main__":
    for s in scripts:
        run(s)
    print("\n✓  All phases complete.")
