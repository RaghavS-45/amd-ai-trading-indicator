"""
run_all.py  –  single entry point that regenerates all results.

Phases run in order:
  1. Verify checksum
  2. Build minute grid
  3. Build features + labels (saves to disk)
  4. Baselines + logistic regression, walk-forward on the train period
  5. Phase 5 experiments E5.1-E5.7 (LightGBM, context/intraday features, threshold)
  5b. Signal diagnostics and volatility-scaled experiments E5b.1-E5b.6
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
    ROOT / "src" / "phase4_baselines.py",
    [ROOT / "src" / "phase5_models.py", "--all"],
    ROOT / "src" / "phase5b_diagnostics.py",
    [ROOT / "src" / "phase5b_models.py", "--all"],
    # Phase 6 will be added here when it is built
]


def run(entry):
    script, *args = entry if isinstance(entry, list) else [entry]
    print(f"\n{'='*60}")
    print(f"  Running: {script.name} {' '.join(args)}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(script), *args], check=False)
    if result.returncode != 0:
        sys.exit(f"FAILED: {script.name}  (exit code {result.returncode})")


if __name__ == "__main__":
    for s in scripts:
        run(s)
    print("\n✓  All phases complete.")
