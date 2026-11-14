"""
run_all.py  –  single entry point that regenerates all results.

Phases run in order:
  1. Verify checksum
  2. Build minute grid
  3. Build features + labels (saves to disk)
  4. Baselines + logistic regression, walk-forward on the train period
  5. Phase 5 experiments E5.1-E5.7 (LightGBM, context/intraday features, threshold)
  5b. Signal diagnostics and volatility-scaled experiments E5b.1-E5b.6
  5c. News diagnostics and experiments E5c.1-E5c.4 (news is downloaded and scored
      beforehand: src/download_news.py, then src/score_news.py in its own environment)
  5d. Training pooled over peer stocks, E5d.1-E5d.4 (peers from src/download_peers.py)
  5e. Fresh-years check of E5.6 and the final research round (history from src/download_history.py)
  5f. Move size: forecasts of the next 10 minutes' realized variance against HAR baselines
  7. Live indicator: freeze the models on training data, check live features equal training features
     (the app itself: uvicorn app.server:app)
  6. Final test (run once, on the frozen app code)

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
    ROOT / "src" / "phase5c_diagnostics.py",
    [ROOT / "src" / "phase5c_models.py", "--all"],
    ROOT / "src" / "phase5d_models.py",
    ROOT / "src" / "phase5d_checks.py",
    [ROOT / "src" / "phase5e_models.py", "--fresh", "--final"],
    ROOT / "src" / "phase5f_models.py",
    ROOT / "src" / "fit_live.py",
    ROOT / "src" / "check_live_parity.py",
    ROOT / "src" / "phase7c_lean_share.py",
    ROOT / "src" / "phase7d_band_by_time.py",
    ROOT / "src" / "make_log_table.py",
    # Phase 6 is not in this list: the final test was run once (report/phase6_run.json) and refuses to run again.
    # To reproduce it: python src/download_test.py, delete report/phase6_*.csv, then python src/phase6_final_test.py --final
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
