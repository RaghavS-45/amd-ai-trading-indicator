"""
make_log_table.py  –  renders report/experiment_log.csv (one row per run, numbers copied from
EXPERIMENTS.md and the phase summary CSVs) as report/experiment_log.md for the report and slides.

Usage:
    python src/make_log_table.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
SRC = ROOT / "report" / "experiment_log.csv"
OUT = ROOT / "report" / "experiment_log.md"

HEADER = """# Experiment log

Every run of the project, including failures, one row each. Details and per-fold tables are in
EXPERIMENTS.md; numbers come from the `report/phase*_summary.csv` files. The hold-out test
(2026-07-01 onward) was not loaded by any run above P6. Seed 42 everywhere (`config.yaml`).

**Data splits**
- **WF9**: AMD 1-minute bars, 2024-11-01 → 2026-06-30 (142,064 rows, 412 days). Walk-forward:
  9 expanding folds of 30 trading days, 1 day left out before each, validation 2025-06-03 → 2026-06-30.
  From Phase 5b on, pass/fail is judged on folds 6–9 (2026-01-07 → 06-30); folds 1–5 were used for design.
- **POOL**: WF9 folds 6–9, but training rows from 10 chip stocks (1,417,350 rows); scored on AMD only.
- **WF28**: 2021 → 2026-06-30 (about 465,000 rows), 28 folds from 2023-02-21. "Fresh" = validation up
  to 2024-10-31, years that no earlier design choice had seen.

**Scoring**
- Direction runs: direction accuracy (± day-clustered SE) and mean profit per trade after costs in
  basis points (bps; 1 bps = 0.01%), one 10-minute trade at a time, $0.02/share round-trip cost.
  **IEX** = entry and exit at IEX closes. **Strict** = entry at the next minute's open, priced on SIP
  (the full market feed). Strict was adopted after Phase 5d showed IEX closes overstate results.
- **Bar** (pass rule, fixed before each phase): accuracy ≥ base rate + 2 SE **and** profit per trade > 0.
- Move-size runs (V0–V4): QLIKE loss on the next 10 minutes' realized variance (lower is better), R² on
  log variance, and how often the price stayed inside the 80% band. A model had to beat V2 by more than
  2 SE on both the fresh years and folds 6–9.
- **Runtime**: wall-clock time of the phase script on the project laptop (macOS, CPU only), for all runs of
  that phase together, excluding data downloads and FinBERT scoring. Phases 4–5d were re-run and timed on
  2026-11-08, and every result file came out identical. Phases 5e and 5f are timings noted at their first run.

"""

COLS = {"run_id": "Run", "date": "Date", "tried": "What was tried", "model_settings": "Model and settings",
        "split": "Split", "scoring": "Scoring", "result": "Result", "bar": "Bar", "outcome": "Outcome",
        "runtime": "Runtime"}


def main():
    log = pd.read_csv(SRC, dtype=str).fillna("")
    lines = [HEADER]
    for phase, rows in log.groupby("phase", sort=False):
        lines.append(f"## Phase {phase}\n")
        lines.append("| " + " | ".join(COLS.values()) + " |")
        lines.append("|" + "---|" * len(COLS))
        for _, r in rows.iterrows():
            lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in COLS) + " |")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"{len(log)} runs -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
