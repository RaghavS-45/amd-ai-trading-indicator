# EXPERIMENTS.md
# Experiment Log

All experiments, including failures, are recorded here for the report.
Date format: YYYY-MM-DD.

---

## Phase 0 – Repo Setup (2026-10-29)

**Status:** ✅ Complete

**What was done:**
- Created repo structure: `data/`, `src/`, `app/`, `models/`, `notebooks/`, `report/`, `tests/`
- Wrote `config.yaml` (single source of truth for all dates, seeds, thresholds)
- Wrote `requirements.txt` with pinned versions
- Wrote `src/download_data.py`, `src/clean_data.py`, `src/features.py`, `src/labels.py`, `src/verify_checksum.py`
- Wrote `run_all.py` (master entry point)
- Wrote `tests/test_no_leakage.py` (automated leakage guard)

**Key decisions:**
- Bar timeframe starts at 1-minute; will review after Phase 1 coverage report
- Label horizon: 10 bars; threshold: 0.15% (15 bps) — larger than assumed round-trip cost of ~1 cent/share
- 3-class target (Up / Down / Neutral) to avoid trading on ambiguous moves
- Credentials only from environment variables

---

## Phase 1 – Data Download (2026-10-29)

**Status:** ⚠️ Superseded by Phase 1b. These numbers are from the old date range
(2023-01-03 to 2024-12-31) and are kept for the record.

**Results:**
| Metric | Value |
|--------|-------|
| Raw rows downloaded | 93,474 |
| Trading days | 251 |
| Expected bars (390/day) | 97,890 |
| Received bars (IEX) | 93,372 |
| **Coverage** | **95.4%** |
| SHA-256 | `5eb38157ebfa9e3ea7d17ee12431f4a5c9c6fbdb2a88eb614a783f83989ea6cc` |

**Decision:** Keep 1-minute bars. 95.4% coverage is excellent — far above the 60% threshold.
The ~4.6% gap (~4,518 missing bars) represents minutes with zero IEX trades, not data errors.
These will become NaN rows after feature computation (rolling windows); handled by `dropna()`.

**No action needed on bar timeframe.**

---

## Phase 1b – Re-download for 2024-11-01 → 2026-10-15 and minute grid (2026-10-31)

**Status:** ✅ Complete

**Why:** Dates moved to 2024-11-01 → 2026-10-15 in `config.yaml` (train to 2026-06-30,
hold-out test from 2026-07-01), so the data was re-downloaded and placed on a complete
minute grid.

**Results:**
| Metric | Value |
|--------|-------|
| Raw rows downloaded | 185,522 |
| Trading days with data | 487 |
| Received session bars (IEX, 09:30–16:00 ET) | 184,063 |
| **Coverage** (vs 390/day) | **96.9%** |
| Grid rows (after dropping days) | 188,640 |
| Filled (synthetic) bars | 4,634 (2.46%) |

**Grid method (`src/build_grid.py`):**
- One row per minute of every XNYS session (exchange-calendars), half-days included.
- A missing minute becomes a flat bar at the previous close with volume 0 and is
  flagged by `is_filled`.
- Days whose share of real bars is below `min_day_coverage` (0.5) are dropped.

**Dropped days:**
| Day | Real bars | Reason |
|-----|-----------|--------|
| 2024-12-23 | 53 / 390 | IEX bars stop at 10:22 ET; rest of the day missing from the feed |
| 2025-01-09 | 0 | Market closed (national day of mourning, President Carter); exchange-calendars 4.5.4 still lists it as a session |
| 2025-03-10 | 0 | No bars in the Alpaca IEX download (source data gap, not a market closure) |
| 2026-10-15 | 0 | Not downloaded: raw data ends 2026-10-14 because Alpaca treats `end="2026-10-15"` as midnight. **Open issue:** the last test day is missing |

**Decisions:**
- Removed `src/clean_data.py`. `build_grid.py` is the only data step, and `run_all.py`
  runs it after checksum verification.
- The first `drop_open_mins` (5) minutes of each session are excluded by row selection
  (`src/sessions.py::tradable_mask`) after features and labels are computed on the full
  grid. They are not deleted from the grid, so they still serve as feature warm-up.

---

<!-- Add new experiment entries below as phases progress -->
