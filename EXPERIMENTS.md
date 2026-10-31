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

## Phase 3a – Per-session features and label tests (2026-10-31)

**Status:** ✅ Complete

**Problem found:** At 09:30 the median |1-min log return| was 0.0127 vs 0.00059 for
the rest of the day (about 20 times larger). The overnight gap leaked into return,
volatility, RSI, Bollinger and MACD features for the first ~30 minutes of each session.

**Change (`src/features.py`):**
- `compute_features()` now computes every feature per session (ET date). The first
  return of a session is NaN, and MACD EMAs use `min_periods`, so no window uses the
  previous day.
- **Bug fix:** `ret_L` was `log_ret.shift(L-1).rolling(L).sum()`, which is the return from
  t-(2L-1) to t-(L-1), not the last L bars as documented (ret_30 lagged by 29 bars).
  Now `ret_L = log(close_t / close_(t-L))`.
- **RSI fix:** RSI was NaN on 23 mid-session rows (14 bars with no down move: loss = 0).
  It is now 100 when there are only gains and 50 when prices are flat.

**Effect on the grid:**
| Metric | Value |
|--------|-------|
| Warm-up per session | 33 bars (features valid from 10:03 ET) |
| Rows with all features valid | 172,602 of 188,640 (91.5%) |
| Remaining NaNs after warm-up | 0 |

`drop_open_mins` (5) has no extra effect now because warm-up covers it. It stays as the
row-selection rule for training and live signals.

**Tests added (18 total, all pass):**
- `test_no_leakage.py`: changing all of day 1 does not change day 2 features; per-session
  warm-up is NaN; `ret_L` equals the last-L-bar log return (these three fail on the old code); RSI is
  100 with only gains and 50 when flat.
- `test_labels.py`: exact 10-minute horizon vs a position-based reference; filled target
  bars; lookup by timestamp (a dropped minute gives NA, not the next row); NA in the last
  10 minutes of each session, including a half-day (last label 12:49); prices after the
  horizon do not change the label; Int8 output.

---

<!-- Add new experiment entries below as phases progress -->
