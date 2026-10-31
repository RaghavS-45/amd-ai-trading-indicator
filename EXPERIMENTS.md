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

## Phase 2 – EDA and move_threshold (2026-10-31)

**Status:** ✅ Complete

**Rows used (`src/eda.py`):** every feature non-NaN, label exists, and
`sessions.tradable_mask`. 167,742 of 188,640 grid rows (88.9%), 486 days, 10:03–15:49 ET.
The script checks that its forward return matches `make_labels()` exactly.

**10-minute forward log return:** mean −0.012 bps, std 35.96 bps.
Percentiles: p1 −98.8, p5 −54.8, p25 −17.5, p50 0.0, p75 +17.6, p95 +54.2, p99 +97.7 bps.

**Cost:** round trip 2 × ($0.005 fee + $0.005 slippage) = $0.02/share = 1.00 bps at the
median close of $199.70 (0.30–2.61 bps over the period). Cost is small next to the
36 bps spread, so the threshold choice is about class balance.

**Class balance by threshold:**
| Threshold | Up | Down | Neutral |
|-----------|----|------|---------|
| 5 bps | 42.0% | 41.9% | 16.0% |
| 10 bps | 34.7% | 34.4% | 30.9% |
| 15 bps | 28.0% | 27.8% | 44.2% |
| 20 bps | 22.4% | 22.3% | 55.3% |
| 30 bps | 14.3% | 14.3% | 71.4% |

**Past vs forward 10-min return:** Pearson +0.0037, Spearman −0.0052. No simple
momentum or mean reversion; consistent with expecting modest accuracy.

**Train vs test (at 15 bps, before the change):**
| Period | Rows | Share > 0 | Std (bps) | Up | Down |
|--------|------|-----------|-----------|----|------|
| Train 2024-11-01 → 2026-06-30 | 142,064 | 49.9% | 35.85 | 28.0% | 27.7% |
| Test 2026-07-01 → 2026-10-15 | 25,678 | 49.2% | 36.60 | 28.0% | 28.3% |

At 10 bps: train Up 34.7% / Down 34.2%, test Up 34.6% / Down 35.2%.

**Filled bars by month:** under 1% for most of 2025, then 2–9% from 2025-12, highest
in the test months (2026-08 8.53%, 2026-10 8.82%). The test and live periods have more
flat, zero-volume minutes than most training data. Watch `vol_ratio` in validation.

**Decision:** `move_threshold` 0.0015 → **0.0010 (10 bps)**: near-equal thirds give the
model the most Up/Down examples, still ~10× the assumed cost.

**Figures:** `report/figures/forward_return_hist.png`, `report/figures/monthly_fill.png`.

---

## Phase 4 – Baselines and logistic regression, walk-forward (2026-10-31)

**Status:** ✅ Complete

**Setup (`src/phase4_baselines.py`):**
- Rows: `dataset.build_dataset(end_date=train_end)`: all features, label, `tradable_mask`;
  train period only (142,064 rows, 412 days). The hold-out test is never loaded.
- Walk-forward (`src/walkforward.py`): expanding window, 9 folds of 30 trading days,
  1 trading day left out between train and validation, first fold trains on 141 days.
  Validation covers 2025-06-03 → 2026-06-30 (270 days, 93,150 rows).
- Models: always-Up; RSI rule (Up if RSI < 30, Down if RSI > 70, else no signal);
  logistic regression (StandardScaler + LogisticRegression, multinomial, seed 42),
  fit per fold on train rows, target = `make_labels()`, predicted Neutral = no signal.
- Metrics (`src/evaluate.py`): direction accuracy (sign of the 10-min forward return;
  a zero return counts as wrong), strict accuracy (label equals the signal, i.e. the move
  beyond 10 bps that way), signals/day, and one non-overlapping 10-minute trade per signal
  (signals while a trade is open are skipped), 10 shares, round-trip cost $0.20/trade at
  1x and $0.60 at 3x. Entry at the signal bar's close is slightly optimistic; the
  slippage assumption is meant to cover it.
- SE: day-clustered (main). Signals on adjacent minutes share most of their 10-minute
  window, so the naive binomial SE is 2-3x too small (always-Up direction: 0.16% naive vs
  0.43% clustered).

**Pooled results (9 folds):**
| Model | Sig/day | Dir acc (±SE) | Strict acc (±SE) | Trades/day | Trade dir acc | P&L 1x | bps/trade 1x | P&L 3x | bps/trade 3x |
|-------|---------|---------------|------------------|------------|---------------|--------|--------------|--------|--------------|
| Always-Up | 345.0 | 50.29% (0.43) | 35.63% (0.47) | 34.8 | 49.94% | −$314 | −0.54 | −$4,072 | −2.45 |
| RSI rule | 81.8 | 50.51% (0.67) | 36.27% (0.62) | 17.1 | 51.01% | +$142 | −0.73 | −$1,706 | −2.63 |
| Logistic | 189.0 | 50.20% (0.43) | 39.47% (0.42) | 21.5 | 50.60% | −$225 | −0.27 | −$2,550 | −2.04 |

Base rate: 50.29% of validation rows have a positive forward return.
RSI P&L is positive in dollars but negative per trade in bps: its winning trades fell in
the later, higher-price folds, so dollar totals weight them more.

**Per fold (logistic):**
| Fold | Validation | Train days | Up share | Dir acc | Strict acc | P&L 1x | P&L 3x |
|------|------------|------------|----------|---------|------------|--------|--------|
| 1 | 2025-06-03 → 07-16 | 141 | 28% | 47.89% | 35.48% | −$60 | −$242 |
| 2 | 2025-07-17 → 08-27 | 171 | 44% | 51.88% | 40.36% | +$102 | −$98 |
| 3 | 2025-08-28 → 10-09 | 201 | 51% | 50.69% | 39.19% | −$26 | −$236 |
| 4 | 2025-10-10 → 11-20 | 231 | 58% | 48.67% | 39.04% | −$458 | −$785 |
| 5 | 2025-11-21 → 2026-01-06 | 261 | 49% | 55.03% | 42.26% | +$234 | +$38 |
| 6 | 2026-01-07 → 02-19 | 291 | 50% | 49.91% | 39.00% | −$137 | −$406 |
| 7 | 2026-02-20 → 04-02 | 321 | 39% | 49.33% | 37.90% | −$134 | −$406 |
| 8 | 2026-04-06 → 05-15 | 351 | 35% | 49.66% | 39.34% | −$493 | −$790 |
| 9 | 2026-05-18 → 06-30 | 381 | 52% | 50.28% | 41.47% | +$747 | +$377 |

Full per-fold tables for all three models: `report/phase4_results.csv`; pooled:
`report/phase4_summary.csv`.

**Conclusions:**
- No model shows direction skill: all three are within 1 SE of the 50.29% base rate.
- Logistic regression's higher strict accuracy (39.5% vs 35.6% for always-Up) comes with
  no direction skill (50.2%). It has learned *when* large moves are likely (volatility
  timing), not which way. Strict accuracy alone would overstate it.
- Nothing is profitable per trade, even at normal costs; folds 8–9 are positive for
  always-Up too, so they reflect the market's drift in that window, not model skill.
- Bar for Phase 5: direction accuracy clearly above ~50.3% (more than 2 clustered SEs,
  about 51.2%) with positive bps/trade at 3x costs.

---

<!-- Add new experiment entries below as phases progress -->
