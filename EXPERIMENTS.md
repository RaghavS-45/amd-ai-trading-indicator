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

## Phase 5 – LightGBM, context and intraday features, confidence threshold (2026-10-31)

**Status:** ❌ Run 2026-10-31: **no experiment passed the bar.** The experiment list below was
committed before any run (commit `cbd0025`) and is unchanged.

**Bar revised (2026-10-31):** the Phase 4 entry said "positive bps/trade at 3x costs". The
bar is now: (a) pooled direction accuracy >= base rate on the same rows + 2 x day-clustered
SE (about 51.2% on the Phase 4 rows) **and** (b) mean bps per trade > 0 at **normal (1x)**
cost. 3x cost is reported as a stress test only. The Phase 4 text is left unchanged.

**Fixed for every experiment:** the rows from `dataset.build_dataset(end_date=train_end)`
(142,064 rows, 412 days); the 9 walk-forward folds from Phase 4 (validation 2025-06-03 →
2026-06-30, 1-day gap); the metrics from `evaluate.evaluate_signals`. The hold-out test
(2026-07-01 → 2026-10-15) is not loaded.

**Feature sets:**
- *base*: the 11 current features.
- *context*: SPY, QQQ, NVDA 1-min bars (Alpaca IEX), same grid and flat-fill rules, joined
  by timestamp (bar t of each symbol, never t+1). Log returns over 1/5/10/30 bars per
  symbol, plus AMD − QQQ and AMD − NVDA 10-bar return.
- *intraday*: distance from session VWAP, distance from the 09:30 open, volume and
  trade-count surprise vs the previous 20 bars of the same session, minutes since open,
  minutes to close (XNYS calendar, so half-days are right).

**Experiments:**
| ID | Model | Features |
|----|-------|----------|
| E5.1 | LightGBM, config params (500 trees, lr 0.05, 31 leaves) | base |
| E5.2 | LightGBM, regularized (300 trees, lr 0.05, 15 leaves, min_child_samples 500, feature and bagging fraction 0.8, L2 1.0) | base |
| E5.3 | LightGBM, params of E5.1 or E5.2, whichever has the higher pooled 1x bps/trade | base + context |
| E5.4 | same params as E5.3 | base + intraday |
| E5.5 | same params as E5.3 | base + context + intraday |
| E5.6 | Logistic regression (control: features vs model) | base + context + intraday |
| E5.7 | Confidence threshold + reliability diagram on the model chosen below | as chosen |

**Rules decided in advance:**
- E5.1–E5.6: most likely class; Neutral = no signal. LightGBM is multiclass,
  deterministic, seed 42, fit per fold on training rows only, fixed tree count (no early
  stopping).
- E5.7 model: the one with the highest pooled 1x bps/trade among E5.1–E5.6. Choosing the
  best of six on the same validation data adds selection bias; the Phase 6 hold-out test is
  the real check.
- E5.7 signal: the more likely of Up/Down, only if its probability >= τ (grid 0.34–0.70,
  step 0.01). τ for fold k is the value with the best pooled 1x bps/trade on the
  out-of-sample predictions of folds 1…k−1, among values giving >= 5 signals/day, then
  applied to fold k. Reported pooled over folds 2–9, with τ and signals/day per fold. The
  τ curve over all folds is shown as an optimistic diagnostic only.
- Reliability diagram: pooled out-of-sample Up and Down probabilities, 10 bins, with counts,
  Brier score and expected calibration error. No calibrator is fitted in Phase 5.
- Every experiment is logged with pass/fail against the bar, including failures.
- Any direction accuracy above ~60% is treated as a leak until proven otherwise.

**Results:**

*Setup as registered:* 142,064 rows, 412 days, 9 folds, validation 2025-06-03 → 2026-06-30.
Context data: SPY 192,133 / QQQ 190,238 / NVDA 195,215 raw rows (coverage 98.6% / 94.9% /
99.3%); no AMD day had under 50% real context bars, so no rows changed and the AMD columns
of the grid are identical to before. Context bars filled: SPY 0.77%, QQQ 4.52%, NVDA 0.03%.

*Deviations and notes (honest record):*
- The base set has **12** features, not 11 as written above (5 returns, volatility,
  vol_ratio, RSI, MACD, Bollinger, 2 time-of-day). Nothing else changes.
- E5.3–E5.5 used the **config** params: E5.1 had higher pooled 1x bps/trade (−0.71 vs −1.57).
- E5.7 used **E5.6** (logistic, all features): the highest 1x bps/trade among E5.1–E5.6.
- `vol_surprise` reaches about −27 on filled bars (volume 0 against a normal recent
  baseline). This follows from the registered definition; not changed.
- Reliability plot: bins with under 50 rows are left off the figure (they distort it, e.g. a
  2-row bin at 1.0); all bins are kept in the printed table and the ECE.

*Pooled over all 9 folds (E5.7: folds 2–9, since τ for fold k needs folds < k):*
| ID | Features | Sig/day | Dir acc (±SE) | Bar | Strict acc | Trades/day | bps/trade 1x | bps/trade 3x | P&L 1x | P&L 3x | Pass |
|----|----------|---------|---------------|-----|------------|------------|--------------|--------------|--------|--------|------|
| E5.1 | LGBM config, base (12) | 236.7 | 50.22% (0.31) | 50.90% | 38.00% | 29.3 | −0.71 | −2.57 | −$606 | −$3,773 | ❌ |
| E5.2 | LGBM regularized, base (12) | 236.3 | 49.78% (0.37) | 51.03% | 37.97% | 27.9 | −1.57 | −3.42 | −$2,073 | −$5,085 | ❌ |
| E5.3 | LGBM, base + context (26) | 229.9 | 50.48% (0.34) | 50.97% | 38.47% | 28.3 | −1.20 | −3.05 | −$2,454 | −$5,513 | ❌ |
| E5.4 | LGBM, base + intraday (18) | 246.0 | 49.50% (0.36) | 51.00% | 37.14% | 29.7 | −1.51 | −3.38 | −$2,551 | −$5,755 | ❌ |
| E5.5 | LGBM, all sets (32) | 238.9 | 49.96% (0.37) | 51.04% | 37.88% | 28.9 | −0.72 | −2.58 | −$1,548 | −$4,666 | ❌ |
| E5.6 | Logistic, all sets (32) | 194.8 | 50.67% (0.40) | 51.09% | 39.70% | 22.4 | **+0.89** | −0.91 | +$1,865 | −$553 | ❌ (direction) |
| E5.7 | Threshold on E5.6 | 54.2 | 50.48% (0.79) | 51.75% | 43.24% | 7.8 | **+1.24** | −0.18 | +$1,276 | +$528 | ❌ (direction) |

Base rate (share of positive forward returns): 50.29% on all 9 folds, 50.17% on folds 2–9.
Bar = base rate + 2 × day-clustered SE of the experiment's direction accuracy.

*E5.6 per fold (logistic, all features):*
| Fold | Validation | Dir acc | bps/trade 1x | P&L 1x |
|------|------------|---------|--------------|--------|
| 1 | 2025-06-03 → 07-16 | 50.93% | −0.29 | −$17 |
| 2 | 2025-07-17 → 08-27 | 51.10% | +1.19 | +$119 |
| 3 | 2025-08-28 → 10-09 | 50.19% | −1.27 | −$113 |
| 4 | 2025-10-10 → 11-20 | 48.53% | −0.68 | −$136 |
| 5 | 2025-11-21 → 2026-01-06 | 55.14% | +4.90 | +$527 |
| 6 | 2026-01-07 → 02-19 | 50.96% | −0.53 | −$104 |
| 7 | 2026-02-20 → 04-02 | 51.52% | +2.16 | +$311 |
| 8 | 2026-04-06 → 05-15 | 48.90% | −0.81 | −$334 |
| 9 | 2026-05-18 → 06-30 | 50.80% | +3.36 | +$1,613 |

Without fold 9: direction 50.65% (±0.43) vs base 50.15%; +0.44 bps/trade; +$252.

*E5.7 per fold (τ chosen on earlier folds, min 5 signals/day):*
| Fold | τ | Sig/day | Dir acc | bps/trade 1x | P&L 1x |
|------|---|---------|---------|--------------|--------|
| 2 | 0.45 | 11.0 | 45.29% | −6.57 | −$87 |
| 3 | 0.44 | 22.8 | 50.29% | +3.16 | +$77 |
| 4 | 0.44 | 84.9 | 51.14% | +0.45 | +$40 |
| 5 | 0.44 | 23.6 | 56.21% | +5.22 | +$135 |
| 6 | 0.44 | 46.8 | 50.32% | +1.64 | +$52 |
| 7 | 0.44 | 24.6 | 45.18% | −3.56 | −$90 |
| 8 | 0.44 | 44.7 | 46.68% | −8.44 | −$720 |
| 9 | 0.39 | 174.9 | 51.51% | +5.57 | +$1,870 |

Without fold 9: direction 49.79% (±0.97), −1.07 bps/trade, −$594, 36.9 signals/day.
Every fold met the 5 signals/day minimum.

*Reliability (E5.6, pooled out-of-sample):* ECE 0.013 (Up), 0.007 (Down). Well calibrated
where almost all predictions sit (0.2–0.5). Above 0.5 the model is **overconfident**:
predicted ~0.53 → observed ~0.43 for both classes (about 1,100 rows each). Figure:
`report/figures/reliability_phase5.png`.

*Optimistic τ curve (E5.6, all folds, hindsight):* direction accuracy stays at 50.6–50.8%
from τ 0.34 to 0.43 and **falls** below 50% above τ 0.45. Higher confidence does not mean
better direction. The large bps/trade at τ ≥ 0.54 come from fewer than 3 signals/day
with direction accuracy below 50% and SEs of 3–15 points: noise. Full curve:
`report/phase5_E5.7_tau_curve.csv`.

**Conclusions:**
- No model, feature set or threshold shows direction skill: every direction accuracy is
  within about 1 SE of the base rate. LightGBM is no better than logistic regression and
  slightly worse on these features.
- Context (SPY/QQQ/NVDA) and intraday features did not help LightGBM (E5.3–E5.5 vs E5.1).
- The only positive 1x results (E5.6, E5.7) depend on fold 9. Without it, E5.6 is barely
  positive and E5.7 is negative. They also come from choosing the best of six on the same
  validation data. They do not meet the bar, and they should not be read as evidence of an edge.
- A confidence threshold cannot create skill the probabilities do not have. Calibration is
  good in the bulk and overconfident in the tail, so a high τ selects worse signals.
- Files: `report/phase5_summary.csv`, `report/phase5_E5.x_folds.csv`,
  `report/phase5_E5.7_tau_curve.csv`, `report/phase5_run.log`; out-of-sample predictions
  `report/phase5_E5.x_oos.parquet` (gitignored, regenerated by `run_all.py`).

---

## Phase 5b – Volatility-scaled features and score signals (2026-11-02)

**Status:** ❌ Run 2026-11-02: **no experiment passed the bar.** The list below was committed
before any run (commit `402bf0f`) and is unchanged.

**Why:** no Phase 5 model passed. Diagnostics were run only on data through 2026-01-06
(end of fold 5), so folds 6-9 stay unused for design. Script: `src/phase5b_diagnostics.py`,
output `report/phase5b_ic.csv`, `report/phase5b_e56_deciles.csv`.

**Diagnostics (design data, 100,424 rows):**
- *A mistake caught:* the first version averaged within-day rank correlations and showed
  `open_dist` at −0.32, negative on 99% of days. Used as a rule, the same feature gets about
  49%. This is a known bias (Nickell 1981): within a day, a price level and its later
  returns are mechanically related. Replaced by pooled correlation with day-clustered SEs,
  which gives `open_dist` −0.005 (t −0.5). The biased column is kept in the CSV as a warning.
- Pooled: 4 of 48 candidates reach |t| > 2, against about 2.4 expected by chance (the script
  now also lists the scaled `norm` versions, which are near-copies: 5 of 57). Largest:
  AMD 1-minute reversal `ret_1` −0.011 (t −3.4), `nvda_ret_1` +0.009 (t 2.8). Both are small.
  The reversal may be bid-ask bounce (Roll 1984), which cannot be earned; hence the delay
  test below.
- Rejected, no evidence: overnight gap, previous-day return, 60/120-minute returns, signed
  volume, QQQ-residual returns, share of filled bars.
- E5.6 by score decile (folds 1-5): not monotonic. The strongest Up decile earned −0.91 bps
  and deciles 6-8 earned +0.19 to +0.91, while the strongest Down decile was right (−2.35 bps,
so the problem is one-sided). This matches Phase 5's overconfident tail. Likely
  cause (a hypothesis): unscaled returns and outliers such as `vol_surprise` −27 drive the
  extreme scores.
- At AMD's median price ($139) the 1x round-trip cost is about 1.4 bps; the mean absolute
  10-minute move is 22.9 bps.

**Hypothesis:** scaling and clipping the inputs, and predicting a continuous scaled return,
makes score strength meaningful, so the strongest scores can pay for costs.

**Fixed:** same rows, folds, costs and `evaluate_signals` as Phase 5. Hold-out test not loaded.

**New `norm` feature set (15):** `z_ret_{1,5,10,30}`, `z_qqq_ret_{1,10}`, `z_nvda_ret_{1,10}`,
`z_vwap`, `z_open` (scaled by the 20-bar volatility x sqrt(bars), clipped at ±4),
`vol_surprise_clip`, `clv_10`, `log_vol`, `norm_tod_sin`, `norm_tod_cos`.

**Target:** `fwd_ret / (volatility x sqrt(10))`, clipped at ±4 (E5b.3: sign of `fwd_ret`,
zero returns dropped).

**Experiments:**
| ID | Model | Signal |
|----|-------|--------|
| E5b.1 | Ridge (alpha 1000), expanding window | sign of every score |
| E5b.2 | Ridge, expanding | strongest 20% of scores |
| E5b.3 | Logistic on sign, expanding | strongest 20% |
| E5b.4 | LightGBM regression (200 trees, lr 0.02, 7 leaves, min_child_samples 2000, feature and bagging fraction 0.7, L2 10) | strongest 20% |
| E5b.5 | Ridge, rolling last 120 training days | strongest 20% |
| E5b.6 | Ridge + LightGBM, scores averaged after scaling | strongest 20% |

**Rules decided in advance:**
- No hyperparameter search. Values above are fixed.
- The 20% cutoff for a fold comes from an inner holdout (last 30 training days; fit on the
  rest, take the |score| quantile). The model is then refit on all training days.
- **Pass:** the unchanged Phase 5 bar (direction accuracy >= base rate + 2 x day-clustered
  SE, and mean 1x bps/trade > 0), judged on **confirmation folds 6-9** only. E5.6 is the
  control on the same rows. Design folds and all folds are reported too.
- Also reported: folds 6-8 alone (fold 9 drove Phase 5's positive result), deciles of the
  score on folds 6-9, and 1x bps/trade with entry one minute late (close t+1). If the edge
  disappears with that delay, it is not tradable and the model does not count as improved,
  even if it passes.
- Folds 6-9 were visible in Phase 5's aggregate tables, so they are not fully virgin; this
  round's design used no analysis of them. With 13 experiments over Phases 5 and 5b, one
  pass could be luck. The hold-out test is the real check.
- If none passes, this is logged as a failure and the model choice for Phase 6 goes back to
  the user. The bar is not changed.
- Any direction accuracy above ~60% is treated as a leak until proven otherwise.

**Results:**

*Confirmation folds 6-9 (2026-01-07 → 06-30), the folds that decide pass/fail:*
| ID | Sig/day | Dir acc (±SE) | Bar | bps/trade 1x | 1 min late | Folds 6-8 only | bps/trade 3x | P&L 1x | Pass |
|----|---------|---------------|-----|--------------|------------|----------------|--------------|--------|------|
| E5.6 (control) | 226.5 | 50.55% (0.57) | 51.72% | +1.19 | +0.84 | +0.26 | −0.22 | +$1,485 | ❌ |
| E5b.1 Ridge, every score | 347.0 | 50.25% (0.38) | 51.34% | +0.02 | −0.54 | −0.09 | −1.45 | −$262 | ❌ |
| E5b.2 Ridge, top 20% | 65.9 | 50.27% (0.95) | 52.47% | −0.91 | −2.39 | −1.26 | −2.39 | −$850 | ❌ |
| E5b.3 Logistic, top 20% | 63.7 | 50.41% (0.91) | 52.39% | −0.11 | −1.21 | −0.91 | −1.56 | +$176 | ❌ |
| E5b.4 LightGBM, top 20% | 60.3 | 51.19% (1.23) | 53.03% | −1.43 | −2.36 | −1.93 | −2.91 | −$684 | ❌ |
| E5b.5 Ridge rolling, top 20% | 51.1 | 48.56% (1.04) | 52.66% | −0.69 | −2.14 | −1.62 | −2.09 | −$358 | ❌ |
| E5b.6 Ridge + LightGBM, top 20% | 63.4 | 51.36% (1.05) | 52.69% | +0.28 | −1.09 | −0.51 | −1.19 | +$336 | ❌ |

*All 9 folds, for comparison with Phase 5:* E5.6 50.67%, +0.89; E5b.1 49.85%, −0.59; E5b.2 50.06%,
−0.44; E5b.3 50.18%, −0.43; E5b.4 50.00%, −1.04; E5b.5 48.91%, −0.95; E5b.6 50.33%, +0.07.
Full table: `report/phase5b_summary.csv`; per fold: `report/phase5b_E5b.x_folds.csv`; deciles:
`report/phase5b_deciles.csv`; log: `report/phase5b_run.log`.

**Conclusions:**
- The hypothesis held only in part. Scores became closer to monotonic on folds 6-9 (E5b.2:
  about −0.1 bps in the lowest decile, about +2 bps in the top two), so strength now means
  something. But that spread is about 2 bps around a positive drift, and it does not pay
  for a two-sided strategy at about 1.4 bps cost. Most of the "Down" deciles still rose.
- Nothing beat the control. Only E5b.6 had a positive 1x result, and it turned negative with
  a one-minute-late entry and without fold 9.
- The delay test matters for the control too: E5.6 keeps +0.84 bps when entry is one minute
  late, but is −0.15 on folds 6-8 (`confirm_ex_last`). Its edge still rests on fold 9.
- 13 experiments across Phases 5 and 5b show no direction skill on one-minute IEX bars and
  this kind of feature. More tuning on the same inputs would only fit noise, so it should
  stop here. Any further attempt needs information the model does not have now.

---

## Phase 5c – News features (2026-11-02)

**Status:** ❌ Run 2026-11-02: **no experiment passed the bar.** The list below was committed
before any run on folds 6-9 (commit `7a4a7d2`) and is unchanged.

**Why:** suggested by the instructor. News is the first new information since Phase 5, and it
is free both historically and live (Alpaca News API, articles from Benzinga).

**Data:** `src/download_news.py` → 2,369 AMD-tagged articles, 2024-11-01 → 2026-06-30 (test
period not downloaded). `src/score_news.py`: FinBERT `ProsusAI/finbert`, pinned revision, on
headlines only (about 7 ms each on CPU), in its own environment (`requirements-news.txt`).

**No look-ahead:** only headlines naming AMD count. An article first affects the bar after the
one it arrived in. If it was edited over 60 s later (4.2% of design articles), it counts as
arriving when edited, since the stored headline may be the edited version. Tested in
`tests/test_news.py`. Limit: Alpaca publishes no news latency figure, so live delivery may be
slower than one bar; the one-minute-late result below tests this only partly.

**Diagnostics (design data through 2026-01-06, `src/phase5c_diagnostics.py`,
`report/phase5c_events.csv`):**
| Articles | Events | Days | Up rate (±SE) | Mean 10-min return | Mean size |
|----------|--------|------|---------------|--------------------|-----------|
| All design rows | 100,424 rows | | 49.6% | | 22.9 bps |
| Named | 332 | 169 | 39.8% (2.8) | −6.2 bps | 30.7 bps |
| Named, at most 2 symbols | 242 | 140 | 39.3% (3.2) | −8.2 bps | 31.2 bps |
| Named, FinBERT positive | 220 | 121 | 37.7% (3.6) | −9.3 bps | 32.1 bps |
| Named, FinBERT negative | 112 | 88 | 43.8% (5.1) | −0.2 bps | 28.0 bps |

- Following sentiment lost money (accuracy about 43%). Positive headlines were followed by
  *falls*. A hand-made word list gave the same picture and is not used further.
- News predicts size of move better than direction: `news_count_30` vs |return| IC 0.045
  (t 3.7), vs direction −0.017 (t −1.7). |1-minute return| is already high 15 minutes before
  articles, so articles partly follow moves instead of causing them.
- **Warning:** the fall after news was found while scanning about 30 subset/threshold/delay
  combinations. Its explanation (attention peak, then reversal) came after seeing it. It may
  be chance. Coverage is also low: about 1.2 named in-session articles per trading day.

**Experiments (same rows, folds, costs, metrics as Phase 5b):**
| ID | Model | Signal |
|----|-------|--------|
| E5c.1 | E5.6 logistic + `news_sent_10`, `news_sent_30`, `news_count_30` | most likely class |
| E5c.2 | Rule, no training | Down on the first bar a named article counts, any sentiment |
| E5c.3 | Rule, no training | opposite of the sentiment's sign on that bar |
| E5c.4 | E5.6 predictions | E5.6, but Up signals are dropped when a named article arrived in the last 30 bars |

**Rules decided in advance:**
- **Pass:** the unchanged bar on **folds 6-9** (direction accuracy >= base rate + 2 x
  day-clustered SE, and mean 1x bps/trade > 0). E5.6 is the control. No parameter is tuned.
- E5c.2 and E5c.3 fire about once a day, so their SE is large and they may fail even if the
  effect is real. That is accepted; the bar is not loosened for it. Their up rate and
  event count on folds 6-9 are reported either way.
- Also reported: all folds, design folds, folds 6-8 alone, and entry one minute late. A
  positive result that disappears when one minute late does not count as an improvement.
- 17 experiments in total after this round. If none passes, this is logged as a failure and
  E5.6 goes to the user as the Phase 6 choice, with no further round on these inputs.
- Any direction accuracy above ~60% on a large sample is treated as a leak until proven
  otherwise. E5c.2/E5c.3 have few events, so there it means checking the alignment first.

**Results:**

*Confirmation folds 6-9 (2026-01-07 → 06-30), the folds that decide pass/fail:*
| ID | Sig/day | Dir acc (±SE) | Bar | Trades | bps/trade 1x | 1 min late | Folds 6-8 only | P&L 1x | Pass |
|----|---------|---------------|-----|--------|--------------|------------|----------------|--------|------|
| E5.6 (control) | 226.5 | 50.55% (0.57) | 51.72% | 3,096 | +1.19 | +0.84 | +0.26 | +$1,485 | ❌ |
| E5c.1 E5.6 + news features | 226.6 | 50.60% (0.58) | 51.73% | 3,094 | +0.72 | +0.21 | +0.01 | +$1,011 | ❌ |
| E5c.2 Down after news | 0.86 | 41.75% (5.07) | 60.71% | 99 | −6.77 | −3.25 | −1.14 | −$379 | ❌ |
| E5c.3 Against sentiment | 0.86 | 42.72% (5.01) | 60.60% | 99 | −8.25 | −5.67 | −4.18 | −$373 | ❌ |
| E5c.4 E5.6, no Up after news | 217.9 | 50.47% (0.60) | 51.77% | 3,048 | +0.78 | +0.42 | +0.37 | +$925 | ❌ |

*Fall after news, by fold (first bar of named articles):* up rate 37.5%, 34.2%, 51.4%, 38.3%,
46.2% on folds 1-5 and 57.1%, 41.2%, 55.8%, 72.7% on folds 6-9. Folds 6-9 pooled: **57.3%
(±5.0)**, 103 events, +3.98 bps, against 39.8% (±2.8) in the design data.

Files: `report/phase5c_summary.csv`, `report/phase5c_E5c.1_folds.csv`, `report/phase5c_run.log`.

*Not pre-registered, a check of a different question (does news predict the size of the move?):*
`news_count_30` vs |10-minute return| IC 0.052 (t 3.0) on folds 1-5 and 0.056 (t 3.1) on
folds 6-9. Comparing rows within the same day, a named article in the last 30 bars means a move
2.35 bps (±1.36) larger on folds 1-5 and 5.65 bps (±1.77) larger on folds 6-9.

**Conclusions:**
- News did not improve direction. Adding news features made the control worse, and removing
  Up signals after news also lowered bps/trade.
- The fall after news did not hold: it reversed on the folds not used to find it. It was a
  chance result from the scan. E5c.2 passes on the design folds (58.76%, +3.98 bps), which
  shows why those folds cannot be the judge.
- Sentiment shows no use for direction on any folds, at this coverage and with headlines
  only. This does not rule out news in general: article text and other sources were not tried.
- News does seem to help with the size of moves, which held on later folds. This is
  volatility information, not direction, and it was a post-hoc check. It could be useful as a
  chart warning ("news in the last 30 minutes: expect larger moves"), not as a signal.
- After 17 experiments E5.6 remains the best model, and has not passed the bar.

---

## Phase 5d – Training pooled over peer stocks (2026-11-02)

**Status:** ⚠️ Run 2026-11-02: **E5d.3 and E5d.4 clear the bar on folds 6-9 but are rejected.**
They fail the pre-registered one-minute-late rule, and the follow-up checks show the edge is
not tradable. The list below was committed before any model ran on the pool (commit `7cea4d0`)
and is unchanged.

**Why:** every model so far looked for a very small signal in one stock's history. Pooling
similar stocks gives about 10 times more examples for estimating it. The scaled `norm`
features make stocks comparable. Live use changes nothing: the model still predicts AMD from
AMD, QQQ and NVDA bars, all free on IEX.

**Data:** free IEX 1-minute bars, 2024-11-01 → 2026-06-30, for NVDA, AVGO, MU, INTC, QCOM, MRVL,
TSM, TXN, LRCX, AMAT (liquid large US-listed semiconductor names, chosen before any
result). `src/download_peers.py`; NVDA's existing raw file is cut at the same date in
`peers.peer_grid`. A peer with under 80% real bars on the grid is left out. Same flat-fill and
sparse-day rules as AMD, on AMD's sessions. A peer row uses the QQQ and NVDA bars of its own
timestamp. No diagnostics were run on peers, so there is no design finding to confirm.

**Experiments (same models and fixed parameters as Phase 5b; only the training rows change):**
| ID | Model | Signal | Same model on AMD alone |
|----|-------|--------|-------------------------|
| E5d.1 | Ridge | sign of every score | E5b.1 |
| E5d.2 | Ridge | strongest 20% | E5b.2 |
| E5d.3 | LightGBM regression | strongest 20% | E5b.4 |
| E5d.4 | Ridge + LightGBM | strongest 20% | E5b.6 |

**Rules decided in advance:**
- Training and the inner holdout's model fit use all symbols; the score cutoff and every
  validation number use **AMD rows only**. Same days, gap, costs and metrics as Phase 5b.
- **Pass:** the unchanged bar on **folds 6-9**. E5.6 is the control. Also reported: all folds,
  design folds, folds 6-8, and entry one minute late. A positive result that disappears when
  one minute late does not count as an improvement.
- No parameter is changed for the larger data, including ridge alpha and LightGBM's leaf size,
  so the only change is the data. A failure here says nothing about other parameters, and they
  are not tried afterwards.
- Stocks move together, so the pool holds fewer independent examples than its row count. This
  affects training only; the SEs come from AMD rows clustered by day.
- 21 experiments in total after this round. If none passes, it is logged as a failure and E5.6
  goes to the user as the Phase 6 choice. The bar is not changed.
- Any direction accuracy above ~60% is treated as a leak until proven otherwise.

**Results:**

*Setup:* TXN was left out by the 80% rule (78.4% real bars). Pool 1,417,350 rows, 10 symbols
including AMD; AMD rows unchanged at 142,064.

*Confirmation folds 6-9, the folds that decide the bar:*
| ID | Sig/day | Dir acc (±SE) | Bar | Bar met | bps/trade 1x | 1 min late | Folds 6-8 only | bps/trade 3x | P&L 1x |
|----|---------|---------------|-----|---------|--------------|------------|----------------|--------------|--------|
| E5.6 (control) | 226.5 | 50.55% (0.57) | 51.72% | ❌ | +1.19 | +0.84 | +0.26 | −0.22 | +$1,485 |
| E5d.1 Ridge, every score | 347.0 | 50.66% (0.30) | 51.18% | ❌ | +0.92 | −0.08 | +0.87 | −0.54 | +$1,379 |
| E5d.2 Ridge, top 20% | 67.1 | 51.74% (0.71) | 52.00% | ❌ | +0.67 | −1.54 | −0.37 | −0.80 | +$1,022 |
| E5d.3 LightGBM, top 20% | 63.5 | **53.33% (0.92)** | 52.42% | ✅ | +1.09 | **−1.05** | +0.09 | −0.38 | +$984 |
| E5d.4 Ridge + LightGBM, top 20% | 65.9 | **52.56% (0.77)** | 52.13% | ✅ | +1.76 | **−0.85** | +1.32 | +0.30 | +$1,589 |

All 9 folds: E5d.3 51.51% against a bar of 51.55%; E5d.4 51.23% against 51.38%. Design folds 1-5:
E5d.3 50.06%, −0.47 bps; E5d.4 50.18%, −0.68 bps. Compared with the same models on AMD alone
(E5b.4 51.19%, E5b.6 51.36% on folds 6-9), pooling did raise IEX-measured accuracy.

*Rejected under the pre-registered rule:* "A positive result that disappears when one minute
late does not count as an improvement." Both turn negative.

*Follow-up checks, not pre-registered (`src/phase5d_checks.py`, `report/phase5d_checks.csv`):*
the same signals on folds 6-9, re-priced. SIP is the full-market feed, used only for checking.
| Model | Dir acc on IEX | Dir acc on SIP (±SE) | bps: close t (as run) | bps: IEX open t+1 | bps: SIP close t | bps: SIP open t+1 |
|-------|----------------|----------------------|-----------------------|-------------------|------------------|-------------------|
| E5.6 | 50.55% | 50.05% (0.59) | +1.19 | +0.93 | +0.85 | +0.88 |
| E5d.3 | 53.33% | 51.20% (0.92) | +1.11 | −1.16 | −1.38 | −1.37 |
| E5d.4 | 52.56% | 50.03% (0.74) | +1.78 | −0.63 | −0.68 | −0.69 |

SIP up rate on the same rows: 50.50%. The small bps differences from the table above (at most
0.02) come from this script using log returns instead of the simulator's price P&L.

**Conclusions:**
- The passes are an artifact of IEX prices, not a tradable edge. IEX's last print in a minute can
  differ from the market price (mean 2.3 bps, Phase 5b sample). The pooled models seem to learn
  when it is off, and it corrects within about a minute. That earns "accuracy" on IEX closes, but
  entering at the next open or scoring on SIP removes it. Entry at the signal bar's own IEX close
  is not available in live trading, since that minute has already ended.
- This explanation fits the numbers but was not tested directly, for example by ablating
  `z_ret_1`. Whatever the mechanism, the tradability result stands.
- **Lesson for Phases 6-8:** IEX-close evaluation can overstate results. Phase 6 and the live
  scorecard must also report entry at the next bar's open, and SIP-scored direction where the
  data is old enough. The earlier 1-minute-late rule existed for this and did its job.
- E5.6 survives these checks on bps (+0.93 at next open, +0.85 on SIP), but not on direction
  (50.05% on SIP, up rate 50.50%), and falls to about zero without fold 9. It has a small
  possible cost-adjusted edge and no direction skill shown.
- After 21 experiments, no model has a tradable direction edge on these free inputs.

---

## Phase 5e – Fresh years and the final research round (2026-11-02)

**Status:** ❌ Run 2026-11-02: **E5.6 failed on the fresh years, and no final-round model passed
on any period.** The list below was committed before any model ran on these years or feature
sets (commit `0c868c6`) and is unchanged. Approved by the user as the last batch before Phase 6.

**Why:** folds 6-9 have now judged 21 experiments, so they are no longer a clean check, and Phase
5d showed IEX closes can overstate results. Data before 2024-11 has not influenced any design
choice, so it gives an independent check without using the hold-out test.

**Data:** free 1-minute IEX bars 2021-01-04 → 2024-10-31 for AMD, SPY, QQQ, NVDA, AVGO, TSM, and
AMD SIP bars (`src/download_history.py`, separate files with checksums), joined to the existing
data to 2026-06-30 (`src/long_history.py`). Same flat-fill, sparse-day and warm-up rules. Test
period rows are cut before anything else, and this is tested. SIP is used for targets and scoring
only; every feature uses IEX prices, as it will live.

**Strict scoring, for this phase and the Phase 6 choice (`src/tradable.py`):** entry at the next
bar's open, exit at the close of t+10, priced on SIP, non-overlapping trades.
**Pass:** direction accuracy on these returns >= their base rate on the same rows + 2 x day-clustered
SE, **and** mean 1x bps/trade > 0. The 1x cost is unchanged. This replaces the IEX-close version
of the bar because of Phase 5d; it is stricter, and it is fixed before any result.

**Step 1 – E5.6, unchanged, on fresh years:** the same logistic model, features, IEX 3-class
labels and most-likely-class signal. Trained on data from 2021 only up to each fold, with the
walk-forward from `history_min_train_days` (500 days) to 2024-10-31. No row after 2024-10-31 is
used. This asks one question: was its small edge real, or mostly fold 9? Reported on strict
scoring and on IEX closes, also split into the first and second half of the folds.

**Step 2 – final round:** ridge + LightGBM (fixed Phase 5b parameters, scores averaged after
scaling), strongest 20% of scores (cutoff from an inner holdout), trained from 2021 on. The target
is the scaled tradable return, not the IEX close. One continuous walk-forward with 30-day blocks
ending 2026-06-30; the last blocks fall on the same dates as folds 6-9.

| ID | Features | Idea and source |
|----|----------|-----------------|
| E5e.1 | `norm` | Baseline: tradable target and longer history alone |
| E5e.2 | `norm` + `z_seas` | Same half hour's mean return over the previous 40 sessions (Heston et al. 2010) |
| E5e.3 | `norm` + scaled AVGO, TSM (1, 10 bars) and NVDA/AVGO/TSM basket returns (1, 5 bars) | Larger chip stocks leading AMD (Chinco et al. 2019) |
| E5e.4 | `norm` + AMD and QQQ first half-hour return from the previous close, last half hour only | Intraday momentum (Gao et al. 2018) |

**Rules decided in advance:**
- **A model qualifies only if it passes strict scoring on BOTH fresh years (validation through
  2024-10-31) and folds 6-9.** Also reported: 2024-11 → 2026-01, and folds 6-8 alone.
- No parameter tuning, and no further variants of these ideas after seeing results.
- E5e.1 is compared with E5e.2-4 on the same rows, so a difference comes from the features.
- **Choice for Phase 6:** a qualifying model with the highest fresh-years bps; if none, E5.6 stays
  the choice if its strict bps/trade is positive on the fresh years; if not, the result goes to the
  user: there would be no evidence of any edge, and the indicator would be reported that way.
- With 25 models run in total, a single pass is still weak evidence. The Deflated Sharpe Ratio
  (Bailey & López de Prado 2014) goes in the report as a further adjustment, not a pass rule.
- Research stops after this round, whatever the outcome. Then Phase 6.
- Any direction accuracy above ~60% is treated as a leak until proven otherwise.

**Results:**

*Setup:* 465,244 rows, 1,346 days, 2021-02-17 → 2026-06-30 (the first weeks are seasonal warm-up;
E5.6 loses them too). AMD filled 2.24%. Filled bars: SPY 1.46%, QQQ 7.81%, NVDA 4.09%, TSM 9.04%,
**AVGO 32.99%**, which is thin in the older years; this weakens E5e.3's AVGO inputs but no rule
set in advance excluded it. Final round: 28 folds, validation 2023-02-21 → 2026-06-30; folds
25-28 are the 2026-01-07 → 06-30 period of folds 6-9.

*Step 1: E5.6, unchanged, fresh years 2023-03-03 → 2024-10-31 (14 folds, 420 days):*
| Scoring | Sig/day | Dir acc (±SE) | Base rate | Bar | bps/trade 1x | bps/trade 3x | Pass |
|---------|---------|---------------|-----------|-----|--------------|--------------|------|
| Strict (SIP, next open) | 206.4 | 50.35% (0.30) | 51.04% | 51.64% | −1.32 | −4.46 | ❌ |
| IEX closes | 206.4 | 50.73% (0.30) | 50.91% | 51.51% | −0.92 | −4.06 | ❌ |
| Strict, first half of folds | 205.2 | 50.15% (0.44) | 51.09% | 51.98% | −1.49 | −5.28 | ❌ |
| Strict, second half of folds | 207.5 | 50.55% (0.41) | 50.99% | 51.81% | −1.15 | −3.64 | ❌ |

*Step 2: final round, strict scoring:*
| ID | Period | Sig/day | Dir acc (±SE) | Base rate | Bar | bps/trade 1x | Pass |
|----|--------|---------|---------------|-----------|-----|--------------|------|
| E5e.1 baseline | Fresh | 67.0 | 50.33% (0.64) | 51.03% | 52.31% | −1.47 | ❌ |
| | Folds 6-9 | 63.6 | 48.13% (1.06) | 50.62% | 52.75% | −2.75 | ❌ |
| E5e.2 same half hour | Fresh | 67.3 | 50.16% (0.65) | 51.03% | 52.32% | −1.20 | ❌ |
| | Folds 6-9 | 67.0 | 48.48% (1.06) | 50.62% | 52.75% | −2.54 | ❌ |
| E5e.3 chip stocks | Fresh | 66.5 | 49.89% (0.63) | 51.03% | 52.30% | −1.40 | ❌ |
| | Folds 6-9 | 63.0 | 48.52% (1.08) | 50.62% | 52.79% | −1.48 | ❌ |
| E5e.4 last half hour | Fresh | 66.8 | 50.42% (0.64) | 51.03% | 52.31% | −1.50 | ❌ |
| | Folds 6-9 | 66.4 | 48.15% (1.12) | 50.62% | 52.87% | −1.89 | ❌ |

Across all 28 folds the correlation between score and tradable return is −0.0003 to −0.0022, and
signals average about −0.1 bps before costs. At most 4 of 28 folds had positive 1x bps/trade.
Files: `report/phase5e_summary.csv`, `report/phase5e_fresh.csv`, `report/phase5e_fresh_run.log`,
`report/phase5e_final_run.log`.

**Conclusions:**
- E5.6's edge was not real. On years it had not seen, it is below the base rate and loses money,
  so the positive numbers on the later folds were mostly chance, as fold 9 suggested. Before costs it
  is about +0.5 bps/trade at about 1.8 bps cost; that gross figure is approximate and post hoc.
- None of the three literature ideas added anything measurable at 10 minutes on free 1-minute data.
  The tradable target and longer history did not help. The earlier IEX-close passes disappear
  when the target itself cannot reward the IEX artifact, which fits the Phase 5d explanation.
- These tests covered more than three years and 25 models, and they do not show that no edge can
  exist. They show no edge these free inputs can find at this horizon, in line with the literature
  (predictability reported at seconds and fading within minutes in liquid stocks).
- By the rule set in advance, research stops here and the Phase 6 choice goes to the user, with no
  evidence of an edge to report.

---

<!-- Add new experiment entries below as phases progress -->
