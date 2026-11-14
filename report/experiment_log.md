# Experiment log

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


## Phase 4

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 2026-10-31 | Always Up (baseline) | No training | WF9 | IEX | 50.29% (±0.43), −0.54 bps | base 50.29% | Baseline | 4 s (phase) |
| B2 | 2026-10-31 | RSI rule (baseline): Up if RSI(14) < 30; Down if > 70 | No training | WF9 | IEX | 50.51% (±0.67), −0.73 bps | base 50.29% | Baseline | 4 s (phase) |
| E4.1 | 2026-10-31 | Logistic regression on 12 price features | StandardScaler + LogisticRegression (max_iter 1000, seed 42) | WF9 | IEX | 50.20% (±0.43), −0.27 bps | base 50.29% | No skill | 4 s (phase) |

## Phase 5

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| E5.1 | 2026-10-31 | LightGBM on 12 price features | 500 trees, lr 0.05, 31 leaves, seed 42 | WF9 | IEX | 50.22% (±0.31), −0.71 bps | 50.90% | ❌ Fail | 2 min 28 s (phase) |
| E5.2 | 2026-10-31 | LightGBM regularized on 12 price features | 300 trees, lr 0.05, 15 leaves, min_child_samples 500, feature/bagging fraction 0.8, L2 1.0 | WF9 | IEX | 49.78% (±0.37), −1.57 bps | 51.03% | ❌ Fail | 2 min 28 s (phase) |
| E5.3 | 2026-10-31 | LightGBM + SPY/QQQ/NVDA context (26 features) | As E5.1 | WF9 | IEX | 50.48% (±0.34), −1.20 bps | 50.97% | ❌ Fail | 2 min 28 s (phase) |
| E5.4 | 2026-10-31 | LightGBM + intraday features: VWAP; open; volume surprise; time (18) | As E5.1 | WF9 | IEX | 49.50% (±0.36), −1.51 bps | 51.00% | ❌ Fail | 2 min 28 s (phase) |
| E5.5 | 2026-10-31 | LightGBM on all 32 features | As E5.1 | WF9 | IEX | 49.96% (±0.37), −0.72 bps | 51.04% | ❌ Fail | 2 min 28 s (phase) |
| E5.6 | 2026-10-31 | Logistic regression on all 32 features | As E4.1 | WF9 | IEX | 50.67% (±0.40), +0.89 bps | 51.09% | ❌ Fail (direction) | 2 min 28 s (phase) |
| E5.7 | 2026-10-31 | Confidence threshold on E5.6 | τ grid 0.34–0.70 step 0.01, chosen on earlier folds, ≥ 5 signals/day | WF9 (folds 2–9) | IEX | 50.48% (±0.79), +1.24 bps | 51.75% | ❌ Fail (direction) | 2 min 28 s (phase) |

## Phase 5b

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| E5b.1 | 2026-11-02 | Ridge on 15 volatility-scaled features; every score | Ridge alpha 1000 | WF9 judged on folds 6–9 | IEX | 50.25% (±0.38), +0.02 bps | 51.34% | ❌ Fail | 20 s (phase) |
| E5b.2 | 2026-11-02 | Ridge; strongest 20% of scores | Ridge alpha 1000, cutoff from inner holdout | WF9 judged on folds 6–9 | IEX | 50.27% (±0.95), −0.91 bps | 52.47% | ❌ Fail | 20 s (phase) |
| E5b.3 | 2026-11-02 | Logistic on sign; strongest 20% | Logistic, cutoff from inner holdout | WF9 judged on folds 6–9 | IEX | 50.41% (±0.91), −0.11 bps | 52.39% | ❌ Fail | 20 s (phase) |
| E5b.4 | 2026-11-02 | LightGBM regression; strongest 20% | 200 trees, lr 0.02, 7 leaves, min_child_samples 2000, fractions 0.7, L2 10 | WF9 judged on folds 6–9 | IEX | 51.19% (±1.23), −1.43 bps | 53.03% | ❌ Fail | 20 s (phase) |
| E5b.5 | 2026-11-02 | Ridge on last 120 training days only; strongest 20% | Ridge alpha 1000 | WF9 judged on folds 6–9 | IEX | 48.56% (±1.04), −0.69 bps | 52.66% | ❌ Fail | 20 s (phase) |
| E5b.6 | 2026-11-02 | Ridge + LightGBM averaged; strongest 20% | As E5b.1 and E5b.4 | WF9 judged on folds 6–9 | IEX | 51.36% (±1.05), +0.28 bps | 52.69% | ❌ Fail | 20 s (phase) |

## Phase 5c

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| E5c.1 | 2026-11-02 | E5.6 + FinBERT news sentiment and news count | As E5.6; FinBERT ProsusAI/finbert, pinned | WF9 judged on folds 6–9 | IEX | 50.60% (±0.58), +0.72 bps | 51.73% | ❌ Fail | 7 s (phase) |
| E5c.2 | 2026-11-02 | Rule: Down after an AMD-named headline | No training | WF9 judged on folds 6–9 | IEX | 41.75% (±5.07), −6.77 bps | 60.71% | ❌ Fail | 7 s (phase) |
| E5c.3 | 2026-11-02 | Rule: against the headline's sentiment | No training | WF9 judged on folds 6–9 | IEX | 42.72% (±5.01), −8.25 bps | 60.60% | ❌ Fail | 7 s (phase) |
| E5c.4 | 2026-11-02 | E5.6 without Up signals after news | As E5.6 | WF9 judged on folds 6–9 | IEX | 50.47% (±0.60), +0.78 bps | 51.77% | ❌ Fail | 7 s (phase) |

## Phase 5d

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| E5d.1 | 2026-11-02 | Ridge trained on 10 chip stocks; every score | As E5b.1 | POOL | IEX | 50.66% (±0.30), +0.92 bps | 51.18% | ❌ Fail | 1 min 17 s (phase) |
| E5d.2 | 2026-11-02 | Ridge on 10 chip stocks; strongest 20% | As E5b.2 | POOL | IEX | 51.74% (±0.71), +0.67 bps | 52.00% | ❌ Fail | 1 min 17 s (phase) |
| E5d.3 | 2026-11-02 | LightGBM on 10 chip stocks; strongest 20% | As E5b.4 | POOL | IEX | 53.33% (±0.92), +1.09 bps; one minute late −1.05 bps; on SIP 51.20% | 52.42% | ⚠️ Bar met; rejected (IEX artifact) | 1 min 17 s (phase) |
| E5d.4 | 2026-11-02 | Ridge + LightGBM on 10 chip stocks; strongest 20% | As E5b.6 | POOL | IEX | 52.56% (±0.77), +1.76 bps; one minute late −0.85 bps; on SIP 50.03% | 52.13% | ⚠️ Bar met; rejected (IEX artifact) | 1 min 17 s (phase) |

## Phase 5e

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| E5e.0 | 2026-11-02 | E5.6 unchanged on unseen years 2023–2024 | As E5.6 | WF28 fresh years | Strict | 50.35% (±0.30), −1.32 bps | 51.64% (base 51.04%) | ❌ Fail | ~40 min (phase; earlier run) |
| E5e.1 | 2026-11-02 | Ridge + LightGBM on tradable target; longer history | As E5b.6 | WF28 | Strict | Fresh 50.33%, −1.47 bps / Folds 6–9 48.13%, −2.75 bps | 52.31% / 52.75% | ❌ Fail | ~40 min (phase; earlier run) |
| E5e.2 | 2026-11-02 | E5e.1 + same half hour's past returns (Heston et al. 2010) | As E5b.6 | WF28 | Strict | Fresh 50.16%, −1.20 bps / Folds 6–9 48.48%, −2.54 bps | 52.32% / 52.75% | ❌ Fail | ~40 min (phase; earlier run) |
| E5e.3 | 2026-11-02 | E5e.1 + AVGO/TSM/NVDA lead returns (Chinco et al. 2019) | As E5b.6 | WF28 | Strict | Fresh 49.89%, −1.40 bps / Folds 6–9 48.52%, −1.48 bps | 52.30% / 52.79% | ❌ Fail | ~40 min (phase; earlier run) |
| E5e.4 | 2026-11-02 | E5e.1 + first half-hour momentum (Gao et al. 2018) | As E5b.6 | WF28 | Strict | Fresh 50.42%, −1.50 bps / Folds 6–9 48.15%, −1.89 bps | 52.31% / 52.87% | ❌ Fail | ~40 min (phase; earlier run) |

## Phase 5f

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 2026-11-06 | Move size: persistence (next RV = last 10 min) | No training | WF28 | QLIKE | Fresh 0.3314 / Folds 6–9 0.3633; 80% band 80.0% / 79.4% | Beat V2 by 2 SE | Baseline | ~25 min (phase; earlier run) |
| V1 | 2026-11-06 | Move size: HAR (Corsi 2009) adapted to 10 minutes | OLS on log RV (4 inputs) | WF28 | QLIKE | Fresh 0.2040 / Folds 6–9 0.2265; R² 0.50 / 0.53; band 80.0% / 79.3% | Beat V2 by 2 SE | ❌ Not better than V2 | ~25 min (phase; earlier run) |
| V2 | 2026-11-06 | Move size: HAR + time of day | OLS on log RV (5 inputs) | WF28 | QLIKE | Fresh 0.2043 / Folds 6–9 0.2251; R² 0.50 / 0.53; band 80.0% / 79.1% | — | ✅ Chosen for the indicator (rule) | ~25 min (phase; earlier run) |
| V3 | 2026-11-06 | Move size: ridge on 15 features (news; QQQ/NVDA volatility; surprises; gap) | Ridge alpha 1, standardized | WF28 | QLIKE | vs V2: fresh −0.0018 (±0.0014), folds 6–9 −0.0088 (±0.0012) | Beat V2 by 2 SE on both | ❌ Fail (fresh years) | ~25 min (phase; earlier run) |
| V4 | 2026-11-06 | Move size: LightGBM on the same 15 features | 500 trees, lr 0.05, 31 leaves, min_child_samples 500, fractions 0.8 | WF28 | QLIKE | vs V2: fresh −0.0028 (±0.0015), folds 6–9 −0.0116 (±0.0024) | Beat V2 by 2 SE on both | ❌ Fail (fresh years; 1.9 SE) | ~25 min (phase; earlier run) |

## Phase 7

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 2026-11-06 | Live app: features match training | check_live_parity.py on 4 training days | Replay | Exact match | Max difference 1.8e-15; no NaN mismatch | — | ✅ Pass | ~2.5 min |
| L2 | 2026-11-06 | Live app first run (from 12:20 ET) | Frozen V2 + E5.6 lean (model 1477dd2f6cc8) | Live IEX | Ranges | On-time signals: 80% range 30 of 39 (77%), typical move 20 of 39 (51%) | 80% / 50% | Smoke test | Live |

## Phase 7b

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| L3 | 2026-11-08 | Review of 2026-11-06 (all 350 signals) | Frozen model | After training | IEX + ranges | Direction 48 of 101 (48%), −$123.05; always Up 53%, −$111.10; RSI 51%, −$85.35; 80% range 86%, typical 52% | 80% / 50% | Ranges held; no direction edge | under 1 min |

## Phase 7c

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| L4 | 2026-11-14 | Arrow shown on 10% vs 20% of signals (daily-reset cut, previous 20 sessions; change made after seeing live results) | E5.6 walk-forward predictions | Train WF | IEX + close entry | 10%: 49.9% ± 0.9, −0.24 bps; 20%: 50.2% ± 0.7, +0.61 bps; base rate 50.35% | 20% not worse than 10% | 20% used; no direction edge at any share | under 1 min |

## Phase 7d

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| L5 | 2026-11-14 | 80% range: wider multiplier in the first and last hour, midday unchanged (change made after seeing live results) | V2 walk-forward (28 folds) | Train WF | Ranges by hour | Worst hourly gap from 80%: 3.1 → 1.4 points (last 4 folds 4.1 → 2.2); coverage 79.97% → 80.58%; interval score unchanged (109.29) | Smaller worst hourly gap; coverage 79-81% | Used; model 6251e5ccaa53 | under 1 min |

## Phase 6

| Run | Date | What was tried | Model and settings | Split | Scoring | Result | Bar | Outcome | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| P6 | — | Final hold-out test 2026-07-01 → 2026-10-14 (touched once) | Frozen V2 + V4 comparison | Test | Strict + ranges | Not run yet (on hold) | Pre-registered before running | Pending | — |
