# AMD 10-Minute Indicator

A live indicator for AMD stock that helps a user judge the next 10 minutes. After every closed
1-minute candle it shows:

- **Move size:** the typical move and an 80% range for the price 10 minutes later.
- **Activity:** calm, normal or busy compared with a normal day at the same time.
- **Direction:** ▲ Up / ▼ Down with a confidence, or "No clear signal". This is a weak hint only.
- **Context:** price levels (VWAP, today's high/low, yesterday's close/high/low), QQQ and NVDA moves, AMD news.
- **A track record:** every signal is stored once and scored 10 minutes later.

Live site: <https://amd-indicator.onrender.com>

University project (IIT Patna). Paper trading and study use only. Not financial advice.

## What the tests showed

| Question | Answer | Evidence |
|---|---|---|
| Can the **size** of the next move be forecast? | Yes | R² about 0.5 on log variance; the 80% range held 80.6% of the time over 840 walk-forward days (Feb 2023 to June 2026) |
| Can the **direction** be forecast? | No | 25 models tested; none beat always guessing "Up" under strict scoring (next-open entry, SIP prices, after costs) |

So the main output is the move-size forecast. The direction arrow is kept because the project brief asks
for it, and the page says plainly that it was right about 50% of the time in testing.

**Final hold-out test** (2026-07-01 to 2026-10-14, 74 days never used before, run once on the frozen app code):

| | Result |
|---|---|
| Move size | R² 0.52. The 80% range held 84.0% of the time (± 0.6): a little too wide, on the safe side, in every part of the day and every month |
| Direction arrows | Right 52.1% of the time (always Up 49.1%, RSI rule 50.4%), but about zero profit after costs under strict scoring (−0.07 bps per trade), so they did not pass |
| LightGBM comparison | Slightly better than HAR on the test (R² 0.527 vs 0.516) |

The ranges in this test came from the HAR model, which was the live model until 14 Nov 2026. After
the test we moved the ranges to our own LightGBM model (see below and Phase 7f in EXPERIMENTS.md).

Every experiment, including the failed ones, is in [EXPERIMENTS.md](EXPERIMENTS.md).

## The models

All are fitted on training data only (up to 2026-06-30) by `src/fit_live.py` and frozen in `models/`.

**Move size: LightGBM** (our own model, live since 14 Nov 2026). Gradient-boosted trees on 15 inputs:
how much AMD moved in the last 10 minutes, the last 30 minutes, today so far, yesterday, and what is
normal for this time of day; the same for QQQ and NVDA over 10 and 30 minutes; the AMD news count of
the last 30 minutes; the overnight gap; volume and trade-count surprises; and the time of day. The
forecast is turned into ranges with multipliers measured on real data:

| | First hour | Midday | Last hour |
|---|---|---|---|
| 80% range | 1.32 | 1.21 | 1.36 |
| Typical move (50%) | 0.65 | 0.60 | 0.69 |

**Baseline and backup: HAR + time of day** (Corsi, 2009, adapted to 10 minutes). A linear model on the
first five inputs above. It is shown beside every forecast and stored with every signal, and it takes
over automatically on any minute where a LightGBM input is missing. The calm / normal / busy level
also comes from HAR.

How the choice was made, in order. On 6 Nov HAR was chosen for the launch, because LightGBM beat it on
2024 to 2026 but not on 2023 to 2024. On 14 Nov the final test showed LightGBM slightly ahead again on
unseen data (R² 0.527 vs 0.516), and we promoted it: when our own trained model is similar to or
slightly better than the published baseline, we use ours. This choice was made after seeing the test,
so the test is not an untouched check of the LightGBM ranges; the live record from 16 Nov is. On the
test days the LightGBM ranges as served held 84.6% of the time (HAR 84.0%) and were slightly wider, so
we claim a small gain in forecast accuracy, not better ranges.

**Direction: logistic regression** on 32 price, market (SPY, QQQ, NVDA) and intraday inputs. It gives
the chance of Up (more than +0.10%), Down (less than −0.10%) and Flat. An arrow is shown when the gap
between Up and Down is large enough; the bar is re-set each morning from the previous 20 sessions so
that about 1 signal in 5 gets an arrow on a normal day.

## Rules the project follows

| Rule | How it is kept |
|---|---|
| No look-ahead | Features use only bars up to the current closed candle; `tests/test_no_leakage.py` checks every feature set |
| No shuffling | Chronological splits and walk-forward validation only (`src/walkforward.py`) |
| No repainting | Signals are written once per closed candle to SQLite; triggers refuse UPDATE and DELETE |
| Same code live and in training | One feature function, `src/features.py::compute_features()`; `src/check_live_parity.py` compares the two |
| Test set touched once | Loaders refuse dates after `train_end`; `src/phase6_final_test.py --final` refuses to run twice; the app never shows a hold-out day |
| Reproducible | Dates and seeds in `config.yaml`, pinned `requirements.txt`, dataset checksums, `run_all.py` |
| Changes after going live are logged | Phases 7c and 7d in EXPERIMENTS.md, chosen on training data only; Phase 7f (LightGBM ranges), chosen after the final test and logged as such |
| No keys in code | Alpaca keys come from environment variables or a git-ignored `.env` |

## Quick start

Needs Python 3.9 and a free Alpaca account (IEX feed).

### 1. Install

```bash
python3.9 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 2. Set Alpaca keys

Create a `.env` file in the repo root (it is git-ignored), or export the same two variables:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### 3. Run the app

The frozen models are in the repo, so the app runs without downloading any training data.

```bash
uvicorn app.server:app
```

Open <http://localhost:8000>. During market hours (09:30 to 16:00 New York) it is live; forecasts start
at 10:03. Outside market hours it shows the last session recomputed after the close, marked REVIEW.

To replay a training day bar by bar (needs the data from step 4):

```bash
APP_MODE=replay REPLAY_DAY=2026-06-15 REPLAY_SPEED=5 uvicorn app.server:app
```

### 4. Reproduce the research

Data files are not in the repo (only their SHA-256 checksums). Download them first:

```bash
python src/download_data.py
```

Repeat with `--symbol SPY`, `--symbol QQQ` and `--symbol NVDA`. Phases 5c to 5f also need
`src/download_news.py`, `src/download_peers.py` and `src/download_history.py`; the exact commands are
at the top of each script. Then:

```bash
python run_all.py
```

This verifies the checksums, builds the minute grid, runs every experiment in order and refits the live
models. The full run takes over an hour (Phase 5e about 40 minutes, Phase 5f about 25).

### 5. Run the tests

```bash
python -m pytest tests/ -v
```

116 tests: leakage, labels, walk-forward splits, trade simulation, the live engine (no repainting, late
bars, restarts) and the scorecard.

## The website

| Page | What it shows |
|---|---|
| Live | Chart with the 80% range drawn 10 minutes ahead, the latest forecast, today's report card |
| History | Any finished day since 16 Oct 2026, with every signal and a CSV download |
| Performance | The live track record by day, the pre-live tests, and every experiment |
| How to read | What each number means and when not to trust it |

The live scorecard grades confident direction signals and paper profit after costs (10 shares,
$0.005/share fee and $0.005/share slippage each way) against two baselines: always Up, and an RSI rule.

## Hosting

The site runs on Render's free plan from the `Dockerfile` (`render.yaml`; auto-deploy is off so a push
never restarts the app in the middle of a session). The free disk is wiped on restart, so
`src/record_sync.py` saves the signal record every 5 minutes to a private Hugging Face dataset and
restores it at start-up. Each save is a dated commit, which also shows when every signal was written.

## Repo structure

```
├── config.yaml            every date, seed, threshold and file path
├── run_all.py             reproduces all results in order
├── EXPERIMENTS.md         log of every experiment, including failures
├── REFERENCES.md          libraries and papers used
├── requirements.txt       pinned versions (research); requirements-app.txt (hosted app only)
├── Dockerfile, render.yaml
├── data/                  checksums only; Parquet files are downloaded
├── models/                frozen live model (joblib) and a readable summary (JSON)
├── report/                result tables and logs of each phase, experiment_log.csv
├── app/
│   ├── server.py          FastAPI server: live, review and replay modes
│   └── static/            the web page (Lightweight Charts)
├── src/
│   ├── download_*.py      bars, news, peers and older history from Alpaca
│   ├── build_grid.py      complete per-session minute grid, quiet minutes filled flat
│   ├── features.py        the one feature function, used in training and live
│   ├── labels.py          Up / Down / Neutral labels, 10 minutes ahead
│   ├── dataset.py, sessions.py, walkforward.py, evaluate.py, tradable.py
│   │                      modelling rows, row selection, splits and scoring
│   ├── volatility.py      move-size targets and features
│   ├── phase4_*.py … phase5f_*.py   the experiments, one script per phase
│   ├── phase7c_*.py, phase7d_*.py   checks behind the two post-launch changes
│   ├── phase6_final_test.py         the final hold-out test (run once)
│   ├── phase6b_lgbm_live.py         the LightGBM ranges as served, on the same days (report only)
│   ├── fit_live.py        fits and freezes the live models
│   ├── live_model.py      turns features into the numbers shown
│   ├── live_engine.py     the minute loop shared by live, review, replay and tests
│   ├── live_sources.py    Alpaca (live) and local files (replay)
│   ├── live_score.py      the live scorecard
│   ├── signal_store.py    write-once SQLite store
│   └── record_sync.py     outside copy of the record
└── tests/                 116 tests
```

## Known limits

- **Free data.** The IEX feed sees only part of the market. Quiet minutes are filled flat and marked on the chart.
- **Earnings and big news.** Moves are larger and the 80% range is missed more often on those days.
- **One day is noisy.** The 80% figure is an average over many days. In testing, about 1 day in 20 had the range hold less than 70% of the time.
- **Direction has no proven edge.** Never act on the arrow alone.
