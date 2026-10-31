# AMD Trading Indicator

An AI-powered trading indicator for AMD stock. A trained LightGBM model predicts
whether the next ~10 minutes of price action is more likely **Up**, **Down**, or
**Neutral**. Signals are displayed live on a candlestick chart.

## Repo Structure

```
ML_Project/
├── config.yaml          ← single source of truth (dates, seeds, thresholds)
├── requirements.txt     ← pinned versions
├── run_all.py           ← reproduce all results from scratch
├── EXPERIMENTS.md       ← every experiment log entry (required for report)
├── data/                ← raw + grid Parquet files, SHA-256 checksum
├── src/
│   ├── download_data.py ← Phase 1: fetch from Alpaca
│   ├── build_grid.py    ← complete per-session minute grid, fill gaps
│   ├── features.py      ← THE shared feature function (train + live)
│   ├── labels.py        ← 3-class label generation
│   ├── sessions.py      ← row selection (skip first drop_open_mins)
│   └── verify_checksum.py
├── models/              ← saved .pkl model files
├── app/                 ← FastAPI server + frontend (Phase 7)
├── notebooks/           ← exploration notebooks (Phase 2)
├── report/              ← figures and final report
└── tests/
    ├── test_no_leakage.py ← automated leakage guard
    └── test_sessions.py   ← opening-minutes row selection
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set Alpaca credentials (never hard-code these)

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

Or create a `.env` file (which is git-ignored):

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

### 3. Download data

```bash
python src/download_data.py
```

### 4. Run the full pipeline

```bash
python run_all.py
```

### 5. Run leakage test

```bash
python -m pytest tests/ -v
```

### 6. Start the live app (Phase 7+)

```bash
uvicorn app.server:app --reload
```

Then open `http://localhost:8000` in your browser.

## Hard Rules (grading constraints)

| Rule | Implementation |
|------|----------------|
| No look-ahead | All features use only data ≤ current bar; tested in `tests/test_no_leakage.py` |
| No shuffle | Chronological train/val/test split; walk-forward only |
| No repainting | Signals written to SQLite once per closed candle, never updated |
| Shared feature fn | `src/features.py::compute_features()` used in both training and live server |
| Reproducibility | `config.yaml` controls all dates/seeds; `run_all.py` is the entry point |
| API keys in env | Never in code or committed |

## Configuration

Edit `config.yaml` to change:
- Date ranges (`train_start`, `train_end`, `test_start`, `test_end`)
- Bar timeframe (`1Min` or `5Min`)
- Label horizon and threshold
- Model hyperparameters

## Paper Trading Note

This is a university project. **No real money is traded.** Signals are
simulated with assumed fees of $0.005/share and $0.005/share slippage.
