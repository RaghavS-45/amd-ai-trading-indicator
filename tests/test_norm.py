"""
test_norm.py
Volatility-scaled features: clipping, flat windows and session warm-up.

Run:
    python -m pytest tests/test_norm.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import CFG, _scaled, compute_features


def _frame(close, n_days=1):
    idx = pd.date_range("2025-01-15 09:30", periods=len(close), freq="1min",
                        tz="America/New_York").tz_convert("UTC")
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
                       "volume": 1000.0, "trade_count": 10.0, "vwap": close}, index=idx)
    for sym in CFG["context_symbols"]:
        df[f"{sym.lower()}_close"] = close
    return df


def test_scaled_formula_and_clip():
    ret = pd.Series([0.01, 0.01, -1.0])
    vol = pd.Series([0.01, 0.005, 0.01])
    z = _scaled(ret, vol, 4)
    assert abs(z[0] - 0.5) < 1e-12
    assert abs(z[1] - 1.0) < 1e-12
    assert z[2] == -CFG["norm_clip"]


def test_flat_window_gives_zero_not_nan():
    z = _scaled(pd.Series([0.0, np.nan]), pd.Series([0.0, 0.0]), 1)
    assert z[0] == 0.0
    assert np.isnan(z[1])


def test_flat_prices_are_usable_after_warmup():
    feat = compute_features(_frame(np.full(100, 100.0)), ["norm"])
    assert feat.iloc[60:].notna().all().all()


def test_all_scaled_values_inside_clip():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.standard_t(2, 200) * 0.001))
    feat = compute_features(_frame(close), ["norm"])
    z_cols = [c for c in feat.columns if c.startswith("z_")]
    assert feat[z_cols].abs().max().max() <= CFG["norm_clip"]


def test_warmup_is_nan():
    feat = compute_features(_frame(100 + np.arange(100) * 0.01), ["norm"])
    assert feat.iloc[:CFG["vol_window"]].isna().any(axis=1).all()
