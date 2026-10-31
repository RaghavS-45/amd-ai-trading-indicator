"""
test_intraday.py
Intraday features use only bars up to t of the same session.

Run:
    python -m pytest tests/test_intraday.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import CFG, compute_features

W = CFG["surprise_window"]


def _frame(days=("2025-01-15", "2025-01-16"), last="15:59", seed=1):
    idx = pd.DatetimeIndex([ts for d in days for ts in pd.date_range(
        f"{d} 09:30", f"{d} {last}", freq="1min", tz="America/New_York")]).tz_convert("UTC")
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.0005, n)),
        "high": close, "low": close, "close": close,
        "volume": rng.integers(100, 5000, n).astype(float),
        "trade_count": rng.integers(1, 100, n).astype(float),
        "vwap": close * (1 + rng.normal(0, 0.0003, n)),
    }, index=idx)


def test_truncation_does_not_change_past():
    df = _frame()
    cut = 600
    full = compute_features(df, ["intraday"]).iloc[:cut]
    short = compute_features(df.iloc[:cut], ["intraday"])
    pd.testing.assert_frame_equal(full, short)


def test_vwap_uses_bars_up_to_t_and_resets_each_session():
    df = _frame()
    feat = compute_features(df, ["intraday"])
    for start, t in [(0, 120), (390, 390 + 45)]:
        s = df.iloc[start:t + 1]
        vwap = (s["vwap"] * s["volume"]).sum() / s["volume"].sum()
        assert feat["vwap_dist"].iloc[t] == pytest.approx(np.log(df["close"].iloc[t] / vwap))


def test_open_dist_uses_first_bar_of_session():
    df = _frame()
    feat = compute_features(df, ["intraday"])
    for start, t in [(0, 77), (390, 500)]:
        expected = np.log(df["close"].iloc[t] / df["open"].iloc[start])
        assert feat["open_dist"].iloc[t] == pytest.approx(expected)


@pytest.mark.parametrize("col,name", [("volume", "vol_surprise"),
                                      ("trade_count", "trades_surprise")])
def test_surprise_excludes_current_bar_and_ignores_future(col, name, t=150):
    df = _frame()
    feat = compute_features(df, ["intraday"])
    x = np.log1p(df[col].to_numpy())
    prev = x[t - W:t]
    expected = (x[t] - prev.mean()) / prev.std(ddof=1)
    assert feat[name].iloc[t] == pytest.approx(expected)

    changed = df.copy()
    changed.iloc[t + 1:, changed.columns.get_loc(col)] *= 50
    pd.testing.assert_series_equal(
        compute_features(changed, ["intraday"])[name].iloc[:t + 1], feat[name].iloc[:t + 1])


def test_surprise_zero_when_baseline_is_flat():
    df = _frame()
    df.iloc[100:100 + W + 1, df.columns.get_loc("volume")] = 0.0
    feat = compute_features(df, ["intraday"])
    assert feat["vol_surprise"].iloc[100 + W] == 0.0


def test_minutes_since_open_and_to_close_full_day():
    feat = compute_features(_frame(), ["intraday"])
    assert feat["min_since_open"].iloc[0] == 0
    assert feat["min_to_close"].iloc[0] == 389
    assert feat["min_to_close"].iloc[389] == 0


def test_min_to_close_half_day():
    df = _frame(days=("2024-11-29",), last="12:59")
    feat = compute_features(df, ["intraday"])
    assert feat["min_to_close"].iloc[-1] == 0
    assert feat["min_to_close"].iloc[0] == 209
