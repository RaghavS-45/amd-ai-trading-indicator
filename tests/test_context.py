"""
test_context.py
Context features (SPY, QQQ, NVDA) must use bar t of each symbol, never t+1,
and fill a missing bar with the previous close, never the next one.

Run:
    python -m pytest tests/test_context.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_grid import join_context
from features import CFG, compute_features

SYMS = [s.lower() for s in CFG["context_symbols"]]
DAYS = ["2025-01-15", "2025-01-16"]


def _index(days=DAYS):
    return pd.DatetimeIndex([ts for d in days for ts in pd.date_range(
        f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")]).tz_convert("UTC")


def _frame(seed=0):
    idx = _index()
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                       "volume": 1000.0, "trade_count": 10.0, "vwap": close,
                       "is_filled": False}, index=idx)
    for s in SYMS:
        df[f"{s}_close"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        df[f"{s}_volume"] = 500.0
        df[f"{s}_is_filled"] = False
    return df


def test_truncation_does_not_change_past():
    df = _frame()
    cut = 250
    full = compute_features(df, ["context"]).iloc[:cut]
    short = compute_features(df.iloc[:cut], ["context"])
    pd.testing.assert_frame_equal(full, short)


@pytest.mark.parametrize("t", [100, 200, 389, 500])
def test_future_context_changes_nothing_up_to_t(t):
    df = _frame()
    changed = df.copy()
    for s in SYMS:
        changed.iloc[t + 1:, changed.columns.get_loc(f"{s}_close")] *= 1.3
    a = compute_features(df, ["context"]).iloc[:t + 1]
    b = compute_features(changed, ["context"]).iloc[:t + 1]
    pd.testing.assert_frame_equal(a, b)


def test_exact_alignment_uses_bar_t():
    df = _frame()
    feat = compute_features(df, ["context"])
    for s in SYMS:
        c = df[f"{s}_close"].to_numpy()
        for t in (50, 123, 300):
            assert feat[f"{s}_ret_1"].iloc[t] == pytest.approx(np.log(c[t] / c[t - 1]), abs=1e-15)
            assert feat[f"{s}_ret_10"].iloc[t] == pytest.approx(np.log(c[t] / c[t - 10]), abs=1e-15)


def test_relative_strength():
    df = _frame()
    feat = compute_features(df, ["context"])
    t = 200
    amd = np.log(df["close"].iloc[t] / df["close"].iloc[t - 10])
    qqq = np.log(df["qqq_close"].iloc[t] / df["qqq_close"].iloc[t - 10])
    assert feat["amd_minus_qqq_ret_10"].iloc[t] == pytest.approx(amd - qqq)


def test_day1_context_does_not_reach_day2():
    df = _frame()
    changed = df.copy()
    for s in SYMS:
        changed.iloc[:390, changed.columns.get_loc(f"{s}_close")] *= 2
    a = compute_features(df, ["context"]).iloc[390:]
    b = compute_features(changed, ["context"]).iloc[390:]
    pd.testing.assert_frame_equal(a, b)


def test_join_fills_missing_bar_with_previous_close():
    asset = _frame()[["open", "high", "low", "close", "volume", "is_filled"]]
    asset = asset[asset.index.tz_convert("America/New_York").date
                  == pd.Timestamp(DAYS[0]).date()]
    raw = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                        "close": np.arange(len(asset), dtype=float) + 10,
                        "volume": 100.0}, index=asset.index)
    k = 100
    raw = raw.drop(raw.index[k])
    joined, real_frac = join_context(asset, "SPY", raw)
    assert joined["spy_close"].iloc[k] == raw["close"].iloc[k - 1]   # previous, not next
    assert joined["spy_is_filled"].iloc[k]
    assert not joined["spy_is_filled"].drop(asset.index[k]).any()
    assert joined["spy_volume"].iloc[k] == 0
    assert (joined.index == asset.index).all()
    assert real_frac.iloc[0] == pytest.approx(1 - 1 / len(asset))
