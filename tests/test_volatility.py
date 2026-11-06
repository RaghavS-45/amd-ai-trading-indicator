"""
test_volatility.py
Move-size columns use earlier data only; targets cover bars t+1..t+10 of the same session.

Run:
    python -m pytest tests/test_volatility.py -v
"""

import sys
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import CFG, compute_features
from long_history import previous_close
from volatility import interval_score, prev_day_rv, qlike, seasonal_rv, sip_targets, squared_returns

DAYS = [str(d.date()) for d in xcals.get_calendar("XNYS").sessions_in_range("2025-02-03", "2025-03-10")]
H = CFG["horizon_bars"]


def _index(days):
    return pd.DatetimeIndex([ts for d in days for ts in pd.date_range(
        f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")]).tz_convert("UTC")


def _close(days=DAYS, seed=4):
    idx = _index(days)
    return pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.001, len(idx)))), index=idx)


def _frame(days=DAYS):
    close = _close(days)
    df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close,
                       "volume": 1000.0, "trade_count": 10.0, "vwap": close, "news_count": 0.0}, index=close.index)
    for sym in CFG["context_symbols"]:
        df[f"{sym.lower()}_close"] = close * 0.5
    df["prev_day_rv"] = prev_day_rv(close)
    df["seas_rv"] = seasonal_rv(close)
    df["prev_close"] = previous_close(close)
    return df


def test_squared_returns_skip_overnight():
    sq = squared_returns(_close())
    first_bars = sq.index.tz_convert("America/New_York").strftime("%H:%M") == "09:30"
    assert sq[first_bars].isna().all()


def test_prev_day_rv_is_previous_session():
    close = _close()
    sq = squared_returns(close)
    day = close.index.tz_convert("America/New_York").date
    expected = sq[day == pd.Timestamp(DAYS[3]).date()].mean() * H
    assert abs(prev_day_rv(close)[day == pd.Timestamp(DAYS[4]).date()].iloc[0] - expected) < 1e-18


def test_seasonal_rv_ignores_today_and_later():
    close = _close()
    changed = close.copy()
    last = close.index.tz_convert("America/New_York").date == pd.Timestamp(DAYS[-1]).date()
    changed[last] *= np.exp(np.random.default_rng(9).normal(0, 0.01, last.sum()))
    pd.testing.assert_series_equal(seasonal_rv(close), seasonal_rv(changed))


def test_sip_targets_use_next_bars_of_same_session():
    idx = _index(DAYS[:2])
    sip = pd.DataFrame({"close": _close(DAYS[:2]).to_numpy()}, index=idx)
    t = sip_targets(idx, sip)
    i = 100
    r = np.log(sip["close"].iloc[i + 1:i + H + 1].to_numpy() / sip["close"].iloc[i:i + H].to_numpy())
    assert abs(t["rv_fwd"].iloc[i] - (r ** 2).sum()) < 1e-15
    assert abs(t["ret_fwd"].iloc[i] - np.log(sip["close"].iloc[i + H] / sip["close"].iloc[i])) < 1e-12
    end_of_day = _index(DAYS[:1])[-H:]
    assert t.loc[end_of_day].isna().all().all()


def test_missing_sip_bar_carried_within_session_only():
    idx = _index(DAYS[:2])
    sip = pd.DataFrame({"close": _close(DAYS[:2]).to_numpy()}, index=idx).drop(idx[50])
    t = sip_targets(idx, sip)
    assert t["rv_fwd"].iloc[45] == t["rv_fwd"].iloc[45]
    first_next_day = _index(DAYS[1:2])[0]
    assert np.isnan(t.loc[_index(DAYS[:1])[-1], "rv_fwd"]) and not np.isnan(t.loc[first_next_day, "rv_fwd"])


def test_vol_features_identical_without_future_rows():
    df = _frame()
    full = compute_features(df, ["vol"])
    cut = len(df) - 250
    pd.testing.assert_frame_equal(full.iloc[:cut], compute_features(df.iloc[:cut], ["vol"]))


def test_qlike_zero_only_at_truth_and_interval_score():
    assert abs(qlike(np.array([1e-4]), np.array([1e-4 + CFG["rv_floor"]]))[0]) < 1e-12
    assert qlike(np.array([1e-4]), np.array([2e-4]))[0] > 0
    s = interval_score(np.array([0.0, 0.03]), np.array([0.01, 0.01]), 0.2)
    assert abs(s[0] - 0.02) < 1e-12 and abs(s[1] - (0.02 + 10 * 0.02)) < 1e-12
