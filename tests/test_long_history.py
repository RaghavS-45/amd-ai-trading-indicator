"""
test_long_history.py
Seasonal, sector and last-half features see no future data; the test period never enters;
tradable returns start at the next bar's open; strict scoring works as stated.

Run:
    python -m pytest tests/test_long_history.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import CFG, compute_features
from long_history import joined_raw, previous_close, seasonal_mean, tradable_return
from tradable import score

DAYS = [str(d.date()) for d in pd.bdate_range("2025-01-06", periods=45)]


def _index(days):
    return pd.DatetimeIndex([ts for d in days for ts in pd.date_range(
        f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")]).tz_convert("UTC")


def _frame(days=DAYS, seed=2):
    idx = _index(days)
    rng = np.random.default_rng(seed)
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.0003, n)), "high": close * 1.001, "low": close * 0.999,
                       "close": close, "volume": 1000.0, "trade_count": 10.0, "vwap": close}, index=idx)
    for sym in set(CFG["context_symbols"] + CFG["sector_symbols"]):
        df[f"{sym.lower()}_close"] = 50 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return df


def _with_columns(df):
    df = df.copy()
    df["seas_mean"] = seasonal_mean(df)
    df["prev_close"] = previous_close(df["close"])
    df["qqq_prev_close"] = previous_close(df["qqq_close"])
    return df


def test_seasonal_uses_only_earlier_sessions():
    df = _frame()
    a = seasonal_mean(df)
    changed = df.copy()
    last_day = changed.index.tz_convert("America/New_York").date == pd.Timestamp(DAYS[-1]).date()
    changed.loc[last_day, ["open", "close"]] *= 1.5
    b = seasonal_mean(changed)
    pd.testing.assert_series_equal(a, b)


def test_seasonal_warmup_is_nan():
    s = seasonal_mean(_frame())
    first = _index(DAYS[:CFG["seasonal_min_days"]])
    assert s.loc[first].isna().all()
    assert s.loc[_index(DAYS[-1:])].notna().all()


def test_seasonal_value_is_mean_of_same_bin():
    df = _frame()
    s = seasonal_mean(df)
    target_day = DAYS[-1]
    ts = pd.Timestamp(f"{target_day} 10:10", tz="America/New_York").tz_convert("UTC")
    rets = []
    for d in DAYS[-1 - CFG["seasonal_lookback_days"]:-1]:
        a = pd.Timestamp(f"{d} 10:00", tz="America/New_York").tz_convert("UTC")
        b = pd.Timestamp(f"{d} 10:29", tz="America/New_York").tz_convert("UTC")
        rets.append(np.log(df.loc[b, "close"] / df.loc[a, "open"]))
    assert abs(s.loc[ts] - np.mean(rets)) < 1e-12


def test_new_sets_identical_without_future_rows():
    df = _with_columns(_frame())
    sets = ["seasonal", "sector"]
    full = compute_features(df, sets)
    cut = len(df) - 200
    part = compute_features(df.iloc[:cut], sets)
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_lasthalf_zero_before_last_half_hour_and_correct_after():
    df = _with_columns(_frame())
    day = _index(DAYS[-1:])
    feat = compute_features(df.loc[day], ["lasthalf"])
    bar_1529 = pd.Timestamp(f"{DAYS[-1]} 15:29", tz="America/New_York").tz_convert("UTC")
    bar_1530 = pd.Timestamp(f"{DAYS[-1]} 15:30", tz="America/New_York").tz_convert("UTC")
    bar_0959 = pd.Timestamp(f"{DAYS[-1]} 09:59", tz="America/New_York").tz_convert("UTC")
    assert feat.loc[:bar_1529].eq(0).all().all()
    expected = np.log(df.loc[bar_0959, "close"] / df.loc[bar_0959, "prev_close"])
    assert abs(feat.loc[bar_1530, "fh_amd"] - expected) < 1e-12
    prev_last = pd.Timestamp(f"{DAYS[-2]} 15:59", tz="America/New_York").tz_convert("UTC")
    assert df.loc[bar_0959, "prev_close"] == df.loc[prev_last, "close"]


def test_joined_raw_never_includes_test_period():
    idx = pd.date_range("2026-06-30 14:00", "2026-07-02 14:00", freq="1h", tz="UTC")
    main = pd.DataFrame({"close": 1.0}, index=idx)
    hist = pd.DataFrame({"close": 2.0}, index=pd.date_range("2024-10-30", periods=4, freq="1D", tz="UTC"))
    out = joined_raw(hist, main)
    assert out.index.max() < pd.Timestamp(CFG["peers_end"], tz="UTC")
    assert (out[out.index < pd.Timestamp(CFG["history_end"], tz="UTC")]["close"] == 2.0).all()
    assert (out[out.index >= pd.Timestamp(CFG["history_end"], tz="UTC")]["close"] == 1.0).all()


def test_tradable_return_starts_at_next_open_and_stays_in_session():
    idx = _index(DAYS[:2])
    sip = pd.DataFrame({"open": np.arange(len(idx)) + 100.0, "close": np.arange(len(idx)) + 200.0}, index=idx)
    ret = tradable_return(idx, sip)
    i = 50
    assert abs(ret.iloc[i] - np.log(sip["close"].iloc[i + 10] / sip["open"].iloc[i + 1])) < 1e-12
    last_bars = _index(DAYS[:1])[-10:]
    assert ret.loc[last_bars].isna().all()


def test_score_pass_needs_direction_and_profit():
    idx = _index(DAYS[:6])
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.002, len(idx))
    x = pd.DataFrame({"trade_ret": ret, "close": 100.0, "date": idx.tz_convert("America/New_York").date}, index=idx)
    assert score(x.assign(signal=np.sign(ret).astype(int)))["pass"]
    assert not score(x.assign(signal=-np.sign(ret).astype(int)))["pass"]
