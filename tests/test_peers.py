"""
test_peers.py
Peer data stays in the train period, and validation rows are always the asset's.

Run:
    python -m pytest tests/test_peers.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peers import CFG, peer_grid, split_rows

DAYS = ["2026-06-29", "2026-06-30"]


def _index(days):
    return pd.DatetimeIndex([ts for d in days for ts in pd.date_range(
        f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")]).tz_convert("UTC")


def _raw(days, seed=1):
    idx = _index(days)
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.001, len(idx))))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 100.0,
                         "trade_count": 5.0, "vwap": close}, index=idx)


def _asset_grid():
    idx = _index(DAYS)
    grid = pd.DataFrame({"close": 1.0}, index=idx)
    for i, sym in enumerate(CFG["context_symbols"]):
        grid[f"{sym.lower()}_close"] = 50.0 + i
    return grid


def test_raw_after_cutoff_is_never_used():
    raw = _raw(DAYS + ["2026-07-01"])
    a = peer_grid(raw, _asset_grid())
    b = peer_grid(raw[raw.index < pd.Timestamp(CFG["peers_end"], tz="UTC")], _asset_grid())
    pd.testing.assert_frame_equal(a, b)
    assert a.index.max() < pd.Timestamp(CFG["peers_end"], tz="UTC")


def test_context_comes_from_same_timestamp():
    grid = peer_grid(_raw(DAYS), _asset_grid())
    for i, sym in enumerate(CFG["context_symbols"]):
        assert (grid[f"{sym.lower()}_close"] == 50.0 + i).all()


def test_missing_peer_bar_is_flat_filled_from_previous_close():
    raw = _raw(DAYS)
    gone = raw.index[100]
    grid = peer_grid(raw.drop(gone), _asset_grid())
    assert grid.loc[gone, "is_filled"]
    assert grid.loc[gone, "close"] == raw["close"].iloc[99]


def test_sparse_peer_day_is_dropped():
    raw = _raw(DAYS)
    first_day = raw.index < _index(DAYS[1:])[0]
    raw = raw[~first_day | (np.arange(len(raw)) % 3 == 0)]
    assert set(peer_grid(raw, _asset_grid()).index.tz_convert("America/New_York").date.astype(str)) == {DAYS[1]}


def test_split_judges_only_the_asset():
    days = list(range(10))
    pool = pd.DataFrame({"date": days * 2, "symbol": [CFG["asset"]] * 10 + ["PEER"] * 10})
    inner, hold, train, val = split_rows(pool, days[:7], days[8:], 2)
    assert set(inner["symbol"]) == {CFG["asset"], "PEER"} and inner["date"].max() == 4
    assert set(train["symbol"]) == {CFG["asset"], "PEER"} and train["date"].max() == 6
    assert set(hold["symbol"]) == {CFG["asset"]} and sorted(hold["date"]) == [5, 6]
    assert set(val["symbol"]) == {CFG["asset"]} and sorted(val["date"]) == [8, 9]
