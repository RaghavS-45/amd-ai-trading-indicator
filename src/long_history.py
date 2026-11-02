"""
long_history.py  –  one continuous grid from history_start to train_end, plus the columns
that need earlier sessions, and tradable returns from SIP.

Raw data at or after peers_end is cut first, so the test period never enters. SIP is used for
targets and scoring only. Features use IEX prices, as they will live.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from build_grid import build_grid, drop_sparse_days, join_context
from sessions import ET, minutes_since_open

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
ONE, ZERO = pd.Timedelta(minutes=1), pd.Timedelta(0)


def _main_raw(sym: str) -> pd.DataFrame:
    return pd.read_parquet(ROOT / CFG["raw_data_template"].format(sym=sym.lower()))


def _hist_raw(sym: str, feed: str = "iex") -> pd.DataFrame:
    return pd.read_parquet(ROOT / CFG["history_template"].format(sym=sym.lower(), feed=feed))


def joined_raw(hist: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """History before history_end, main data up to peers_end; never test-period rows."""
    start = pd.Timestamp(CFG["history_end"], tz="UTC")
    stop = pd.Timestamp(CFG["peers_end"], tz="UTC")
    hist = hist[pd.to_datetime(hist.index, utc=True) < start]
    idx = pd.to_datetime(main.index, utc=True)
    return pd.concat([hist, main[(idx >= start) & (idx < stop)]]).sort_index()


def seasonal_mean(grid: pd.DataFrame) -> pd.Series:
    """For each row: mean return of its half-hour bin over previous sessions, today excluded."""
    day = pd.Series(grid.index.tz_convert(ET).date, index=grid.index)
    half_hour = pd.Series(minutes_since_open(grid.index) // 30, index=grid.index)
    grouped = grid.groupby([day.rename("day"), half_hour.rename("bin")])
    table = np.log(grouped["close"].last() / grouped["open"].first()).unstack("bin")
    past = table.shift(1).rolling(CFG["seasonal_lookback_days"], min_periods=CFG["seasonal_min_days"]).mean()
    keys = pd.MultiIndex.from_arrays([day, half_hour])
    return pd.Series(past.stack().reindex(keys).to_numpy(), index=grid.index)


def previous_close(close: pd.Series) -> pd.Series:
    """Last close of the previous session, on every row of a session."""
    day = pd.Series(close.index.tz_convert(ET).date, index=close.index)
    last = close.groupby(day).last().shift(1)
    return pd.Series(day.map(last).to_numpy(dtype=float), index=close.index)


def load_grid() -> pd.DataFrame:
    start, end = CFG["history_start"], CFG["train_end"]
    grid = build_grid(joined_raw(_hist_raw(CFG["asset"]), _main_raw(CFG["asset"])), start, end)
    grid = grid[grid["close"].notna()]
    grid, dropped = drop_sparse_days(grid, CFG["min_day_coverage"])
    print(f"AMD grid {len(grid):,} rows, sparse days dropped {len(dropped)}, filled {grid['is_filled'].mean():.2%}")
    for sym in dict.fromkeys(CFG["context_symbols"] + CFG["sector_symbols"]):
        grid, _ = join_context(grid, sym, joined_raw(_hist_raw(sym), _main_raw(sym)), start, end)
        print(f"{sym:<5} filled {grid[f'{sym.lower()}_is_filled'].mean():.2%}")
    grid["seas_mean"] = seasonal_mean(grid)
    grid["prev_close"] = previous_close(grid["close"])
    grid["qqq_prev_close"] = previous_close(grid["qqq_close"])
    return grid


def load_sip() -> pd.DataFrame:
    hist = _hist_raw(CFG["asset"], "sip")
    main = pd.read_parquet(ROOT / CFG["sip_check_path"])
    return joined_raw(hist, main)


def at(col: pd.Series, idx: pd.DatetimeIndex, delta: pd.Timedelta) -> np.ndarray:
    return col.reindex(idx + delta).to_numpy()


def tradable_return(index: pd.DatetimeIndex, sip: pd.DataFrame) -> pd.Series:
    """Log return from SIP open of bar t+1 to SIP close of bar t+horizon, same session only."""
    end = index + pd.Timedelta(minutes=CFG["horizon_bars"])
    same_day = index.tz_convert(ET).date == end.tz_convert(ET).date
    ret = np.log(at(sip["close"], index, pd.Timedelta(minutes=CFG["horizon_bars"])) / at(sip["open"], index, ONE))
    return pd.Series(np.where(same_day, ret, np.nan), index=index)
