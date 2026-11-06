"""
volatility.py  –  grid columns and targets for forecasting the size of the next move.

Realized variance (RV) = sum of squared 1-minute log returns. Grid columns use IEX closes and
earlier sessions only. Targets use SIP closes, within the same session:
  rv_fwd   – RV over bars t+1 .. t+horizon (close t to close t+horizon)
  ret_fwd  – log return from close t to close t+horizon, for the band
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from long_history import ET, load_grid
from news import news_count_column
from sessions import minutes_since_open

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def _day(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.tz_convert(ET).date, index=index)


def squared_returns(close: pd.Series) -> pd.Series:
    """Squared 1-minute log returns; the first bar of each session gets none."""
    day = _day(close.index)
    return (np.log(close / close.groupby(day).shift(1))) ** 2


def prev_day_rv(close: pd.Series) -> pd.Series:
    """Previous session's mean squared return, scaled to a 10-minute horizon."""
    day = _day(close.index)
    per_day = squared_returns(close).groupby(day).mean().shift(1) * CFG["horizon_bars"]
    return pd.Series(day.map(per_day).to_numpy(dtype=float), index=close.index)


def seasonal_rv(close: pd.Series) -> pd.Series:
    """Mean RV of this 10-minute bin over previous sessions; today excluded."""
    day = _day(close.index)
    bin_ = pd.Series(minutes_since_open(close.index) // CFG["horizon_bars"], index=close.index)
    table = squared_returns(close).groupby([day.rename("day"), bin_.rename("bin")]).sum(min_count=1).unstack("bin")
    past = table.shift(1).rolling(CFG["seasonal_rv_days"], min_periods=CFG["seasonal_rv_min_days"]).mean()
    keys = pd.MultiIndex.from_arrays([day, bin_])
    return pd.Series(past.stack().reindex(keys).to_numpy(), index=close.index)


def sip_targets(index: pd.DatetimeIndex, sip: pd.DataFrame) -> pd.DataFrame:
    """rv_fwd and ret_fwd from SIP closes on the grid, carried forward inside a session only."""
    day = _day(index)
    close = sip["close"].reindex(index).groupby(day).ffill()
    sq = squared_returns(close)
    h = CFG["horizon_bars"]
    rv_fwd = sq.groupby(day).transform(lambda x: x.rolling(h, min_periods=h).sum().shift(-h))
    ret_fwd = np.log(close.groupby(day).shift(-h) / close)
    return pd.DataFrame({"rv_fwd": rv_fwd, "ret_fwd": ret_fwd}, index=index)


def load_news_all() -> pd.DataFrame:
    hist = pd.read_parquet(ROOT / CFG["news_hist_raw_path"])
    main = pd.read_parquet(ROOT / CFG["news_raw_path"])
    created = pd.to_datetime(hist["created_at"], utc=True)
    hist = hist[created < pd.Timestamp(CFG["history_end"], tz="UTC")]
    return pd.concat([hist, main], ignore_index=True)


def load_vol_grid() -> pd.DataFrame:
    grid = load_grid()
    grid["prev_day_rv"] = prev_day_rv(grid["close"])
    grid["seas_rv"] = seasonal_rv(grid["close"])
    grid["news_count"] = news_count_column(grid.index, load_news_all())
    return grid


def qlike(rv: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = (rv + CFG["rv_floor"]) / forecast
    return ratio - np.log(ratio) - 1


def interval_score(ret: np.ndarray, half_width: np.ndarray, alpha: float) -> np.ndarray:
    """Gneiting & Raftery (2007) score of the band [-w, w]: width plus a penalty for misses."""
    lo, hi = -half_width, half_width
    return (hi - lo) + (2 / alpha) * (lo - ret) * (ret < lo) + (2 / alpha) * (ret - hi) * (ret > hi)
