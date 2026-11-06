"""
features.py  –  THE shared feature function.
Used in training (Phase 3+) AND the live server (Phase 7).
Both must call compute_features() from this file – never define features twice.

Rule: every feature uses only data up to the current row (no look-ahead).
The automated leakage test in tests/test_no_leakage.py verifies this.

Features are computed per session (ET date): no window reaches across the
overnight gap into the previous day. The first rows of each session are NaN
until every window has enough same-session history (33 bars with the
current config, i.e. features start at 10:03 ET).

Feature sets (config feature_sets, or the sets argument):
  base     – asset returns, volatility, volume ratio, RSI, MACD, Bollinger, time of day
  context  – returns of the context symbols (grid columns <sym>_close), bar t only
  intraday – VWAP / open distance, volume and trade-count surprise, minutes since
             open / to close
  norm     – returns and distances scaled by rolling volatility and clipped at
             +-norm_clip, close location, clipped volume surprise, log volatility,
             time of day
  news     – sentiment sum and article count over recent bars of the same session,
             from grid-aligned columns news_sent, news_count (see news.py)
  seasonal – mean return of this half hour over previous sessions, scaled by volatility,
             from grid column seas_mean (see long_history.py)
  sector   – scaled returns of larger chip stocks and of their equal-weight basket
  vol      – log realized variance of AMD, QQQ, NVDA over recent bars, of the session so far,
             of the previous session and of this 10-minute bin on past sessions (grid columns
             prev_day_rv, seas_rv, see volatility.py); news count; overnight gap; surprises
  lasthalf – first half hour's return from the previous close, for AMD and QQQ; nonzero
             only in the session's last half hour (grid columns prev_close, qqq_prev_close)
"""

from pathlib import Path
from typing import Optional, Sequence

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yaml

from sessions import ET, minutes_since_open

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


FEATURE_SETS = ("base", "context", "intraday", "norm", "news", "seasonal", "sector", "lasthalf", "vol")
_CAL = None


def compute_features(df: pd.DataFrame, sets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Input : a DataFrame with columns [open, high, low, close, volume]
            sorted chronologically, index = DatetimeTZAware (UTC).
    Output: a DataFrame of features, same index. Warm-up NaN rows at the
            start of every session are kept so index alignment is
            preserved; callers must dropna() themselves.

    No future data is ever read inside this function.
    """
    sets = CFG["feature_sets"] if sets is None else sets
    unknown = set(sets) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")
    ordered = [s for s in FEATURE_SETS if s in sets]

    session = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    parts = [_session_features(g, ordered) for _, g in df.groupby(session, sort=False)]
    if not parts:
        return _session_features(df, ordered)
    return pd.concat(parts)


def _session_features(df: pd.DataFrame, sets: Sequence[str]) -> pd.DataFrame:
    builders = {"base": _base, "context": _context, "intraday": _intraday, "norm": _norm, "news": _news,
                "seasonal": _seasonal, "sector": _sector, "lasthalf": _lasthalf,
                "vol": _vol}
    return pd.concat([builders[s](df) for s in sets], axis=1)


def _base(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # ── 1. Log returns ──────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for lag in CFG["return_lags"]:
        feat[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))

    # ── 2. Rolling volatility ───────────────────────────────────────
    feat["volatility"] = log_ret.rolling(CFG["vol_window"]).std()

    # ── 3. Volume relative to rolling mean ──────────────────────────
    vol_ma = df["volume"].rolling(CFG["volume_ma_window"]).mean()
    feat["vol_ratio"] = df["volume"] / vol_ma

    # ── 4. RSI ──────────────────────────────────────────────────────
    feat["rsi"] = _rsi(df["close"], CFG["rsi_period"])

    # ── 5. MACD histogram ───────────────────────────────────────────
    feat["macd_hist"] = _macd_hist(
        df["close"],
        CFG["macd_fast"],
        CFG["macd_slow"],
        CFG["macd_signal"],
    )

    # ── 6. Bollinger Band position ──────────────────────────────────
    feat["bb_pos"] = _bb_position(df["close"], CFG["bb_window"], CFG["bb_std"])

    # ── 7. Time of day (sine/cosine encoding, avoids discontinuity) ──
    session_length = 390  # minutes in a regular session
    angle = 2 * np.pi * minutes_since_open(df.index) / session_length
    feat["tod_sin"] = np.sin(angle)
    feat["tod_cos"] = np.cos(angle)

    return feat


def _context(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    asset = CFG["asset"].lower()
    asset_ret_10 = np.log(df["close"] / df["close"].shift(10))
    for sym in CFG["context_symbols"]:
        s = sym.lower()
        close = df[f"{s}_close"]
        for lag in CFG["context_return_lags"]:
            feat[f"{s}_ret_{lag}"] = np.log(close / close.shift(lag))
        if sym != "SPY":
            feat[f"{asset}_minus_{s}_ret_10"] = asset_ret_10 - np.log(close / close.shift(10))
    return feat


def _intraday(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # Session VWAP from bar VWAPs; filled bars have volume 0 and add nothing
    cum_vol = df["volume"].cumsum()
    vwap = (df["vwap"] * df["volume"]).cumsum() / cum_vol.replace(0, np.nan)
    feat["vwap_dist"] = np.log(df["close"] / vwap)

    # First bar of the session passed in (live code must pass the whole session)
    feat["open_dist"] = np.log(df["close"] / df["open"].iloc[0])

    w = CFG["surprise_window"]
    for col, name in [("volume", "vol_surprise"), ("trade_count", "trades_surprise")]:
        x = np.log1p(df[col])
        base = x.shift(1).rolling(w)  # previous w bars, current bar excluded
        mean, std = base.mean(), base.std()
        z = (x - mean) / std.where(std > 0)
        feat[name] = z.where(~((std == 0) & mean.notna()), 0.0)

    feat["min_since_open"] = minutes_since_open(df.index)
    close_utc = _session_close(df.index[0])
    bar_end = df.index + pd.Timedelta(minutes=1)
    feat["min_to_close"] = (close_utc - bar_end) / pd.Timedelta(minutes=1)
    return feat


def _scaled(ret: pd.Series, vol: pd.Series, bars: int) -> pd.Series:
    """ret / (vol * sqrt(bars)), clipped; a flat window (vol 0) gives 0."""
    z = ret / (vol.where(vol > 0) * np.sqrt(bars))
    return z.mask((vol == 0) & ret.notna(), 0.0).clip(-CFG["norm_clip"], CFG["norm_clip"])


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    w = CFG["vol_window"]
    vol = np.log(df["close"] / df["close"].shift(1)).rolling(w).std()

    for lag in CFG["norm_return_lags"]:
        feat[f"z_ret_{lag}"] = _scaled(np.log(df["close"] / df["close"].shift(lag)), vol, lag)

    for sym in CFG["norm_context_symbols"]:
        s = sym.lower()
        close = df[f"{s}_close"]
        sym_vol = np.log(close / close.shift(1)).rolling(w).std()
        for lag in CFG["norm_context_lags"]:
            feat[f"z_{s}_ret_{lag}"] = _scaled(np.log(close / close.shift(lag)), sym_vol, lag)

    intraday = _intraday(df)
    feat["z_vwap"] = _scaled(intraday["vwap_dist"], vol, CFG["norm_level_bars"])
    feat["z_open"] = _scaled(intraday["open_dist"], vol, CFG["norm_level_bars"])
    feat["vol_surprise_clip"] = intraday["vol_surprise"].clip(-CFG["norm_clip"], CFG["norm_clip"])

    # Filled bars have high == low: close location 0
    location = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    feat["clv_10"] = location.fillna(0.0).rolling(10).mean()

    feat["log_vol"] = np.log(vol.where(vol > 0, 1e-6))
    angle = 2 * np.pi * minutes_since_open(df.index) / 390
    feat["norm_tod_sin"] = np.sin(angle)
    feat["norm_tod_cos"] = np.cos(angle)
    return feat


def _news(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["news_sent_10"] = df["news_sent"].rolling(10, min_periods=1).sum()
    feat["news_sent_30"] = df["news_sent"].rolling(30, min_periods=1).sum()
    feat["news_count_30"] = df["news_count"].rolling(30, min_periods=1).sum()
    return feat


def _seasonal(df: pd.DataFrame) -> pd.DataFrame:
    vol = np.log(df["close"] / df["close"].shift(1)).rolling(CFG["vol_window"]).std()
    return pd.DataFrame({"z_seas": _scaled(df["seas_mean"], vol, 30)}, index=df.index)


def _sector(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    w = CFG["vol_window"]
    one_bar = pd.DataFrame({s: np.log(df[f"{s.lower()}_close"] / df[f"{s.lower()}_close"].shift(1))
                            for s in CFG["sector_symbols"]})
    for sym in ("AVGO", "TSM"):
        vol = one_bar[sym].rolling(w).std()
        for lag in (1, 10):
            close = df[f"{sym.lower()}_close"]
            feat[f"z_{sym.lower()}_ret_{lag}"] = _scaled(np.log(close / close.shift(lag)), vol, lag)
    basket = one_bar.mean(axis=1, skipna=False)
    vol = basket.rolling(w).std()
    for lag in (1, 5):
        feat[f"z_basket_ret_{lag}"] = _scaled(basket.rolling(lag).sum(), vol, lag)
    return feat


def _lasthalf(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    bars = CFG["last_half_minutes"]
    bar_end = df.index + pd.Timedelta(minutes=1)
    in_last = ((_session_close(df.index[0]) - bar_end) / pd.Timedelta(minutes=1)) < bars
    for name, close, prev in [("fh_amd", "close", "prev_close"), ("fh_qqq", "qqq_close", "qqq_prev_close")]:
        first = np.log(df[close].iloc[bars - 1] / df[prev].iloc[bars - 1]) if len(df) >= bars else np.nan
        feat[name] = np.where(in_last, first, 0.0)
    return feat


def _log_rv(x: pd.Series) -> pd.Series:
    return np.log(x + CFG["rv_floor"])


def _vol(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    sq = np.log(df["close"] / df["close"].shift(1)) ** 2
    feat["lrv_10"] = _log_rv(sq.rolling(10).sum())
    feat["lrv_30"] = _log_rv(sq.rolling(30).sum() / 3)
    feat["lrv_sess"] = _log_rv(sq.expanding(min_periods=10).mean() * 10)
    feat["lrv_prev_day"] = _log_rv(df["prev_day_rv"])
    feat["lrv_seas"] = _log_rv(df["seas_rv"])
    for sym in ("qqq", "nvda"):
        sym_sq = np.log(df[f"{sym}_close"] / df[f"{sym}_close"].shift(1)) ** 2
        feat[f"{sym}_lrv_10"] = _log_rv(sym_sq.rolling(10).sum())
        feat[f"{sym}_lrv_30"] = _log_rv(sym_sq.rolling(30).sum() / 3)
    feat["news_count_30"] = df["news_count"].rolling(30, min_periods=1).sum()
    feat["abs_gap"] = np.abs(np.log(df["open"].iloc[0] / df["prev_close"].iloc[0]))
    intraday = _intraday(df)
    feat["vol_surprise_clip"] = intraday["vol_surprise"].clip(-CFG["norm_clip"], CFG["norm_clip"])
    feat["trades_surprise_clip"] = intraday["trades_surprise"].clip(-CFG["norm_clip"], CFG["norm_clip"])
    angle = 2 * np.pi * minutes_since_open(df.index) / 390
    feat["vol_tod_sin"] = np.sin(angle)
    feat["vol_tod_cos"] = np.cos(angle)
    return feat


def _session_close(ts: pd.Timestamp) -> pd.Timestamp:
    global _CAL
    if _CAL is None:
        _CAL = xcals.get_calendar("XNYS")
    day = pd.Timestamp(ts.tz_convert(ET).date())
    return _CAL.session_close(day)


# ── Private helper functions ────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - (100 / (1 + gain / loss))  # loss == 0 -> rs = inf -> 100
    return rsi.mask((gain == 0) & (loss == 0), 50.0)


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line - signal_line


def _bb_position(close: pd.Series, window: int, n_std: float) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # 0 = at lower band, 0.5 = at mid, 1 = at upper band
    return (close - lower) / (upper - lower).replace(0, np.nan)
