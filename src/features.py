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
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sessions import ET, minutes_since_open

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : a DataFrame with columns [open, high, low, close, volume]
            sorted chronologically, index = DatetimeTZAware (UTC).
    Output: a DataFrame of features, same index. Warm-up NaN rows at the
            start of every session are kept so index alignment is
            preserved; callers must dropna() themselves.

    No future data is ever read inside this function.
    """
    session = pd.Series(df.index.tz_convert(ET).date, index=df.index)
    parts = [_session_features(g) for _, g in df.groupby(session, sort=False)]
    if not parts:
        return _session_features(df)
    return pd.concat(parts)


def _session_features(df: pd.DataFrame) -> pd.DataFrame:
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
