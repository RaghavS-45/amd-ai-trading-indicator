"""
features.py  –  THE shared feature function.
Used in training (Phase 3+) AND the live server (Phase 7).
Both must call compute_features() from this file – never define features twice.

Rule: every feature uses only data up to the current row (no look-ahead).
The automated leakage test in tests/test_no_leakage.py verifies this.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input : a DataFrame with columns [open, high, low, close, volume]
            sorted chronologically, index = DatetimeTZAware (UTC).
    Output: a DataFrame of features, same index, NaN rows at the top
            (due to rolling windows) are intentionally kept so index
            alignment is preserved; callers must dropna() themselves.

    No future data is ever read inside this function.
    """
    feat = pd.DataFrame(index=df.index)

    # ── 1. Log returns ──────────────────────────────────────────────
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for lag in CFG["return_lags"]:
        feat[f"ret_{lag}"] = log_ret.shift(lag - 1).rolling(lag).sum()
        # ret_1 = last 1-bar return; ret_5 = sum of last 5 bar returns

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
    minutes_since_open = _minutes_since_open(df.index)
    session_length = 390  # minutes in a regular session
    angle = 2 * np.pi * minutes_since_open / session_length
    feat["tod_sin"] = np.sin(angle)
    feat["tod_cos"] = np.cos(angle)

    return feat


# ── Private helper functions ────────────────────────────────────────

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


def _bb_position(close: pd.Series, window: int, n_std: float) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    # 0 = at lower band, 0.5 = at mid, 1 = at upper band
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _minutes_since_open(index: pd.DatetimeIndex) -> pd.Series:
    import pytz
    et = pytz.timezone("America/New_York")
    local = index.tz_convert(et)
    open_time = pd.Timestamp("09:30").time()
    minutes = pd.Series(
        [(t.hour * 60 + t.minute) - (9 * 60 + 30) for t in local.time],
        index=index,
    )
    return minutes.clip(lower=0)
