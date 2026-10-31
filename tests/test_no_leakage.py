"""
test_no_leakage.py
Automated leakage test: compute_features() must produce identical results
for the first N rows regardless of whether future rows are present.

Run:
    python -m pytest tests/test_no_leakage.py -v
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import CFG, compute_features


def _make_fake_ohlcv(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    volume = rng.integers(10_000, 100_000, n).astype(float)
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.001, n)),
            "high": close * (1 + abs(rng.normal(0, 0.002, n))),
            "low":  close * (1 - abs(rng.normal(0, 0.002, n))),
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(
            "2024-01-02 14:35:00", periods=n, freq="1min", tz="UTC"
        ),
    )
    return df


def test_features_identical_without_future_rows():
    """
    Core leakage test:
    Features on rows 0..N-1 computed with the full DataFrame (N rows)
    must equal features computed with only the first N rows.
    Any difference means a feature is peeking at future data.
    """
    N = 200   # total rows
    cut = 150  # check that first 'cut' rows are identical

    full_df = _make_fake_ohlcv(N)
    short_df = full_df.iloc[:cut]

    feat_full = compute_features(full_df).iloc[:cut]
    feat_short = compute_features(short_df)

    # Align columns
    cols = feat_full.columns.tolist()
    feat_full = feat_full[cols]
    feat_short = feat_short[cols]

    # Drop NaN rows (rolling warm-up period) – both should have the same NaNs
    mask_full = feat_full.notna().all(axis=1)
    mask_short = feat_short.notna().all(axis=1)
    assert (mask_full == mask_short).all(), (
        "NaN pattern differs between full and short DataFrame."
    )

    for col in cols:
        diff = (feat_full[col] - feat_short[col]).abs()
        assert diff.max() < 1e-9, (
            f"Feature '{col}' differs between full and truncated DataFrame. "
            f"Max diff = {diff.max():.2e}. Possible look-ahead leakage!"
        )


def _make_two_sessions() -> pd.DataFrame:
    days = ["2025-01-15", "2025-01-16"]
    idx = pd.DatetimeIndex(
        [ts for d in days for ts in pd.date_range(
            f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")]
    ).tz_convert("UTC")
    df = _make_fake_ohlcv(len(idx))
    df.index = idx
    return df


def test_features_do_not_cross_overnight_gap():
    """Changing every price of day 1 must not change any feature of day 2."""
    df = _make_two_sessions()
    day1 = slice(0, 390)
    changed = df.copy()
    changed.iloc[day1, :4] *= 1.5
    changed.iloc[day1, 4] *= 3

    a = compute_features(df).iloc[390:]
    b = compute_features(changed).iloc[390:]
    pd.testing.assert_frame_equal(a, b)


def test_session_warmup_is_nan():
    """Return features need same-session history; the first bar has none."""
    df = _make_two_sessions()
    feat = compute_features(df)
    for start in (0, 390):
        first = feat.iloc[start]
        assert first.drop(["tod_sin", "tod_cos"]).isna().all()
        for lag in CFG["return_lags"]:
            col = feat[f"ret_{lag}"].iloc[start:start + 390]
            assert col.iloc[:lag].isna().all()
            assert col.iloc[lag:].notna().all()


def test_return_lags_cover_last_n_bars():
    df = _make_two_sessions()
    feat = compute_features(df)
    t = 200
    for lag in CFG["return_lags"]:
        expected = np.log(df["close"].iloc[t] / df["close"].iloc[t - lag])
        assert abs(feat[f"ret_{lag}"].iloc[t] - expected) < 1e-12


def test_feature_columns_stable():
    """Feature names must not change between calls – model depends on column order."""
    df = _make_fake_ohlcv(100)
    cols1 = list(compute_features(df).columns)
    cols2 = list(compute_features(df).columns)
    assert cols1 == cols2


def test_rsi_defined_without_losses():
    """Only gains -> RSI 100; flat prices -> RSI 50; warm-up stays NaN."""
    df = _make_two_sessions()
    p = CFG["rsi_period"]
    df.iloc[100:100 + p + 1, 3] = np.linspace(100, 101, p + 1)
    df.iloc[300:300 + p + 1, 3] = 100.0
    rsi = compute_features(df)["rsi"]
    assert rsi.iloc[100 + p] == 100
    assert rsi.iloc[300 + p] == 50
    assert rsi.iloc[:p].isna().all()
