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

from features import compute_features


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


def test_feature_columns_stable():
    """Feature names must not change between calls – model depends on column order."""
    df = _make_fake_ohlcv(100)
    cols1 = list(compute_features(df).columns)
    cols2 = list(compute_features(df).columns)
    assert cols1 == cols2
