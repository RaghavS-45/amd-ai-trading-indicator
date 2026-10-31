"""
dataset.py  –  the modelling dataset: features, label and forward return on
usable rows (every feature present, label exists, tradable_mask).

Usage (from other scripts):
    from dataset import build_dataset
    data = build_dataset(end_date=CFG["train_end"])   # never loads the test period
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from features import compute_features
from labels import make_labels
from sessions import ET, tradable_mask

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


META_COLS = ["label", "fwd_ret", "close", "date"]


def feature_columns(data: pd.DataFrame) -> list:
    return [c for c in data.columns if c not in META_COLS]


def forward_return(df: pd.DataFrame) -> pd.Series:
    """Log return to t + horizon_bars minutes, same timestamp lookup as labels.py."""
    target = df.index + pd.Timedelta(minutes=CFG["horizon_bars"])
    future = df["close"].reindex(target).to_numpy()
    same_day = df.index.tz_convert(ET).date == target.tz_convert(ET).date
    future = np.where(same_day, future, np.nan)
    return pd.Series(np.log(future / df["close"].to_numpy()), index=df.index)


def usable_mask(feat: pd.DataFrame, lab: pd.Series) -> np.ndarray:
    return feat.notna().all(axis=1).to_numpy() & lab.notna().to_numpy() & tradable_mask(feat.index)


def build_dataset(end_date: Optional[str] = None,
                  grid: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Feature columns + label, fwd_ret, close, date (ET) for usable rows.
    Rows after end_date (ET date) are dropped before anything else.
    """
    if grid is None:
        grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    if end_date is not None:
        dates = grid.index.tz_convert(ET).date
        grid = grid[dates <= pd.Timestamp(end_date).date()]

    feat = compute_features(grid)
    lab = make_labels(grid)
    keep = usable_mask(feat, lab)

    data = feat[keep].copy()
    data["label"] = lab[keep].astype(int)
    data["fwd_ret"] = forward_return(grid)[keep]
    data["close"] = grid["close"][keep]
    data["date"] = data.index.tz_convert(ET).date
    if end_date is not None:
        assert data["date"].max() <= pd.Timestamp(end_date).date()
    return data
