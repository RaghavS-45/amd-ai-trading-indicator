"""
peers.py  –  puts a peer stock on the asset's minute grid, and splits pooled rows.

Peers are used only for training. A peer row gets the same context columns as the asset
row at that timestamp. Raw peer data after train_end is cut before anything else.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

from build_grid import build_grid, drop_sparse_days

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def peer_grid(raw: pd.DataFrame, asset_grid: pd.DataFrame) -> pd.DataFrame:
    """asset_grid must already stop at train_end."""
    cutoff = pd.Timestamp(CFG["peers_end"], tz="UTC")
    raw = raw[pd.to_datetime(raw.index, utc=True) < cutoff]
    grid = build_grid(raw).reindex(asset_grid.index)
    grid = grid[grid["close"].notna()]
    grid, _ = drop_sparse_days(grid, CFG["min_day_coverage"])
    for sym in CFG["context_symbols"]:
        grid[f"{sym.lower()}_close"] = asset_grid[f"{sym.lower()}_close"].reindex(grid.index)
    return grid


def split_rows(pool: pd.DataFrame, train_days: list, val_days: list, n_hold: int) -> Tuple[pd.DataFrame, ...]:
    """inner and train: all symbols. hold and val: the asset only, the rows the model is judged on."""
    asset = pool[pool["symbol"] == CFG["asset"]]
    inner = pool[pool["date"].isin(set(train_days[:-n_hold]))]
    train = pool[pool["date"].isin(set(train_days))]
    hold = asset[asset["date"].isin(set(train_days[-n_hold:]))]
    val = asset[asset["date"].isin(set(val_days))]
    return inner, hold, train, val
