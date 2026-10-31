"""
walkforward.py  –  expanding-window walk-forward splits over trading days.

Validation windows are consecutive, non-overlapping blocks of val_window_days,
laid out so the last one ends on the last available day. Each fold trains on
every earlier day except the gap_days immediately before validation.
Dates after train_end are refused: the hold-out test is never split.
"""

import datetime as dt
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

Split = Tuple[List[dt.date], List[dt.date]]


def walk_forward_splits(dates: Iterable[dt.date],
                        min_train_days: Optional[int] = None,
                        val_days: Optional[int] = None,
                        gap_days: Optional[int] = None,
                        last_allowed: Optional[str] = None) -> List[Split]:
    min_train_days = CFG["wf_min_train_days"] if min_train_days is None else min_train_days
    val_days = CFG["val_window_days"] if val_days is None else val_days
    gap_days = CFG["wf_gap_days"] if gap_days is None else gap_days
    last_allowed = CFG["train_end"] if last_allowed is None else last_allowed

    days = sorted(set(dates))
    if days and days[-1] > pd.Timestamp(last_allowed).date():
        raise ValueError(f"Dates after {last_allowed} passed to walk-forward (hold-out test?)")

    n_folds = (len(days) - min_train_days - gap_days) // val_days
    if n_folds < 1:
        raise ValueError("Not enough days for one walk-forward fold")

    first_val = len(days) - n_folds * val_days
    splits = []
    for k in range(n_folds):
        v0 = first_val + k * val_days
        splits.append((days[:v0 - gap_days], days[v0:v0 + val_days]))
    return splits
