"""
test_walkforward.py
Walk-forward splits: no overlap, gap respected, chronological order, and the
hold-out test period is refused.

Run:
    python -m pytest tests/test_walkforward.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from walkforward import walk_forward_splits

DAYS = list(pd.bdate_range("2024-11-01", periods=412).date)
LAST = str(DAYS[-1])


@pytest.fixture(params=[(120, 30, 1), (120, 30, 3), (50, 10, 1), (100, 25, 0)])
def splits(request):
    min_train, val, gap = request.param
    return walk_forward_splits(DAYS, min_train, val, gap, last_allowed=LAST), min_train, val, gap


def test_train_and_val_never_overlap(splits):
    folds, *_ = splits
    for train, val in folds:
        assert not set(train) & set(val)


def test_gap_respected(splits):
    folds, _, _, gap = splits
    pos = {d: i for i, d in enumerate(DAYS)}
    for train, val in folds:
        assert pos[val[0]] - pos[train[-1]] - 1 >= gap
        skipped = DAYS[pos[train[-1]] + 1:pos[val[0]]]
        assert not set(skipped) & (set(train) | set(val))


def test_chronological(splits):
    folds, *_ = splits
    for train, val in folds:
        assert train == sorted(train) and val == sorted(val)
        assert max(train) < min(val)
    for (_, v1), (_, v2) in zip(folds, folds[1:]):
        assert max(v1) < min(v2)


def test_validation_windows_tile_the_tail(splits):
    folds, _, val_days, _ = splits
    vals = [d for _, v in folds for d in v]
    assert len(vals) == len(set(vals)) == len(folds) * val_days
    assert vals == DAYS[-len(vals):]
    assert all(len(v) == val_days for _, v in folds)


def test_expanding_train_and_minimum_size(splits):
    folds, min_train, _, _ = splits
    assert len(folds[0][0]) >= min_train
    assert all(t[0] == DAYS[0] for t, _ in folds)
    sizes = [len(t) for t, _ in folds]
    assert sizes == sorted(sizes)


def test_dates_after_train_end_refused():
    with pytest.raises(ValueError):
        walk_forward_splits(DAYS, 120, 30, 1, last_allowed=str(DAYS[-2]))


def test_default_config_layout():
    folds = walk_forward_splits(DAYS, 120, 30, 1, last_allowed=LAST)
    assert len(folds) == 9
    assert len(folds[0][0]) == 141
