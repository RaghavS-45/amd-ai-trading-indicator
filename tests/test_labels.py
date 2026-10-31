"""
test_labels.py
make_labels() must look exactly horizon_bars minutes ahead by timestamp,
use filled bars like real ones, and be NA when the target leaves the session.

Run:
    python -m pytest tests/test_labels.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labels import CFG, make_labels

H = CFG["horizon_bars"]
THR = CFG["move_threshold"]
SESSIONS = [  # (date, last bar ET); 2024-11-29 is a half-day
    ("2024-11-29", "12:59"),
    ("2025-01-15", "15:59"),
    ("2025-01-16", "15:59"),
]
FILLED = slice(100, 107)  # positions in the first full session (after the half-day)


def _make_grid() -> pd.DataFrame:
    parts = []
    for day, last in SESSIONS:
        parts.append(pd.date_range(
            f"{day} 09:30", f"{day} {last}", freq="1min", tz="America/New_York"))
    idx = pd.DatetimeIndex([ts for p in parts for ts in p]).tz_convert("UTC")

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    df = pd.DataFrame({"close": close}, index=idx)
    df["is_filled"] = False

    # Flat filled run, as build_grid.py produces it
    start = 210 + FILLED.start
    stop = 210 + FILLED.stop
    df.iloc[start:stop, 0] = df["close"].iloc[start - 1]
    df.iloc[start:stop, 1] = True
    return df


def _session_bounds(df: pd.DataFrame):
    day = df.index.tz_convert("America/New_York").date
    pos = 0
    for d in pd.unique(day):
        n = int((day == d).sum())
        yield pos, pos + n
        pos += n


def _expected(df: pd.DataFrame) -> pd.Series:
    """Reference labels by row position inside each session (grid is complete)."""
    out = pd.Series(pd.NA, index=df.index, dtype="Int8")
    close = df["close"].to_numpy()
    for start, stop in _session_bounds(df):
        for i in range(start, stop - H):
            r = np.log(close[i + H] / close[i])
            out.iloc[i] = 1 if r > THR else (-1 if r < -THR else 0)
    return out


def test_labels_match_exact_horizon():
    df = _make_grid()
    lab = make_labels(df)
    pd.testing.assert_series_equal(lab, _expected(df), check_names=False)
    assert set(lab.dropna().unique()) == {-1, 0, 1}


def test_filled_target_uses_carried_close():
    df = _make_grid()
    lab = make_labels(df)
    t = 210 + FILLED.start + 3 - H  # target is inside the filled run
    assert df["is_filled"].iloc[t + H]
    r = np.log(df["close"].iloc[t + H] / df["close"].iloc[t])
    assert lab.iloc[t] == (1 if r > THR else (-1 if r < -THR else 0))


def test_lookup_is_by_time_not_row():
    df = _make_grid()
    k = 300
    dropped = df.drop(df.index[k])
    lab = make_labels(dropped)
    assert pd.isna(lab.loc[df.index[k - H]])
    assert not pd.isna(lab.loc[df.index[k - H - 1]])
    assert not pd.isna(lab.loc[df.index[k - H + 1]])


def test_na_at_session_end():
    df = _make_grid()
    lab = make_labels(df).to_numpy()
    for start, stop in _session_bounds(df):
        assert not pd.isna(lab[stop - H - 1])
        assert pd.isna(lab[stop - H:stop]).all()
        assert not pd.isna(lab[start:stop - H]).any()


def test_half_day_last_label_at_1249():
    df = _make_grid()
    lab = make_labels(df)
    et = lab.index.tz_convert("America/New_York")
    half = lab[et.date == pd.Timestamp("2024-11-29").date()]
    last_valid = half.dropna().index[-1].tz_convert("America/New_York")
    assert last_valid.strftime("%H:%M") == "12:49"


@pytest.mark.parametrize("t", [50, 250, 500])
def test_prices_beyond_horizon_do_not_matter(t):
    df = _make_grid()
    base = make_labels(df).iloc[t]
    changed = df.copy()
    changed.iloc[t + H + 1:, 0] *= 2
    assert make_labels(changed).iloc[t] == base


def test_output_dtype_and_values():
    lab = make_labels(_make_grid())
    assert str(lab.dtype) == "Int8"
    assert set(lab.dropna().unique()) <= {-1, 0, 1}
