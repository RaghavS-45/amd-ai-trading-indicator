"""
test_sessions.py
tradable_mask() must exclude exactly the first drop_open_mins minutes of each
session, in both EST and EDT.

Run:
    python -m pytest tests/test_sessions.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sessions import CFG, tradable_mask


@pytest.mark.parametrize("day", ["2025-01-15", "2025-07-15"])  # EST, EDT
def test_first_minutes_excluded(day):
    n = CFG["drop_open_mins"]
    idx = pd.date_range(
        f"{day} 09:30", periods=n + 3, freq="1min", tz="America/New_York"
    ).tz_convert("UTC")
    mask = tradable_mask(idx)

    assert len(mask) == len(idx)
    assert not mask[:n].any()
    assert mask[n:].all()


def test_later_session_minutes_included():
    idx = pd.DatetimeIndex(
        ["2025-01-15 12:00", "2025-01-15 15:59"], tz="America/New_York"
    ).tz_convert("UTC")
    assert tradable_mask(idx).all()
