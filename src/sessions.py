"""
sessions.py  –  shared row selection for training, validation and live signals.

Features and labels are computed on the full grid (the opening minutes act as
warm-up). Rows are then selected with tradable_mask(), which excludes the first
drop_open_mins minutes of every session. The grid file itself is never trimmed.

Every NYSE session (half-days included) opens at market_open ET, so only the
ET wall clock is needed, not the exchange calendar.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
ET = "America/New_York"


def minutes_since_open(index: pd.DatetimeIndex) -> np.ndarray:
    """ET wall-clock minutes since market_open for each timestamp (UTC index)."""
    local = index.tz_convert(ET)
    open_h, open_m = (int(x) for x in CFG["market_open"].split(":"))
    return np.asarray(local.hour * 60 + local.minute - (open_h * 60 + open_m))


def tradable_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for rows that may be used for training, validation or live signals."""
    return minutes_since_open(index) >= CFG["drop_open_mins"]
