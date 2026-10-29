"""
labels.py  –  time-based 3-class target label.

KEY DESIGN:  the label for bar at time t is computed by looking up the
price at  t + horizon_minutes  using an explicit index lookup on the
complete minute grid.  We never use row-offset shifts (shift(-N)) because
those are only safe when every consecutive row is exactly 1 minute apart.
The complete grid guarantees that, but explicit time lookup is safer and
more auditable.

Classes:
  1  = Up      :  fwd_return > +threshold
 -1  = Down    :  fwd_return < -threshold
  0  = Neutral :  |fwd_return| <= threshold

Label is NaN when:
  - The target timestamp is outside the same session
    (e.g., last 10 bars of a session where t+horizon falls into next day)
  - The target timestamp is in the next session (crosses overnight gap)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def make_labels(df: pd.DataFrame) -> pd.Series:
    """
    Input : complete-grid DataFrame (output of build_grid.py) with
            'close' column and UTC DatetimeIndex.
    Output: nullable Int8 Series with values {-1, 0, 1, <NA>}.

    Only uses time-based lookup; never reads future rows directly.
    """
    horizon_min = CFG["horizon_bars"]   # horizon in minutes
    threshold   = CFG["move_threshold"]

    # ── 1. Compute target timestamps ──────────────────────────────
    target_times = df.index + pd.Timedelta(minutes=horizon_min)

    # ── 2. Look up close prices at those timestamps ────────────────
    # reindex returns NaN for any target time not present in df
    # (i.e., outside the session or beyond end of data)
    future_close = df["close"].reindex(target_times)
    future_close.index = df.index   # restore source index

    # ── 3. Null if target crosses a session boundary ───────────────
    # Even though most cross-session targets produce NaN via reindex
    # (because t+horizon is outside the grid), we add an explicit check
    # for robustness (e.g., if a session grid happened to start exactly
    # at t+horizon on the next day).
    future_close = _null_cross_session(df, future_close, horizon_min)

    # ── 4. Forward return ─────────────────────────────────────────
    fwd_return = np.log(future_close / df["close"])

    # ── 5. Classify ───────────────────────────────────────────────
    label = pd.Series(0, index=df.index, dtype="int8")
    label[fwd_return >  threshold] =  1
    label[fwd_return < -threshold] = -1
    label[fwd_return.isna()] = pd.NA

    return label.astype("Int8")   # nullable integer supports <NA>


def _null_cross_session(df: pd.DataFrame,
                         future_close: pd.Series,
                         horizon_min: int) -> pd.Series:
    """Set future_close to NaN where target date ≠ source date (ET)."""
    import pytz
    et = pytz.timezone("America/New_York")

    source_dates = pd.Series(
        [t.date() for t in df.index.tz_convert(et)],
        index=df.index,
    )
    target_times = df.index + pd.Timedelta(minutes=horizon_min)
    target_dates = pd.Series(
        [t.date() for t in target_times.tz_convert(et)],
        index=df.index,
    )

    crosses = source_dates.values != target_dates.values
    future_close = future_close.copy()
    future_close[crosses] = np.nan
    return future_close
