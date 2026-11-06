"""
check_live_parity.py  –  the live engine must compute the same features as training.

Replays training days through the engine (bar by bar, as live) and compares its features with
the ones the training pipeline computes for the same days: the move-size features from the long
grid (volatility.load_vol_grid) and the direction-lean features from the main grid file.
Writes report/live_parity.csv. Training days only.

Usage: python src/check_live_parity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features import compute_features
from live_engine import CAL, ONE, Engine
from live_model import HAR, LEAN_SETS, MOVE_SETS, load_model
from live_sources import ReplaySource
from sessions import ET
from signal_store import Store
from volatility import load_vol_grid

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DAYS = ["2025-03-04", "2025-11-28", "2026-04-09", "2026-06-15"]   # includes a half-day


def live_features(day: str) -> pd.DataFrame:
    clock = {"t": None}
    eng = Engine(ReplaySource(), Store(":memory:"), load_model(), lambda: clock["t"])
    eng.open_session(day)
    last = None
    for upto in pd.date_range(eng.open_utc, eng.last_bar, freq="1min"):
        clock["t"] = upto + ONE
        eng.step(upto)
        f = compute_features(eng.frame, MOVE_SETS + LEAN_SETS)
        last = f if last is None else last
        last.loc[upto] = f.loc[upto]          # keep each row as it was when its bar closed
    return last


def main():
    vol_grid = load_vol_grid()
    main_grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    lean_cols = load_model()["lean"]["columns"]
    rows = []
    for day in DAYS:
        live = live_features(day)
        pick = lambda g: g[g.index.tz_convert(ET).date == pd.Timestamp(day).date()]
        train_move = compute_features(pick(vol_grid), MOVE_SETS)
        train_lean = compute_features(pick(main_grid), LEAN_SETS)
        for name, train, cols in [("move", train_move, HAR + ["news_count_30"]), ("lean", train_lean, lean_cols)]:
            a, b = live[cols], train[cols].reindex(live.index)
            both = a.notna() & b.notna()
            diff = (a - b).abs().where(both)
            rows.append({"day": day, "part": name, "rows": int(both.all(axis=1).sum()),
                         "nan_mismatch": int((a.notna() != b.notna()).to_numpy().sum()),
                         "max_abs_diff": float(np.nanmax(diff.to_numpy())),
                         "worst_column": diff.max().idxmax()})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "report" / "live_parity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
