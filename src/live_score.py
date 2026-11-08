"""
live_score.py  –  the live scorecard, scored the way the brief grades it: accuracy of confident
signals and paper profit after costs, for the indicator and two simple baselines (always Up, and
the Phase 4 RSI rule: Up if RSI < 30, Down if RSI > 70). One trade at a time, entered at the signal
bar's close and closed 10 minutes later (tradable.non_overlapping), cost as in config.yaml.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tradable import COST, non_overlapping

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
LEAN_SIGNAL = {"weak up": 1, "weak down": -1}


def lean_signal(lean: pd.Series) -> np.ndarray:
    return lean.map(LEAN_SIGNAL).fillna(0).astype(int).to_numpy()


def rsi_signal(rsi: pd.Series) -> np.ndarray:
    r = rsi.to_numpy(dtype=float)
    return np.where(r < 30, 1, np.where(r > 70, -1, 0))


def score_one(index: pd.DatetimeIndex, signal: np.ndarray, close: np.ndarray, after: np.ndarray) -> dict:
    ret = np.log(after / close)
    on = signal != 0
    hit = np.sign(ret[on]) == signal[on]
    rows = non_overlapping(index, signal)
    shares = CFG["shares_per_trade"]
    pnl = shares * (signal[rows] * (after[rows] - close[rows]) - COST)
    net_bps = (signal[rows] * ret[rows] - COST / close[rows]) * 1e4
    return {"signals": int(on.sum()), "correct": int(hit.sum()),
            "accuracy": float(hit.mean()) if on.any() else None,
            "trades": int(len(rows)), "wins": int((pnl > 0).sum()),
            "pnl_usd": float(pnl.sum()), "bps_per_trade": float(net_bps.mean()) if len(rows) else None}


def scorecard(scored: pd.DataFrame) -> dict:
    """scored: one row per on-time signal with an outcome; columns ts, close, close_after, lean, rsi."""
    if not len(scored):
        return {}
    scored = scored.sort_values("ts")
    idx = pd.DatetimeIndex(pd.to_datetime(scored["ts"], utc=True))
    close, after = scored["close"].to_numpy(float), scored["close_after"].to_numpy(float)
    signals = {"indicator": lean_signal(scored["lean"]),
               "always_up": np.ones(len(scored), dtype=int),
               "rsi_rule": rsi_signal(scored["rsi"])}
    return {k: score_one(idx, s, close, after) for k, s in signals.items()}
