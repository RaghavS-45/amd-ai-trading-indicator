"""
tradable.py  –  the strict scoring for Phase 5e: what a live trade could actually have earned.
Enter at the next bar's open and exit at t + horizon, priced on SIP, one trade at a time.
Pass: direction accuracy >= base rate + 2 x day-clustered SE, and mean bps/trade > 0 at 1x cost.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import BPS, clustered_se

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
COST = 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"])


def non_overlapping(index: pd.DatetimeIndex, signal: np.ndarray) -> np.ndarray:
    horizon = np.timedelta64(CFG["horizon_bars"], "m")
    times = index.values
    rows, free_at = [], None
    for i in np.flatnonzero(signal != 0):
        if free_at is None or times[i] >= free_at:
            rows.append(i)
            free_at = times[i] + horizon
    return np.asarray(rows, dtype=int)


def score(x: pd.DataFrame) -> dict:
    """x has columns signal, trade_ret (tradable log return), close, date."""
    ok = x["trade_ret"].notna().to_numpy()
    x = x[ok]
    sig, ret = x["signal"].to_numpy(), x["trade_ret"].to_numpy()
    on = sig != 0
    hit = (np.sign(ret[on]) == sig[on]).astype(float)
    base = float((ret > 0).mean())
    acc, se = float(hit.mean()) if on.any() else np.nan, clustered_se(hit, x["date"].to_numpy()[on])
    rows = non_overlapping(x.index, sig)
    cost = COST / x["close"].to_numpy()[rows]
    net = (sig[rows] * ret[rows] - cost) * BPS
    net3 = (sig[rows] * ret[rows] - 3 * cost) * BPS
    bar = base + 2 * se
    bps = float(net.mean()) if len(rows) else np.nan
    return {"rows": len(x), "days": x["date"].nunique(), "signals_per_day": on.sum() / x["date"].nunique(),
            "up_share": float((sig[on] == 1).mean()) if on.any() else np.nan, "base_rate": base,
            "dir_acc": acc, "dir_se_day": se, "dir_bar": bar, "trades": len(rows), "bps_1x": bps,
            "bps_3x": float(net3.mean()) if len(rows) else np.nan, "pass": bool(acc >= bar and bps > 0)}
