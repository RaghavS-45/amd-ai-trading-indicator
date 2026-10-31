"""
evaluate.py  –  signal metrics shared by Phases 4-6.

A signal is +1 (Up), -1 (Down) or 0 (no signal) per row of a dataset built by
dataset.build_dataset() (columns fwd_ret, label, close, date; UTC index).

- direction accuracy: sign(fwd_ret) == signal; a zero return counts as wrong
- strict accuracy:    label == signal (move beyond move_threshold that way)
- trades:             one 10-minute trade per signal, entered at the signal
                      bar's close; signals while a trade is open are skipped
- standard errors:    day-clustered (main), naive binomial, binomial on trades
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
BPS = 1e4


def binomial_se(p: float, n: int) -> float:
    return float(np.sqrt(p * (1 - p) / n)) if n else np.nan


def clustered_se(correct: np.ndarray, groups: np.ndarray) -> float:
    """SE of a proportion when observations are correlated within groups (days)."""
    n = len(correct)
    if n == 0:
        return np.nan
    p = correct.mean()
    resid = pd.Series(correct - p).groupby(groups).sum().to_numpy()
    g = len(resid)
    if g < 2:
        return np.nan
    return float(np.sqrt(g / (g - 1) * (resid ** 2).sum()) / n)


def simulate_trades(data: pd.DataFrame, signal: np.ndarray, cost_mult: float = 1.0) -> pd.DataFrame:
    """Non-overlapping trades: entry at close t, exit at close t + horizon."""
    horizon = np.timedelta64(CFG["horizon_bars"], "m")
    times = data.index.values  # datetime64[ns] UTC
    rows = []
    free_at = None
    for i in np.flatnonzero(signal != 0):
        if free_at is not None and times[i] < free_at:
            continue
        rows.append(i)
        free_at = times[i] + horizon

    rows = np.asarray(rows, dtype=int)
    shares = CFG["shares_per_trade"]
    direction = signal[rows]
    entry = data["close"].to_numpy()[rows]
    exit_ = entry * np.exp(data["fwd_ret"].to_numpy()[rows])
    cost = shares * 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"]) * cost_mult
    pnl = shares * direction * (exit_ - entry) - cost
    return pd.DataFrame({
        "direction": direction,
        "entry": entry,
        "exit": exit_,
        "pnl": pnl,
        "pnl_bps": pnl / (shares * entry) * BPS,
        "date": data["date"].to_numpy()[rows],
    }, index=data.index[rows])


def evaluate_signals(data: pd.DataFrame, signal: np.ndarray, n_days: int) -> dict:
    signal = np.asarray(signal)
    on = signal != 0
    n = int(on.sum())
    dir_ok = (np.sign(data["fwd_ret"].to_numpy()[on]) == signal[on]).astype(float)
    strict_ok = (data["label"].to_numpy()[on] == signal[on]).astype(float)
    days = data["date"].to_numpy()[on]

    out = {
        "signals": n,
        "signals_per_day": n / n_days if n_days else np.nan,
        "up_share": float((signal[on] == 1).mean()) if n else np.nan,
        "dir_acc": dir_ok.mean() if n else np.nan,
        "dir_se_day": clustered_se(dir_ok, days),
        "dir_se_naive": binomial_se(dir_ok.mean(), n) if n else np.nan,
        "strict_acc": strict_ok.mean() if n else np.nan,
        "strict_se_day": clustered_se(strict_ok, days),
        "strict_se_naive": binomial_se(strict_ok.mean(), n) if n else np.nan,
    }

    for k, tag in [(1.0, "1x"), (3.0, "3x")]:
        tr = simulate_trades(data, signal, cost_mult=k)
        out[f"pnl_{tag}"] = float(tr["pnl"].sum())
        out[f"bps_per_trade_{tag}"] = float(tr["pnl_bps"].mean()) if len(tr) else np.nan
        out[f"win_rate_{tag}"] = float((tr["pnl"] > 0).mean()) if len(tr) else np.nan
        if tag == "1x":
            t_ok = (np.sign(tr["exit"] - tr["entry"]) == tr["direction"]).astype(float)
            out["trades"] = len(tr)
            out["trades_per_day"] = len(tr) / n_days if n_days else np.nan
            out["trade_dir_acc"] = float(t_ok.mean()) if len(tr) else np.nan
            out["trade_dir_se"] = binomial_se(out["trade_dir_acc"], len(tr)) if len(tr) else np.nan
    return out
