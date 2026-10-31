"""
threshold.py  –  confidence-threshold signals and reliability statistics.

A thresholded signal takes the more likely of Up/Down and fires only if its
probability >= tau. For walk-forward honesty, tau for fold k is chosen on the
out-of-sample predictions of folds 1..k-1 only (choose_tau / walk_forward_tau).
"""

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import yaml

from evaluate import evaluate_signals

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())

TAU_GRID = np.round(np.arange(0.34, 0.701, 0.01), 2)


def threshold_signals(p_down: np.ndarray, p_up: np.ndarray, tau: float) -> np.ndarray:
    direction = np.where(p_up >= p_down, 1, -1)
    conf = np.maximum(p_up, p_down)
    return np.where(conf >= tau, direction, 0)


def tau_curve(oos: pd.DataFrame, taus: Sequence[float] = TAU_GRID) -> pd.DataFrame:
    """Metrics for each tau on the given out-of-sample rows (pooled)."""
    n_days = oos["date"].nunique()
    rows = []
    for tau in taus:
        sig = threshold_signals(oos["p_down"].to_numpy(), oos["p_up"].to_numpy(), tau)
        res = evaluate_signals(oos, sig, n_days) if (sig != 0).any() else {"signals": 0}
        res["tau"] = tau
        rows.append(res)
    return pd.DataFrame(rows).set_index("tau")


def choose_tau(oos: pd.DataFrame, min_signals_per_day: float,
               taus: Sequence[float] = TAU_GRID) -> float:
    """Best 1x bps/trade among taus with enough signals; never a silent tau."""
    curve = tau_curve(oos, taus)
    curve = curve[curve["signals"] > 0]
    ok = curve[curve["signals_per_day"] >= min_signals_per_day]
    if ok.empty:
        return float(curve["signals_per_day"].idxmax())
    return float(ok["bps_per_trade_1x"].idxmax())


def walk_forward_tau(oos: pd.DataFrame, min_signals_per_day: float) -> Dict[int, float]:
    """tau for each fold k >= 2, chosen on folds < k only."""
    folds = sorted(oos["fold"].unique())
    return {k: choose_tau(oos[oos["fold"] < k], min_signals_per_day) for k in folds[1:]}


def reliability(prob: np.ndarray, outcome: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": prob, "y": outcome})
    out = df.groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"), count=("y", "size"))
    out["lo"], out["hi"] = edges[out.index], edges[out.index + 1]
    return out


def expected_calibration_error(rel: pd.DataFrame) -> float:
    w = rel["count"] / rel["count"].sum()
    return float((w * (rel["mean_pred"] - rel["observed"]).abs()).sum())


def brier_multiclass(proba: np.ndarray, labels: np.ndarray, classes=(-1, 0, 1)) -> float:
    onehot = np.stack([(labels == c).astype(float) for c in classes], axis=1)
    return float(((proba - onehot) ** 2).sum(axis=1).mean())
