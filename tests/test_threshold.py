"""
test_threshold.py
Confidence threshold: chosen on earlier folds only, respects the minimum
signal rate, never picks a silent threshold.

Run:
    python -m pytest tests/test_threshold.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import threshold
from threshold import choose_tau, threshold_signals, walk_forward_tau


def _oos(folds=4, days_per_fold=5, rows_per_day=60, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    day0 = pd.Timestamp("2025-03-03")
    for f in range(1, folds + 1):
        for d in range(days_per_fold):
            day = (day0 + pd.offsets.BDay((f - 1) * days_per_fold + d)).date()
            idx = pd.date_range(f"{day} 15:00", periods=rows_per_day, freq="1min", tz="UTC")
            p_up = rng.uniform(0.2, 0.6, rows_per_day)
            p_down = rng.uniform(0.2, 0.6, rows_per_day)
            fwd = rng.normal(0, 0.003, rows_per_day)
            parts.append(pd.DataFrame({
                "p_up": p_up, "p_down": p_down, "p_neutral": 1 - p_up,
                "fwd_ret": fwd, "label": np.sign(fwd).astype(int),
                "close": 100.0, "date": day, "fold": f}, index=idx))
    return pd.concat(parts)


def test_threshold_signals():
    sig = threshold_signals(np.array([0.5, 0.2, 0.3]), np.array([0.3, 0.6, 0.3]), 0.45)
    assert list(sig) == [-1, 1, 0]


def test_tau_for_fold_k_uses_only_earlier_folds(monkeypatch):
    oos = _oos()
    seen = []
    real = threshold.choose_tau

    def spy(data, *a, **k):
        seen.append(set(data["fold"]))
        return real(data, *a, **k)

    monkeypatch.setattr(threshold, "choose_tau", spy)
    taus = walk_forward_tau(oos, min_signals_per_day=5)
    assert sorted(taus) == [2, 3, 4]
    assert seen == [{1}, {1, 2}, {1, 2, 3}]


def test_fold_k_data_cannot_change_its_own_tau():
    oos = _oos()
    base = walk_forward_tau(oos, 5)
    changed = oos.copy()
    changed.loc[changed["fold"] == 3, "fwd_ret"] *= -1
    assert walk_forward_tau(changed, 5)[3] == base[3]


def test_min_signals_per_day_respected():
    oos = _oos()
    n_days = oos["date"].nunique()
    for min_spd in (5, 20, 40):
        tau = choose_tau(oos, min_spd)
        sig = threshold_signals(oos["p_down"].to_numpy(), oos["p_up"].to_numpy(), tau)
        assert (sig != 0).sum() / n_days >= min_spd


def test_never_silent_when_constraint_impossible():
    oos = _oos()
    tau = choose_tau(oos, min_signals_per_day=10_000)
    sig = threshold_signals(oos["p_down"].to_numpy(), oos["p_up"].to_numpy(), tau)
    assert (sig != 0).any()
