"""
test_evaluate.py
Trade simulation (non-overlapping, costs) and accuracy definitions.

Run:
    python -m pytest tests/test_evaluate.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluate import CFG, clustered_se, evaluate_signals, simulate_trades

H = CFG["horizon_bars"]
SHARES = CFG["shares_per_trade"]
RT_COST = SHARES * 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"])


def _data(n=30, fwd=0.001, close=100.0):
    idx = pd.date_range("2025-01-15 15:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "close": close,
        "fwd_ret": fwd,
        "label": 0,
        "date": idx.tz_convert("America/New_York").date,
    }, index=idx)


def test_signals_during_open_trade_are_skipped():
    data = _data()
    sig = np.ones(len(data), dtype=int)
    tr = simulate_trades(data, sig)
    assert list(tr.index) == list(data.index[::H])


def test_pnl_and_costs():
    data = _data(n=1, fwd=np.log(1.01))
    for k in (1.0, 3.0):
        up = simulate_trades(data, np.array([1]), cost_mult=k)
        down = simulate_trades(data, np.array([-1]), cost_mult=k)
        assert up["pnl"].iloc[0] == pytest.approx(SHARES * 1.0 - RT_COST * k)
        assert down["pnl"].iloc[0] == pytest.approx(-SHARES * 1.0 - RT_COST * k)


def test_zero_return_counts_as_wrong():
    data = _data(n=2, fwd=0.0)
    res = evaluate_signals(data, np.array([1, -1]), n_days=1)
    assert res["dir_acc"] == 0.0


def test_strict_accuracy_uses_label():
    data = _data(n=3, fwd=0.0005)
    data["label"] = [1, 0, -1]
    res = evaluate_signals(data, np.array([1, 1, 0]), n_days=1)
    assert res["dir_acc"] == 1.0
    assert res["strict_acc"] == 0.5
    assert res["signals"] == 2


def test_clustered_se_larger_with_correlated_days():
    correct = np.repeat([1.0, 0.0], 50)
    days = np.repeat([1, 2], 50)
    naive = np.sqrt(0.25 / 100)
    assert clustered_se(correct, days) > naive
