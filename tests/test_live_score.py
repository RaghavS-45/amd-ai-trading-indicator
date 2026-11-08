"""
test_live_score.py
The live direction scorecard: accuracy counts only confident signals, trades never overlap,
costs are taken off every trade, and the RSI rule matches Phase 4.

Run:
    python -m pytest tests/test_live_score.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from live_score import CFG, rsi_signal, scorecard
from tradable import COST


def _rows():
    ts = pd.date_range("2025-03-14 14:40", periods=12, freq="1min", tz="UTC")
    close = np.full(12, 100.0)
    after = np.array([101, 99, 100.5, 100, 100, 100, 100, 100, 100, 100, 99, 100], dtype=float)
    lean = ["weak up", "weak down", "none", None, "none", "none", "none", "none", "none", "none",
            "weak down", "weak up"]
    rsi = [25, 50, 75, np.nan, 50, 50, 50, 50, 50, 50, 50, 50]
    return pd.DataFrame({"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "close": close, "close_after": after,
                         "lean": lean, "rsi": rsi})


def test_indicator_scores_only_confident_signals_without_overlap():
    r = scorecard(_rows())["indicator"]
    assert r["signals"] == 4 and r["correct"] == 3          # the last Up signal ends flat: wrong
    assert r["trades"] == 2                                # 14:40, then the next signal at/after 14:50
    shares = CFG["shares_per_trade"]
    assert r["pnl_usd"] == pytest.approx(shares * (1 - COST) + shares * (1 - COST))


def test_baselines():
    sc = scorecard(_rows())
    assert sc["always_up"]["signals"] == 12 and sc["always_up"]["trades"] == 2
    assert sc["rsi_rule"]["signals"] == 2 and sc["rsi_rule"]["correct"] == 1   # RSI 75 says Down, price rose
    assert list(rsi_signal(pd.Series([10, 30, 50, 70, 90, np.nan]))) == [1, 0, 0, 0, -1, 0]


def test_empty():
    assert scorecard(_rows().iloc[:0]) == {}
