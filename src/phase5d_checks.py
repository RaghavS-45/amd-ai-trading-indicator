"""
phase5d_checks.py  –  are the Phase 5d passes tradable? Not pre-registered.
Re-prices each model's trades with a next-bar-open entry and on full-market (SIP) closes.
SIP covers train-period dates only; it is free for data over 15 minutes old, and is used here
only to check results, never as a model input.

Usage: python src/phase5d_checks.py
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

from evaluate import BPS, clustered_se, simulate_trades

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
MODELS = {"E5.6": "phase5_E5.6_oos", "E5b.4": "phase5b_E5b.4_oos", "E5b.6": "phase5b_E5b.6_oos",
          "E5d.3": "phase5d_E5d.3_oos", "E5d.4": "phase5d_E5d.4_oos"}
ZERO, ONE = pd.Timedelta(0), pd.Timedelta(minutes=1)
TEN = pd.Timedelta(minutes=CFG["horizon_bars"])


def load_sip() -> pd.DataFrame:
    path = ROOT / CFG["sip_check_path"]
    if not path.exists():
        client = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
        request = StockBarsRequest(symbol_or_symbols=CFG["asset"], timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=pd.Timestamp(CFG["train_start"], tz="UTC").to_pydatetime(),
                                   end=pd.Timestamp(CFG["peers_end"], tz="UTC").to_pydatetime(),
                                   feed="sip", adjustment="split")
        df = client.get_stock_bars(request).df.xs(CFG["asset"], level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        df.to_parquet(path)
        (ROOT / CFG["checksum_template"].format(sym=path.stem.split("_1min")[0] + "_sip")).write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return pd.read_parquet(path)


def at(col: pd.Series, idx: pd.DatetimeIndex, delta: pd.Timedelta) -> np.ndarray:
    return col.reindex(idx + delta).to_numpy()


def check(name: str, x: pd.DataFrame, grid: pd.DataFrame, sip: pd.DataFrame, scope: str) -> dict:
    trades = simulate_trades(x, x["signal"].to_numpy())
    i, d = trades.index, trades["direction"].to_numpy()
    cost = 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"]) / trades["entry"].to_numpy() * BPS
    end, sip_end = at(grid["close"], i, TEN), at(sip["close"], i, TEN)

    def net(exit_px, entry_px):
        return float(np.nanmean(d * np.log(exit_px / entry_px) * BPS - cost))

    on = x["signal"] != 0
    idx = x.index[on]
    sip_ret = np.log(at(sip["close"], idx, TEN) / at(sip["close"], idx, ZERO))
    ok = ~np.isnan(sip_ret)
    hit = (np.sign(sip_ret[ok]) == x["signal"][on].to_numpy()[ok]).astype(float)
    iex_hit = (np.sign(x["fwd_ret"][on]) == x["signal"][on]).astype(float)
    all_sip = np.log(at(sip["close"], x.index, TEN) / at(sip["close"], x.index, ZERO))
    return {"model": name, "folds": scope, "trades": len(trades),
            "iex_dir": iex_hit.mean(), "sip_dir": hit.mean(),
            "sip_dir_se": clustered_se(hit, x["date"][on].to_numpy()[ok]), "sip_up_rate": np.nanmean(all_sip > 0),
            "bps_iex_close_t": net(end, x.loc[i, "close"].to_numpy()),
            "bps_iex_open_t1": net(end, at(grid["open"], i, ONE)),
            "bps_iex_close_t1": net(end, at(grid["close"], i, ONE)),
            "bps_sip_close_t": net(sip_end, at(sip["close"], i, ZERO)),
            "bps_sip_open_t1": net(sip_end, at(sip["open"], i, ONE))}


def main():
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index < pd.Timestamp(CFG["peers_end"], tz="UTC")]
    sip = load_sip()
    first = CFG["confirm_from_fold"]
    rows = []
    for name, stem in MODELS.items():
        oos = pd.read_parquet(REPORT / f"{stem}.parquet")
        for scope, mask in [("6-9", oos["fold"] >= first), ("6-8", (oos["fold"] >= first) & (oos["fold"] < 9)),
                            ("1-5", oos["fold"] < first)]:
            rows.append(check(name, oos[mask], grid, sip, scope))
    table = pd.DataFrame(rows).set_index(["model", "folds"])
    table.to_csv(REPORT / "phase5d_checks.csv")
    pd.set_option("display.width", 220)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
