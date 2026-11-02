"""
download_peers.py  –  1-minute IEX bars of the peer stocks, train period only.
A peer whose raw file already exists (NVDA, also a context symbol) is left alone.

Usage: python src/download_peers.py
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def main():
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    client = StockHistoricalDataClient(key, secret)
    for sym in CFG["peer_symbols"]:
        out = ROOT / CFG["raw_data_template"].format(sym=sym.lower())
        if out.exists():
            print(f"{sym}: exists, kept")
            continue
        request = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=pd.Timestamp(CFG["train_start"], tz="UTC").to_pydatetime(),
                                   end=pd.Timestamp(CFG["peers_end"], tz="UTC").to_pydatetime(),
                                   feed="iex", adjustment="split")
        df = client.get_stock_bars(request).df.xs(sym, level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        df.to_parquet(out)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        (ROOT / CFG["checksum_template"].format(sym=sym.lower())).write_text(sha + "\n")
        print(f"{sym}: {len(df):,} rows, {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()
