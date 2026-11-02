"""
download_history.py  –  1-minute bars from before the main dataset, in separate files.
Files that already exist are kept, so an interrupted run can be restarted.

Usage: python src/download_history.py --symbols AMD SPY QQQ NVDA --feed iex
"""

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    args = ap.parse_args()
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    client = StockHistoricalDataClient(key, secret)
    for sym in args.symbols:
        names = dict(sym=sym.lower(), feed=args.feed)
        out = ROOT / CFG["history_template"].format(**names)
        if out.exists():
            print(f"{sym}: exists, kept", flush=True)
            continue
        request = StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=pd.Timestamp(CFG["history_start"], tz="UTC").to_pydatetime(),
                                   end=pd.Timestamp(CFG["history_end"], tz="UTC").to_pydatetime(),
                                   feed=args.feed, adjustment="split")
        df = client.get_stock_bars(request).df.xs(sym, level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        df.to_parquet(out)
        sha = hashlib.sha256(out.read_bytes()).hexdigest()
        (ROOT / CFG["history_checksum_template"].format(**names)).write_text(sha + "\n")
        print(f"{sym} {args.feed}: {len(df):,} rows, {df.index.min()} to {df.index.max()}", flush=True)


if __name__ == "__main__":
    main()
