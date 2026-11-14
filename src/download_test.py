"""
download_test.py  –  the two files the final test (Phase 6) needs beyond the IEX bars already in
the raw files: AMD SIP bars (targets and scoring only) and AMD news (V4's news count only), for
test_start to test_end. Existing files are kept, never overwritten. Looks at no results.

Usage: python src/download_test.py
"""

import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / CFG["news_checksum_template"].format(name=path.stem)).write_text(sha + "\n")


def main():
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    start = pd.Timestamp(CFG["test_start"], tz="UTC").to_pydatetime()
    end = (pd.Timestamp(CFG["test_end"], tz="UTC") + pd.Timedelta(days=1)).to_pydatetime()   # whole last day

    sip_path = ROOT / CFG["test_sip_path"]
    if sip_path.exists():
        print("SIP bars: exists, kept")
    else:
        request = StockBarsRequest(symbol_or_symbols=CFG["asset"], timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                                   start=start, end=end, feed="sip", adjustment="split")
        df = StockHistoricalDataClient(key, secret).get_stock_bars(request).df.xs(CFG["asset"], level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        save(df, sip_path)
        print(f"SIP bars: {len(df):,} rows, {df.index.min()} to {df.index.max()}")

    news_path = ROOT / CFG["test_news_path"]
    if news_path.exists():
        print("News: exists, kept")
    else:
        request = NewsRequest(symbols=CFG["asset"], include_content=False, limit=None, start=start, end=end)
        articles = NewsClient(key, secret).get_news(request).data["news"]
        news = pd.DataFrame([{"id": a.id, "created_at": a.created_at, "updated_at": a.updated_at,
                              "headline": a.headline, "source": a.source, "n_symbols": len(a.symbols)}
                             for a in articles]).sort_values(["created_at", "id"]).reset_index(drop=True)
        save(news, news_path)
        print(f"News: {len(news):,} articles, {news['created_at'].min()} to {news['created_at'].max()}")


if __name__ == "__main__":
    main()
