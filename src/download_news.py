"""
download_news.py  –  AMD articles from the Alpaca News API (free, from Benzinga).
Writes news_raw_path and a checksum. Stops at news_end, so the test period stays out
until Phase 6.

Usage: python src/download_news.py              (train period)
       python src/download_news.py --history    (history_start to history_end, for counts)
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def main():
    history = argparse.ArgumentParser()
    history.add_argument("--history", action="store_true")
    history = history.parse_args().history
    start, end = (CFG["history_start"], CFG["history_end"]) if history else (CFG["train_start"], CFG["news_end"])
    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        sys.exit("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
    request = NewsRequest(symbols=CFG["asset"], include_content=False, limit=None,
                          start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
                          end=pd.Timestamp(end, tz="UTC").to_pydatetime())
    articles = NewsClient(key, secret).get_news(request).data["news"]
    news = pd.DataFrame([{"id": a.id, "created_at": a.created_at, "updated_at": a.updated_at,
                          "headline": a.headline, "source": a.source, "n_symbols": len(a.symbols)}
                         for a in articles])
    news = news.sort_values(["created_at", "id"]).reset_index(drop=True)
    out = ROOT / CFG["news_hist_raw_path" if history else "news_raw_path"]
    news.to_parquet(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / CFG["news_checksum_template"].format(name=out.stem)).write_text(sha + "\n")
    print(f"Saved {len(news):,} articles, {news['created_at'].min()} to {news['created_at'].max()}")


if __name__ == "__main__":
    main()
