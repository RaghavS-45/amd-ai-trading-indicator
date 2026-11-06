"""
live_sources.py  –  where the indicator gets bars and news.

AlpacaSource: free Alpaca IEX bars and the free news API (keys from environment variables).
ReplaySource: the local raw files, for training-period days only (never after train_end), so the
app can be shown and tested while the market is closed without touching the test period.
Both return bars as {symbol: DataFrame[open, high, low, close, volume, trade_count, vwap]} with a
UTC index of bar open times, and news as DataFrame[id, created_at, updated_at, headline, source].
"""

import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
BAR_FIELDS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
NEWS_FIELDS = ["id", "created_at", "updated_at", "headline", "source"]


def _between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


class AlpacaSource:
    def __init__(self):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.historical.news import NewsClient
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables.")
        self.bars_client = StockHistoricalDataClient(key, secret)
        self.news_client = NewsClient(key, secret)

    def bars(self, symbols, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        """Bars opening in [start, end)."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        req = StockBarsRequest(symbol_or_symbols=list(symbols), timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                               start=start.to_pydatetime(), end=end.to_pydatetime(), feed="iex",
                               adjustment="split", limit=None)
        df = self.bars_client.get_stock_bars(req).df
        out = {}
        for sym in symbols:
            if len(df) and sym in df.index.get_level_values("symbol"):
                part = df.xs(sym, level="symbol")[BAR_FIELDS].copy()
                part.index = pd.to_datetime(part.index, utc=True)
                out[sym] = _between(part.sort_index(), start, end)
            else:
                out[sym] = pd.DataFrame(columns=BAR_FIELDS, index=pd.DatetimeIndex([], tz="UTC"))
        return out

    def news(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        from alpaca.data.requests import NewsRequest
        req = NewsRequest(symbols=CFG["asset"], include_content=False, limit=None,
                          start=start.to_pydatetime(), end=end.to_pydatetime())
        articles = self.news_client.get_news(req).data["news"]
        return pd.DataFrame([{"id": a.id, "created_at": a.created_at, "updated_at": a.updated_at,
                              "headline": a.headline, "source": a.source} for a in articles],
                            columns=NEWS_FIELDS)


class ReplaySource:
    def __init__(self):
        from long_history import _hist_raw, _main_raw, joined_raw
        from volatility import load_news_all
        self.limit = pd.Timestamp(CFG["train_end"], tz="America/New_York") + pd.Timedelta(days=1)
        self._raw = {}
        self._loader = lambda sym: joined_raw(_hist_raw(sym), _main_raw(sym))
        news = load_news_all()
        for col in ("created_at", "updated_at"):
            news[col] = pd.to_datetime(news[col], utc=True)
        self._news = news[NEWS_FIELDS]

    def _check(self, end: pd.Timestamp):
        if end > self.limit:
            raise ValueError(f"Replay is limited to training days (up to {CFG['train_end']}); "
                             "later days belong to the hold-out test.")

    def bars(self, symbols, start: pd.Timestamp, end: pd.Timestamp) -> dict:
        self._check(end)
        out = {}
        for sym in symbols:
            if sym not in self._raw:
                raw = self._loader(sym)
                raw.index = pd.to_datetime(raw.index, utc=True)
                self._raw[sym] = raw[BAR_FIELDS]
            out[sym] = _between(self._raw[sym], start, end)
        return out

    def news(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        self._check(end)
        n = self._news
        return n[(n["created_at"] >= start) & (n["created_at"] < end)].reset_index(drop=True)
