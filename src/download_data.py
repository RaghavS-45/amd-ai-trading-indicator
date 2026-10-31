"""
download_data.py
Downloads 1-minute bars from Alpaca (IEX feed) for the date range in
config.yaml, writes a Parquet file, and saves a SHA-256 checksum.
Default symbol is the asset (AMD); context symbols use --symbol.

Usage:
    python src/download_data.py                 # AMD
    python src/download_data.py --symbol SPY    # context symbol

Environment variables (never hard-coded):
    ALPACA_API_KEY
    ALPACA_SECRET_KEY
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")  # reads .env if it exists; no-op if absent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def load_credentials():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        sys.exit(
            "ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        )
    return api_key, secret_key


def build_timeframe(tf_str: str) -> TimeFrame:
    if tf_str == "1Min":
        return TimeFrame(1, TimeFrameUnit.Minute)
    if tf_str == "5Min":
        return TimeFrame(5, TimeFrameUnit.Minute)
    raise ValueError(f"Unsupported bar_timeframe: {tf_str}")


def raw_path(symbol: str) -> Path:
    return ROOT / CFG["raw_data_template"].format(sym=symbol.lower())


def checksum_path(symbol: str) -> Path:
    return ROOT / CFG["checksum_template"].format(sym=symbol.lower())


def download(api_key: str, secret_key: str, symbol: str) -> pd.DataFrame:
    client = StockHistoricalDataClient(api_key, secret_key)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=build_timeframe(CFG["bar_timeframe"]),
        start=CFG["train_start"],
        end=CFG["test_end"],
        feed="iex",          # free-plan feed
        adjustment="split",  # handle stock splits
    )

    print(f"Downloading {symbol} {CFG['bar_timeframe']} bars "
          f"{CFG['train_start']} → {CFG['test_end']} (IEX feed) ...")
    bars = client.get_stock_bars(request)
    df = bars.df  # MultiIndex: (symbol, timestamp)

    # Drop the symbol level – one symbol per request
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  Raw rows downloaded: {len(df):,}")
    return df


def save(df: pd.DataFrame, symbol: str) -> None:
    out_path = raw_path(symbol)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    # SHA-256 checksum so we can verify the dataset is unchanged later
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    checksum_path(symbol).write_text(sha + "\n")
    print(f"  Saved → {out_path}  ({len(df):,} rows)")
    print(f"  SHA-256: {sha}")


def coverage_report(df: pd.DataFrame) -> None:
    """
    Prints how many 1-min bars we received vs how many we should expect
    during regular market hours (09:30-16:00 ET, Mon-Fri).
    IEX only records bars when trades happen on that exchange,
    so coverage < 100% is expected and normal.
    """
    import pytz
    et = pytz.timezone("America/New_York")
    local = df.index.tz_convert(et)

    # Keep only regular-session minutes
    session = local[
        (local.time >= pd.Timestamp("09:30").time())
        & (local.time < pd.Timestamp("16:00").time())
        & (local.weekday < 5)
    ]

    # Count unique trading days in the download
    trading_days = session.normalize().nunique()

    bars_per_day_expected = 390  # 09:30–16:00 = 390 minutes
    if CFG["bar_timeframe"] == "5Min":
        bars_per_day_expected = 78

    expected_total = trading_days * bars_per_day_expected
    received = len(session)
    pct = 100.0 * received / expected_total if expected_total else 0

    print("\n── Coverage Report ──────────────────────────────────")
    print(f"  Trading days in range : {trading_days}")
    print(f"  Expected bars         : {expected_total:,}  "
          f"({bars_per_day_expected}/day)")
    print(f"  Received bars (IEX)   : {received:,}")
    print(f"  Coverage              : {pct:.1f}%")
    print("─────────────────────────────────────────────────────")

    if pct < 60:
        print(
            "\n⚠  Coverage below 60%. Consider switching bar_timeframe to '5Min'\n"
            "   in config.yaml and re-running. 5-min bars aggregate more trades\n"
            "   so coverage improves, but the shortest label horizon must then\n"
            "   be ≥10 bars (50 min). Re-run and re-check before proceeding."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=CFG["asset"])
    symbol = parser.parse_args().symbol.upper()
    api_key, secret_key = load_credentials()
    df = download(api_key, secret_key, symbol)
    save(df, symbol)
    coverage_report(df)
    print(f"\n{symbol} download complete. Review coverage numbers above before proceeding.")
