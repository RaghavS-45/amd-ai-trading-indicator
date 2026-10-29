"""
build_grid.py
Rebuilds the raw Alpaca Parquet onto a complete 09:30 – 15:59 ET minute
grid for every NYSE trading day (including half-day sessions).

Missing minutes are forward-filled:
    open = high = low = close = previous close,  volume = 0
An `is_filled` boolean column marks synthetic bars.

Usage:
    python src/build_grid.py
"""

from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
ET = "America/New_York"


# ── Grid construction ──────────────────────────────────────────────

def _build_session_grid(cal, start: str, end: str) -> pd.DatetimeIndex:
    """
    Return a DatetimeIndex (UTC) covering every 1-min bar of every
    NYSE session in [start, end].  Half-day sessions automatically
    produce shorter grids (e.g., 09:30–12:59 instead of 09:30–15:59).
    """
    sessions = cal.sessions_in_range(start, end)
    grids = []
    for session in sessions:
        open_utc = cal.session_open(session)
        close_utc = cal.session_close(session)
        # Last bar = one minute before the official close
        # (close marks the END of the session, not a bar open)
        last_bar_utc = close_utc - pd.Timedelta(minutes=1)
        grids.append(pd.date_range(open_utc, last_bar_utc, freq="1min"))
    return pd.DatetimeIndex(
        [ts for g in grids for ts in g], name="timestamp"
    )


def build_grid(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex df_raw onto the complete minute grid and forward-fill gaps.

    Parameters
    ----------
    df_raw : DataFrame with UTC DatetimeIndex and at least
             columns [open, high, low, close, volume].

    Returns
    -------
    DataFrame on the complete grid, with is_filled column.
    """
    cal = xcals.get_calendar("XNYS")
    full_grid = _build_session_grid(
        cal, CFG["train_start"], CFG["test_end"]
    )

    df_raw = df_raw.copy()
    df_raw.index = pd.to_datetime(df_raw.index, utc=True)

    # Reindex: rows with no Alpaca data become all-NaN
    df = df_raw.reindex(full_grid)

    # Record which bars are synthetic BEFORE forward-filling
    is_filled = df["close"].isna()

    # Forward-fill close price across the whole grid
    close_ff = df["close"].ffill()

    # For filled bars: flat candle at previous close, zero volume
    df["close"] = close_ff
    df["open"] = df["open"].where(~is_filled, close_ff)
    df["high"] = df["high"].where(~is_filled, close_ff)
    df["low"]  = df["low"].where(~is_filled, close_ff)
    df["volume"] = df["volume"].fillna(0.0)
    df["volume"] = df["volume"].where(~is_filled, 0.0)

    # Optional Alpaca columns
    if "vwap" in df.columns:
        df["vwap"] = df["vwap"].where(~is_filled, close_ff)
    if "trade_count" in df.columns:
        df["trade_count"] = df["trade_count"].fillna(0).where(~is_filled, 0)

    df["is_filled"] = is_filled.astype(bool)
    return df


# ── Reporting helpers ──────────────────────────────────────────────

def _half_day_sessions(cal, start: str, end: str) -> pd.DatetimeIndex:
    """Return sessions whose NYSE close is before 16:00 ET."""
    sessions = cal.sessions_in_range(start, end)
    half = []
    for session in sessions:
        close_et = cal.session_close(session).tz_convert(ET)
        if close_et.hour < 16:
            half.append(session)
    return pd.DatetimeIndex(half)


def coverage_report(df_raw: pd.DataFrame, df_grid: pd.DataFrame,
                    cal, start: str, end: str) -> None:
    """Print missing-bar counts by hour and top-10 worst days."""
    missing = df_grid[df_grid["is_filled"]].copy()
    missing_et = missing.index.tz_convert(ET)

    # ── by hour of day ─────────────────────────────────────────────
    by_hour = (
        pd.Series(missing_et.hour, name="hour")
        .value_counts()
        .sort_index()
        .rename("missing_bars")
    )
    total_by_hour = (
        pd.Series(df_grid.index.tz_convert(ET).hour, name="hour")
        .value_counts()
        .sort_index()
        .rename("total_bars")
    )
    hour_df = pd.concat([total_by_hour, by_hour], axis=1).fillna(0)
    hour_df["fill_pct"] = 100 * hour_df["missing_bars"] / hour_df["total_bars"]

    print("\n── Missing bars by hour (ET) ────────────────────────────")
    print(hour_df.to_string(float_format="%.1f"))

    # ── top 10 worst days ──────────────────────────────────────────
    by_day = (
        pd.Series(missing_et.date, name="date")
        .value_counts()
        .head(10)
        .rename("missing_bars")
    )
    print("\n── Top 10 worst days (most missing bars) ────────────────")
    print(by_day.to_string())

    # ── half-day sessions ──────────────────────────────────────────
    half_days = _half_day_sessions(cal, start, end)
    print(f"\n── Half-day sessions in range: {len(half_days)} ─────────────────")
    for s in half_days:
        c = cal.session_close(s).tz_convert(ET)
        print(f"  {s.date()}  (closes {c.strftime('%H:%M')} ET)")

    # ── overall summary ────────────────────────────────────────────
    total = len(df_grid)
    n_filled = int(df_grid["is_filled"].sum())
    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Grid bars total   : {total:,}")
    print(f"  Real (IEX) bars   : {total - n_filled:,}")
    print(f"  Synthetic (filled): {n_filled:,}  ({100*n_filled/total:.2f}%)")
    print("─────────────────────────────────────────────────────────")


# ── Entry point ────────────────────────────────────────────────────

def main():
    raw_path  = ROOT / CFG["raw_data_path"]
    grid_path = ROOT / CFG["grid_data_path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}\nRun src/download_data.py first."
        )

    df_raw = pd.read_parquet(raw_path)
    print(f"Raw rows loaded : {len(df_raw):,}")

    cal = xcals.get_calendar("XNYS")
    df_grid = build_grid(df_raw)
    print(f"Grid rows built : {len(df_grid):,}")

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    df_grid.to_parquet(grid_path)
    print(f"Saved → {grid_path}")

    coverage_report(
        df_raw, df_grid, cal, CFG["train_start"], CFG["test_end"]
    )


if __name__ == "__main__":
    main()
