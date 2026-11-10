"""
signal_store.py  –  SQLite store for the indicator. Signals, outcomes and the bars they were
computed from are written once (INSERT OR IGNORE) and never changed: triggers refuse UPDATE and
DELETE, so a stored signal cannot repaint.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

SIGNAL_COLS = ["ts", "session", "created_at", "late", "close", "forecast_rv", "typical_bps", "band_bps",
               "typical_lo", "typical_hi", "band_lo", "band_hi", "activity", "level", "warning",
               "p_up", "p_flat", "p_down", "lean", "news_30", "qqq_ret_10", "nvda_ret_10",
               "filled_30", "model_id"]
OUTCOME_COLS = ["ts", "session", "target_ts", "close_after", "ret_bps", "in_typical", "in_band"]
BAR_COLS = ["symbol", "ts", "session", "open", "high", "low", "close", "volume", "trade_count", "vwap"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  ts TEXT PRIMARY KEY, session TEXT, created_at TEXT, late INTEGER, close REAL, forecast_rv REAL,
  typical_bps REAL, band_bps REAL, typical_lo REAL, typical_hi REAL, band_lo REAL, band_hi REAL,
  activity REAL, level TEXT, warning INTEGER, p_up REAL, p_flat REAL, p_down REAL, lean TEXT,
  news_30 REAL, qqq_ret_10 REAL, nvda_ret_10 REAL, filled_30 REAL, model_id TEXT);
CREATE TABLE IF NOT EXISTS outcomes (
  ts TEXT PRIMARY KEY, session TEXT, target_ts TEXT, close_after REAL, ret_bps REAL,
  in_typical INTEGER, in_band INTEGER);
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT, ts TEXT, session TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
  trade_count REAL, vwap REAL, PRIMARY KEY (symbol, ts));
"""
FROZEN = "".join(
    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{t} rows are never modified'); END;\n"
    for t in ("signals", "outcomes", "bars") for op in ("UPDATE", "DELETE"))


class Store:
    def __init__(self, path):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(path), check_same_thread=False)
        self.con.executescript(SCHEMA + FROZEN)

    def _insert(self, table: str, cols: list, rows: list) -> int:
        if not rows:
            return 0
        sql = f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        before = self.con.total_changes
        with self.con:
            self.con.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
        return self.con.total_changes - before

    def add_signals(self, rows: list) -> int:
        return self._insert("signals", SIGNAL_COLS, rows)

    def add_outcomes(self, rows: list) -> int:
        return self._insert("outcomes", OUTCOME_COLS, rows)

    def add_bars(self, rows: list) -> int:
        return self._insert("bars", BAR_COLS, rows)

    def _read(self, table: str, session: Optional[str]) -> pd.DataFrame:
        sql, args = f"SELECT * FROM {table}", ()
        if session is not None:
            sql, args = sql + " WHERE session = ?", (session,)
        return pd.read_sql_query(sql + " ORDER BY ts", self.con, params=args)

    def signals(self, session: Optional[str] = None) -> pd.DataFrame:
        return self._read("signals", session)

    def outcomes(self, session: Optional[str] = None) -> pd.DataFrame:
        return self._read("outcomes", session)

    def live_sessions(self) -> list:
        rows = self.con.execute("SELECT DISTINCT session FROM signals WHERE late = 0").fetchall()
        return sorted(r[0] for r in rows)

    def bars(self, session: str) -> pd.DataFrame:
        return self._read("bars", session)
