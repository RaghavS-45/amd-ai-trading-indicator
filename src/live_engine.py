"""
live_engine.py  –  the indicator's minute loop, shared by live mode and replay.

Each step finalizes the bars that opened up to `upto` (a minute is final live_bar_delay_seconds
after it closes; a bar that arrives later is ignored, so inputs never change). Missing minutes are
filled flat with build_grid, exactly as in training. Grid columns that need earlier sessions
(prev_day_rv, seas_rv, prev_close) come from the last live_history_sessions sessions, with sparse
days dropped as in training. Features come from features.compute_features on the current session
only. A signal is made for every tradable bar whose 10-minute target lies inside the session; it
is written once to SQLite and never changed. Outcomes are written once the target bar is final.
"""

import hashlib
from pathlib import Path
from typing import Callable, Optional

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import yaml

from build_grid import CONTEXT_COLS, build_grid, drop_sparse_days
from features import compute_features
from live_model import LEAN_SETS, MOVE_SETS, direction_lean, move_forecast
from live_score import scorecard
from long_history import previous_close
from news import first_bar
from sessions import ET, tradable_mask
from signal_store import Store
from volatility import prev_day_rv, seasonal_rv

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
ONE = pd.Timedelta(minutes=1)
H = pd.Timedelta(minutes=CFG["horizon_bars"])
ASSET = CFG["asset"]
SYMBOLS = [ASSET] + [s for s in CFG["context_symbols"] if s != ASSET]
CAL = xcals.get_calendar("XNYS")
LATE_AFTER = pd.Timedelta(seconds=CFG["live_bar_delay_seconds"] + 60)


def model_id() -> str:
    path = ROOT / CFG["live_info_path"]
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "unknown"


def _iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _num(x) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else float(x)


class Engine:
    def __init__(self, source, store: Store, model: dict, clock: Callable[[], pd.Timestamp]):
        self.source, self.store, self.model, self.clock = source, store, model, clock
        self.model_id = model_id()
        self.day = None
        self.review = False   # a past session recomputed after the close: score every signal

    # ── session setup ───────────────────────────────────────────────
    def open_session(self, day) -> None:
        day = pd.Timestamp(day)
        self.day, self.session = None, str(day.date())
        self.open_utc, self.close_utc = CAL.session_open(day), CAL.session_close(day)
        self.last_bar = self.close_utc - ONE
        past = CAL.sessions_window(day, -(CFG["live_history_sessions"] + 1))[:-1]
        self.first_day = past[0]
        self.history = self.source.bars(SYMBOLS, CAL.session_open(past[0]), self.open_utc)
        stored = self.store.bars(self.session)
        self.today = {}
        for sym in SYMBOLS:
            rows = stored[stored["symbol"] == sym].set_index("ts")
            rows.index = pd.to_datetime(rows.index, utc=True)
            self.today[sym] = rows.drop(columns=["symbol", "session"])
        self.final_upto = max([self.open_utc - ONE] + [d.index.max() for d in self.today.values() if len(d)])
        self.done_upto = self.open_utc - ONE
        self.news = {}
        self.frame = None
        self.feat = None
        self.day = day   # set last: a failed start is retried on the next tick

    # ── one step ────────────────────────────────────────────────────
    def step(self, upto: pd.Timestamp) -> int:
        """Finalize bars opening up to `upto`, then write new signals and outcomes."""
        upto = min(upto, self.last_bar)
        if upto > self.final_upto:
            new = self.source.bars(SYMBOLS, self.final_upto + ONE, upto + ONE)
            rows = []
            for sym, df in new.items():
                df = df[(df.index > self.final_upto) & (df.index <= upto)]
                self.today[sym] = pd.concat([self.today[sym], df]).sort_index() if len(self.today[sym]) else df
                rows += [dict(symbol=sym, ts=_iso(ts), session=self.session, **{k: float(v) for k, v in r.items()})
                         for ts, r in df.iterrows()]
            self.store.add_bars(rows)
            self.final_upto = upto
        self._update_news(self.final_upto)
        self.frame = self._frame(self.final_upto)
        n = self._emit(self.final_upto)
        self.done_upto = self.final_upto
        return n

    def _update_news(self, upto: pd.Timestamp) -> None:
        news = self.source.news(self.open_utc - pd.Timedelta(minutes=CFG["live_news_minutes"]), upto + ONE)
        if not len(news):
            return
        news = news.assign(created_at=pd.to_datetime(news["created_at"], utc=True),
                           updated_at=pd.to_datetime(news["updated_at"], utc=True))
        news["bar"] = first_bar(news)
        for _, a in news[news["bar"] <= upto + ONE].iterrows():
            if a["id"] not in self.news:
                self.news[a["id"]] = {"bar": a["bar"], "headline": a["headline"], "source": a["source"]}

    def _frame(self, upto: pd.Timestamp) -> pd.DataFrame:
        start, end = str(self.first_day.date()), self.session
        raw = {s: pd.concat([self.history[s], self.today[s]]) for s in SYMBOLS}
        grid = build_grid(raw[ASSET], start, end)
        grid = grid[(grid.index <= upto) & grid["close"].notna()]
        is_today = grid.index >= self.open_utc
        past, _ = drop_sparse_days(grid[~is_today], CFG["min_day_coverage"])
        grid = pd.concat([past, grid[is_today]])
        for sym in SYMBOLS[1:]:
            ctx = build_grid(raw[sym], start, end).reindex(grid.index)[CONTEXT_COLS]
            grid = grid.join(ctx.add_prefix(f"{sym.lower()}_"))
        grid["prev_day_rv"] = prev_day_rv(grid["close"])
        grid["seas_rv"] = seasonal_rv(grid["close"])
        grid["prev_close"] = previous_close(grid["close"])
        grid["qqq_prev_close"] = previous_close(grid["qqq_close"])
        named = pd.Series([a["headline"] for a in self.news.values()], dtype=object)
        named = named.str.lower().str.contains(CFG["news_named_pattern"], regex=True).to_numpy()
        bars = pd.Series([a["bar"] for a in self.news.values()], dtype="datetime64[ns, UTC]")[named]
        grid["news_count"] = bars.value_counts().reindex(grid.index).fillna(0.0).to_numpy()
        return grid[grid.index >= self.open_utc]

    def _emit(self, upto: pd.Timestamp) -> int:
        today = self.frame
        if not len(today):
            return 0
        feat = self.feat = compute_features(today, MOVE_SETS + LEAN_SETS)
        done = set(self.store.signals(self.session)["ts"])
        ok = tradable_mask(feat.index) & (feat.index + H <= self.last_bar)
        rows_idx = [ts for ts in feat.index[ok] if _iso(ts) not in done]
        move = move_forecast(self.model, feat.loc[rows_idx].dropna(subset=["lrv_10", "lrv_30", "lrv_sess",
                                                                            "lrv_prev_day", "lrv_seas"]))
        lean = direction_lean(self.model, feat.loc[move.index])
        now = self.clock()
        filled = today["is_filled"].rolling(30, min_periods=1).mean()
        rows = []
        for ts, m in move.iterrows():
            close = float(today.at[ts, "close"])
            ln = lean.loc[ts] if ts in lean.index else None
            rows.append({
                "ts": _iso(ts), "session": self.session, "created_at": _iso(now),
                "late": int(now > ts + ONE + LATE_AFTER), "close": close,
                "forecast_rv": float(m["forecast_rv"]),
                "typical_bps": float(m["typical_hw"] * 1e4), "band_bps": float(m["band_hw"] * 1e4),
                "typical_lo": close * np.exp(-m["typical_hw"]), "typical_hi": close * np.exp(m["typical_hw"]),
                "band_lo": close * np.exp(-m["band_hw"]), "band_hi": close * np.exp(m["band_hw"]),
                "activity": float(m["activity"]), "level": m["level"], "warning": int(m["warning"]),
                "p_up": None if ln is None else float(ln["p_up"]),
                "p_flat": None if ln is None else float(ln["p_flat"]),
                "p_down": None if ln is None else float(ln["p_down"]),
                "lean": None if ln is None else ln["lean"],
                "news_30": float(feat.at[ts, "news_count_30"]),
                "qqq_ret_10": _num(feat.at[ts, "qqq_ret_10"]), "nvda_ret_10": _num(feat.at[ts, "nvda_ret_10"]),
                "filled_30": float(filled.at[ts]), "model_id": self.model_id})
        added = self.store.add_signals(rows)
        self._outcomes(upto)
        return added

    def _outcomes(self, upto: pd.Timestamp) -> None:
        sig = self.store.signals(self.session)
        have = set(self.store.outcomes(self.session)["ts"])
        rows = []
        for _, s in sig.iterrows():
            ts = pd.Timestamp(s["ts"])
            target = ts + H
            if s["ts"] in have or target > upto:
                continue
            after = float(self.frame.at[target, "close"])
            ret = np.log(after / s["close"])
            rows.append({"ts": s["ts"], "session": self.session, "target_ts": _iso(target), "close_after": after,
                         "ret_bps": float(ret * 1e4), "in_typical": int(abs(ret) * 1e4 <= s["typical_bps"]),
                         "in_band": int(abs(ret) * 1e4 <= s["band_bps"])})
        self.store.add_outcomes(rows)

    # ── what the page shows ─────────────────────────────────────────
    def grade(self, sig: pd.DataFrame, out: pd.DataFrame):
        """Range record and direction scorecard for these signals."""
        scored = out[out["ts"].isin(set(sig["ts"]))]
        rsi = self.feat["rsi"].rename(index=_iso) if self.feat is not None else pd.Series(dtype=float)
        graded = scored.merge(sig[["ts", "close", "lean"]], on="ts").assign(rsi=lambda d: d["ts"].map(rsi))
        rec = {"scored": int(len(scored)), "in_band": int(scored["in_band"].sum()),
               "in_typical": int(scored["in_typical"].sum())}
        return rec, scorecard(graded)

    def state(self) -> dict:
        if self.frame is None:
            return {"session": None}
        bars = self.frame[["open", "high", "low", "close", "is_filled"]]
        sig = self.store.signals(self.session)
        out = self.store.outcomes(self.session)
        news = sorted(({"time": _iso(a["bar"] - ONE), "headline": a["headline"], "source": a["source"]}
                       for a in self.news.values() if a["bar"] >= self.open_utc), key=lambda a: a["time"])
        rec, card = self.grade(sig if self.review else sig[sig["late"] == 0], out)
        offset = self.open_utc.tz_convert(ET).utcoffset().total_seconds()
        return {
            "session": self.session, "open": _iso(self.open_utc), "close": _iso(self.close_utc),
            "et_offset_seconds": int(offset), "horizon_minutes": CFG["horizon_bars"],
            "last_bar": _iso(self.final_upto) if self.final_upto >= self.open_utc else None,
            "bars": [{"time": _iso(ts), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                      "filled": bool(r.is_filled)} for ts, r in bars.iterrows()],
            "signals": sig.replace({np.nan: None}).to_dict("records"),
            "outcomes": out.to_dict("records"),
            "news": news,
            "record": dict(rec, late_signals=int((sig["late"] == 1).sum()), review=self.review),
            "scorecard": card,
            "costs": {"shares": CFG["shares_per_trade"],
                      "per_share_each_way": CFG["fee_per_share"] + CFG["slippage_per_share"]},
            "model": {"id": self.model_id, "train_end": self.model["train_end"],
                      "band_coverage": CFG["band_coverage"], "typical_coverage": CFG["typical_coverage"]},
        }
