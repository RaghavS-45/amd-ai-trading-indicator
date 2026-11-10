"""
server.py  –  the AMD indicator web app (FastAPI). A background thread runs the engine once a
minute; the page polls /api/state.

  live   (default): free Alpaca IEX bars and news; signals stored in signals_db_path.
          Outside market hours it shows the last session recomputed after the close (marked as a
          review, not live signals, and kept in memory only).
  /api/state?day=YYYY-MM-DD shows a past session after the test period, recomputed after the close
  (a review), plus that day's live record if the app was running then. /api/days lists them.
  replay: plays a training-period day bar by bar from local files (in memory only).

Usage:
  uvicorn app.server:app
  APP_MODE=replay REPLAY_DAY=2026-06-15 REPLAY_SPEED=5 uvicorn app.server:app
"""

import os
import sys
import threading
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import json

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import _rsi  # noqa: E402
from live_engine import ASSET, CAL, ONE, Engine  # noqa: E402
from live_score import scorecard  # noqa: E402
from live_model import load_model  # noqa: E402
import record_sync  # noqa: E402
from signal_store import Store  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DELAY = pd.Timedelta(seconds=CFG["live_bar_delay_seconds"])
STATIC = Path(__file__).parent / "static"
FIRST_PAST = pd.Timestamp(CFG["test_end"]) + pd.Timedelta(days=1)   # never show the hold-out test period


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def latest_session(now: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(now.tz_convert("America/New_York").date())
    if CAL.is_session(day) and now >= CAL.session_open(day):
        return day
    return CAL.date_to_session(day, "previous") if not CAL.is_session(day) else CAL.previous_session(day)


def market_hours(now: pd.Timestamp) -> dict:
    day = latest_session(now)
    is_open = CAL.session_open(day) <= now < CAL.session_close(day)
    nxt = CAL.next_open(now)
    return {"open_now": bool(is_open), "closes": CAL.session_close(day).isoformat() if is_open else None,
            "next_open": nxt.isoformat()}


class Runner:
    def __init__(self, mode: str):
        self.mode = mode
        self.lock = threading.Lock()
        self.status = {"mode": mode, "message": "starting", "error": None, "review": False}
        self.model = load_model()
        self.current = None
        self.snapshot = {"session": None}
        self.source, self.store = None, None
        self.past, self.past_lock = OrderedDict(), threading.Lock()

    def start(self):
        target = self._replay if self.mode == "replay" else self._live
        threading.Thread(target=target, daemon=True).start()

    def state(self) -> dict:
        return dict(self.snapshot, status=dict(self.status, server_time=utc_now().isoformat()),
                    market=market_hours(utc_now()))

    def publish(self, engine):
        self.current = engine
        with self.lock:
            self.snapshot = engine.state()

    def _run(self, fn):
        try:
            with self.lock:
                fn()
            self.status["error"] = None
        except Exception as e:  # keep serving; retry next minute
            self.status["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    # ── replay ──────────────────────────────────────────────────────
    def _replay(self):
        from live_sources import ReplaySource
        day = pd.Timestamp(os.environ.get("REPLAY_DAY", CFG["replay_default_day"]))
        speed = float(os.environ.get("REPLAY_SPEED", CFG["replay_speed"]))
        clock = {"t": None}
        eng = Engine(ReplaySource(), Store(":memory:"), self.model, lambda: clock["t"])
        self._run(lambda: eng.open_session(day))
        for upto in pd.date_range(eng.open_utc, eng.last_bar, freq="1min"):
            clock["t"] = upto + ONE + DELAY
            self.status["message"] = f"Replay of {day.date()} (training data), {speed:g} bars per second"
            self._run(lambda: eng.step(upto))
            self.publish(eng)
            time.sleep(1 / speed)
        self.status["message"] = f"Replay of {day.date()} finished"

    # ── past sessions ───────────────────────────────────────────────
    def days(self) -> dict:
        if self.source is None:
            return {"days": []}
        latest = latest_session(utc_now())
        with self.lock:
            live = set(self.store.live_sessions())
        done = utc_now() >= CAL.session_close(latest) + DELAY + ONE
        days = [str(d.date()) for d in CAL.sessions_in_range(FIRST_PAST, latest) if d < latest or done]
        return {"days": [{"day": d, "live": d in live} for d in reversed(days)]}

    def past_state(self, day: str) -> dict:
        if day not in {d["day"] for d in self.days()["days"]}:
            return {"session": None, "status": dict(self.status, error=f"{day}: not an available past session")}
        status = dict(self.status, review=True, error=None, server_time=utc_now().isoformat(),
                      message=f"Session {day} recomputed after the close (a review, not live signals)")
        try:
            snap = self._review(day)
        except Exception as e:
            traceback.print_exc()
            return {"session": None, "status": dict(status, error=f"{type(e).__name__}: {e}")}
        return dict(snap, status=status)

    def _review(self, day: str) -> dict:
        """A finished session recomputed in memory (every signal scored), plus the signals the app
        showed live on time that day, if it was running. Cached: a past session never changes."""
        with self.past_lock:
            if day not in self.past:
                eng = Engine(self.source, Store(":memory:"), self.model, utc_now)
                eng.review = True
                eng.open_session(pd.Timestamp(day))
                eng.step(eng.last_bar)
                snap = eng.state()
                with self.lock:
                    sig, out = self.store.signals(day), self.store.outcomes(day)
                sig = sig[sig["late"] == 0]
                if len(sig):
                    rec, card = eng.grade(sig, out)
                    snap["live_record"] = {"signals": int(len(sig)), "first": sig["ts"].iloc[0],
                                           "last": sig["ts"].iloc[-1], "record": rec, "scorecard": card}
                self.past[day] = snap
                while len(self.past) > CFG["past_days_in_memory"]:   # small free hosts: 512 MB
                    self.past.popitem(last=False)
            self.past.move_to_end(day)
            return self.past[day]

    # ── track record over every live day ────────────────────────────
    def performance(self) -> dict:
        """On-time signals of every day the app ran live, scored as on the page (from SQLite only)."""
        if self.store is None:
            return {"days": [], "total": None}
        with self.lock:
            sig, out = self.store.signals(), self.store.outcomes()
            bars = {d: self.store.bars(d) for d in sig["session"].unique()}
        sig = sig[sig["late"] == 0]
        days, graded = [], []
        for day, s in sig.groupby("session"):
            b = bars[day]
            close = b[b["symbol"] == ASSET].set_index(pd.to_datetime(b.loc[b["symbol"] == ASSET, "ts"], utc=True))["close"]
            grid = pd.date_range(CAL.session_open(pd.Timestamp(day)), close.index.max(), freq="1min")
            rsi = _rsi(close.reindex(grid).ffill(), CFG["rsi_period"])   # quiet minutes filled flat, as build_grid
            rsi.index = grid.strftime("%Y-%m-%dT%H:%M:%SZ")
            scored = out[out["ts"].isin(set(s["ts"]))]
            g = scored.merge(s[["ts", "close", "lean"]], on="ts").assign(rsi=lambda d: d["ts"].map(rsi))
            graded.append(g)
            days.append({"day": day, "first": s["ts"].iloc[0], "last": s["ts"].iloc[-1], "signals": int(len(s)),
                         "scored": int(len(g)), "in_band": int(g["in_band"].sum()),
                         "in_typical": int(g["in_typical"].sum()), "scorecard": scorecard(g)})
        if not days:
            return {"days": [], "total": None}
        g = pd.concat(graded)
        total = {"scored": int(len(g)), "in_band": int(g["in_band"].sum()), "in_typical": int(g["in_typical"].sum()),
                 "signals": int(sum(d["signals"] for d in days)), "scorecard": scorecard(g)}
        return {"days": days[::-1], "total": total,
                "costs": {"shares": CFG["shares_per_trade"],
                          "per_share_each_way": CFG["fee_per_share"] + CFG["slippage_per_share"]}}

    # ── live ────────────────────────────────────────────────────────
    def _live(self):
        from live_sources import AlpacaSource
        source = self.source = AlpacaSource()
        db = ROOT / CFG["signals_db_path"]
        record_sync.restore_until_done(db, self.status)   # hosts whose disk is wiped on restart
        self.store = Store(db)
        record_sync.start(db)
        live = Engine(source, self.store, self.model, utc_now)
        while True:
            now = utc_now()
            day = latest_session(now)
            close = CAL.session_close(day)
            if now < close + DELAY + ONE:
                if live.day != day:
                    self._run(lambda: live.open_session(day))
                upto = (now - DELAY).floor("min") - ONE
                if upto >= live.open_utc:
                    self._run(lambda: live.step(upto))
                self.publish(live)
                self.status.update(review=False, message="Live: IEX bars, a minute is final "
                                   f"{CFG['live_bar_delay_seconds']}s after it closes")
            elif live.day == day:
                if live.done_upto < live.last_bar:
                    self._run(lambda: live.step(live.last_bar))
                self.publish(live)
                self.status.update(review=False, message="Market closed: today's live signals")
            else:
                if self.snapshot.get("session") != str(day.date()):
                    try:   # not under self.lock: _review takes past_lock first, like a page request
                        self.snapshot = dict(self._review(str(day.date())))
                        self.status["error"] = None
                    except Exception as e:
                        self.status["error"] = f"{type(e).__name__}: {e}"
                        traceback.print_exc()
                self.status.update(review=True, message=f"Market closed: session {day.date()} recomputed "
                                   "after the close (a review, not live signals)")
            wake = (utc_now().floor("min") + ONE + DELAY - utc_now()).total_seconds()
            time.sleep(max(wake, 1.0))


runner = Runner(os.environ.get("APP_MODE", "live"))
app = FastAPI(title="AMD 10-minute indicator")


@app.on_event("startup")
def _start():
    runner.start()


@app.get("/api/state")
def api_state(day: Optional[str] = None):
    return runner.past_state(day) if day else runner.state()


@app.api_route("/api/days", methods=["GET", "HEAD"])   # HEAD: uptime monitors
def api_days():
    return runner.days()


@app.get("/api/performance")
def api_performance():
    return runner.performance()


@app.get("/api/experiments")
def api_experiments():
    log = pd.read_csv(ROOT / "report" / "experiment_log.csv", dtype=str).fillna("")
    return {"runs": log.to_dict("records")}


@app.get("/api/model")
def api_model():
    info = json.loads((ROOT / CFG["live_info_path"]).read_text())
    return {"train_end": info["train_end"], "move": {k: info["move"][k] for k in
                                                     ("columns", "coef", "rows", "first_day", "last_day")},
            "lean": {k: info["lean"][k] for k in ("rows", "first_day", "last_day")},
            "lean_features": len(info["lean"]["columns"]),
            "band_coverage": CFG["band_coverage"], "typical_coverage": CFG["typical_coverage"],
            "horizon_minutes": CFG["horizon_bars"], "drop_open_mins": CFG["drop_open_mins"]}


@app.middleware("http")
async def _revalidate(request, call_next):   # pages and scripts always fresh after a redeploy
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def index():
    return FileResponse(STATIC / "index.html")
