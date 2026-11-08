"""
server.py  –  the AMD indicator web app (FastAPI). A background thread runs the engine once a
minute; the page polls /api/state.

  live   (default): free Alpaca IEX bars and news; signals stored in signals_db_path.
          Outside market hours it shows the last session recomputed after the close (marked as a
          review, not live signals, and kept in memory only).
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
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.responses import FileResponse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from live_engine import CAL, ONE, Engine  # noqa: E402
from live_model import load_model  # noqa: E402
from signal_store import Store  # noqa: E402

CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DELAY = pd.Timedelta(seconds=CFG["live_bar_delay_seconds"])
STATIC = Path(__file__).parent / "static"


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def latest_session(now: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(now.tz_convert("America/New_York").date())
    if CAL.is_session(day) and now >= CAL.session_open(day):
        return day
    return CAL.date_to_session(day, "previous") if not CAL.is_session(day) else CAL.previous_session(day)


class Runner:
    def __init__(self, mode: str):
        self.mode = mode
        self.lock = threading.Lock()
        self.status = {"mode": mode, "message": "starting", "error": None, "review": False}
        self.model = load_model()
        self.current = None
        self.snapshot = {"session": None}

    def start(self):
        target = self._replay if self.mode == "replay" else self._live
        threading.Thread(target=target, daemon=True).start()

    def state(self) -> dict:
        return dict(self.snapshot, status=dict(self.status, server_time=utc_now().isoformat()))

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

    # ── live ────────────────────────────────────────────────────────
    def _live(self):
        from live_sources import AlpacaSource
        source = AlpacaSource()
        live = Engine(source, Store(ROOT / CFG["signals_db_path"]), self.model, utc_now)
        review = None
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
                if live.final_upto < live.last_bar:
                    self._run(lambda: live.step(live.last_bar))
                self.publish(live)
                self.status.update(review=False, message="Market closed: today's live signals")
            else:
                if review is None or review.day != day:
                    review = Engine(source, Store(":memory:"), self.model, utc_now)
                    review.review = True
                    self._run(lambda: review.open_session(day))
                    self._run(lambda: review.step(review.last_bar))
                self.publish(review)
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
def api_state():
    return runner.state()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
