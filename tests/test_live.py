"""
test_live.py
The live engine: same features as the training grid, no repainting, late bars ignored,
stored rows never change, signals only where the 10-minute target is inside the session,
replay limited to training days.

Run:
    python -m pytest tests/test_live.py -v
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_grid import CONTEXT_COLS, build_grid, drop_sparse_days
from features import CFG, compute_features
from live_engine import CAL, ONE, SYMBOLS, Engine
from live_model import HAR, LEAN_SETS, MOVE_SETS, adaptive_cut, direction_lean, move_forecast
from live_sources import BAR_FIELDS, NEWS_FIELDS, ReplaySource
from long_history import previous_close
from sessions import tradable_mask
from signal_store import Store
from volatility import prev_day_rv, seasonal_rv

DAY = pd.Timestamp("2025-03-14")
SESSIONS = CAL.sessions_window(DAY, -(CFG["live_history_sessions"] + 1))
STEPS = 75   # 09:30 to 10:44: past every warm-up and some outcomes


def _raw(seed: int, drop: float = 0.05) -> pd.DataFrame:
    idx = pd.DatetimeIndex([ts for d in SESSIONS for ts in
                            pd.date_range(CAL.session_open(d), CAL.session_close(d) - ONE, freq="1min")])
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, 2e-4, len(idx))), "close": close,
                       "volume": rng.integers(100, 5000, len(idx)).astype(float),
                       "trade_count": rng.integers(1, 80, len(idx)).astype(float)}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) * 1.0005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.9995
    df["vwap"] = (df["high"] + df["low"]) / 2
    return df[rng.random(len(idx)) > drop][BAR_FIELDS]


RAW = {sym: _raw(i) for i, sym in enumerate(SYMBOLS)}
NEWS = pd.DataFrame([{"id": 1, "created_at": CAL.session_open(DAY) + pd.Timedelta(minutes=12, seconds=20),
                      "updated_at": CAL.session_open(DAY) + pd.Timedelta(minutes=12, seconds=20),
                      "headline": "AMD wins a contract", "source": "test"}], columns=NEWS_FIELDS)


class FakeSource:
    def __init__(self, raw=RAW, late=None):
        self.raw, self.late, self.now = raw, late or {}, None

    def bars(self, symbols, start, end):
        out = {}
        for s in symbols:
            df = self.raw[s][(self.raw[s].index >= start) & (self.raw[s].index < end)]
            hidden = [ts for ts, seen in self.late.get(s, {}).items() if self.now is not None and self.now < seen]
            out[s] = df.drop(index=[ts for ts in hidden if ts in df.index])
        return out

    def news(self, start, end):
        n = NEWS[(NEWS["created_at"] >= start) & (NEWS["created_at"] < end)]
        return n.reset_index(drop=True)


def _model():
    cols = list(compute_features(_training_grid().iloc[:400], LEAN_SETS).columns)
    rng = np.random.default_rng(0)
    lean = LogisticRegression().fit(pd.DataFrame(rng.normal(size=(300, len(cols))), columns=cols),
                                   rng.integers(-1, 2, 300))
    return {"train_end": CFG["train_end"],
            "move": {"coef": [0.1, 0.5, 0.25, 0.1, 0.05], "intercept": -0.1, "smear": 1.3, "z_typical": 0.6,
                     "z_band": 1.2, "ratio_median": 0.75, "activity_cuts": [0.7, 1.5], "warning_cut": 2.1},
            "lean": {"model": lean, "columns": cols, "cut": 0.05}}


def _training_grid() -> pd.DataFrame:
    """The training recipe (build_grid.py, long_history.py, volatility.py) on the same raw bars."""
    start, end = str(SESSIONS[0].date()), str(DAY.date())
    grid = build_grid(RAW[SYMBOLS[0]], start, end)
    grid, _ = drop_sparse_days(grid[grid["close"].notna()], CFG["min_day_coverage"])
    for sym in SYMBOLS[1:]:
        ctx = build_grid(RAW[sym], start, end).reindex(grid.index)[CONTEXT_COLS]
        grid = grid.join(ctx.add_prefix(f"{sym.lower()}_"))
    grid["prev_day_rv"] = prev_day_rv(grid["close"])
    grid["seas_rv"] = seasonal_rv(grid["close"])
    grid["prev_close"] = previous_close(grid["close"])
    grid["qqq_prev_close"] = previous_close(grid["qqq_close"])
    grid["news_count"] = 0.0
    grid.loc[CAL.session_open(DAY) + pd.Timedelta(minutes=13), "news_count"] = 1.0
    return grid


MODEL = _model()


def _run(source, store, steps=STEPS, at_once=False):
    clock = {"t": None}
    eng = Engine(source, store, MODEL, lambda: clock["t"])
    eng.open_session(DAY)
    minutes = pd.date_range(eng.open_utc, periods=steps, freq="1min")
    for upto in ([minutes[-1]] if at_once else minutes):
        clock["t"] = source.now = upto + ONE + pd.Timedelta(seconds=CFG["live_bar_delay_seconds"])
        eng.step(upto)
    return eng


VALUE_COLS = ["close", "forecast_rv", "band_lo", "band_hi", "activity", "p_up", "p_down", "news_30", "filled_30"]


def test_engine_features_match_training_grid():
    eng = _run(FakeSource(), Store(":memory:"))
    live = compute_features(eng.frame, MOVE_SETS + LEAN_SETS)
    grid = _training_grid()
    train = compute_features(grid[grid.index.tz_convert("America/New_York").date == DAY.date()],
                             MOVE_SETS + LEAN_SETS).reindex(live.index)
    pd.testing.assert_frame_equal(live, train, check_exact=False, rtol=1e-12, atol=1e-12)
    assert live[HAR].dropna().shape[0] > 30


def test_no_repainting_step_by_step_equals_after_the_fact():
    a = _run(FakeSource(), Store(":memory:")).store.signals()
    b = _run(FakeSource(), Store(":memory:"), at_once=True).store.signals()
    assert len(a) > 30 and (a["late"] == 0).all() and (b["late"].iloc[:-2] == 1).all()
    pd.testing.assert_frame_equal(a[["ts"] + VALUE_COLS], b[["ts"] + VALUE_COLS])


def test_review_scores_every_signal_live_skips_late_ones():
    eng = _run(FakeSource(), Store(":memory:"), at_once=True)
    assert eng.state()["record"]["scored"] == 0 and eng.state()["scorecard"] == {}
    eng.review = True
    st = eng.state()
    assert st["record"]["scored"] == len(eng.store.outcomes()) > 0 and st["record"]["review"]
    assert st["scorecard"]["always_up"]["signals"] == st["record"]["scored"]


def test_late_bar_is_ignored_and_signals_unchanged():
    ts = CAL.session_open(DAY) + pd.Timedelta(minutes=40)
    raw = {s: d.copy() for s, d in RAW.items()}
    if ts not in raw["AMD"].index:
        raw["AMD"].loc[ts] = raw["AMD"].iloc[0]
        raw["AMD"] = raw["AMD"].sort_index()
    late = FakeSource(raw, late={"AMD": {ts: ts + pd.Timedelta(minutes=5)}})
    eng = _run(late, Store(":memory:"))
    assert bool(eng.frame.at[ts, "is_filled"])
    stored = eng.store.bars(str(DAY.date()))
    assert pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ") not in set(stored.loc[stored["symbol"] == "AMD", "ts"])


def test_restart_keeps_frozen_bars_and_signals():
    store = Store(":memory:")
    first = _run(FakeSource(), store, steps=50).store.signals()
    again = _run(FakeSource(), store, steps=STEPS).store.signals()
    pd.testing.assert_frame_equal(first, again.iloc[:len(first)].reset_index(drop=True))


def test_stored_rows_cannot_change(tmp_path):
    store = Store(tmp_path / "s.db")
    _run(FakeSource(), store)
    con = sqlite3.connect(str(tmp_path / "s.db"), isolation_level=None)
    for sql in ("UPDATE signals SET close = 0", "DELETE FROM signals", "UPDATE bars SET close = 0",
                "DELETE FROM outcomes"):
        with pytest.raises(sqlite3.DatabaseError, match="never modified"):
            con.execute(sql)
    n = len(store.signals())
    assert store.add_signals(store.signals().to_dict("records")) == 0 and len(store.signals()) == n


def test_signals_only_on_tradable_bars_with_target_in_session():
    eng = _run(FakeSource(), Store(":memory:"), steps=390)
    ts = pd.to_datetime(eng.store.signals()["ts"], utc=True)
    minutes = (ts - eng.open_utc) / ONE
    assert minutes.min() >= CFG["drop_open_mins"]
    assert (ts + pd.Timedelta(minutes=CFG["horizon_bars"]) <= eng.last_bar).all()
    out = eng.store.outcomes()
    assert len(out) == len(ts)
    assert out["in_band"].between(0, 1).all()
    sc = eng.state()["scorecard"]
    assert set(sc) == {"indicator", "always_up", "rsi_rule"} and sc["always_up"]["signals"] == len(out)


def test_news_counts_from_the_bar_after_arrival():
    eng = _run(FakeSource(), Store(":memory:"))
    first = CAL.session_open(DAY) + pd.Timedelta(minutes=13)
    assert eng.frame.at[first, "news_count"] == 1 and eng.frame["news_count"].sum() == 1


def test_replay_refuses_test_period():
    src = ReplaySource.__new__(ReplaySource)
    src.limit = pd.Timestamp(CFG["train_end"], tz="America/New_York") + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="hold-out test"):
        src._check(pd.Timestamp(CFG["test_start"], tz="UTC") + pd.Timedelta(hours=15))


def test_direction_cut_comes_from_earlier_sessions_only():
    eng = _run(FakeSource(), Store(":memory:"), at_once=True)
    old = _training_grid()
    old = old[old.index < eng.open_utc]
    feat = compute_features(old, LEAN_SETS)
    feat = feat[tradable_mask(feat.index) & (feat.index + pd.Timedelta(minutes=CFG["horizon_bars"])).isin(old.index)]
    days = sorted(set(feat.index.tz_convert("America/New_York").date))[-CFG["lean_sessions"]:]
    feat = feat[np.isin(feat.index.tz_convert("America/New_York").date, days)]
    shown = direction_lean(MODEL, feat, eng.lean_cut)["lean"] != "none"
    assert abs(shown.mean() - CFG["lean_share"]) < 0.005
    assert eng.lean_cut != MODEL["lean"]["cut"]
    assert adaptive_cut(MODEL, feat.iloc[:CFG["lean_min_rows"] - 1]) == MODEL["lean"]["cut"]


def test_range_multiplier_follows_the_part_of_the_day():
    day = _training_grid()
    feat = compute_features(day[day.index >= CAL.session_open(DAY)], MOVE_SETS).dropna(subset=HAR)
    parts = dict(MODEL["move"], z_band_parts={"first": 1.3, "mid": 1.1, "last": 1.4},
                 z_typical_parts={"first": 0.65, "mid": 0.55, "last": 0.7})
    one, three = move_forecast(MODEL, feat), move_forecast({"move": parts}, feat)
    hour = three.index.tz_convert("America/New_York").hour
    ratio = three["band_hw"] / one["band_hw"] * MODEL["move"]["z_band"]
    assert np.allclose(ratio[hour < 11], 1.3) and np.allclose(ratio[hour >= 15], 1.4)
    assert np.allclose(ratio[(hour >= 11) & (hour < 15)], 1.1)
    assert {(hour < 11).any(), (hour >= 15).any()} == {True}
    assert three["forecast_rv"].equals(one["forecast_rv"])


class _Const:   # stands in for the frozen LightGBM model
    def predict(self, x, **kw):
        return np.full(len(x), np.log(4e-6))


def test_lightgbm_is_the_main_range_and_har_the_backup():
    day = _training_grid()
    feat = compute_features(day[day.index >= CAL.session_open(DAY)], MOVE_SETS).dropna(subset=HAR)
    extra = ["qqq_lrv_10", "news_count_30"]
    feat = feat[np.isfinite(feat[extra]).all(axis=1)].copy()
    feat.iloc[5, feat.columns.get_loc("qqq_lrv_10")] = np.nan   # one minute with a missing input
    lgbm = {"model": _Const(), "columns": HAR + extra, "smear": 1.0,
            "z_band_parts": {p: 1.5 for p in ("first", "mid", "last")},
            "z_typical_parts": {p: 0.5 for p in ("first", "mid", "last")}}
    har, both = move_forecast(MODEL, feat), move_forecast({"move": dict(MODEL["move"], lgbm=lgbm)}, feat)
    backup = both["move_model"] == "har"
    assert backup.sum() == 1 and backup.iloc[5] and len(both) == len(har)
    assert np.allclose(both["band_hw"][~backup], 1.5 * 2e-3) and np.allclose(both["typical_hw"][~backup], 0.5 * 2e-3)
    assert both["band_hw"][backup].equals(har["band_hw"][backup])          # the backup row is HAR's range
    assert both["har_band_hw"].equals(har["band_hw"])                      # HAR is always kept beside it
    assert both[["activity", "level", "warning"]].equals(har[["activity", "level", "warning"]])


def test_old_record_gets_the_new_columns_and_keeps_its_rows(tmp_path):
    from signal_store import ADDED, SIGNAL_COLS
    old_cols = [c for c in SIGNAL_COLS if c not in ADDED]
    con = sqlite3.connect(str(tmp_path / "old.db"))
    con.execute(f"CREATE TABLE signals ({', '.join(old_cols)}, PRIMARY KEY (ts))")
    con.execute(f"INSERT INTO signals (ts, session, late, close) VALUES ('2026-11-06T15:03:00Z', '2026-11-06', 0, 250.0)")
    con.commit()
    con.close()
    store = Store(tmp_path / "old.db")
    row = store.signals().iloc[0]
    assert row["close"] == 250.0 and row["move_model"] is None and set(ADDED) <= set(store.signals().columns)
    eng = _run(FakeSource(), store)
    assert set(eng.store.signals(eng.session)["move_model"]) == {"har"}   # the test model has no LightGBM
