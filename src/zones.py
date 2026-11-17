"""
zones.py  –  support and resistance zones for one session, built only from earlier sessions.

Sources (each a band of width `zone_width_atr` x the average daily range):
  day     previous session's high and low
  week    previous calendar week's high and low
  month   high and low of the last `zone_month_sessions` sessions
  swing   prices where AMD turned at least twice in the last `zone_swing_sessions` sessions
          (pivots on 15-minute bars), merged into clusters

Overlapping zones are merged. A zone above the previous close is resistance, below it support.
Hold-out test days are never used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sessions import ET

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def _daily(real: pd.DataFrame) -> pd.DataFrame:
    date = real.index.tz_convert(ET).date
    return real.groupby(date).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))


def _pivots(real: pd.DataFrame, k: int) -> list:
    """Swing highs and lows on 15-minute bars, inside each session only."""
    out = []
    for _, d in real.groupby(real.index.tz_convert(ET).date):
        b = d.resample("15min").agg({"high": "max", "low": "min"}).dropna()
        hi, lo = b["high"].to_numpy(), b["low"].to_numpy()
        for i in range(k, len(b) - k):
            if hi[i] == hi[i - k:i + k + 1].max():
                out.append((float(hi[i]), "high"))
            if lo[i] == lo[i - k:i + k + 1].min():
                out.append((float(lo[i]), "low"))
    return out


def _band(price: float, side: str, w: float) -> tuple:
    return (price - w, price) if side == "high" else (price, price + w)


def compute_zones(past: pd.DataFrame, day: pd.Timestamp) -> list:
    """Zones for session `day` from `past` (grid rows of earlier sessions, is_filled column)."""
    real = past[~past["is_filled"]]
    dates = real.index.tz_convert(ET).date
    test = (dates >= pd.Timestamp(CFG["test_start"]).date()) & (dates <= pd.Timestamp(CFG["test_end"]).date())
    real = real[~test]
    if not len(real):
        return []
    daily = _daily(real)
    atr = float((daily["high"] - daily["low"]).tail(14).mean())
    w = CFG["zone_width_atr"] * atr
    raw = []

    last = daily.iloc[-1]
    raw += [("Prev day high", last["high"], "high"), ("Prev day low", last["low"], "low")]

    week = pd.Series(pd.to_datetime(daily.index)).dt.to_period("W-FRI").to_numpy()
    this_week = pd.Timestamp(day).to_period("W-FRI")
    prev = daily[week < this_week]
    if len(prev):
        wk = pd.Series(pd.to_datetime(prev.index)).dt.to_period("W-FRI").to_numpy()
        pw = prev[wk == wk.max()]
        raw += [("Prev week high", pw["high"].max(), "high"), ("Prev week low", pw["low"].min(), "low")]

    month = daily.tail(CFG["zone_month_sessions"])
    raw += [("Month high", month["high"].max(), "high"), ("Month low", month["low"].min(), "low")]

    recent = real[pd.Index(real.index.tz_convert(ET).date).isin(daily.index[-CFG["zone_swing_sessions"]:])]
    piv = sorted(_pivots(recent, CFG["zone_pivot_bars"]))
    groups, cur = [], []
    for p in piv:
        if cur and p[0] - cur[0][0] > w:
            groups.append(cur); cur = []
        cur.append(p)
    if cur:
        groups.append(cur)
    zones = [{"lo": lo, "hi": hi, "highs": int(side == "high"), "lows": int(side == "low"), "labels": [name],
              "touches": 1} for name, price, side in raw for lo, hi in [_band(price, side, w)]]
    for g in groups:
        if len(g) < CFG["zone_min_touches"]:
            continue
        prices = [p for p, _ in g]
        mid = float(np.mean(prices))
        zones.append({"lo": min(min(prices), mid - w / 2), "hi": max(max(prices), mid + w / 2),
                      "highs": sum(s == "high" for _, s in g), "lows": sum(s == "low" for _, s in g),
                      "labels": [f"Swing ×{len(g)}"], "touches": len(g)})

    zones.sort(key=lambda z: z["lo"])
    merged = []
    for z in zones:
        if merged and z["lo"] <= merged[-1]["hi"]:
            m = merged[-1]
            m.update(hi=max(m["hi"], z["hi"]), highs=m["highs"] + z["highs"], lows=m["lows"] + z["lows"],
                     labels=m["labels"] + z["labels"], touches=m["touches"] + z["touches"])
        else:
            merged.append(dict(z))
    ref = float(last["close"])
    mid = lambda z: (z["lo"] + z["hi"]) / 2
    keep = [z for z in merged if abs(mid(z) - ref) <= CFG["zone_max_atr"] * atr]
    keep = sorted(sorted(keep, key=lambda z: abs(mid(z) - ref))[:CFG["zone_max_count"]], key=lambda z: z["lo"])
    return [{"lo": round(z["lo"], 2), "hi": round(z["hi"], 2),
             "kind": "resistance" if mid(z) > ref else "support",
             "label": " · ".join(dict.fromkeys(z["labels"])), "strength": int(z["touches"])} for z in keep]
