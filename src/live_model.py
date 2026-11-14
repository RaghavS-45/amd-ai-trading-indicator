"""
live_model.py  –  turns feature rows into what the indicator shows, using the frozen model from
src/fit_live.py. Used by the live server, replay and Phase 6, so all three read the same numbers.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
HAR = ["lrv_10", "lrv_30", "lrv_sess", "lrv_prev_day", "lrv_seas"]
MOVE_SETS = ["vol"]
LEAN_SETS = ["base", "context", "intraday"]
LEVELS = ("calm", "normal", "busy")


PARTS = ("first", "mid", "last")


def part_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    """first = before 11:00 New York, last = from 15:00, mid = between."""
    hour = index.tz_convert("America/New_York").hour
    return np.where(hour < 11, PARTS[0], np.where(hour >= 15, PARTS[2], PARTS[1]))


def load_model(path=None) -> dict:
    return joblib.load(path or ROOT / CFG["live_model_path"])


def move_forecast(model: dict, feat: pd.DataFrame) -> pd.DataFrame:
    """Forecast 10-minute variance, typical and 80% half widths (log return), activity vs normal.
    Ranges come from LightGBM when the frozen model has one (HAR on rows with a missing input);
    activity and the warning always come from HAR against the time-of-day normal."""
    m = model["move"]
    x = feat[HAR].to_numpy(dtype=float)
    log_f = x @ np.asarray(m["coef"]) + m["intercept"]
    forecast = np.exp(log_f) * m["smear"]
    ratio = np.exp(log_f - feat["lrv_seas"].to_numpy(dtype=float)) / m["ratio_median"]
    lo, hi = m["activity_cuts"]
    level = np.where(ratio < lo, LEVELS[0], np.where(ratio > hi, LEVELS[2], LEVELS[1]))
    z_typical, z_band = m["z_typical"], m["z_band"]
    if "z_band_parts" in m:   # one multiplier per part of the day (Phase 7d)
        part = pd.Series(part_of_day(feat.index))
        z_typical, z_band = part.map(m["z_typical_parts"]).to_numpy(), part.map(m["z_band_parts"]).to_numpy()
    out = pd.DataFrame({"forecast_rv": forecast,
                        "typical_hw": z_typical * np.sqrt(forecast),
                        "band_hw": z_band * np.sqrt(forecast),
                        "activity": ratio, "level": level,
                        "warning": ratio > m["warning_cut"]}, index=feat.index)
    out["move_model"] = "har"
    for c in ("forecast_rv", "typical_hw", "band_hw"):
        out[f"har_{c}"] = out[c]   # the published baseline, kept beside the main forecast
    if "lgbm" in m:
        _use_lgbm(m["lgbm"], feat, out)
    return out[np.isfinite(log_f)]


def _use_lgbm(g: dict, feat: pd.DataFrame, out: pd.DataFrame) -> None:
    """LightGBM (V4) is the main forecast on rows where all its inputs are present; HAR stays elsewhere."""
    x = feat[g["columns"]]
    ok = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    if not ok.any():
        return
    forecast = np.exp(g["model"].predict(x[ok], num_threads=1)) * g["smear"]
    part = pd.Series(part_of_day(feat.index[ok]))
    out.loc[ok, "forecast_rv"] = forecast
    out.loc[ok, "typical_hw"] = part.map(g["z_typical_parts"]).to_numpy() * np.sqrt(forecast)
    out.loc[ok, "band_hw"] = part.map(g["z_band_parts"]).to_numpy() * np.sqrt(forecast)
    out.loc[ok, "move_model"] = "lgbm"


def lean_edge(model: dict, feat: pd.DataFrame) -> pd.Series:
    """p_up - p_down on rows with complete inputs."""
    out = direction_lean(model, feat)
    return out["p_up"] - out["p_down"]


def adaptive_cut(model: dict, past_feat: pd.DataFrame) -> float:
    """Cut that shows a direction on lean_share of the previous lean_sessions sessions' rows.
    Uses model outputs only, never outcomes. Falls back to the frozen cut with too little history."""
    days = pd.Series(past_feat.index.tz_convert("America/New_York").date, index=past_feat.index)
    keep = days.isin(sorted(days.unique())[-CFG["lean_sessions"]:]).to_numpy()
    edge = lean_edge(model, past_feat[keep]).abs()
    if len(edge) < CFG["lean_min_rows"]:
        return float(model["lean"]["cut"])
    return float(np.quantile(edge, 1 - CFG["lean_share"]))


def direction_lean(model: dict, feat: pd.DataFrame, cut: float = None) -> pd.DataFrame:
    d = model["lean"]
    cut = d["cut"] if cut is None else cut
    x = feat[d["columns"]]
    ok = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    classes = list(d["model"].classes_)
    p = d["model"].predict_proba(x[ok]) if ok.any() else np.empty((0, len(classes)))
    out = pd.DataFrame({"p_down": p[:, classes.index(-1)], "p_flat": p[:, classes.index(0)],
                        "p_up": p[:, classes.index(1)]}, index=x.index[ok])
    edge = out["p_up"] - out["p_down"]
    out["lean"] = np.where(edge >= cut, "weak up", np.where(edge <= -cut, "weak down", "none"))
    return out
