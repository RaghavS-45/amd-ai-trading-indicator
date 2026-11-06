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


def load_model(path=None) -> dict:
    return joblib.load(path or ROOT / CFG["live_model_path"])


def move_forecast(model: dict, feat: pd.DataFrame) -> pd.DataFrame:
    """Forecast 10-minute variance, typical and 80% half widths (log return), activity vs normal."""
    m = model["move"]
    x = feat[HAR].to_numpy(dtype=float)
    log_f = x @ np.asarray(m["coef"]) + m["intercept"]
    forecast = np.exp(log_f) * m["smear"]
    ratio = np.exp(log_f - feat["lrv_seas"].to_numpy(dtype=float)) / m["ratio_median"]
    lo, hi = m["activity_cuts"]
    level = np.where(ratio < lo, LEVELS[0], np.where(ratio > hi, LEVELS[2], LEVELS[1]))
    out = pd.DataFrame({"forecast_rv": forecast,
                        "typical_hw": m["z_typical"] * np.sqrt(forecast),
                        "band_hw": m["z_band"] * np.sqrt(forecast),
                        "activity": ratio, "level": level,
                        "warning": ratio > m["warning_cut"]}, index=feat.index)
    return out[np.isfinite(log_f)]


def direction_lean(model: dict, feat: pd.DataFrame) -> pd.DataFrame:
    d = model["lean"]
    x = feat[d["columns"]]
    ok = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    classes = list(d["model"].classes_)
    p = d["model"].predict_proba(x[ok]) if ok.any() else np.empty((0, len(classes)))
    out = pd.DataFrame({"p_down": p[:, classes.index(-1)], "p_flat": p[:, classes.index(0)],
                        "p_up": p[:, classes.index(1)]}, index=x.index[ok])
    edge = out["p_up"] - out["p_down"]
    out["lean"] = np.where(edge >= d["cut"], "weak up", np.where(edge <= -d["cut"], "weak down", "none"))
    return out
