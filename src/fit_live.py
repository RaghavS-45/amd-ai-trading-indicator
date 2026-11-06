"""
fit_live.py  –  fits and freezes the indicator's models on training data only (up to train_end).

Move size: HAR + time of day (V2 in Phase 5f), fitted as in its last walk-forward fold: the last
inner_holdout_days set the smearing factor and the band multipliers, then the model is refit on
all training days. Activity cut points and the direction-lean cut come from the last year of
training (activity_from to train_end). Direction lean: E5.6 (logistic, base + context + intraday).

Writes live_model_path (joblib) and live_model_info_path (JSON, committed).
Usage: python src/fit_live.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LinearRegression

from dataset import build_dataset, feature_columns
from live_model import HAR, LEAN_SETS, MOVE_SETS
from long_history import load_sip
from phase5_models import make_model
from volatility import load_vol_grid, sip_targets

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def log_target(rv):
    return np.log(rv + CFG["rv_floor"])


def fit_move(recent_from) -> dict:
    grid = load_vol_grid()
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=MOVE_SETS)
    data = data.join(sip_targets(grid.index, load_sip()).loc[data.index])
    data = data[data["rv_fwd"].notna() & data["ret_fwd"].notna()]
    days = sorted(data["date"].unique())
    hold_days = set(days[-CFG["inner_holdout_days"]:])
    inner, hold = data[~data["date"].isin(hold_days)], data[data["date"].isin(hold_days)]

    ols = LinearRegression().fit(inner[HAR], log_target(inner["rv_fwd"]))
    h = np.exp(ols.predict(hold[HAR]))
    smear = float(np.mean(hold["rv_fwd"] / h))
    scaled = np.abs(hold["ret_fwd"]) / np.sqrt(h * smear)
    z_typical = float(np.quantile(scaled, CFG["typical_coverage"]))
    z_band = float(np.quantile(scaled, CFG["band_coverage"]))

    ols = LinearRegression().fit(data[HAR], log_target(data["rv_fwd"]))
    recent = data[data["date"] >= recent_from]
    raw_ratio = np.exp(ols.predict(recent[HAR]) - recent["lrv_seas"])
    median = float(np.median(raw_ratio))
    ratio = raw_ratio / median
    return {"coef": ols.coef_.tolist(), "intercept": float(ols.intercept_), "columns": HAR,
            "smear": smear, "z_typical": z_typical, "z_band": z_band, "ratio_median": median,
            "activity_cuts": [float(np.quantile(ratio, q)) for q in CFG["activity_quantiles"]],
            "warning_cut": float(np.quantile(ratio, CFG["warning_quantile"])),
            "rows": len(data), "first_day": str(days[0]), "last_day": str(days[-1]),
            "holdout_days": [str(min(hold_days)), str(max(hold_days))]}


def fit_lean(recent_from) -> dict:
    data = build_dataset(end_date=CFG["train_end"], sets=LEAN_SETS)
    cols = feature_columns(data)
    model = make_model("logistic", None).fit(data[cols], data["label"])
    recent = data[data["date"] >= recent_from]
    p = model.predict_proba(recent[cols])
    classes = list(model.classes_)
    edge = np.abs(p[:, classes.index(1)] - p[:, classes.index(-1)])
    return {"model": model, "columns": cols, "cut": float(np.quantile(edge, CFG["lean_quantile"])),
            "rows": len(data), "first_day": str(data["date"].min()), "last_day": str(data["date"].max())}


def main():
    recent_from = pd.Timestamp(CFG["activity_from"]).date()
    move = fit_move(recent_from)
    lean = fit_lean(recent_from)
    model = {"move": move, "lean": lean, "train_end": CFG["train_end"]}
    out = ROOT / CFG["live_model_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    info = {"train_end": CFG["train_end"], "move": move,
            "lean": {k: v for k, v in lean.items() if k != "model"},
            "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (ROOT / CFG["live_info_path"]).write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
