"""
fit_live.py  –  fits and freezes the indicator's models on training data only (up to train_end).

Move size: HAR + time of day (V2 in Phase 5f), fitted as in its last walk-forward fold: the last
inner_holdout_days set the smearing factor and the band multipliers (one per part of the day, Phase 7d), then the model is refit on
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
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from dataset import build_dataset, feature_columns
from live_model import HAR, LEAN_SETS, MOVE_SETS, part_of_day
from long_history import load_sip
from phase5_models import make_model
from phase5f_models import LGBM_PARAMS
from volatility import load_vol_grid, sip_targets

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def log_target(rv):
    return np.log(rv + CFG["rv_floor"])


def multipliers(hold: pd.DataFrame, h: np.ndarray) -> dict:
    """Smearing factor and range multipliers from the inner hold-out: all-day, and per part of the day (Phase 7d)."""
    smear = float(np.mean(hold["rv_fwd"] / h))
    scaled = np.abs(hold["ret_fwd"]) / np.sqrt(h * smear)
    z_typical = float(np.quantile(scaled, CFG["typical_coverage"]))
    z_band = float(np.quantile(scaled, CFG["band_coverage"]))
    by_part = pd.Series(scaled.to_numpy(), index=hold.index).groupby(part_of_day(hold.index))
    z_typical_parts = by_part.quantile(CFG["typical_coverage"]).to_dict()
    z_band_parts = by_part.quantile(CFG["band_coverage"]).to_dict()
    z_typical_parts["mid"], z_band_parts["mid"] = z_typical, z_band   # midday keeps the all-day multiplier
    return {"smear": smear, "z_typical": z_typical, "z_band": z_band, "z_typical_parts": z_typical_parts,
            "z_band_parts": z_band_parts}


def fit_lgbm(data: pd.DataFrame, inner: pd.DataFrame, hold: pd.DataFrame) -> dict:
    """V4 of Phase 5f, fitted exactly as the comparison model scored in Phase 6."""
    cols = [c for c in feature_columns(data) if c not in ("rv_fwd", "ret_fwd")]
    h = np.exp(LGBMRegressor(**LGBM_PARAMS).fit(inner[cols], log_target(inner["rv_fwd"])).predict(hold[cols]))
    model = LGBMRegressor(**LGBM_PARAMS).fit(data[cols], log_target(data["rv_fwd"]))
    return dict(multipliers(hold, h), model=model, columns=cols, params={k: v for k, v in LGBM_PARAMS.items()})


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
    mult = multipliers(hold, h)

    ols = LinearRegression().fit(data[HAR], log_target(data["rv_fwd"]))
    recent = data[data["date"] >= recent_from]
    raw_ratio = np.exp(ols.predict(recent[HAR]) - recent["lrv_seas"])
    median = float(np.median(raw_ratio))
    ratio = raw_ratio / median
    return {"coef": ols.coef_.tolist(), "intercept": float(ols.intercept_), "columns": HAR,
            **mult, "lgbm": fit_lgbm(data, inner, hold), "ratio_median": median,
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
    info = {"train_end": CFG["train_end"],
            "move": dict(move, lgbm={k: v for k, v in move["lgbm"].items() if k != "model"}),
            "lean": {k: v for k, v in lean.items() if k != "model"},
            "versions": {"sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__}}
    (ROOT / CFG["live_info_path"]).write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
