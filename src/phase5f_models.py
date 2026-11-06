"""
phase5f_models.py  –  pre-registered Phase 5f experiment (see EXPERIMENTS.md): forecasting the
size of AMD's next 10-minute move against HAR baselines. Same walk-forward as Phase 5e.

Usage: python src/phase5f_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from long_history import load_sip
from phase5_models import set_columns
from volatility import interval_score, load_vol_grid, qlike, sip_targets
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
FRESH_END = pd.Timestamp(CFG["fresh_end"]).date()
CONFIRM_START = pd.Timestamp(CFG["explore_end"]).date()
HAR = ["lrv_10", "lrv_30", "lrv_sess", "lrv_prev_day"]
MODELS = {
    "V0": ("persistence", ["lrv_10"]),
    "V1": ("ols", HAR),
    "V2": ("ols", HAR + ["lrv_seas"]),
    "V3": ("ridge", None),
    "V4": ("lgbm", None),
}
BASELINE = "V2"
LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                   colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                   random_state=CFG["random_seed"], deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)


def log_target(rv: pd.Series) -> pd.Series:
    return np.log(rv + CFG["rv_floor"])


def fit_predict(kind: str, cols: list, fit: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    if kind == "persistence":
        return rows[cols[0]].to_numpy()
    model = {"ols": lambda: LinearRegression(),
             "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=CFG["vol_ridge_alpha"])),
             "lgbm": lambda: LGBMRegressor(**LGBM_PARAMS)}[kind]()
    model.fit(fit[cols], log_target(fit["rv_fwd"]))
    return model.predict(rows[cols])


def run_model(name: str, kind: str, cols: list, data: pd.DataFrame, splits) -> pd.DataFrame:
    n_hold = CFG["inner_holdout_days"]
    z = None
    parts = []
    for k, (train_days, val_days) in enumerate(splits, 1):
        inner = data[data["date"].isin(set(train_days[:-n_hold]))]
        hold = data[data["date"].isin(set(train_days[-n_hold:]))]
        train = data[data["date"].isin(set(train_days))]
        val = data[data["date"].isin(set(val_days))]
        h = np.exp(fit_predict(kind, cols, inner, hold))
        smear = float(np.mean(hold["rv_fwd"] / h))
        z = float(np.quantile(np.abs(hold["ret_fwd"]) / np.sqrt(h * smear), CFG["band_coverage"]))
        forecast = np.exp(fit_predict(kind, cols, train, val)) * smear
        parts.append(val[["rv_fwd", "ret_fwd", "close", "date"]].assign(
            forecast=forecast, half_width=z * np.sqrt(forecast), fold=k))
    oos = pd.concat(parts)
    oos.to_parquet(REPORT / f"phase5f_{name}_oos.parquet")
    print(f"{name} done ({kind}, {len(cols)} features), last band multiplier {z:.3f}", flush=True)
    return oos


def metrics(x: pd.DataFrame, base: pd.DataFrame) -> dict:
    loss = qlike(x["rv_fwd"].to_numpy(), x["forecast"].to_numpy())
    base_loss = qlike(base["rv_fwd"].to_numpy(), base["forecast"].to_numpy())
    daily = pd.Series(loss - base_loss, index=x.index).groupby(x["date"]).mean()
    y, f = np.log(x["rv_fwd"] + CFG["rv_floor"]), np.log(x["forecast"])
    inside = np.abs(x["ret_fwd"]) <= x["half_width"]
    alpha = 1 - CFG["band_coverage"]
    return {"rows": len(x), "days": x["date"].nunique(), "qlike": float(loss.mean()),
            "qlike_vs_V2": float(daily.mean()), "qlike_vs_V2_se": float(daily.std(ddof=1) / np.sqrt(len(daily))),
            "mse_log": float(((y - f) ** 2).mean()), "r2_log": float(1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "coverage": float(inside.mean()), "band_width_bps": float((2 * x["half_width"]).mean() * 1e4),
            "interval_score_bps": float(interval_score(x["ret_fwd"].to_numpy(), x["half_width"].to_numpy(), alpha).mean() * 1e4)}


def main():
    grid = load_vol_grid()
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=["vol"])
    data = data.join(sip_targets(grid.index, load_sip()).loc[data.index])
    data = data[data["rv_fwd"].notna() & data["ret_fwd"].notna()]
    all_cols = set_columns(grid.iloc[-4000:], ["vol"])
    splits = walk_forward_splits(data["date"], min_train_days=CFG["history_min_train_days"])
    print(f"Rows {len(data):,}, days {data['date'].nunique()}, {data['date'].min()} to {data['date'].max()}, "
          f"{len(splits)} folds from {splits[0][1][0]}; features {all_cols}", flush=True)

    oos = {name: run_model(name, kind, cols or all_cols, data, splits) for name, (kind, cols) in MODELS.items()}
    scopes = {"fresh": lambda d: d <= FRESH_END, "confirm": lambda d: d > CONFIRM_START,
              "between": lambda d: (d > FRESH_END) & (d <= CONFIRM_START), "all": lambda d: d == d}
    rows = []
    for name, x in oos.items():
        for scope, pick in scopes.items():
            m = pick(x["date"])
            res = metrics(x[m], oos[BASELINE][m])
            res["beats_V2"] = bool(res["qlike_vs_V2"] + 2 * res["qlike_vs_V2_se"] < 0)
            rows.append(dict(res, model=name, scope=scope))
    table = pd.DataFrame(rows).set_index(["model", "scope"])
    table.to_csv(REPORT / "phase5f_summary.csv")
    pd.set_option("display.width", 230)
    print("\n", table.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
