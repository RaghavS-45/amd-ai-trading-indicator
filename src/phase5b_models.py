"""
phase5b_models.py  –  pre-registered Phase 5b experiments (see EXPERIMENTS.md).

Volatility-scaled, clipped features; a continuous score instead of three classes.
Same rows, folds, costs and metrics as Phase 5. Pass/fail is judged on the confirmation
folds (confirm_from_fold onward), which the design did not look at.

A fold's score cutoff comes from an inner holdout: fit on train minus the last
inner_holdout_days, take the |score| quantile on those days, then refit on all of train.

Usage:
    python src/phase5b_models.py --all
    python src/phase5b_models.py --exp E5b.1 E5b.2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from evaluate import BPS, evaluate_signals, simulate_trades
from phase4_baselines import print_table
from phase5_models import pass_fail, set_columns
from sessions import ET
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"

TOP = CFG["signal_share"]
EXPERIMENTS = {
    "E5b.1": dict(parts=["ridge"], share=1.0, rolling=False),
    "E5b.2": dict(parts=["ridge"], share=TOP, rolling=False),
    "E5b.3": dict(parts=["logistic"], share=TOP, rolling=False),
    "E5b.4": dict(parts=["lgbm"], share=TOP, rolling=False),
    "E5b.5": dict(parts=["ridge"], share=TOP, rolling=True),
    "E5b.6": dict(parts=["ridge", "lgbm"], share=TOP, rolling=False),
}
LGBM_PARAMS = dict(n_estimators=200, learning_rate=0.02, num_leaves=7, min_child_samples=2000,
                   colsample_bytree=0.7, subsample=0.7, subsample_freq=1, reg_lambda=10.0,
                   random_state=CFG["random_seed"], deterministic=True, force_row_wise=True,
                   n_jobs=4, verbose=-1)


def scaled_target(data: pd.DataFrame) -> pd.Series:
    vol = data["volatility"].where(data["volatility"] > 0)
    z = data["fwd_ret"] / (vol * np.sqrt(CFG["horizon_bars"]))
    return z.fillna(0.0).clip(-CFG["norm_clip"], CFG["norm_clip"])


def fit_score(part: str, fit: pd.DataFrame, rows: pd.DataFrame, cols: list) -> np.ndarray:
    if part == "logistic":
        fit = fit[fit["fwd_ret"] != 0]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=CFG["random_seed"]))
        model.fit(fit[cols], fit["fwd_ret"] > 0)
        return model.predict_proba(rows[cols])[:, 1] - 0.5
    model = (make_pipeline(StandardScaler(), Ridge(alpha=CFG["ridge_alpha"])) if part == "ridge"
             else LGBMRegressor(**LGBM_PARAMS))
    model.fit(fit[cols], scaled_target(fit))
    return model.predict(rows[cols])


def fold_scores(spec: dict, data: pd.DataFrame, train_days: list, val_days: list, cols: list):
    """Validation scores and the cutoff from the inner holdout. Never reads validation labels."""
    days = train_days[-CFG["rolling_train_days"]:] if spec["rolling"] else train_days
    n_hold = CFG["inner_holdout_days"]
    inner = data[data["date"].isin(set(days[:-n_hold]))]
    hold = data[data["date"].isin(set(days[-n_hold:]))]
    train = data[data["date"].isin(set(days))]
    val = data[data["date"].isin(set(val_days))]

    hold_total, val_total = np.zeros(len(hold)), np.zeros(len(val))
    for part in spec["parts"]:
        h = fit_score(part, inner, hold, cols)
        scale = h.std()
        hold_total += h / scale
        val_total += fit_score(part, train, val, cols) / scale
    cutoff = 0.0 if spec["share"] >= 1 else float(np.quantile(np.abs(hold_total), 1 - spec["share"]))
    return val, val_total, cutoff


def delayed_return(grid: pd.DataFrame) -> pd.Series:
    """Log return from close t+1 to close t+horizon: the trade if the fill is one minute late."""
    late = grid["close"].reindex(grid.index + pd.Timedelta(minutes=1)).to_numpy()
    end = grid["close"].reindex(grid.index + pd.Timedelta(minutes=CFG["horizon_bars"])).to_numpy()
    return pd.Series(np.log(end / late), index=grid.index)


def delayed_bps(oos: pd.DataFrame, signal: np.ndarray) -> float:
    trades = simulate_trades(oos, signal)
    if trades.empty:
        return np.nan
    cost = 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"]) / trades["entry"]
    return float((trades["direction"] * oos.loc[trades.index, "delay_ret"] - cost).mean() * BPS)


def summarize(oos: pd.DataFrame) -> dict:
    sig = oos["signal"].to_numpy()
    res = evaluate_signals(oos, sig, oos["date"].nunique())
    res.update(pass_fail(res, float((oos["fwd_ret"] > 0).mean())))
    res["delay_bps_1x"] = delayed_bps(oos, sig)
    return res


def report_scopes(exp: str, oos: pd.DataFrame) -> list:
    first = CFG["confirm_from_fold"]
    last = oos["fold"].max()
    rows = []
    for scope, rows_in in [("confirm", oos["fold"] >= first), ("design", oos["fold"] < first),
                           ("all", oos["fold"] > 0), ("confirm_ex_last", (oos["fold"] >= first) & (oos["fold"] < last))]:
        res = summarize(oos[rows_in])
        res.update(experiment=exp, scope=scope)
        rows.append(res)
    return rows


def score_deciles(exp: str, oos: pd.DataFrame) -> pd.DataFrame:
    conf = oos[oos["fold"] >= CFG["confirm_from_fold"]]
    bins = pd.qcut(conf["score"], 10, labels=False, duplicates="drop")
    out = conf.groupby(bins).agg(mean_score=("score", "mean"), mean_ret_bps=("fwd_ret", lambda x: x.mean() * BPS),
                                 up_share=("fwd_ret", lambda x: (x > 0).mean()), rows=("fwd_ret", "size"))
    out.insert(0, "experiment", exp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--exp", nargs="*", default=[])
    args = ap.parse_args()
    exps = list(EXPERIMENTS) if args.all else args.exp

    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index.tz_convert(ET).date <= pd.Timestamp(CFG["train_end"]).date()]
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=["base", "context", "intraday", "norm"])
    data["delay_ret"] = delayed_return(grid).loc[data.index]
    cols = set_columns(grid, ["norm"])
    splits = walk_forward_splits(data["date"])
    print(f"Rows {len(data):,}, days {data['date'].nunique()}, folds {len(splits)}, {len(cols)} features: {cols}")

    control = pd.read_parquet(REPORT / "phase5_E5.6_oos.parquet").join(data[["delay_ret"]], how="inner")
    rows = report_scopes("E5.6_control", control)
    deciles = []
    for exp in exps:
        spec = EXPERIMENTS[exp]
        parts, per_fold = [], []
        for k, (train_days, val_days) in enumerate(splits, 1):
            val, score, cutoff = fold_scores(spec, data, train_days, val_days, cols)
            signal = np.where(np.abs(score) >= cutoff, np.sign(score), 0).astype(int)
            res = evaluate_signals(val, signal, len(val_days))
            res.update(fold=k, val_start=val_days[0], val_end=val_days[-1], cutoff=cutoff)
            per_fold.append(res)
            parts.append(val[["fwd_ret", "label", "close", "date", "delay_ret"]].assign(
                score=score, signal=signal, fold=k))
        oos = pd.concat(parts)
        oos.to_parquet(REPORT / f"phase5b_{exp}_oos.parquet")
        pd.DataFrame(per_fold).set_index("fold").to_csv(REPORT / f"phase5b_{exp}_folds.csv")
        rows += report_scopes(exp, oos)
        deciles.append(score_deciles(exp, oos))
        print(f"\n{exp} per fold")
        print_table(pd.DataFrame(per_fold).set_index("fold"), "Fold")

    table = pd.DataFrame(rows).set_index(["experiment", "scope"])
    table.to_csv(REPORT / "phase5b_summary.csv")
    if deciles:
        pd.concat(deciles).to_csv(REPORT / "phase5b_deciles.csv")
    show = ["signals_per_day", "dir_acc", "dir_se_day", "dir_bar", "strict_acc", "trades_per_day",
            "bps_per_trade_1x", "delay_bps_1x", "bps_per_trade_3x", "pnl_1x", "pass"]
    pd.set_option("display.width", 220)
    print("\n", table[show].round(4).to_string())


if __name__ == "__main__":
    main()
