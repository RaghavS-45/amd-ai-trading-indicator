"""
phase5e_models.py  –  pre-registered Phase 5e experiments (see EXPERIMENTS.md).

--fresh: E5.6 with no changes, trained on 2021 onward, on years no model has seen.
--final: tradable target and the three literature ideas, one continuous walk-forward.

Usage:
    python src/phase5e_models.py --fresh
    python src/phase5e_models.py --final
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from long_history import load_grid, load_sip, tradable_return
from phase5_models import CLASSES, set_columns
from phase5b_models import fit_score
from sessions import tradable_mask
from tradable import score
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
ALL_SETS = ["base", "context", "intraday", "norm", "seasonal", "sector", "lasthalf"]
EXPERIMENTS = {"E5e.1": ["norm"], "E5e.2": ["norm", "seasonal"], "E5e.3": ["norm", "sector"],
               "E5e.4": ["norm", "lasthalf"]}
FRESH_END = pd.Timestamp(CFG["fresh_end"]).date()
CONFIRM_START = pd.Timestamp(CFG["explore_end"]).date()


def load_rows():
    grid = load_grid()
    sip = load_sip()
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=ALL_SETS)
    assert tradable_mask(data.index).all()
    data["trade_ret"] = tradable_return(data.index, sip)
    print(f"Rows {len(data):,}, days {data['date'].nunique()}, {data['date'].min()} to {data['date'].max()}, "
          f"no SIP target {data['trade_ret'].isna().mean():.2%}")
    return grid, data


def fit_part(part: str, fit: pd.DataFrame, rows: pd.DataFrame, cols: list) -> np.ndarray:
    fit = fit[fit["trade_ret"].notna()]
    return fit_score(part, fit.assign(fwd_ret=fit["trade_ret"]), rows, cols)


def scopes(exp: str, oos: pd.DataFrame) -> list:
    groups = {"fresh": oos["date"] <= FRESH_END, "confirm": oos["date"] > CONFIRM_START,
              "confirm_ex_last": (oos["date"] > CONFIRM_START) & (oos["fold"] < oos["fold"].max()),
              "between": (oos["date"] > FRESH_END) & (oos["date"] <= CONFIRM_START)}
    return [dict(score(oos[m]), experiment=exp, scope=name) for name, m in groups.items() if m.any()]


def show(rows: list, path: Path):
    table = pd.DataFrame(rows).set_index(["experiment", "scope"])
    table.to_csv(path)
    pd.set_option("display.width", 220)
    print("\n", table.round(4).to_string(), flush=True)


def run_fresh():
    grid, data = load_rows()
    data = data[data["date"] <= FRESH_END]
    cols = set_columns(grid, ["base", "context", "intraday"])
    splits = walk_forward_splits(data["date"], min_train_days=CFG["history_min_train_days"], last_allowed=CFG["fresh_end"])
    print(f"Fresh check: {len(splits)} folds, validation {splits[0][1][0]} to {splits[-1][1][-1]}, {len(cols)} features")
    parts = []
    for k, (train_days, val_days) in enumerate(splits, 1):
        train = data[data["date"].isin(set(train_days))]
        val = data[data["date"].isin(set(val_days))]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=CFG["random_seed"]))
        model.fit(train[cols], train["label"])
        signal = CLASSES[model.predict_proba(val[cols]).argmax(axis=1)]
        parts.append(val[["fwd_ret", "trade_ret", "close", "date"]].assign(signal=signal, fold=k))
        print(f"fold {k} {val_days[0]} to {val_days[-1]}", flush=True)
    oos = pd.concat(parts)
    oos.to_parquet(REPORT / "phase5e_fresh_E5.6_oos.parquet")
    tradable = dict(score(oos), experiment="E5.6", scope="fresh_tradable")
    iex = dict(score(oos.assign(trade_ret=oos["fwd_ret"])), experiment="E5.6", scope="fresh_iex_close")
    last = oos["fold"].max()
    by_half = [dict(score(oos[m]), experiment="E5.6", scope=name) for name, m in
               {"first_half": oos["fold"] <= last // 2, "second_half": oos["fold"] > last // 2}.items()]
    show([tradable, iex] + by_half, REPORT / "phase5e_fresh.csv")


def run_final():
    grid, data = load_rows()
    splits = walk_forward_splits(data["date"], min_train_days=CFG["history_min_train_days"])
    print(f"Final round: {len(splits)} folds, validation {splits[0][1][0]} to {splits[-1][1][-1]}")
    n_hold = CFG["inner_holdout_days"]
    rows = []
    for exp, sets in EXPERIMENTS.items():
        cols = set_columns(grid.iloc[-4000:], sets)
        parts = []
        for k, (train_days, val_days) in enumerate(splits, 1):
            inner = data[data["date"].isin(set(train_days[:-n_hold]))]
            hold = data[data["date"].isin(set(train_days[-n_hold:]))]
            train = data[data["date"].isin(set(train_days))]
            val = data[data["date"].isin(set(val_days))]
            hold_total, val_total = np.zeros(len(hold)), np.zeros(len(val))
            for part in ("ridge", "lgbm"):
                h = fit_part(part, inner, hold, cols)
                hold_total += h / h.std()
                val_total += fit_part(part, train, val, cols) / h.std()
            cutoff = float(np.quantile(np.abs(hold_total), 1 - CFG["signal_share"]))
            signal = np.where(np.abs(val_total) >= cutoff, np.sign(val_total), 0).astype(int)
            parts.append(val[["fwd_ret", "trade_ret", "close", "date"]].assign(score=val_total, signal=signal, fold=k))
        oos = pd.concat(parts)
        oos.to_parquet(REPORT / f"phase5e_{exp}_oos.parquet")
        rows += scopes(exp, oos)
        print(f"{exp} done, {len(cols)} features: {cols}", flush=True)
        show(rows, REPORT / "phase5e_summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    if args.fresh:
        run_fresh()
    if args.final:
        run_final()


if __name__ == "__main__":
    main()
