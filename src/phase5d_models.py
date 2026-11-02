"""
phase5d_models.py  –  pre-registered Phase 5d experiments (see EXPERIMENTS.md).
Same models as Phase 5b, trained on AMD plus peers, judged only on AMD rows. The
validation folds, costs, metrics and bar are unchanged.

Usage: python src/phase5d_models.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dataset import build_dataset
from evaluate import evaluate_signals
from peers import peer_grid, split_rows
from phase4_baselines import print_table
from phase5_models import set_columns
from phase5b_models import delayed_return, fit_score, report_scopes
from sessions import ET
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
SETS = ["base", "norm"]
KEEP = ["fwd_ret", "volatility", "label", "close", "date", "symbol"]

TOP = CFG["signal_share"]
EXPERIMENTS = {
    "E5d.1": dict(parts=["ridge"], share=1.0),
    "E5d.2": dict(parts=["ridge"], share=TOP),
    "E5d.3": dict(parts=["lgbm"], share=TOP),
    "E5d.4": dict(parts=["ridge", "lgbm"], share=TOP),
}


def load_pool(grid: pd.DataFrame, cols: list) -> pd.DataFrame:
    frames = [build_dataset(end_date=CFG["train_end"], grid=grid, sets=SETS).assign(symbol=CFG["asset"])]
    for sym in CFG["peer_symbols"]:
        raw = pd.read_parquet(ROOT / CFG["raw_data_template"].format(sym=sym.lower()))
        peer = peer_grid(raw, grid)
        real = (~peer["is_filled"]).mean()
        keep = real >= CFG["peer_min_coverage"]
        print(f"{sym:<5} real bars {real:.1%}, grid rows {len(peer):,}, {'used' if keep else 'LEFT OUT'}")
        if keep:
            frames.append(build_dataset(end_date=CFG["train_end"], grid=peer, sets=SETS).assign(symbol=sym))
    return pd.concat([f[cols + KEEP] for f in frames])


def main():
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index.tz_convert(ET).date <= pd.Timestamp(CFG["train_end"]).date()]
    cols = set_columns(grid, ["norm"])
    pool = load_pool(grid, cols)
    asset = pool[pool["symbol"] == CFG["asset"]]
    splits = walk_forward_splits(asset["date"])
    print(f"Pool {len(pool):,} rows, AMD {len(asset):,}, symbols {pool['symbol'].nunique()}, folds {len(splits)}")
    delay = delayed_return(grid).loc[asset.index]

    # score, holdout std and holdout scores per model part and fold, shared across experiments
    fits = {}
    for k, (train_days, val_days) in enumerate(splits, 1):
        inner, hold, train, val = split_rows(pool, train_days, val_days, CFG["inner_holdout_days"])
        assert set(val["symbol"]) == {CFG["asset"]} and train["date"].max() < val["date"].min()
        for part in ("ridge", "lgbm"):
            h = fit_score(part, inner, hold, cols)
            fits[part, k] = (h / h.std(), fit_score(part, train, val, cols) / h.std())
        print(f"fold {k} fitted on {len(train):,} rows", flush=True)

    control = pd.read_parquet(REPORT / "phase5_E5.6_oos.parquet").join(delay.rename("delay_ret"), how="inner")
    rows = report_scopes("E5.6_control", control)
    for exp, spec in EXPERIMENTS.items():
        parts, per_fold = [], []
        for k, (train_days, val_days) in enumerate(splits, 1):
            _, _, _, val = split_rows(pool, train_days, val_days, CFG["inner_holdout_days"])
            hold_score = sum(fits[p, k][0] for p in spec["parts"])
            score = sum(fits[p, k][1] for p in spec["parts"])
            cutoff = 0.0 if spec["share"] >= 1 else float(np.quantile(np.abs(hold_score), 1 - spec["share"]))
            signal = np.where(np.abs(score) >= cutoff, np.sign(score), 0).astype(int)
            res = evaluate_signals(val, signal, len(val_days))
            res.update(fold=k, val_start=val_days[0], val_end=val_days[-1], cutoff=cutoff)
            per_fold.append(res)
            parts.append(val[["fwd_ret", "label", "close", "date"]].assign(
                delay_ret=delay.loc[val.index], score=score, signal=signal, fold=k))
        per_fold = pd.DataFrame(per_fold).set_index("fold")
        per_fold.to_csv(REPORT / f"phase5d_{exp}_folds.csv")
        oos = pd.concat(parts)
        oos.to_parquet(REPORT / f"phase5d_{exp}_oos.parquet")
        rows += report_scopes(exp, oos)
        print(f"\n{exp} per fold")
        print_table(per_fold, "Fold")

    table = pd.DataFrame(rows).set_index(["experiment", "scope"])
    table.to_csv(REPORT / "phase5d_summary.csv")
    show = ["signals_per_day", "dir_acc", "dir_se_day", "dir_bar", "trades_per_day", "bps_per_trade_1x",
            "delay_bps_1x", "bps_per_trade_3x", "pnl_1x", "pass"]
    pd.set_option("display.width", 220)
    print("\n", table[show].round(4).to_string())


if __name__ == "__main__":
    main()
