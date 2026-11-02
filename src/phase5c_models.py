"""
phase5c_models.py  –  pre-registered Phase 5c news experiments (see EXPERIMENTS.md).
Same rows, folds, metrics and bar as Phase 5b; pass/fail on the confirmation folds.
Control and E5c.2-E5c.4 reuse the out-of-sample E5.6 rows.

Usage: python src/phase5c_models.py --all
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from evaluate import evaluate_signals
from news import load_scored, news_columns
from phase4_baselines import print_table
from phase5_models import CLASSES, set_columns
from phase5b_models import delayed_return, report_scopes
from sessions import ET
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
SETS = ["base", "context", "intraday", "news"]


def run_logistic(data: pd.DataFrame, cols: list, splits) -> pd.DataFrame:
    parts, per_fold = [], []
    for k, (train_days, val_days) in enumerate(splits, 1):
        train = data[data["date"].isin(set(train_days))]
        val = data[data["date"].isin(set(val_days))]
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=CFG["random_seed"]))
        model.fit(train[cols], train["label"])
        signal = CLASSES[model.predict_proba(val[cols]).argmax(axis=1)]
        res = evaluate_signals(val, signal, len(val_days))
        res.update(fold=k, val_start=val_days[0], val_end=val_days[-1])
        per_fold.append(res)
        parts.append(val[["fwd_ret", "label", "close", "date", "delay_ret"]].assign(signal=signal, fold=k))
    per_fold = pd.DataFrame(per_fold).set_index("fold")
    per_fold.to_csv(REPORT / "phase5c_E5c.1_folds.csv")
    print("E5c.1 per fold")
    print_table(per_fold, "Fold")
    return pd.concat(parts)


def main():
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index.tz_convert(ET).date <= pd.Timestamp(CFG["train_end"]).date()]
    grid = grid.join(news_columns(grid.index, load_scored()))
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=SETS)
    data["delay_ret"] = delayed_return(grid).loc[data.index]
    data["bar_count"] = grid["news_count"].loc[data.index]
    data["bar_sent"] = grid["news_sent"].loc[data.index]
    cols = set_columns(grid, SETS)
    print(f"Rows {len(data):,}, {len(cols)} features, bars with a named article {(data['bar_count'] > 0).sum():,}")

    control = pd.read_parquet(REPORT / "phase5_E5.6_oos.parquet").join(
        data[["delay_ret", "bar_count", "bar_sent", "news_count_30"]], how="inner")
    assert len(control) > 0

    fade = control.assign(signal=np.where(control["bar_count"] > 0, -1, 0))
    contrary = control.assign(signal=np.where(control["bar_count"] > 0, -np.sign(control["bar_sent"]), 0).astype(int))
    veto = control.assign(signal=control["signal"].where(~((control["signal"] == 1) & (control["news_count_30"] > 0)), 0))

    rows = report_scopes("E5.6_control", control)
    rows += report_scopes("E5c.1", run_logistic(data, cols, walk_forward_splits(data["date"])))
    rows += report_scopes("E5c.2", fade)
    rows += report_scopes("E5c.3", contrary)
    rows += report_scopes("E5c.4", veto)
    table = pd.DataFrame(rows).set_index(["experiment", "scope"])
    table.to_csv(REPORT / "phase5c_summary.csv")
    show = ["signals_per_day", "up_share", "dir_acc", "dir_se_day", "dir_bar", "trades", "bps_per_trade_1x",
            "delay_bps_1x", "bps_per_trade_3x", "pnl_1x", "pass"]
    pd.set_option("display.width", 220)
    print("\n", table[show].round(4).to_string())


if __name__ == "__main__":
    main()
