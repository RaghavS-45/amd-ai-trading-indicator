"""
phase4_baselines.py  –  walk-forward evaluation of always-Up, an RSI rule and
logistic regression on the train period only (hold-out test never loaded).

Writes report/phase4_results.csv (per fold) and report/phase4_summary.csv (pooled).

Usage:
    python src/phase4_baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset, feature_columns
from evaluate import evaluate_signals
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def always_up(train: pd.DataFrame, val: pd.DataFrame, cols) -> np.ndarray:
    return np.ones(len(val), dtype=int)


def rsi_rule(train: pd.DataFrame, val: pd.DataFrame, cols) -> np.ndarray:
    rsi = val["rsi"].to_numpy()
    return np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))


def logistic(train: pd.DataFrame, val: pd.DataFrame, cols) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=CFG["random_seed"]),
    )
    model.fit(train[cols], train["label"])
    return model.predict(val[cols]).astype(int)


MODELS = {"always_up": always_up, "rsi_rule": rsi_rule, "logistic": logistic}

SHOW = [
    ("signals_per_day", "Sig/day", "{:.1f}"),
    ("dir_acc", "Dir acc", "{:.2%}"),
    ("dir_se_day", "±SE", "{:.2%}"),
    ("strict_acc", "Strict", "{:.2%}"),
    ("strict_se_day", "±SE", "{:.2%}"),
    ("trades_per_day", "Trd/day", "{:.1f}"),
    ("trade_dir_acc", "Trd dir", "{:.2%}"),
    ("pnl_1x", "P&L 1x $", "{:,.0f}"),
    ("bps_per_trade_1x", "bps/trd", "{:+.2f}"),
    ("pnl_3x", "P&L 3x $", "{:,.0f}"),
    ("bps_per_trade_3x", "bps/trd", "{:+.2f}"),
]


def print_table(df: pd.DataFrame, index_name: str):
    header = f"  {index_name:<12}" + "".join(f"{h:>11}" for _, h, _ in SHOW)
    print(header)
    for idx, row in df.iterrows():
        cells = "".join(f"{fmt.format(row[k]):>11}" for k, _, fmt in SHOW)
        print(f"  {str(idx):<12}{cells}")


def main():
    data = build_dataset(end_date=CFG["train_end"])
    cols = feature_columns(data)
    splits = walk_forward_splits(data["date"])

    print(f"Rows (train period, usable): {len(data):,}   days: {data['date'].nunique()}")
    print(f"Folds: {len(splits)}   val window: {CFG['val_window_days']} days   "
          f"gap: {CFG['wf_gap_days']} day(s)   min train: {CFG['wf_min_train_days']} days")
    print(f"Validation span: {splits[0][1][0]} → {splits[-1][1][-1]}")
    print(f"Costs: ${CFG['fee_per_share']} fee + ${CFG['slippage_per_share']} slippage per share "
          f"per side, {CFG['shares_per_trade']} shares/trade; 3x = triple costs")

    per_fold, pooled_val, pooled_sig = [], [], {m: [] for m in MODELS}
    for k, (train_days, val_days) in enumerate(splits, 1):
        train = data[data["date"].isin(set(train_days))]
        val = data[data["date"].isin(set(val_days))]
        assert train["date"].max() < val["date"].min()
        pooled_val.append(val)
        for name, fn in MODELS.items():
            sig = fn(train, val, cols)
            pooled_sig[name].append(sig)
            res = evaluate_signals(val, sig, len(val_days))
            res.update(model=name, fold=k, train_days=len(train_days),
                       val_start=val_days[0], val_end=val_days[-1])
            per_fold.append(res)

    val_all = pd.concat(pooled_val)
    n_val_days = val_all["date"].nunique()
    summary = []
    for name in MODELS:
        res = evaluate_signals(val_all, np.concatenate(pooled_sig[name]), n_val_days)
        res["model"] = name
        summary.append(res)

    summary = pd.DataFrame(summary).set_index("model")
    per_fold = pd.DataFrame(per_fold)

    print(f"\n── Pooled over {len(splits)} folds ({n_val_days} validation days, "
          f"{len(val_all):,} rows) ──")
    print_table(summary, "Model")
    print("  ±SE = day-clustered standard error. Naive binomial SE for comparison:")
    for name, row in summary.iterrows():
        print(f"    {name:<10} dir {row['dir_se_naive']:.2%}   strict {row['strict_se_naive']:.2%}   "
              f"trade-dir (binomial on {int(row['trades']):,} trades) {row['trade_dir_se']:.2%}")
    print(f"  Share of validation rows with fwd_ret > 0: {(val_all['fwd_ret'] > 0).mean():.2%}")

    for name in MODELS:
        print(f"\n── Per fold: {name} ──")
        t = per_fold[per_fold["model"] == name].set_index("fold")
        print_table(t, "Fold")

    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    per_fold.to_csv(out / "phase4_results.csv", index=False)
    summary.to_csv(out / "phase4_summary.csv")
    print(f"\nSaved → report/phase4_results.csv, report/phase4_summary.csv")


if __name__ == "__main__":
    main()
