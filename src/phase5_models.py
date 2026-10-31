"""
phase5_models.py  –  pre-registered Phase 5 experiments (see EXPERIMENTS.md).

E5.1-E5.6: LightGBM / logistic regression on base, context and intraday feature
sets, same walk-forward folds and metrics as Phase 4.
E5.7: confidence threshold (tau chosen on earlier folds only) + reliability
diagram on the pre-registered choice among E5.1-E5.6.

Train period only; the hold-out test is never loaded.

Usage:
    python src/phase5_models.py --all
    python src/phase5_models.py --exp E5.1 E5.2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import build_dataset
from evaluate import evaluate_signals
from features import compute_features
from phase4_baselines import print_table
from threshold import (brier_multiclass, expected_calibration_error, reliability,
                       tau_curve, threshold_signals, walk_forward_tau)
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
FIG_DIR = REPORT / "figures"
ALL_SETS = ["base", "context", "intraday"]
CLASSES = np.array([-1, 0, 1])

LGBM_COMMON = dict(objective="multiclass", random_state=CFG["random_seed"],
                   deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)
LGBM_PARAMS = {
    "config": dict(n_estimators=CFG["lgbm_n_estimators"], learning_rate=CFG["lgbm_learning_rate"],
                   num_leaves=CFG["lgbm_num_leaves"]),
    "regularized": dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                        min_child_samples=500, colsample_bytree=0.8, subsample=0.8,
                        subsample_freq=1, reg_lambda=1.0),
}

EXPERIMENTS = {
    "E5.1": ("lgbm", "config", ["base"]),
    "E5.2": ("lgbm", "regularized", ["base"]),
    "E5.3": ("lgbm", "best_of_E5.1_E5.2", ["base", "context"]),
    "E5.4": ("lgbm", "best_of_E5.1_E5.2", ["base", "intraday"]),
    "E5.5": ("lgbm", "best_of_E5.1_E5.2", ["base", "context", "intraday"]),
    "E5.6": ("logistic", None, ["base", "context", "intraday"]),
}

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ebeae6"
UP_COLOR, DOWN_COLOR = "#2a78d6", "#eb6834"
MIN_BIN_ROWS = 50  # bins with fewer rows are left off the reliability plot (kept in the table)


def summary_path() -> Path:
    return REPORT / "phase5_summary.csv"


def load_summary() -> pd.DataFrame:
    p = summary_path()
    return pd.read_csv(p, index_col="experiment") if p.exists() else pd.DataFrame()


def save_summary_row(exp: str, row: dict):
    s = load_summary()
    s = s.drop(index=exp, errors="ignore")
    s = pd.concat([s, pd.DataFrame([row], index=pd.Index([exp], name="experiment"))])
    s.sort_index().to_csv(summary_path())


def set_columns(grid: pd.DataFrame, sets) -> list:
    return list(compute_features(grid.iloc[:400], sets).columns)


def resolve_params(name: str) -> str:
    if name != "best_of_E5.1_E5.2":
        return name
    s = load_summary()
    missing = {"E5.1", "E5.2"} - set(s.index)
    if missing:
        raise SystemExit(f"Run {sorted(missing)} first (E5.3-E5.5 use the better of E5.1/E5.2).")
    best = s.loc[["E5.1", "E5.2"], "bps_per_trade_1x"].idxmax()
    return "config" if best == "E5.1" else "regularized"


def make_model(kind: str, params: str):
    if kind == "lgbm":
        return LGBMClassifier(**LGBM_COMMON, **LGBM_PARAMS[params])
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=CFG["random_seed"]))


def pass_fail(res: dict, base_rate: float) -> dict:
    bar = base_rate + 2 * res["dir_se_day"]
    ok_dir = res["dir_acc"] >= bar
    ok_bps = res["bps_per_trade_1x"] > 0
    return {"base_rate": base_rate, "dir_bar": bar, "pass_dir": ok_dir,
            "pass_bps_1x": ok_bps, "pass": bool(ok_dir and ok_bps)}


def run_model_experiment(exp: str, data: pd.DataFrame, grid: pd.DataFrame, splits):
    kind, params, sets = EXPERIMENTS[exp]
    params = resolve_params(params) if kind == "lgbm" else None
    cols = set_columns(grid, sets)
    print(f"\n{'=' * 70}\n{exp}: {kind}{' (' + params + ')' if params else ''}, "
          f"sets {'+'.join(sets)}, {len(cols)} features\n{'=' * 70}")

    per_fold, oos_parts = [], []
    for k, (train_days, val_days) in enumerate(splits, 1):
        train = data[data["date"].isin(set(train_days))]
        val = data[data["date"].isin(set(val_days))]
        model = make_model(kind, params)
        model.fit(train[cols], train["label"])
        proba = model.predict_proba(val[cols])
        assert list(model.classes_) == list(CLASSES)
        sig = CLASSES[proba.argmax(axis=1)]
        res = evaluate_signals(val, sig, len(val_days))
        res.update(fold=k, val_start=val_days[0], val_end=val_days[-1], train_days=len(train_days))
        per_fold.append(res)
        oos_parts.append(pd.DataFrame({
            "p_down": proba[:, 0], "p_neutral": proba[:, 1], "p_up": proba[:, 2],
            "signal": sig, "fold": k}, index=val.index).join(val[["fwd_ret", "label", "close", "date"]]))

    oos = pd.concat(oos_parts)
    oos.to_parquet(REPORT / f"phase5_{exp}_oos.parquet")
    per_fold = pd.DataFrame(per_fold).set_index("fold")
    per_fold.to_csv(REPORT / f"phase5_{exp}_folds.csv")

    pooled = evaluate_signals(oos, oos["signal"].to_numpy(), oos["date"].nunique())
    base_rate = float((oos["fwd_ret"] > 0).mean())
    pooled.update(pass_fail(pooled, base_rate))
    pooled.update(model=kind, params=params or "", sets="+".join(sets), n_features=len(cols),
                  brier=brier_multiclass(oos[["p_down", "p_neutral", "p_up"]].to_numpy(),
                                         oos["label"].to_numpy()))
    save_summary_row(exp, pooled)

    print_table(pd.DataFrame([pooled], index=[exp]), "Pooled")
    print(f"  Day-clustered SE: dir {pooled['dir_se_day']:.2%}, strict {pooled['strict_se_day']:.2%}   "
          f"Base rate {base_rate:.2%} → bar {pooled['dir_bar']:.2%}   "
          f"Up share of signals {pooled['up_share']:.1%}   Brier {pooled['brier']:.4f}")
    print(f"  PASS: {pooled['pass']}  (direction {pooled['pass_dir']}, bps/trade 1x > 0 {pooled['pass_bps_1x']})")
    print_table(per_fold, "Fold")


def plot_reliability(rels: dict, path: Path, title: str):
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(8, 8), facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [3, 1.3]}, sharex=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_2, linewidth=1)
    ax.text(0.62, 0.66, "perfect calibration", color=INK_2, fontsize=9, rotation=38)
    width = 0.035
    for (name, rel), color, off in zip(rels.items(), [UP_COLOR, DOWN_COLOR], [-width / 2, width / 2]):
        rel = rel[rel["count"] >= MIN_BIN_ROWS]
        ax.plot(rel["mean_pred"], rel["observed"], color=color, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        last = rel.iloc[-1]
        ax.text(last["mean_pred"] + 0.015, last["observed"], name, color=INK, fontsize=10, va="center")
        axc.bar((rel["lo"] + rel["hi"]) / 2 + off, rel["count"], width=width,
                color=color, edgecolor=SURFACE, linewidth=1)
    for a in (ax, axc):
        a.set_facecolor(SURFACE)
        a.tick_params(colors=INK_2, labelsize=9)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#d6d5d0")
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=18)
    ax.text(0, 1.01, f"Bins with fewer than {MIN_BIN_ROWS} rows not shown", transform=ax.transAxes,
            color=INK_2, fontsize=9, va="bottom")
    ax.set_ylabel("Observed frequency of the class", color=INK_2)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    axc.set_ylabel("Rows per bin", color=INK_2)
    axc.set_xlabel("Predicted probability (bin mean)", color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def run_threshold_experiment():
    s = load_summary()
    candidates = [e for e in EXPERIMENTS if e in s.index]
    if len(candidates) < len(EXPERIMENTS):
        raise SystemExit("Run E5.1-E5.6 first (E5.7 picks among them).")
    chosen = s.loc[candidates, "bps_per_trade_1x"].idxmax()
    oos = pd.read_parquet(REPORT / f"phase5_{chosen}_oos.parquet")
    min_spd = CFG["min_signals_per_day"]
    print(f"\n{'=' * 70}\nE5.7: confidence threshold on {chosen} "
          f"(highest pooled 1x bps/trade among E5.1-E5.6), min {min_spd} signals/day\n{'=' * 70}")

    # Reliability (pooled out-of-sample, all folds)
    rels = {"Up": reliability(oos["p_up"].to_numpy(), (oos["label"] == 1).to_numpy().astype(float)),
            "Down": reliability(oos["p_down"].to_numpy(), (oos["label"] == -1).to_numpy().astype(float))}
    print("\n── Reliability (pooled out-of-sample, 10 bins) ──")
    for name, rel in rels.items():
        print(f"  {name}: ECE {expected_calibration_error(rel):.4f}")
        print(rel[["lo", "hi", "count", "mean_pred", "observed"]].round(4).to_string())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIG_DIR / "reliability_phase5.png"
    plot_reliability(rels, fig_path, f"Reliability, {chosen} (pooled out-of-sample, 9 folds)")

    # Walk-forward tau: fold k uses folds < k only
    taus = walk_forward_tau(oos, min_spd)
    rows, sig_all = [], []
    eval_rows = oos[oos["fold"].isin(taus.keys())]
    for k, tau in taus.items():
        f = oos[oos["fold"] == k]
        sig = threshold_signals(f["p_down"].to_numpy(), f["p_up"].to_numpy(), tau)
        sig_all.append(sig)
        res = evaluate_signals(f, sig, f["date"].nunique()) if (sig != 0).any() else {"signals": 0}
        res.update(fold=k, tau=tau)
        rows.append(res)
    per_fold = pd.DataFrame(rows).set_index("fold")
    per_fold.to_csv(REPORT / "phase5_E5.7_folds.csv")

    pooled = evaluate_signals(eval_rows, np.concatenate(sig_all), eval_rows["date"].nunique())
    base_rate = float((eval_rows["fwd_ret"] > 0).mean())
    pooled.update(pass_fail(pooled, base_rate))
    pooled.update(model=f"threshold on {chosen}", params="", sets="", n_features=np.nan,
                  ece_up=expected_calibration_error(rels["Up"]),
                  ece_down=expected_calibration_error(rels["Down"]))
    save_summary_row("E5.7", pooled)

    print(f"\n── Walk-forward threshold, pooled folds 2-9 ({eval_rows['date'].nunique()} days) ──")
    print_table(pd.DataFrame([pooled], index=["E5.7"]), "Pooled")
    print(f"  Day-clustered SE: dir {pooled['dir_se_day']:.2%}   Base rate {base_rate:.2%} → "
          f"bar {pooled['dir_bar']:.2%}   PASS: {pooled['pass']}  (direction {pooled['pass_dir']}, "
          f"bps/trade 1x > 0 {pooled['pass_bps_1x']})")
    print("\n── Per fold (tau chosen on earlier folds) ──")
    print("  " + "  ".join(f"fold {k}: tau {t:.2f}" for k, t in taus.items()))
    print_table(per_fold, "Fold")

    curve = tau_curve(oos)
    curve.to_csv(REPORT / "phase5_E5.7_tau_curve.csv")
    print("\n── Optimistic diagnostic: tau curve over ALL folds (tau chosen in hindsight) ──")
    print(f"  {'tau':>5} {'Sig/day':>8} {'Dir acc':>8} {'±SE':>6} {'bps/trd 1x':>11} {'bps/trd 3x':>11}")
    for tau, r in curve.iterrows():
        if r["signals"] == 0:
            print(f"  {tau:>5.2f}  no signals")
            continue
        print(f"  {tau:>5.2f} {r['signals_per_day']:>8.1f} {r['dir_acc']:>8.2%} {r['dir_se_day']:>6.2%} "
              f"{r['bps_per_trade_1x']:>+11.2f} {r['bps_per_trade_3x']:>+11.2f}")
    print(f"\nSaved → {fig_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="*", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    exps = list(EXPERIMENTS) + ["E5.7"] if args.all else args.exp
    if not exps:
        parser.error("give --all or --exp E5.x ...")

    REPORT.mkdir(exist_ok=True)
    model_exps = [e for e in exps if e in EXPERIMENTS]
    if model_exps:
        grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
        data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=ALL_SETS)
        splits = walk_forward_splits(data["date"])
        print(f"Rows (train period, usable): {len(data):,}   days: {data['date'].nunique()}   "
              f"folds: {len(splits)}   validation {splits[0][1][0]} → {splits[-1][1][-1]}")
        for exp in model_exps:
            run_model_experiment(exp, data, grid, splits)
    if "E5.7" in exps:
        run_threshold_experiment()


if __name__ == "__main__":
    main()
