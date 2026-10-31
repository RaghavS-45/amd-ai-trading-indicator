"""
eda.py
Exploratory statistics for choosing move_threshold. Uses the project's own
compute_features(), make_labels() and tradable_mask(), on rows where every
feature is non-NaN and the label exists.

Figures are written to report/figures/.

Usage:
    python src/eda.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import labels
from dataset import forward_return, usable_mask
from features import compute_features
from sessions import ET

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
FIG_DIR = ROOT / "report" / "figures"

THRESHOLDS_BPS = [5, 10, 15, 20, 30]
BPS = 1e4

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = "#2a78d6"


def labels_at(df: pd.DataFrame, threshold: float) -> pd.Series:
    """make_labels() with move_threshold temporarily overridden."""
    saved = labels.CFG["move_threshold"]
    labels.CFG["move_threshold"] = threshold
    try:
        return labels.make_labels(df)
    finally:
        labels.CFG["move_threshold"] = saved


def class_shares(lab: pd.Series) -> dict:
    n = len(lab)
    return {name: (lab == v).sum() / n for name, v in [("Up", 1), ("Down", -1), ("Neutral", 0)]}


def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d0")
    ax.grid(axis="y", color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_histogram(fwd_bps: pd.Series, thr_bps: float, path: Path):
    lim = float(np.ceil(4 * fwd_bps.std() / 10) * 10)
    outside = (fwd_bps.abs() > lim).mean()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.hist(fwd_bps.clip(-lim, lim), bins=np.arange(-lim, lim + 2, 2),
            color=SERIES, edgecolor=SURFACE, linewidth=1)
    for x in (-thr_bps, thr_bps):
        ax.axvline(x, color=INK_2, linestyle="--", linewidth=1.2)
    top = ax.get_ylim()[1]
    ax.text(thr_bps + 1, top * 0.95, f"+{thr_bps:g} bps", color=INK_2, fontsize=9)
    ax.text(-thr_bps - 1, top * 0.95, f"−{thr_bps:g} bps", color=INK_2, fontsize=9, ha="right")
    style(ax, f"{CFG['horizon_bars']}-minute forward log return (usable rows)",
          f"Forward return, bps (clipped at ±{lim:g}; {outside:.2%} of rows outside)",
          "Rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_monthly_fill(fill_pct: pd.Series, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    x = np.arange(len(fill_pct))
    ax.bar(x, fill_pct.to_numpy(), width=0.8, color=SERIES, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(fill_pct.index, rotation=60, ha="right")
    worst = int(np.argmax(fill_pct.to_numpy()))
    ax.text(worst, fill_pct.iloc[worst], f"{fill_pct.iloc[worst]:.1f}%",
            ha="center", va="bottom", color=INK, fontsize=9)
    style(ax, "Filled (synthetic) bars by month, full grid", "Month", "Filled bars, % of grid rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    feat = compute_features(grid)
    lab = labels.make_labels(grid)
    fwd = forward_return(grid)

    # Our forward return must agree with make_labels() (same NaNs, same classes)
    assert (lab.isna().to_numpy() == fwd.isna().to_numpy()).all()
    thr = CFG["move_threshold"]
    ref = np.where(fwd > thr, 1, np.where(fwd < -thr, -1, 0))
    assert (lab.dropna().to_numpy() == ref[lab.notna().to_numpy()]).all()

    usable = usable_mask(feat, lab)
    df = grid[usable]
    fwd_u = fwd[usable]
    fwd_bps = fwd_u * BPS
    days = pd.Series(df.index.tz_convert(ET).date, index=df.index)

    print("── Usable rows ──────────────────────────────────────────")
    print(f"  Grid rows       : {len(grid):,}")
    print(f"  Usable rows     : {len(df):,}  ({len(df) / len(grid):.1%})")
    print(f"  Usable days     : {days.nunique()}")
    print(f"  ET time range   : {df.index.tz_convert(ET).time.min()} – {df.index.tz_convert(ET).time.max()}")

    print(f"\n── {CFG['horizon_bars']}-minute forward log return ──────────────────────")
    print(f"  Mean : {fwd_bps.mean():+.3f} bps")
    print(f"  Std  : {fwd_bps.std():.2f} bps")
    print(f"  Percentiles (bps): " + ", ".join(
        f"p{q}={fwd_bps.quantile(q / 100):+.1f}" for q in (1, 5, 25, 50, 75, 95, 99)))

    rt_dollars = 2 * (CFG["fee_per_share"] + CFG["slippage_per_share"])
    cost_bps = rt_dollars / df["close"] * BPS
    print("\n── Assumed round-trip cost ──────────────────────────────")
    print(f"  2 × (fee {CFG['fee_per_share']} + slippage {CFG['slippage_per_share']}) = ${rt_dollars:.3f}/share")
    print(f"  At median close ${df['close'].median():.2f}: {rt_dollars / df['close'].median() * BPS:.2f} bps"
          f"  (range {cost_bps.min():.2f}–{cost_bps.max():.2f} bps over the period)")

    print("\n── Class balance by move threshold ──────────────────────")
    print(f"  {'Threshold':>10}  {'Up':>7}  {'Down':>7}  {'Neutral':>8}")
    for t in THRESHOLDS_BPS:
        s = class_shares(labels_at(grid, t / BPS)[usable])
        mark = "  ← config" if abs(t / BPS - thr) < 1e-12 else ""
        print(f"  {t:>6} bps  {s['Up']:>7.1%}  {s['Down']:>7.1%}  {s['Neutral']:>8.1%}{mark}")

    past = feat.loc[usable, "ret_10"]
    print("\n── Past 10-min return vs forward 10-min return ──────────")
    print(f"  Pearson  : {past.corr(fwd_u):+.4f}")
    print(f"  Spearman : {past.corr(fwd_u, method='spearman'):+.4f}")

    print("\n── Train vs test period ─────────────────────────────────")
    print(f"  {'Period':<28} {'Rows':>8}  {'Share >0':>8}  {'Std bps':>7}  {'Up@cfg':>6}  {'Down@cfg':>8}")
    lab_u = lab[usable]
    for name, a, b in [("train", CFG["train_start"], CFG["train_end"]),
                       ("test", CFG["test_start"], CFG["test_end"])]:
        sel = (days >= pd.Timestamp(a).date()) & (days <= pd.Timestamp(b).date())
        f = fwd_bps[sel]
        s = class_shares(lab_u[sel])
        print(f"  {name + ' ' + a + '→' + b:<28} {sel.sum():>8,}  {(f > 0).mean():>8.1%}  "
              f"{f.std():>7.2f}  {s['Up']:>6.1%}  {s['Down']:>8.1%}")

    month = grid.index.tz_convert(ET).strftime("%Y-%m")
    fill_pct = grid["is_filled"].groupby(month).mean() * 100
    print("\n── Filled-bar share by month (full grid) ────────────────")
    for m, v in fill_pct.items():
        print(f"  {m}  {v:5.2f}%")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = FIG_DIR / "forward_return_hist.png"
    fill_path = FIG_DIR / "monthly_fill.png"
    plot_histogram(fwd_bps, thr * BPS, hist_path)
    plot_monthly_fill(fill_pct, fill_path)
    print(f"\nSaved → {hist_path.relative_to(ROOT)}")
    print(f"Saved → {fill_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
