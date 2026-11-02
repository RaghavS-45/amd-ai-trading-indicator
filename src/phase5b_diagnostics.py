"""
phase5b_diagnostics.py  –  where is there signal? Design data only: nothing after
explore_end (end of walk-forward fold 5) is loaded.

Pooled rank correlation of each candidate with the 10-minute forward return, with
day-clustered SEs. The mean within-day correlation is written next to it as a warning:
it is biased negative for level-like features (Nickell 1981), so it must not be used.
Candidates not in features.py were screened here and rejected.

Usage: python src/phase5b_diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dataset import build_dataset
from sessions import ET

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"


def extra_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    day = pd.Series(grid.index.tz_convert(ET).date, index=grid.index)
    last, first = grid.groupby(day)["close"].last(), grid.groupby(day)["open"].first()
    out = pd.DataFrame({"gap": day.map(np.log(first / last.shift(1))),
                        "prevday_ret": day.map(np.log(last.shift(1) / last.shift(2)))}, index=grid.index)

    def per_session(x):
        o = pd.DataFrame(index=x.index)
        ret = np.log(x["close"] / x["close"].shift(1))
        for n in (60, 120):
            o[f"ret_{n}"] = np.log(x["close"] / x["close"].shift(n))
        signed = np.sign(x["close"] - x["open"]) * x["volume"]
        for n in (10, 30):
            o[f"signed_vol_{n}"] = signed.rolling(n).sum() / x["volume"].rolling(n).sum().replace(0, np.nan)
        qqq = np.log(x["qqq_close"] / x["qqq_close"].shift(1))
        beta = ((ret * qqq).rolling(30).mean() / (qqq * qqq).rolling(30).mean().replace(0, np.nan)).clip(-1, 5)
        for n in (5, 10, 30):
            o[f"resid_qqq_{n}"] = np.log(x["close"] / x["close"].shift(n)) - beta * np.log(
                x["qqq_close"] / x["qqq_close"].shift(n))
        o["filled_10"] = x["is_filled"].astype(float).rolling(10).mean()
        return o

    return out.join(pd.concat([per_session(x) for _, x in grid.groupby(day, sort=False)]))


def pooled_ic(x: pd.Series, y: pd.Series, days: pd.Series):
    z = pd.DataFrame({"x": x.rank(), "y": y.rank(), "d": days}).dropna()
    a = (z["x"] - z["x"].mean()) / z["x"].std()
    b = (z["y"] - z["y"].mean()) / z["y"].std()
    u = a * b
    r = u.mean()
    resid = (u - r).groupby(z["d"]).sum()
    return r, np.sqrt((resid ** 2).sum()) / len(z)


def within_day_ic(x: pd.Series, y: pd.Series, days: pd.Series) -> float:
    z = pd.DataFrame({"x": x, "y": y, "d": days}).dropna()
    z = z[z.groupby("d")["x"].transform("nunique") > 2]
    if z.empty:
        return np.nan
    ranks = z.groupby("d")[["x", "y"]].rank()
    a = ranks["x"] - ranks["x"].groupby(z["d"]).transform("mean")
    b = ranks["y"] - ranks["y"].groupby(z["d"]).transform("mean")
    cov = (a * b).groupby(z["d"]).sum()
    return (cov / np.sqrt((a * a).groupby(z["d"]).sum() * (b * b).groupby(z["d"]).sum())).mean()


def main():
    end = pd.Timestamp(CFG["explore_end"]).date()
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index.tz_convert(ET).date <= end]
    data = build_dataset(grid=grid, sets=["base", "context", "intraday", "norm"])
    assert data["date"].max() <= end
    data = data.join(extra_candidates(grid).loc[data.index])

    rows = []
    for f in [c for c in data.columns if c not in ("label", "fwd_ret", "close", "date")]:
        r, se = pooled_ic(data[f], data["fwd_ret"], data["date"])
        rows.append({"feature": f, "pooled_ic": r, "se_day": se, "t": r / se,
                     "within_day_ic_biased": within_day_ic(data[f], data["fwd_ret"], data["date"])})
    table = pd.DataFrame(rows).set_index("feature").sort_values("t")
    table.to_csv(REPORT / "phase5b_ic.csv")
    print(f"Rows {len(data):,}, {data['date'].min()} to {data['date'].max()}")
    print(table.round(4).to_string())
    print(f"|t| > 2: {(table['t'].abs() > 2).sum()} of {len(table)}")

    oos = pd.read_parquet(REPORT / "phase5_E5.6_oos.parquet")
    oos = oos[oos["fold"] < CFG["confirm_from_fold"]]
    edge = oos["p_up"] - oos["p_down"]
    deciles = oos.groupby(pd.qcut(edge, 10, labels=False)).agg(
        mean_ret_bps=("fwd_ret", lambda x: x.mean() * 1e4), up_share=("fwd_ret", lambda x: (x > 0).mean()))
    deciles.insert(0, "mean_edge", edge.groupby(pd.qcut(edge, 10, labels=False)).mean())
    deciles.to_csv(REPORT / "phase5b_e56_deciles.csv")
    print("\nE5.6 by score decile, folds 1-5:\n", deciles.round(4).to_string())


if __name__ == "__main__":
    main()
