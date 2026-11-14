"""
phase7d_band_by_time.py  –  should the 80% range use one multiplier per part of the day?
Re-runs the V2 (HAR + time of day) walk-forward of Phase 5f on training data only and compares:
  A  one multiplier, from the last inner_holdout_days of each fold's training days (as live now)
  B  three multipliers (first hour, midday, last hour), same days
  C  three multipliers, from the last 2 x inner_holdout_days
  D  as B for the first and last hour, but midday keeps the all-day multiplier of A
Writes report/phase7d_summary.csv.
Usage: python src/phase7d_band_by_time.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from dataset import build_dataset
from live_model import HAR, MOVE_SETS, part_of_day
from long_history import load_sip
from volatility import interval_score, load_vol_grid, sip_targets
from walkforward import walk_forward_splits

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
RECENT_FOLDS = 4


def log_target(rv):
    return np.log(rv + CFG["rv_floor"])


def fold(data, train_days, val_days, n_hold, by_part):
    inner = data[data["date"].isin(set(train_days[:-n_hold]))]
    hold = data[data["date"].isin(set(train_days[-n_hold:]))]
    train = data[data["date"].isin(set(train_days))]
    val = data[data["date"].isin(set(val_days))]
    h = np.exp(LinearRegression().fit(inner[HAR], log_target(inner["rv_fwd"])).predict(hold[HAR]))
    smear = float(np.mean(hold["rv_fwd"] / h))
    scaled = pd.Series(np.abs(hold["ret_fwd"].to_numpy()) / np.sqrt(h * smear), index=hold.index)
    forecast = np.exp(LinearRegression().fit(train[HAR], log_target(train["rv_fwd"])).predict(val[HAR])) * smear
    out = val[["ret_fwd", "date"]].assign(part=part_of_day(val.index))
    for name, q in [("band", CFG["band_coverage"]), ("typical", CFG["typical_coverage"])]:
        if by_part:
            z = scaled.groupby(part_of_day(hold.index)).quantile(q)
            if by_part == "edges":
                z["mid"] = float(np.quantile(scaled, q))
            mult = out["part"].map(z).to_numpy()
        else:
            mult = float(np.quantile(scaled, q))
        out[name] = mult * np.sqrt(forecast)
    return out


def main():
    grid = load_vol_grid()
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=MOVE_SETS)
    data = data.join(sip_targets(grid.index, load_sip()).loc[data.index])
    data = data[data["rv_fwd"].notna() & data["ret_fwd"].notna()]
    splits = walk_forward_splits(data["date"], min_train_days=CFG["history_min_train_days"])
    n = CFG["inner_holdout_days"]
    variants = {"A one multiplier": (n, False), "B three, 30 days": (n, True), "C three, 60 days": (2 * n, True),
                "D edges only": (n, "edges")}
    rows = []
    for name, (n_hold, by_part) in variants.items():
        oos = pd.concat([fold(data, tr, va, n_hold, by_part).assign(fold=k) for k, (tr, va) in enumerate(splits, 1)])
        hour = oos.index.tz_convert("America/New_York").hour
        inside = (np.abs(oos["ret_fwd"]) <= oos["band"]).astype(float)
        typ = (np.abs(oos["ret_fwd"]) <= oos["typical"]).astype(float)
        score = interval_score(oos["ret_fwd"].to_numpy(), oos["band"].to_numpy(), 1 - CFG["band_coverage"]) * 1e4
        for scope, m in [("all", np.ones(len(oos), bool)), ("recent", (oos["fold"] > len(splits) - RECENT_FOLDS).to_numpy())]:
            by_hour = inside[m].groupby(hour[m]).mean()
            daily = inside[m].groupby(oos["date"][m]).mean()
            rows.append({"variant": name, "scope": scope, "rows": int(m.sum()), "coverage": inside[m].mean(),
                         **{f"h{h}": v for h, v in by_hour.items()},
                         "worst_gap": float((by_hour - CFG["band_coverage"]).abs().max()),
                         "typical": typ[m].mean(), "typ_worst_gap": float((typ[m].groupby(hour[m]).mean()
                                                                           - CFG["typical_coverage"]).abs().max()),
                         "width_bps": float(2 * oos["band"][m].mean() * 1e4), "interval_score_bps": float(score[m].mean()),
                         "day_sd": float(daily.std()), "day_p05": float(daily.quantile(0.05))})
    table = pd.DataFrame(rows).set_index(["variant", "scope"])
    table.to_csv(ROOT / "report" / "phase7d_summary.csv")
    pd.set_option("display.width", 250)
    print(f"{len(splits)} folds, {data['date'].min()} to {data['date'].max()}\n", (table * 1).round(4).to_string())


if __name__ == "__main__":
    main()
