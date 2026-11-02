"""
phase5c_diagnostics.py  –  does anything follow AMD-named news? Design data only:
nothing after explore_end is read. Numbers are logged in EXPERIMENTS.md, Phase 5c.

Usage: python src/phase5c_diagnostics.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dataset import build_dataset
from news import first_bar, load_scored
from sessions import ET

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def main():
    end = pd.Timestamp(CFG["explore_end"]).date()
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    grid = grid[grid.index.tz_convert(ET).date <= end]
    data = build_dataset(grid=grid, sets=["base"])
    news = load_scored()
    news = news[pd.to_datetime(news["created_at"], utc=True).dt.tz_convert(ET).dt.date <= end]
    named = news["headline"].str.lower().str.contains(CFG["news_named_pattern"], regex=True)
    edge = news["fb_pos"] - news["fb_neg"]
    groups = {"all": named | True, "named": named, "named_focused": named & (news["n_symbols"] <= 2),
              "named_pos": named & (edge > 0), "named_neg": named & (edge < 0)}
    rows = []
    for label, mask in groups.items():
        bars = first_bar(news[mask]).drop_duplicates()
        x = data.loc[data.index.intersection(bars)]
        up = (x["fwd_ret"] > 0).astype(float)
        resid = (up - up.mean()).groupby(x["date"]).sum()
        rows.append({"group": label, "events": len(x), "days": x["date"].nunique(), "up_rate": up.mean(),
                     "se_day": np.sqrt((resid ** 2).sum()) / len(x),
                     "mean_ret_bps": x["fwd_ret"].mean() * 1e4, "mean_abs_bps": x["fwd_ret"].abs().mean() * 1e4})
    table = pd.DataFrame(rows).set_index("group")
    table.to_csv(ROOT / "report" / "phase5c_events.csv")
    print(f"All design rows: up rate {(data['fwd_ret'] > 0).mean():.3f}, "
          f"mean abs {data['fwd_ret'].abs().mean() * 1e4:.1f} bps")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
