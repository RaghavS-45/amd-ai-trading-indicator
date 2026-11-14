"""
phase6b_lgbm_live.py  –  the live model after the switch to LightGBM ranges (EXPERIMENTS.md Phase 7f),
scored on the hold-out period with the Phase 6 code. This is a report, not a choice: LightGBM was
promoted because of the Phase 6 comparison on these same days, so its numbers here are not an
untouched test. The live record from 2026-11-16 on is.

Checks first that the HAR baseline inside the new model gives exactly the Phase 6 ranges.

Usage: python src/phase6b_lgbm_live.py --dry-run   (last training days; also compares with the live engine)
       python src/phase6b_lgbm_live.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluate import BPS, clustered_se
from live_model import load_model
from phase6_final_test import add_outcomes, app_rows, check_against_engine, load_inputs
from volatility import qlike

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"


def table(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows["ret_fwd"].notna() & rows["rv_fwd"].notna()]
    move = np.abs(x["ret_fwd"])
    month = pd.Series([f"{d:%Y-%m}" for d in x["date"]], index=x.index)
    scopes = [("all", np.ones(len(x), dtype=bool))]
    scopes += [(f"part: {p}", (x["part"] == p).to_numpy()) for p in ("first", "mid", "last")]
    scopes += [(f"month: {m}", (month == m).to_numpy()) for m in sorted(month.unique())]
    y = np.log(x["rv_fwd"] + CFG["rv_floor"])
    out = []
    for name, m in scopes:
        inside = (move[m] <= x["band_hw"][m]).to_numpy(dtype=float)
        gain = pd.Series(qlike(x["rv_fwd"][m].to_numpy(), x["forecast_rv"][m].to_numpy())
                         - qlike(x["rv_fwd"][m].to_numpy(), x["har_forecast_rv"][m].to_numpy()),
                         index=x.index[m]).groupby(x["date"][m]).mean()
        r2 = lambda f: float(1 - ((y[m] - np.log(f[m])) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())
        out.append({"scope": name, "rows": int(m.sum()), "days": x["date"][m].nunique(),
                    "on_backup": float((x["move_model"][m] == "har").mean()),
                    "in_80_range": inside.mean(), "se_day": clustered_se(inside, x["date"][m].to_numpy()),
                    "in_typical": float((move[m] <= x["typical_hw"][m]).mean()),
                    "width_bps": float(2 * x["band_hw"][m].mean() * BPS),
                    "har_in_80_range": float((move[m] <= x["har_band_hw"][m]).mean()),
                    "har_width_bps": float(2 * x["har_band_hw"][m].mean() * BPS),
                    "r2_log": r2(x["forecast_rv"]), "har_r2_log": r2(x["har_forecast_rv"]),
                    "qlike_vs_har": float(gain.mean()), "qlike_vs_har_se": float(gain.std(ddof=1) / np.sqrt(len(gain)))})
    return pd.DataFrame(out).set_index("scope")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run
    model = load_model()
    grid, sip, news, days = load_inputs(dry)
    rows, _ = app_rows(grid, news, days, model)
    rows = add_outcomes(rows, grid, sip)
    print(f"{'DRY RUN on training days' if dry else 'Hold-out period'}: {len(rows):,} rows, {days[0]} to {days[-1]}")
    if dry:
        check_against_engine(rows, pd.Timestamp(CFG["replay_default_day"]).date(), model)
    t = table(rows)
    if not dry:
        p6 = pd.read_csv(REPORT / "phase6_ranges.csv", index_col="scope")
        assert np.allclose(t["har_in_80_range"], p6["in_80_range"]) and np.allclose(t["har_width_bps"], p6["width_bps"])
        print("HAR baseline inside the new model = the Phase 6 ranges, exactly.")
        t.to_csv(REPORT / "phase6b_ranges.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()
