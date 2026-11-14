"""
phase6_final_test.py  –  the final hold-out test (see EXPERIMENTS.md Phase 6), run once.

Scores what the frozen app would have shown on every test-period signal row, with the same
functions the live engine calls (live_model.move_forecast, adaptive_cut, direction_lean):
  ranges     – coverage of the 80% and typical ranges on SIP closes, by part of the day and month
  move size  – QLIKE and R² of the indicator's model (V2, HAR) with V4 (LightGBM) beside it
  activity   – mean |move| by calm / normal / busy and under the warning
  direction  – the arrows, always Up and the RSI rule, strict scoring (next-open entry on SIP)
               and IEX closes as on the live page
Nothing is fitted, tuned or chosen on test rows. V4 is fitted on training data as in Phase 5f.

Usage: python src/phase6_final_test.py --dry-run   (last training days: checks the code, results mean nothing)
       python src/phase6_final_test.py --final     (the test period; refuses to run twice)
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

from dataset import build_dataset, feature_columns, forward_return
from evaluate import BPS, clustered_se
from features import compute_features
from live_engine import model_id
from live_model import HAR, LEAN_SETS, MOVE_SETS, adaptive_cut, direction_lean, load_model, move_forecast, part_of_day
from live_score import lean_signal, rsi_signal
from long_history import ONE, load_sip, previous_close, tradable_return
from news import news_count_column
from phase5f_models import LGBM_PARAMS, log_target
from sessions import ET, tradable_mask
from tradable import COST, non_overlapping, score
from volatility import interval_score, load_vol_grid, prev_day_rv, qlike, seasonal_rv, sip_targets

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
REPORT = ROOT / "report"
H = pd.Timedelta(minutes=CFG["horizon_bars"])
TABLES = ["ranges", "models", "activity", "direction", "days"]
LEAK_ALARM = 0.60


def checked(path: Path) -> Path:
    """A test-period file must exist and match its checksum."""
    sha = ROOT / CFG["news_checksum_template"].format(name=path.stem)
    if not path.exists() or not sha.exists():
        sys.exit(f"Missing {path.name} or its checksum: run python src/download_test.py first.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha.read_text().strip():
        sys.exit(f"Checksum mismatch: {path.name}")
    return path


def load_inputs(dry: bool):
    """Grid, SIP bars and news. A dry run cuts everything at train_end before any other step."""
    grid = pd.read_parquet(ROOT / CFG["grid_data_path"])
    date = grid.index.tz_convert(ET).date
    news = pd.read_parquet(ROOT / CFG["news_raw_path"])
    if dry:
        grid = grid[date <= pd.Timestamp(CFG["train_end"]).date()]
        sip = pd.read_parquet(ROOT / CFG["sip_check_path"])
        days = sorted(set(grid.index.tz_convert(ET).date))[-CFG["phase6_dry_days"]:]
    else:
        grid = grid[date <= pd.Timestamp(CFG["test_end"]).date()]
        sip = pd.read_parquet(checked(ROOT / CFG["test_sip_path"]))
        news = pd.concat([news, pd.read_parquet(checked(ROOT / CFG["test_news_path"]))], ignore_index=True)
        days = sorted(d for d in set(grid.index.tz_convert(ET).date) if d >= pd.Timestamp(CFG["test_start"]).date())
    sip.index = pd.to_datetime(sip.index, utc=True)
    return grid, sip, news, days


def app_rows(grid: pd.DataFrame, news: pd.DataFrame, days: list, model: dict):
    """What the app shows on every signal row of `days`, built as live_engine does."""
    grid = grid.copy()
    grid["prev_day_rv"] = prev_day_rv(grid["close"])
    grid["seas_rv"] = seasonal_rv(grid["close"])
    grid["prev_close"] = previous_close(grid["close"])
    grid["qqq_prev_close"] = previous_close(grid["qqq_close"])
    grid["news_count"] = news_count_column(grid.index, news)
    all_days = sorted(set(grid.index.tz_convert(ET).date))
    first = all_days[max(all_days.index(days[0]) - CFG["live_history_sessions"], 0)]
    grid = grid[grid.index.tz_convert(ET).date >= first]

    feat = compute_features(grid, MOVE_SETS + LEAN_SETS)
    date = pd.Series(feat.index.tz_convert(ET).date, index=feat.index)
    was_signal = tradable_mask(feat.index) & (feat.index + H).isin(grid.index)   # target inside the session
    move = move_forecast(model, feat[was_signal].dropna(subset=HAR))
    parts = []
    for day in days:
        earlier = sorted(d for d in set(date) if d < day)[-CFG["live_history_sessions"]:]
        cut = adaptive_cut(model, feat[was_signal & date.isin(earlier).to_numpy()])
        rows = move.index[date.loc[move.index] == day]
        parts.append(direction_lean(model, feat.loc[rows], cut).assign(cut=cut))
    lean = pd.concat(parts)
    out = move[date.loc[move.index].isin(days).to_numpy()].join(lean)
    out["date"], out["close"], out["rsi"] = date.loc[out.index], grid["close"].loc[out.index], feat["rsi"].loc[out.index]
    out["part"] = part_of_day(out.index)
    return out, feat


def add_outcomes(rows: pd.DataFrame, grid: pd.DataFrame, sip: pd.DataFrame) -> pd.DataFrame:
    """SIP targets for the ranges, the tradable return (SIP open t+1 to SIP close t+10), and the IEX move."""
    rows = rows.join(sip_targets(grid.index, sip).loc[rows.index])
    rows["trade_ret"] = tradable_return(rows.index, sip)
    rows["entry"] = sip["open"].reindex(rows.index + ONE).to_numpy()
    rows["iex_ret"] = forward_return(grid).loc[rows.index]
    return rows


def fit_v4(model_cols=None):
    """V4 as in the last Phase 5f fold: smearing factor and band multiplier from the last
    inner_holdout_days of training, then refit on all training days. Training data only."""
    grid = load_vol_grid()
    data = build_dataset(end_date=CFG["train_end"], grid=grid, sets=["vol"])
    data = data.join(sip_targets(grid.index, load_sip()).loc[data.index])
    data = data[data["rv_fwd"].notna() & data["ret_fwd"].notna()]
    cols = model_cols or [c for c in feature_columns(data) if c not in ("rv_fwd", "ret_fwd")]
    days = sorted(data["date"].unique())
    hold_days = set(days[-CFG["inner_holdout_days"]:])
    inner, hold = data[~data["date"].isin(hold_days)], data[data["date"].isin(hold_days)]
    h = np.exp(LGBMRegressor(**LGBM_PARAMS).fit(inner[cols], log_target(inner["rv_fwd"])).predict(hold[cols]))
    smear = float(np.mean(hold["rv_fwd"] / h))
    z = float(np.quantile(np.abs(hold["ret_fwd"]) / np.sqrt(h * smear), CFG["band_coverage"]))
    full = LGBMRegressor(**LGBM_PARAMS).fit(data[cols], log_target(data["rv_fwd"]))
    return {"model": full, "columns": cols, "smear": smear, "z_band": z, "rows": len(data), "last_day": str(days[-1])}


# ── tables ──────────────────────────────────────────────────────────
def range_table(rows: pd.DataFrame, model: dict) -> pd.DataFrame:
    x = rows[rows["ret_fwd"].notna()]
    move = np.abs(x["ret_fwd"])
    old_hw = model["move"]["z_band"] * np.sqrt(x["forecast_rv"])   # one all-day multiplier, as before Phase 7d
    month = pd.Series([f"{d:%Y-%m}" for d in x["date"]], index=x.index)
    scopes = [("all", np.ones(len(x), dtype=bool))]
    scopes += [(f"part: {p}", (x["part"] == p).to_numpy()) for p in ("first", "mid", "last")]
    scopes += [(f"month: {m}", (month == m).to_numpy()) for m in sorted(month.unique())]
    out = []
    for name, m in scopes:
        inside = (move[m] <= x["band_hw"][m]).to_numpy(dtype=float)
        out.append({"scope": name, "rows": int(m.sum()), "days": x["date"][m].nunique(),
                    "in_80_range": inside.mean(), "se_day": clustered_se(inside, x["date"][m].to_numpy()),
                    "in_typical": float((move[m] <= x["typical_hw"][m]).mean()),
                    "in_80_one_multiplier": float((move[m] <= old_hw[m]).mean()),
                    "width_bps": float(2 * x["band_hw"][m].mean() * BPS),
                    "interval_score_bps": float(interval_score(x["ret_fwd"][m].to_numpy(), x["band_hw"][m].to_numpy(),
                                                               1 - CFG["band_coverage"]).mean() * BPS)})
    return pd.DataFrame(out).set_index("scope")


def model_table(rows: pd.DataFrame, v4: dict, feat: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows["rv_fwd"].notna() & rows["ret_fwd"].notna()]
    x = x[feat.loc[x.index, v4["columns"]].notna().all(axis=1).to_numpy()]   # same rows for both models
    f4 = np.exp(v4["model"].predict(feat.loc[x.index, v4["columns"]])) * v4["smear"]
    forecasts = {"V2 HAR (the indicator)": (x["forecast_rv"].to_numpy(), x["band_hw"].to_numpy()),
                 "V4 LightGBM (comparison)": (f4, v4["z_band"] * np.sqrt(f4))}
    y = np.log(x["rv_fwd"] + CFG["rv_floor"])
    base = qlike(x["rv_fwd"].to_numpy(), forecasts["V2 HAR (the indicator)"][0])
    out = []
    for name, (f, hw) in forecasts.items():
        loss = qlike(x["rv_fwd"].to_numpy(), f)
        daily = pd.Series(loss - base, index=x.index).groupby(x["date"]).mean()
        diff, se = float(daily.mean()), float(daily.std(ddof=1) / np.sqrt(len(daily)))
        out.append({"model": name, "rows": len(x), "days": x["date"].nunique(), "qlike": float(loss.mean()),
                    "qlike_vs_V2": diff, "qlike_vs_V2_se": se, "beats_V2": bool(diff + 2 * se < 0),
                    "r2_log": float(1 - ((y - np.log(f)) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                    "in_80_range": float((np.abs(x["ret_fwd"]) <= hw).mean()), "width_bps": float(2 * hw.mean() * BPS),
                    "interval_score_bps": float(interval_score(x["ret_fwd"].to_numpy(), hw,
                                                               1 - CFG["band_coverage"]).mean() * BPS)})
    return pd.DataFrame(out).set_index("model")


def activity_table(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows["ret_fwd"].notna()].assign(move_bps=lambda d: np.abs(d["ret_fwd"]) * BPS)
    groups = [(f"level: {lv}", x["level"] == lv) for lv in ("calm", "normal", "busy")]
    groups += [("warning: off", ~x["warning"]), ("warning: on", x["warning"])]
    return pd.DataFrame([{"group": g, "rows": int(m.sum()), "share": float(m.mean()),
                          "mean_move_bps": float(x["move_bps"][m].mean()),
                          "median_move_bps": float(x["move_bps"][m].median()),
                          "in_80_range": float((np.abs(x["ret_fwd"][m]) <= x["band_hw"][m]).mean())}
                         for g, m in groups]).set_index("group")


def score_signal(rows: pd.DataFrame, signal: np.ndarray, ret: str, entry: str) -> dict:
    """tradable.score plus paper profit in dollars on shares_per_trade shares."""
    x = rows.assign(signal=signal, trade_ret=rows[ret])
    x = x[x["trade_ret"].notna() & x[entry].notna()]
    res = score(x)
    t = x.iloc[non_overlapping(x.index, x["signal"].to_numpy())]
    gain = t["signal"] * t[entry] * (np.exp(t["trade_ret"]) - 1)
    res["pnl_usd"] = float(CFG["shares_per_trade"] * (gain - COST).sum())
    return res


def direction_table(rows: pd.DataFrame, model: dict) -> pd.DataFrame:
    x = rows[rows["lean"].notna()]   # rows where the app had a direction model output
    edge = (x["p_up"] - x["p_down"]).to_numpy()
    p = x[["p_down", "p_flat", "p_up"]].to_numpy()
    fixed = model["lean"]["cut"]
    signals = {"Indicator arrows (as shown)": lean_signal(x["lean"]),
               "Arrows, fixed cut (before Phase 7c)": np.where(edge >= fixed, 1, np.where(edge <= -fixed, -1, 0)),
               "E5.6 most likely class": np.array([-1, 0, 1])[p.argmax(axis=1)],
               "Always Up": np.ones(len(x), dtype=int),
               "RSI rule": rsi_signal(x["rsi"])}
    out = []
    for scoring, ret, entry in [("strict: next open, SIP", "trade_ret", "entry"), ("IEX closes (live page)", "iex_ret", "close")]:
        for name, sig in signals.items():
            out.append(dict(score_signal(x, sig, ret, entry), signal=name, scoring=scoring))
    return pd.DataFrame(out).set_index(["scoring", "signal"])


def day_table(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows["ret_fwd"].notna()].assign(
        inside=lambda d: np.abs(d["ret_fwd"]) <= d["band_hw"], arrow=lambda d: d["lean"].isin(["weak up", "weak down"]),
        right=lambda d: np.sign(d["trade_ret"]) == lean_signal(d["lean"]))
    g = x.groupby("date")
    return pd.DataFrame({"signals": g.size(), "in_80_range": g["inside"].mean(), "arrows": g["arrow"].sum(),
                         "arrow_share": g["arrow"].mean(), "arrows_right": x[x["arrow"]].groupby("date")["right"].mean(),
                         "cut": g["cut"].first(), "warning_share": g["warning"].mean()})


# ── dry-run check: the same numbers as the live engine ──────────────
def check_against_engine(rows: pd.DataFrame, day, model: dict) -> None:
    from live_engine import Engine
    from live_sources import ReplaySource
    from signal_store import Store
    eng = Engine(ReplaySource(), Store(":memory:"), model, lambda: pd.Timestamp.now(tz="UTC"))
    eng.open_session(pd.Timestamp(day))
    eng.step(eng.last_bar)
    sig = eng.store.signals(eng.session).set_index("ts")
    sig.index = pd.to_datetime(sig.index, utc=True)
    mine = rows[rows["date"] == day]
    assert sig.index.equals(mine.index), "signal rows differ from the engine"
    assert np.allclose(sig["band_bps"], mine["band_hw"] * BPS) and np.allclose(sig["typical_bps"], mine["typical_hw"] * BPS)
    assert (sig["level"] == mine["level"]).all() and (sig["warning"] == mine["warning"].astype(int)).all()
    assert (sig["lean"].fillna("-") == mine["lean"].fillna("-")).all(), "arrows differ from the engine"
    assert np.allclose(sig["p_up"].astype(float), mine["p_up"], equal_nan=True)
    print(f"Engine check {day}: {len(sig)} signals, ranges, activity and arrows identical to the live engine.")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--final", action="store_true")
    dry = ap.parse_args().dry_run
    done = [t for t in TABLES if (REPORT / f"phase6_{t}.csv").exists()]
    if not dry and done:
        sys.exit(f"The final test has already been run (report/phase6_{done[0]}.csv exists). It runs once.")

    model = load_model()
    model["move"].pop("lgbm", None)   # Phase 6 scored HAR ranges, the model as frozen on 2026-11-14; LightGBM as served is in phase6b
    grid, sip, news, days = load_inputs(dry)
    rows, feat = app_rows(grid, news, days, model)
    rows = add_outcomes(rows, grid, sip)
    title = "DRY RUN on training days: these numbers mean nothing" if dry else "FINAL TEST"
    print(f"\n{title}\nModel {model_id()}, {len(rows):,} signal rows, {len(days)} days, {days[0]} to {days[-1]}")
    if dry:
        check_against_engine(rows, pd.Timestamp(CFG["replay_default_day"]).date(), model)

    tables = {"ranges": range_table(rows, model), "models": model_table(rows, fit_v4(), feat),
              "activity": activity_table(rows), "direction": direction_table(rows, model), "days": day_table(rows)}
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    for name, t in tables.items():
        print(f"\n── {name} ──\n{t.round(4).to_string()}")
    if tables["direction"]["dir_acc"].max() > LEAK_ALARM:
        print(f"\nWARNING: a direction accuracy above {LEAK_ALARM:.0%}. Treat as a bug or leak until checked.")
    if dry:
        return
    for name, t in tables.items():
        t.to_csv(REPORT / f"phase6_{name}.csv")
    rows.to_parquet(REPORT / "phase6_rows.parquet")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (REPORT / "phase6_run.json").write_text(json.dumps(
        {"run_at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "git_commit": commit, "model_id": model_id(),
         "first_day": str(days[0]), "last_day": str(days[-1]), "days": len(days), "rows": len(rows)}, indent=2) + "\n")
    print("\nWritten: report/phase6_*.csv, report/phase6_run.json")


if __name__ == "__main__":
    main()
