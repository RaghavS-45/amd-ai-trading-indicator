"""
news.py  –  aligns scored articles to the minute grid without look-ahead.

Only articles whose headline names the asset count. An article arrives at created_at, or
at updated_at if it was edited more than news_edit_gap_seconds later (the stored headline
may be the edited one). It first affects the bar news_delay_bars after its arrival minute:
an article at 10:03:20 sits in bar 10:03 (closes 10:04), so it first counts on bar 10:04.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def first_bar(news: pd.DataFrame) -> pd.Series:
    created = pd.to_datetime(news["created_at"], utc=True)
    updated = pd.to_datetime(news["updated_at"], utc=True)
    edited = (updated - created).dt.total_seconds() > CFG["news_edit_gap_seconds"]
    arrival = created.where(~edited, updated)
    return arrival.dt.floor("min") + pd.Timedelta(minutes=CFG["news_delay_bars"])


def news_columns(index: pd.DatetimeIndex, news: pd.DataFrame) -> pd.DataFrame:
    """news_sent (sum of FinBERT pos - neg) and news_count per bar; articles outside the grid drop out."""
    named = news["headline"].str.lower().str.contains(CFG["news_named_pattern"], regex=True)
    use = news[named].assign(bar=first_bar(news[named]), edge=lambda x: x["fb_pos"] - x["fb_neg"])
    per_bar = use.groupby("bar")["edge"].agg(["sum", "size"])
    return pd.DataFrame({"news_sent": per_bar["sum"].reindex(index).fillna(0.0).to_numpy(),
                         "news_count": per_bar["size"].reindex(index).fillna(0.0).to_numpy()}, index=index)


def load_scored() -> pd.DataFrame:
    return pd.read_parquet(ROOT / CFG["news_scored_path"])
