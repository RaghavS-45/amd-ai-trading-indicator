"""
test_news.py
News alignment: an article may only affect bars after the one it arrived in.

Run:
    python -m pytest tests/test_news.py -v
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import compute_features
from news import news_columns

IDX = pd.date_range("2025-01-15 09:30", periods=60, freq="1min", tz="America/New_York").tz_convert("UTC")
T = IDX[20]


def _article(offset_s=20, edit_s=1, headline="AMD Wins Deal"):
    created = T + pd.Timedelta(seconds=offset_s)
    return pd.DataFrame({"created_at": [created], "updated_at": [created + pd.Timedelta(seconds=edit_s)],
                         "headline": [headline], "fb_pos": [0.9], "fb_neg": [0.1], "n_symbols": [1]})


def test_article_first_counts_on_next_bar():
    cols = news_columns(IDX, _article())
    assert cols["news_count"].iloc[:21].sum() == 0
    assert cols["news_count"].iloc[21] == 1
    assert abs(cols["news_sent"].iloc[21] - 0.8) < 1e-12


def test_edited_article_arrives_when_edited():
    cols = news_columns(IDX, _article(edit_s=600))
    assert cols["news_count"].iloc[:31].sum() == 0
    assert cols["news_count"].iloc[31] == 1


def test_unnamed_headline_ignored():
    assert news_columns(IDX, _article(headline="Nvidia Wins Deal"))["news_count"].sum() == 0


def test_news_features_ignore_future_and_previous_session():
    frame = pd.DataFrame({"news_sent": 0.0, "news_count": 0.0}, index=IDX)
    frame.iloc[30, 0], frame.iloc[30, 1] = 0.8, 1
    full = compute_features(frame, ["news"])
    cut = compute_features(frame.iloc[:30], ["news"])
    assert (full.iloc[:30] == cut).all().all()
    assert full["news_count_30"].iloc[59] == 1

    next_day = IDX + pd.Timedelta(days=1)
    two = pd.concat([frame, pd.DataFrame({"news_sent": 0.0, "news_count": 0.0}, index=next_day)])
    assert compute_features(two, ["news"]).loc[next_day, "news_count_30"].sum() == 0
