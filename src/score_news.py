"""
score_news.py  –  FinBERT probabilities for each headline, pinned model revision.
Runs in its own environment (requirements-news.txt) because of torch.

Usage: .venv-news/bin/python src/score_news.py
"""

import hashlib
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent.parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
BATCH = 32


def main():
    name, rev = CFG["finbert_model"], CFG["finbert_revision"]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(name, revision=rev).eval()
    news = pd.read_parquet(ROOT / CFG["news_raw_path"])
    heads = news["headline"].tolist()
    parts = []
    with torch.no_grad():
        for i in range(0, len(heads), BATCH):
            batch = tok(heads[i:i + BATCH], return_tensors="pt", padding=True, truncation=True, max_length=64)
            parts.append(torch.softmax(model(**batch).logits, 1))
    proba = torch.cat(parts).numpy()
    idx = {v.lower(): k for k, v in model.config.id2label.items()}
    news["fb_pos"], news["fb_neg"] = proba[:, idx["positive"]], proba[:, idx["negative"]]
    out = ROOT / CFG["news_scored_path"]
    news.to_parquet(out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT / CFG["news_checksum_template"].format(name=out.stem)).write_text(sha + "\n")
    print(f"Scored {len(news):,} headlines")


if __name__ == "__main__":
    main()
