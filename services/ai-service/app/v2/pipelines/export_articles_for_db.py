"""Export articles_max.csv into a PostgreSQL-friendly CSV.

The source sentiment_score is a confidence value, not signed polarity.  Signed
sentiment is derived from the aligned entity-sentiment artifact.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "training_data" / "v2"
ARTICLES = DATA / "articles_max.csv"
ENTITY_SENTIMENT = DATA / "article_entity_sentiment.npy"
OUTPUT = DATA / "articles_db_import.csv"


def stable_url(row):
    identity = f"{row.published_at}|{row.source}|{row.title}".encode("utf-8", "ignore")
    return f"dataset://articles_max/{hashlib.sha256(identity).hexdigest()}"


def main():
    articles = pd.read_csv(ARTICLES)
    entity = np.load(ENTITY_SENTIMENT, mmap_mode="r")
    if len(articles) != len(entity):
        raise ValueError(f"row mismatch: articles={len(articles)}, entity={len(entity)}")

    nonzero = (entity != 0).sum(axis=1)
    signed = np.divide(
        entity.sum(axis=1),
        nonzero,
        out=np.zeros(len(entity), dtype=np.float32),
        where=nonzero > 0,
    )

    out = pd.DataFrame({
        "time": pd.to_datetime(articles["published_at"], utc=True, errors="coerce"),
        "url": [stable_url(row) for row in articles.itertuples(index=False)],
        "source": articles["source"].fillna("unknown"),
        "title": articles["title"].fillna("Không có tiêu đề"),
        "sentiment_score": signed,
        "raw_score": [
            json.dumps({
                "content": "" if pd.isna(content) else str(content),
                "dataset_confidence": None if pd.isna(confidence) else float(confidence),
                "origin": "articles_max.csv",
            }, ensure_ascii=False)
            for content, confidence in zip(articles["content"], articles["sentiment_score"])
        ],
    }).dropna(subset=["time"])
    out.to_csv(OUTPUT, index=False)
    print(f"exported={len(out)} output={OUTPUT}")


if __name__ == "__main__":
    main()
