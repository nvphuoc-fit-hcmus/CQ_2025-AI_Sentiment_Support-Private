"""Prepare a TimescaleDB-compatible historical news CSV from SAFE-Alert artifacts."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


SYMBOL_PATTERNS = {
    "BTCUSDT": (r"\bbitcoin\b", r"\bBTC\b"),
    "ETHUSDT": (r"\bethereum\b", r"\bether\b", r"\bETH\b"),
    "BNBUSDT": (r"\bbinance coin\b", r"\bBNB\b"),
    "SOLUSDT": (r"\bsolana\b", r"\bSOL\b"),
    "XRPUSDT": (r"\bripple\b", r"\bXRP\b"),
    "DOGEUSDT": (r"\bdogecoin\b", r"\bDOGE\b"),
    "ADAUSDT": (r"\bcardano\b", r"\bADA\b"),
    "AVAXUSDT": (r"\bavalanche\b", r"\bAVAX\b"),
    "DOTUSDT": (r"\bpolkadot\b", r"\bDOT\b"),
    "POLUSDT": (r"\bpolygon\b", r"\bmatic\b", r"\bPOL\b"),
}


def infer_symbols(title: str, content: str) -> list[str]:
    text = f"{title} {content}"
    matched = [
        symbol
        for symbol, patterns in SYMBOL_PATTERNS.items()
        if any(re.search(pattern, text, flags=re.IGNORECASE if pattern.islower() else 0) for pattern in patterns)
    ]
    return matched or ["ALL"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", required=True, type=Path)
    parser.add_argument("--entity-sentiment", required=True, type=Path)
    parser.add_argument("--factor-labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", default="2024-08-28")
    parser.add_argument("--end-exclusive", default="2025-08-28")
    args = parser.parse_args()

    entity_sentiment = np.load(args.entity_sentiment, mmap_mode="r")
    factor_labels = np.load(args.factor_labels, mmap_mode="r")
    if entity_sentiment.shape != factor_labels.shape:
        raise ValueError(f"Artifact shape mismatch: {entity_sentiment.shape} vs {factor_labels.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.articles.open(encoding="utf-8-sig", newline="") as source, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.writer(target)
        writer.writerow(["time", "url", "source", "title", "sentiment_score", "raw_score"])

        for index, row in enumerate(reader):
            published_at = (row.get("published_at") or "").strip()
            day = published_at[:10]
            if not (args.start <= day < args.end_exclusive):
                continue
            if index >= entity_sentiment.shape[0]:
                raise IndexError(f"Article row {index} exceeds sentiment artifact length")

            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            source_name = (row.get("source") or "unknown").strip()

            weights = np.asarray(factor_labels[index], dtype=np.float64)
            sentiments = np.asarray(entity_sentiment[index], dtype=np.float64)
            weight_sum = float(weights.sum())
            signed_sentiment = float(np.dot(weights, sentiments) / weight_sum) if weight_sum > 0 else 0.0
            signed_sentiment = max(-1.0, min(1.0, signed_sentiment))

            digest = hashlib.sha256(f"{published_at}\n{title}".encode("utf-8")).hexdigest()[:24]
            synthetic_url = f"historical://articles-max/{digest}"
            raw_score = {
                "symbols": infer_symbols(title, content),
                "import_source": "articles_max_v2",
                "original_score": row.get("sentiment_score"),
                "factor_grounded": True,
            }

            writer.writerow([
                published_at,
                synthetic_url,
                source_name,
                title,
                f"{signed_sentiment:.8f}",
                json.dumps(raw_score, ensure_ascii=False, separators=(",", ":")),
            ])
            written += 1

    print(json.dumps({
        "rows_written": written,
        "start": args.start,
        "end_exclusive": args.end_exclusive,
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
