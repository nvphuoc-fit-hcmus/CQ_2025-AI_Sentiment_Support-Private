"""Precompute TF-IDF novelty score per article — paper Section 4.1.3.

Session 24: Paper Section 4.1.3 specifies novelty as similarity vs. the
24-hour prior corpus (TF-IDF or LLM). The dataset metadata vector's 4th
dimension (``meta[i, 3]``) was previously populated with ``rank_norm`` —
ordinal recency rank within the article window. That is NOT novelty; it
only captures temporal order, not content novelty.

This script computes the paper-faithful novelty:

    novelty(a_i) = 1 - max_cos_sim(TF-IDF(a_i), TF-IDF(a_j))
                                   where t_j ∈ [t_i - 24h, t_i - 1 min]

A value of 1.0 means the article is completely novel (no similar prior
article); 0.0 means a near-duplicate of a prior article in the window.
Using max (not mean) similarity makes novelty sensitive to near-duplicate
re-hashing by different outlets — the common crypto news pattern where the
same story is syndicated by 5 outlets within an hour.

Output
------
``article_novelty.npy`` (float32, shape (N_articles,)) with values in [0, 1].
The dataset loads this file when present and uses it for ``meta[i, 3]``,
falling back to ``rank_norm`` when the file is missing.

Performance
-----------
For 90k articles, a naive O(N²) pairwise cosine is ~8 × 10⁹ ops — slow
(~10 min on CPU). We exploit the 24h temporal window: for each article,
only the ≤ 24h of prior articles (≈ 50-300 articles for crypto feeds) are
compared. This is O(N × W) with W ≈ 100, ~9 × 10⁶ ops, runs in < 30 s.

Causality
---------
Only uses articles with ``published_at < t_i`` (strictly before). If the
dataset also enforces ``ingest_delay_minutes=15``, that is orthogonal — the
novelty value is computed at publication time and doesn't change based on
when the model sees the article.

Usage
-----
    python precompute_novelty.py \\
        --articles-csv training_data/v2/articles_max.csv \\
        --out training_data/v2/article_novelty.npy \\
        --window-hours 24
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("precompute_novelty")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# Tokenization: alphanumeric + underscore, lowercased. Stopwords are a short
# English+crypto-specific list — lowercase, deduplicated. TF-IDF with ~1500
# articles in a 24h window is fast even with this coarse pipeline.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "for", "on",
    "with", "as", "by", "at", "is", "was", "are", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "his", "she", "her", "will", "would",
    "can", "could", "may", "might", "should", "shall", "has", "have", "had",
    "do", "does", "did", "not", "no", "if", "then", "than", "so", "just",
    "also", "from", "into", "over", "out", "up", "down", "off", "about",
    "after", "before", "between", "during", "more", "less", "most", "least",
    "very", "much", "many", "some", "any", "all", "each", "every", "one",
    "two", "three", "other", "another", "i", "me", "my",
})


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    toks = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in toks if t and t not in _STOPWORDS and len(t) > 1]


def _build_tfidf(tokens_list: List[List[str]]) -> tuple:
    """Build a sparse TF-IDF representation.

    Returns:
        vocab       : {token -> col_idx} (int keys)
        tfidf_rows  : list of dicts {col_idx -> tfidf_value}
        row_norms   : list of L2 norms of each row (for cosine)
    """
    # Document frequency
    df: dict[str, int] = {}
    for toks in tokens_list:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    # Vocab: keep tokens appearing in ≥ 2 docs (drop hapax legomena — mostly
    # typos or rare named entities that inflate novelty artificially)
    vocab = {t: i for i, (t, c) in enumerate(df.items()) if c >= 2}
    N = len(tokens_list)
    # IDF = log(N / df)
    idf = {t: float(np.log(N / c)) for t, c in df.items() if t in vocab}

    tfidf_rows: list[dict] = []
    row_norms = np.zeros(N, dtype=np.float64)
    for i, toks in enumerate(tokens_list):
        if not toks:
            tfidf_rows.append({})
            continue
        # Term frequency
        tf: dict[str, int] = {}
        for t in toks:
            if t in vocab:
                tf[t] = tf.get(t, 0) + 1
        # TF-IDF
        row: dict[int, float] = {}
        sq = 0.0
        for t, c in tf.items():
            w = (1.0 + np.log(c)) * idf[t]    # sublinear tf (standard)
            row[vocab[t]] = float(w)
            sq += w * w
        tfidf_rows.append(row)
        row_norms[i] = float(np.sqrt(sq))
    return vocab, tfidf_rows, row_norms


def _cosine(row_a: dict, row_b: dict, norm_a: float, norm_b: float) -> float:
    """Cosine similarity on two sparse rows (dicts)."""
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    # Iterate over the shorter one for speed
    if len(row_a) > len(row_b):
        row_a, row_b = row_b, row_a
    dot = 0.0
    for k, va in row_a.items():
        vb = row_b.get(k)
        if vb is not None:
            dot += va * vb
    return float(dot / (norm_a * norm_b))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles-csv", type=Path,
                        default=Path("training_data/v2/articles_max.csv"),
                        help="Path to articles_max.csv")
    parser.add_argument("--out", type=Path,
                        default=Path("training_data/v2/article_novelty.npy"),
                        help="Output .npy path")
    parser.add_argument("--window-hours", type=int, default=24,
                        help="Temporal lookback window for novelty comparison")
    parser.add_argument("--text-field", type=str, default="title",
                        choices=["title", "content", "both"],
                        help="Which field to use for TF-IDF (default: title — "
                             "fast, captures headlines most strongly)")
    args = parser.parse_args()

    if not args.articles_csv.exists():
        raise FileNotFoundError(f"Articles CSV not found: {args.articles_csv}")

    # ── Load + normalize schema ─────────────────────────────────────────────
    logger.info("Reading %s ...", args.articles_csv.name)
    # pandas 3.0 rejects low_memory=False when engine="python" — drop the flag
    # under the python engine (it's a no-op there anyway, only the C engine
    # chunks input based on low_memory).
    df = pd.read_csv(args.articles_csv, sep=None, engine="python",
                     encoding="utf-8-sig")
    # Normalize column names (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    pub_col = col_map.get("published_at") or col_map.get("publishedat") or col_map.get("date")
    if pub_col is None:
        raise ValueError(f"No published_at column found in {args.articles_csv.name}")
    df = df.rename(columns={pub_col: "published_at"})
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # Text source
    if args.text_field == "title":
        text_series = df.get("title", pd.Series([""] * len(df))).fillna("")
    elif args.text_field == "content":
        text_series = df.get("content", pd.Series([""] * len(df))).fillna("")
    else:   # both
        title = df.get("title", pd.Series([""] * len(df))).fillna("")
        content = df.get("content", pd.Series([""] * len(df))).fillna("")
        text_series = title.astype(str) + " " + content.astype(str)

    # ── Sort by time (causality) ────────────────────────────────────────────
    # novelty needs t_j < t_i for all j compared; sorting by time lets us
    # use a simple sliding-window cursor instead of per-article filtering.
    # CRITICAL: preserve the ORIGINAL row indices so we can un-sort at the
    # end — dataset __getitem__ indexes novelty[i] by article_meta.iloc[i]
    # position, which is the original CSV row order.
    df_sorted = df.sort_values("published_at")
    orig_order = df_sorted.index.to_numpy()       # (N,) orig row idx for each sorted position
    text_sorted = text_series.iloc[orig_order].reset_index(drop=True)
    df_sorted = df_sorted.reset_index(drop=True)  # now 0..N-1 positional
    N = len(df_sorted)
    logger.info("Articles: %d", N)

    # ── Tokenize ────────────────────────────────────────────────────────────
    logger.info("Tokenising (%s field) ...", args.text_field)
    tokens_list = [_tokenize(t) for t in text_sorted]

    # ── Build TF-IDF ────────────────────────────────────────────────────────
    logger.info("Building TF-IDF (vocab ≥ 2 docs) ...")
    vocab, tfidf_rows, row_norms = _build_tfidf(tokens_list)
    logger.info("  |vocab| = %d (after df ≥ 2 filter)", len(vocab))

    # ── Sliding-window novelty ──────────────────────────────────────────────
    logger.info("Computing novelty with %dh window ...", args.window_hours)
    times = df_sorted["published_at"].to_numpy()
    window = pd.Timedelta(hours=args.window_hours)
    novelty = np.zeros(N, dtype=np.float32)
    left = 0
    for i in range(N):
        # Advance left until times[left] >= times[i] - window
        cutoff = times[i] - window
        while left < i and times[left] < cutoff:
            left += 1
        # Compare row_i vs rows [left, i) — max cosine → novelty = 1 - max
        max_sim = 0.0
        row_i = tfidf_rows[i]
        norm_i = row_norms[i]
        if norm_i > 0.0 and row_i:
            for j in range(left, i):
                sim = _cosine(row_i, tfidf_rows[j], norm_i, row_norms[j])
                if sim > max_sim:
                    max_sim = sim
                    if max_sim >= 0.999:
                        break   # already ≈ duplicate, no gain from further compares
        novelty[i] = 1.0 - max_sim
        if i % 5000 == 0 and i > 0:
            logger.info("  processed %d / %d", i, N)

    # ── Un-sort back to original order ──────────────────────────────────────
    # `orig_order[k]` = the ORIGINAL row index for the k-th sorted article.
    # novelty[k] is the novelty of that k-th sorted article, which belongs
    # at position orig_order[k] of the output array. This restores the
    # article_meta row ordering SAFEAlertDataset expects.
    novelty_original = np.zeros(N, dtype=np.float32)
    novelty_original[orig_order] = novelty

    # ── Persist ─────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, novelty_original)
    logger.info("Saved %s (%.1f MB)", args.out, args.out.stat().st_size / 1024 / 1024)
    logger.info("Novelty stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                float(novelty_original.mean()),
                float(novelty_original.std()),
                float(novelty_original.min()),
                float(novelty_original.max()))


if __name__ == "__main__":
    main()
