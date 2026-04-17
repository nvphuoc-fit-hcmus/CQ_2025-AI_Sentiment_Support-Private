"""
Precompute target-based Fine-grained Sentiment Analysis (FSA) per article.

For each article, computes sentiment scores TARGETING each of the 10 financial
factors (not just a global positive/negative). This is "target-based FSA" as
described in the thesis: sentiment is analyzed per financial entity/aspect.

Method:
  1. Split each article into sentences.
  2. For each factor, filter sentences containing that factor's keywords.
  3. Run ProsusAI/finbert on matched sentences (financial domain model).
  4. Aggregate: net_sentiment = mean(positive_prob) - mean(negative_prob).
  5. Articles with no keyword match for a factor get sentiment = 0.0 (neutral).

Output: article_entity_sentiment.npy of shape (N_articles, 10)
  - Each row is a per-factor sentiment vector in [-1, +1].
  - Aligned with articles_max.csv row order (same as article_factor_labels.npy).

Usage:
    python precompute_entity_sentiment.py
    python precompute_entity_sentiment.py --batch_size 16
    python precompute_entity_sentiment.py --articles_path /path/to/articles_max.csv
"""

import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Load SA/.env (pipelines -> v2 -> app -> ai-service -> services -> SA)
try:
    from dotenv import load_dotenv
    sa_env = Path(__file__).parent.parent.parent.parent.parent.parent / ".env"
    if sa_env.exists():
        load_dotenv(sa_env)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Factor keyword ontology — reused from precompute_factor_labels.py
# Same 10 factors, same keywords for consistency.
# ---------------------------------------------------------------------------
FACTOR_KEYWORDS = {
    "institutional_inflow": [
        "institution", "fund", "grayscale", "microstrategy", "blackrock",
        "corporate", "treasury", "bitcoin purchase", "acquisition", "accumul",
        "pension", "endowment", "sovereign", "wealth fund", "adoption",
        "purchased", "invested", "allocat",
    ],
    "etf_flow": [
        "etf", "spot etf", "bitcoin etf", "sec approval", "inflow", "outflow",
        "bitcoin fund", "crypto fund", "futures etf", "bitcoin trust",
        "grayscale premium", "discount", "nav",
    ],
    "regulatory_easing": [
        "approved", "legal", "regulatory clarity", "compliant", "licensed",
        "framework", "clearance", "authorize", "regulation", "bill",
        "congress", "law", "policy", "guidance", "exemption", "sandbox",
    ],
    "regulatory_tightening": [
        "ban", "crackdown", "illegal", "sanction", "sec sue", "enforcement",
        "restrict", "prohibited", "suspension", "warning", "probe",
        "investigation", "fine", "penalty", "delist", "kyc", "aml",
    ],
    "exchange_risk": [
        "hack", "exploit", "exchange down", "withdrawal halt", "insolvent",
        "ftx", "celsius", "rug", "compromised", "security", "breach",
        "stolen", "loss", "bankrupt", "withdrawal", "freeze", "halted",
    ],
    "liquidity_squeeze": [
        "liquidity", "leverage", "liquidation", "margin call", "funding rate",
        "squeeze", "cascade", "deleverag", "short squeeze", "long squeeze",
        "open interest", "perp", "futures", "basis", "contango",
    ],
    "whale_accumulation": [
        "whale", "transaction", "on-chain", "address", "wallet", "accumulate",
        "hodl", "large buy", "holdings", "large transaction", "miner",
        "mining", "cold storage", "staking", "hodler", "accumulation",
    ],
    "macro_uncertainty": [
        "inflation", "fed", "interest rate", "recession", "gdp", "cpi",
        "fomc", "yield", "economy", "growth", "rate hike", "rate cut",
        "bank", "dollar", "usd", "dxy", "treasury", "bonds", "equity",
    ],
    "protocol_upgrade": [
        "upgrade", "fork", "halving", "taproot", "merge", "protocol",
        "layer2", "lightning", "launch", "update", "mainnet", "testnet",
        "snapshot", "airdrop", "defi", "nft", "smart contract", "validator",
    ],
    "network_outage": [
        "outage", "congestion", "fees spike", "mempool", "hash rate",
        "51%", "network issue", "downtime", "slow", "backlog", "stuck",
        "unconfirmed", "difficulty", "block time", "hashpower", "reorg",
    ],
}

FACTOR_NAMES = list(FACTOR_KEYWORDS.keys())
N_FACTORS    = len(FACTOR_NAMES)  # 10

# Sentence splitter: split on ". ", "! ", "? " preserving short segments.
_SENT_PATTERN = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, removing empty/short ones."""
    if not text or not isinstance(text, str):
        return []
    sentences = _SENT_PATTERN.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def _get_factor_sentences(title: str, content: str, factor_idx: int) -> list[str]:
    """Return sentences from title+content that mention factor_idx keywords."""
    keywords = FACTOR_KEYWORDS[FACTOR_NAMES[factor_idx]]
    # Title counts as 3 sentences for weighting (mirrors factor_labels.py)
    all_sentences = _split_sentences(title) * 3 + _split_sentences(content)

    matched = []
    for sent in all_sentences:
        sent_lower = sent.lower()
        if any(kw.lower() in sent_lower for kw in keywords):
            matched.append(sent)
    return matched


def compute_entity_sentiment(
    articles_df: pd.DataFrame,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Compute per-factor sentiment for all articles using FinBERT.

    For each article × factor pair:
      - Extract sentences mentioning that factor's keywords.
      - Run ProsusAI/finbert → (positive, negative, neutral) probabilities.
      - net_sentiment = mean(positive) - mean(negative) in [-1, +1].
      - If no keyword match: 0.0 (neutral signal, not noise).

    Args:
        articles_df: DataFrame with columns [title, content].
        batch_size:  FinBERT inference batch size (default 16).

    Returns:
        entity_sentiment: (N_articles, 10) float32 array, values in [-1, +1].
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print("[entity_sentiment] Loading ProsusAI/finbert ...")
    finbert = hf_pipeline(
        "text-classification",
        model="ProsusAI/finbert",
        top_k=None,          # Return all 3 labels
        truncation=True,
        max_length=128,
    )
    print("[entity_sentiment] Model loaded. Starting target-based FSA ...")

    N = len(articles_df)
    entity_sentiment = np.zeros((N, N_FACTORS), dtype=np.float32)

    n_matched_total = 0   # (article, factor) pairs with keyword match
    n_total_pairs   = N * N_FACTORS

    rows = list(articles_df.itertuples(index=False))

    for i, row in enumerate(rows):
        title   = str(getattr(row, "title",   "") or "")
        content = str(getattr(row, "content", "") or "")[:2000]  # cap at 2000 chars

        for f_idx in range(N_FACTORS):
            sentences = _get_factor_sentences(title, content, f_idx)

            if not sentences:
                # No keyword match → neutral (0.0), not noise
                entity_sentiment[i, f_idx] = 0.0
                continue

            # Batch FinBERT inference (up to batch_size sentences)
            sentences_trimmed = sentences[:batch_size]  # avoid huge batches per factor

            try:
                results = finbert(sentences_trimmed)
                pos_scores = []
                neg_scores = []
                for label_list in results:
                    label_dict = {r["label"].lower(): r["score"] for r in label_list}
                    pos_scores.append(label_dict.get("positive", 0.0))
                    neg_scores.append(label_dict.get("negative", 0.0))

                net = float(np.mean(pos_scores) - np.mean(neg_scores))
                net = max(-1.0, min(1.0, net))  # clamp to [-1, +1]
                entity_sentiment[i, f_idx] = net
                n_matched_total += 1

            except Exception as exc:
                entity_sentiment[i, f_idx] = 0.0  # fallback to neutral
                if i < 5:
                    print(f"    [FinBERT error] article {i}, factor {f_idx}: {exc}", flush=True)

        if (i + 1) % 100 == 0 or (i + 1) == N:
            pct = (i + 1) / N * 100
            print(f"  Progress: {i+1}/{N} articles ({pct:.1f}%)", flush=True)

    match_rate = n_matched_total / n_total_pairs * 100
    print(f"\n[entity_sentiment] Done.")
    print(f"  Matched (article, factor) pairs: {n_matched_total}/{n_total_pairs} ({match_rate:.1f}%)")
    print(f"  Sentiment range: [{entity_sentiment.min():.3f}, {entity_sentiment.max():.3f}]")
    print(f"  Mean |sentiment|: {np.abs(entity_sentiment).mean():.4f}")

    return entity_sentiment


def print_statistics(entity_sentiment: np.ndarray) -> None:
    """Print per-factor sentiment statistics."""
    print("\n" + "=" * 60)
    print("ENTITY SENTIMENT STATISTICS (per factor)")
    print("=" * 60)
    print(f"  {'Factor':<25} {'Mean':>8} {'Std':>8} {'Pos%':>7} {'Neg%':>7}")
    print(f"  {'-'*60}")
    for f_idx, fname in enumerate(FACTOR_NAMES):
        col  = entity_sentiment[:, f_idx]
        mean = col.mean()
        std  = col.std()
        pos  = (col > 0.1).mean() * 100
        neg  = (col < -0.1).mean() * 100
        print(f"  {fname:<25} {mean:>8.4f} {std:>8.4f} {pos:>6.1f}% {neg:>6.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute target-based FSA entity sentiment per article."
    )

    # Resolve default paths
    script_dir    = Path(__file__).resolve().parent
    service_root  = script_dir.parent.parent.parent     # ai-service/
    default_data  = service_root / "training_data"
    default_data_v2 = default_data / "v2"

    parser.add_argument(
        "--articles_path",
        type=Path,
        default=None,
        help="Path to articles_max.csv (default: training_data/articles_max.csv)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save article_entity_sentiment.npy (default: same as articles)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="FinBERT inference batch size per factor (default: 16)",
    )
    args = parser.parse_args()

    # Resolve articles path
    if args.articles_path is None:
        search_root = default_data_v2 if default_data_v2.exists() else default_data
        for candidate in ["articles_max.csv", "articles.csv"]:
            p = search_root / candidate
            if p.exists():
                args.articles_path = p
                break
    if args.articles_path is None or not args.articles_path.exists():
        print(f"[ERROR] articles_max.csv not found in {search_root}")
        sys.exit(1)

    if args.output_dir is None:
        args.output_dir = args.articles_path.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output_dir / "article_entity_sentiment.npy"

    print("[START] Precomputing target-based entity sentiment (FSA)")
    print(f"  Articles:    {args.articles_path.name}")
    print(f"  Output:      {output_path.name}")
    print(f"  Model:       ProsusAI/finbert (financial domain)")
    print(f"  Factors:     {N_FACTORS} ({', '.join(FACTOR_NAMES)})")
    print(f"  Batch size:  {args.batch_size}")
    print()

    # Load articles
    print(f"[LOAD] Reading {args.articles_path.name} ...")
    articles_df = pd.read_csv(args.articles_path)
    N = len(articles_df)
    print(f"[OK] Loaded {N} articles")

    for col in ["title", "content"]:
        if col not in articles_df.columns:
            print(f"[WARN] Column '{col}' not found — using empty string")
            articles_df[col] = ""

    # Compute
    entity_sentiment = compute_entity_sentiment(articles_df, batch_size=args.batch_size)

    # Validate
    assert entity_sentiment.shape == (N, N_FACTORS), \
        f"Shape mismatch: expected ({N}, {N_FACTORS}), got {entity_sentiment.shape}"
    assert not np.isnan(entity_sentiment).any(), "NaN detected in entity_sentiment!"
    assert (np.abs(entity_sentiment) <= 1.0 + 1e-6).all(), "Values outside [-1, +1]!"

    # Save
    np.save(output_path, entity_sentiment)
    print(f"\n[OK] Saved: {output_path}")
    print(f"     Shape:     {entity_sentiment.shape}  (N_articles × N_factors)")
    print(f"     Dtype:     {entity_sentiment.dtype}")
    print(f"     File size: {output_path.stat().st_size / 1e6:.2f} MB")

    print_statistics(entity_sentiment)

    print(f"\n[DONE] Run training:")
    print(f"  python train_safe_alert.py --symbol BTCUSDT --horizon 1h --epochs 60")
    print(f"  (train_safe_alert.py auto-loads article_entity_sentiment.npy)")


if __name__ == "__main__":
    main()
