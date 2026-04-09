"""
SAFE-Alert — News Relevance Scorer.

Scores news articles by recency + source credibility + keyword relevance
for use in SelectiveAttention news selection (Eq.9-13 of thesis).

Used by:
  - nlp/news_selector.py   (select_top_k_news)
  - pipelines/live_infer.py (_nlp_for_window)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# SOURCE CREDIBILITY SCORES (hand-tuned, financial news domain)
# ─────────────────────────────────────────────────────────────

SOURCE_CREDIBILITY: dict[str, float] = {
    "coindesk":         0.95,
    "bitcoin magazine": 0.95,
    "cointelegraph":    0.90,
    "decrypt":          0.85,
    "the block":        0.85,
    "cryptoslate":      0.80,
    "coingecko":        0.80,
    "newsbtc":          0.75,
    "u.today":          0.70,
    "rss":              0.65,
    "reddit":           0.55,
    "twitter":          0.50,
    "unknown":          0.50,
}

# Keyword sets that signal high relevance for crypto price movement
PRICE_RELEVANCE_KEYWORDS = [
    # Macro signals
    "bitcoin", "btc", "price", "rally", "dump", "crash", "surge", "drop",
    # Institutional
    "etf", "institution", "grayscale", "blackrock", "microstrategy",
    # Regulatory
    "sec", "cftc", "ban", "approved", "regulation", "legal",
    # On-chain
    "whale", "accumulate", "exchange", "liquidation", "funding rate",
    # Macro-economic
    "inflation", "fed", "interest rate", "recession", "fomc",
    # Protocol
    "halving", "upgrade", "fork", "lightning",
]


@dataclass
class RelevanceConfig:
    """
    Configuration for relevance scoring.

    Attributes:
        recency_weight     : Weight for recency score (how fresh the article is)
        credibility_weight : Weight for source credibility
        keyword_weight     : Weight for keyword relevance to price movement
        recency_halflife_h : Half-life in hours for exponential recency decay
    """
    recency_weight: float = 0.50
    credibility_weight: float = 0.25
    keyword_weight: float = 0.25
    recency_halflife_h: float = 4.0    # Score halves every 4 hours


# Per-horizon configs: shorter horizon = weight recency more
HORIZON_RELEVANCE_CONFIGS: dict[str, RelevanceConfig] = {
    "1h":  RelevanceConfig(recency_weight=0.60, credibility_weight=0.20, keyword_weight=0.20, recency_halflife_h=2.0),
    "4h":  RelevanceConfig(recency_weight=0.45, credibility_weight=0.30, keyword_weight=0.25, recency_halflife_h=6.0),
    "24h": RelevanceConfig(recency_weight=0.30, credibility_weight=0.35, keyword_weight=0.35, recency_halflife_h=12.0),
}


def _recency_score(published_at: pd.Series, current_time: pd.Timestamp,
                   halflife_h: float = 4.0) -> pd.Series:
    """
    Exponential decay recency: score = exp(-lambda * age_hours)
    lambda = ln(2) / halflife_h  →  score halves every halflife_h hours.
    """
    try:
        # Ensure timezone alignment
        if current_time.tz is None:
            current_time = current_time.tz_localize("UTC")
        pub = pd.to_datetime(published_at, utc=True, errors="coerce")
    except Exception:
        pub = pd.to_datetime(published_at, errors="coerce")

    age_h = (current_time - pub).dt.total_seconds() / 3600.0
    age_h = age_h.clip(lower=0.0)
    lam = np.log(2) / max(halflife_h, 0.1)
    return np.exp(-lam * age_h).fillna(0.0)


def _credibility_score(source: pd.Series) -> pd.Series:
    """Map source name to credibility score."""
    def _lookup(s: str) -> float:
        s = str(s).lower().strip()
        for key, score in SOURCE_CREDIBILITY.items():
            if key in s:
                return score
        return SOURCE_CREDIBILITY["unknown"]

    return source.apply(_lookup)


def _keyword_score(text: pd.Series, keywords: list[str] = PRICE_RELEVANCE_KEYWORDS) -> pd.Series:
    """
    Fraction of keywords present in lowercased text (capped at 1.0).
    """
    def _count(t: str) -> float:
        t = str(t).lower()
        hits = sum(1 for kw in keywords if kw in t)
        return min(1.0, hits / max(1, len(keywords) * 0.10))  # 10% hit rate = score 1.0

    return text.apply(_count)


def add_relevance_scores(
    news_df: pd.DataFrame,
    current_time: Optional[pd.Timestamp] = None,
    cfg: Optional[RelevanceConfig] = None,
) -> pd.DataFrame:
    """
    Add `relevance_score` column to news DataFrame.

    relevance_score = w_rec * recency + w_cred * credibility + w_kw * keyword

    Args:
        news_df      : DataFrame with at least [published_at] column.
                       Optional: [source], [title] / [text] / [content]
        current_time : Reference time for recency (default: now UTC)
        cfg          : RelevanceConfig (default: equal weights, 4h halflife)

    Returns:
        DataFrame with `relevance_score` column added.
    """
    if news_df.empty:
        news_df = news_df.copy()
        news_df["relevance_score"] = pd.Series(dtype=float)
        return news_df

    if current_time is None:
        current_time = pd.Timestamp.now("UTC")
    if cfg is None:
        cfg = RelevanceConfig()

    out = news_df.copy()

    # 1. Recency
    time_col = "published_at" if "published_at" in out.columns else "created_at"
    if time_col in out.columns:
        rec = _recency_score(out[time_col], current_time, cfg.recency_halflife_h)
    else:
        rec = pd.Series(0.5, index=out.index)

    # 2. Source credibility
    src_col = "source" if "source" in out.columns else None
    if src_col:
        cred = _credibility_score(out[src_col])
    else:
        cred = pd.Series(SOURCE_CREDIBILITY["unknown"], index=out.index)

    # 3. Keyword relevance (use title > text > content)
    text_col = None
    for col in ("title", "text", "content"):
        if col in out.columns:
            text_col = col
            break
    if text_col:
        kw = _keyword_score(out[text_col])
    else:
        kw = pd.Series(0.0, index=out.index)

    out["relevance_score"] = (
        cfg.recency_weight     * rec
        + cfg.credibility_weight * cred
        + cfg.keyword_weight     * kw
    ).clip(0.0, 1.0)

    return out
