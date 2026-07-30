from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from .relevance_scorer import RelevanceConfig, HORIZON_RELEVANCE_CONFIGS, add_relevance_scores


@dataclass
class NewsSelectionConfig:
    lookback_minutes: int = 60
    top_k: int = 3
    relevance_cfg: RelevanceConfig = field(default_factory=RelevanceConfig)


# SAFE-Alert: Per-horizon news selection (Section 3.4.3 of paper)
# 15m: very narrow window, top-3 (breaking news only, high recency weight)
# 1h:  narrow window, top-4 (focus on breaking news)
# 4h:  wide window, top-5 (broader context)
# 24h: full-day window, top-8 (macro context)
# top_k must match SAFEAlertNet.K_h_map: K_15m=3, K_1h=4, K_4h=5, K_24h=8
# (model is initialized with those K values in train_safe_alert.py main())
HORIZON_SELECTION_CONFIGS: dict[str, NewsSelectionConfig] = {
    "15m": NewsSelectionConfig(
        lookback_minutes=30,
        top_k=3,  # matches model K_15m=3
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["15m"],
    ),
    "1h": NewsSelectionConfig(
        lookback_minutes=60,
        top_k=4,  # matches model K_1h=4
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["1h"],
    ),
    "4h": NewsSelectionConfig(
        lookback_minutes=240,
        top_k=5,  # matches model K_4h=5
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["4h"],
    ),
    "24h": NewsSelectionConfig(
        lookback_minutes=1440,
        top_k=8,  # matches model K_24h=8
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["24h"],
    ),
}

# Candidate pools handed to SAFEAlertNet. These mirror the training data
# windows; the learned selective-attention layer, not this rule-based
# pre-filter, is responsible for the final horizon Top-K decision.
HORIZON_CANDIDATE_CONFIGS: dict[str, NewsSelectionConfig] = {
    "15m": NewsSelectionConfig(
        lookback_minutes=24 * 60,
        top_k=16,
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["15m"],
    ),
    "1h": NewsSelectionConfig(
        lookback_minutes=24 * 60,
        top_k=16,
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["1h"],
    ),
    "4h": NewsSelectionConfig(
        lookback_minutes=96 * 60,
        top_k=32,
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["4h"],
    ),
    "24h": NewsSelectionConfig(
        lookback_minutes=168 * 60,
        top_k=32,
        relevance_cfg=HORIZON_RELEVANCE_CONFIGS["24h"],
    ),
}


def select_top_k_news(
    news_df: pd.DataFrame,
    current_time: pd.Timestamp,
    cfg: NewsSelectionConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or NewsSelectionConfig()

    if news_df.empty:
        return news_df.copy()

    start_time = current_time - timedelta(minutes=cfg.lookback_minutes)

    window = news_df[
        (news_df["published_at"] >= start_time) &
        (news_df["published_at"] < current_time)
    ].copy()

    if window.empty:
        return window

    window = add_relevance_scores(
        window,
        current_time=current_time,
        cfg=cfg.relevance_cfg,
    )

    window = window.sort_values(
        by=["relevance_score", "published_at"],
        ascending=[False, False]
    ).head(cfg.top_k).reset_index(drop=True)

    return window


def select_news_for_horizon(
    news_df: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon: str,
) -> pd.DataFrame:
    """
    SAFE-Alert Component 1 — Per-horizon selective news modeling.

    Different horizons use different lookback windows and top-K:
      15m: lookback=30min,   top_k=3  (breaking news only)
      1h:  lookback=60min,   top_k=4  (breaking news focus)
      4h:  lookback=240min,  top_k=5  (broader context)
      24h: lookback=1440min, top_k=8  (macro context)

    Returns selected DataFrame with relevance_score column.
    """
    cfg = HORIZON_SELECTION_CONFIGS.get(horizon, HORIZON_SELECTION_CONFIGS["1h"])
    return select_top_k_news(news_df, current_time, cfg)


def select_candidates_for_horizon(
    news_df: pd.DataFrame,
    current_time: pd.Timestamp,
    horizon: str,
) -> pd.DataFrame:
    """Build the candidate pool consumed by SAFEAlertNet selective attention."""
    cfg = HORIZON_CANDIDATE_CONFIGS.get(
        horizon,
        HORIZON_CANDIDATE_CONFIGS["1h"],
    )
    return select_top_k_news(news_df, current_time, cfg)


def format_selected_news_for_output(selected_news: pd.DataFrame) -> list[dict]:
    """Format selected news articles for inference output JSON."""
    if selected_news.empty:
        return []
    rows = []
    for _, row in selected_news.iterrows():
        rows.append({
            "title":           str(row.get("title", ""))[:120],
            "source":          str(row.get("source", "unknown")),
            "published_at":    str(row.get("published_at", "")),
            "relevance_score": round(float(row.get("relevance_score", 0.0)), 4),
            "sentiment_score": round(float(row.get("sentiment_score", 0.0)), 4),
        })
    return rows


def aggregate_selected_news_features(selected_news: pd.DataFrame) -> dict:
    if selected_news.empty:
        return {
            "selected_news_count": 0,
            "top1_sentiment": 0.0,
            "top3_avg_sentiment": 0.0,
            "max_sentiment_strength": 0.0,
            "weighted_sentiment": 0.0,
            "selected_distinct_sources": 0,
            "selected_avg_title_len": 0.0,
            "selected_avg_content_len": 0.0,
            "selected_avg_relevance": 0.0,
        }

    sentiments = selected_news["sentiment_score"].fillna(0.0).astype(float)
    relevance = selected_news["relevance_score"].fillna(0.0).astype(float)

    if relevance.sum() > 0:
        weighted_sent = float((sentiments * relevance).sum() / relevance.sum())
    else:
        weighted_sent = float(sentiments.mean())

    return {
        "selected_news_count": int(len(selected_news)),
        "top1_sentiment": float(sentiments.iloc[0]),
        "top3_avg_sentiment": float(sentiments.mean()),
        "max_sentiment_strength": float(sentiments.abs().max()),
        "weighted_sentiment": weighted_sent,
        "selected_distinct_sources": int(selected_news["source"].nunique()),
        "selected_avg_title_len": float(selected_news["title"].fillna("").str.len().mean()),
        "selected_avg_content_len": float(selected_news["content"].fillna("").str.len().mean()),
        "selected_avg_relevance": float(relevance.mean()),
    }
