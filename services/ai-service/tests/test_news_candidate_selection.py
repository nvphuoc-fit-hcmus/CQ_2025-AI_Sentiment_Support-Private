import pandas as pd

from app.v2.nlp.news_selector import select_candidates_for_horizon


def _news_frame(now: pd.Timestamp, count: int, spacing_minutes: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "published_at": now - pd.Timedelta(minutes=(i + 1) * spacing_minutes),
                "title": f"Bitcoin ETF market update {i}",
                "content": "Bitcoin ETF regulation and market liquidity update.",
                "source": "coindesk",
                "sentiment_score": 0.1,
            }
            for i in range(count)
        ]
    )


def test_1h_candidate_pool_is_not_pretrimmed_to_model_top_k():
    now = pd.Timestamp("2026-07-30T12:00:00Z")
    selected = select_candidates_for_horizon(
        _news_frame(now, count=30, spacing_minutes=30),
        now,
        "1h",
    )
    assert len(selected) == 16


def test_4h_candidate_pool_matches_training_capacity():
    now = pd.Timestamp("2026-07-30T12:00:00Z")
    selected = select_candidates_for_horizon(
        _news_frame(now, count=40, spacing_minutes=60),
        now,
        "4h",
    )
    assert len(selected) == 32


def test_candidate_pool_returns_available_count_when_news_is_sparse():
    now = pd.Timestamp("2026-07-30T12:00:00Z")
    selected = select_candidates_for_horizon(
        _news_frame(now, count=3, spacing_minutes=60),
        now,
        "1h",
    )
    assert len(selected) == 3
