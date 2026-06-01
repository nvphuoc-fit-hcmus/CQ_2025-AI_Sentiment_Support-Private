"""Unit tests for SAFEAlertDataset.

Tests verify temporal alignment, feature extraction correctness, and
precomputed artifact handling — the three most failure-prone areas
identified in the pre-training audit.

Run:
    cd services/ai-service
    pytest tests/test_dataset.py -v
"""
from __future__ import annotations

import sys
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
import numpy as np
import pandas as pd
import torch

_AI = Path(__file__).resolve().parent.parent
_V2 = _AI / "app" / "v2"
for p in (str(_AI), str(_V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipelines.safe_alert_dataset import SAFEAlertDataset  # noqa: E402


# ── Minimal fixture builders ──────────────────────────────────────────────────

def _make_candles(n: int = 20, start: str = "2023-01-01", freq_min: int = 60) -> pd.DataFrame:
    """Return n hourly OHLCV candles starting at `start`."""
    times = pd.date_range(start, periods=n, freq=f"{freq_min}min", tz="UTC")
    prices = 20000.0 + np.arange(n) * 10.0
    return pd.DataFrame({
        "timestamp": times,
        "open":  prices,
        "high":  prices + 50,
        "low":   prices - 50,
        "close": prices + np.random.uniform(-20, 20, n),
        "volume": np.random.uniform(100, 500, n),
    })


def _make_articles(
    candle_df: pd.DataFrame,
    n_per_candle: int = 5,
    delay_min: int = 0,
    emb_dim: int = 32,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return article_meta DataFrame + embeddings aligned to candles."""
    rows = []
    embs = []
    for ts in candle_df["timestamp"]:
        for k in range(n_per_candle):
            # Publish article ~30 min before candle close
            pub_time = ts - timedelta(minutes=30 + k * 5)
            rows.append({
                "timestamp": pub_time,
                "title": f"Article {k} at {ts}",
                "content": "BTC price analysis " * 10,
                "source": "coindesk",
            })
            embs.append(np.random.randn(emb_dim).astype(np.float32))
    meta = pd.DataFrame(rows)
    embs_arr = np.stack(embs)
    # L2-normalise (dataset expects unit-norm embeddings)
    norms = np.linalg.norm(embs_arr, axis=1, keepdims=True).clip(min=1e-8)
    return meta, embs_arr / norms


def _make_dataset(
    n_candles: int = 20,
    n_per_candle: int = 5,
    horizon: str = "1h",
    delay_min: int = 0,
    emb_dim: int = 32,
    **kwargs,
) -> SAFEAlertDataset:
    candle_df = _make_candles(n_candles, freq_min=60)
    article_meta, embeddings = _make_articles(candle_df, n_per_candle=n_per_candle, emb_dim=emb_dim)
    return SAFEAlertDataset(
        candle_df=candle_df,
        article_meta=article_meta,
        embeddings=embeddings,
        horizon=horizon,
        ingest_delay_minutes=delay_min,
        min_articles=1,
        articles_per_candle=n_per_candle,
        emb_dim=emb_dim,
        **kwargs,
    )


# ── 1. Basic construction ─────────────────────────────────────────────────────

def test_dataset_construction_no_crash():
    ds = _make_dataset()
    assert len(ds) > 0


def test_valid_idx_excludes_last_horizon_candles():
    """valid_idx must not include the last `horizon_steps` candles."""
    n = 15
    ds = _make_dataset(n_candles=n, horizon="1h")  # horizon_steps=1
    for idx in ds.valid_idx:
        assert idx < n - ds.horizon_steps, (
            f"valid_idx {idx} is within horizon range of last candle (n={n}, "
            f"horizon_steps={ds.horizon_steps})"
        )


def test_dataset_len_matches_valid_idx():
    ds = _make_dataset()
    assert len(ds) == len(ds.valid_idx)


# ── 2. Temporal alignment ─────────────────────────────────────────────────────

def test_no_future_articles_in_window():
    """Articles must be visible by candle-close decision time, not candle open."""
    ds = _make_dataset(n_candles=15, n_per_candle=5, delay_min=0)
    for sample_i in range(min(len(ds), 5)):
        candle_idx = ds.valid_idx[sample_i]
        decision_time = ds._get_decision_timestamp(candle_idx)
        window_start = decision_time - pd.Timedelta(hours=ds.lookback_hours)
        art_indices = ds.candle_to_articles.get(candle_idx, [])
        for ai in art_indices:
            art_time = ds.article_meta.iloc[ai]["timestamp"]
            assert window_start <= art_time <= decision_time, (
                f"Article {ai} published at {art_time} is outside "
                f"decision window [{window_start}, {decision_time}] "
                f"for candle {candle_idx}"
            )


def test_ingest_delay_excludes_recent_articles():
    """With delay_min=60, articles 30 min before close must be excluded."""
    n = 15
    candle_df = _make_candles(n, freq_min=60)
    # All articles published 30 min before their candle — should be excluded with delay=60
    rows, embs = [], []
    for ts in candle_df["timestamp"]:
        pub = ts + timedelta(minutes=30)
        rows.append({
            "timestamp": pub,
            "title": "Recent article before close",
            "content": "BTC price analysis " * 10,
            "source": "coindesk",
        })
        embs.append(np.ones(32, dtype=np.float32))
    meta = pd.DataFrame(rows)
    embs_arr = np.stack(embs)
    embs_arr /= np.linalg.norm(embs_arr, axis=1, keepdims=True)

    ds_no_delay  = SAFEAlertDataset(candle_df, meta, embs_arr, horizon="1h",
                                    ingest_delay_minutes=0, min_articles=1,
                                    articles_per_candle=5, emb_dim=32)
    ds_with_delay = SAFEAlertDataset(candle_df, meta, embs_arr, horizon="1h",
                                     ingest_delay_minutes=60, min_articles=1,
                                     articles_per_candle=5, emb_dim=32)

    # With 60-min delay, articles 30 min before candle are too recent → excluded
    assert len(ds_with_delay) < len(ds_no_delay) or (
        # OR: same candles but fewer articles per candle
        sum(len(v) for v in ds_with_delay.candle_to_articles.values()) <
        sum(len(v) for v in ds_no_delay.candle_to_articles.values())
    )


def test_article_after_open_before_close_is_visible_after_delay():
    """Regression: usable intrabar news is kept when it clears delay by close."""
    candle_df = _make_candles(6, freq_min=60)
    open_ts = candle_df.loc[0, "timestamp"]
    rows = [
        {
            "timestamp": open_ts + timedelta(minutes=30),  # visible at +45m
            "title": "Visible intrabar article",
            "content": "BTC price analysis " * 10,
            "source": "coindesk",
        },
        {
            "timestamp": open_ts + timedelta(minutes=50),  # visible at +65m
            "title": "Too late intrabar article",
            "content": "BTC price analysis " * 10,
            "source": "coindesk",
        },
    ]
    meta = pd.DataFrame(rows)
    embs = np.ones((len(rows), 32), dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    ds = SAFEAlertDataset(
        candle_df,
        meta,
        embs,
        horizon="1h",
        ingest_delay_minutes=15,
        min_articles=0,
        articles_per_candle=2,
        emb_dim=32,
    )

    included_titles = {
        ds.article_meta.iloc[i]["title"]
        for i in ds.candle_to_articles.get(0, [])
    }
    assert "Visible intrabar article" in included_titles
    assert "Too late intrabar article" not in included_titles


# ── 3. __getitem__ output shapes ─────────────────────────────────────────────

def test_getitem_shapes():
    K = 4
    emb_dim = 32
    ds = _make_dataset(n_per_candle=6, emb_dim=emb_dim)
    ds.articles_per_candle = K
    item = ds[0]

    assert item["market_features"].ndim == 1, "market_features must be 1D"
    assert item["article_embeddings"].shape[0] == K
    assert item["article_embeddings"].shape[1] == emb_dim
    assert item["article_metadata"].shape[0] == K
    assert item["article_mask"].shape == (K,)
    assert item["direction"] in (0, 1, 2)
    assert item["factor"].shape == (K, 10), f"factor shape wrong: {item['factor'].shape}"


def test_article_mask_ones_for_valid_articles():
    """Positions with real articles must have mask=1, padding must have mask=0."""
    ds = _make_dataset(n_per_candle=3)
    ds.articles_per_candle = 8  # more slots than articles per candle
    item = ds[0]
    mask = item["article_mask"]
    n_real = mask.sum().item()
    assert 1 <= n_real <= 3, f"Expected 1-3 valid article slots, got {n_real}"


def test_direction_label_range():
    ds = _make_dataset()
    for i in range(min(len(ds), 10)):
        d = ds[i]["direction"]
        assert d in (0, 1, 2), f"direction={d} outside {{0,1,2}}"


# ── 4. Factor label alignment ─────────────────────────────────────────────────

def test_precomputed_factor_labels_shape_accepted():
    """Factor labels with correct shape (n_articles, 10) must be accepted."""
    n_candles = 15
    n_per = 4
    candle_df = _make_candles(n_candles)
    meta, embs = _make_articles(candle_df, n_per_candle=n_per)
    n_arts = len(meta)
    # Valid uniform factor distribution
    factor_labels = np.ones((n_arts, 10), dtype=np.float32) / 10.0
    ds = SAFEAlertDataset(
        candle_df, meta, embs, horizon="1h",
        factor_labels=factor_labels, min_articles=1,
        articles_per_candle=n_per, emb_dim=32,
    )
    assert ds.precomputed_factor_labels is not None


def test_precomputed_factor_labels_wrong_shape_falls_back():
    """Factor labels with wrong shape must NOT raise — fall back to on-the-fly."""
    n_candles = 15
    n_per = 4
    candle_df = _make_candles(n_candles)
    meta, embs = _make_articles(candle_df, n_per_candle=n_per)
    # Wrong shape: (5, 10) instead of (n_articles, 10)
    bad_labels = np.ones((5, 10), dtype=np.float32) / 10.0
    ds = SAFEAlertDataset(
        candle_df, meta, embs, horizon="1h",
        factor_labels=bad_labels, min_articles=1,
        articles_per_candle=n_per, emb_dim=32,
    )
    assert ds.precomputed_factor_labels is None, (
        "Wrong-shape factor_labels should fall back (precomputed_factor_labels=None)"
    )


def test_entity_sentiment_wrong_shape_raises():
    """entity_sentiment with wrong shape must raise ValueError (fail-fast contract)."""
    n_candles = 15
    candle_df = _make_candles(n_candles)
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    bad_sentiment = np.zeros((5, 10), dtype=np.float32)  # wrong n_articles
    with pytest.raises(ValueError, match="entity_sentiment"):
        SAFEAlertDataset(
            candle_df, meta, embs, horizon="1h",
            entity_sentiment=bad_sentiment, min_articles=1,
            articles_per_candle=3, emb_dim=32,
        )


# ── 5. Direction label correctness ───────────────────────────────────────────

def test_direction_up_when_price_rises_above_epsilon():
    """If close[t+1] > open[t+1] + epsilon, direction must be 2 (UP)."""
    candle_df = _make_candles(5, freq_min=60)
    # Manually set a big price jump at index 1 (label for candle 0)
    candle_df.loc[1, "open"]  = 20000.0
    candle_df.loc[1, "close"] = 20400.0  # +2%, well above epsilon=0.2%
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    ds = SAFEAlertDataset(candle_df, meta, embs, horizon="1h",
                          min_articles=1, articles_per_candle=3, emb_dim=32)
    if len(ds) > 0:
        item = ds[0]  # candle_idx=0, label_candle=1
        assert item["direction"] == 2, (
            f"Expected UP (2) for +2% return, got {item['direction']}"
        )


def test_direction_down_when_price_falls_below_epsilon():
    """If close[t+1] < open[t+1] - epsilon, direction must be 0 (DOWN)."""
    candle_df = _make_candles(5, freq_min=60)
    candle_df.loc[1, "open"]  = 20000.0
    candle_df.loc[1, "close"] = 19600.0  # -2%, well below epsilon=-0.2%
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    ds = SAFEAlertDataset(candle_df, meta, embs, horizon="1h",
                          min_articles=1, articles_per_candle=3, emb_dim=32)
    if len(ds) > 0:
        item = ds[0]
        assert item["direction"] == 0, (
            f"Expected DOWN (0) for -2% return, got {item['direction']}"
        )


# ── 6. Return label continuity ───────────────────────────────────────────────

def test_return_label_finite():
    """All return labels in the dataset must be finite."""
    ds = _make_dataset()
    for i in range(len(ds)):
        r = ds[i]["return"]
        assert math.isfinite(float(r)), f"return label at idx {i} is non-finite: {r}"


# ── 7. Candle interval guard ─────────────────────────────────────────────────

def test_candle_interval_mismatch_raises():
    """4h candles with horizon='1h' must raise ValueError."""
    candle_df = _make_candles(20, freq_min=240)  # 4h candles
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    with pytest.raises(ValueError, match="candle interval"):
        SAFEAlertDataset(candle_df, meta, embs, horizon="1h",
                         min_articles=1, articles_per_candle=3, emb_dim=32)


def test_candle_interval_correct_passes():
    """Matching candle interval must NOT raise."""
    candle_df = _make_candles(20, freq_min=60)  # 1h candles
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    ds = SAFEAlertDataset(candle_df, meta, embs, horizon="1h",
                          min_articles=1, articles_per_candle=3, emb_dim=32)
    assert len(ds) >= 0


def test_horizon_steps_derive_from_native_4h_candles():
    """4h horizon on 4h candles is one candle ahead, not four candles ahead."""
    candle_df = _make_candles(20, freq_min=240)
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    ds = SAFEAlertDataset(candle_df, meta, embs, horizon="4h",
                          min_articles=1, articles_per_candle=3, emb_dim=32)
    assert ds.horizon_steps == 1


def test_horizon_steps_derive_from_1h_base_candles():
    """4h horizon on 1h candles remains four candle steps ahead."""
    candle_df = _make_candles(20, freq_min=60)
    meta, embs = _make_articles(candle_df, n_per_candle=3)
    ds = SAFEAlertDataset(candle_df, meta, embs, horizon="4h",
                          min_articles=1, articles_per_candle=3, emb_dim=32)
    assert ds.horizon_steps == 4
