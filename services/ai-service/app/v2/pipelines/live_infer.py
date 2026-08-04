"""
SAFE-Alert Live Inference Adapter.

Kết nối market_cache (Kafka live candles) + MongoDB news -> v2 ML pipeline.

Flow:
  1. get_candles(symbol, limit=250, interval="1h")  ← market_cache
  2. build_market_features()                         ← technical indicators + lags
  3. _fetch_news_mongodb()                           ← MongoDB crawler_db (4h window)
  4. _nlp_features_for_window()   × 4 windows       ← VADER scoring per 1h slot
  5. align columns với training features
  6. SAFEAlertNet inference 1h + 4h                  ← PyTorch model (sole path)
  7. alert_info from should_alert flags (Eq.26)
  8. save latest_signal_{symbol}.json

Fallbacks:
  - market_cache rỗng -> ValueError (caller should handle)
  - MongoDB không connect -> NLP features = 0 (model vẫn chạy, ~8% less accurate)
  - SAFEAlertNet chưa train -> neutral/HOLD defaults cho cả 2 horizon
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Use FinBERT for sentiment scoring if USE_FINBERT=1 env var is set.
# FinBERT downloads ~440 MB on first run. Default: VADER only (faster, no download).
USE_FINBERT = os.getenv("USE_FINBERT", "0").strip() in ("1", "true", "yes")

# ── Path setup (same pattern as infer_realtime_v2.py) ──────────
CURRENT_FILE  = Path(__file__).resolve()
PIPELINES_DIR = CURRENT_FILE.parent
V2_DIR        = PIPELINES_DIR.parent
SERVICE_ROOT  = V2_DIR.parent.parent
APP_DIR       = V2_DIR.parent
SERVICE_ROOT  = APP_DIR.parent

for _p in [str(V2_DIR), str(APP_DIR), str(SERVICE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Internal imports ────────────────────────────────────────────
from preprocessing.market_features import (
    build_market_features,
    extract_multiframe_market_features,  # NEW: For true 5-timeframe support
)
from preprocessing.news_features import add_nlp_scores, aggregate_nlp_window, get_vader_score
from nlp.news_selector import (
    aggregate_selected_news_features,
    select_candidates_for_horizon,
    select_news_for_horizon,
)
from nlp.relevance_scorer import add_relevance_scores
from alerts.alert_decider import decide_alert_multimodal_eq26
from pipelines.utils import ARTIFACT_DIR, load_safe_alert_policy
from models.safe_alert_net import FACTOR_KEYWORDS, FACTOR_NAMES, META_DIM
from pipelines.precompute_factor_labels import compute_factor_probs_for_article
from pipelines.precompute_entity_sentiment import compute_one_article_entity_sentiment

logger = logging.getLogger("v2.live_infer")

# ── SAFEAlertNet constants ───────────────────────────────────────
_SAFE_ALERT_MODELS: dict = {}   # cache: (symbol, horizon) -> SAFEAlertNet instance
_SAFE_ALERT_MARKET_DIM = 63
_SAFE_ALERT_MAX_ARTICLES = 8    # K_max articles per inference call
_RUNTIME_EMBEDDER = None
_FACTOR_LABEL_CACHE = None
_ENTITY_SENT_CACHE = None
_FACTOR_URL_MAP = None
_FACTOR_TITLE_TS_MAP = None
# R3 #C3: FinBERT classification pipeline for runtime per-factor sentiment.
# Training used ProsusAI/finbert classifier per-factor (see
# precompute_entity_sentiment.py). Runtime must match — previous VADER×keyword
# approximation produced same-sign per-factor sentiments, contradicting training
# distribution (where an article can be +etf_flow AND -exchange_risk). This
# cache lazily loads the classifier on first cache miss.
_FINBERT_CLASSIFIER = None

# ── Config ──────────────────────────────────────────────────────
NLP_LAGS          = [1, 2, 3]          # lag cols: vader_mean_lag1-3 etc.
NLP_LAG_COLS      = ["vader_mean", "nlp_score_mean", "bullish_ratio", "bearish_ratio"]
NEWS_LOOKBACK_H   = 168                # covers the widest deployed candidate window
CANDLE_LIMIT      = 250                # need ≥ 200 for SMA50 + lags + LSTM seq_len

DROP_COLS = {
    "symbol", "timestamp", "timestamp_dt",
    "target_return_1h", "target_return_4h",
    "target_direction_1h", "target_direction_4h",
}


class _FinBertEmbedder:
    """Lazy runtime embedder for live articles."""

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> bool:
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModel.from_pretrained(self.MODEL_NAME).to(self._device).eval()
            return True
        except Exception as exc:
            logger.warning("Runtime FinBERT embedding unavailable: %s", exc)
            self._tokenizer = None
            self._model = None
            return False

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        if not self._load():
            return np.zeros((len(texts), 768), dtype=np.float32)

        try:
            import torch
            clean = [t[:512] if isinstance(t, str) else "" for t in texts]
            inputs = self._tokenizer(
                clean,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
            return emb / norms
        except Exception as exc:
            logger.warning("Runtime article embedding failed: %s", exc)
            return np.zeros((len(texts), 768), dtype=np.float32)


def _get_runtime_embedder() -> _FinBertEmbedder:
    global _RUNTIME_EMBEDDER
    if _RUNTIME_EMBEDDER is None:
        _RUNTIME_EMBEDDER = _FinBertEmbedder()
    return _RUNTIME_EMBEDDER


def _get_finbert_classifier():
    """Lazy-load ProsusAI/finbert classifier for runtime entity-sentiment
    computation. Matches train-time FinBERT used in precompute_entity_sentiment.py."""
    global _FINBERT_CLASSIFIER
    if _FINBERT_CLASSIFIER is not None:
        return _FINBERT_CLASSIFIER
    try:
        from transformers import pipeline as hf_pipeline
        _FINBERT_CLASSIFIER = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            top_k=None,          # return all 3 labels (positive/negative/neutral)
            truncation=True,
            max_length=128,
        )
        logger.info("[live_infer] FinBERT classifier loaded for runtime entity sentiment.")
        return _FINBERT_CLASSIFIER
    except Exception as exc:
        logger.warning("[live_infer] FinBERT classifier load failed — "
                       "entity_sentiment will fall back to zeros for cache-miss articles: %s", exc)
        _FINBERT_CLASSIFIER = False  # sentinel: tried and failed
        return None


def _estimate_entity_sentiment(title: str, content: str, source: str = "") -> np.ndarray:
    """R3 #C3 fix: Per-factor FinBERT entity sentiment at runtime.

    PREVIOUSLY (broken): returned ``VADER(overall) × factor_probs`` — a single
    scalar VADER sentiment scaled by keyword-based factor probabilities. This
    forces EVERY factor to have the same SIGN (positive/negative), which
    contradicts training distribution: articles can simultaneously have
    ``+etf_flow`` and ``-exchange_risk``.

    NOW (correct): delegate to ``compute_one_article_entity_sentiment`` which
    IS the same function precompute_entity_sentiment.py uses for training
    artefacts. For cache-hit articles we use the cached LLM labels; for cache
    miss we run FinBERT per-factor at runtime. Adds ~200-500 ms latency on
    cache miss but ensures train/inference distribution parity.

    Falls back to zeros only if FinBERT fails to load (e.g. no internet,
    disk full) — the model then gets a neutral 10-dim FSA vector for those
    articles, which is the same behaviour as training-time unmatched articles.
    """
    classifier = _get_finbert_classifier()
    if classifier is None or classifier is False:
        return np.zeros(len(FACTOR_NAMES), dtype=np.float32)
    try:
        return compute_one_article_entity_sentiment(
            title=title, content=content, finbert_pipeline=classifier, batch_size=16,
        )
    except Exception as exc:
        logger.warning("[live_infer] Runtime entity sentiment failed: %s", exc)
        return np.zeros(len(FACTOR_NAMES), dtype=np.float32)


def _infer_factor_distribution(title: str, content: str, source: str = "") -> np.ndarray:
    """Infer factor relevance using the same article-level scorer as precompute."""
    probs, _ = compute_factor_probs_for_article(
        title=title,
        content=content,
        source=source,
        deterministic_prior=True,
    )
    return probs.astype(np.float32)


def _load_factor_label_cache() -> None:
    """Load precomputed factor labels for live lookup by url/title+date."""
    global _FACTOR_LABEL_CACHE, _ENTITY_SENT_CACHE, _FACTOR_URL_MAP, _FACTOR_TITLE_TS_MAP
    if _FACTOR_LABEL_CACHE is not None and _FACTOR_URL_MAP is not None:
        return
    try:
        training_dir = SERVICE_ROOT / "training_data"
        training_v2 = training_dir / "v2"
        if training_v2.exists():
            training_dir = training_v2
        labels_path = training_dir / "article_factor_labels.npy"
        entities_path = training_dir / "article_entity_sentiment.npy"
        meta_path = training_dir / "articles_max.csv"
        if not labels_path.exists() or not meta_path.exists():
            return
        labels = np.load(labels_path)
        entities = np.load(entities_path) if entities_path.exists() else None
        meta = pd.read_csv(meta_path)
        if labels.shape[0] != len(meta):
            return
        url_map = {}
        title_ts_map = {}
        for i, row in meta.iterrows():
            url = str(row.get("url", "")).strip()
            if url:
                url_map[url] = i
            title = str(row.get("title", "")).strip().lower()
            ts_raw = row.get("timestamp", "")
            date_key = ""
            try:
                ts = pd.to_datetime(ts_raw, errors="coerce")
                if not pd.isna(ts):
                    date_key = ts.date().isoformat()
            except Exception:
                date_key = ""
            if title:
                key = f"{title}|{date_key}" if date_key else title
                title_ts_map[key] = i
        _FACTOR_LABEL_CACHE = labels.astype(np.float32)
        _ENTITY_SENT_CACHE = entities.astype(np.float32) if entities is not None else None
        _FACTOR_URL_MAP = url_map
        _FACTOR_TITLE_TS_MAP = title_ts_map
    except Exception:
        _FACTOR_LABEL_CACHE = None
        _ENTITY_SENT_CACHE = None
        _FACTOR_URL_MAP = None
        _FACTOR_TITLE_TS_MAP = None


def _lookup_factor_cache(row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (factor_probs, entity_sent) from precomputed cache if possible."""
    _load_factor_label_cache()
    if _FACTOR_LABEL_CACHE is None or _FACTOR_URL_MAP is None:
        return None, None
    idx = None
    url = str(row.get("url", "")).strip()
    if url and url in _FACTOR_URL_MAP:
        idx = _FACTOR_URL_MAP[url]
    if idx is None:
        title = str(row.get("title", "")).strip().lower()
        date_key = ""
        ts_raw = row.get("published_at", row.get("created_at", ""))
        try:
            ts = pd.to_datetime(ts_raw, errors="coerce")
            if not pd.isna(ts):
                date_key = ts.date().isoformat()
        except Exception:
            date_key = ""
        if title:
            key = f"{title}|{date_key}" if date_key else title
            idx = _FACTOR_TITLE_TS_MAP.get(key)
    if idx is None:
        return None, None
    fac = _FACTOR_LABEL_CACHE[idx]
    ent = _ENTITY_SENT_CACHE[idx] if _ENTITY_SENT_CACHE is not None else None
    return fac, ent


def _try_parse_factor_vector(row: pd.Series, keys: list[str], expected_dim: int) -> np.ndarray | None:
    """Parse factor vectors from row fields when upstream precompute provides them."""
    for key in keys:
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        vec = None
        if isinstance(raw, (list, np.ndarray)):
            vec = np.asarray(raw, dtype=np.float32)
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                vec = np.asarray(parsed, dtype=np.float32)
            except Exception:
                vec = None
        if vec is None or vec.ndim != 1:
            continue
        if vec.shape[0] < expected_dim:
            vec = np.pad(vec, (0, expected_dim - vec.shape[0]))
        elif vec.shape[0] > expected_dim:
            vec = vec[:expected_dim]
        return vec.astype(np.float32)
    return None


def _temperature_softmax(logits, temperature: float):
    """Apply scalar temperature scaling to logits for calibrated live decisions."""
    import torch

    temp = max(float(temperature), 1e-3)
    return torch.softmax(logits / temp, dim=-1)


# ──────────────────────────────────────────────────────────────
# NLP HELPERS
# ──────────────────────────────────────────────────────────────

def _empty_nlp() -> dict:
    return {
        "news_count":            0,
        "vader_mean":            0.0,
        "vader_std":             0.0,
        "vader_max":             0.0,
        "vader_min":             0.0,
        "finbert_mean":          0.0,
        "bullish_ratio":         0.0,
        "bearish_ratio":         0.0,
        "nlp_score_mean":        0.0,
        "nlp_score_max_abs":     0.0,
        "selected_news_count":   0,
        "weighted_sentiment":    0.0,
        "selected_avg_relevance":0.0,
    }


def _fetch_news_mongodb(
    symbol: str,
    hours: int = NEWS_LOOKBACK_H,
    reference_time: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch recent news from MongoDB. Returns empty DataFrame on failure."""
    try:
        import pymongo
        mongo_url = os.getenv("MONGO_URL", "mongodb://mongo-crawler:27017")
        mongo_db  = os.getenv("MONGO_DB", "crawler_db")
        client = pymongo.MongoClient(mongo_url, serverSelectionTimeoutMS=3_000)
        db     = client[mongo_db]

        if reference_time is None:
            ref_ts = datetime.now(timezone.utc)
        else:
            ref_ts = pd.Timestamp(reference_time)
            ref_ts = ref_ts.tz_localize("UTC") if ref_ts.tzinfo is None else ref_ts.tz_convert("UTC")
            ref_ts = ref_ts.to_pydatetime()

        since = ref_ts - timedelta(hours=hours)
        # Filter by symbol when stored (news_crawler sets this); also accept
        # docs without symbol field (crawler-service doesn't always set it).
        sym_filter = {"$or": [{"symbol": symbol}, {"symbol": {"$exists": False}}, {"symbol": None}]}
        query = {"created_at": {"$gte": since, "$lt": ref_ts}, **sym_filter}
        docs  = list(
            db["news_articles"].find(
                query,
                {"_id": 0, "title": 1, "content": 1, "source": 1,
                 "created_at": 1, "published_at": 1,
                  "sentiment_score": 1, "sentiment": 1,
                 "url": 1, "article_id": 1, "id": 1}
            ).sort("created_at", -1).limit(500)
        )
        if not docs:
            return pd.DataFrame()

        df = pd.DataFrame(docs)

        # Normalise timestamp: crawler-service uses "created_at", news_crawler uses "created_at"
        # Use published_at if available, else fall back to created_at
        if "published_at" not in df.columns:
            df["published_at"] = df.get("created_at")
        else:
            df["published_at"] = df["published_at"].fillna(df.get("created_at"))

        # Parse datetime, handling missing timezone info
        try:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors='coerce')
        except Exception:
            # Fallback: parse without forcing UTC, then localize
            df["published_at"] = pd.to_datetime(df["published_at"], errors='coerce')
            if df["published_at"].dt.tz is None:
                df["published_at"] = df["published_at"].dt.tz_localize("UTC")

        # Normalise sentiment: crawler-service stores "sentiment" (dict or scalar),
        # news_crawler stores "sentiment_score" (scalar).
        if "sentiment_score" not in df.columns:
            df["sentiment_score"] = 0.0
        if "sentiment" in df.columns:
            def _extract_score(s):
                if isinstance(s, dict):
                    pos = float(s.get("positive", s.get("pos", 0)) or 0)
                    neg = float(s.get("negative", s.get("neg", 0)) or 0)
                    return pos - neg
                try:
                    return float(s)
                except Exception:
                    return 0.0
            mask = df["sentiment_score"] == 0.0
            df.loc[mask, "sentiment_score"] = df.loc[mask, "sentiment"].apply(_extract_score)
            df.drop(columns=["sentiment"], inplace=True, errors="ignore")

        if "source" not in df.columns:
            df["source"] = "unknown"
        logger.info("Fetched %d news articles from MongoDB", len(df))
        return df

    except Exception as e:
        logger.warning("MongoDB news fetch failed: %s — using empty NLP features", e)
        return pd.DataFrame()


def _nlp_for_window(
    news_df: pd.DataFrame,
    window_end: pd.Timestamp,
    lookback_minutes: int = 60,
    symbol: str = "BTCUSDT",
) -> dict:
    """Compute aggregated NLP features for a single 1h window ending at window_end."""
    if news_df.empty:
        return _empty_nlp()

    window_start = window_end - timedelta(minutes=lookback_minutes)
    window = news_df[
        (news_df["published_at"] >= window_start) &
        (news_df["published_at"] <  window_end)
    ].copy()

    if window.empty:
        return _empty_nlp()

    # Score with VADER (default) or FinBERT if USE_FINBERT=1 env var is set
    if "nlp_score" not in window.columns:
        window = add_nlp_scores(window, symbol=symbol, use_finbert=USE_FINBERT)

    agg = aggregate_nlp_window(window)

    # Selected-news features (top-3 by relevance)
    if "relevance_score" not in window.columns:
        window = add_relevance_scores(window, current_time=window_end)

    top3 = window.nlargest(3, "relevance_score")
    sel  = aggregate_selected_news_features(top3)
    agg["selected_news_count"]    = sel["selected_news_count"]
    agg["weighted_sentiment"]     = sel["weighted_sentiment"]
    agg["selected_avg_relevance"] = sel["selected_avg_relevance"]

    return agg


# ──────────────────────────────────────────────────────────────
# FEATURE BUILDER
# ──────────────────────────────────────────────────────────────

def _load_feature_cols(symbol: str, horizon: str) -> list[str]:
    """
    Load feature_cols from saved meta JSON (XGB legacy path only).

    NOTE: SAFEAlertNet does NOT use column-aligned features — it receives a
    pre-extracted 63-dim tensor from extract_multiframe_market_features() when
    use_multiframe=True (primary path), or falls back to extracting numeric
    columns from the DataFrame row (pseudo-timeframe fallback).
    This function is called for the pseudo-timeframe fallback only.

    Returns empty list [] when no saved metadata exists; _run_safe_alert_net
    handles empty feature_cols by auto-selecting numeric columns from the row.
    """
    sym = symbol.lower()

    # Try SAFE-Alert metrics first
    safe_path = ARTIFACT_DIR / f"safe_alert_{sym}_{horizon}_metrics.json"
    if safe_path.exists():
        try:
            meta = json.loads(safe_path.read_text(encoding="utf-8"))
            if "feature_cols" in meta:
                return meta["feature_cols"]
        except Exception as e:
            logger.warning("Failed to load %s: %s", safe_path, e)

    # Try XGB metrics as legacy fallback
    xgb_path = ARTIFACT_DIR / f"xgb_direction_{sym}_{horizon}_metrics.json"
    if xgb_path.exists():
        try:
            meta = json.loads(xgb_path.read_text(encoding="utf-8"))
            cols = meta.get("feature_cols", [])
            if cols:
                return cols
        except Exception as e:
            logger.warning("Failed to load %s: %s", xgb_path, e)

    # Return empty list — _run_safe_alert_net will auto-select numeric columns
    logger.info("No feature_cols metadata for %s/%s; auto-select will be used.", symbol, horizon)
    return []


def build_live_feature_df(symbol: str, news_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, bool]:
    """
    Build a feature DataFrame from live market data + live news (HYBRID Eq.16-19 SUPPORT).

    HYBRID Multi-Timeframe Approach:
    - If 1m candles available from crawler: Extract TRUE 5-timeframe features (1m, 5m, 15m, 1h, 4h)
    - If only 1h candles available: Use PSEUDO-TIMEFRAME (all 5 encoders receive same 63D features)

    Args:
        symbol  : e.g. "BTCUSDT"
        news_df : Optional pre-fetched news. If None, fetched from MongoDB.

    Returns:
        (DataFrame, bool):
            - DataFrame: Last row is the feature vector for current candle
            - bool: True if using TRUE 5-timeframe (1m data), False if PSEUDO-timeframe (1h data fallback)
    """
    # ── 1. Live candles ────────────────────────────────────────
    try:
        from app.market_cache import get_candles
    except ImportError:
        # Fallback: direct import when running from v2/pipelines
        from market_cache import get_candles  # type: ignore

    print(f"[INFO] [build_live_feature_df] Fetching candles for {symbol}...")

    # HYBRID APPROACH (Eq.16-19 Multi-timeframe Support):
    # 1. Try 1m candles first (true 5-timeframe via extract_multiframe_market_features)
    # 2. Use 1h candles for NLP pipeline (existing compatibility)
    # 3. Inject extracted 1m features into final DataFrame when available

    use_multiframe = False
    mkt_features_1m = None

    try:
        # Try to get 1m candles for true 5-timeframe extraction (Eq.16-19)
        candles_1m = get_candles(symbol, limit=300, interval="1m")  # 300 1m ≈ 5 hours
        if len(candles_1m) >= 250:  # Minimum for 5-timeframe resampling
            print(f"[INFO] [build_live_feature_df] Got {len(candles_1m)} 1m candles -> Extracting TRUE 5-TIMEFRAME (Eq.16-19)")
            candles_1m = sorted(candles_1m, key=lambda x: x["time"])
            df_1m = pd.DataFrame(candles_1m)
            df_1m = df_1m.rename(columns={"time": "timestamp"})

            # Extract true 5-timeframe features: [1m, 5m, 15m, 1h, 4h] × 12 + 3 = 63
            mkt_features_1m = extract_multiframe_market_features(df_1m)
            print(f"[OK] [build_live_feature_df] Extracted 5-timeframe market features: shape={mkt_features_1m.shape}")
            use_multiframe = True
        else:
            print(f"[WARN]  [build_live_feature_df] Only {len(candles_1m)} 1m candles available (need ≥250), will use pseudo-timeframe")
    except Exception as e:
        print(f"[WARN]  [build_live_feature_df] 1m candle fetch failed ({e}), will use pseudo-timeframe")

    # Always fetch 1h candles for NLP pipeline (existing compatibility)
    candles = get_candles(symbol, limit=CANDLE_LIMIT, interval="1h")
    print(f"[INFO] [build_live_feature_df] Got {len(candles)} 1h candles (min needed: 60)")
    if len(candles) < 60:
        raise ValueError(
            f"Not enough 1h candles in market_cache for {symbol}: "
            f"got {len(candles)}, need ≥60. "
            "Wait for Kafka to populate or run collect_training_data.py."
        )

    # Sort oldest->newest
    candles = sorted(candles, key=lambda x: x["time"])

    df = pd.DataFrame(candles)
    df = df.rename(columns={"time": "timestamp"})
    df["symbol"] = symbol

    # ── 2. Market features ─────────────────────────────────────
    # build_market_features drops NaN rows, returns clean DataFrame
    # Live data can contain a flat indicator window (for example StochRSI when
    # RSI is constant). Dropping on *every* engineered column can therefore
    # remove all otherwise valid candles and make manual refresh appear stuck.
    # Keep the rows, forward-fill from past candles only, and use neutral values
    # for indicators that remain undefined. Training keeps the strict default.
    df = build_market_features(df, add_lags=True, drop_incomplete=False)
    df = df.replace([np.inf, -np.inf], np.nan).ffill()
    neutral_values = {
        "rsi_14": 50.0,
        "stoch_rsi": 0.5,
        "bb_pos": 0.5,
    }
    df = df.fillna(value=neutral_values).fillna(0.0)
    if df.empty:
        raise ValueError("build_market_features() returned empty DataFrame.")

    # Keep timestamp_dt for NLP window alignment
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) if "timestamp" in df.columns and not isinstance(df["timestamp"].iloc[0], (pd.Timestamp, str)) else pd.to_datetime(df["timestamp"])


    # ── 3. Fetch news ──────────────────────────────────────────
    if news_df is None:
        news_df = _fetch_news_mongodb(symbol, hours=NEWS_LOOKBACK_H, reference_time=df["timestamp_dt"].iloc[-1])

    # Ensure UTC timestamps
    if not news_df.empty:
        if news_df["published_at"].dt.tz is None:
            news_df = news_df.copy()
            news_df["published_at"] = news_df["published_at"].dt.tz_localize("UTC")

    # ── 4. NLP features for last candle (lag 0) ────────────────
    last_ts = df["timestamp_dt"].iloc[-1]
    nlp_now = _nlp_for_window(news_df, last_ts, lookback_minutes=60, symbol=symbol)
    for col, val in nlp_now.items():
        if col not in df.columns:
            df[col] = 0.0
        df.iloc[-1, df.columns.get_loc(col)] = float(val)

    # ── 5. NLP lag features (lag 1, 2, 3) ─────────────────────
    # lag=1 -> news window ending 1h before last_ts (matches training's shift(1))
    # lag=2 -> ending 2h before, lag=3 -> ending 3h before
    for lag in NLP_LAGS:
        lag_end = last_ts - timedelta(hours=lag)
        nlp_lag = _nlp_for_window(news_df, lag_end, lookback_minutes=60, symbol=symbol)
        for col in NLP_LAG_COLS:
            col_lag = f"{col}_lag{lag}"
            if col_lag not in df.columns:
                df[col_lag] = 0.0
            df.iloc[-1, df.columns.get_loc(col_lag)] = float(nlp_lag.get(col, 0.0))

    # ── 6. Align columns with training features ────────────────
    feat_1h = _load_feature_cols(symbol, "1h")
    feat_4h = _load_feature_cols(symbol, "4h")
    all_feat = set(feat_1h) | set(feat_4h)

    # DEBUG: Print actual columns from build_market_features + NLP
    print(f"[build_live_feature_df] Current columns: {sorted(df.columns.tolist())}")
    print(f"[build_live_feature_df] feat_1h ({len(feat_1h)}): {feat_1h[:10] if feat_1h else 'EMPTY'}")
    print(f"[build_live_feature_df] feat_4h ({len(feat_4h)}): {feat_4h[:10] if feat_4h else 'EMPTY'}")

    for col in all_feat:
        if col not in df.columns:
            df[col] = 0.0

    # ── 7. Inject multiframe features if available (Eq.16-19) ────
    if use_multiframe and mkt_features_1m is not None:
        if "market_features_63d" not in df.columns:
            # Object dtype is required because each cell stores one complete
            # 63-dimensional NumPy vector.
            df["market_features_63d"] = pd.Series(
                [None] * len(df), index=df.index, dtype="object"
            )
        # Scalar object assignment prevents pandas from interpreting the
        # vector as values to broadcast across multiple columns.
        df.at[df.index[-1], "market_features_63d"] = np.asarray(
            mkt_features_1m, dtype=np.float32
        ).reshape(-1)
        print(f"[build_live_feature_df] Injected TRUE 5-TIMEFRAME market features")
    else:
        print(f"[build_live_feature_df] Using PSEUDO-TIMEFRAME mode (1h data for all 5 encoders)")

    df["symbol"] = symbol
    return df, use_multiframe  # MODIFIED: Return multiframe flag


# ──────────────────────────────────────────────────────────────
# SAFE-ALERT NET INFERENCE (PyTorch path)
# ──────────────────────────────────────────────────────────────

def _load_safe_alert_net(symbol: str, horizon: str):
    """
    Load trained SAFEAlertNet from disk (cached per symbol+horizon).
    Falls back to 1h model if horizon-specific model not found.
    Returns model or None if checkpoint not found.
    """
    # All non-BTC demo symbols share the same transferred BTC weights. Cache
    # one model instance per horizon instead of loading ten identical copies.
    key = ("btcusdt" if symbol.upper() != "BTCUSDT" else symbol.lower(), horizon)
    if key in _SAFE_ALERT_MODELS:
        return _SAFE_ALERT_MODELS[key]

    # Checkpoint lookup order (matches train_safe_alert.py save convention):
    # 1. FINAL.pt (best by model_score, saved at end of Stage 3)
    # 2. best_epoch*.pt (best during Stage 3)
    # 3. Legacy: safe_alert_{symbol}_{horizon}.pt
    sym = symbol.lower()
    candidate_paths = [
        ARTIFACT_DIR / f"safe_alert_{sym}_{horizon}_FINAL.pt",
        *sorted(ARTIFACT_DIR.glob(f"safe_alert_{sym}_{horizon}_best_epoch*.pt"), reverse=True),
        *sorted(ARTIFACT_DIR.glob(f"safe_alert_{sym}_{horizon}_stage3_epoch*.pt"),
                key=lambda x: int(x.stem.split("epoch")[-1]), reverse=True),
        ARTIFACT_DIR / f"safe_alert_{sym}_{horizon}.pt",
    ]

    # Demo transfer mode for coins without their own trained checkpoint.
    # Candidate-specific artifacts still take precedence.
    if symbol.upper() != "BTCUSDT":
        candidate_paths.extend([
            ARTIFACT_DIR / f"safe_alert_btcusdt_{horizon}_FINAL.pt",
            ARTIFACT_DIR / f"safe_alert_btcusdt_{horizon}.pt",
        ])

    ckpt_path = None
    for p in candidate_paths:
        if isinstance(p, Path) and p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        # Fallback: search fold subdirectories (newest fold first, e.g. fold_3, fold_2, fold_1)
        fold_dirs = sorted(ARTIFACT_DIR.glob("fold_*"), reverse=True)
        for fold_dir in fold_dirs:
            for fname in [f"safe_alert_{horizon}_FINAL.pt"]:
                p = fold_dir / fname
                if p.exists():
                    ckpt_path = p
                    print(f"[live_infer] Found checkpoint in fold dir: {fold_dir.name}/{fname}")
                    break
            if ckpt_path is not None:
                break

    if ckpt_path is None:
        print(f"[live_infer] No checkpoint found for {symbol}/{horizon}")
        logger.debug("SAFEAlertNet checkpoint not found for %s/%s", symbol, horizon)
        _SAFE_ALERT_MODELS[key] = None
        return None

    print(f"[live_infer] Loading SAFEAlertNet from: {ckpt_path.name}")

    try:
        import torch
        from models.safe_alert_net import SAFEAlertNet, FACTOR_CLASSES as _FCL
        # Read the checkpoint before constructing the network.  Walk-forward
        # artifacts may have been trained with the scalar, bar, or hybrid
        # market encoder; constructing the default scalar network and relying
        # on strict=False silently leaves large parts of the model random.
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            print(f"[INFO] Extracted model_state from checkpoint ({len(state)} keys)")
        else:
            state = checkpoint
            print(f"[INFO] Using checkpoint directly ({len(state)} keys)")

        state_keys = tuple(state.keys())
        if any(k.startswith("market_enc.scalar_encoders.") for k in state_keys):
            market_input_mode = "hybrid"
        elif any(k.startswith("market_enc.encoders.0.net.") for k in state_keys):
            market_input_mode = "bar"
        else:
            market_input_mode = "scalar"
        bar_seq_len = int(checkpoint.get("bar_seq_len", 20)) if isinstance(checkpoint, dict) else 20
        bar_feat_dim = int(checkpoint.get("bar_feat_dim", 10)) if isinstance(checkpoint, dict) else 10
        print(f"[live_infer] Creating SAFEAlertNet (market_dim={_SAFE_ALERT_MARKET_DIM}, "
              f"K_15m=3, K_1h=4, K_4h=5, K_24h=8, market_input_mode={market_input_mode})...")
        # Session 26: explicit K_24h=8 + n_factors to protect against default
        # drift. Loading checkpoint afterwards will verify state_dict shapes.
        model = SAFEAlertNet(
            market_dim=_SAFE_ALERT_MARKET_DIM, has_news=True,
            K_15m=3, K_1h=4, K_4h=5, K_24h=8,
            n_factors=_FCL,
            market_input_mode=market_input_mode,
            bar_seq_len=bar_seq_len,
            bar_feat_dim=bar_feat_dim,
        )
        print(f"[live_infer] Loading state dict from {ckpt_path.name}...")

        # strict=False keeps live inference tolerant to checkpoint schema drift.
        _result = model.load_state_dict(state, strict=False)
        if _result.missing_keys or _result.unexpected_keys:
            print(
                f"[WARN] state_dict mismatch — missing={list(_result.missing_keys)[:5]}, "
                f"unexpected={list(_result.unexpected_keys)[:5]} (first 5). "
                "Checkpoint schema may differ from the current model."
            )
        model.eval()

        # P0 #1: rehydrate preprocessing state (market scaler) attached to the
        # model instance so _run_safe_alert_net can apply the SAME z-score
        # normalization that training used. Without this, raw features at
        # inference would drift far outside the training distribution and
        # produce garbage predictions (the production blocker).
        preproc = checkpoint.get("preprocessing_state") if isinstance(checkpoint, dict) else None
        if preproc is not None and preproc.get("market_feature_mean") is not None:
            import torch as _torch
            model._market_feature_mean = _torch.as_tensor(preproc["market_feature_mean"]).float()
            model._market_feature_std  = _torch.as_tensor(preproc["market_feature_std"]).float()
            model._market_feature_clip = float(preproc.get("market_feature_clip", 8.0))
            print(f"[OK] Loaded preprocessing state (mean/std/clip={model._market_feature_clip})")
        else:
            # Checkpoint predates the P0 #1 fix — warn loudly so operators know
            # predictions will be on RAW features and are likely unreliable.
            model._market_feature_mean = None
            model._market_feature_std  = None
            model._market_feature_clip = 8.0
            import warnings as _warnings
            _warnings.warn(
                f"[live_infer] Checkpoint {ckpt_path.name} has NO preprocessing_state "
                f"(pre-P0#1 artifact). Live inference will use RAW market features — "
                f"predictions will drift from training distribution. Retrain to refresh.",
                RuntimeWarning, stacklevel=2,
            )

        # Also attach trained temperature (for _temperature_softmax) when present.
        if isinstance(checkpoint, dict) and "temperature" in checkpoint:
            model._ckpt_temperature = float(checkpoint.get("temperature") or 1.0)
        else:
            model._ckpt_temperature = 1.0

        _SAFE_ALERT_MODELS[key] = model
        print(f"[OK] SAFEAlertNet loaded successfully for {symbol}/{horizon}")
        logger.info("[OK] SAFEAlertNet loaded successfully for %s/%s", symbol, horizon)
        return model
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to load SAFEAlertNet for {symbol}/{horizon}: {e}")
        print(traceback.format_exc())
        logger.error(f"[ERROR] Failed to load SAFEAlertNet for {symbol}/{horizon}: {e}")
        logger.error(traceback.format_exc())
        _SAFE_ALERT_MODELS[key] = None
        return None


def _build_article_tensors(
    news_df: pd.DataFrame,
    horizon: str,
    current_time: pd.Timestamp,
    symbol: str,
):
    """
    Build live article tensors aligned with training inputs.

    Uses per-horizon candidate selection, runtime FinBERT embeddings when
    available, and 14-dim metadata = 4 base features + 10 target-based
    sentiment slots (zero-filled when no factor match is found).
    Returns (emb_tensor, meta_tensor, meta_dicts) or None if no articles.
    """
    import torch

    # Return None if no data or missing published_at column
    if news_df.empty or "published_at" not in news_df.columns:
        return None

    window = select_candidates_for_horizon(news_df, current_time, horizon)

    if window.empty:
        return None

    lookback_h = 1 if horizon == "1h" else (4 if horizon == "4h" else 24)
    now_ts = current_time.timestamp()
    texts = [
        f"{str(row.get('title', '')).strip()} {str(row.get('content', '')).strip()}".strip()
        for _, row in window.iterrows()
    ]
    runtime_embeddings = _get_runtime_embedder().encode(texts)

    emb_list  = []
    meta_list = []
    meta_dicts = []

    for idx, (_, row) in enumerate(window.iterrows()):
        if "embedding" in row and isinstance(row["embedding"], (list, np.ndarray)):
            emb = np.asarray(row["embedding"], dtype=np.float32)
            if emb.shape != (768,):
                emb = runtime_embeddings[idx]
        else:
            emb = runtime_embeddings[idx]
        emb_list.append(emb)

        pub_ts = row["published_at"].timestamp() if hasattr(row["published_at"], "timestamp") else now_ts
        age_h  = max(0.0, (now_ts - pub_ts) / 3600.0)
        recency = float(np.exp(-age_h / (lookback_h + 1)))

        text_len = len(str(row.get("content", row.get("title", ""))))
        length_norm = min(1.0, text_len / 2000.0)

        # Paper Section 3.3.1 — same tier whitelist used at training time.
        # Keeps training / live-inference feature distributions consistent.
        from pipelines.safe_alert_dataset import (
            SOURCE_TIER1, SOURCE_TIER2,
            SOURCE_TIER1_SCORE, SOURCE_TIER2_SCORE,
        )
        src = str(row.get("source", "unknown")).lower()
        if any(needle in src for needle in SOURCE_TIER1):
            source_cred = SOURCE_TIER1_SCORE
        elif any(needle in src for needle in SOURCE_TIER2):
            source_cred = SOURCE_TIER2_SCORE
        else:
            source_cred = 0.30  # unknown source fallback (matches dataset default)

        sentiment_score = float(row.get("sentiment_score", row.get("nlp_score", 0.0)) or 0.0)
        novelty = min(1.0, abs(sentiment_score))

        factor_probs, entity_sent = _lookup_factor_cache(row)

        if factor_probs is None:
            factor_probs = _try_parse_factor_vector(
                row,
                keys=["factor_probs", "factor_probs_json", "factor_distribution", "factor_label"],
                expected_dim=len(FACTOR_NAMES),
            )
        if entity_sent is None:
            entity_sent = _try_parse_factor_vector(
                row,
                keys=["factor_sentiment", "entity_sentiment", "fsa_vector"],
                expected_dim=len(FACTOR_NAMES),
            )

        if factor_probs is None:
            factor_probs = _infer_factor_distribution(
                str(row.get("title", "")),
                str(row.get("content", "")),
                str(row.get("source", "unknown")),
            )
        if entity_sent is None:
            entity_sent = _estimate_entity_sentiment(
                str(row.get("title", "")),
                str(row.get("content", "")),
                str(row.get("source", "unknown")),
            )
        meta_vec = np.concatenate(
            [np.array([recency, length_norm, source_cred, novelty], dtype=np.float32), entity_sent]
        )
        if meta_vec.shape[0] < META_DIM:
            meta_vec = np.pad(meta_vec, (0, META_DIM - meta_vec.shape[0]))

        meta_list.append(meta_vec[:META_DIM].tolist())
        meta_dicts.append({
            "title": str(row.get("title", "")),
            "source": str(row.get("source", "unknown")),
            "url": str(row.get("url", "") or ""),
            "content": str(row.get("content", "") or ""),
            "published_at": str(row.get("published_at", "")),
            "relevance_score": float(row.get("relevance_score", 0.0)),
            "sentiment_score": sentiment_score,
            "category": str(row.get("category", "") or ""),
            "symbols": row.get("symbols", []) if isinstance(row.get("symbols", []), list) else [],
            "factor_probs": factor_probs.tolist(),
            "factor_sentiment": entity_sent.tolist(),
        })

    emb_tensor  = torch.tensor(np.stack(emb_list), dtype=torch.float32)   # (K, 768)
    meta_tensor = torch.tensor(meta_list, dtype=torch.float32)            # (K, 14)
    return emb_tensor, meta_tensor, meta_dicts


def _run_safe_alert_net(
    model,
    market_row: pd.Series,
    feature_cols: list[str],
    news_df: pd.DataFrame,
    horizon: str,
    current_time: pd.Timestamp,
    use_multiframe: bool = False,  # NEW: True if 1m 5-timeframe data, False if 1h pseudo-timeframe
    tau_h: float = 0.65,
    gamma_h: float = 0.60,
    temperature_h: float = 1.0,
    policy_confidence_source: str = "raw",
    symbol: str | None = None,
) -> dict | None:
    """
    Run a single SAFEAlertNet forward pass for one sample.

    Eq.16-19 Support:
    - If use_multiframe=True: Extract true 5-timeframe market features (stored as "market_features_63d")
    - If use_multiframe=False: Extract 63D features from columns (pseudo-timeframe fallback)

    Returns dict with signal/confidence/explanation or None on error.
    """
    import torch
    import torch.nn.functional as F
    print(f"[RUN] [_run_safe_alert_net] Starting inference for {horizon} ({'TRUE 5-TIMEFRAME' if use_multiframe else 'PSEUDO-TIMEFRAME'})...")
    try:
        # Market feature vector (B=1, M=63)
        print(f"[RUN] [_run_safe_alert_net] Building market features (M={_SAFE_ALERT_MARKET_DIM})...")

        # NEW: Handle true 5-timeframe or pseudo-timeframe
        if use_multiframe and "market_features_63d" in market_row:
            # Extract pre-computed 5-timeframe features
            mkt_vals = market_row["market_features_63d"].astype(np.float32).tolist()
            print(f"   [OK] [_run_safe_alert_net] Using TRUE 5-TIMEFRAME market features")
        else:
            # Fallback: if feature_cols is empty, use all numeric columns from market_row (exclude metadata)
            if not feature_cols:
                metadata_cols = {"symbol", "timestamp", "timestamp_dt", "market_features_63d"}
                feature_cols = [col for col in market_row.index if col not in metadata_cols and isinstance(market_row[col], (int, float, np.integer, np.floating))]
                print(f"   [_run_safe_alert_net] Using fallback feature_cols: {len(feature_cols)} columns from market_row")

            mkt_vals = []
            for col in feature_cols[:_SAFE_ALERT_MARKET_DIM]:
                v = market_row.get(col, 0.0)
                mkt_vals.append(float(v) if pd.notna(v) else 0.0)
            # Pad/trim to MARKET_DIM
            while len(mkt_vals) < _SAFE_ALERT_MARKET_DIM:
                mkt_vals.append(0.0)
            mkt_vals = mkt_vals[:_SAFE_ALERT_MARKET_DIM]
            print(f"   [DATA] [_run_safe_alert_net] Using PSEUDO-TIMEFRAME market features ({len(mkt_vals)} cols)")
        mkt_tensor = torch.tensor([mkt_vals], dtype=torch.float32)  # (1, M)

        # P0 #1: apply training-time z-score normalization. The scaler was
        # fit on train split only and persisted in the checkpoint; without
        # this the model sees raw features (distribution mismatch).
        # apply_market_normalization shares the exact math with
        # Dataset._normalize_market_features so training and inference cannot
        # silently diverge.
        from pipelines.safe_alert_dataset import apply_market_normalization
        mkt_tensor = apply_market_normalization(
            mkt_tensor,
            mean=getattr(model, "_market_feature_mean", None),
            std=getattr(model, "_market_feature_std",  None),
            clip=float(getattr(model, "_market_feature_clip", 8.0)),
        )

        print(f"[RUN] [_run_safe_alert_net] Market tensor shape: {mkt_tensor.shape}, sum={mkt_tensor.sum():.2f}, mean={mkt_tensor.mean():.6f}")

        # Article tensors
        print(f"[RUN] [_run_safe_alert_net] Building article tensors...")
        art_result = _build_article_tensors(news_df, horizon, current_time, str(market_row.get("symbol", "BTCUSDT")))
        if art_result is not None:
            emb_t, meta_t, meta_dicts = art_result
            # Add batch dim
            art_emb  = emb_t.unsqueeze(0)   # (1, K, 768)
            art_meta = meta_t.unsqueeze(0)   # (1, K, 4)
            art_mask = torch.ones(1, emb_t.shape[0], dtype=torch.bool)  # (1, K)
            print(f"[RUN] [_run_safe_alert_net] Article shapes: emb={art_emb.shape}, meta={art_meta.shape}, mask={art_mask.shape}")
        else:
            art_emb = art_meta = art_mask = None
            meta_dicts = []
            print(f"[RUN] [_run_safe_alert_net] No articles found")

        print(f"[RUN] [_run_safe_alert_net] Calling model forward pass...")
        # The model accepts symbol for API compatibility; the paper-final
        # Eq.9 query uses market summary + horizon embedding only.
        with torch.no_grad():
            out = model(
                market_feat=mkt_tensor,
                horizon=horizon,
                article_emb=art_emb,
                article_mask=art_mask,
                article_meta_vec=art_meta,
                symbol=symbol,
            )
        print(f"[OK] [_run_safe_alert_net] Model forward pass successful!")
        print(f"[RUN] [_run_safe_alert_net] Output keys: {out.keys()}")

        dir_logits = out["dir_logits"][0]     # (3,)
        confidence = float(out["confidence"][0])
        probs_tensor = _temperature_softmax(dir_logits, temperature_h)
        probs      = probs_tensor.tolist()  # [P_down, P_neutral, P_up]
        pred_class = int(dir_logits.argmax())                # 0=DOWN, 1=NEUTRAL, 2=UP
        print(f"[OK] [_run_safe_alert_net] Confidence: {confidence}, Signal: {['DOWN', 'NEUTRAL', 'UP'][pred_class]}")

        # Alert decision Eq.26
        policy_confidence = confidence
        if str(policy_confidence_source).strip().lower() == "position":
            policy_confidence = confidence * (1.0 - float(probs_tensor[1].item()))
        should_alert = bool((policy_confidence >= float(tau_h)) and (float(probs_tensor.max().item()) >= float(gamma_h)))

        # Structured explanation
        explanation = {}
        if out["attn_weights"] is not None and out["p_fac_all"] is not None:
            exp_out = model.explainer(
                alpha_tilde=out["attn_weights"][0],
                p_fac=out["p_fac_all"][0],
                article_meta=meta_dicts,
                K_h=model.K_h_map.get(horizon, 3),
                P=3,
                direction=pred_class,
                confidence=confidence,
                horizon=horizon,
                symbol=str(market_row.get("symbol", "BTCUSDT")),
            )
            explanation = exp_out
            selected_titles = set(exp_out.get("selected_news", []))
            explanation["selected_news"] = [
                {
                    key: item.get(key)
                    for key in (
                        "title", "source", "url", "content", "published_at",
                        "relevance_score", "sentiment_score", "category", "symbols",
                    )
                }
                for item in meta_dicts
                if item.get("title") in selected_titles
            ]

        dir_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        return {
            "signal":       dir_map.get(pred_class, "NEUTRAL"),
            "confidence":   confidence,
            "policy_confidence": float(policy_confidence),
            "policy_confidence_source": str(policy_confidence_source),
            "probs":        {"DOWN": probs[0], "NEUTRAL": probs[1], "UP": probs[2]},
            "should_alert": should_alert,
            "temperature": float(temperature_h),
            "selected_news":  explanation.get("selected_news", []),
            "top_factors":    explanation.get("top_factors", []),
            "factors_text":   explanation.get("factors_text", ""),
            "nl_explanation": explanation.get("nl_explanation", ""),
            "source":         "safe_alert_net",
            "news_fetched_count": int(len(news_df)),
            "candidate_count": int(len(meta_dicts)),
            "selected_count": int(len(explanation.get("selected_news", []))),
        }

    except Exception as e:
        logger.warning("SAFEAlertNet inference failed for %s: %s", horizon, e)
        return None


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def run_live_inference(symbol: str, news_df: pd.DataFrame | None = None) -> dict:
    """
    Run full SAFE-Alert inference from live market + news data.

    Args:
        symbol  : e.g. "BTCUSDT"
        news_df : Optional pre-fetched news DataFrame. Fetched from MongoDB if None.

    Returns:
        Signal dict (same format as infer_realtime_v2.py output) + "data_source": "live".
    """
    print(f"[START] [run_live_inference] Starting for {symbol}...")
    logger.info("Running live inference for %s...", symbol)

    df_live, use_multiframe = build_live_feature_df(symbol, news_df=news_df)
    row     = df_live.iloc[-1]

    if news_df is None:
        news_df = _fetch_news_mongodb(symbol, hours=NEWS_LOOKBACK_H, reference_time=row["timestamp_dt"])

    print(f"[INFO] [run_live_inference] Using {'TRUE 5-TIMEFRAME' if use_multiframe else 'PSEUDO-TIMEFRAME'} market features")

    print(f"[INFO] [run_live_inference] Last candle - timestamp: {row.get('timestamp', 'N/A')}, RSI: {row.get('rsi_14', 'N/A'):.2f}, close: {row.get('close', 'N/A'):.2f}")

    # SAFEAlertNet – sole inference path (Eq.26 alert decision per horizon)
    _SAN_SIGNAL_MAP = {"UP": "BUY", "DOWN": "SELL", "NEUTRAL": "HOLD", "BUY": "BUY", "SELL": "SELL"}

    def _neutral_horizon(horizon: str) -> dict:
        return {
            "signal": "HOLD", "confidence": 0.0, "final_prob": 0.333,
            "probs": {"UP": 0.333, "DOWN": 0.333, "NEUTRAL": 0.334},
            "selected_news": [], "top_factors": [], "factors_text": "", "explanation": "",
            "should_alert": False, "model_used": "SAFEAlertNet",
            "horizon": horizon,
        }

    current_time = pd.Timestamp(row["timestamp_dt"])
    current_time = current_time.tz_localize("UTC") if current_time.tz is None else current_time.tz_convert("UTC")
    feat_1h = _load_feature_cols(symbol, "1h")
    feat_4h = _load_feature_cols(symbol, "4h")
    p1_thresh = load_safe_alert_policy(symbol, "1h")
    p4_thresh = load_safe_alert_policy(symbol, "4h")

    san_model_1h = _load_safe_alert_net(symbol, "1h")
    san_1h_raw = None
    if san_model_1h is not None:
        print(f"[RUN] [1h] Model loaded, running inference...")
        san_1h_raw = _run_safe_alert_net(
            san_model_1h, row, feat_1h, news_df, "1h", current_time,
            use_multiframe=use_multiframe,
            tau_h=float(p1_thresh["tau"]), gamma_h=float(p1_thresh["gamma"]),
            temperature_h=float(p1_thresh.get("temperature", 1.0)),
            policy_confidence_source=str(p1_thresh.get("policy_confidence_source", "raw")),
            symbol=symbol,
        )
        if san_1h_raw:
            print(f"[OK] [1h] Inference result: signal={san_1h_raw.get('signal')}, conf={san_1h_raw.get('confidence')}")
        else:
            print(f"[WARN] [1h] Inference returned None")
    else:
        print(f"[WARN] [1h] Model NOT loaded - using neutral fallback")

    if san_1h_raw is not None:
        model_signal = san_1h_raw.get("signal", "NEUTRAL")
        model_conf = float(san_1h_raw.get("confidence", 0.0))
        probs = san_1h_raw.get("probs", {})

        print(f"[INFO] [1h] Model result: signal={model_signal}, conf={model_conf:.3f}")

        h1h = {
            "signal":        _SAN_SIGNAL_MAP.get(model_signal, "HOLD"),
            "confidence":    model_conf,
            "policy_confidence": san_1h_raw.get("policy_confidence", model_conf),
            "policy_confidence_source": san_1h_raw.get("policy_confidence_source", "raw"),
            "final_prob":    probs.get("UP", 0.333),
            "probs":         probs,
            "selected_news": san_1h_raw.get("selected_news", []),
            "top_factors":   san_1h_raw.get("top_factors", []),
            "factors_text":  san_1h_raw.get("factors_text", ""),
            "explanation":   san_1h_raw.get("nl_explanation", ""),
            "should_alert":  san_1h_raw.get("should_alert", False),
            "model_used":    "SAFEAlertNet" if symbol.upper() == "BTCUSDT" else "SAFEAlertNet · BTC transfer demo",
            "horizon":       "1h",
        }
    else:
        h1h = _neutral_horizon("1h")

    san_model_4h = _load_safe_alert_net(symbol, "4h")
    san_4h_raw = None
    if san_model_4h is not None:
        print(f"[RUN] [4h] Model loaded, running inference...")
        san_4h_raw = _run_safe_alert_net(
            san_model_4h, row, feat_4h, news_df, "4h", current_time,
            use_multiframe=use_multiframe,
            tau_h=float(p4_thresh["tau"]), gamma_h=float(p4_thresh["gamma"]),
            temperature_h=float(p4_thresh.get("temperature", 1.0)),
            policy_confidence_source=str(p4_thresh.get("policy_confidence_source", "raw")),
            symbol=symbol,
        )
        if san_4h_raw:
            print(f"[OK] [4h] Inference result: signal={san_4h_raw.get('signal')}, conf={san_4h_raw.get('confidence')}")
    else:
        print(f"[WARN] [4h] Model NOT loaded - using neutral fallback")

    if san_4h_raw is not None:
        model_signal = san_4h_raw.get("signal", "NEUTRAL")
        model_conf = float(san_4h_raw.get("confidence", 0.0))
        probs = san_4h_raw.get("probs", {})

        print(f"[INFO] [4h] Model result: signal={model_signal}, conf={model_conf:.3f}")

        h4h = {
            "signal":        _SAN_SIGNAL_MAP.get(model_signal, "HOLD"),
            "confidence":    model_conf,
            "policy_confidence": san_4h_raw.get("policy_confidence", model_conf),
            "policy_confidence_source": san_4h_raw.get("policy_confidence_source", "raw"),
            "final_prob":    probs.get("UP", 0.333),
            "probs":         probs,
            "selected_news": san_4h_raw.get("selected_news", []),
            "top_factors":   san_4h_raw.get("top_factors", []),
            "factors_text":  san_4h_raw.get("factors_text", ""),
            "explanation":   san_4h_raw.get("nl_explanation", ""),
            "should_alert":  san_4h_raw.get("should_alert", False),
            "model_used":    "SAFEAlertNet" if symbol.upper() == "BTCUSDT" else "SAFEAlertNet · BTC transfer demo",
            "horizon":       "4h",
        }
    else:
        h4h = _neutral_horizon("4h")

    alert_decision = decide_alert_multimodal_eq26(
        signal_1h=h1h["signal"],
        confidence_1h=h1h.get("policy_confidence", h1h["confidence"]),
        max_prob_1h=max(h1h["probs"].values()) if h1h["probs"] else 0.0,
        tau_1h=float(p1_thresh["tau"]),
        gamma_1h=float(p1_thresh["gamma"]),
        signal_4h=h4h["signal"],
        confidence_4h=h4h.get("policy_confidence", h4h["confidence"]),
        max_prob_4h=max(h4h["probs"].values()) if h4h["probs"] else 0.0,
        tau_4h=float(p4_thresh["tau"]),
        gamma_4h=float(p4_thresh["gamma"]),
    )

    alert_info = {
        "alert": alert_decision["alert"],
        "level": alert_decision["level"],
        "signal": alert_decision.get("signal", h1h["signal"]),
        "reason": alert_decision["reason"],
        "reasons": [
            f"1h SAFEAlertNet: {h1h['signal']} (conf={h1h['confidence']:.3f}, policy_conf={h1h.get('policy_confidence', h1h['confidence']):.3f}, p={max(h1h['probs'].values()) if h1h['probs'] else 0.0:.3f})",
            f"4h SAFEAlertNet: {h4h['signal']} (conf={h4h['confidence']:.3f}, policy_conf={h4h.get('policy_confidence', h4h['confidence']):.3f}, p={max(h4h['probs'].values()) if h4h['probs'] else 0.0:.3f})",
        ],
    }

    def _to_py(v):
        if not isinstance(v, (list, dict)) and pd.isna(v):
            return None
        return v.item() if hasattr(v, "item") else v

    result = {
        "timestamp":   str(current_time),
        "symbol":      symbol,
        "data_source": "live",
        "alert":       alert_info,
        "horizon_1h":  h1h,
        "horizon_4h":  h4h,
        "top_inputs": {
            "rsi_14":         _to_py(row.get("rsi_14")),
            "macd_hist":      _to_py(row.get("macd_hist")),
            "bb_pos":         _to_py(row.get("bb_pos")),
            "stoch_rsi":      _to_py(row.get("stoch_rsi")),
            "log_return_1":   _to_py(row.get("log_return_1")),
            "return_5":       _to_py(row.get("return_5")),
            "volume_spike":   _to_py(row.get("volume_spike")),
            "vader_mean":     _to_py(row.get("vader_mean")),
            "nlp_score_mean": _to_py(row.get("nlp_score_mean")),
            "bullish_ratio":  _to_py(row.get("bullish_ratio")),
            "news_count":     _to_py(row.get("news_count")),
            "news_fetched_count": int(len(news_df)),
            "candidate_count_1h": int(
                san_1h_raw.get("candidate_count", 0) if san_1h_raw else 0
            ),
            "selected_count_1h": int(
                san_1h_raw.get("selected_count", 0) if san_1h_raw else 0
            ),
            "candidate_count_4h": int(
                san_4h_raw.get("candidate_count", 0) if san_4h_raw else 0
            ),
            "selected_count_4h": int(
                san_4h_raw.get("selected_count", 0) if san_4h_raw else 0
            ),
        },
    }

    # Save to disk (per-symbol file)
    out_path = ARTIFACT_DIR / f"latest_signal_{symbol.lower()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Signal saved -> %s  [alert=%s level=%s]",
                out_path.name, result["alert"].get("alert"), result["alert"].get("level"))

    return result


def get_cached_signal(symbol: str) -> dict | None:
    """Return last cached signal dict from disk, or None if not found."""
    path = ARTIFACT_DIR / f"latest_signal_{symbol.lower()}.json"
    if not path.exists():
        # Backward-compat: try generic latest_signal.json
        path = ARTIFACT_DIR / "latest_signal.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None
