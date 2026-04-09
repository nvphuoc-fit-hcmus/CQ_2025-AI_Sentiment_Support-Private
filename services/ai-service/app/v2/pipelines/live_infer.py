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
from preprocessing.news_features import add_nlp_scores, aggregate_nlp_window
from nlp.relevance_scorer import add_relevance_scores
from nlp.news_selector import aggregate_selected_news_features
from pipelines.utils import ARTIFACT_DIR, load_safe_alert_policy

logger = logging.getLogger("v2.live_infer")

# ── SAFEAlertNet constants ───────────────────────────────────────
_SAFE_ALERT_MODELS: dict = {}   # cache: (symbol, horizon) -> SAFEAlertNet instance
_SAFE_ALERT_MARKET_DIM = 63
_SAFE_ALERT_MAX_ARTICLES = 8    # K_max articles per inference call

# ── Config ──────────────────────────────────────────────────────
NLP_LAGS          = [1, 2, 3]          # lag cols: vader_mean_lag1-3 etc.
NLP_LAG_COLS      = ["vader_mean", "nlp_score_mean", "bullish_ratio", "bearish_ratio"]
NEWS_LOOKBACK_H   = 5                  # fetch 5h of news (enough for lag 0..3 + margin)
CANDLE_LIMIT      = 250                # need ≥ 200 for SMA50 + lags + LSTM seq_len

DROP_COLS = {
    "symbol", "timestamp", "timestamp_dt",
    "target_return_1h", "target_return_4h",
    "target_direction_1h", "target_direction_4h",
}


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


def _fetch_news_mongodb(symbol: str, hours: int = NEWS_LOOKBACK_H) -> pd.DataFrame:
    """Fetch recent news from MongoDB. Returns empty DataFrame on failure."""
    try:
        import pymongo
        mongo_url = os.getenv("MONGO_URL", "mongodb://mongo-crawler:27017")
        mongo_db  = os.getenv("MONGO_DB", "crawler_db")
        client = pymongo.MongoClient(mongo_url, serverSelectionTimeoutMS=3_000)
        db     = client[mongo_db]

        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        # Filter by symbol when stored (news_crawler sets this); also accept
        # docs without symbol field (crawler-service doesn't always set it).
        sym_filter = {"$or": [{"symbol": symbol}, {"symbol": {"$exists": False}}, {"symbol": None}]}
        query = {"created_at": {"$gte": since}, **sym_filter}
        docs  = list(
            db["news_articles"].find(
                query,
                {"_id": 0, "title": 1, "content": 1, "source": 1,
                 "created_at": 1, "published_at": 1,
                 "sentiment_score": 1, "sentiment": 1}
            ).limit(500)
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
    df = build_market_features(df, add_lags=True)
    if df.empty:
        raise ValueError("build_market_features() returned empty DataFrame.")

    # Keep timestamp_dt for NLP window alignment
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True) if "timestamp" in df.columns and not isinstance(df["timestamp"].iloc[0], (pd.Timestamp, str)) else pd.to_datetime(df["timestamp"])


    # ── 3. Fetch news ──────────────────────────────────────────
    if news_df is None:
        news_df = _fetch_news_mongodb(symbol, hours=NEWS_LOOKBACK_H)

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
            df["market_features_63d"] = None
        df.iloc[-1, df.columns.get_loc("market_features_63d")] = mkt_features_1m
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
    key = (symbol.lower(), horizon)
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

    ckpt_path = None
    for p in candidate_paths:
        if isinstance(p, Path) and p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        # Fallback: search fold subdirectories (newest fold first, e.g. fold_3, fold_2, fold_1)
        fold_dirs = sorted(ARTIFACT_DIR.glob("fold_*"), reverse=True)
        for fold_dir in fold_dirs:
            for fname in [f"safe_alert_{horizon}_FINAL.pt",
                          f"safe_alert_1h_FINAL.pt"]:  # 1h as secondary fallback
                p = fold_dir / fname
                if p.exists():
                    ckpt_path = p
                    print(f"[live_infer] Found checkpoint in fold dir: {fold_dir.name}/{fname}")
                    break
            if ckpt_path is not None:
                break

    if ckpt_path is None:
        # Final fallback: try 1h model when 4h not found (root artifacts dir)
        if horizon == "4h":
            print(f"[live_infer] 4h model not found, falling back to 1h model...")
            for p in [
                ARTIFACT_DIR / f"safe_alert_{sym}_1h_FINAL.pt",
                *sorted(ARTIFACT_DIR.glob(f"safe_alert_{sym}_1h_best_epoch*.pt"), reverse=True),
            ]:
                if isinstance(p, Path) and p.exists():
                    ckpt_path = p
                    break

    if ckpt_path is None:
        print(f"[live_infer] No checkpoint found for {symbol}/{horizon}")
        logger.debug("SAFEAlertNet checkpoint not found for %s/%s", symbol, horizon)
        _SAFE_ALERT_MODELS[key] = None
        return None

    print(f"[live_infer] Loading SAFEAlertNet from: {ckpt_path.name}")

    try:
        import torch
        from models.safe_alert_net import SAFEAlertNet
        print(f"[live_infer] Creating SAFEAlertNet (market_dim={_SAFE_ALERT_MARKET_DIM}, K_1h=4, K_4h=5)...")
        model = SAFEAlertNet(market_dim=_SAFE_ALERT_MARKET_DIM, has_news=True, K_1h=4, K_4h=5)
        print(f"[live_infer] Loading state dict from {ckpt_path.name}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle nested checkpoint format (extract model_state if present)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state = checkpoint["model_state"]
            print(f"[INFO] Extracted model_state from checkpoint ({len(state)} keys)")
        else:
            state = checkpoint
            print(f"[INFO] Using checkpoint directly ({len(state)} keys)")

        model.load_state_dict(state)
        model.eval()
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


def _build_article_tensors(news_df: pd.DataFrame, horizon: str, current_time: pd.Timestamp):
    """
    Build article_emb (K, 768), article_meta (K, 4), article_meta_list from news_df.
    Uses pre-computed 'embedding' column if available, else zeros.
    Window: 1h lookback for 1h horizon, 4h lookback for 4h horizon.
    Returns (emb_tensor, meta_tensor, meta_dicts) or None if no articles.
    """
    import torch

    # Return None if no data or missing published_at column
    if news_df.empty or "published_at" not in news_df.columns:
        return None

    lookback_h = 1 if horizon == "1h" else 4
    cutoff = current_time - timedelta(hours=lookback_h)
    window = news_df[
        (news_df["published_at"] >= cutoff) &
        (news_df["published_at"] <= current_time)
    ].head(_SAFE_ALERT_MAX_ARTICLES).copy()

    if window.empty:
        return None

    K = len(window)
    now_ts = current_time.timestamp()

    emb_list  = []
    meta_list = []
    meta_dicts = []

    for _, row in window.iterrows():
        # Embedding: use stored vector if available, else zeros
        if "embedding" in row and isinstance(row["embedding"], (list, np.ndarray)):
            emb = np.asarray(row["embedding"], dtype=np.float32)
            if len(emb) != 768:
                emb = np.zeros(768, dtype=np.float32)
        else:
            emb = np.zeros(768, dtype=np.float32)
        emb_list.append(emb)

        # Metadata: [recency_norm, length_norm, source_cred, novelty_norm]
        pub_ts = row["published_at"].timestamp() if hasattr(row["published_at"], "timestamp") else now_ts
        age_h  = max(0.0, (now_ts - pub_ts) / 3600.0)
        recency = float(np.exp(-age_h / (lookback_h + 1)))

        text_len = len(str(row.get("content", row.get("title", ""))))
        length_norm = min(1.0, text_len / 2000.0)

        src = str(row.get("source", "unknown")).lower()
        SOURCE_CRED = {"reuters": 0.98, "bloomberg": 0.98, "coindesk": 0.85,
                       "cointelegraph": 0.80, "decrypt": 0.75}
        source_cred = SOURCE_CRED.get(src, 0.50)

        sentiment_score = float(row.get("sentiment_score", row.get("nlp_score", 0.0)) or 0.0)
        novelty = min(1.0, abs(sentiment_score))

        meta_list.append([recency, length_norm, source_cred, novelty])
        meta_dicts.append({"title": str(row.get("title", ""))})

    emb_tensor  = torch.tensor(np.stack(emb_list), dtype=torch.float32)   # (K, 768)
    meta_tensor = torch.tensor(meta_list, dtype=torch.float32)            # (K, 4)
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
    from models.safe_alert_net import HORIZON_VOCAB, FACTOR_NAMES

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
        print(f"[RUN] [_run_safe_alert_net] Market tensor shape: {mkt_tensor.shape}, sum={mkt_tensor.sum():.2f}, mean={mkt_tensor.mean():.6f}")

        # Article tensors
        print(f"[RUN] [_run_safe_alert_net] Building article tensors...")
        art_result = _build_article_tensors(news_df, horizon, current_time)
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
        with torch.no_grad():
            out = model(
                market_feat=mkt_tensor,
                horizon=horizon,
                article_emb=art_emb,
                article_mask=art_mask,
                article_meta_vec=art_meta,
            )
        print(f"[OK] [_run_safe_alert_net] Model forward pass successful!")
        print(f"[RUN] [_run_safe_alert_net] Output keys: {out.keys()}")

        dir_logits = out["dir_logits"][0]     # (3,)
        confidence = float(out["confidence"][0])
        probs      = F.softmax(dir_logits, dim=-1).tolist()  # [P_down, P_neutral, P_up]
        pred_class = int(dir_logits.argmax())                # 0=DOWN, 1=NEUTRAL, 2=UP
        print(f"[OK] [_run_safe_alert_net] Confidence: {confidence}, Signal: {['DOWN', 'NEUTRAL', 'UP'][pred_class]}")

        # Alert decision Eq.26
        alert = model.alert_decision(
            out["dir_logits"], out["confidence"],
            tau_h=tau_h, gamma_h=gamma_h
        )
        should_alert = bool(alert[0].item())

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
            )
            explanation = exp_out

        dir_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
        return {
            "signal":       dir_map.get(pred_class, "NEUTRAL"),
            "confidence":   confidence,
            "probs":        {"DOWN": probs[0], "NEUTRAL": probs[1], "UP": probs[2]},
            "should_alert": should_alert,
            "selected_news":  explanation.get("selected_news", []),
            "top_factors":    explanation.get("top_factors", []),
            "nl_explanation": explanation.get("nl_explanation", ""),
            "source":         "safe_alert_net",
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

    # Fetch news once here so the same DataFrame is passed to both
    # build_live_feature_df (NLP feature computation) and infer_horizon
    # (per-horizon selective news selection). If we let build_live_feature_df
    # fetch internally, the outer news_df stays None and infer_horizon gets
    # no news -> selected_news always empty.
    if news_df is None:
        news_df = _fetch_news_mongodb(symbol, hours=NEWS_LOOKBACK_H)

    df_live, use_multiframe = build_live_feature_df(symbol, news_df=news_df)
    row     = df_live.iloc[-1]

    print(f"[INFO] [run_live_inference] Using {'TRUE 5-TIMEFRAME' if use_multiframe else 'PSEUDO-TIMEFRAME'} market features")

    print(f"[INFO] [run_live_inference] Last candle - timestamp: {row.get('timestamp', 'N/A')}, RSI: {row.get('rsi_14', 'N/A'):.2f}, close: {row.get('close', 'N/A'):.2f}")

    # SAFEAlertNet – sole inference path (Eq.26 alert decision per horizon)
    _SAN_SIGNAL_MAP = {"UP": "BUY", "DOWN": "SELL", "NEUTRAL": "HOLD", "BUY": "BUY", "SELL": "SELL"}

    def _neutral_horizon(horizon: str) -> dict:
        return {
            "signal": "HOLD", "confidence": 0.0, "final_prob": 0.333,
            "probs": {"UP": 0.333, "DOWN": 0.333, "NEUTRAL": 0.334},
            "selected_news": [], "top_factors": [], "explanation": "",
            "should_alert": False, "model_used": "SAFEAlertNet",
            "horizon": horizon,
        }

    current_time = pd.Timestamp.now(tz="UTC")  # FIXED: Keep UTC timezone for consistency
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
        )
        if san_1h_raw:
            print(f"[OK] [1h] Inference result: signal={san_1h_raw.get('signal')}, conf={san_1h_raw.get('confidence')}")
        else:
            print(f"[WARN] [1h] Inference returned None")
    else:
        print(f"[WARN] [1h] Model NOT loaded - using neutral fallback")

    if san_1h_raw is not None:
        # Check confidence vs technical signal
        model_signal = san_1h_raw.get("signal", "NEUTRAL")
        model_conf = san_1h_raw.get("confidence", 0.0)
        probs = san_1h_raw.get("probs", {})

        rsi = row.get("rsi_14", 50.0)
        macd = row.get("macd_hist", 0.0)
        bb_pos = row.get("bb_pos", 0.5)

        print(f"[INFO] [1h] Model result: signal={model_signal}, conf={model_conf:.3f}, RSI={rsi:.1f}, MACD={macd:.2f}, BB={bb_pos:.2f}")

        # Enhanced confidence: blend model output with model's probability certainty
        enhanced_conf = model_conf
        if probs:
            max_prob = max(probs.values())
            # Only boost confidence when model is MORE certain (max_prob > 0.40)
            # Normalize: 0.40 -> 0.0 (no boost), 1.0 -> 1.0 (high boost)
            if max_prob > 0.40:
                model_certainty = (max_prob - 0.40) / 0.60  # range [0, 1]
                # Boost: keep model_conf, but allow up to increased certainty boost
                boost = model_certainty * 0.3  # max +0.30 boost
                enhanced_conf = model_conf + boost  # additive boost, not blending
                print(f"   [1h] Confidence boost: base={model_conf:.3f}, certainty={model_certainty:.3f} -> {enhanced_conf:.3f} (+{boost:.3f})")

        # ALWAYS check EXTREME conditions first (override model completely)
        if rsi >= 90:  # Extreme overbought
            print(f"[WARN] [1h] EXTREME OVERBOUGHT: RSI={rsi:.1f} >= 90 -> forcing SELL with 0.95 confidence")
            model_signal = "SELL"
            enhanced_conf = 0.95  # Very high confidence
        elif rsi <= 10:  # Extreme oversold
            print(f"[WARN] [1h] EXTREME OVERSOLD: RSI={rsi:.1f} <= 10 -> forcing BUY with 0.95 confidence")
            model_signal = "BUY"
            enhanced_conf = 0.95  # Very high confidence
        # If not extreme, check if model not confident (conf < 0.4) for technical override
        elif model_conf < 0.4:
            signal_changed = False
            technical_strength = 0.0

            # Strong technical signals - override SIGNAL + boost confidence
            if rsi < 30 and macd < 0:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = min(1.0, (30 - rsi) / 30 * 0.5 + abs(macd) / 100 * 0.5)
            elif rsi > 70 and macd > 0:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = min(1.0, (rsi - 70) / 30 * 0.5 + macd / 100 * 0.5)
            # Moderate signals: RSI 30-50 + MACD < 0 = BUY lean
            elif 30 <= rsi <= 50 and macd < -2:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = min(1.0, abs(macd) / 100 * 0.8)
            # Moderate signals: RSI 50-70 + MACD > 0 = SELL lean
            elif 50 <= rsi <= 70 and macd > 2:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = min(1.0, macd / 100 * 0.8)
            # Bollinger Bands: price at extremes
            elif bb_pos < 0.2 and rsi < 40:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = (0.2 - bb_pos) / 0.2 * 0.4 + (40 - rsi) / 40 * 0.4
            elif bb_pos > 0.8 and rsi > 60:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = (bb_pos - 0.8) / 0.2 * 0.4 + (rsi - 60) / 40 * 0.4

            if signal_changed:
                # Blend enhanced_conf with technical strength (80% model, 20% technical boost)
                technical_boost = technical_strength * 0.2
                enhanced_conf = enhanced_conf + technical_boost
                print(f"[CHANGE] [1h] Technical override: RSI={rsi:.1f}, MACD={macd:.2f}, BB={bb_pos:.2f} -> signal={model_signal}, conf={enhanced_conf:.3f} (+{technical_boost:.3f})")

        # Clamp confidence to [0, 1]
        enhanced_conf = min(1.0, max(0.0, enhanced_conf))
        model_conf = enhanced_conf

        h1h = {
            "signal":        _SAN_SIGNAL_MAP.get(model_signal, "HOLD"),
            "confidence":    model_conf,
            "final_prob":    san_1h_raw.get("probs", {}).get("UP", 0.333),
            "probs":         san_1h_raw.get("probs", {}),
            "selected_news": san_1h_raw.get("selected_news", []),
            "top_factors":   san_1h_raw.get("top_factors", []),
            "explanation":   san_1h_raw.get("nl_explanation", ""),
            "should_alert":  san_1h_raw.get("should_alert", False),
            "model_used":    "SAFEAlertNet",
            "horizon":       "1h",
        }
    else:
        # Use technical indicators to generate mock signal
        rsi = row.get("rsi_14", 50.0)
        macd = row.get("macd_hist", 0.0)

        # Simple trading logic: RSI > 70 = SELL, RSI < 30 = BUY, else HOLD
        if rsi > 70:
            mock_signal = "SELL"
            mock_conf = min(0.85, (rsi - 70) / 30 * 0.85)
        elif rsi < 30:
            mock_signal = "BUY"
            mock_conf = min(0.85, (30 - rsi) / 30 * 0.85)
        else:
            mock_signal = "NEUTRAL"
            mock_conf = 0.3

        h1h = _neutral_horizon("1h")
        if mock_signal != "NEUTRAL":
            h1h["signal"] = {"BUY": "BUY", "SELL": "SELL"}.get(mock_signal, "HOLD")
            h1h["confidence"] = mock_conf
            h1h["final_prob"] = 0.6 if mock_signal in ["BUY", "SELL"] else 0.333

    san_model_4h = _load_safe_alert_net(symbol, "4h")
    san_4h_raw = None
    if san_model_4h is not None:
        print(f"[RUN] [4h] Model loaded, running inference...")
        san_4h_raw = _run_safe_alert_net(
            san_model_4h, row, feat_4h, news_df, "4h", current_time,
            use_multiframe=use_multiframe,
            tau_h=float(p4_thresh["tau"]), gamma_h=float(p4_thresh["gamma"]),
        )
        if san_4h_raw:
            print(f"[OK] [4h] Inference result: signal={san_4h_raw.get('signal')}, conf={san_4h_raw.get('confidence')}")
    else:
        print(f"[WARN] [4h] Model NOT loaded - using neutral fallback")

    if san_4h_raw is not None:
        # Check confidence vs technical signal (4h)
        model_signal = san_4h_raw.get("signal", "NEUTRAL")
        model_conf = san_4h_raw.get("confidence", 0.0)
        probs = san_4h_raw.get("probs", {})

        rsi = row.get("rsi_14", 50.0)
        macd = row.get("macd_hist", 0.0)
        bb_pos = row.get("bb_pos", 0.5)

        print(f"[INFO] [4h] Model result: signal={model_signal}, conf={model_conf:.3f}, RSI={rsi:.1f}, MACD={macd:.2f}, BB={bb_pos:.2f}")

        # Enhanced confidence: blend model output with model's probability certainty
        enhanced_conf = model_conf
        if probs:
            max_prob = max(probs.values())
            # Only boost confidence when model is MORE certain (max_prob > 0.40)
            # Normalize: 0.40 -> 0.0 (no boost), 1.0 -> 1.0 (high boost)
            if max_prob > 0.40:
                model_certainty = (max_prob - 0.40) / 0.60  # range [0, 1]
                # Boost: keep model_conf, but allow up to increased certainty boost
                boost = model_certainty * 0.3  # max +0.30 boost
                enhanced_conf = model_conf + boost  # additive boost, not blending
                print(f"   [4h] Confidence boost: base={model_conf:.3f}, certainty={model_certainty:.3f} -> {enhanced_conf:.3f} (+{boost:.3f})")

        # ALWAYS check EXTREME conditions first (override model completely)
        if rsi >= 90:  # Extreme overbought
            print(f"[WARN] [4h] EXTREME OVERBOUGHT: RSI={rsi:.1f} >= 90 -> forcing SELL with 0.95 confidence")
            model_signal = "SELL"
            enhanced_conf = 0.95  # Very high confidence
        elif rsi <= 10:  # Extreme oversold
            print(f"[WARN] [4h] EXTREME OVERSOLD: RSI={rsi:.1f} <= 10 -> forcing BUY with 0.95 confidence")
            model_signal = "BUY"
            enhanced_conf = 0.95  # Very high confidence
        # If not extreme, check if model not confident (conf < 0.4) for technical override
        elif model_conf < 0.4:
            signal_changed = False
            technical_strength = 0.0

            # Strong technical signals - override SIGNAL + boost confidence
            if rsi < 30 and macd < 0:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = min(1.0, (30 - rsi) / 30 * 0.5 + abs(macd) / 100 * 0.5)
            elif rsi > 70 and macd > 0:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = min(1.0, (rsi - 70) / 30 * 0.5 + macd / 100 * 0.5)
            # Moderate signals: RSI 30-50 + MACD < 0 = BUY lean
            elif 30 <= rsi <= 50 and macd < -2:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = min(1.0, abs(macd) / 100 * 0.8)
            # Moderate signals: RSI 50-70 + MACD > 0 = SELL lean
            elif 50 <= rsi <= 70 and macd > 2:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = min(1.0, macd / 100 * 0.8)
            # Bollinger Bands: price at extremes
            elif bb_pos < 0.2 and rsi < 40:
                model_signal = "BUY"
                signal_changed = True
                technical_strength = (0.2 - bb_pos) / 0.2 * 0.4 + (40 - rsi) / 40 * 0.4
            elif bb_pos > 0.8 and rsi > 60:
                model_signal = "SELL"
                signal_changed = True
                technical_strength = (bb_pos - 0.8) / 0.2 * 0.4 + (rsi - 60) / 40 * 0.4

            if signal_changed:
                # Blend enhanced_conf with technical strength (80% model, 20% technical boost)
                technical_boost = technical_strength * 0.2
                enhanced_conf = enhanced_conf + technical_boost
                print(f"[CHANGE] [4h] Technical override: RSI={rsi:.1f}, MACD={macd:.2f}, BB={bb_pos:.2f} -> signal={model_signal}, conf={enhanced_conf:.3f} (+{technical_boost:.3f})")

        # Clamp confidence to [0, 1]
        enhanced_conf = min(1.0, max(0.0, enhanced_conf))
        model_conf = enhanced_conf

        h4h = {
            "signal":        _SAN_SIGNAL_MAP.get(model_signal, "HOLD"),
            "confidence":    model_conf,
            "final_prob":    san_4h_raw.get("probs", {}).get("UP", 0.333),
            "probs":         san_4h_raw.get("probs", {}),
            "selected_news": san_4h_raw.get("selected_news", []),
            "top_factors":   san_4h_raw.get("top_factors", []),
            "explanation":   san_4h_raw.get("nl_explanation", ""),
            "should_alert":  san_4h_raw.get("should_alert", False),
            "model_used":    "SAFEAlertNet",
            "horizon":       "4h",
        }
    else:
        # Use technical indicators + 4h trend for mock signal (same RSI logic as 1h)
        rsi = row.get("rsi_14", 50.0)

        # Same simple trading logic as 1h: RSI > 70 = SELL, RSI < 30 = BUY, else HOLD
        if rsi > 70:
            mock_signal = "SELL"
            mock_conf = min(0.80, (rsi - 70) / 30 * 0.80)
        elif rsi < 30:
            mock_signal = "BUY"
            mock_conf = min(0.80, (30 - rsi) / 30 * 0.80)
        else:
            mock_signal = "NEUTRAL"
            mock_conf = 0.3

        h4h = _neutral_horizon("4h")
        if mock_signal != "NEUTRAL":
            h4h["signal"] = {"BUY": "BUY", "SELL": "SELL"}.get(mock_signal, "HOLD")
            h4h["confidence"] = mock_conf
            h4h["final_prob"] = 0.6 if mock_signal in ["BUY", "SELL"] else 0.333

    # Derive alert_info from SAFEAlertNet should_alert flags (both horizons)
    alert_1h = h1h["should_alert"] or h1h["confidence"] >= 0.80  # HIGH confidence triggers alert
    alert_4h = h4h["should_alert"] or h4h["confidence"] >= 0.80  # HIGH confidence triggers alert
    if alert_1h and alert_4h:
        alert_level = "high"
        alert_triggered = True
    elif alert_1h or alert_4h:
        alert_level = "medium"
        alert_triggered = True
    else:
        alert_level = "low"
        alert_triggered = False

    alert_info = {
        "alert":   alert_triggered,
        "level":   alert_level,
        "signal":  h1h["signal"],
        "reasons": [
            f"1h SAFEAlertNet: {h1h['signal']} (conf={h1h['confidence']:.3f})",
            f"4h SAFEAlertNet: {h4h['signal']} (conf={h4h['confidence']:.3f})",
        ],
    }

    def _to_py(v):
        if not isinstance(v, (list, dict)) and pd.isna(v):
            return None
        return v.item() if hasattr(v, "item") else v

    result = {
        "timestamp":   str(pd.Timestamp.now(tz="UTC")),
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
