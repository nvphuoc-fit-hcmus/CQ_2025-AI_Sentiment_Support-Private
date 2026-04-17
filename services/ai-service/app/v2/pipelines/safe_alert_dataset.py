"""
PyTorch Dataset for SAFE-Alert training.

Per-candle article windows: for each candle, select articles within [t-H, t]
where H is lookback horizon (1h, 4h, etc).

Label: next candle direction {0 (DOWN), 1 (NEUTRAL), 2 (UP)}
Factor labels: {0-9} - extracted from article keywords
"""

import sys
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing.market_features import extract_multiframe_market_features


# Load factor keywords
FACTOR_KEYWORDS = {
    "institutional_inflow":  ["institution","fund","grayscale","microstrategy","blackrock","corporate","treasury","bitcoin purchase","acquisition","buy","accumul"],
    "etf_flow":              ["etf","spot etf","bitcoin etf","sec approval","inflow","outflow","etf listing","issuance"],
    "regulatory_easing":     ["approved","legal","regulatory clarity","compliant","licensed","framework","clearance","authorize"],
    "regulatory_tightening": ["ban","crackdown","illegal","sanction","sec sue","enforcement","restrict","prohibited","suspension"],
    "exchange_risk":         ["hack","exploit","exchange down","withdrawal halt","insolvent","ftx","celsius","rug","compromised"],
    "liquidity_squeeze":     ["liquidity","leverage","liquidation","margin call","funding rate","squeeze","cascade","deleverag"],
    "whale_accumulation":    ["whale","transaction","on-chain","address","wallet","accumulate","hodl","large buy","holdings"],
    "macro_uncertainty":     ["inflation","fed","interest rate","recession","gdp","cpi","fomc","yield","economy","growth"],
    "protocol_upgrade":      ["upgrade","fork","halving","taproot","merge","protocol","layer2","lightning","launch","update"],
    "network_outage":        ["outage","congestion","fees spike","mempool","hash rate","51%","network issue","downtime"],
}

FACTOR_NAMES = list(FACTOR_KEYWORDS.keys())


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp column name to 'timestamp'."""
    normalized = {
        col: col.strip().lstrip("\ufeff").lower()
        for col in df.columns
    }
    if "timestamp" in normalized.values():
        inv = {v: k for k, v in normalized.items()}
        return df.rename(columns={inv["timestamp"]: "timestamp"})
    candidates = [
        "time", "date", "datetime", "open_time", "open_time_ms", "timestamp_ms", "ts",
        "published_at", "publishedat",
    ]
    for col in candidates:
        if col in normalized.values():
            inv = {v: k for k, v in normalized.items()}
            return df.rename(columns={inv[col]: "timestamp"})
    for col_raw, col_norm in normalized.items():
        if "time" in col_norm:
            return df.rename(columns={col_raw: "timestamp"})
    return df


class SAFEAlertDataset(torch.utils.data.Dataset):
    """SAFE-Alert training dataset (per-candle windows)."""

    def __init__(
        self,
        candle_df: pd.DataFrame,           # OHLCV + features
        article_embeddings: np.ndarray,    # (N_articles, 768) FinBERT
        article_meta: pd.DataFrame,        # article metadata (timestamp, source, length, etc)
        article_to_candle: Dict,           # {article_id → candle_idx} mapping
        symbol: str = "BTCUSDT",
        horizon: str = "1h",
        lookback_hours: int = 24,
        articles_per_candle: int = 8,
        min_articles: int = 0,
        precomputed_features: Optional[np.ndarray] = None,  # (52542, 63) precomputed market features
        factor_labels: Optional[np.ndarray] = None,          # (N_articles, 10) precomputed factor probs
        entity_sentiment: Optional[np.ndarray] = None,       # (N_articles, 10) target-based FSA scores [-1,+1]
        min_factor_confidence: float = 0.6,
    ):
        """
        Args:
            candle_df: candles with OHLCV + technical features
            article_embeddings: FinBERT 768-dim embeddings (precomputed)
            article_meta: article metadata
            article_to_candle: map article_id to nearest candle
            symbol: trading pair
            horizon: "15m", "1h", "4h", "24h"
            lookback_hours: days of articles to consider around each candle
            articles_per_candle: max articles per sample (pad/truncate to this)
            min_articles: skip candles with < min_articles. Default 0 keeps no-news
                          states so training matches production fallbacks.
            precomputed_features: optional (N_candles, 63) precomputed market features for speed
            factor_labels: optional (N_articles, 10) precomputed factor probability distributions.
                           Indexed in the same order as article_meta (articles_max.csv row order).
                           When provided, _extract_factor_distributions uses these directly
                           instead of on-the-fly keyword matching, fixing Lfac stuck at log(10).
            min_factor_confidence: drop low-confidence article occurrences from each
                           candle window when precomputed factor labels are available.
                           If all articles are filtered out, keep the single best article
                           to avoid empty-news windows caused only by confidence filtering.
        """
        self.candle_df = candle_df.reset_index(drop=True)

        # Horizon → how many candles forward to look for the label (PDF Eq.50)
        # 15m: 1 step ahead on 15m candles, 1h: 1 step on 1h, 4h: 4 steps on 1h, 24h: 24 steps on 1h
        _HORIZON_STEPS = {"15m": 1, "1h": 1, "4h": 4, "24h": 24}
        self.horizon_steps = _HORIZON_STEPS.get(horizon, 1)

        # Neutral-band threshold ε_h (PDF Eq.51): horizon-dependent to avoid labelling
        # small noise moves as directional.  Wider bands for longer horizons because
        # random-walk variance grows with √h.
        _EPSILON_H = {"15m": 0.001, "1h": 0.002, "4h": 0.005, "24h": 0.010}
        self.epsilon_h = _EPSILON_H.get(horizon, 0.002)
        # ✅ FIX #12: L2 normalize embeddings for scale consistency
        embeddings = torch.from_numpy(article_embeddings).float()
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # ✅ FIX #13: Log warning if zero embeddings detected
        zero_embs = (embeddings == 0).all(dim=1).sum().item()
        if zero_embs > 0:
            print(f"[WARN] SAFEAlertDataset: {zero_embs} zero embeddings detected out of {len(embeddings)}", flush=True)

        self.article_embeddings = embeddings
        self.article_meta = _ensure_timestamp_column(article_meta).reset_index(drop=True)
        self.article_to_candle = article_to_candle
        self.symbol = symbol
        self.horizon = horizon
        self.lookback_hours = lookback_hours
        self.articles_per_candle = articles_per_candle
        self.min_articles = min_articles
        self.min_factor_confidence = float(min_factor_confidence)
        self.market_feature_mean: Optional[torch.Tensor] = None
        self.market_feature_std: Optional[torch.Tensor] = None
        self.market_feature_clip: float = 8.0

        # Load or use precomputed market features
        if precomputed_features is not None:
            if len(precomputed_features) != len(self.candle_df):
                raise ValueError(
                    "precomputed_features length mismatch: "
                    f"got {len(precomputed_features)}, expected {len(self.candle_df)}. "
                    "Recompute features for the filtered candle dataframe."
                )
            self.precomputed_features = torch.from_numpy(precomputed_features).float()
            self.use_precomputed = True
        else:
            self.precomputed_features = None
            self.use_precomputed = False

        # Load or use precomputed factor probability distributions
        # Shape: (N_articles, 10) float32, indexed same order as article_meta
        if factor_labels is not None:
            expected_shape = (len(article_meta), len(FACTOR_NAMES))
            if factor_labels.shape != expected_shape:
                print(f"[WARN] SAFEAlertDataset: factor_labels shape {factor_labels.shape} "
                      f"does not match expected {expected_shape}. Falling back to on-the-fly.", flush=True)
                self.precomputed_factor_labels = None
            else:
                self.precomputed_factor_labels = factor_labels.astype(np.float32)
                self.article_max_prob = self.precomputed_factor_labels.max(axis=1)
                print(f"[OK] SAFEAlertDataset: using precomputed factor labels {factor_labels.shape} "
                      f"(fixes Lfac stuck at log(10)=2.303)", flush=True)
        else:
            self.precomputed_factor_labels = None
            self.article_max_prob = None

        # Load target-based FSA entity sentiment scores
        # Shape: (N_articles, 10) float32 in [-1, +1], one score per factor per article
        # This enables fine-grained sentiment analysis per financial entity/factor
        if entity_sentiment is not None:
            expected_shape = (len(article_meta), len(FACTOR_NAMES))
            if entity_sentiment.shape != expected_shape:
                print(f"[WARN] SAFEAlertDataset: entity_sentiment shape {entity_sentiment.shape} "
                      f"does not match expected {expected_shape}. Ignoring.", flush=True)
                self.entity_sentiment = None
            else:
                self.entity_sentiment = entity_sentiment.astype(np.float32)
                print(f"[OK] SAFEAlertDataset: using target-based FSA entity sentiment {entity_sentiment.shape}", flush=True)
        else:
            self.entity_sentiment = None

        # Source credibility: log-normalized corpus frequency.
        # cred(s) = log(count_s + 1) / log(count_max + 1) ∈ [0, 1].
        # More frequently published sources are treated as more established.
        # Replaces the hardcoded 7-source map where >80% of articles got 0.5 (constant).
        self._source_cred_map = self._build_source_cred_map()

        # Convert timestamps to datetime
        self.candle_df = _ensure_timestamp_column(self.candle_df)
        if "timestamp" not in self.candle_df.columns:
            raise KeyError("timestamp")
        if "timestamp" not in self.article_meta.columns:
            raise KeyError("timestamp")
        self.candle_df["timestamp"] = pd.to_datetime(self.candle_df["timestamp"], errors="coerce")
        self.article_meta["timestamp"] = pd.to_datetime(self.article_meta["timestamp"], errors="coerce")

        # Build reverse mapping: candle_idx → list of article indices
        self._build_candle_article_map()

        # Precompute valid sample indices (fast lookup)
        # Must leave horizon_steps candles at the end for label access (PDF Eq.50)
        max_idx = len(self.candle_df) - 1 - self.horizon_steps
        self.valid_idx = []
        for i in range(max_idx + 1):
            if len(self.candle_to_articles.get(i, [])) >= min_articles:
                self.valid_idx.append(i)

        print(f"[OK] SAFEAlertDataset: {len(self.valid_idx)}/{len(candle_df)} candles with articles", flush=True)

    def fit_market_scaler(self, dataset_indices: List[int] | range, clip_value: float = 8.0) -> None:
        """Fit train-only z-score normalization for market features.

        `dataset_indices` are indices in dataset space (i.e. positions inside `self.valid_idx`),
        not raw candle indices. This keeps scaling aligned with walk-forward subsets and avoids
        leakage from validation/test windows.
        """
        self.market_feature_clip = float(clip_value)

        if len(dataset_indices) == 0:
            self.market_feature_mean = None
            self.market_feature_std = None
            return

        candle_indices = [self.valid_idx[int(i)] for i in dataset_indices]
        if self.use_precomputed:
            train_feat = self.precomputed_features[candle_indices]
        else:
            train_feat = []
            for candle_idx in candle_indices:
                candle = self.candle_df.iloc[candle_idx]
                train_feat.append(torch.from_numpy(self._extract_market_features(candle, candle_idx)).float())
            train_feat = torch.stack(train_feat, dim=0)

        train_feat = torch.nan_to_num(train_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        mean = train_feat.mean(dim=0)
        std = train_feat.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)

        self.market_feature_mean = mean.float()
        self.market_feature_std = std.float()

    def _normalize_market_features(self, market_features: torch.Tensor) -> torch.Tensor:
        market_features = torch.nan_to_num(market_features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if self.market_feature_mean is not None and self.market_feature_std is not None:
            market_features = (market_features - self.market_feature_mean) / self.market_feature_std
            market_features = torch.clamp(
                market_features,
                min=-self.market_feature_clip,
                max=self.market_feature_clip,
            )
        return market_features

    def _build_source_cred_map(self) -> dict:
        """Combined source credibility: 0.6 * freq_score + 0.4 * length_score.

        freq_score  = log(count+1) / log(count_max+1)  — established sources publish more
        length_score = mean_content_len / 500, clipped to [0,1] — professional sources
                       write longer, more detailed articles vs. social/spam sources.

        Returns dict: source_name_lower → float in [0, 1].
        Unknown sources fall back to 0.3 at lookup time.
        """
        if 'source' not in self.article_meta.columns:
            return {}

        src_col = self.article_meta['source'].dropna().str.lower().str.strip()
        counts = src_col.value_counts()
        if counts.empty:
            return {}

        # Frequency score
        log_max = float(np.log1p(counts.iloc[0]))
        if log_max == 0:
            return {}
        freq_scores = {src: float(np.log1p(cnt) / log_max) for src, cnt in counts.items()}

        # Length score: mean content length per source, normalized to [0,1]
        length_scores = {}
        if 'content' in self.article_meta.columns:
            tmp = self.article_meta.copy()
            tmp['_src'] = src_col
            tmp['_len'] = tmp['content'].fillna('').astype(str).str.len()
            mean_len = tmp.groupby('_src')['_len'].mean()
            max_len = float(mean_len.max()) if len(mean_len) > 0 else 1.0
            if max_len > 0:
                length_scores = {src: float(min(l / max_len, 1.0)) for src, l in mean_len.items()}

        cred_map = {}
        for src in freq_scores:
            fs = freq_scores[src]
            ls = length_scores.get(src, fs)  # fallback to freq if no length data
            cred_map[src] = float(0.6 * fs + 0.4 * ls)

        print(f"[OK] SAFEAlertDataset: source_cred built from {len(cred_map)} sources "
              f"(top: {list(cred_map.keys())[:3]}, range=[{min(cred_map.values()):.2f},{max(cred_map.values()):.2f}])",
              flush=True)
        return cred_map

    def _build_candle_article_map(self):
        """Build candle_idx → article indices mapping using fast vectorized ops."""
        self.candle_to_articles = {}

        # Convert to numpy for fast operations
        candle_times = self.candle_df['timestamp'].values
        article_times = self.article_meta['timestamp'].values

        # For each candle (vectorized)
        for candle_idx in range(len(candle_times)):
            candle_time = candle_times[candle_idx]
            window_start = candle_time - np.timedelta64(self.lookback_hours, 'h')

            # Find articles in window
            # Strictly past-only news window: exclude articles at exact candle timestamp
            # to prevent ambiguous same-bar information leakage.
            mask = (article_times >= window_start) & (article_times < candle_time)
            article_indices = np.where(mask)[0].tolist()

            if len(article_indices) > 0:
                # Sort by recency
                article_indices.sort(
                    key=lambda i: article_times[i],
                    reverse=True
                )
                self.candle_to_articles[candle_idx] = article_indices

            # Progress report every 10k candles
            if (candle_idx + 1) % 10000 == 0:
                print(f"  Indexed {candle_idx + 1}/{len(candle_times)} candles...", flush=True)

    def _get_articles_for_candle(self, candle_idx: int) -> List[int]:
        """Get article indices from precomputed mapping."""
        return self.candle_to_articles.get(candle_idx, [])

    def __len__(self) -> int:
        return len(self.valid_idx)

    def _filter_article_indices_by_factor_confidence(self, article_indices: List[int]) -> List[int]:
        """Drop low-confidence article occurrences while keeping one best fallback."""
        if (
            self.min_factor_confidence <= 0.0
            or self.article_max_prob is None
            or len(article_indices) == 0
        ):
            return article_indices

        filtered = [
            art_idx for art_idx in article_indices
            if art_idx < len(self.article_max_prob)
            and self.article_max_prob[art_idx] >= self.min_factor_confidence
        ]
        if filtered:
            return filtered

        valid_original = [
            art_idx for art_idx in article_indices
            if art_idx < len(self.article_max_prob)
        ]
        if not valid_original:
            return article_indices

        best_art_idx = max(valid_original, key=lambda art_idx: float(self.article_max_prob[art_idx]))
        return [best_art_idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        candle_idx = self.valid_idx[idx]

        # Get current candle (ALWAYS needed for labels)
        candle = self.candle_df.iloc[candle_idx]

        # ✅ OPTIMIZATION: Use precomputed features (O(1) lookup) instead of computing each time
        if self.use_precomputed:
            market_features = self.precomputed_features[candle_idx]
        else:
            # Fallback: compute on-the-fly (slow, ~3-5 sec per sample)
            market_features_np = self._extract_market_features(candle, candle_idx)
            market_features = torch.from_numpy(market_features_np).float()
        market_features = self._normalize_market_features(market_features)

        # Articles (lookback window)
        article_indices = self._get_articles_for_candle(candle_idx)
        article_indices = self._filter_article_indices_by_factor_confidence(article_indices)
        article_emb, article_meta, article_mask = self._extract_articles(article_indices, candle_idx)

        if not torch.isfinite(market_features).all():
            raise ValueError(f"Non-finite market_features at candle_idx={candle_idx}")
        if not torch.isfinite(article_emb).all():
            raise ValueError(f"Non-finite article_embeddings at candle_idx={candle_idx}")
        if not torch.isfinite(article_meta).all():
            raise ValueError(f"Non-finite article_metadata at candle_idx={candle_idx}")

        # Label candle: horizon_steps ahead (PDF Eq.50)
        # 1h → candle_idx+1, 4h → candle_idx+4
        label_candle = self.candle_df.iloc[candle_idx + self.horizon_steps]
        exec_candle  = self.candle_df.iloc[candle_idx + 1]  # P_exec = next open (no look-ahead)

        # Direction label (PDF Eq.51)
        direction_label = self._get_direction_label(candle, label_candle, exec_candle)

        # Factor label (from articles)
        fac_labels = self._extract_factor_distributions(article_indices)

        # Return label (PDF Eq.50): r = (P_close_{t+h} - P_exec) / P_exec
        # P_exec = open of next 1h candle (execution price, avoids look-ahead bias)
        # P_close_{t+h} = close of the horizon candle (h steps ahead)
        ret_raw = (label_candle['close'] - exec_candle['open']) / exec_candle['open']
        ret_label = torch.tensor(ret_raw, dtype=torch.float32)

        return {
            "market_features": market_features,  # Already tensor from precomputed or computed above
            "article_embeddings": article_emb,                                  # (K, 768)
            "article_metadata": article_meta,                                   # (K, 4) base or (K, 14) with entity_sentiment
            "article_mask": article_mask,  # ✅ Use computed mask (with padding 0s)
            "direction": torch.tensor(direction_label, dtype=torch.long),
            "factor": fac_labels,             # ✅ FIX #10: (K, C) tensor, not scalar
            "return": ret_label,
        }

    def _extract_market_features(self, candle: pd.Series, candle_idx: int) -> np.ndarray:
        """
        Extract 63 market features from candle + historical context (NO PADDING).

        Eq.16-19 OPTIMIZED:
        - 5 timeframes × 12 indicators = 60 features
        - 3 aggregate cross-timeframe features = 3 features
        - Total: 63 features (ALL REAL DATA, zero padding)

        Args:
            candle: Single row from candle_df (should have OHLCV at minimum)
            candle_idx: Index of candle in dataframe (for fast lookup)

        Returns:
            (63,) array: 5×12 + 3 = 63 ALL REAL (no zeros except where unavailable)
        """
        # ✅ BUG FIX #2: Use direct index instead of timestamp re-lookup
        try:
            # Look back 250 candles for resampling to 5 timeframes
            lookback_idx = max(0, candle_idx - 250)
            df_window = self.candle_df.iloc[lookback_idx:candle_idx+1].copy()

            if len(df_window) >= 50:  # Minimum for resampling
                # Build 5-timeframe features: returns (63,) directly (no padding!)
                features = extract_multiframe_market_features(df_window)
                return features[:63].astype(np.float32)  # Ensure exactly 63
        except Exception as e:
            # Fall back to simple extraction if multiframe fails
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Multiframe extraction failed: {e}, using fallback")

        # FALLBACK: Extract all available technical indicators from candle row
        # ✅ BUG FIX #5: Use all actual features from dataset, minimize zero-padding
        # Try to extract as many real technical indicators as possible
        possible_features = [
            'close', 'rsi_14', 'rsi_7', 'rsi_21', 'macd', 'macd_hist', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_pos', 'atr', 'stoch_k', 'stoch_d',
            'stoch_rsi', 'log_return_1', 'log_return_5', 'log_return_20', 'return_5',
            'return_20', 'volume_spike', 'volatility_20', 'ema_12', 'ema_26', 'sma_20',
            'high', 'low', 'open', 'volume'
        ]
        features = []
        for col in possible_features:
            if col in candle.index:
                val = candle[col]
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)
            else:
                features.append(0.0)

        features = np.array(features, dtype=np.float32)
        if len(features) < 63:
            features = np.pad(features, (0, 63 - len(features)), 'constant', constant_values=0.0)
        return features[:63].astype(np.float32)

    def _extract_articles(self, article_indices: List[int], candle_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract article embeddings + metadata, pad to articles_per_candle.

        article_metadata layout (per PDF Eq.8 + target-based FSA extension):
          [0] recency_norm  - linear decay over 24h window: 1.0 (just pub) → 0.0 (24h old)
          [1] length_norm   - content length / max_source_length, clipped to [0, 1]
          [2] source_cred   - corpus-frequency credibility: log(count+1)/log(max+1) ∈ [0,1]
          [3] rank_norm     - relative recency rank in window: 1.0 (most recent) → 0.0 (oldest)
          [4:14] entity_sent - per-factor FinBERT sentiment [-1, +1]
                               (only if precomputed entity_sentiment provided)
        Total: 4 dims (base) or 14 dims (with target-based FSA)
        """
        K        = self.articles_per_candle
        meta_dim = 14 if self.entity_sentiment is not None else 4
        emb  = torch.zeros(K, 768,     dtype=torch.float32)
        meta = torch.zeros(K, meta_dim, dtype=torch.float32)
        mask = torch.zeros(K,           dtype=torch.float32)

        for i, art_idx in enumerate(article_indices[:K]):
            if art_idx < len(self.article_embeddings):
                emb[i] = self.article_embeddings[art_idx]

                # CRITICAL FIX: Correct metadata computation per PDF Eq.8
                # Meta should be: [recency_norm, length_norm, source_cred, novelty_norm]
                article = self.article_meta.iloc[art_idx]

                # Metadata 0: Recency — linear decay over 24h lookback window.
                # 1.0 = published at candle time, 0.0 = published 24h ago.
                try:
                    art_time    = pd.Timestamp(article['timestamp'] if 'timestamp' in article.index else datetime.utcnow())
                    candle_time = pd.Timestamp(self.candle_df.iloc[candle_idx]['timestamp'])
                    age_hours   = (candle_time - art_time).total_seconds() / 3600
                    recency     = max(0.0, 1.0 - (age_hours / 24))
                except (TypeError, ValueError, KeyError):
                    recency = 0.5

                meta[i, 0] = float(recency)

                # Metadata 1: Length normalization — content length relative to corpus max.
                try:
                    content_len = len(str(article['content'] if 'content' in article.index else ''))
                except (TypeError, AttributeError):
                    content_len = 0
                length_norm = min(1.0, content_len / 500.0)
                meta[i, 1] = float(length_norm)

                # Metadata 2: Source credibility — corpus-frequency log-normalized [0,1].
                # Built in __init__ from article_meta; unknown sources default to 0.3.
                try:
                    source = str(article['source'] if 'source' in article.index else '').lower().strip()
                except (TypeError, AttributeError):
                    source = ''
                source_cred = self._source_cred_map.get(source, 0.3)
                meta[i, 2] = float(source_cred)

                # Metadata 3: Relative rank within window (0=oldest, 1=most recent).
                # Distinct from recency (dim 0) which is absolute age decay.
                # Rank encodes ordering signal: is this among the freshest in the window?
                n_valid = min(len(article_indices), K)
                rank_score = float(n_valid - 1 - i) / max(n_valid - 1, 1)  # 1.0=most recent, 0.0=oldest
                meta[i, 3] = float(rank_score)

                # Metadata [4:14]: Target-based FSA entity sentiment (per factor)
                # Only populated when entity_sentiment precomputed file is available.
                if self.entity_sentiment is not None and art_idx < len(self.entity_sentiment):
                    meta[i, 4:14] = torch.from_numpy(self.entity_sentiment[art_idx])

                mask[i] = 1.0

        return emb, meta, mask

    def _get_direction_label(self, curr: pd.Series, label_candle: pd.Series,
                             exec_candle: pd.Series = None) -> int:
        """Compute direction: 0 (DOWN), 1 (NEUTRAL), 2 (UP).

        PDF Eq.51: r = (P_close_{t+h} - P_exec) / P_exec
        P_exec = open of next candle (execution price, no look-ahead).
        P_close_{t+h} = close of horizon candle (h steps ahead).

        For 1h: exec_candle == label_candle (same candle +1 step).
        For 4h: exec_candle = candle+1, label_candle = candle+4.
        """
        p_exec = exec_candle['open'] if exec_candle is not None else label_candle['open']
        ret = (label_candle['close'] - p_exec) / p_exec
        # PDF Eq.51: horizon-dependent neutral band ε_h
        # 15m: 0.1%, 1h: 0.2%, 4h: 0.5%, 24h: 1.0%
        eps = self.epsilon_h
        if ret > eps:
            return 2  # UP
        elif ret < -eps:
            return 0  # DOWN
        else:
            return 1  # NEUTRAL

    def _extract_factor_distributions(self, article_indices: List[int]) -> torch.Tensor:
        """
        Extract factor distributions PER-ARTICLE (Eq.14-15 PDF).

        Returns: (K, C) tensor - factor probability distribution for each article.
        - K = number of articles (padded to articles_per_candle)
        - C = 10 factor classes

        Each article gets a softmax probability over all 10 factors.

        When self.precomputed_factor_labels is set (loaded from article_factor_labels.npy),
        uses precomputed distributions indexed by article_idx.  This fixes Lfac stuck near
        log(10)=2.303 by searching both title AND content with extended keywords.

        Falls back to on-the-fly title-only keyword matching when not available.
        """
        K = self.articles_per_candle
        C = len(FACTOR_NAMES)  # 10 factor classes
        factor_probs = torch.zeros(K, C, dtype=torch.float32)

        # Uniform distribution (fallback for padding positions or missing articles)
        uniform = torch.ones(C, dtype=torch.float32) / C

        for i in range(K):
            if i < len(article_indices):
                art_idx = article_indices[i]
                if art_idx < len(self.article_meta):
                    # --- Path 1: use precomputed factor labels (preferred) ---
                    if self.precomputed_factor_labels is not None and art_idx < len(self.precomputed_factor_labels):
                        factor_probs[i] = torch.from_numpy(
                            self.precomputed_factor_labels[art_idx]
                        ).float()
                    else:
                        # --- Path 2: on-the-fly keyword matching (title only, fallback) ---
                        title = str(self.article_meta.iloc[art_idx].get('title', '')).lower()

                        scores = np.zeros(C, dtype=np.float32)
                        for j, (fname, keywords) in enumerate(FACTOR_KEYWORDS.items()):
                            for kw in keywords:
                                if kw.lower() in title:
                                    scores[j] += 1

                        # T=0.3: sharpens matched articles so dominant factor is clear.
                        # avoids near-uniform labels giving Lfac~log(C) with no gradient.
                        if scores.sum() > 0:
                            LABEL_TEMP = 0.3
                            factor_probs[i] = torch.softmax(
                                torch.tensor(scores / LABEL_TEMP, dtype=torch.float32), dim=0
                            )
                        else:
                            factor_probs[i] = uniform
                else:
                    factor_probs[i] = uniform
            else:
                # Padding position: uniform distribution
                factor_probs[i] = uniform

        return factor_probs  # (K, C)
