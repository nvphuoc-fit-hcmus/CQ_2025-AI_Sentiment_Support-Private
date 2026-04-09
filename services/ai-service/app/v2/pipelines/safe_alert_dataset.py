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
        min_articles: int = 1,
        precomputed_features: Optional[np.ndarray] = None,  # (52542, 63) precomputed market features
        factor_labels: Optional[np.ndarray] = None,          # (N_articles, 10) precomputed factor probs
        entity_sentiment: Optional[np.ndarray] = None,       # (N_articles, 10) target-based FSA scores [-1,+1]
    ):
        """
        Args:
            candle_df: candles with OHLCV + technical features
            article_embeddings: FinBERT 768-dim embeddings (precomputed)
            article_meta: article metadata
            article_to_candle: map article_id to nearest candle
            symbol: trading pair
            horizon: "1h", "4h"
            lookback_hours: days of articles to consider around each candle
            articles_per_candle: max articles per sample (pad/truncate to this)
            min_articles: skip candles with < min_articles
            precomputed_features: optional (N_candles, 63) precomputed market features for speed
            factor_labels: optional (N_articles, 10) precomputed factor probability distributions.
                           Indexed in the same order as article_meta (articles_max.csv row order).
                           When provided, _extract_factor_distributions uses these directly
                           instead of on-the-fly keyword matching, fixing Lfac stuck at log(10).
        """
        self.candle_df = candle_df.reset_index(drop=True)

        # Horizon → how many 1h candles forward to look for the label (PDF Eq.50)
        # 1h: next candle (+1), 4h: 4 steps ahead (+4), 24h: 24 steps ahead (+24)
        _HORIZON_STEPS = {"1h": 1, "4h": 4, "24h": 24}
        self.horizon_steps = _HORIZON_STEPS.get(horizon, 1)
        # ✅ FIX #12: L2 normalize embeddings for scale consistency
        embeddings = torch.from_numpy(article_embeddings).float()
        embeddings = embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # ✅ FIX #13: Log warning if zero embeddings detected
        zero_embs = (embeddings == 0).all(dim=1).sum().item()
        if zero_embs > 0:
            print(f"[WARN] SAFEAlertDataset: {zero_embs} zero embeddings detected out of {len(embeddings)}", flush=True)

        self.article_embeddings = embeddings
        self.article_meta = article_meta.reset_index(drop=True)
        self.article_to_candle = article_to_candle
        self.symbol = symbol
        self.horizon = horizon
        self.lookback_hours = lookback_hours
        self.articles_per_candle = articles_per_candle
        self.min_articles = min_articles

        # Load or use precomputed market features
        if precomputed_features is not None:
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
                print(f"[OK] SAFEAlertDataset: using precomputed factor labels {factor_labels.shape} "
                      f"(fixes Lfac stuck at log(10)=2.303)", flush=True)
        else:
            self.precomputed_factor_labels = None

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

        # Convert timestamps to datetime
        self.candle_df['timestamp'] = pd.to_datetime(self.candle_df['timestamp'])
        self.article_meta['timestamp'] = pd.to_datetime(self.article_meta['timestamp'])

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
            mask = (article_times >= window_start) & (article_times <= candle_time)
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

        # Articles (lookback window)
        article_indices = self._get_articles_for_candle(candle_idx)
        article_emb, article_meta, article_mask = self._extract_articles(article_indices, candle_idx)

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
            "article_metadata": article_meta,                                   # (K, 4)
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
          [0] recency_norm      - exponential decay over 1 week
          [1] length_norm       - content length / 500 chars
          [2] source_cred       - source credibility score [0, 1]
          [3] novelty_norm      - temporal freshness (24h decay)
          [4:14] entity_sent    - per-factor FinBERT sentiment [-1, +1]
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

                # Metadata 0: Recency (how fresh)
                # Use exponential decay based on timestamp difference
                try:
                    # ✅ FIX #8: Use bracket notation for pandas Series (more reliable)
                    art_time = pd.Timestamp(article['timestamp'] if 'timestamp' in article.index else datetime.utcnow())
                    # FIX: Use the current candle's timestamp, not undefined self.df/self.indices
                    candle_time = pd.Timestamp(self.candle_df.iloc[candle_idx]['timestamp'])
                    age_hours = (candle_time - art_time).total_seconds() / 3600
                    recency = max(0.0, 1.0 - (age_hours / 168))  # 1-week decay
                except:
                    recency = 0.5

                meta[i, 0] = float(recency)

                # Metadata 1: Length normalization (content length / max_len)
                try:
                    content_len = len(str(article['content'] if 'content' in article.index else ''))
                except:
                    content_len = 0
                length_norm = min(1.0, content_len / 500.0)  # Normalize by 500 chars
                meta[i, 1] = float(length_norm)

                # Metadata 2: Source credibility (based on source name)
                try:
                    source = str(article['source'] if 'source' in article.index else 'Unknown').lower()
                except:
                    source = 'Unknown'
                source_cred_map = {
                    'bitcoin magazine': 0.95, 'cointelegraph': 0.90,
                    'decrypt': 0.85, 'reddit': 0.60, 'rss': 0.70,
                    'coingecko': 0.80, 'newsbtc': 0.75
                }
                source_cred = next((v for k, v in source_cred_map.items() if k in source), 0.5)
                meta[i, 2] = float(source_cred)

                # Metadata 3: Novelty (temporal recency: how fresh the article is)
                # ✅ FIX #9: Changed from position-based (1.0 - i/K) to temporal recency
                # Temporal decay: 1.0 for article published at candle time, 0.0 after 24 hours
                try:
                    article_time = pd.Timestamp(article.get('timestamp', datetime.utcnow()))
                    candle_time = pd.Timestamp(self.candle_df.iloc[candle_idx]['timestamp'])
                    age_hours = (candle_time - article_time).total_seconds() / 3600
                    novelty = max(0.0, 1.0 - (age_hours / 24))  # Decay over 24 hours
                except:
                    novelty = 0.5
                meta[i, 3] = float(novelty)

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
        # Balanced thresholds: ~33% each class
        if ret > 0.002:      # +0.2% threshold
            return 2  # UP
        elif ret < -0.002:   # -0.2% threshold
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