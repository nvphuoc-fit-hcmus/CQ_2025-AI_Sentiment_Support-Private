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


# Load factor keywords from the single source of truth (factor_ontology.json).
# Previously this module hard-coded a shorter keyword list that drifted from the
# JSON authoritative version used by safe_alert_net, causing the fallback factor
# label builder (_build_factor_labels, below) to use stale keywords when
# precomputed .npy labels were unavailable.
_ONTOLOGY_PATH = Path(__file__).parent.parent / "factor_ontology.json"
if not _ONTOLOGY_PATH.exists():
    raise FileNotFoundError(
        f"factor_ontology.json missing at {_ONTOLOGY_PATH}. "
        "This file is the single source of truth for FACTOR_KEYWORDS; "
        "precompute scripts and dataset fallback must load from it."
    )
with open(_ONTOLOGY_PATH, "r", encoding="utf-8") as _f:
    _ONTOLOGY = json.load(_f)
FACTOR_NAMES: list[str] = list(_ONTOLOGY["FACTOR_NAMES"])
FACTOR_KEYWORDS: dict[str, list[str]] = dict(_ONTOLOGY["FACTOR_KEYWORDS"])


# Paper Section 3.3.1 "domain expertise / established sources" — whitelist of
# recognized crypto/finance journalism outlets. Tier 1 = flagship domain-expert
# publications; Tier 2 = active crypto-native sites with moderate authority.
# Matching is case-insensitive substring so raw values like "CoinDesk",
# "coindesk.com", or "feeds.coindesk.com/rss" all resolve correctly.
SOURCE_TIER1 = (
    "coindesk", "cointelegraph", "theblock", "the block",
    "decrypt", "bloomberg", "reuters", "wsj", "wall street journal",
    "financial times", "ft.com", "cnbc", "bitcoinmagazine", "forbes",
    # Session 26 — data-driven additions after inspecting articles_max.csv:
    # blockworks covers 5.6 % of the corpus and is a dedicated crypto-native
    # publication with institutional coverage (Blockworks Research, Fed
    # reporting). Tier-1 matches its editorial depth.
    "blockworks",
)
SOURCE_TIER2 = (
    "cryptoslate", "cryptonews", "newsbtc", "cryptopotato",
    "u.today", "utoday", "beincrypto", "bein crypto",
    "coinmarketcap", "cryptobriefing", "ambcrypto",
    "cryptodaily", "coingape", "bitcoinist", "trustnodes",
    # Session 26 — cryptopanic is an aggregator (republishes others) rather
    # than a primary source; it covers 21.1 % of the corpus. Placing it in
    # tier-2 gives its content credibility weight without conflating it
    # with flagship newsrooms that do original reporting.
    "cryptopanic",
)
SOURCE_TIER1_SCORE = 0.9
SOURCE_TIER2_SCORE = 0.7
SOURCE_UNKNOWN_CAP = 0.6  # unknown sources capped below established tiers


def apply_market_normalization(
    features: torch.Tensor,
    mean: Optional[torch.Tensor],
    std: Optional[torch.Tensor],
    clip: float = 8.0,
) -> torch.Tensor:
    """Standalone version of Dataset._normalize_market_features.

    Use this in live_infer/backtest paths where no Dataset instance is
    available — load (mean, std, clip) from a checkpoint and call this
    before feeding the tensor to the model. Training path still goes
    through Dataset._normalize_market_features for locality; the two
    MUST stay in lockstep (same math) so a distribution mismatch can
    never silently reappear.
    """
    features = torch.nan_to_num(features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if mean is not None and std is not None:
        mean_t = torch.as_tensor(mean, dtype=features.dtype, device=features.device)
        std_t  = torch.as_tensor(std,  dtype=features.dtype, device=features.device)
        features = (features - mean_t) / std_t
        features = torch.clamp(features, min=-float(clip), max=float(clip))
    return features


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
        article_embeddings: Optional[np.ndarray] = None,    # (N_articles, D) FinBERT
        article_meta: Optional[pd.DataFrame] = None,        # article metadata (timestamp, source, length, etc)
        article_to_candle: Optional[Dict] = None,           # {article_id → candle_idx} mapping
        symbol: str = "BTCUSDT",
        horizon: str = "1h",
        lookback_hours: int = 24,
        articles_per_candle: int = 8,
        min_articles: int = 0,
        precomputed_features: Optional[np.ndarray] = None,  # (52542, 63) precomputed market features
        factor_labels: Optional[np.ndarray] = None,          # (N_articles, 10) precomputed factor probs
        entity_sentiment: Optional[np.ndarray] = None,       # (N_articles, 10) target-based FSA scores [-1,+1]
        min_factor_confidence: float = 0.6,
        ingest_delay_minutes: int = 15,
        embeddings: Optional[np.ndarray] = None,
        emb_dim: Optional[int] = None,
        strict_features: bool = True,   # Session 23 P0 #2: fail if precomputed missing (silent zero-padding hurts metrics)
        # Session 23 P0 #3: per-timeframe bar sequences from
        # precompute_market_bars.py. If provided, __getitem__ emits
        # ``market_bars`` key with shape (5, L_δ, d_δ) per sample so the
        # upgraded MultiTimescaleMarketEncoder(use_bar_sequences=True) can
        # consume paper-faithful Eq.16 input. None → scalar-only legacy path
        # (kept for backward compatibility and ablation 'w/o_bar_sequences').
        market_bars: Optional[dict] = None,
        # Session 24 — paper Section 4.1.3 TF-IDF novelty from
        # precompute_novelty.py. Shape (N_articles,) float32 in [0, 1].
        # When None, dataset falls back to rank_norm (deviation D4).
        article_novelty: Optional[np.ndarray] = None,
        # Sprint 4 Phase 8.5 — optional override of paper ε_h. None →
        # paper-canonical (1h: 0.002). When set, MUST also be passed to
        # MultiObjectiveLoss to keep label generator and Lret_sign mask
        # synchronized; train_safe_alert.py threads the same value to
        # both sides via the YAML key ``epsilon_h_override``.
        epsilon_h_override: Optional[float] = None,
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
            ingest_delay_minutes: simulated crawler→ML pipeline ingest delay (default 15).
                           Paper Section 4.1.3 requires "timestamp ingest" but the
                           dataset exposes only ``published_at``. Requiring
                           published_at + delay <= candle_close when filtering
                           articles replicates the paper's temporal-consistency
                           constraint. Set to 0 to disable the simulated delay.
            embeddings: backward-compatible alias for ``article_embeddings``.
            emb_dim: optional expected embedding width for legacy callers/tests.
        """
        # Backward compatibility for older call sites:
        #   SAFEAlertDataset(candle_df, article_meta, embeddings, ...)
        #   SAFEAlertDataset(..., embeddings=embeddings, ...)
        if isinstance(article_embeddings, pd.DataFrame):
            legacy_article_meta = article_embeddings
            legacy_embeddings = article_meta
            article_meta = legacy_article_meta
            article_embeddings = legacy_embeddings
        if article_embeddings is None:
            article_embeddings = embeddings
        if article_meta is None:
            raise TypeError("SAFEAlertDataset requires article_meta.")
        if article_embeddings is None:
            raise TypeError("SAFEAlertDataset requires article_embeddings (or embeddings alias).")
        if article_to_candle is None:
            article_to_candle = {}

        self.candle_df = candle_df.reset_index(drop=True)

        # Horizon duration requested by the experiment. The candle step count
        # is derived after timestamp interval inference below.
        _HORIZON_DURATIONS = {
            "15m": pd.Timedelta(minutes=15),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "24h": pd.Timedelta(hours=24),
        }
        if horizon not in _HORIZON_DURATIONS:
            raise ValueError(
                f"Unknown horizon {horizon!r}. Expected one of {list(_HORIZON_DURATIONS)}. "
                "Silent default to 1-step + 0.002 ε_h (previous behaviour) corrupted "
                "direction labels when callers mistyped the horizon (e.g. '2h')."
            )
        self.horizon_duration = _HORIZON_DURATIONS[horizon]
        # Provisional only; recomputed from observed candle interval below.
        self.horizon_steps = 1

        # Neutral-band threshold ε_h (PDF Eq.51): horizon-dependent to avoid labelling
        # small noise moves as directional.  Wider bands for longer horizons because
        # random-walk variance grows with √h.  Sprint 4 Phase 8.5 — single source of
        # truth via constants.get_epsilon_h so dataset (label generator) and loss
        # (Lret_sign mask) cannot drift apart on samples in the override band.
        from constants import get_epsilon_h
        self.epsilon_h = get_epsilon_h(horizon, epsilon_h_override)
        if epsilon_h_override is not None:
            print(
                f"[OK] SAFEAlertDataset: ε_h override active — "
                f"using {self.epsilon_h} for horizon={horizon} "
                f"(paper canonical = {get_epsilon_h(horizon)})"
            )
        # ✅ FIX #12: L2 normalize embeddings for scale consistency
        embeddings = torch.from_numpy(article_embeddings).float()
        if emb_dim is not None and embeddings.ndim == 2 and embeddings.shape[1] != int(emb_dim):
            raise ValueError(
                f"SAFEAlertDataset: embeddings dim {embeddings.shape[1]} does not match emb_dim={emb_dim}."
            )
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
        # Ingest delay (paper Section 4.1.3 "temporal consistency"):
        # the raw CSV only has ``published_at`` but production systems do NOT
        # see an article the moment it is published — a crawler/ingest
        # pipeline introduces a delay (typically 5-30 min). Simulating this
        # delay by requiring published_at + delay <= candle_close prevents the
        # optimistic leak where the model is trained on articles that would
        # not yet be visible at inference. Set to 0 to disable the simulated
        # delay.
        self.ingest_delay = pd.Timedelta(minutes=int(ingest_delay_minutes))
        self.market_feature_mean: Optional[torch.Tensor] = None
        self.market_feature_std: Optional[torch.Tensor] = None
        self.market_feature_clip: float = 8.0
        self.market_bar_mean: Optional[torch.Tensor] = None
        self.market_bar_std: Optional[torch.Tensor] = None
        self.market_bar_clip: float = 8.0

        # Store strict_features flag so __getitem__ can honour it
        self.strict_features = bool(strict_features)

        # Session 23 P0 #3: bind bar-sequence arrays if supplied. Expected
        # keys are ``bars_{1m,5m,15m,1h,4h}``, each shape (N_candles, L, d).
        # Validation:
        #   - all 5 timeframes must be present (paper Eq.16 lists 5 TFs)
        #   - N_candles must match len(candle_df) (positional alignment)
        #   - L and d must be consistent across all TFs (simpler API)
        #   - no NaN / Inf (hard fail — indicates stale precompute)
        self._bar_tf_order = ["1m", "5m", "15m", "1h", "4h"]  # must match model N_TIMEFRAMES order
        self.market_bars = None
        if market_bars is not None:
            required = {f"bars_{tf}" for tf in self._bar_tf_order}
            missing = required - set(market_bars.keys())
            if missing:
                raise ValueError(
                    f"SAFEAlertDataset: market_bars missing timeframes {sorted(missing)}. "
                    f"Re-run precompute_market_bars.py to include all 5 TFs."
                )
            # Shape consistency
            first = market_bars[f"bars_{self._bar_tf_order[0]}"]
            if first.ndim != 3:
                raise ValueError(
                    f"market_bars[bars_{self._bar_tf_order[0]}] must be 3D "
                    f"(N, L, d); got shape {first.shape}"
                )
            N_expected, L_expected, d_expected = first.shape
            if N_expected != len(self.candle_df):
                raise ValueError(
                    f"market_bars N={N_expected} != len(candle_df)={len(self.candle_df)}. "
                    f"Re-run precompute_market_bars.py against the current horizon CSV."
                )
            for tf in self._bar_tf_order:
                arr = market_bars[f"bars_{tf}"]
                if arr.shape != (N_expected, L_expected, d_expected):
                    raise ValueError(
                        f"market_bars[bars_{tf}] shape {arr.shape} != "
                        f"({N_expected}, {L_expected}, {d_expected}); all TFs must match."
                    )
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"market_bars[bars_{tf}] contains NaN/Inf — stale precompute?"
                    )
            # Pre-stack into (N, 5, L, d) for fast __getitem__ slicing.
            self.market_bars = np.stack(
                [market_bars[f"bars_{tf}"] for tf in self._bar_tf_order],
                axis=1,
            ).astype(np.float32)
            print(
                f"[OK] SAFEAlertDataset: market_bars loaded, shape={self.market_bars.shape} "
                f"(N_candles, n_TFs, L_δ, d_δ) — paper Eq.16 mode",
                flush=True,
            )

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
        #
        # Session 23 P1 #4 fix: shape mismatch now fails fast instead of
        # silently falling back to keyword matching. Previous warn-and-
        # fallback behaviour hid a stealth bug: when factor_labels.npy was
        # generated for N_old articles but the dataset now has N_new > N_old
        # (new articles crawled since the last precompute), the warning only
        # appeared once and the fallback path quietly produced lower-quality
        # keyword labels for ALL articles in that run. Lfac silently
        # degraded. Failing fast forces the user to either:
        #   (a) re-run precompute_factor_labels.py for the current corpus, or
        #   (b) pass factor_labels=None explicitly to opt into the fallback.
        if factor_labels is not None:
            expected_shape = (len(article_meta), len(FACTOR_NAMES))
            if factor_labels.shape != expected_shape:
                raise ValueError(
                    f"SAFEAlertDataset: factor_labels shape {factor_labels.shape} "
                    f"!= expected {expected_shape}. This usually means the "
                    f".npy file is stale (articles added since last precompute). "
                    f"Re-run precompute_factor_labels.py, OR pass factor_labels=None "
                    f"to explicitly use the keyword fallback (lower quality)."
                )
            self.precomputed_factor_labels = factor_labels.astype(np.float32)
            self.article_max_prob = self.precomputed_factor_labels.max(axis=1)
            # Session 26 — per-factor class weights for Lfac.
            # Corpus inspection found severe imbalance: macro=32.8 %,
            # network_outage=0.6 % (55× gap). Without class weights the
            # network factor receives near-zero gradient and is effectively
            # dead — w/o_factor ablation would be identical to "with factor"
            # for the rare classes. Sqrt-softened inverse-frequency weighting
            # (same recipe as Ldir class_weights) restores a learnable
            # signal for the minority factors.
            top1_factor = self.precomputed_factor_labels.argmax(axis=1)
            counts = np.bincount(top1_factor, minlength=len(FACTOR_NAMES)).astype(np.float64)
            # Floor at 1 to avoid divide-by-zero for factors that never appear as top1.
            freqs = np.clip(counts / max(counts.sum(), 1), 1e-4, None)
            raw_w = 1.0 / np.sqrt(freqs)                  # inverse-sqrt-freq
            raw_w = raw_w / raw_w.mean()                  # normalize: mean(w) = 1
            self.factor_class_weights = raw_w.astype(np.float32)  # (C,) numpy
            _w_str = ", ".join(
                f"{FACTOR_NAMES[i]}={raw_w[i]:.2f}" for i in range(len(FACTOR_NAMES))
            )
            print(f"[OK] SAFEAlertDataset: using precomputed factor labels {factor_labels.shape} "
                  f"(fixes Lfac stuck at log(10)=2.303)", flush=True)
            print(f"[OK] SAFEAlertDataset: factor class weights (sqrt-softened balanced, mean=1): "
                  f"{_w_str}", flush=True)
        else:
            self.precomputed_factor_labels = None
            self.article_max_prob = None
            self.factor_class_weights = None

        # Load target-based FSA entity sentiment scores
        # Shape: (N_articles, 10) float32 in [-1, +1], one score per factor per article.
        # This enables fine-grained sentiment analysis per financial entity/factor.
        #
        # Alignment contract: entity_sentiment[i] MUST correspond to article_meta.iloc[i]
        # — this is a positional alignment, not an index-based one. A misaligned file
        # silently degrades META effective dimensionality from 14 → 4 (drops 10 FSA
        # features) because the __getitem__ fallback zero-fills rows whose global index
        # is out of bounds. We now fail fast on shape mismatch, and count runtime
        # out-of-range hits in __getitem__ so users spot alignment drift early.
        self._entity_sentiment_oob_count = 0
        self._entity_sentiment_warned = False
        if entity_sentiment is not None:
            expected_shape = (len(article_meta), len(FACTOR_NAMES))
            if entity_sentiment.shape != expected_shape:
                # Raise rather than silently-ignore: the file was provided deliberately,
                # so a shape mismatch almost always indicates a stale/misbuilt artifact
                # that would strip 10 features if we proceeded. Better to crash loudly.
                raise ValueError(
                    f"SAFEAlertDataset: entity_sentiment shape {entity_sentiment.shape} "
                    f"does not match expected {expected_shape} "
                    f"(n_articles={len(article_meta)}, n_factors={len(FACTOR_NAMES)}). "
                    "Re-run precompute_entity_sentiment.py against the current "
                    "articles_max.csv to rebuild the aligned artifact."
                )
            if not np.isfinite(entity_sentiment).all():
                raise ValueError(
                    "SAFEAlertDataset: entity_sentiment contains NaN/Inf. "
                    "Re-run precompute_entity_sentiment.py."
                )
            self.entity_sentiment = entity_sentiment.astype(np.float32)
            print(f"[OK] SAFEAlertDataset: using target-based FSA entity sentiment {entity_sentiment.shape}", flush=True)
        else:
            self.entity_sentiment = None

        # Session 24 — paper Section 4.1.3 TF-IDF novelty. Shape (N_articles,)
        # float32 in [0, 1]. None → fallback to rank_norm (deviation D4).
        if article_novelty is not None:
            if article_novelty.ndim != 1 or len(article_novelty) != len(article_meta):
                raise ValueError(
                    f"article_novelty shape {article_novelty.shape} does not "
                    f"match expected ({len(article_meta)},). Re-run "
                    f"precompute_novelty.py against the current articles CSV."
                )
            if not np.isfinite(article_novelty).all():
                raise ValueError("article_novelty contains NaN/Inf.")
            self.article_novelty = article_novelty.astype(np.float32)
            print(
                f"[OK] SAFEAlertDataset: TF-IDF novelty loaded, shape={article_novelty.shape} "
                f"(mean={float(article_novelty.mean()):.3f}) — paper Section 4.1.3",
                flush=True,
            )
        else:
            self.article_novelty = None
            print(
                "[INFO] SAFEAlertDataset: no precomputed novelty — falling back "
                "to rank_norm for META[3] (deviation D4). Run precompute_novelty.py "
                "to enable paper-faithful TF-IDF novelty.",
                flush=True,
            )

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

        # Session 25 — paper Section 4.1.4 market regime (stratification
        # variable, not a model input). Precompute a coarse 3-class label
        # per candle so downstream analysis can stratify metrics by regime
        # ("does the model lose more in bear vs bull?") without re-reading
        # OHLCV. Uses a simple rule grounded in practice:
        #   BULL    — trailing 30-bar return > +3%   AND vol in lower 66%
        #   BEAR    — trailing 30-bar return < −3%   AND vol in lower 66%
        #   VOLATILE — trailing 30-bar vol in top 33%
        #   SIDEWAYS — everything else
        # The paper does not mandate a specific regime definition; this rule
        # is transparent and deterministic. Regime is emitted as a string key
        # in each sample so it's easy to group-by in analysis notebooks.
        self._regime_labels = self._compute_regime_labels()

        # P1 #10: Fail-fast if the candle sampling interval does not match the
        # requested horizon. The exact step count is derived below from the
        # observed candle interval, preventing 4h-native candles from being
        # mislabeled as 16h-ahead samples.
        self.candle_interval = self.horizon_duration
        try:
            sorted_ts = self.candle_df["timestamp"].dropna().sort_values().drop_duplicates()
            if len(sorted_ts) >= 3:
                # Median of consecutive diffs — robust to occasional gaps.
                diff_sec = float(sorted_ts.diff().dropna().median().total_seconds())
                observed_min = diff_sec / 60.0
                if diff_sec <= 0:
                    raise ValueError(
                        f"SAFEAlertDataset: non-positive candle interval detected ({observed_min:.1f} min)."
                    )
                self.candle_interval = pd.Timedelta(seconds=diff_sec)
        except (TypeError, AttributeError, KeyError):
            # Timestamp column may be partially missing for some downstream
            # callers (backtest with empty article_to_candle etc.). Don't
            # block construction — the error will surface clearly later.
            pass
        ratio = self.horizon_duration / self.candle_interval
        horizon_steps_float = float(ratio)
        horizon_steps_round = int(round(horizon_steps_float))
        if horizon_steps_round < 1 or abs(horizon_steps_float - horizon_steps_round) > 1e-3:
            raise ValueError(
                f"SAFEAlertDataset: horizon='{horizon}' ({self.horizon_duration}) is not an "
                f"integer multiple of candle interval {self.candle_interval} "
                f"(ratio={horizon_steps_float:.4f}). Cannot build exact paper Eq.50 labels."
            )
        self.horizon_steps = horizon_steps_round
        self.candle_interval_np = np.timedelta64(int(self.candle_interval.value), "ns")
        print(
            f"[OK] SAFEAlertDataset: news cutoff uses candle close "
            f"(timestamp + {self.candle_interval.total_seconds()/60.0:.1f}m) "
            f"with ingest_delay={self.ingest_delay}; horizon={horizon} "
            f"=> steps_ahead={self.horizon_steps}",
            flush=True,
        )

        # Session 26 — paper §4.1.3 quality filter. MUST run BEFORE the
        # candle-article mapping so filtered articles never enter any
        # candle window. Mask is per-row-index over the ORIGINAL article
        # corpus (so precomputed article_embeddings / factor_labels /
        # entity_sentiment indexing by art_idx stays valid — we simply
        # never reference the filtered indices).
        self.article_quality_mask = self._compute_article_quality_mask()

        # Build reverse mapping: candle_idx → list of article indices
        self._build_candle_article_map()

        # Pre-epoch alignment checks — fail fast instead of lazy-warn after batches fire.
        self._validate_entity_sentiment_coverage()
        self._validate_factor_label_coverage()   # P1 #17
        self._validate_source_coverage()

        # Precompute valid sample indices (fast lookup)
        # Must leave horizon_steps candles at the end for label access (PDF Eq.50)
        max_idx = len(self.candle_df) - 1 - self.horizon_steps
        self.valid_idx = []
        for i in range(max_idx + 1):
            if len(self.candle_to_articles.get(i, [])) >= min_articles:
                self.valid_idx.append(i)

        print(f"[OK] SAFEAlertDataset: {len(self.valid_idx)}/{len(candle_df)} candles with articles", flush=True)

        # P2 #25: One-shot summary so the reviewer sees data dimensions without
        # grepping logs. Runs once after construction; cheap because all stats
        # are already computed.
        self._log_dataset_summary()

    def _log_dataset_summary(self) -> None:
        """Print a compact dataset summary (P2 #25). Called once from __init__."""
        try:
            ts_min = pd.to_datetime(self.candle_df["timestamp"].min())
            ts_max = pd.to_datetime(self.candle_df["timestamp"].max())
            # Article-count distribution per candle (only candles with articles).
            art_counts = np.array(
                [len(v) for v in self.candle_to_articles.values() if len(v) > 0],
                dtype=np.int64,
            )
            if art_counts.size == 0:
                art_stats = "0 (no article windows)"
            else:
                art_stats = (f"avg={art_counts.mean():.1f}, "
                             f"min={art_counts.min()}, "
                             f"max={art_counts.max()}, "
                             f"median={int(np.median(art_counts))}")

            # Sample-level direction class balance (cheap — reuses epsilon_h).
            # Sprint 7 — cache labels in self._direction_labels indexed by
            # position-in-valid_idx (= dataset __getitem__ index). Used by
            # BalancedBatchSampler to build class-balanced batches without
            # invoking __getitem__ (which loads articles/embeddings/bars).
            # Sentinel -1 marks samples where label computation failed.
            self._direction_labels = np.full(len(self.valid_idx), -1, dtype=np.int8)
            dir_counts = np.zeros(3, dtype=np.int64)
            for j, i in enumerate(self.valid_idx):
                try:
                    curr  = self.candle_df.iloc[i]
                    label = self.candle_df.iloc[i + self.horizon_steps]
                    exec_ = self.candle_df.iloc[i + 1]
                    d = self._get_direction_label(curr, label, exec_)
                    self._direction_labels[j] = d
                    dir_counts[d] += 1
                except Exception:
                    continue
            total = max(int(dir_counts.sum()), 1)

            print("=" * 78)
            print("[DATA SUMMARY]")
            print(f"  Horizon:           {self.horizon} (epsilon_h={self.epsilon_h}, "
                  f"steps_ahead={self.horizon_steps})")
            print(f"  Candles:           {len(self.candle_df)} "
                  f"from {ts_min} -> {ts_max}")
            print(f"  Articles corpus:   {len(self.article_meta)} "
                  f"(embedding dim={self.article_embeddings.shape[-1]})")
            print(f"  Valid samples:     {len(self.valid_idx)} "
                  f"(>= {self.min_articles} articles/candle)")
            print(f"  Articles/candle:   {art_stats}")
            print(f"  Class balance:     DOWN={dir_counts[0]/total:5.1%}  "
                  f"NEUTRAL={dir_counts[1]/total:5.1%}  "
                  f"UP={dir_counts[2]/total:5.1%}  (n={total})")
            print(f"  Factor ontology:   {len(FACTOR_NAMES)} classes")
            print(f"  Entity sentiment:  "
                  f"{'enabled (14-dim META)' if self.entity_sentiment is not None else 'disabled (4-dim META)'}")
            print(f"  Factor labels:     "
                  f"{'precomputed' if self.precomputed_factor_labels is not None else 'on-the-fly keyword'}")
            print("=" * 78, flush=True)
        except Exception as exc:
            print(f"[WARN] Dataset summary failed: {exc}", flush=True)

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

        if self.market_bars is not None:
            self.fit_market_bar_scaler(dataset_indices, clip_value=clip_value)

    def fit_market_bar_scaler(self, dataset_indices: List[int] | range, clip_value: float = 8.0) -> None:
        """Fit train-only z-score normalization for per-timeframe bar tensors.

        The cached bar features store OHLC as absolute log prices. In a
        walk-forward split those levels drift by several train standard
        deviations as BTC reprices over years, which makes the sequence encoder
        learn timestamp/regime level instead of local candle shape. Before
        fitting stats, convert the first four OHLC log columns into values
        relative to the latest close in each sample/timeframe sequence.
        """
        self.market_bar_clip = float(clip_value)

        if self.market_bars is None or len(dataset_indices) == 0:
            self.market_bar_mean = None
            self.market_bar_std = None
            return

        candle_indices = np.asarray([self.valid_idx[int(i)] for i in dataset_indices], dtype=np.int64)
        train_bars = torch.from_numpy(self.market_bars[candle_indices]).float()
        train_bars, _ = self._stationarize_market_bars(train_bars)
        train_bars = torch.nan_to_num(train_bars, nan=0.0, posinf=1e4, neginf=-1e4)

        mean = train_bars.mean(dim=(0, 2), keepdim=False).unsqueeze(1)  # (5, 1, d)
        std = train_bars.std(dim=(0, 2), unbiased=False, keepdim=False).unsqueeze(1)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)

        self.market_bar_mean = mean.float()
        self.market_bar_std = std.float()
        print(
            "[OK] SAFEAlertDataset: market_bar scaler fit "
            f"(train-only, relative OHLC, shape={tuple(self.market_bar_mean.shape)}, "
            f"clip={self.market_bar_clip:g})",
            flush=True,
        )

    def _stationarize_market_bars(self, market_bars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert absolute OHLC log levels to relative levels per sample/TF."""
        bars = torch.nan_to_num(market_bars.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        zero_row = bars.abs().sum(dim=-1, keepdim=True) == 0
        if bars.shape[-1] >= 4:
            bars = bars.clone()
            last_close_log = bars[..., -1:, 3:4]
            bars[..., :4] = bars[..., :4] - last_close_log
            bars = torch.where(zero_row, torch.zeros_like(bars), bars)
        return bars, zero_row

    def _normalize_market_bars(self, market_bars: torch.Tensor) -> torch.Tensor:
        market_bars, zero_row = self._stationarize_market_bars(market_bars)
        if self.market_bar_mean is not None and self.market_bar_std is not None:
            mean = self.market_bar_mean.to(device=market_bars.device, dtype=market_bars.dtype)
            std = self.market_bar_std.to(device=market_bars.device, dtype=market_bars.dtype)
            market_bars = (market_bars - mean) / std
            market_bars = torch.clamp(
                market_bars,
                min=-self.market_bar_clip,
                max=self.market_bar_clip,
            )
            market_bars = torch.where(zero_row, torch.zeros_like(market_bars), market_bars)
        return market_bars

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

    # ── Preprocessing state persistence (for production inference) ────────────
    def get_preprocessing_state(self) -> Dict[str, Optional[torch.Tensor]]:
        """Export fitted preprocessing state for checkpoint persistence.

        Without this, live_infer and backtest receive RAW features while training
        uses z-score normalized ones — a silent distribution mismatch that makes
        production predictions garbage. Callers (trainer) snapshot this dict into
        the checkpoint so downstream loaders can rehydrate the exact normalization.
        """
        return {
            "market_feature_mean": None if self.market_feature_mean is None
                                   else self.market_feature_mean.detach().cpu().clone(),
            "market_feature_std":  None if self.market_feature_std is None
                                   else self.market_feature_std.detach().cpu().clone(),
            "market_feature_clip": float(self.market_feature_clip),
            "market_bar_mean": None if self.market_bar_mean is None
                               else self.market_bar_mean.detach().cpu().clone(),
            "market_bar_std":  None if self.market_bar_std is None
                               else self.market_bar_std.detach().cpu().clone(),
            "market_bar_clip": float(self.market_bar_clip),
        }

    def load_preprocessing_state(self, state: Dict) -> None:
        """Rehydrate preprocessing state (e.g. from a training checkpoint)."""
        if state is None:
            return
        mean = state.get("market_feature_mean")
        std  = state.get("market_feature_std")
        clip = state.get("market_feature_clip", 8.0)
        self.market_feature_mean = None if mean is None else torch.as_tensor(mean).float()
        self.market_feature_std  = None if std  is None else torch.as_tensor(std).float()
        self.market_feature_clip = float(clip)
        bar_mean = state.get("market_bar_mean")
        bar_std = state.get("market_bar_std")
        bar_clip = state.get("market_bar_clip", 8.0)
        self.market_bar_mean = None if bar_mean is None else torch.as_tensor(bar_mean).float()
        self.market_bar_std = None if bar_std is None else torch.as_tensor(bar_std).float()
        self.market_bar_clip = float(bar_clip)

    def _compute_regime_labels(self) -> np.ndarray:
        """Session 25 — paper Section 4.1.4 market regime classifier.

        Returns (N_candles,) int8 array: 0=SIDEWAYS, 1=BULL, 2=BEAR,
        3=VOLATILE. Labels are purely backward-looking (rolling 30-bar
        window) so they are safe to emit alongside labels without leak.

        Regime is a STRATIFICATION variable per paper Section 4.1.4 —
        it does NOT feed the model. Downstream analysis groups evaluation
        metrics by regime to answer "where does the model underperform?".
        """
        closes = self.candle_df["close"].to_numpy(dtype=np.float64)
        N = len(closes)
        window = 30  # 30 one-hour bars ≈ 1.25 days — captures short-term state

        # Trailing log-return over window, and rolling std of 1-bar returns.
        # Both computed with a simple forward-only window to guarantee
        # causality (each regime label uses only past prices).
        roll_ret = np.zeros(N, dtype=np.float32)
        roll_vol = np.zeros(N, dtype=np.float32)
        for i in range(N):
            start = max(0, i - window + 1)
            win = closes[start: i + 1]
            if len(win) >= 5:
                roll_ret[i] = float(np.log(max(win[-1], 1e-8) / max(win[0], 1e-8)))
                bar_rets = np.diff(np.log(np.maximum(win, 1e-8)))
                roll_vol[i] = float(np.std(bar_rets)) if len(bar_rets) >= 2 else 0.0

        # Volatility band: top-33% of the full-history distribution.
        vol_threshold = float(np.quantile(roll_vol[roll_vol > 0], 0.66)) if (roll_vol > 0).any() else 0.0

        labels = np.zeros(N, dtype=np.int8)   # default SIDEWAYS
        bull_mask   = (roll_ret > 0.03) & (roll_vol <= vol_threshold)
        bear_mask   = (roll_ret < -0.03) & (roll_vol <= vol_threshold)
        volatile_mask = roll_vol > vol_threshold
        labels[bull_mask] = 1
        labels[bear_mask] = 2
        labels[volatile_mask] = 3   # volatile overrides bull/bear by design

        # Distribution summary for log
        unique, counts = np.unique(labels, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique, counts)}
        print(
            f"[OK] SAFEAlertDataset: market regime labels computed "
            f"(SIDEWAYS={dist.get(0, 0)}, BULL={dist.get(1, 0)}, "
            f"BEAR={dist.get(2, 0)}, VOLATILE={dist.get(3, 0)}) — paper Section 4.1.4",
            flush=True,
        )
        return labels

    def _build_source_cred_map(self) -> dict:
        """Paper Section 3.3.1 source credibility — tiered domain-expert whitelist.

        Tier 1 (flagship crypto/finance journalism → 0.9):  CoinDesk,
        Cointelegraph, The Block, Decrypt, Bloomberg, Reuters, WSJ, FT, CNBC,
        Bitcoin Magazine, Forbes.

        Tier 2 (active crypto-native sites, moderate authority → 0.7):
        CryptoSlate, CryptoNews, NewsBTC, CryptoPotato, U.Today, BeInCrypto,
        CoinMarketCap, CryptoBriefing, AMBCrypto, etc.

        Unknown sources: fall back to frequency + length heuristic, capped at
        SOURCE_UNKNOWN_CAP (0.6) so established outlets always dominate.

        Returns dict: source_name_lower → float in [0, 0.9].
        Unknown sources fall back to 0.3 at lookup time (below the cap).
        """
        if 'source' not in self.article_meta.columns:
            return {}

        src_col = self.article_meta['source'].dropna().str.lower().str.strip()
        counts = src_col.value_counts()
        if counts.empty:
            return {}

        def _tier_lookup(src_name: str) -> Optional[float]:
            for needle in SOURCE_TIER1:
                if needle in src_name:
                    return SOURCE_TIER1_SCORE
            for needle in SOURCE_TIER2:
                if needle in src_name:
                    return SOURCE_TIER2_SCORE
            return None

        # Heuristic fallback for unknown sources
        log_max = float(np.log1p(counts.iloc[0]))
        freq_scores = (
            {src: float(np.log1p(cnt) / log_max) for src, cnt in counts.items()}
            if log_max > 0 else {src: 0.0 for src in counts.index}
        )
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
        n_t1 = n_t2 = n_unk = 0
        for src in freq_scores:
            tier_score = _tier_lookup(src)
            if tier_score is not None:
                cred_map[src] = tier_score
                if tier_score == SOURCE_TIER1_SCORE:
                    n_t1 += 1
                else:
                    n_t2 += 1
            else:
                fs = freq_scores[src]
                ls = length_scores.get(src, fs)
                heuristic = 0.6 * fs + 0.4 * ls
                cred_map[src] = float(min(heuristic, SOURCE_UNKNOWN_CAP))
                n_unk += 1

        print(
            f"[OK] SAFEAlertDataset: source_cred built from {len(cred_map)} sources "
            f"(tier1={n_t1}, tier2={n_t2}, unknown_heuristic={n_unk}, "
            f"range=[{min(cred_map.values()):.2f},{max(cred_map.values()):.2f}]) — paper Section 3.3.1",
            flush=True,
        )
        return cred_map

    # Paper §4.1.3 runtime quality filter — applied BEFORE candle-article
    # mapping so filtered articles never appear in any candle window. Data
    # inspection (90 346 BTC articles) found ~2.6 % of rows are low-quality:
    # empty-title crawler failures, HTTP error pages, UI placeholders
    # ("x icon"), and ~2.4 % exact-title duplicates. A semantic near-dup
    # detector (TF-IDF / MinHash) would cost a full corpus sort for <0.4 %
    # additional catches; we therefore skip it. Upstream crawler-service
    # catches the long tail (see D15 / D18).
    _QUALITY_MIN_TITLE_CHARS   = 10      # drop empty titles + "x icon" (~1.5 %)
    _QUALITY_MIN_CONTENT_CHARS = 100     # 1.5 % of BTC corpus — safe floor
    _QUALITY_BAD_TITLE_PATTERNS = (
        "error 500", "server error", "that’s an error", "that's an error",
        "page not found", "404 not found", "access denied",
    )

    def _compute_article_quality_mask(self) -> np.ndarray:
        """Return (N_articles,) bool mask — True = keep this article.

        Four filters applied (conjunctive):
          1. Title ≥ _QUALITY_MIN_TITLE_CHARS non-whitespace chars
          2. Content ≥ _QUALITY_MIN_CONTENT_CHARS chars
          3. Title does not match any known garbage pattern (error pages etc.)
          4. Exact title+source is kept only on first occurrence — remaining
             duplicates dropped. Ordered by published_at so the earliest copy
             wins (preserves original reporting, drops syndication reposts).

        Per-filter drop counts are logged for reviewer inspection — the
        thesis can reference these numbers as evidence of data hygiene.
        """
        N = len(self.article_meta)
        keep = np.ones(N, dtype=bool)

        title_col = self.article_meta.get("title")
        if title_col is None:
            return keep  # no title column → cannot filter, keep all
        titles = title_col.fillna("").astype(str).str.strip()

        content_col = self.article_meta.get("content")
        contents = (
            content_col.fillna("").astype(str).str.len()
            if content_col is not None
            else pd.Series([self._QUALITY_MIN_CONTENT_CHARS] * N, index=self.article_meta.index)
        )

        # (1) title length
        short_title = titles.str.len() < self._QUALITY_MIN_TITLE_CHARS
        # (2) content length
        short_content = contents < self._QUALITY_MIN_CONTENT_CHARS
        # (3) known bad patterns (case-insensitive contains)
        titles_lower = titles.str.lower()
        bad_pattern = pd.Series(False, index=self.article_meta.index)
        for pat in self._QUALITY_BAD_TITLE_PATTERNS:
            bad_pattern = bad_pattern | titles_lower.str.contains(pat, regex=False, na=False)

        keep &= (~short_title).values
        keep &= (~short_content).values
        keep &= (~bad_pattern).values

        # (4) exact (title, source) dedup — keep earliest published_at.
        # Sort stable by timestamp, mark first occurrence as keep, rest drop.
        src_col = self.article_meta.get("source")
        sources = src_col.fillna("").astype(str).str.lower().str.strip() if src_col is not None else pd.Series([""] * N)
        key = titles.str.lower() + "||" + sources
        if "timestamp" in self.article_meta.columns:
            order = self.article_meta["timestamp"].values.argsort(kind="stable")
        else:
            order = np.arange(N)
        seen: set = set()
        dedup_keep = np.ones(N, dtype=bool)
        key_vals = key.values
        for i in order:
            if not keep[i]:
                continue  # already filtered by 1-3, no need to dedup
            k = key_vals[i]
            if k in seen:
                dedup_keep[i] = False
            else:
                seen.add(k)
        keep &= dedup_keep

        # Raw per-filter counts (not mutually exclusive — an article can fail
        # multiple filters). The exact dedup counter is scoped to articles
        # that would otherwise have been kept, so it reflects "additional
        # drops beyond the hard-quality filters".
        st_arr = short_title.values
        sc_arr = short_content.values
        bp_arr = bad_pattern.values
        survived_hard = (~st_arr) & (~sc_arr) & (~bp_arr)
        n_short_title   = int(st_arr.sum())
        n_short_content = int(sc_arr.sum())
        n_bad_pattern   = int(bp_arr.sum())
        n_dup           = int(((~dedup_keep) & survived_hard).sum())
        n_kept          = int(keep.sum())
        print(
            f"[OK] SAFEAlertDataset: quality filter — "
            f"dropped {N - n_kept}/{N} articles "
            f"(short_title={n_short_title}, short_content={n_short_content}, "
            f"bad_pattern={n_bad_pattern}, exact_title_dup={n_dup}), "
            f"kept {n_kept} ({100.0*n_kept/max(N,1):.1f}%) — paper §4.1.3",
            flush=True,
        )
        return keep

    def _build_candle_article_map(self):
        """Build candle_idx → article indices mapping using fast vectorized ops."""
        self.candle_to_articles = {}

        # Convert to numpy for fast operations
        candle_times = pd.to_datetime(
            self.candle_df["timestamp"], utc=True
        ).to_numpy(dtype="datetime64[ns]")
        article_times = pd.to_datetime(
            self.article_meta["timestamp"], utc=True
        ).to_numpy(dtype="datetime64[ns]")

        # OHLCV CSV timestamps are candle OPEN times. The decision point for
        # both market bars and news is candle CLOSE, so shift the whole
        # article window forward by one candle interval. This keeps the
        # lookback length constant while admitting the most recent news that
        # would have cleared the ingest pipeline by decision time.
        delay_np = np.timedelta64(int(self.ingest_delay.total_seconds()), 's')
        lookback_np = np.timedelta64(int(float(self.lookback_hours) * 3600), 's')

        # For each candle (vectorized)
        for candle_idx in range(len(candle_times)):
            candle_time = candle_times[candle_idx]
            decision_time = candle_time + self.candle_interval_np
            window_start = decision_time - lookback_np

            # Find articles in window.
            # Three constraints for leak-free + quality-controlled training:
            #   (1) article_times >= window_start  — inside the lookback.
            #   (2) article_times + ingest_delay <= decision_time — would have
            #       been visible to the production pipeline by candle close
            #       (paper §4.1.3 "temporal consistency"). Using candle close
            #       here matches precompute_market_bars.py and avoids a
            #       one-candle freshness lag in news windows.
            #   (3) article_quality_mask[i] — paper §4.1.3 step 1+4 filter
            #       (empty-title / short-content / exact-title-dup / garbage
            #       patterns). Filtered indices are excluded from every
            #       candle window, so they never reach the model.
            mask = (
                (article_times >= window_start)
                & (article_times + delay_np <= decision_time)
            )
            if self.article_quality_mask is not None:
                mask = mask & self.article_quality_mask
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

    def _get_decision_timestamp(self, candle_idx: int) -> pd.Timestamp:
        """Return the candle-close decision timestamp for an OHLCV row.

        The candle CSV timestamp is the bar open time. Decisions are made at
        bar close, consistent with precomputed market bars and the paper's
        pseudo-online setting.
        """
        return pd.Timestamp(self.candle_df.iloc[candle_idx]["timestamp"]) + self.candle_interval

    def _validate_entity_sentiment_coverage(self) -> None:
        """Full pre-epoch scan: every art_idx referenced by the candle→articles
        map must fit inside ``self.entity_sentiment``. Fails fast on alignment
        drift — the lazy __getitem__ warning only fires after training starts
        and can be hidden behind thousands of already-completed batches.

        P1 #16: vectorised flatten over all article indices — O(total refs)
        instead of O(n_candles × n_articles) Python loop.
        """
        if self.entity_sentiment is None:
            return
        n_es = len(self.entity_sentiment)
        # Flatten all article references in one pass (vectorised).
        all_refs = np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in self.candle_to_articles.values()]
        ) if self.candle_to_articles else np.empty(0, dtype=np.int64)
        if all_refs.size == 0:
            return
        oob_mask = all_refs >= n_es
        oob_articles = int(oob_mask.sum())
        if oob_articles > 0:
            max_oob_idx = int(all_refs[oob_mask].max())
            # Unique candles containing at least one OOB ref — approximate via set.
            oob_candles = sum(
                1 for v in self.candle_to_articles.values()
                if any(a >= n_es for a in v)
            )
            raise ValueError(
                f"SAFEAlertDataset: entity_sentiment covers {n_es} articles but "
                f"{oob_articles} references (across {oob_candles} candles) exceed that, "
                f"max_oob_idx={max_oob_idx}. articles_max.csv likely grew past the "
                f"precomputed FSA file. Re-run precompute_entity_sentiment.py."
            )
        print(f"[OK] SAFEAlertDataset: entity_sentiment coverage validated "
              f"({n_es} articles, 0 OOB references).", flush=True)

    def _validate_factor_label_coverage(self) -> None:
        """P1 #17: symmetric check for article_factor_labels.npy alignment.

        When factor labels are precomputed but the article corpus has grown
        past the .npy file, the dataset silently falls back to on-the-fly
        keyword matching for late articles → Lfac degrades. Fail-fast mirrors
        the entity_sentiment check so both precomputed artefacts stay in sync.
        """
        if self.precomputed_factor_labels is None:
            return
        n_fl = len(self.precomputed_factor_labels)
        if not self.candle_to_articles:
            return
        all_refs = np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in self.candle_to_articles.values()]
        )
        if all_refs.size == 0:
            return
        oob_mask = all_refs >= n_fl
        if oob_mask.any():
            max_oob_idx = int(all_refs[oob_mask].max())
            n_oob = int(oob_mask.sum())
            import warnings
            warnings.warn(
                f"[safe_alert_dataset] factor_labels covers {n_fl} articles but "
                f"{n_oob} references reach {max_oob_idx}. Those articles fall back "
                f"to on-the-fly keyword matching, weakening Lfac. Re-run "
                f"precompute_factor_labels.py to refresh alignment.",
                RuntimeWarning, stacklevel=2,
            )
        else:
            print(f"[OK] SAFEAlertDataset: factor_labels coverage validated "
                  f"({n_fl} articles, 0 OOB references).", flush=True)

    def _validate_source_coverage(self) -> None:
        """Warn if a large fraction of articles have no source (or blank) — they
        would silently use the 0.3 credibility fallback."""
        if 'source' not in self.article_meta.columns:
            return
        src_col = self.article_meta['source'].fillna('').astype(str).str.lower().str.strip()
        n_total = len(src_col)
        if n_total == 0:
            return
        n_blank = int((src_col == '').sum())
        ratio = n_blank / n_total
        if ratio > 0.10:
            import warnings
            warnings.warn(
                f"[safe_alert_dataset] {n_blank}/{n_total} ({ratio:.1%}) articles "
                f"have blank/missing source and will use the 0.3 credibility fallback. "
                f"Consider enriching article_meta['source'] to improve signal.",
                RuntimeWarning, stacklevel=2,
            )
        # Always print a summary line so runs leave a trace regardless of warning.
        print(f"[OK] SAFEAlertDataset: source coverage "
              f"{n_total - n_blank}/{n_total} articles with source "
              f"(blank ratio={ratio:.1%}).", flush=True)

    def __len__(self) -> int:
        return len(self.valid_idx)

    def get_direction_labels(self) -> np.ndarray:
        """Sprint 7 — return cached direction labels indexed by __getitem__ index.

        Used by BalancedBatchSampler to pre-compute class membership without
        invoking the heavy __getitem__ path (which loads articles/embeddings/
        market bars). Returns int8 array of shape [len(self)] with values in
        {0=DOWN, 1=NEUTRAL, 2=UP, -1=label-compute failed}. Sentinel -1 should
        be filtered out by callers if encountered.
        """
        if (not hasattr(self, "_direction_labels")
                or len(self._direction_labels) != len(self.valid_idx)):
            labels = np.full(len(self.valid_idx), -1, dtype=np.int8)
            for j, i in enumerate(self.valid_idx):
                try:
                    curr = self.candle_df.iloc[i]
                    label = self.candle_df.iloc[i + self.horizon_steps]
                    exec_ = self.candle_df.iloc[i + 1]
                    labels[j] = self._get_direction_label(curr, label, exec_)
                except Exception:
                    continue
            self._direction_labels = labels
        return self._direction_labels

    def _filter_article_indices_by_factor_confidence(self, article_indices: List[int]) -> List[int]:
        """Drop low-confidence article occurrences while keeping one best fallback.

        Fallback ordering:
          1. If any article passes the threshold → return the filtered list.
          2. Else if at least one article has a known max_prob entry → return the
             single highest-max_prob article so the candle is not article-less.
          3. Else (all art_idx are out-of-range vs article_max_prob) → return the
             original list unfiltered. This is only safe when article_max_prob
             is stale; we count these occurrences so misalignment surfaces.
        """
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
            # Every art_idx is out-of-range — article_max_prob is stale vs the
            # current article corpus. Counter surfaces this; per-call warning
            # would flood logs, so we only warn once.
            self._factor_prob_stale_count = getattr(self, "_factor_prob_stale_count", 0) + 1
            if not getattr(self, "_factor_prob_stale_warned", False):
                import warnings
                warnings.warn(
                    f"[safe_alert_dataset] article_max_prob covers "
                    f"{len(self.article_max_prob)} articles but the candle→articles "
                    f"map references indices beyond that. min_factor_confidence "
                    f"filter is skipped for those candles. Re-run "
                    f"precompute_factor_labels.py.",
                    RuntimeWarning, stacklevel=2,
                )
                self._factor_prob_stale_warned = True
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

        # Label candle: horizon_steps ahead (PDF Eq.50). The step count is
        # horizon duration / observed candle interval.
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

        # Optional volatility-extension label. Computed as the standard
        # deviation of per-candle returns over the horizon window [t+1, t+h],
        # giving σ_h for the model's optional vol_head. For horizon="1h" there is
        # only ONE realised return, so σ is trivially 0 — we fall back to
        # |return| as a volatility-proxy in that case so the label has
        # useful variance. Non-negative by construction.
        horizon_end_idx = candle_idx + 1 + self.horizon_steps
        horizon_end_idx = min(horizon_end_idx, len(self.candle_df))
        if self.horizon_steps >= 2:
            win = self.candle_df.iloc[candle_idx + 1: horizon_end_idx]
            if len(win) >= 2:
                # intra-window 1-bar returns
                closes = win["close"].to_numpy(dtype=np.float64)
                opens = win["open"].to_numpy(dtype=np.float64)
                bar_rets = (closes - opens) / np.maximum(opens, 1e-8)
                vol_raw = float(np.std(bar_rets, ddof=0))
            else:
                vol_raw = float(abs(ret_raw))
        else:
            # Single-step horizon (1h): use |return| as volatility proxy
            # (no within-horizon dispersion to compute a std from).
            vol_raw = float(abs(ret_raw))
        vol_label = torch.tensor(max(vol_raw, 0.0), dtype=torch.float32)

        sample = {
            "market_features": market_features,  # Already tensor from precomputed or computed above
            "article_embeddings": article_emb,                                  # (K, 768)
            "article_metadata": article_meta,                                   # (K, 4) base or (K, 14) with entity_sentiment
            "article_mask": article_mask,  # ✅ Use computed mask (with padding 0s)
            "direction": torch.tensor(direction_label, dtype=torch.long),
            "factor": fac_labels,             # ✅ FIX #10: (K, C) tensor, not scalar
            "return": ret_label,
            # Realised-volatility label for the optional vol_head auxiliary
            # regression. Paper-final runs keep lambda_vol=0.
            "volatility": vol_label,
            # Session 25 — paper Section 4.1.4 market regime stratification.
            # Integer in {0: SIDEWAYS, 1: BULL, 2: BEAR, 3: VOLATILE}. NOT a
            # model input; downstream analysis groups metrics by regime.
            "regime": torch.tensor(int(self._regime_labels[candle_idx]), dtype=torch.long),
        }
        # Session 23 P0 #3: add bar sequences when precomputed (paper Eq.16).
        # Shape per sample: (n_TFs=5, L_δ, d_δ). DataLoader collates to
        # (B, 5, L, d) which the model splits back into a list of 5 tensors
        # for MultiTimescaleMarketEncoder's sequence-mode encoders.
        if self.market_bars is not None:
            sample["market_bars"] = self._normalize_market_bars(torch.from_numpy(self.market_bars[candle_idx]))
        return sample

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
        n_extracted = len(features)
        if n_extracted < 63:
            features = np.pad(features, (0, 63 - n_extracted), 'constant', constant_values=0.0)

        # Zero-padding fallback is legitimate at inference (stream warmup) but is a
        # SILENT DATA-QUALITY HAZARD during training: if precomputed features are
        # missing, 34/63 dims will be zero and the market encoder is trained on
        # half-blank inputs, typically costing 15-30 % of held-out Accuracy.
        # Session 23 P0 #2 fix: in strict_features mode (default True) we
        # fail fast when > 15% of feature dims would be zero-padded. This
        # catches the "silent 50%-blank inputs" failure where training
        # completes but with severely degraded features. Non-strict mode
        # preserves the warn-and-continue behaviour for legacy callers.
        STRICT_PAD_THRESHOLD = 0.15   # > 15% pad → hard error in strict mode
        pad_frac = (63 - n_extracted) / 63.0
        if n_extracted < 63:
            if self.strict_features and pad_frac > STRICT_PAD_THRESHOLD:
                raise RuntimeError(
                    f"[safe_alert_dataset] Fallback market-feature extraction "
                    f"would zero-pad {pad_frac*100:.1f}% of dims "
                    f"({n_extracted}/63 real). Run precompute_market_features.py "
                    f"first, or pass strict_features=False to allow this "
                    f"(NOT recommended for reported metrics)."
                )
            if not getattr(self, "_warned_zero_padding", False):
                import logging, warnings
                logger = logging.getLogger(__name__)
                pad_pct = pad_frac * 100
                msg = (
                    f"[safe_alert_dataset] Fallback market-feature extraction: "
                    f"{n_extracted}/63 real dims, {pad_pct:.0f}% zero-padded. "
                    f"Strict mode allows up to {STRICT_PAD_THRESHOLD*100:.0f}% "
                    f"pad; current pad is within tolerance. Consider running "
                    f"precompute_market_features.py for full quality."
                )
                logger.warning(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                self._warned_zero_padding = True

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
                # 1.0 = published at decision time, 0.0 = published at the
                # start of the lookback window.
                try:
                    art_time    = pd.Timestamp(article['timestamp'] if 'timestamp' in article.index else datetime.utcnow())
                    decision_time = self._get_decision_timestamp(candle_idx)
                    age_hours   = (decision_time - art_time).total_seconds() / 3600
                    lookback_hours = max(float(self.lookback_hours), 1e-6)
                    recency = max(0.0, min(1.0, 1.0 - (age_hours / lookback_hours)))
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

                # Metadata 3: Novelty (paper Section 4.1.3).
                # Session 24: prefer TF-IDF novelty from precompute_novelty.py
                # (1 - max_cos_sim vs. prior 24h corpus) when available; fall
                # back to rank_norm (relative recency rank within window)
                # otherwise. Rank was the pre-Session-24 behaviour and is
                # documented as deviation D4; precomputed novelty closes the
                # deviation and matches paper Section 4.1.3 exactly.
                if self.article_novelty is not None and art_idx < len(self.article_novelty):
                    meta[i, 3] = float(self.article_novelty[art_idx])
                else:
                    # Fallback: rank within window (0=oldest, 1=most recent).
                    n_valid = min(len(article_indices), K)
                    rank_score = float(n_valid - 1 - i) / max(n_valid - 1, 1)
                    meta[i, 3] = float(rank_score)

                # Metadata [4:14]: Target-based FSA entity sentiment (per factor)
                # Only populated when entity_sentiment precomputed file is available.
                # Previously an art_idx out of range silently fell back to zeros,
                # degrading META 14→4 on misaligned runs. Now we keep the zero
                # fallback (to avoid crashing training on one bad index) but log
                # and count occurrences so misalignment surfaces quickly.
                # Session 23 P1 #5 fix: log the count at decay intervals (1,
                # 10, 100, 1000, 10000, ...) so a persistent misalignment is
                # visible throughout training, not just once per dataset
                # instance. Previous "warn once" behaviour meant a stale
                # entity_sentiment.npy could silently degrade META 14→4 on
                # thousands of articles with only a single early warning
                # buried in the log.
                if self.entity_sentiment is not None:
                    if art_idx < len(self.entity_sentiment):
                        meta[i, 4:14] = torch.from_numpy(self.entity_sentiment[art_idx])
                    else:
                        self._entity_sentiment_oob_count += 1
                        import logging, warnings
                        logger = logging.getLogger(__name__)
                        # Log-decay reporting: warn at 1, 10, 100, 1000, 10k...
                        count = self._entity_sentiment_oob_count
                        should_log = (
                            count == 1
                            or (count > 0 and count == 10 ** int(np.log10(count)))
                        )
                        if should_log:
                            msg = (
                                f"[safe_alert_dataset] entity_sentiment OOB "
                                f"count={count}: art_idx={art_idx} >= "
                                f"n_entity_sentiment={len(self.entity_sentiment)}. "
                                f"Affected articles use zero FSA vector (META 14→4). "
                                f"Re-run precompute_entity_sentiment.py against the "
                                f"current articles_max.csv to fix."
                            )
                            logger.warning(msg)
                            if not self._entity_sentiment_warned:
                                warnings.warn(msg, RuntimeWarning, stacklevel=2)
                                self._entity_sentiment_warned = True

                mask[i] = 1.0

        return emb, meta, mask

    def _get_direction_label(self, curr: pd.Series, label_candle: pd.Series,
                             exec_candle: pd.Series = None) -> int:
        """Compute direction: 0 (DOWN), 1 (NEUTRAL), 2 (UP).

        PDF Eq.51: r = (P_close_{t+h} - P_exec) / P_exec
        P_exec = open of next candle (execution price, no look-ahead).
        P_close_{t+h} = close of horizon candle (h steps ahead).

        For 1h: exec_candle == label_candle (same candle +1 step).
        For longer horizons, label_candle is derived from horizon duration / candle interval.
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

                        # Session 23 P0 #1 fix: LABEL_TEMP unified with
                        # precompute_factor_labels.py (was 0.3 here vs 0.5
                        # there). The mismatch made runtime-fallback labels
                        # systematically sharper than offline Qwen-precomputed
                        # labels, causing a distribution shift on the rare
                        # articles that miss .npy coverage → Lfac gradient
                        # jumps at article-coverage boundaries. 0.5 matches
                        # the precompute script and gives softer, more-honest
                        # per-article factor distributions on fallback.
                        LABEL_TEMP = 0.5
                        if scores.sum() > 0:
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
