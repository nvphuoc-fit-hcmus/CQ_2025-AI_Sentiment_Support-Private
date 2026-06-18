"""
Baseline comparison for SAFE-Alert (PDF Section 4.3 / 5.1 / related work).

Fourteen baselines (4 existing + 8 added + 2 PDF parity baselines):

  Existing baselines:
  1.  market_only            -- 2-layer MLP on 63-dim market features, no news
  2.  all_news_fusion        -- average ALL article embeddings + market MLP (no selection)
  3.  always_alert           -- trivial: always fires alert, no training
  4.  raw_prob_threshold     -- market-only + raw max(softmax) confidence (no conf head)

  New baselines (PDF Section 4.3 related work):
  5.  sentiment_market       -- compact 8-dim sentiment bottleneck + market MLP
  6.  nsm_style              -- soft dot-product attention over all articles (no Top-K)
  7.  llm_factor             -- pre-computed factor label features + market (no neural decomp.)
  8.  sep_style              -- news+market MLP; max-softmax abstaining (no Lcal / Lrisk)
  9.  finin_style            -- multi-head cross-attention fusion; no selective mechanism
  10. interleaved            -- (article, market) pair encoding -> mask-weighted aggregation
  11. temperature_scaled     -- MarketOnly + post-hoc temperature scaling (Guo et al., 2017)
    12. selective_forecasting  -- AllNewsFusion + coverage-optimized abstaining threshold
    13. current_price_predictor -- production-style price-only momentum fallback
    14. all_news_llm_expl      -- all-news forecast + post-hoc LLM explanation

All baselines train on train split.
Threshold-based components are tuned on val split.
Final comparison metrics are reported on held-out test split.
Results are saved to artifact_dir/baseline_results.json for thesis comparison tables.

Usage:
    # Run all 14 baselines
  python run_baselines.py --symbol BTCUSDT --horizon 1h --epochs 30

  # Run specific baselines
  python run_baselines.py --baselines market_only sentiment_market temperature_scaled

  # Quick ablation (fast baselines only)
  python run_baselines.py --baselines always_alert market_only all_news_fusion
"""

import sys
import json
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))

from safe_alert_dataset import SAFEAlertDataset
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_brier_score,
    compute_alert_precision, mini_backtest, compute_model_selection_score,
    compute_auc,
)
from utils import ARTIFACT_DIR

torch.manual_seed(42)
np.random.seed(42)

# ── Shared hyper-parameters ───────────────────────────────────────────────────
_MARKET_DIM   = 63
_NEWS_EMB_DIM = 768
_FACTOR_DIM   = 10
_HIDDEN       = 128
_DROPOUT      = 0.3
_N_CLASSES    = 3
_ACCUM_STEPS  = 2


# =============================================================================
# BASELINE MODEL DEFINITIONS — EXISTING (Baselines 1 & 2 / 4)
# =============================================================================

class MarketOnlyMLP(nn.Module):
    """Baseline 1 & 4: 2-layer MLP on 63-dim market features.

    No news, no attention. Used directly as Baseline 1 (market_only) and as
    the backbone for Baseline 4 (raw_prob_threshold) and Baseline 11
    (temperature_scaled), which modify how confidence is computed at inference.
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(market_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)   # learned confidence (Baseline 1)

    def forward(self, market_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(market_feat)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class AllNewsFusionMLP(nn.Module):
    """Baseline 2: Uniform average of ALL article embeddings + market features.

    No selective attention, no Top-K gating. Tests whether SAFE-Alert's
    selective news encoding (Eq.10-13) adds value over naive all-news averaging.
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.news_proj = nn.Sequential(
            nn.Linear(news_emb_dim, 128),
            nn.GELU(),
        )
        fused_dim = market_dim + 128
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_f  = article_mask.float().unsqueeze(-1)              # (B, K, 1)
        denom   = mask_f.sum(dim=1).clamp(min=1e-8)               # (B, 1)
        avg_emb = (article_emb * mask_f).sum(dim=1) / denom       # (B, 768)
        news_h  = self.news_proj(avg_emb)                          # (B, 128)
        fused   = torch.cat([market_feat, news_h], dim=-1)
        h = self.net(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


# =============================================================================
# ADVANCED TIME-SERIES BASELINES: PatchTST & iTransformer (CaiTien §4)
# Market-only backbones; use the same 63-dim precomputed feature vector as
# MarketOnlyMLP so comparisons reflect architecture differences only.
# =============================================================================

class PatchTSTMarketOnly(nn.Module):
    """PatchTST-style transformer on market features (Nie et al. 2023).

    Treats the 63-dim market feature vector as a sequence of non-overlapping
    patches of size patch_len, applies a Transformer encoder, then pools for
    direction + confidence prediction.

    Simplified variant: input is (B, 63) already precomputed features
    (not raw OHLCV). We reshape into patches along the feature dimension.
    """
    def __init__(self, feat_dim: int = _MARKET_DIM, patch_len: int = 9,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_classes: int = _N_CLASSES, dropout: float = _DROPOUT):
        super().__init__()
        # Number of patches (pad if not divisible)
        self.patch_len = patch_len
        n_patches = math.ceil(feat_dim / patch_len)
        self.pad_len = n_patches * patch_len - feat_dim
        self.n_patches = n_patches

        self.patch_proj = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dir_head  = nn.Linear(d_model, n_classes)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, market_feat: torch.Tensor, **_) -> Dict[str, torch.Tensor]:
        B = market_feat.shape[0]
        # Pad to multiple of patch_len
        if self.pad_len > 0:
            market_feat = F.pad(market_feat, (0, self.pad_len))
        x = market_feat.view(B, self.n_patches, self.patch_len)  # (B, P, L)
        x = self.patch_proj(x)                                   # (B, P, d_model)
        x = self.transformer(x)                                  # (B, P, d_model)
        x = self.norm(x.mean(dim=1))                             # (B, d_model)
        return {
            "dir_logits":  self.dir_head(x),
            "confidence":  torch.sigmoid(self.conf_head(x)).squeeze(-1),
        }


class iTransformerMarketOnly(nn.Module):
    """iTransformer on market features (Liu et al. 2024, ICLR 2024).

    iTransformer inverts the attention axis: treats each feature dimension
    as a 'variate token' rather than each time step. Here we have a single
    timestep with 63 features, so we embed each feature independently and
    apply attention across the feature dimension.
    """
    def __init__(self, feat_dim: int = _MARKET_DIM, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 n_classes: int = _N_CLASSES, dropout: float = _DROPOUT):
        super().__init__()
        self.feat_embed = nn.Linear(1, d_model)      # embed each scalar feature
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dir_head  = nn.Linear(d_model, n_classes)
        self.conf_head = nn.Linear(d_model, 1)

    def forward(self, market_feat: torch.Tensor, **_) -> Dict[str, torch.Tensor]:
        # market_feat: (B, 63)  → treat each feature as a variate token
        x = market_feat.unsqueeze(-1)       # (B, 63, 1)
        x = self.feat_embed(x)              # (B, 63, d_model)
        x = self.transformer(x)             # (B, 63, d_model)
        x = self.norm(x.mean(dim=1))        # (B, d_model)  — pool across variates
        return {
            "dir_logits":  self.dir_head(x),
            "confidence":  torch.sigmoid(self.conf_head(x)).squeeze(-1),
        }


# =============================================================================
# EVIDENCE-SELECTION BASELINES 15 & 16 (CaiTien.md Section 4 + Section 6)
# =============================================================================

class KSelectNewsFusionMLP(nn.Module):
    """Baselines 15-17: AllNewsFusion but limited to K articles by a selection rule.

    selection_mode:
      "random"          — K articles chosen uniformly at random per sample per call
      "most_recent"     — first K valid articles in the sequence (dataset sorts
                          articles most-recent-first, so index 0 = newest)
      "source_priority" — K articles with highest embedding L2 norm, used as a
                          proxy for source quality (higher-quality sources produce
                          more informative, content-dense embeddings). Requires no
                          external source metadata — purely self-supervised signal.

    Compared to AllNewsFusion (Baseline 2, all articles averaged) and
    SAFE-Alert (learned selective attention):
    • random_k_evidence        shows the floor for K-article selection without learning.
    • most_recent_k_evidence   shows a strong heuristic (recency bias).
    • source_priority_k_evidence shows an embedding-norm quality heuristic.
    If SAFE-Alert's selective attention > all three, the learned selector adds value.
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT,
                 k: int = 4,
                 selection_mode: str = "random"):
        super().__init__()
        assert selection_mode in ("random", "most_recent", "source_priority"), \
            f"selection_mode must be 'random', 'most_recent', or 'source_priority', got {selection_mode!r}"
        self.k = int(k)
        self.selection_mode = selection_mode
        self.news_proj = nn.Sequential(nn.Linear(news_emb_dim, 128), nn.GELU())
        fused_dim = market_dim + 128
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),   nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def _select_mask(self, article_mask: torch.Tensor,
                     article_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K = article_mask.shape
        sel = torch.zeros_like(article_mask, dtype=torch.bool)
        for b in range(B):
            valid = article_mask[b].bool().nonzero(as_tuple=False).squeeze(-1)
            n = valid.numel()
            if n == 0:
                continue
            k_pick = min(self.k, n)
            if self.selection_mode == "random":
                perm = torch.randperm(n, device=article_mask.device)[:k_pick]
                chosen = valid[perm]
            elif self.selection_mode == "source_priority":
                # Select K articles with highest embedding L2 norm as source-quality proxy.
                # Embeddings from high-credibility sources (e.g. CoinDesk, Reuters) tend to
                # have higher magnitude than noisy/low-signal aggregator content.
                if article_emb is not None:
                    norms = article_emb[b, valid].norm(dim=-1)           # (n,)
                    _, top_idx = norms.topk(k_pick, largest=True)
                    chosen = valid[top_idx]
                else:
                    chosen = valid[:k_pick]   # fallback to most_recent if no emb available
            else:  # most_recent: first k_pick valid indices (newest first)
                chosen = valid[:k_pick]
            sel[b, chosen] = True
        return sel

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        sel_mask = self._select_mask(
            article_mask,
            article_emb if self.selection_mode == "source_priority" else None,
        ).float().unsqueeze(-1)  # (B,K,1)
        denom    = sel_mask.sum(dim=1).clamp(min=1e-8)                    # (B,1)
        avg_emb  = (article_emb * sel_mask).sum(dim=1) / denom            # (B,768)
        news_h   = self.news_proj(avg_emb)
        fused    = torch.cat([market_feat, news_h], dim=-1)
        h = self.net(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


# =============================================================================
# BASELINE MODEL DEFINITIONS — NEW (Baselines 5-12, PDF Section 4.3)
# =============================================================================

class SentimentMarketFusionModel(nn.Module):
    """Baseline 5: Sentiment+Market Fusion (PDF Section 4.3, related work).

    Maps article embeddings through a narrow sentiment bottleneck (8-dim,
    Tanh-bounded) before fusing with market features. The bottleneck
    explicitly models sentiment as a compact polarity signal rather than a
    high-dimensional embedding — analogous to dictionary-based or simple
    classifier sentiment systems common in the financial NLP literature.

    Contrast with AllNewsFusion (Baseline 2): that baseline retains a full
    128-dim news projection; here the bottleneck is intentionally narrow to
    simulate aggregated sentiment scores.
    """
    _SENTIMENT_DIM = 8

    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(news_emb_dim, self._SENTIMENT_DIM),
            nn.Tanh(),   # bounded in [-1, +1], mimicking polarity scores
        )
        self.net = nn.Sequential(
            nn.Linear(market_dim + self._SENTIMENT_DIM, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_f    = article_mask.float().unsqueeze(-1)
        avg_emb   = (article_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        sentiment = self.sentiment_encoder(avg_emb)                # (B, 8)
        fused     = torch.cat([market_feat, sentiment], dim=-1)
        h = self.net(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class NSMStyleModel(nn.Module):
    """Baseline 6: NSM-style soft attention over articles (PDF Section 4.3).

    Implements scaled dot-product attention where the market feature vector
    acts as the query and article embeddings are keys and values. All K
    articles contribute via soft attention weights (no hard Top-K gating).

    Direct ablation of SAFE-Alert's selective Top-K mechanism (Eq.12):
    tests whether hard gating adds value over standard soft attention.
    No confidence head, no factor module, no Lsel or Lrisk losses.
    """
    _ATTN_DIM = 64

    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        D = self._ATTN_DIM
        self.q_proj = nn.Linear(market_dim, D)
        self.k_proj = nn.Linear(news_emb_dim, D)
        self.v_proj = nn.Linear(news_emb_dim, D)
        self.net = nn.Sequential(
            nn.Linear(market_dim + D, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = self.q_proj(market_feat).unsqueeze(1)   # (B, 1, D)
        k = self.k_proj(article_emb)                # (B, K, D)
        v = self.v_proj(article_emb)                # (B, K, D)
        scores = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)  # (B, 1, K)
        # Mask padding positions with -inf before softmax
        pad_mask = (~article_mask.bool()).unsqueeze(1)              # (B, 1, K)
        scores   = scores.masked_fill(pad_mask, -1e9)
        attn = F.softmax(scores, dim=-1)            # (B, 1, K) — soft weights
        ctx  = (attn @ v).squeeze(1)                # (B, D)
        fused = torch.cat([market_feat, ctx], dim=-1)
        h = self.net(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class LLMFactorModel(nn.Module):
    """Baseline 7: LLMFactor-inspired forecasting (PDF Section 4.3).

    ⚠️  FIDELITY CAVEAT (P1 #13):
    Wang et al. LLMFactor [6] uses Sequential Knowledge-Guided Prompting to
    make an LLM GENERATE the factor taxonomy PER SAMPLE at inference time and
    feeds that as the causal explanation for the direction prediction. This
    baseline uses a STATIC factor ontology (10 classes, pre-computed by the
    keyword/LLM pipeline in precompute_factor_labels.py) and feeds the
    resulting distribution as numeric features to a vanilla MLP head. It
    therefore tests a STRICTLY WEAKER form of the LLMFactor hypothesis:

      "Does even a simple MLP over pre-computed factor distributions
       beat plain market-only or all-news baselines?"

    It is NOT a faithful replica of Wang et al.'s end-to-end prompting
    pipeline. We keep this baseline because:
      • Factor features themselves carry real signal (the pseudo-labels are
        LLM-generated in precompute_factor_labels.py with method=qwen/gemini).
      • It isolates whether SAFE-Alert's gains come from the factor module
        specifically vs. the selective news + confidence gate jointly.
      • Running a true per-sample prompting inference loop at 90k+ samples
        is thesis-scope prohibitive; the pseudo-labels already capture the
        representational benefit without the inference cost.

    Paper must flag this explicitly in the baseline description — do NOT
    describe this as "LLMFactor [6]" outright, say "LLMFactor-inspired" or
    "static factor-feature MLP".

    Input:
      factor_labels: (B, K, 10) per-article pseudo-label distributions;
                     mean-pooled across valid articles to obtain a 10-dim
                     factor signal, then projected and fused with market.
    """
    _FACTOR_PROJ = 32

    def __init__(self, market_dim: int = _MARKET_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.factor_proj = nn.Sequential(
            nn.Linear(_FACTOR_DIM, self._FACTOR_PROJ), nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.Linear(market_dim + self._FACTOR_PROJ, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(self, market_feat: torch.Tensor,
                factor_labels: Optional[torch.Tensor],
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = market_feat.shape[0]
        if factor_labels is not None:
            mask_f     = article_mask.float().unsqueeze(-1)          # (B, K, 1)
            denom      = mask_f.sum(dim=1).clamp(min=1e-8)           # (B, 1)
            avg_factor = (factor_labels * mask_f).sum(dim=1) / denom # (B, 10)
            factor_h   = self.factor_proj(avg_factor)
        else:
            factor_h = torch.zeros(B, self._FACTOR_PROJ, device=market_feat.device)
        fused = torch.cat([market_feat, factor_h], dim=-1)
        h = self.net(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class SEPStyleModel(nn.Module):
    """Baseline 8: SEP-style Selective Prediction (PDF Section 4.3).

    Trains a news+market classifier. At inference, abstains when
    max(softmax(logits)) < tau (no alert). The threshold tau is optimized
    on the validation set for a target coverage rate.

    Implements the standard selective prediction framework (Geifman &
    El-Yaniv, 2017): uncertainty = 1 - max_softmax; abstain when uncertain.

    Key differences from SAFE-Alert:
    - No dedicated confidence head (uses max-softmax directly)
    - No calibration loss Lcal (Eq.35)
    - No selective risk loss Lrisk (Eq.37)
    - Threshold chosen post-hoc for target coverage, not learned end-to-end
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.news_proj = nn.Linear(news_emb_dim, 64)
        self.net = nn.Sequential(
            nn.Linear(market_dim + 64, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head = nn.Linear(hidden, n_classes)
        # No dedicated confidence head — max(softmax) is used at inference

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask_f  = article_mask.float().unsqueeze(-1)
        avg_emb = (article_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-8)
        news_h  = F.gelu(self.news_proj(avg_emb))
        fused   = torch.cat([market_feat, news_h], dim=-1)
        h       = self.net(fused)
        dir_logits = self.dir_head(h)
        # Confidence = max softmax probability (no dedicated head)
        confidence = F.softmax(dir_logits, dim=-1).max(dim=-1).values
        return {"dir_logits": dir_logits, "confidence": confidence}


class FININStyleModel(nn.Module):
    """Baseline 9: FININ-style full multimodal model (PDF Section 4.3).

    Architecture:
      - Market encoder  : Linear -> LayerNorm -> GELU
      - Cross-attention : multi-head (Q=market, K/V=articles) + residual + LayerNorm
      - Fusion MLP      : concat(market_h, news_ctx) -> 2-layer MLP

    Represents a competitive multimodal baseline with learned cross-attention
    and a dedicated confidence head, but without:
    - Top-K hard article selection (Eq.12)
    - Factor decomposition module (Eq.14-15)
    - Calibration loss Lcal (Eq.35)
    - Faithfulness loss Lfaith (Eq.36)
    - Selective risk loss Lrisk (Eq.37)

    Tests the value of SAFE-Alert's curriculum training and multi-objective
    loss over a strong multimodal attention baseline.
    """
    _D_MODEL = 128
    _N_HEADS  = 4

    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        D, H   = self._D_MODEL, self._N_HEADS
        self.d_model  = D
        self.n_heads  = H
        self.head_dim = D // H
        # Market encoder
        self.market_enc = nn.Sequential(
            nn.Linear(market_dim, D), nn.LayerNorm(D), nn.GELU(),
        )
        # Cross-attention projections
        self.q_proj   = nn.Linear(D, D)
        self.k_proj   = nn.Linear(news_emb_dim, D)
        self.v_proj   = nn.Linear(news_emb_dim, D)
        self.out_proj  = nn.Linear(D, D)
        self.attn_norm = nn.LayerNorm(D)
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(D * 2, D), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(D, D // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(D // 2, n_classes)
        self.conf_head = nn.Linear(D // 2, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, _ = article_emb.shape
        D, H, Dh = self.d_model, self.n_heads, self.head_dim
        # Market encoding
        m = self.market_enc(market_feat)            # (B, D)
        # Multi-head cross-attention (Q from market, K/V from articles)
        q = self.q_proj(m).unsqueeze(1)             # (B, 1, D)
        k = self.k_proj(article_emb)                # (B, K, D)
        v = self.v_proj(article_emb)                # (B, K, D)
        # Reshape for multi-head
        q = q.view(B,  1, H, Dh).transpose(1, 2)   # (B, H, 1, Dh)
        k = k.view(B,  K, H, Dh).transpose(1, 2)   # (B, H, K, Dh)
        v = v.view(B,  K, H, Dh).transpose(1, 2)   # (B, H, K, Dh)
        scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)  # (B, H, 1, K)
        # Mask padding positions
        pad_mask = (~article_mask.bool()).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        scores = scores.masked_fill(pad_mask, -1e9)
        attn = F.softmax(scores, dim=-1)            # (B, H, 1, K)
        ctx  = (attn @ v).squeeze(2)                # (B, H, Dh)
        ctx  = ctx.transpose(1, 2).contiguous().view(B, D)  # (B, D)
        ctx  = self.out_proj(ctx)
        ctx  = self.attn_norm(ctx + m)              # residual + layer norm
        # Fusion
        fused = torch.cat([m, ctx], dim=-1)         # (B, 2D)
        h = self.fusion(fused)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class InterleavedModel(nn.Module):
    """Baseline 10: Interleaved Text-Time-Series model (PDF Section 4.3).

    For each article, concatenates the article embedding with the concurrent
    market feature vector (same candle window), forming an interleaved
    (article, market) pair. Each pair is processed jointly through a shared
    encoder. The resulting pair representations are mean-pooled over articles.

    Models the hypothesis that article impact depends on the concurrent
    market regime — unlike separate-encoder + late-fusion designs (e.g.
    AllNewsFusion, FININStyle). Inspired by interleaved text-time-series
    approaches in financial NLP (Ding et al., 2021 and related work).
    """
    _PAIR_HIDDEN = 128

    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        pair_dim = news_emb_dim + market_dim
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_dim, self._PAIR_HIDDEN),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.agg_net = nn.Sequential(
            nn.Linear(self._PAIR_HIDDEN, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.dir_head  = nn.Linear(hidden // 2, n_classes)
        self.conf_head = nn.Linear(hidden // 2, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, K, _ = article_emb.shape
        # Expand market features to per-article dimension, then interleave
        mkt_exp = market_feat.unsqueeze(1).expand(-1, K, -1)  # (B, K, market_dim)
        pairs   = torch.cat([article_emb, mkt_exp], dim=-1)   # (B, K, D+market_dim)
        h_pairs = self.pair_encoder(pairs)                     # (B, K, pair_hidden)
        # Mask-weighted mean pooling over articles
        mask_f = article_mask.float().unsqueeze(-1)            # (B, K, 1)
        denom  = mask_f.sum(dim=1).clamp(min=1e-8)             # (B, 1)
        h_agg  = (h_pairs * mask_f).sum(dim=1) / denom         # (B, pair_hidden)
        h = self.agg_net(h_agg)
        return {
            "dir_logits": self.dir_head(h),
            "confidence": torch.sigmoid(self.conf_head(h)).squeeze(-1),
        }


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for confidence calibration (Guo et al., 2017).

    Optimizes a single scalar temperature T on the validation set by
    minimizing negative log-likelihood: T* = argmin_T NLL(softmax(z/T), y).

    T > 1 : softens the distribution (reduces overconfidence).
    T < 1 : sharpens the distribution (rarely helpful in practice).
    T = 1 : no change (uncalibrated model).

    This is used in Baseline 11 (temperature_scaled) as a post-hoc calibration
    step applied to a trained MarketOnly model.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(self, logits_np: np.ndarray, labels_np: np.ndarray) -> float:
        """Fit temperature T on validation logits/labels. Returns optimal T."""
        self.train()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)
        logits_t = torch.tensor(logits_np, dtype=torch.float32)
        labels_t = torch.tensor(labels_np, dtype=torch.long)

        def _nll_closure():
            optimizer.zero_grad()
            scaled = logits_t / self.temperature.clamp(min=0.05)
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(_nll_closure)
        self.eval()
        return float(self.temperature.clamp(min=0.05).item())


# =============================================================================
# TRAINING HELPERS — MARKET-ONLY (Baselines 1, 4, 11)
# =============================================================================

def _train_epoch_market_only(model: MarketOnlyMLP, loader, optimizer,
                              device: str, accum: int = _ACCUM_STEPS) -> float:
    """One training epoch for the market-only baseline."""
    model.train()
    total_loss, count = 0.0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        market_feat = batch["market_features"].to(device)
        dir_labels  = batch["direction"].to(device)
        out  = model(market_feat)
        loss = F.cross_entropy(out["dir_logits"], dir_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        (loss / accum).backward()
        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        count      += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate_market_only(model: MarketOnlyMLP, loader,
                           device: str) -> Tuple[np.ndarray, ...]:
    """Return (preds, labels, confidence, ret_labels, dir_probs) arrays."""
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []
    for batch in loader:
        market_feat = batch["market_features"].to(device)
        out   = model(market_feat)
        probs = F.softmax(out["dir_logits"], dim=-1)
        preds_l.append(probs.argmax(dim=-1).cpu().numpy())
        labels_l.append(batch["direction"].numpy())
        conf_l.append(out["confidence"].cpu().numpy())
        ret_l.append(batch["return"].numpy())
        probs_l.append(probs.cpu().numpy())
    return (np.concatenate(preds_l), np.concatenate(labels_l),
            np.concatenate(conf_l),  np.concatenate(ret_l),
            np.concatenate(probs_l))


# =============================================================================
# TRAINING HELPERS — NEWS+MARKET (Baselines 2, 5, 6, 8, 9, 10, 12)
# =============================================================================

def _train_epoch_news_market(model: nn.Module, loader, optimizer,
                              device: str, accum: int = _ACCUM_STEPS) -> float:
    """One training epoch for any model with forward(market, article_emb, mask)."""
    model.train()
    total_loss, count = 0.0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        market_feat  = batch["market_features"].to(device)
        article_emb  = batch["article_embeddings"].to(device)
        article_mask = batch["article_mask"].to(device)
        dir_labels   = batch["direction"].to(device)
        out  = model(market_feat, article_emb, article_mask)
        loss = F.cross_entropy(out["dir_logits"], dir_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        (loss / accum).backward()
        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        count      += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate_news_market(model: nn.Module, loader,
                           device: str) -> Tuple[np.ndarray, ...]:
    """Evaluate any model with forward(market, article_emb, mask)."""
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []
    for batch in loader:
        market_feat  = batch["market_features"].to(device)
        article_emb  = batch["article_embeddings"].to(device)
        article_mask = batch["article_mask"].to(device)
        out   = model(market_feat, article_emb, article_mask)
        probs = F.softmax(out["dir_logits"], dim=-1)
        preds_l.append(probs.argmax(dim=-1).cpu().numpy())
        labels_l.append(batch["direction"].numpy())
        conf_l.append(out["confidence"].cpu().numpy())
        ret_l.append(batch["return"].numpy())
        probs_l.append(probs.cpu().numpy())
    return (np.concatenate(preds_l), np.concatenate(labels_l),
            np.concatenate(conf_l),  np.concatenate(ret_l),
            np.concatenate(probs_l))


# =============================================================================
# TRAINING HELPERS — FACTOR-LABEL MODEL (Baseline 7)
# =============================================================================

def _train_epoch_factor(model: LLMFactorModel, loader, optimizer,
                         device: str, accum: int = _ACCUM_STEPS) -> float:
    """One training epoch for LLMFactorModel (uses batch['factor'], not article_emb)."""
    model.train()
    total_loss, count = 0.0, 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        market_feat  = batch["market_features"].to(device)
        article_mask = batch["article_mask"].to(device)
        dir_labels   = batch["direction"].to(device)
        factor_labs  = batch.get("factor")
        if factor_labs is not None:
            factor_labs = factor_labs.to(device)
        out  = model(market_feat, factor_labs, article_mask)
        loss = F.cross_entropy(out["dir_logits"], dir_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        (loss / accum).backward()
        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        count      += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate_factor(model: LLMFactorModel, loader,
                      device: str) -> Tuple[np.ndarray, ...]:
    """Evaluate LLMFactorModel."""
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []
    for batch in loader:
        market_feat  = batch["market_features"].to(device)
        article_mask = batch["article_mask"].to(device)
        factor_labs  = batch.get("factor")
        if factor_labs is not None:
            factor_labs = factor_labs.to(device)
        out   = model(market_feat, factor_labs, article_mask)
        probs = F.softmax(out["dir_logits"], dim=-1)
        preds_l.append(probs.argmax(dim=-1).cpu().numpy())
        labels_l.append(batch["direction"].numpy())
        conf_l.append(out["confidence"].cpu().numpy())
        ret_l.append(batch["return"].numpy())
        probs_l.append(probs.cpu().numpy())
    return (np.concatenate(preds_l), np.concatenate(labels_l),
            np.concatenate(conf_l),  np.concatenate(ret_l),
            np.concatenate(probs_l))


# =============================================================================
# SHARED METRIC COMPUTATION
# =============================================================================

def _derive_alert_thresholds(
    confidence: np.ndarray,
    dir_probs: Optional[np.ndarray] = None,
    tau_percentile: float = 70.0,
    gamma_percentile: float = 60.0,
) -> Tuple[float, float]:
    """Derive alert thresholds on a tuning split (typically validation)."""
    max_prob = dir_probs.max(axis=-1) if dir_probs is not None else confidence
    tau = float(np.percentile(confidence, tau_percentile))
    gamma = float(np.percentile(max_prob, gamma_percentile))
    return tau, gamma


def _compute_metrics(preds: np.ndarray, labels: np.ndarray,
                     confidence: np.ndarray, ret_labels: np.ndarray,
                     dir_probs: Optional[np.ndarray] = None,
                     tau: Optional[float] = None,
                     gamma: Optional[float] = None) -> Dict:
    """Compute the full metric set using the same functions as SAFEAlertTrainer."""
    if tau is None or gamma is None:
        tau, gamma = _derive_alert_thresholds(confidence, dir_probs)

    macro_f1 = compute_macro_f1(preds, labels)
    mcc      = compute_mcc(preds, labels)
    ece      = compute_ece(confidence, preds, labels)
    brier    = compute_brier_score(confidence, preds, labels)

    alert_prec, alert_cov, _ = compute_alert_precision(
        confidence, preds, labels,
        dir_probs=dir_probs, tau=tau, gamma=gamma,
    )
    backtest     = mini_backtest(confidence, preds, ret_labels,
                                  tau=tau, gamma=gamma, dir_probs=dir_probs)
    alert_sharpe = backtest["alert_sharpe"]
    score        = compute_model_selection_score(
        macro_f1, mcc, ece, alert_sharpe,
        alert_coverage=backtest.get("alert_coverage", 0.0),
    )
    auc          = compute_auc(dir_probs, labels) if dir_probs is not None else 0.5

    return {
        "macro_f1":        macro_f1,
        "mcc":             mcc,
        "ece":             ece,
        "brier":           brier,
        "alert_precision": alert_prec,
        "alert_coverage":  alert_cov,
        "alert_sharpe":    alert_sharpe,
        "model_score":     score,
        "auc":             auc,
    }


# =============================================================================
# GENERIC RUNNER FOR NEWS+MARKET BASELINES (Baselines 5, 6, 8, 9, 10)
# =============================================================================

def _run_news_market_baseline(
        model_class, baseline_num: int, name: str, desc: str,
    train_loader, eval_loader, args,
        model_kwargs: Optional[Dict] = None,
        use_factor: bool = False,
) -> Dict:
    """Generic runner for trainable news+market or factor-based baselines."""
    print(f"\n{'='*60}")
    print(f"  BASELINE {baseline_num}: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")

    model_kwargs = model_kwargs or {}
    model     = model_class(**model_kwargs).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    train_fn = _train_epoch_factor    if use_factor else _train_epoch_news_market
    eval_fn  = _evaluate_factor       if use_factor else _evaluate_news_market

    for epoch in range(1, args.epochs + 1):
        loss = train_fn(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    preds, labels, confidence, ret_labels, dir_probs = eval_fn(
        model, eval_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# =============================================================================
# BASELINE RUNNERS — EXISTING (1-4)
# =============================================================================

def run_market_only(train_loader, eval_loader, args) -> Dict:
    """Baseline 1: 2-layer MLP on market features only, with learned confidence head."""
    print(f"\n{'='*60}")
    print(f"  BASELINE 1: Market-Only MLP")
    print(f"  No news input. Tests whether news adds value over market features alone.")
    print(f"{'='*60}")
    model     = MarketOnlyMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")
    preds, labels, confidence, ret_labels, dir_probs = _evaluate_market_only(
        model, eval_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_lightgbm_market_only(train_loader, eval_loader, args) -> Dict:
    """LightGBM (or sklearn GBT) on market features — strong tabular baseline.

    No news, no deep learning. Tests whether gradient-boosted trees can match
    the neural market-only MLP on the same 63-dim feature set (CaiTien §4).
    Falls back to sklearn GradientBoostingClassifier if lightgbm is unavailable.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 17: LightGBM Market-Only")
    print(f"  Gradient-boosted trees on {_MARKET_DIM}-dim market features.")
    print(f"  No news. Strong tabular baseline vs. deep model.")
    print(f"{'='*60}")

    try:
        import lightgbm as lgb
        _USE_LGB = True
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as _GBT  # noqa: F401
        _USE_LGB = False
        print("  [WARN] lightgbm not installed; falling back to sklearn GradientBoostingClassifier")

    def _collect(loader):
        X_list, y_list, ret_list = [], [], []
        for batch in loader:
            mf = batch["market_features"].numpy()
            if mf.ndim > 2:
                mf = mf.reshape(mf.shape[0], -1)
            X_list.append(mf)
            y_list.append(batch["direction"].numpy())
            ret_list.append(batch["return"].numpy())
        return (np.concatenate(X_list),
                np.concatenate(y_list).astype(np.int64),
                np.concatenate(ret_list))

    print("  Collecting features from loaders...")
    X_train, y_train, _   = _collect(train_loader)
    X_eval,  y_eval,  ret = _collect(eval_loader)
    print(f"  Train: {X_train.shape}  Eval: {X_eval.shape}")

    if _USE_LGB:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            colsample_bytree=0.8, subsample=0.8, min_child_samples=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )

    print("  Training...")
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_eval)   # (N, n_present_classes)

    # Map classifier classes → columns 0/1/2 (DOWN/NEUTRAL/UP).
    # Some classes may be absent from training data.
    full_proba = np.zeros((len(y_eval), 3), dtype=np.float32)
    for col_idx, cls in enumerate(clf.classes_):
        full_proba[:, int(cls)] = proba[:, col_idx]

    preds      = full_proba.argmax(axis=1).astype(np.int64)
    confidence = full_proba.max(axis=1).astype(np.float32)

    metrics = _compute_metrics(preds, y_eval, confidence, ret, full_proba)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    return metrics


def run_all_news_fusion(train_loader, eval_loader, args) -> Dict:
    """Baseline 2: Average ALL articles (no selection) + market features."""
    print(f"\n{'='*60}")
    print(f"  BASELINE 2: All-News Fusion (uniform average, no Top-K selection)")
    print(f"  Tests whether SAFE-Alert's selective attention adds value over naive fusion.")
    print(f"{'='*60}")
    model     = AllNewsFusionMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_news_market(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")
    preds, labels, confidence, ret_labels, dir_probs = _evaluate_news_market(
        model, eval_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def run_always_alert(eval_loader, args) -> Dict:
    """Baseline 3: Always fire alert (conf=1.0, direction=majority class).

    No training. Establishes the lower bound: how well does a naive strategy
    that always fires an alert perform compared to the learned model?
    Direction prediction: majority class (class 1 = NEUTRAL in most datasets).
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 3: Always-Alert (trivial, no training)")
    print(f"  Lower bound: alerts on every sample using the majority class.")
    print(f"{'='*60}")
    all_labels, all_ret_labels = [], []
    for batch in eval_loader:
        all_labels.append(batch["direction"].numpy())
        all_ret_labels.append(batch["return"].numpy())
    labels     = np.concatenate(all_labels)
    ret_labels = np.concatenate(all_ret_labels)
    N = len(labels)
    majority_class = int(np.bincount(labels.astype(np.int64)).argmax())
    preds      = np.full(N, majority_class, dtype=np.int64)
    confidence = np.ones(N, dtype=np.float32)
    dir_probs  = np.zeros((N, 3), dtype=np.float32)
    dir_probs[:, majority_class] = 1.0
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  Majority class: {majority_class} | N={N}")
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    return metrics


def run_raw_prob_threshold(train_loader, eval_loader, args) -> Dict:
    """Baseline 4: Market-Only model with raw max(softmax) threshold as confidence.

    Trains MarketOnlyMLP but at inference uses max(softmax) directly as the
    alert confidence (ignoring the learned confidence head). Tests whether the
    dedicated confidence head adds value over raw probability thresholding.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 4: Raw-Probability Threshold (no learned confidence head)")
    print(f"  Tests whether dedicated confidence calibration adds value over max-softmax.")
    print(f"{'='*60}")
    model     = MarketOnlyMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            market_feat = batch["market_features"].to(args.device)
            out   = model(market_feat)
            probs = F.softmax(out["dir_logits"], dim=-1)
            preds_l.append(probs.argmax(dim=-1).cpu().numpy())
            labels_l.append(batch["direction"].numpy())
            conf_l.append(probs.max(dim=-1).values.cpu().numpy())  # raw max-softmax
            ret_l.append(batch["return"].numpy())
            probs_l.append(probs.cpu().numpy())
    preds      = np.concatenate(preds_l)
    labels     = np.concatenate(labels_l)
    confidence = np.concatenate(conf_l)
    ret_labels = np.concatenate(ret_l)
    dir_probs  = np.concatenate(probs_l)
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# =============================================================================
# BASELINE RUNNERS — NEW (5-12, PDF Section 4.3)
# =============================================================================

def run_sentiment_market(train_loader, eval_loader, args) -> Dict:
    """Baseline 5: Sentiment+Market Fusion (compact 8-dim sentiment bottleneck)."""
    return _run_news_market_baseline(
        SentimentMarketFusionModel, 5,
        "Sentiment+Market Fusion",
        "8-dim Tanh sentiment bottleneck + market MLP; simulates aggregate polarity scores.",
        train_loader, eval_loader, args,
    )


def run_nsm_style(train_loader, eval_loader, args) -> Dict:
    """Baseline 6: NSM-style soft attention (no Top-K hard gating)."""
    return _run_news_market_baseline(
        NSMStyleModel, 6,
        "NSM-style Soft Attention",
        "Scaled dot-product soft attention over all K articles; ablates SAFE-Alert Eq.12.",
        train_loader, eval_loader, args,
    )


def run_llm_factor(train_loader, eval_loader, args) -> Dict:
    """Baseline 7: LLMFactor-style — explicit factor labels as input features."""
    return _run_news_market_baseline(
        LLMFactorModel, 7,
        "LLMFactor-style Forecasting",
        "Pre-computed factor distributions (10-dim) as input; no neural decomposition.",
        train_loader, eval_loader, args,
        use_factor=True,
    )


def run_sep_style(train_loader, eval_loader, args) -> Dict:
    """Baseline 8: SEP-style selective prediction (max-softmax abstaining)."""
    return _run_news_market_baseline(
        SEPStyleModel, 8,
        "SEP-style Selective Prediction",
        "News+market MLP; max-softmax abstaining; no Lcal or Lrisk losses.",
        train_loader, eval_loader, args,
    )


def run_finin_style(train_loader, eval_loader, args) -> Dict:
    """Baseline 9: FININ-style multi-head cross-attention fusion."""
    return _run_news_market_baseline(
        FININStyleModel, 9,
        "FININ-style Multimodal",
        "Multi-head cross-attention (Q=market, K/V=articles); no factor / selective losses.",
        train_loader, eval_loader, args,
    )


def run_interleaved(train_loader, eval_loader, args) -> Dict:
    """Baseline 10: Interleaved text-time-series pair encoding."""
    return _run_news_market_baseline(
        InterleavedModel, 10,
        "Interleaved Text-Time-Series",
        "Each (article, market) pair encoded jointly; mask-weighted mean pooling.",
        train_loader, eval_loader, args,
    )


def run_temperature_scaled(train_loader, val_loader, eval_loader, args) -> Dict:
    """Baseline 11: MarketOnly + post-hoc temperature scaling (Guo et al., 2017).

    Step 1: Train a MarketOnly MLP backbone.
    Step 2: Collect raw logits on the validation set.
    Step 3: Fit temperature T* by minimizing NLL on val logits/labels.
    Step 4: Recompute calibrated predictions using logits / T*.

    This isolates the contribution of proper confidence calibration (vs
    SAFE-Alert's end-to-end Lcal loss) and tests post-hoc recalibration as
    an alternative to learned calibration.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 11: Temperature-Scaled Threshold (Guo et al., 2017)")
    print(f"  MarketOnly MLP + post-hoc temperature scaling on val logits.")
    print(f"{'='*60}")

    # Step 1: Train base MarketOnly model
    base_model = MarketOnlyMLP().to(args.device)
    optimizer  = optim.AdamW(base_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(base_model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  [Base] Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    # Step 2: Collect raw validation logits (before any calibration)
    base_model.eval()
    logits_l, labels_l = [], []
    with torch.no_grad():
        for batch in val_loader:
            market_feat = batch["market_features"].to(args.device)
            out = base_model(market_feat)
            logits_l.append(out["dir_logits"].cpu().numpy())
            labels_l.append(batch["direction"].numpy())
    logits_np = np.concatenate(logits_l)
    labels_np = np.concatenate(labels_l)

    # Step 3: Fit optimal temperature T* on val set
    scaler = TemperatureScaler()
    T_opt  = scaler.calibrate(logits_np, labels_np)
    print(f"  [Calibrate] Optimal temperature T* = {T_opt:.4f}  "
          f"({'softened' if T_opt > 1.0 else 'sharpened'} distribution)")

    # Step 4: Evaluate calibrated model on held-out evaluation split
    eval_logits_l, eval_labels_l, eval_ret_l = [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            market_feat = batch["market_features"].to(args.device)
            out = base_model(market_feat)
            eval_logits_l.append(out["dir_logits"].cpu().numpy())
            eval_labels_l.append(batch["direction"].numpy())
            eval_ret_l.append(batch["return"].numpy())

    eval_logits_np = np.concatenate(eval_logits_l)
    eval_labels_np = np.concatenate(eval_labels_l)
    eval_ret_np    = np.concatenate(eval_ret_l)

    with torch.no_grad():
        scaled_logits = torch.tensor(eval_logits_np) / max(T_opt, 0.05)
        probs_np      = F.softmax(scaled_logits, dim=-1).numpy()
    preds_np      = probs_np.argmax(axis=-1)
    confidence_np = probs_np.max(axis=-1)   # calibrated max-softmax as confidence

    tau, gamma = _derive_alert_thresholds(
        confidence=np.max(F.softmax(torch.tensor(logits_np) / max(T_opt, 0.05), dim=-1).numpy(), axis=-1),
        dir_probs=F.softmax(torch.tensor(logits_np) / max(T_opt, 0.05), dim=-1).numpy(),
    )

    metrics = _compute_metrics(
        preds_np, eval_labels_np, confidence_np, eval_ret_np, probs_np,
        tau=tau, gamma=gamma,
    )
    metrics["temperature"] = T_opt
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_selective_forecasting(train_loader, val_loader, eval_loader, args,
                               target_coverage: float = 0.30) -> Dict:
    """Baseline 12 — Selective Forecasting (inspired by Brusokas et al. [8]).

    ── Protocol ─────────────────────────────────────────────────────────────────
    Step 1: Train AllNewsFusion (news+market, uniform attention, no factor head).
    Step 2: On validation split, record max(softmax(dir_logits)) per sample.
    Step 3: τ* = percentile(max_prob, 100·(1-target_coverage)) — chosen so that
            exactly target_coverage fraction of validation samples pass the gate.
    Step 4: Apply τ*=γ* on held-out evaluation split; report full PDF Table-4
            metrics plus alert_coverage.

    ── Distinction from Baseline 4 (Raw-Probability Threshold) ────────────────
    Baseline 4 uses MARKET-ONLY MLP + default alert threshold (no coverage goal).
    Baseline 12 uses NEWS+MARKET fusion + COVERAGE-CONSTRAINED percentile
    threshold tuned on validation. The coverage constraint is the selective-
    forecasting contribution: it enforces a fixed operational alert rate so
    downstream comparison against SAFE-Alert is conducted at equal coverage,
    isolating *quality-per-alert* from *how often to alert*.

    ── Deviation from Brusokas Time-Energy Model ──────────────────────────────
    Brusokas et al. [8] use an Energy-Based Model (EBM) that is trained jointly
    with the forecaster to estimate prediction quality; tau* is chosen on the
    energy score. Our implementation substitutes **max-softmax** as the quality
    signal because SAFE-Alert's task is 3-class direction classification, not
    time-series regression. Max-softmax is a standard, strong selective-
    prediction signal for classifiers (Hendrycks & Gimpel 2017); training a
    separate EBM is Brusokas-specific to regression and not informative here.
    The COVERAGE-CONSTRAINED TUNING PROCEDURE, however, is preserved verbatim.

    ── Contrast with SAFE-Alert ───────────────────────────────────────────────
    Baseline 12 picks τ* POST-HOC on validation.
    SAFE-Alert jointly optimises τ_h end-to-end via:
      - L_cal  (Eq.35) — confidence calibration
      - L_risk (Eq.37) — coverage-aware selective risk
      - L_sel  (Eq.34) — evidence-selection regularisation
      - Threshold calibration grid search (Section 3.7.1) with multi-objective
        penalty on coverage + Sharpe + precision.
    Expected: SAFE-Alert beats Baseline 12 at equal target coverage on the
    Sharpe / Precision / ECE trio, confirming end-to-end joint optimization
    adds value over post-hoc abstention.

    Args:
        target_coverage: Desired alert fraction at deployment (default 0.30).
            Paper default is 0.35 (κ); we set 0.30 here to create a slightly
            more selective baseline and widen the margin for comparison.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 12: Selective Forecasting (coverage-optimized abstaining)")
    print(f"  AllNewsFusion + threshold tau* for {target_coverage:.0%} target coverage.")
    print(f"{'='*60}")

    # Step 1: Train AllNewsFusion base model
    model     = AllNewsFusionMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_news_market(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  [Base] Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    # Step 2: Evaluate on validation set for threshold tuning
    _, _, _, _, dir_probs = _evaluate_news_market(
        model, val_loader, args.device
    )

    # Step 3: Coverage-constrained threshold
    max_probs = dir_probs.max(axis=-1)           # (N,) model confidence via max softmax
    tau_star  = float(np.percentile(max_probs, (1.0 - target_coverage) * 100))
    actual_cov = float((max_probs >= tau_star).mean())
    print(f"  [Threshold] tau* = {tau_star:.4f} | actual_coverage = {actual_cov:.3f} "
          f"(target = {target_coverage:.2f})")

    # Step 4: Evaluate tuned threshold on held-out evaluation split
    preds, labels, _, ret_labels, dir_probs = _evaluate_news_market(
        model, eval_loader, args.device
    )
    confidence = dir_probs.max(axis=-1)
    metrics = _compute_metrics(
        preds, labels, confidence, ret_labels, dir_probs,
        tau=tau_star, gamma=tau_star,
    )
    metrics["abstain_threshold"] = tau_star
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    print(f"           AlertCov={metrics['alert_coverage']:.4f}  AUC={metrics['auc']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def run_current_price_predictor(train_loader, eval_loader, args) -> Dict:
    """Baseline 13: Production-style price-only momentum fallback.

    Uses a simple momentum proxy from market features to emulate a lightweight
    price predictor without news, explanation, or selective gating.
    """
    print(f"\n{'='*60}")
    print("  BASELINE 13: Current Price Predictor (price-only fallback)")
    print("  Momentum threshold rule from market features, no news input.")
    print(f"{'='*60}")

    train_signal = []
    for batch in train_loader:
        mf = batch["market_features"]
        train_signal.append(mf[:, 0].cpu().numpy())
    train_signal = np.concatenate(train_signal)

    q_low = float(np.percentile(train_signal, 33))
    q_high = float(np.percentile(train_signal, 67))
    center = float(np.median(train_signal))
    spread = float(np.std(train_signal) + 1e-8)

    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []
    for batch in eval_loader:
        signal = batch["market_features"][:, 0].cpu().numpy()
        labels = batch["direction"].numpy()
        rets = batch["return"].numpy()

        preds = np.full(signal.shape[0], 1, dtype=np.int64)  # neutral
        preds[signal <= q_low] = 0
        preds[signal >= q_high] = 2

        dist = np.abs(signal - center) / spread
        conf = np.clip(0.5 + 0.25 * dist, 0.5, 0.99).astype(np.float32)

        probs = np.full((signal.shape[0], 3), 0.1, dtype=np.float32)
        probs[:, 1] = 0.8
        probs[preds == 0] = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        probs[preds == 2] = np.array([0.1, 0.2, 0.7], dtype=np.float32)

        preds_l.append(preds)
        labels_l.append(labels)
        conf_l.append(conf)
        ret_l.append(rets)
        probs_l.append(probs)

    preds = np.concatenate(preds_l)
    labels = np.concatenate(labels_l)
    confidence = np.concatenate(conf_l)
    ret_labels = np.concatenate(ret_l)
    dir_probs = np.concatenate(probs_l)

    tau, gamma = _derive_alert_thresholds(confidence, dir_probs)
    metrics = _compute_metrics(
        preds, labels, confidence, ret_labels, dir_probs,
        tau=tau, gamma=gamma,
    )
    metrics["q_low"] = q_low
    metrics["q_high"] = q_high

    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    return metrics


def run_all_news_llm_expl(train_loader, eval_loader, args) -> Dict:
    """Baseline 14: All-News + post-hoc LLM explanation.

    Forecasting path is identical to all-news fusion; explanation is treated as
    post-hoc and does not alter predictions.
    """
    metrics = _run_news_market_baseline(
        AllNewsFusionMLP, 14,
        "All-News + LLM Explanation",
        "All-news forecast with post-hoc LLM explanation (prediction unchanged).",
        train_loader, eval_loader, args,
    )
    metrics["posthoc_explanation"] = "llm_template"
    return metrics


# =============================================================================
# DATA LOADING (mirrors train_safe_alert.py)
# =============================================================================

def load_data(args):
    script_file   = Path(__file__).resolve()
    service_root  = script_file.parent.parent.parent.parent

    default_data = service_root / "training_data"
    default_data_v2 = default_data / "v2"
    data_root = args.data_path or default_data
    emb_root  = args.embeddings_path or default_data
    if args.data_path is None and default_data_v2.exists():
        data_root = default_data_v2
    if args.embeddings_path is None and default_data_v2.exists():
        emb_root = default_data_v2

    # Candles
    symbol_upper = args.symbol.upper()
    for candidate in [
        data_root / f"{symbol_upper}_{args.horizon}_ohlcv.csv",
        data_root / "candles_max.csv",
        data_root / "candles_aligned.csv",
        data_root / f"{args.symbol.lower()}_training_dataset_v2.csv",
    ]:
        if candidate.exists():
            candle_df = pd.read_csv(candidate)
            print(f"[OK] Candles: {candidate.name} ({len(candle_df)} rows)")
            break
    else:
        logger.error("No candle CSV found in %s", data_root)
        sys.exit(1)

    # Embeddings
    embeddings = None
    for candidate in [
        emb_root / "btcusdt_article_embeddings_max.npy",
        emb_root / f"{args.symbol.lower()}_article_embeddings.npy",
        emb_root / "article_embeddings.npy",
    ]:
        if candidate.exists():
            embeddings = np.load(candidate)
            print(f"[OK] Embeddings: {candidate.name} {embeddings.shape}")
            break
    if embeddings is None:
        logger.error("No embeddings .npy found in %s", emb_root)
        sys.exit(1)

    # Articles
    articles_df = None
    for candidate in [
        emb_root / "articles_max.csv",
        emb_root / "articles.csv",
    ]:
        if candidate.exists():
            articles_df = pd.read_csv(candidate)
            print(f"[OK] Articles: {candidate.name} ({len(articles_df)} rows)")
            break
    if articles_df is None:
        logger.error("No articles CSV found in %s", emb_root)
        sys.exit(1)

    # Precomputed market features (optional — 5-10x speedup)
    precomp = None
    pp = emb_root / "features_precomputed.npy"
    if pp.exists():
        precomp = np.load(pp)
        if len(precomp) != len(candle_df):
            if len(precomp) > len(candle_df):
                print(
                    "[WARN] features_precomputed.npy has more rows than filtered candles; "
                    f"truncating {len(precomp)} -> {len(candle_df)} to keep alignment."
                )
                precomp = precomp[:len(candle_df)]
            else:
                raise ValueError(
                    "features_precomputed.npy is shorter than filtered candles: "
                    f"got {len(precomp)} rows, expected {len(candle_df)}. "
                    "Re-run precompute_market_features.py."
                )
        print(f"[OK] Precomputed features: {pp.name} {precomp.shape}")

    # Factor labels (optional — required for Baseline 7)
    factor_labels = None
    fp = emb_root / "article_factor_labels.npy"
    if fp.exists():
        factor_labels = np.load(fp)
        print(f"[OK] Factor labels: {fp.name} {factor_labels.shape}")
    else:
        print(f"[INFO] Factor labels not found ({fp.name}) — Baseline 7 will use zero factors.")

    # Entity sentiment (optional, but required for META_DIM=14)
    entity_sentiment = None
    es_path = emb_root / "article_entity_sentiment.npy"
    if es_path.exists():
        entity_sentiment = np.load(es_path)
        print(f"[OK] Entity sentiment: {es_path.name} {entity_sentiment.shape}")

    return candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment


def build_loaders(candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment, args):
    from torch.utils.data import Subset, DataLoader
    import train_safe_alert as _ts_ref
    _eps_override = getattr(_ts_ref, '_EPSILON_H_OVERRIDE', None)

    print("[BUILD] Creating SAFEAlertDataset (indexing candles↔articles)...", flush=True)
    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=args.horizon,
        precomputed_features=precomp,
        factor_labels=factor_labels,
        entity_sentiment=entity_sentiment,
        epsilon_h_override=_eps_override,
    )
    print(f"[BUILD] Dataset ready: {len(dataset)} samples", flush=True)

    train_size = int(0.70 * len(dataset))
    val_size   = int(0.15 * len(dataset))

    # P0 #2: Fit the market-feature z-score scaler on train ONLY, then share
    # across val/test so the baselines see the SAME preprocessing distribution
    # SAFE-Alert training uses. Previously this call was missing here and all
    # 14 baselines trained on raw (unnormalized) features while SAFE-Alert
    # trained on z-scored features — an unfair preprocessing advantage that
    # biased the head-to-head comparison. Critical for paper fairness.
    dataset.fit_market_scaler(range(0, train_size))

    train_set = Subset(dataset, range(0, train_size))
    val_set   = Subset(dataset, range(train_size, train_size + val_size))
    test_set  = Subset(dataset, range(train_size + val_size, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"[OK] Split: train={len(train_set)}, val={len(val_set)}, "
          f"test (held out)={len(dataset) - train_size - val_size} samples")
    print(f"[OK] Market scaler fit on train split (baselines and SAFE-Alert use identical preprocessing).")
    return train_loader, val_loader, test_loader


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary_table(results: Dict[str, Dict]):
    cols = ["Baseline", "F1", "MCC", "AUC", "ECE", "Sharpe", "AlertCov", "Score"]
    widths = [30, 7, 7, 7, 7, 8, 9, 8]

    def _fmt(s, w):
        return str(s).ljust(w)[:w]

    sep    = "+" + "+".join("-" * w for w in widths) + "+"
    header = "|" + "|".join(_fmt(h, w) for h, w in zip(cols, widths)) + "|"

    print("\n" + sep)
    print(header)
    print(sep)

    for name, m in results.items():
        row = [
            name,
            f"{m.get('macro_f1',     0):.4f}",
            f"{m.get('mcc',          0):.4f}",
            f"{m.get('auc',          0):.4f}",
            f"{m.get('ece',          0):.4f}",
            f"{m.get('alert_sharpe', 0):.4f}",
            f"{m.get('alert_coverage',0):.4f}",
            f"{m.get('model_score',  0):.4f}",
        ]
        print("|" + "|".join(_fmt(v, w) for v, w in zip(row, widths)) + "|")

    print(sep)


# =============================================================================
# MAIN
def _run_k_select_news_baseline(
    baseline_num: int, name: str, desc: str,
    selection_mode: str,
    train_loader, eval_loader, args,
) -> Dict:
    """Runner for Random-K and Most-recent-K evidence baselines (15 & 16)."""
    import train_safe_alert as _ts
    k = _ts._TOP_K_MAP.get(args.horizon, 4)
    return _run_news_market_baseline(
        KSelectNewsFusionMLP,
        baseline_num=baseline_num, name=name, desc=desc,
        train_loader=train_loader, eval_loader=eval_loader, args=args,
        model_kwargs={"k": k, "selection_mode": selection_mode},
    )


# =============================================================================

_ALL_BASELINES = [
    "market_only",
    "all_news_fusion",
    "always_alert",
    "raw_prob_threshold",
    "sentiment_market",
    "nsm_style",
    "llm_factor",
    "sep_style",
    "finin_style",
    "interleaved",
    "temperature_scaled",
    "selective_forecasting",
    "current_price_predictor",
    "all_news_llm_expl",
    # Evidence-selection ablation baselines (CaiTien.md Section 4 + Section 6)
    # These use the same AllNewsFusion backbone but with different article selection:
    # random_k_evidence:    average of K random articles  (not most-relevant)
    # most_recent_k_evidence: average of K most-recent articles (not most-relevant)
    # Expected result: SAFE-Alert's selective attention > all three heuristics.
    "random_k_evidence",
    "most_recent_k_evidence",
    # Source-quality-proxy K-selection: rank articles by embedding L2 norm
    # (Baseline 17 — CaiTien §6: source-priority evidence selection)
    "source_priority_k_evidence",
    # Tabular ML baseline (CaiTien §4)
    "lightgbm_market_only",
    # Advanced time-series transformer baselines (CaiTien §4)
    "patchtst_market_only",
    "itransformer_market_only",
]


def _run_market_arch_baseline(
    model_class, baseline_num: int, name: str, desc: str,
    train_loader, eval_loader, args,
    model_kwargs: Optional[Dict] = None,
) -> Dict:
    """Generic runner for market-only architecture baselines (PatchTST, iTransformer).

    Uses _train_epoch_market_only / _evaluate_market_only so only market_features
    are passed to the model — article embeddings are never loaded.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE {baseline_num}: {name}")
    print(f"  {desc}")
    print(f"{'='*60}")
    model = model_class(**(model_kwargs or {})).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")
    preds, labels, confidence, ret_labels, dir_probs = _evaluate_market_only(
        model, eval_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f}  MCC={metrics['mcc']:.4f}  "
          f"ECE={metrics['ece']:.4f}  Sharpe={metrics['alert_sharpe']:.4f}  "
          f"Score={metrics['model_score']:.4f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _run_all_baselines(to_run, train_loader, val_loader, test_loader, args) -> Dict:
    """Run all requested baselines and return metrics dict."""
    results: Dict[str, Dict] = {}
    n_total = len(to_run)
    done = 0

    def _run(name, fn, *fn_args):
        nonlocal done
        done += 1
        print(f"\n[{done}/{n_total}] Starting baseline: {name}", flush=True)
        result = fn(*fn_args)
        f1 = result.get('macro_f1', 0)
        sh = result.get('alert_sharpe', 0)
        print(f"[{done}/{n_total}] Done {name}: F1={f1:.3f} Sharpe={sh:.3f}", flush=True)
        return result

    if "market_only" in to_run:
        results["market_only"] = _run("market_only", run_market_only, train_loader, test_loader, args)
    if "all_news_fusion" in to_run:
        results["all_news_fusion"] = _run("all_news_fusion", run_all_news_fusion, train_loader, test_loader, args)
    if "always_alert" in to_run:
        results["always_alert"] = _run("always_alert", run_always_alert, test_loader, args)
    if "raw_prob_threshold" in to_run:
        results["raw_prob_threshold"] = _run("raw_prob_threshold", run_raw_prob_threshold, train_loader, test_loader, args)
    if "sentiment_market" in to_run:
        results["sentiment_market"] = _run("sentiment_market", run_sentiment_market, train_loader, test_loader, args)
    if "nsm_style" in to_run:
        results["nsm_style"] = _run("nsm_style", run_nsm_style, train_loader, test_loader, args)
    if "llm_factor" in to_run:
        results["llm_factor"] = _run("llm_factor", run_llm_factor, train_loader, test_loader, args)
    if "sep_style" in to_run:
        results["sep_style"] = _run("sep_style", run_sep_style, train_loader, test_loader, args)
    if "finin_style" in to_run:
        results["finin_style"] = _run("finin_style", run_finin_style, train_loader, test_loader, args)
    if "interleaved" in to_run:
        results["interleaved"] = _run("interleaved", run_interleaved, train_loader, test_loader, args)
    if "temperature_scaled" in to_run:
        results["temperature_scaled"] = _run("temperature_scaled", run_temperature_scaled, train_loader, val_loader, test_loader, args)
    if "selective_forecasting" in to_run:
        results["selective_forecasting"] = _run("selective_forecasting", run_selective_forecasting,
            train_loader, val_loader, test_loader, args, args.target_coverage)
    if "current_price_predictor" in to_run:
        results["current_price_predictor"] = _run("current_price_predictor", run_current_price_predictor, train_loader, test_loader, args)
    if "all_news_llm_expl" in to_run:
        results["all_news_llm_expl"] = _run("all_news_llm_expl", run_all_news_llm_expl, train_loader, test_loader, args)
    if "random_k_evidence" in to_run:
        done += 1
        print(f"\n[{done}/{n_total}] Starting baseline: random_k_evidence", flush=True)
        r = _run_k_select_news_baseline(
            baseline_num=15, name="Random-K Evidence",
            desc="AllNewsFusion with K random articles per candle.",
            selection_mode="random",
            train_loader=train_loader, eval_loader=test_loader, args=args)
        print(f"[{done}/{n_total}] Done random_k_evidence: F1={r.get('macro_f1',0):.3f} Sharpe={r.get('alert_sharpe',0):.3f}", flush=True)
        results["random_k_evidence"] = r
    if "most_recent_k_evidence" in to_run:
        done += 1
        print(f"\n[{done}/{n_total}] Starting baseline: most_recent_k_evidence", flush=True)
        r = _run_k_select_news_baseline(
            baseline_num=16, name="Most-Recent-K Evidence",
            desc="AllNewsFusion with K most-recent articles per candle.",
            selection_mode="most_recent",
            train_loader=train_loader, eval_loader=test_loader, args=args)
        print(f"[{done}/{n_total}] Done most_recent_k_evidence: F1={r.get('macro_f1',0):.3f} Sharpe={r.get('alert_sharpe',0):.3f}", flush=True)
        results["most_recent_k_evidence"] = r
    if "source_priority_k_evidence" in to_run:
        done += 1
        print(f"\n[{done}/{n_total}] Starting baseline: source_priority_k_evidence", flush=True)
        r = _run_k_select_news_baseline(
            baseline_num=17, name="Source-Priority-K Evidence",
            desc="AllNewsFusion with K articles ranked by embedding L2 norm (source-quality proxy).",
            selection_mode="source_priority",
            train_loader=train_loader, eval_loader=test_loader, args=args)
        print(f"[{done}/{n_total}] Done source_priority_k_evidence: F1={r.get('macro_f1',0):.3f} Sharpe={r.get('alert_sharpe',0):.3f}", flush=True)
        results["source_priority_k_evidence"] = r
    if "lightgbm_market_only" in to_run:
        results["lightgbm_market_only"] = _run(
            "lightgbm_market_only", run_lightgbm_market_only,
            train_loader, test_loader, args)
    if "patchtst_market_only" in to_run:
        results["patchtst_market_only"] = _run_market_arch_baseline(
            PatchTSTMarketOnly,
            baseline_num=19, name="PatchTST (Market-Only)",
            desc="PatchTST-style patch transformer on 63-dim market features (Nie et al. 2023).",
            train_loader=train_loader, eval_loader=test_loader, args=args,
        )
    if "itransformer_market_only" in to_run:
        results["itransformer_market_only"] = _run_market_arch_baseline(
            iTransformerMarketOnly,
            baseline_num=20, name="iTransformer (Market-Only)",
            desc="iTransformer variate-attention transformer on 63-dim market features (Liu et al. 2024).",
            train_loader=train_loader, eval_loader=test_loader, args=args,
        )
    return results


def main():
    script_file  = Path(__file__).resolve()
    service_root = script_file.parent.parent.parent.parent
    default_data = service_root / "training_data"

    parser = ArgumentParser(description="SAFE-Alert baseline comparison (20 baselines)")
    parser.add_argument("--symbol",          default="BTCUSDT")
    parser.add_argument("--horizon",         default="1h", choices=["1h", "4h"])
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--output",          type=Path,  default=None,
                        help="Path for baseline_results.json (default: artifact_dir/...)")
    parser.add_argument("--target_coverage", type=float, default=0.30,
                        help="Target alert coverage for Baseline 12 (default 0.30 = 30%)")
    parser.add_argument("--baselines",       nargs="*",  default=None,
                        choices=_ALL_BASELINES,
                            help=f"Which baselines to run (default: all). "
                             f"Choices: {_ALL_BASELINES}")
    parser.add_argument("--walk_forward",    action="store_true", default=False,
                        help="Run baselines with walk-forward CV (same protocol as SAFE-Alert).")
    parser.add_argument("--n_folds",         type=int,   default=4)
    parser.add_argument("--embargo_steps",   type=int,   default=24)
    parser.add_argument("--data_path",       type=Path,  default=default_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_data)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "train_config_research_best.yaml",
                        help="YAML config (same as train_safe_alert.py). "
                             "Baselines 5-10 use SAFEAlertNet variants, so they "
                             "MUST honour the same YAML hyperparameters as the "
                             "full model for fair comparison — otherwise "
                             "baseline-vs-full-model deltas mix config noise "
                             "with architectural effect.")
    args = parser.parse_args()

    # Session 26 — apply YAML overrides so baselines use the same
    # λ schedule / top_k / faith_margin / ingest_delay / etc. as the
    # full model. Without this, run_baselines.py would silently use
    # module defaults even when user edited YAML, making comparisons
    # unreliable.
    try:
        import yaml as _yaml
        import train_safe_alert as _train
        if args.config and args.config.exists():
            with open(args.config) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            _train._apply_config_overrides(_cfg)
            print(f"[CONFIG] Loaded {args.config}")
        else:
            _train._apply_config_overrides({})
    except Exception as _e:
        print(f"[CONFIG WARN] Could not apply config overrides: {_e}")

    output_path = args.output or (args.artifact_dir / "baseline_results.json")
    to_run      = args.baselines if args.baselines else _ALL_BASELINES

    print(f"[START] SAFE-Alert Baseline Study ({len(to_run)} baselines)")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Horizon   : {args.horizon}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : {args.device}")
    print(f"  Baselines : {to_run}")

    candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment = load_data(args)

    if args.walk_forward:
        import train_safe_alert as _ts_ref_wf
        _eps_override_wf = getattr(_ts_ref_wf, '_EPSILON_H_OVERRIDE', None)
        from train_safe_alert import walk_forward_split
        from torch.utils.data import Subset, DataLoader
        from safe_alert_dataset import SAFEAlertDataset

        dataset = SAFEAlertDataset(
            candle_df=candle_df,
            article_embeddings=embeddings,
            article_meta=articles_df,
            article_to_candle={},
            symbol=args.symbol,
            horizon=args.horizon,
            precomputed_features=precomp,
            factor_labels=factor_labels,
            entity_sentiment=entity_sentiment,
            epsilon_h_override=_eps_override_wf,
        )
        folds = list(walk_forward_split(len(dataset), n_folds=args.n_folds,
                                        embargo_steps=args.embargo_steps))
        print(f"[WALK-FORWARD] {args.n_folds} folds | embargo={args.embargo_steps}")

        fold_results_all: Dict[str, list] = {}

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            print(f"\n{'='*60}")
            print(f"  FOLD {fold_idx}/{len(folds)} | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            print(f"{'='*60}")

            dataset.fit_market_scaler(train_idx)
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=args.batch_size, shuffle=False, num_workers=0)

            fold_results = _run_all_baselines(to_run, train_loader, val_loader, test_loader, args)
            for name, metrics in fold_results.items():
                fold_results_all.setdefault(name, []).append(metrics)

        # Aggregate across folds
        results = {}
        for name, fold_list in fold_results_all.items():
            agg = {}
            keys = [k for k in fold_list[0] if isinstance(fold_list[0][k], (int, float))]
            for k in keys:
                vals = [f[k] for f in fold_list if k in f]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            results[name] = agg
        print("\n[WALK-FORWARD] Aggregated over all folds.")
    else:
        train_loader, val_loader, test_loader = build_loaders(
            candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment, args
        )
        print("[INFO] All final baseline metrics are reported on held-out test split.")
        results = _run_all_baselines(to_run, train_loader, val_loader, test_loader, args)

    # ── Summary and save ─────────────────────────────────────────────────────
    print_summary_table(results)

    def _to_serialisable(d):
        return {k: float(v) if isinstance(v, (np.floating, np.integer, float)) else v
                for k, v in d.items()}

    json_results = {k: _to_serialisable(v) for k, v in results.items()}
    json_results["_meta"] = {
        "symbol":           args.symbol,
        "horizon":          args.horizon,
        "epochs":           args.epochs,
        "device":           args.device,
        "protocol":         f"walk_forward_{args.n_folds}fold" if args.walk_forward else "single_split_70_15_15",
        "final_eval_split": "test",
        "target_coverage":  args.target_coverage,
        "n_baselines_run":  len(results),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Baseline results saved: {output_path}")


if __name__ == "__main__":
    main()
