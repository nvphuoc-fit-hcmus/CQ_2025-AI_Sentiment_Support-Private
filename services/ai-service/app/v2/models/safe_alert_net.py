"""
SAFE-Alert Neural Network — match 100% PDF architecture.

Eq references match paper Section 3.

Architecture:
  Phi_sel  : SelectiveNewsEncoder (Eq. 8-13)
             - horizon-conditioned article encoding
             - W_m metadata embedding
             - temperature tau softmax
             - adaptive Top-Kh gating
  Phi_fac  : FactorModule (Eq. 14-15)
             - 10-class factor ontology
             - factor embedding matrix U
  Phi_mkt  : MultiTimescaleMarketEncoder (Eq. 16-19)
             - pseudo-timeframes from lag feature groups
             - horizon-conditioned attention beta_delta
  Phi_fus  : CrossAttentionFusion (Eq. 20-22)
             - dual cross-attention (news->mkt, fac->mkt)
             - h_emb in MLP input
  Phi_pred : Prediction heads (Eq. 23-25)
             - 3-class direction {-1,0,+1}
             - return regression
             - confidence sigmoid
  Phi_exp  : ExplanationModule (Eq. 27-29)
             - structured output
             - template NL explanation
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Constants ──
EMB_DIM    = 768
META_DIM   = 14   # [0-3]: recency_norm, length_norm, source_cred, novelty_norm
                  # [4-13]: target-based FSA entity sentiment per factor [-1,+1]
                  # Falls back to 4 dims if entity_sentiment not precomputed.

# --- Factor Ontology: Load from JSON if available ---
import json
from pathlib import Path

ONTOLOGY_PATH = Path(__file__).parent.parent / "factor_ontology.json"  # v2/factor_ontology.json
_DEFAULT_FACTOR_NAMES = [
    "institutional_inflow", "etf_flow", "regulatory_easing",
    "regulatory_tightening", "exchange_risk", "liquidity_squeeze",
    "whale_accumulation", "macro_uncertainty", "protocol_upgrade",
    "network_outage",
]
_DEFAULT_FACTOR_KEYWORDS = {
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

try:
    if ONTOLOGY_PATH.exists():
        with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
            _ontology = json.load(f)
        FACTOR_NAMES = _ontology["FACTOR_NAMES"]
        FACTOR_KEYWORDS = _ontology["FACTOR_KEYWORDS"]
    else:
        FACTOR_NAMES = _DEFAULT_FACTOR_NAMES
        FACTOR_KEYWORDS = _DEFAULT_FACTOR_KEYWORDS
except Exception as e:
    FACTOR_NAMES = _DEFAULT_FACTOR_NAMES
    FACTOR_KEYWORDS = _DEFAULT_FACTOR_KEYWORDS

FACTOR_CLASSES = len(FACTOR_NAMES)
N_TIMEFRAMES = 5  # Eq.16-19 spec: 5 timeframes (1m, 5m, 15m, 1h, 4h)
HORIZON_VOCAB = {"1h": 0, "4h": 1, "24h": 2}  # horizon embedding (PDF: h ∈ {15m,1h,4h,24h})
HORIZON_EMB_DIM = 16
MARKET_DIM = 63  # kept for backward compat


class ArticleEncoder(nn.Module):
    """Eq.8: e_i = Phi_news(x_i, m_i, h) — horizon-conditioned article encoding.
    PDF includes m_i (metadata) as part of the encoder input, not just the scorer.
    """
    def __init__(self, emb_dim: int = EMB_DIM, meta_dim: int = META_DIM,
                 article_dim: int = 128,
                 horizon_emb_dim: int = HORIZON_EMB_DIM, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim + meta_dim + horizon_emb_dim, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, article_dim), nn.LayerNorm(article_dim),
            nn.GELU(),  # non-linearity prevents representation collapse
        )

    def forward(self, emb: torch.Tensor, meta: torch.Tensor,
                h_emb: torch.Tensor) -> torch.Tensor:
        """
        emb:  (B, K, 768)
        meta: (B, K, META_DIM)   [recency, length, source_cred, novelty]
        h_emb:(B, H)
        Returns: (B, K, article_dim)
        """
        B, K, _ = emb.shape
        h_exp = h_emb.unsqueeze(1).expand(B, K, -1)
        return self.net(torch.cat([emb, meta, h_exp], dim=-1))  # (B,K,article_dim)


class SelectiveAttention(nn.Module):
    """
    Eq.9-13: Scoring + Top-Kh gating.
    a_i = v^T tanh(W_e*e_i + W_q*q + W_m*m_i)
    alpha = softmax(a/tau) * mask
    alpha_tilde = alpha * I(i in TopK(a, Kh))
    Z_news = sum alpha_tilde_i e_i
    """
    def __init__(self, article_dim: int = 128, query_dim: int = 64,
                 meta_dim: int = META_DIM, tau_init: float = 1.0):
        super().__init__()
        self.W_e  = nn.Linear(article_dim, query_dim, bias=False)
        self.W_q  = nn.Linear(query_dim,   query_dim, bias=False)
        self.W_m  = nn.Linear(meta_dim,    query_dim, bias=False)
        self.v    = nn.Linear(query_dim, 1, bias=False)
        self.tau  = nn.Parameter(torch.tensor(tau_init))

    def forward(self, articles: torch.Tensor, query: torch.Tensor,
                mask: torch.Tensor, meta: torch.Tensor,
                K_h: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        articles: (B,K,article_dim)
        query:    (B,query_dim)
        mask:     (B,K) bool
        meta:     (B,K,meta_dim)
        K_h:      int top-K for this horizon
        Returns: z_news (B,article_dim), attn_weights (B,K), a_scores (B,K)
        """
        q_exp = query.unsqueeze(1)  # (B,1,query_dim)
        scores = self.v(torch.tanh(
            self.W_e(articles) + self.W_q(q_exp) + self.W_m(meta)
        )).squeeze(-1)  # (B,K)

        # Temperature softmax with padding mask (Eq.11)
        # Numerical stability: use finfo().min/2 instead of -1e9 for masked_fill
        scores_masked = scores.masked_fill((mask == 0).bool(), torch.finfo(scores.dtype).min / 2)
        alpha = F.softmax(scores_masked / self.tau.clamp(min=0.1), dim=-1)
        # Zero out padding positions (exp(min/tau) ≈ 0 but clamp for safety)
        alpha = alpha * mask.float()

        # Soft Top-Kh gating (Eq.12): I(i in TopK(a, K_h)) using raw scores a,
        # not post-softmax alpha — matches PDF notation exactly.
        if K_h > 0 and K_h < mask.shape[1]:
            topk_vals, _ = scores_masked.topk(K_h, dim=-1)
            threshold = topk_vals[:, -1:].detach()
            # Use >= (not >) to include ties at the K_h threshold value
            alpha_tilde = alpha * (scores_masked >= threshold - 1e-5).float()
        else:
            alpha_tilde = alpha

        z_news = (alpha_tilde.unsqueeze(-1) * articles).sum(dim=1)  # (B,article_dim)
        return z_news, alpha_tilde, scores


class FactorModule(nn.Module):
    """
    Eq.14-15:
    p^fac_i = softmax(W_fac e_i + b_fac)
    Z^fac = sum alpha_tilde_i U p^fac_i
    """
    def __init__(self, article_dim: int = 128, n_factors: int = FACTOR_CLASSES,
                 factor_emb_dim: int = 64):
        super().__init__()
        self.W_fac = nn.Linear(article_dim, n_factors)
        self.U     = nn.Linear(n_factors, factor_emb_dim, bias=False)  # U in R^{d_f x C}
        self.n_factors = n_factors

    def forward(self, articles: torch.Tensor,
                alpha_tilde: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        articles:    (B,K,article_dim)
        alpha_tilde: (B,K)
        Returns: z_fac (B,factor_emb_dim), p_fac (B,K,C)
        """
        p_fac = F.softmax(self.W_fac(articles), dim=-1)  # (B,K,C) Eq.14
        weighted = (alpha_tilde.unsqueeze(-1) * p_fac).sum(dim=1)  # (B,C)
        z_fac = self.U(weighted)  # (B,factor_emb_dim) Eq.15
        return z_fac, p_fac


class MultiTimescaleMarketEncoder(nn.Module):
    """
    Eq.16-19: Multi-timescale market encoder with horizon-conditioned attention.

    Feature layout in 63-dim vector (from market_features.py):
        dims  0-11  : 1m  timeframe (12 indicators)
        dims 12-23  : 5m  timeframe (12 indicators)
        dims 24-35  : 15m timeframe (12 indicators)
        dims 36-47  : 1h  timeframe (12 indicators)
        dims 48-59  : 4h  timeframe (12 indicators)
        dims 60-62  : cross-timeframe aggregates (RSI_agg, Momentum, Volume_trend)

    Each per-timeframe encoder receives its 12 features + 3 cross-timeframe = 15 dims,
    so all 5 encoders see different data (true multi-timescale, not the same vector 5×).
    """
    # Feature layout constants
    TF_FEATURES   = 12   # per-timeframe features (12 indicators each)
    CROSS_FEATURES = 3   # cross-timeframe features (dims 60-62)
    ENCODER_DIM    = TF_FEATURES + CROSS_FEATURES  # = 15

    def __init__(self, market_dim: int = 63, n_timeframes: int = N_TIMEFRAMES,
                 dm: int = 64, horizon_emb_dim: int = HORIZON_EMB_DIM):
        super().__init__()
        self.n_timeframes = n_timeframes   # was hardcoded to 5, now uses parameter
        self.dm = dm

        # Per-timeframe encoder (Eq.17): each gets 15-dim input (12 tf + 3 cross)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.ENCODER_DIM, dm), nn.LayerNorm(dm), nn.GELU(),
            ) for _ in range(n_timeframes)
        ])

        # Horizon-conditioned attention weights (Eq.18)
        self.w_delta = nn.ModuleList([nn.Linear(dm, 1, bias=False) for _ in range(n_timeframes)])
        self.W_delta = nn.ModuleList([nn.Linear(dm, dm, bias=False) for _ in range(n_timeframes)])
        self.W_h     = nn.Linear(horizon_emb_dim, dm, bias=False)

    def _split_features(self, feat: torch.Tensor) -> list:
        """
        Split (B, 63) into 5 tensors of shape (B, 15) = 12 tf-specific + 3 cross.
        Each timeframe gets its own 12 features + the shared 3 cross-timeframe features.
        """
        cross = feat[:, 60:63]  # (B, 3) — shared cross-timeframe context
        chunks = []
        for i in range(self.n_timeframes):
            tf_feat = feat[:, i*12 : (i+1)*12]          # (B, 12)
            chunks.append(torch.cat([tf_feat, cross], dim=-1))  # (B, 15)
        return chunks

    def forward(self, market_feat_list, h_emb: torch.Tensor):
        """
        Eq.16-19: Horizon-weighted market encoding across 5 REAL timeframes.

        Args:
            market_feat_list: Either:
              - Single tensor (B, 63): split into 5×(B,15) by timeframe — REAL multi-timescale
              - List of 5 tensors (B, 15): already split (e.g. from live inference)
            h_emb: (B, horizon_emb_dim)

        Returns:
            Z_mkt: (B, dm)     — horizon-weighted aggregate (for MLP concat)
            u_stack: (B, 5, dm) — per-timeframe vectors (K/V for CrossAttentionFusion)
        """
        # Split single 63-dim tensor into 5 real timeframe chunks
        if isinstance(market_feat_list, torch.Tensor):
            market_feat_list = self._split_features(market_feat_list)

        # Eq.17: Encode each timeframe independently (different features = different encodings)
        u_list = []
        for feat_tau, enc in zip(market_feat_list, self.encoders):
            u_list.append(enc(feat_tau))  # (B, dm)

        # Eq.18: Horizon-conditioned attention β_δ
        h_proj = self.W_h(h_emb)  # (B, dm)
        beta_logits = []
        for i, u_tau in enumerate(u_list):
            score = self.w_delta[i](torch.tanh(self.W_delta[i](u_tau) + h_proj))  # (B, 1)
            beta_logits.append(score)

        beta = F.softmax(torch.cat(beta_logits, dim=-1), dim=-1)  # (B, 5)

        # Eq.19: Weighted sum Z_mkt = Σ β_δ · u_δ
        Z_mkt = sum(beta[:, i:i+1] * u_list[i] for i in range(self.n_timeframes))  # (B, dm)

        # Stack individual timeframe encodings for use as K/V in CrossAttentionFusion.
        # CrossAttn(Q=news, K=u_stack, V=u_stack) attends to the 5 market timeframes
        # that are most relevant given the news context — meaningful cross-attention
        # instead of single-token K/V where softmax(1 element)=1 → output=V always.
        u_stack = torch.stack(u_list, dim=1)  # (B, 5, dm)

        return Z_mkt, u_stack


class CrossAttentionFusion(nn.Module):
    """
    Eq.20-22:
    Z_tilde_news = CrossAttn(Z_news, Z_mkt)
    Z_tilde_fac  = CrossAttn(Z_fac, Z_mkt)
    Z_fus  = MLP([Z_tilde_news; Z_tilde_fac; Z_mkt; h_emb])
    """
    def __init__(self, news_dim: int = 128, fac_dim: int = 64, mkt_dim: int = 64,
                 horizon_emb_dim: int = HORIZON_EMB_DIM, out_dim: int = 64, nhead: int = 4):
        super().__init__()
        # Cross-attention: Q=proj(Z_news), K/V=Z_mkt (Eq.20)
        # proj_news first maps news_dim→mkt_dim so all three heads operate in mkt_dim space
        self.proj_news = nn.Linear(news_dim, mkt_dim)
        self.ca_news = nn.MultiheadAttention(embed_dim=mkt_dim, num_heads=nhead,
                                              kdim=mkt_dim, vdim=mkt_dim, batch_first=True)

        # Cross-attention: Q=proj(Z_fac), K/V=Z_mkt (Eq.21)
        # proj_fac maps fac_dim→mkt_dim so all attention operates in mkt_dim space
        self.proj_fac = nn.Linear(fac_dim, mkt_dim)
        fac_heads = min(nhead, mkt_dim // 16)
        fac_heads = max(fac_heads, 1)
        self.ca_fac  = nn.MultiheadAttention(embed_dim=mkt_dim, num_heads=fac_heads,
                                              kdim=mkt_dim, vdim=mkt_dim, batch_first=True)

        fused_in = mkt_dim + mkt_dim + mkt_dim + horizon_emb_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_in, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, out_dim), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, z_news: torch.Tensor, z_fac: torch.Tensor,
                z_mkt: torch.Tensor, h_emb: torch.Tensor,
                u_stack: torch.Tensor = None) -> torch.Tensor:
        """
        z_news:  (B, news_dim)
        z_fac:   (B, fac_dim)
        z_mkt:   (B, mkt_dim)   — aggregated market (for MLP concat)
        h_emb:   (B, horizon_emb_dim)
        u_stack: (B, 5, mkt_dim) — per-timeframe encodings used as K/V

        Using u_stack (5 tokens) as K/V instead of z_mkt (1 token) makes
        attention meaningful: softmax over 5 keys gives non-trivial weights,
        so the attended output depends on Q (news/fac), not just V.
        Previously z_mkt.unsqueeze(1) → 1 token → softmax([scalar])=1.0 → output=V=z_mkt
        regardless of Q — news contribution was zero.
        """
        # K/V: use per-timeframe encodings if available, else fall back to z_mkt
        if u_stack is not None:
            kv = u_stack  # (B, 5, mkt_dim) — 5 meaningful K/V tokens
        else:
            kv = z_mkt.unsqueeze(1)  # (B, 1, mkt_dim) — fallback (legacy)

        # Eq.20: CrossAttn(Z_news, Z_mkt_sequence) — news attends to market timeframes
        z_news_p = self.proj_news(z_news).unsqueeze(1)  # (B, 1, mkt_dim)
        z_news_t, _ = self.ca_news(z_news_p, kv, kv)   # (B, 1, mkt_dim)
        z_news_t = z_news_t.squeeze(1)                   # (B, mkt_dim)

        # Eq.21: CrossAttn(Z_fac, Z_mkt_sequence) — factors attend to market timeframes
        z_fac_p = self.proj_fac(z_fac).unsqueeze(1)    # (B, 1, fac_dim→mkt_dim)
        z_fac_t, _ = self.ca_fac(z_fac_p, kv, kv)     # (B, 1, mkt_dim)
        z_fac_t = z_fac_t.squeeze(1)                    # (B, mkt_dim)

        # Eq.22: MLP([Z_tilde_news; Z_tilde_fac; Z_mkt; h_emb])
        cat = torch.cat([z_news_t, z_fac_t, z_mkt, h_emb], dim=-1)
        return self.mlp(cat)  # (B, out_dim)


class ExplanationModule(nn.Module):
    """
    Eq.27-29: Structured + language explanation.
    Returns structured dict and template NL string.
    """
    def forward(self, alpha_tilde: torch.Tensor, p_fac: torch.Tensor,
                article_meta: list[dict], K_h: int, P: int = 3,
                direction: int = 1, confidence: float = 0.0,
                horizon: str = "1h") -> dict:
        """
        alpha_tilde: (K,) attention weights for single sample
        p_fac:       (K, C) factor distributions
        article_meta: list of {"title":...} dicts
        Returns explanation dict
        """
        K = alpha_tilde.shape[0]

        # Eq.27: Selected news S^news = {n_i | i in TopK(a, Kh)}
        selected_indices = torch.nonzero(alpha_tilde > 0, as_tuple=False).squeeze(-1)
        if selected_indices.numel() == 0:
            selected_news = []
        else:
            selected_news = [
                article_meta[i]["title"] if i < len(article_meta) else ""
                for i in selected_indices.tolist()
            ]

        # Eq.28: Top-P factors S^fac = TopP(sum alpha_tilde_i p^fac_i, P)
        avg_factor = (alpha_tilde.unsqueeze(-1) * p_fac).sum(dim=0)  # (C,)
        if selected_indices.numel() == 0:
            top_factors = []
        else:
            topP_indices = avg_factor.topk(min(P, len(FACTOR_NAMES))).indices.tolist()
            top_factors = [FACTOR_NAMES[i] for i in topP_indices]

        # Eq.29: Phi_txt — template NL explanation
        dir_str = {0: "giam (DOWN)", 1: "trung tinh (NEUTRAL)", 2: "tang (UP)"}.get(direction, "?")
        factor_str = " va ".join(top_factors[:2]) if top_factors else "khong ro"
        nl = (f"Tin hieu: BTC co kha nang {dir_str} trong {horizon} toi. "
              f"Do tin cay: {confidence:.2f}. "
              f"Bang chung chinh: {'; '.join(selected_news[:2]) if selected_news else 'N/A'}. "
              f"Yeu to chi phoi: {factor_str}.")

        return {
            "selected_news":  selected_news,
            "top_factors":    top_factors,
            "factor_dist":    avg_factor.tolist(),
            "nl_explanation": nl,
        }


class SAFEAlertNet(nn.Module):
    """
    Full SAFE-Alert as per PDF Section 3.
    horizon: "1h", "4h", or "24h" -- determines K_h and horizon embedding
    """
    def __init__(self, market_dim: int = MARKET_DIM, article_dim: int = 128,
                 query_dim: int = 64, n_factors: int = FACTOR_CLASSES,
                 has_news: bool = True, K_1h: int = 8, K_4h: int = 6, K_24h: int = 8):
        super().__init__()
        self.has_news    = has_news
        self.K_h_map     = {"1h": K_1h, "4h": K_4h, "24h": K_24h}
        self.article_dim = article_dim

        # Horizon embedding (learned)
        self.h_emb_table = nn.Embedding(len(HORIZON_VOCAB), HORIZON_EMB_DIM)

        # Market encoder (Eq.16-19)
        mkt_dm = 64
        self.market_enc = MultiTimescaleMarketEncoder(market_dim=market_dim, dm=mkt_dm)

        # Query projection for attention (Eq.9)
        self.query_proj = nn.Linear(mkt_dm + HORIZON_EMB_DIM, query_dim)

        if has_news:
            # Article encoder (Eq.8)
            self.article_enc = ArticleEncoder(article_dim=article_dim)
            # Selective attention (Eq.10-13)
            self.sel_attn = SelectiveAttention(article_dim=article_dim, query_dim=query_dim)
            # Factor module (Eq.14-15)
            fac_dim = 64
            self.factor_mod = FactorModule(article_dim=article_dim, n_factors=n_factors,
                                           factor_emb_dim=fac_dim)
            # Fusion (Eq.20-22)
            self.fusion = CrossAttentionFusion(news_dim=article_dim, fac_dim=fac_dim,
                                               mkt_dim=mkt_dm, out_dim=64)
            fused_dim = self.fusion.out_dim
        else:
            fused_dim = 64

        # Market-only fallback (used when has_news=True but no articles)
        self.mkt_only = nn.Sequential(nn.Linear(mkt_dm, 64), nn.GELU())

        self.final_dropout = nn.Dropout(0.3)  # before prediction heads

        # Prediction heads (Eq.23-25)
        self.dir_head  = nn.Linear(fused_dim, 3)  # 3-class: DOWN(0), NEUTRAL(1), UP(2)
        self.ret_head  = nn.Linear(fused_dim, 1)  # return regression
        self.conf_head = nn.Linear(fused_dim, 1)  # confidence
        # Zero-init ret_head bias: prevents spurious positive offset before training
        nn.init.zeros_(self.ret_head.bias)

        # Explanation module (Eq.27-29)
        self.explainer = ExplanationModule()

    def forward(self, market_feat: torch.Tensor, horizon: str,
                article_emb: Optional[torch.Tensor] = None,
                article_mask: Optional[torch.Tensor] = None,
                article_meta_vec: Optional[torch.Tensor] = None,
                ablation: Optional[str] = None) -> dict:
        """
        market_feat:     (B, M)
        horizon:         "1h", "4h", or "24h"
        article_emb:     (B, K, 768)
        article_mask:    (B, K) bool
        article_meta_vec:(B, K, META_DIM)  [recency, length, source_cred, novelty]
        ablation:        one of None, "w/o_selective_news", "w/o_factor",
                         "w/o_market", "w/o_confidence", "w/o_horizon"
                         Controls which module is disabled for ablation study.
        """
        B = market_feat.shape[0]
        dev = market_feat.device

        # Horizon embedding — ablation w/o_horizon: replace with zeros
        h_idx = torch.tensor([HORIZON_VOCAB.get(horizon, 0)] * B, device=dev)
        h_emb = self.h_emb_table(h_idx)  # (B, HORIZON_EMB_DIM)
        if ablation == "w/o_horizon":
            h_emb = torch.zeros_like(h_emb)

        # Market encoding (Eq.16-19)
        z_mkt, u_stack = self.market_enc(market_feat, h_emb)  # (B, dm), (B, 5, dm)

        # ablation w/o_market: replace market encoding with zeros
        if ablation == "w/o_market":
            z_mkt = torch.zeros_like(z_mkt)
            u_stack = torch.zeros_like(u_stack)

        # Query (Eq.9)
        query = self.query_proj(torch.cat([z_mkt, h_emb], dim=-1))  # (B, query_dim)

        attn_w = factor_z = p_fac_all = None
        selected_mask = None

        if self.has_news and article_emb is not None and article_mask is not None:
            K_h = self.K_h_map.get(horizon, 3)

            # Metadata features (needed by both encoder Eq.8 and scorer Eq.10)
            if article_meta_vec is None:
                article_meta_vec = torch.zeros(B, article_emb.shape[1], META_DIM, device=dev)

            # Article encoding (Eq.8): e_i = Phi_news(x_i, m_i, h)
            e_i = self.article_enc(article_emb, article_meta_vec, h_emb)  # (B,K,article_dim)

            if ablation == "w/o_selective_news":
                # Average ALL articles (uniform weights), no Top-K gating
                mask_float = article_mask.float()  # (B, K)
                denom = mask_float.sum(dim=1, keepdim=True).clamp(min=1e-8)
                alpha_tilde = mask_float / denom   # uniform over valid articles
                z_news = (alpha_tilde.unsqueeze(-1) * e_i).sum(dim=1)  # (B, article_dim)
                selected_mask = article_mask.bool()
            else:
                # Selective attention (Eq.10-13)
                z_news, alpha_tilde, _ = self.sel_attn(e_i, query, article_mask,
                                                        article_meta_vec, K_h)
                selected_mask = alpha_tilde > 0

            if ablation == "w/o_factor":
                # Disable factor module: set z_fac = zeros
                z_fac = torch.zeros(B, self.factor_mod.U.out_features, device=dev)
                p_fac_all = torch.zeros(B, article_emb.shape[1], self.factor_mod.n_factors, device=dev)
            else:
                # Factor module (Eq.14-15)
                z_fac, p_fac_all = self.factor_mod(e_i, alpha_tilde)

            # Mask out samples with no articles
            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news = z_news * has_any

            # Fusion (Eq.20-22): pass u_stack so CrossAttn uses 5 K/V tokens (not 1)
            fused = self.fusion(z_news, z_fac, z_mkt, h_emb, u_stack=u_stack)
            attn_w = alpha_tilde
            factor_z = z_fac
        else:
            fused = self.mkt_only(z_mkt)

        fused = self.final_dropout(fused)

        # Prediction heads (Eq.23-25)
        dir_logits = self.dir_head(fused)                              # (B,3)
        ret_pred   = self.ret_head(fused).squeeze(-1)                    # (B,) plain linear — Eq.24: r̂ = W_r Z^fus + b_r
        confidence = torch.sigmoid(self.conf_head(fused)).squeeze(-1)  # (B,) in [0,1]

        # ablation w/o_confidence: replace confidence with fixed 0.5
        if ablation == "w/o_confidence":
            confidence = torch.full_like(confidence, 0.5)

        return {
            "dir_logits":   dir_logits,
            "ret_pred":     ret_pred,
            "confidence":   confidence,
            "attn_weights": attn_w,
            "selected_mask": selected_mask,
            "p_fac_all":    p_fac_all,
        }

    def forward_masked(self, market_feat: torch.Tensor, horizon: str,
                       article_emb: Optional[torch.Tensor] = None,
                       article_mask: Optional[torch.Tensor] = None,
                       article_meta_vec: Optional[torch.Tensor] = None,
                       ablation: Optional[str] = None) -> dict:
        """Forward pass with selected articles removed (for Lfaith loss, Eq.36).

        Computes p̂_masked by redistributing attention to unselected articles only,
        so the model predicts "as if top-K selected articles were absent".

        Returns: dict with "dir_logits" key (for masked prediction)
        """
        B = market_feat.shape[0]
        dev = market_feat.device

        # Horizon embedding
        h_idx = torch.tensor([HORIZON_VOCAB.get(horizon, 0)] * B, device=dev)
        h_emb = self.h_emb_table(h_idx)  # (B, HORIZON_EMB_DIM)

        # Market encoding
        z_mkt, u_stack = self.market_enc(market_feat, h_emb)  # (B, dm), (B, 5, dm)

        # Query
        query = self.query_proj(torch.cat([z_mkt, h_emb], dim=-1))  # (B, query_dim)

        if self.has_news and article_emb is not None and article_mask is not None:
            K_h = self.K_h_map.get(horizon, 3)

            # Metadata features
            if article_meta_vec is None:
                article_meta_vec = torch.zeros(B, article_emb.shape[1], META_DIM, device=dev)

            # Article encoding
            e_i = self.article_enc(article_emb, article_meta_vec, h_emb)  # (B,K,article_dim)

            # Selective attention (but then MASK it for masked prediction)
            z_news, alpha_tilde, _ = self.sel_attn(e_i, query, article_mask,
                                                    article_meta_vec, K_h)

            # Lfaith (Eq.36): predict WITHOUT the articles selected by top-K.
            # Step 1 — identify which articles were selected (alpha_tilde > 0).
            # Step 2 — redistribute attention weight only to UNSELECTED articles,
            #           renormalizing so weights still sum to 1 (proper simplex).
            # This gives the model's prediction "as if selected articles were absent",
            # not "with selected articles zeroed while keeping their weight" (which
            # would make z_news ≈ 0 and conflate "no articles" with "other articles").
            selection_mask = (alpha_tilde > 0).float()          # (B, K): 1 = selected
            alpha_unsel    = alpha_tilde * (1.0 - selection_mask)  # zero out selected
            alpha_renorm   = alpha_unsel / (                        # renormalize to simplex
                alpha_unsel.sum(dim=1, keepdim=True).clamp(min=1e-8)
            )

            z_news_masked = (alpha_renorm.unsqueeze(-1) * e_i).sum(dim=1)  # (B, article_dim)
            z_news = z_news_masked

            # Factor module on unselected articles with renormalized attention
            z_fac, _ = self.factor_mod(e_i, alpha_renorm)

            # Mask out samples with no articles
            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news = z_news * has_any

            # Fusion — pass u_stack (5 K/V tokens) for meaningful cross-attention
            fused = self.fusion(z_news, z_fac, z_mkt, h_emb, u_stack=u_stack)
        else:
            fused = self.mkt_only(z_mkt)

        fused = self.final_dropout(fused)

        # Prediction heads (masked)
        dir_logits_masked = self.dir_head(fused)  # (B, 3)

        return {"dir_logits": dir_logits_masked}

    def alert_decision(self, dir_logits: torch.Tensor, confidence: torch.Tensor,
                       horizon: str = "1h",
                       tau_h: float | None = None,
                       gamma_h: float | None = None) -> torch.Tensor:
        """Eq.26: A^{(h)} = 1 if ĉ >= tau_h AND max(p̂) >= gamma_h else 0

        Args:
            dir_logits: (B, 3) direction predictions
            confidence: (B,) model confidence [0, 1]
            horizon: "15m", "1h", "4h", "24h" (used for default thresholds)
            tau_h: Override confidence threshold (from policy file). Uses default if None.
            gamma_h: Override max-prob threshold (from policy file). Uses default if None.

        Returns:
            (B,) binary alerts
        """
        # Default thresholds per horizon (used only when not overridden by policy file)
        THRESHOLD_MAP = {
            "15m": {"tau": 0.80, "gamma": 0.75},
            "1h":  {"tau": 0.78, "gamma": 0.72},
            "4h":  {"tau": 0.75, "gamma": 0.70},
            "24h": {"tau": 0.72, "gamma": 0.68},
        }
        defaults = THRESHOLD_MAP.get(horizon, THRESHOLD_MAP["1h"])
        if tau_h is None:
            tau_h = defaults["tau"]
        if gamma_h is None:
            gamma_h = defaults["gamma"]

        # Eq.26: Both conditions must be true
        max_prob = F.softmax(dir_logits, dim=-1).max(dim=-1).values
        alert = ((confidence >= tau_h) & (max_prob >= gamma_h)).long()

        return alert


def compute_factor_pseudolabels(texts: list[str]) -> torch.Tensor:
    """
    Compute soft factor pseudo-labels from article text via keyword matching.
    Returns: (N, C) float tensor (softmax-normalised counts per factor).
    """
    C = len(FACTOR_NAMES)
    labels = torch.zeros(len(texts), C)
    for i, text in enumerate(texts):
        low = text.lower()
        counts = torch.zeros(C)
        for j, fname in enumerate(FACTOR_NAMES):
            kws = FACTOR_KEYWORDS.get(fname, [])
            counts[j] = sum(1 for kw in kws if kw in low)
        if counts.sum() == 0:
            counts[7] = 1.0  # default -> macro_uncertainty (most common "noise")
        labels[i] = counts / counts.sum()
    return labels
