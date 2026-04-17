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
HORIZON_VOCAB = {"15m": 0, "1h": 1, "4h": 2, "24h": 3}  # horizon embedding (PDF: h ∈ {15m,1h,4h,24h})
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
                horizon: str = "1h", symbol: str = "BTC") -> dict:
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
        nl = (f"Tin hieu: {symbol} co kha nang {dir_str} trong {horizon} toi. "
              f"Do tin cay: {confidence:.2f}. "
              f"Bang chung chinh: {'; '.join(selected_news[:2]) if selected_news else 'N/A'}. "
              f"Yeu to chi phoi: {factor_str}.")
        factors_text = ", ".join(top_factors) if top_factors else "khong ro"

        return {
            "selected_news":  selected_news,
            "top_factors":    top_factors,
            "factor_dist":    avg_factor.tolist(),
            "factors_text":   factors_text,
            "nl_explanation": nl,
        }


class SAFEAlertNet(nn.Module):
    """
    Full SAFE-Alert as per PDF Section 3.
    horizon: "15m", "1h", "4h", or "24h" -- determines K_h and horizon embedding
    """
    def __init__(self, market_dim: int = MARKET_DIM, article_dim: int = 128,
                 query_dim: int = 64, n_factors: int = FACTOR_CLASSES,
                 has_news: bool = True, K_15m: int = 3, K_1h: int = 4, K_4h: int = 5, K_24h: int = 8):
        super().__init__()
        self.has_news    = has_news
        self.K_h_map     = {"15m": K_15m, "1h": K_1h, "4h": K_4h, "24h": K_24h}
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

    # ── Shared encoder (Eq.8-19) ──────────────────────────────────────────────

    def _encode_common(
        self,
        market_feat: torch.Tensor,
        horizon: str,
        article_emb: Optional[torch.Tensor],
        article_mask: Optional[torch.Tensor],
        article_meta_vec: Optional[torch.Tensor],
        ablation: Optional[str] = None,
    ) -> dict:
        """Shared market + horizon + article encoding (Eq.8-19).

        Extracts the common prefix of forward() and forward_masked() to eliminate
        code duplication. Both callers receive identical intermediate representations
        and apply their own article-weighting logic on top.

        Returns dict keys: h_emb, z_mkt, u_stack, query, e_i, K_h, article_meta_vec.
        e_i and K_h are None when no articles are available.
        """
        B   = market_feat.shape[0]
        dev = market_feat.device

        # Horizon embedding (Eq.HORIZON_EMB) — w/o_horizon: zero out
        if horizon not in HORIZON_VOCAB:
            raise ValueError(f"Unknown horizon '{horizon}'. Must be one of {list(HORIZON_VOCAB.keys())}")
        h_idx = torch.tensor([HORIZON_VOCAB[horizon]] * B, device=dev)
        h_emb = self.h_emb_table(h_idx)              # (B, HORIZON_EMB_DIM)
        if ablation == "w/o_horizon":
            h_emb = torch.zeros_like(h_emb)

        # Market encoding (Eq.16-19) — w/o_market: zero out
        z_mkt, u_stack = self.market_enc(market_feat, h_emb)  # (B, dm), (B, 5, dm)
        if ablation == "w/o_market":
            z_mkt   = torch.zeros_like(z_mkt)
            u_stack = torch.zeros_like(u_stack)

        # Query projection (Eq.9)
        query = self.query_proj(torch.cat([z_mkt, h_emb], dim=-1))  # (B, query_dim)

        # Article encoding (Eq.8) — only when articles are present
        e_i = K_h = None
        if self.has_news and article_emb is not None and article_mask is not None:
            K_h = self.K_h_map.get(horizon, 3)
            if article_meta_vec is None:
                article_meta_vec = torch.zeros(B, article_emb.shape[1], META_DIM, device=dev)
            e_i = self.article_enc(article_emb, article_meta_vec, h_emb)  # (B, K, article_dim)

        return {
            "h_emb": h_emb, "z_mkt": z_mkt, "u_stack": u_stack,
            "query": query, "e_i": e_i, "K_h": K_h,
            "article_meta_vec": article_meta_vec,
        }

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(self, market_feat: torch.Tensor, horizon: str,
                article_emb: Optional[torch.Tensor] = None,
                article_mask: Optional[torch.Tensor] = None,
                article_meta_vec: Optional[torch.Tensor] = None,
                ablation: Optional[str] = None) -> dict:
        """Full forward pass (Eq.8-25).

        market_feat:      (B, M)
        horizon:          "15m" | "1h" | "4h" | "24h"
        article_emb:      (B, K, 768)
        article_mask:     (B, K) float, 1=valid, 0=padding
        article_meta_vec: (B, K, META_DIM)
        ablation:         one of None | "w/o_selective_news" | "w/o_factor" |
                          "w/o_market" | "w/o_confidence" | "w/o_horizon"
        """
        B   = market_feat.shape[0]
        dev = market_feat.device

        enc            = self._encode_common(market_feat, horizon, article_emb,
                                             article_mask, article_meta_vec, ablation)
        h_emb          = enc["h_emb"]
        z_mkt, u_stack = enc["z_mkt"], enc["u_stack"]
        query          = enc["query"]
        e_i, K_h       = enc["e_i"], enc["K_h"]
        article_meta_vec = enc["article_meta_vec"]

        attn_w = p_fac_all = selected_mask = soft_gates = None

        if e_i is not None:
            if ablation == "w/o_selective_news":
                # Uniform attention over all valid articles — no Top-K gating
                mask_float  = article_mask.float()
                alpha_tilde = mask_float / mask_float.sum(dim=1, keepdim=True).clamp(min=1e-8)
                z_news       = (alpha_tilde.unsqueeze(-1) * e_i).sum(dim=1)
                selected_mask = article_mask.bool()
            else:
                # Selective attention (Eq.10-13)
                z_news, alpha_tilde, a_scores = self.sel_attn(
                    e_i, query, article_mask, article_meta_vec, K_h
                )
                selected_mask = alpha_tilde > 0
                # Sigmoid soft-gates for Lsel sel_term (Eq.34): σ(a_i) ∈ [0,1].
                # Σσ(a_i) is not constant (unlike softmax), so sel_term = (Σσ - K_h)²
                # carries a real gradient that trains the scorer to activate ≈ K_h articles.
                soft_gates = torch.sigmoid(a_scores) * article_mask.float()  # (B, K)

            if ablation == "w/o_factor":
                z_fac     = torch.zeros(B, self.factor_mod.U.out_features, device=dev)
                p_fac_all = torch.zeros(B, article_emb.shape[1], self.factor_mod.n_factors, device=dev)
            else:
                z_fac, p_fac_all = self.factor_mod(e_i, alpha_tilde)  # Eq.14-15

            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news  = z_news * has_any
            fused   = self.fusion(z_news, z_fac, z_mkt, h_emb, u_stack=u_stack)  # Eq.20-22
            attn_w  = alpha_tilde
        else:
            fused = self.mkt_only(z_mkt)

        fused      = self.final_dropout(fused)
        dir_logits = self.dir_head(fused)                             # (B, 3) Eq.23
        ret_pred   = self.ret_head(fused).squeeze(-1)                 # (B,)   Eq.24
        confidence = torch.sigmoid(self.conf_head(fused)).squeeze(-1) # (B,)   Eq.25
        if ablation == "w/o_confidence":
            confidence = torch.full_like(confidence, 0.5)

        return {
            "dir_logits":    dir_logits,
            "ret_pred":      ret_pred,
            "confidence":    confidence,
            "attn_weights":  attn_w,
            "selected_mask": selected_mask,
            "p_fac_all":     p_fac_all,
            "soft_gates":    soft_gates,
        }

    def forward_masked(self, market_feat: torch.Tensor, horizon: str,
                       article_emb: Optional[torch.Tensor] = None,
                       article_mask: Optional[torch.Tensor] = None,
                       article_meta_vec: Optional[torch.Tensor] = None) -> dict:
        """Faithfulness forward pass: predict WITHOUT the top-K selected articles (Eq.36).

        Redistributes attention weight only to UNSELECTED articles (renormalized simplex),
        giving the model's prediction "as if the selected evidence were absent".
        When all valid articles are selected (K_h ≥ n_valid), z_news is zeroed
        so the prediction degrades gracefully to market-only signal.

        Returns: {"dir_logits": (B, 3)}
        """
        enc            = self._encode_common(market_feat, horizon, article_emb,
                                             article_mask, article_meta_vec, ablation=None)
        h_emb          = enc["h_emb"]
        z_mkt, u_stack = enc["z_mkt"], enc["u_stack"]
        query          = enc["query"]
        e_i, K_h       = enc["e_i"], enc["K_h"]
        article_meta_vec = enc["article_meta_vec"]

        if e_i is not None:
            z_news, alpha_tilde, _ = self.sel_attn(e_i, query, article_mask,
                                                    article_meta_vec, K_h)
            # Faithfulness masking (Eq.36):
            # zero out selected articles, renormalize remaining to proper simplex.
            selection_mask = (alpha_tilde > 0).float()
            alpha_unsel    = alpha_tilde * (1.0 - selection_mask)
            unsel_sum      = alpha_unsel.sum(dim=1, keepdim=True)
            has_unsel      = (unsel_sum.squeeze(1) > 1e-8).float()   # (B,) 1=has unselected
            alpha_renorm   = alpha_unsel / unsel_sum.clamp(min=1e-8)

            z_news = (alpha_renorm.unsqueeze(-1) * e_i).sum(dim=1) * has_unsel.unsqueeze(-1)
            z_fac, _ = self.factor_mod(e_i, alpha_renorm * has_unsel.unsqueeze(-1))

            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news  = z_news * has_any
            fused   = self.fusion(z_news, z_fac, z_mkt, h_emb, u_stack=u_stack)
        else:
            fused = self.mkt_only(z_mkt)

        fused = self.final_dropout(fused)
        return {"dir_logits": self.dir_head(fused)}

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


    def generate_explanation(
        self,
        outputs: dict,
        article_meta: "list[dict]",
        sample_idx: int = 0,
        P: int = 3,
        horizon: str = "1h",
        symbol: str = "BTC",
    ) -> dict:
        """Generate structured + NL explanation for one prediction (Eq.27-29).

        This is the public inference-time API that wraps ExplanationModule.
        Call after `forward()` to produce human-readable alerts.

        Args:
            outputs:      dict returned by forward() for the batch
            article_meta: list of article metadata dicts with "title" key
            sample_idx:   which sample in the batch to explain (default 0)
            P:            top-P factors to include (default 3)
            horizon:      prediction horizon string for NL text
            symbol:       ticker symbol for NL text

        Returns:
            dict with keys: selected_news, top_factors, factor_dist,
                            factors_text, nl_explanation
        """
        attn_weights = outputs.get("attn_weights")   # (B, K) or None
        p_fac_all    = outputs.get("p_fac_all")      # (B, K, C) or None
        dir_logits   = outputs.get("dir_logits")     # (B, 3)
        confidence   = outputs.get("confidence")     # (B,)

        dir_pred = int(dir_logits[sample_idx].argmax().item()) if dir_logits is not None else 1
        conf_val = float(confidence[sample_idx].item()) if confidence is not None else 0.0

        if attn_weights is None or p_fac_all is None:
            # Market-only mode: no article attention available
            dir_str = {0: "giam (DOWN)", 1: "trung tinh (NEUTRAL)", 2: "tang (UP)"}.get(dir_pred, "?")
            return {
                "selected_news":  [],
                "top_factors":    [],
                "factor_dist":    [],
                "factors_text":   "khong co tin tuc",
                "nl_explanation": (
                    f"Tin hieu: {symbol} co kha nang {dir_str} trong {horizon} toi. "
                    f"Do tin cay: {conf_val:.2f}. "
                    f"Bang chung chinh: N/A (market-only mode). "
                    f"Yeu to chi phoi: khong ro."
                ),
            }

        alpha_single = attn_weights[sample_idx]  # (K,)
        p_fac_single = p_fac_all[sample_idx]     # (K, C)

        return self.explainer(
            alpha_tilde=alpha_single,
            p_fac=p_fac_single,
            article_meta=article_meta,
            K_h=alpha_single.shape[0],
            P=P,
            direction=dir_pred,
            confidence=conf_val,
            horizon=horizon,
            symbol=symbol,
        )


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
