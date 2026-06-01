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
             - 5 timeframe bar-sequence or scalar/hybrid market features
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


# ── Gradient scaling utility (Session 21 audit fix: factor-grounded coupling) ──
class _GradScale(torch.autograd.Function):
    """Identity in forward, scales gradient in backward.

    Used to couple the factor pathway into fusion so that the direction loss
    (Ldir) still flows gradient back through factor_mod — but at a throttled
    magnitude so Ldir does not overpower the explicit Lfac supervision.

    Before: ``z_fac.detach()`` — factor output completely isolated, so factors
            learned ONLY from Lfac (keyword-based pseudo-labels) and had zero
            influence on prediction. This made "factor-grounded" vestigial:
            the factor module was a parallel decoration, not a grounding path.

    After:  ``_GradScale.apply(z_fac, 0.1)`` — factor output still affects
            prediction (forward value unchanged), and Ldir sends 10% gradient
            back through factor_mod. Small enough to preserve Stage-1 training
            stability (the original reason for .detach()), large enough for
            factors and predictions to co-learn representations that actually
            align. Sole change needed to promote "factor-grounded" from
            [VESTIGIAL] to [GENUINE] per Session 22 audit.
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward x unchanged; scale its gradient by `scale` in backward."""
    return _GradScale.apply(x, scale)


# ── Constants ──
EMB_DIM    = 768
META_DIM   = 14   # [0-3]: recency_norm, length_norm, source_cred, novelty_norm
                  # [4-13]: target-based FSA entity sentiment per factor [-1,+1]
                  # Falls back to 4 dims if entity_sentiment not precomputed.

# --- Factor Ontology: Load from JSON if available ---
import json
from pathlib import Path

ONTOLOGY_PATH = Path(__file__).parent.parent / "factor_ontology.json"  # v2/factor_ontology.json

# P2 #18: factor_ontology.json is the SINGLE SOURCE OF TRUTH. Previously this
# file carried a duplicate default list that went out of sync with the JSON
# (shorter keyword lists → fewer pseudo-label matches → Lfac regression on
# articles that would otherwise match). Now we fail loudly if the JSON is
# missing rather than silently using a stale in-code fallback.
if not ONTOLOGY_PATH.exists():
    raise FileNotFoundError(
        f"Factor ontology not found at {ONTOLOGY_PATH}. This file is the "
        "single source of truth for FACTOR_NAMES and FACTOR_KEYWORDS; all "
        "consumers (training, inference, baselines) load from it. Restore "
        "the file from version control."
    )
with open(ONTOLOGY_PATH, "r", encoding="utf-8") as _f:
    _ontology = json.load(_f)
FACTOR_NAMES: list[str] = list(_ontology["FACTOR_NAMES"])
FACTOR_KEYWORDS: dict[str, list[str]] = dict(_ontology["FACTOR_KEYWORDS"])

# Sanity invariants — mismatch here means the JSON was hand-edited incorrectly.
if len(FACTOR_NAMES) != len(FACTOR_KEYWORDS):
    raise ValueError(
        f"factor_ontology.json is internally inconsistent: "
        f"{len(FACTOR_NAMES)} names vs {len(FACTOR_KEYWORDS)} keyword lists."
    )
for _name in FACTOR_NAMES:
    if _name not in FACTOR_KEYWORDS:
        raise ValueError(f"factor_ontology.json: '{_name}' has no keyword list.")

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
    # Temperature bounds prevent softmax collapse/uniform degeneracy during training.
    # tau too small (<0.1) → softmax spikes, single-article dominance; too large (>5)
    # → uniform over all K articles, selectivity lost. Paper Eq.11 doesn't bound tau;
    # we clamp to a practical range on every forward pass.
    TAU_MIN = 0.1
    TAU_MAX = 5.0

    def __init__(self, article_dim: int = 128, query_dim: int = 64,
                 meta_dim: int = META_DIM, tau_init: float = 1.0):
        super().__init__()
        self.W_e  = nn.Linear(article_dim, query_dim, bias=False)
        self.W_q  = nn.Linear(query_dim,   query_dim, bias=False)
        self.W_m  = nn.Linear(meta_dim,    query_dim, bias=False)
        self.v    = nn.Linear(query_dim, 1, bias=False)
        # tau is stored as a raw unconstrained scalar; _tau_clamped() maps it to
        # (TAU_MIN, TAU_MIN + softplus(tau)) with full gradient flow. Hard clamp
        # (.clamp(min, max)) creates a dead zone where grad=0 when tau leaves
        # bounds, preventing recovery. softplus is smooth everywhere.
        self.tau  = nn.Parameter(torch.tensor(tau_init))

    def _tau_clamped(self) -> torch.Tensor:
        # TAU_MIN + softplus(tau) keeps effective temperature > TAU_MIN always,
        # with continuous gradient. softplus(x) ≈ x for x>>0 (linear regime),
        # ≈ 0 for x<<0 (floor at TAU_MIN). No upper hard bound — the optimizer
        # will learn the right scale; extreme tau is prevented by Lsel entropy.
        return self.TAU_MIN + F.softplus(self.tau)

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

        # Temperature softmax with padding mask (Eq.11).
        # Numerical stability: fill with a large negative derived from the valid
        # score maximum (not a hardcoded sentinel), so /tau can't overflow to -inf.
        with torch.no_grad():
            max_valid = scores.masked_fill(~mask.bool(), float("-inf")).amax(
                dim=-1, keepdim=True
            )
            max_valid = torch.where(torch.isfinite(max_valid), max_valid,
                                    torch.zeros_like(max_valid))
        fill_val = max_valid - 40.0  # exp(-40) ≈ 4e-18, safely zero after softmax
        scores_masked = torch.where(mask.bool(), scores, fill_val)
        tau = self._tau_clamped()
        alpha = F.softmax(scores_masked / tau, dim=-1)
        # Zero out padding positions (exp(min/tau) ≈ 0 but clamp for safety)
        alpha = alpha * mask.float()

        # Soft Top-Kh gating (Eq.12): I(i in TopK(a, K_h)) using raw scores a,
        # not post-softmax alpha — matches PDF notation exactly.
        #
        # Use scatter from the returned topk INDICES directly instead of a
        # threshold-comparison trick. Value-based comparison ("scores >= τ-ε")
        # can select MORE than K_h articles when many scores tie near the cut
        # (common in early training when the scorer is near-uniform). Scatter
        # guarantees exactly K_h non-zero positions per sample.
        #
        # Straight-through estimator (PDF Section 3.3.2 mentions this relaxation
        # as an optional improvement):
        #   Forward:  α̃ = α · 𝕀(TopK)   — exactly K_h non-zero positions (paper Eq.12)
        #   Backward: gradient of the downstream loss w.r.t. α flows through ALL
        #             positions, including the K-n_valid that are zero in forward.
        #             Non-selected articles can therefore learn how to *avoid*
        #             being dropped (score must fall further to be un-competitive),
        #             yielding smoother selection learning than a pure hard gate.
        # The identity  α̃ = α + (α_hard − α).detach()  preserves the forward
        # numerical value (α_hard) while routing d/dα straight through as if the
        # gate were the identity.
        if K_h > 0 and K_h < mask.shape[1]:
            _, topk_idx = scores_masked.topk(K_h, dim=-1)
            topk_mask = torch.zeros_like(scores_masked, dtype=torch.bool)
            topk_mask.scatter_(1, topk_idx, True)
            # Intersect with padding mask so we never activate a padded slot,
            # even if its masked-fill score coincides with a real article's score.
            topk_mask = topk_mask & mask.bool()
            alpha_hard  = alpha * topk_mask.float()              # forward value
            alpha_tilde = alpha + (alpha_hard - alpha).detach()  # STE: soft gradient
        else:
            alpha_tilde = alpha

        z_news = (alpha_tilde.unsqueeze(-1) * articles).sum(dim=1)  # (B,article_dim)
        return z_news, alpha_tilde, scores


class FactorModule(nn.Module):
    """
    Eq.14-15:
    p^fac_i = softmax(W_fac e_i + b_fac)
    Z^fac = sum alpha_tilde_i U p^fac_i

    Paper-final implementation keeps Eq.14-15 literal: no factor dropout or
    extra stochastic masking is applied inside this module.
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


class BarSequenceEncoder(nn.Module):
    """
    Per-timeframe bar-sequence encoder — paper-faithful realisation of Eq.17.

    Input  : (B, L_δ, d_δ) — sequence of L_δ bars with d_δ features each
    Output : (B, dm)       — single vector summarising the sequence

    Architecture:
        Conv1d(d_δ, hidden, kernel=3, padding=1)  — local temporal pattern
        GroupNorm(4, hidden) + GELU
        Conv1d(hidden, hidden, kernel=3, padding=1) — second temporal hop
        GroupNorm(4, hidden) + GELU
        AdaptiveAvgPool1d(1) + flatten              — global pool
        Linear(hidden, dm) + LayerNorm + GELU

    Paper Eq.16 specifies a sequence of recent bars per timeframe, not only
    scalar aggregate indicators. This encoder implements that bar-sequence
    path for the multi-timescale market module.
    """
    def __init__(self, seq_len: int = 20, feat_dim: int = 10,
                 hidden: int = 32, out_dim: int = 64):
        super().__init__()
        self.seq_len = int(seq_len)
        self.feat_dim = int(feat_dim)
        self.hidden = int(hidden)
        self.out_dim = int(out_dim)
        # Conv1d over time axis; input shape after transpose: (B, feat_dim, L).
        # GroupNorm instead of BatchNorm to keep small-batch training stable
        # (SAFE-Alert uses batch_size=8 per paper Table 4).
        groups = 4 if self.hidden >= 4 else 1
        self.net = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, self.hidden),
            nn.GELU(),
            nn.Conv1d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, self.hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),    # (B, hidden, 1)
            nn.Flatten(),                # (B, hidden)
        )
        self.proj = nn.Sequential(
            nn.Linear(self.hidden, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L_δ, d_δ) bar sequence.
        Returns:
            (B, out_dim) sequence summary vector.
        """
        if x.dim() != 3:
            raise ValueError(
                f"BarSequenceEncoder expects (B, L, d); got shape {tuple(x.shape)}"
            )
        # Conv1d expects channels-first: (B, d, L)
        x = x.transpose(1, 2).contiguous()
        h = self.net(x)           # (B, hidden)
        return self.proj(h)        # (B, out_dim)


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

    Bar-sequence mode uses one ``BarSequenceEncoder`` per timeframe and consumes
    (B, L_δ, d_δ) tensors, matching paper Eq.16. Scalar mode remains available
    for ablations and backwards-compatible inference.
    """
    # Feature layout constants (scalar path — legacy)
    TF_FEATURES   = 12   # per-timeframe features (12 indicators each)
    CROSS_FEATURES = 3   # cross-timeframe features (dims 60-62)
    ENCODER_DIM    = TF_FEATURES + CROSS_FEATURES  # = 15

    # Bar-sequence path defaults (paper Eq.16)
    # L_δ = 20 bars per timeframe by default. The paper does not specify
    # L_δ, but 20 bars is standard for technical analysis (captures major
    # momentum/mean-reversion patterns at each scale). Per-bar feature dim
    # d_δ = 10 is enough for OHLCV + 5 derived indicators.
    BAR_SEQ_LEN    = 20
    BAR_FEAT_DIM   = 10

    def __init__(self, market_dim: int = 63, n_timeframes: int = N_TIMEFRAMES,
                 dm: int = 64, horizon_emb_dim: int = HORIZON_EMB_DIM,
                 use_bar_sequences: bool = False,
                 market_input_mode: Optional[str] = None,
                 bar_seq_len: int = None,
                 bar_feat_dim: int = None):
        super().__init__()
        self.n_timeframes = n_timeframes
        self.dm = dm
        self.bar_seq_len = int(bar_seq_len or self.BAR_SEQ_LEN)
        self.bar_feat_dim = int(bar_feat_dim or self.BAR_FEAT_DIM)

        # Resolve mode. Precedence: explicit ``market_input_mode`` wins; else fall
        # back to legacy ``use_bar_sequences`` bool. Three modes:
        #   "scalar" — engineered 63-dim indicators (OLD baseline, paper Eq.18+).
        #   "bar"    — paper-literal Eq.16 bar sequences (raw OHLCV per TF).
        #   "hybrid" — both paths, fused via gated residual:
        #               u_τ = u_scalar_τ + gate ⊙ proj(u_bar_τ)
        #              with ``gate`` init at 0 → at training start the model is
        #              identical to scalar baseline. The gate learns to mix in
        #              bar information only if it improves the objective. This
        #              prevents low-information bar features from corrupting a
        #              well-engineered scalar trunk.
        if market_input_mode is None:
            market_input_mode = "bar" if bool(use_bar_sequences) else "scalar"
        if market_input_mode not in ("scalar", "bar", "hybrid"):
            raise ValueError(
                f"market_input_mode must be one of 'scalar'|'bar'|'hybrid'; got {market_input_mode!r}"
            )
        self.market_input_mode = market_input_mode
        # Backward-compat flag: True iff bar pathway is constructed (bar/hybrid).
        self.use_bar_sequences = market_input_mode in ("bar", "hybrid")

        # Per-timeframe encoder construction.
        # In "scalar" or "bar" mode, we build a single self.encoders ModuleList so
        # legacy checkpoints (which only know about ``self.encoders``) keep loading.
        # In "hybrid" mode we expose two ModuleLists side-by-side.
        if market_input_mode == "scalar":
            self.encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.ENCODER_DIM, dm), nn.LayerNorm(dm), nn.GELU(),
                ) for _ in range(n_timeframes)
            ])
            self.scalar_encoders = None
            self.bar_encoders = None
        elif market_input_mode == "bar":
            self.encoders = nn.ModuleList([
                BarSequenceEncoder(
                    seq_len=self.bar_seq_len,
                    feat_dim=self.bar_feat_dim,
                    out_dim=dm,
                ) for _ in range(n_timeframes)
            ])
            self.scalar_encoders = None
            self.bar_encoders = None
        else:  # hybrid
            self.encoders = None
            self.scalar_encoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.ENCODER_DIM, dm), nn.LayerNorm(dm), nn.GELU(),
                ) for _ in range(n_timeframes)
            ])
            self.bar_encoders = nn.ModuleList([
                BarSequenceEncoder(
                    seq_len=self.bar_seq_len,
                    feat_dim=self.bar_feat_dim,
                    out_dim=dm,
                ) for _ in range(n_timeframes)
            ])

        # Gated residual for hybrid mode.
        # bar_proj aligns bar feature space → scalar feature space (random init).
        # bar_gate is a per-channel scalar, init at 0 so the residual contribution
        # starts at exactly 0 — the model begins identical to scalar baseline.
        # Single global gate (per-channel) is shared across timeframes for
        # simplicity; the per-TF u_scalar_τ already gives some path diversity.
        if market_input_mode == "hybrid":
            self.bar_proj = nn.Linear(dm, dm)
            self.bar_gate = nn.Parameter(torch.zeros(dm))
        else:
            self.bar_proj = None
            self.bar_gate = None

        # Horizon-conditioned attention weights (Eq.18) — same for both paths.
        #
        # Paper Eq.18: β_δ = softmax_δ( wᵀ · tanh(W_δ · u^(δ) + W_h · h_emb) )
        # Parameter sharing per paper's notation:
        #   • ``w``   : shared scoring vector (no δ subscript)   — one Linear
        #   • ``W_δ`` : per-timeframe projection (δ subscript)   — ModuleList of n_timeframes
        #   • ``W_h`` : shared horizon projection (no δ subscript) — one Linear
        #
        # Paper Eq.18: per-timeframe scorer w_delta and projection W_delta,
        # with a shared horizon projection W_h.
        self.w       = nn.ModuleList([nn.Linear(dm, 1, bias=False) for _ in range(n_timeframes)])
        self.W_delta = nn.ModuleList([nn.Linear(dm, dm, bias=False) for _ in range(n_timeframes)])
        self.W_h     = nn.Linear(horizon_emb_dim, dm, bias=False)

    def _split_features(self, feat: torch.Tensor) -> list:
        """
        Split (B, 63) into 5 tensors of shape (B, 15) = 12 tf-specific + 3 cross.
        Each timeframe gets its own 12 features + the shared 3 cross-timeframe features.
        Only used in scalar mode (``use_bar_sequences=False``).
        """
        cross = feat[:, 60:63]  # (B, 3) — shared cross-timeframe context
        chunks = []
        for i in range(self.n_timeframes):
            tf_feat = feat[:, i*12 : (i+1)*12]          # (B, 12)
            chunks.append(torch.cat([tf_feat, cross], dim=-1))  # (B, 15)
        return chunks

    def forward(self, market_feat_list, h_emb: torch.Tensor,
                market_bars: Optional[list] = None):
        """
        Eq.16-19: Horizon-weighted market encoding across 5 REAL timeframes.

        Args:
            market_feat_list: input shape depends on ``self.market_input_mode``:
              - "scalar" (legacy):
                  * Single tensor (B, 63) — split into 5×(B,15) per timeframe, OR
                  * List of 5 tensors (B, 15) — already split (e.g. live infer).
              - "bar" (paper Eq.16):
                  * List of 5 tensors (B, L_δ, d_δ) — per-timeframe bar sequences.
              - "hybrid" (default for new training, see __init__ docstring):
                  * Single (B, 63) tensor or list of 5×(B, 15) for the scalar path.
                  * Bar path passed via the ``market_bars`` kwarg below.
            h_emb: (B, horizon_emb_dim)
            market_bars: list of 5 × (B, L, d) bar sequences. Used in "hybrid"
                mode only; ignored in "scalar"/"bar" mode (in "bar" mode the
                bar list is passed via ``market_feat_list``). When ``None`` in
                hybrid mode, the model gracefully falls back to scalar-only
                (gate path contributes 0).

        Returns:
            Z_mkt: (B, dm)     — horizon-weighted aggregate (for MLP concat)
            u_stack: (B, 5, dm) — per-timeframe vectors (K/V for CrossAttentionFusion)
        """
        if self.market_input_mode == "scalar":
            if isinstance(market_feat_list, torch.Tensor):
                market_feat_list = self._split_features(market_feat_list)
            u_list = [enc(f) for enc, f in zip(self.encoders, market_feat_list)]

        elif self.market_input_mode == "bar":
            # Sequence mode: caller must pass a list of 5 × (B, L, d) tensors.
            if isinstance(market_feat_list, torch.Tensor):
                raise ValueError(
                    "market_input_mode='bar' requires a list of 5 bar-sequence "
                    "tensors (B, L_δ, d_δ), not a single scalar tensor."
                )
            if len(market_feat_list) != self.n_timeframes:
                raise ValueError(
                    f"Expected {self.n_timeframes} bar sequences, got {len(market_feat_list)}."
                )
            u_list = [enc(b) for enc, b in zip(self.encoders, market_feat_list)]

        else:  # "hybrid"
            # Scalar branch always runs: split (B, 63) → 5×(B, 15) → encode.
            if isinstance(market_feat_list, torch.Tensor):
                scalar_chunks = self._split_features(market_feat_list)
            else:
                scalar_chunks = market_feat_list
            u_scalar = [enc(f) for enc, f in zip(self.scalar_encoders, scalar_chunks)]

            if market_bars is None:
                # Graceful degradation: bars unavailable → use scalar-only path.
                # Equivalent to gate=0; identical numerics to a scalar-only
                # encoder (the bar sub-graph is simply not invoked).
                u_list = u_scalar
            else:
                if len(market_bars) != self.n_timeframes:
                    raise ValueError(
                        f"hybrid mode expects {self.n_timeframes} bar tensors, "
                        f"got {len(market_bars)}."
                    )
                u_bar = [enc(b) for enc, b in zip(self.bar_encoders, market_bars)]
                # Gated residual fusion per timeframe.
                # gate (per-channel, init=0) starts the model at the scalar baseline;
                # bar_proj aligns bar repr space → scalar repr space.
                u_list = [
                    u_s + self.bar_gate * self.bar_proj(u_b)
                    for u_s, u_b in zip(u_scalar, u_bar)
                ]

        # Eq.18: horizon-conditioned attention
        # β_δ = softmax_δ(w_δ^T · tanh(W_δ · u^(δ) + W_h · h_emb)).
        # w_δ is per-TF (matches paper subscript), W_δ per-TF, W_h shared.
        h_proj = self.W_h(h_emb)  # (B, dm)
        beta_logits = []
        for i, u_tau in enumerate(u_list):
            score = self.w[i](torch.tanh(self.W_delta[i](u_tau) + h_proj))  # (B, 1)
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
        # Sprint 10 anti-overfit: fusion MLP dropout 0.1 → 0.2.
        self.mlp = nn.Sequential(
            nn.Linear(fused_in, 128), nn.GELU(), nn.Dropout(0.2),
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
    Language is configurable via ``lang`` ('vi' or 'en') on the forward pass.
    """
    # Per-language phrase tables (Eq.29 template).
    # Session 23 P3 #13: required placeholders are validated at class-load
    # time so template drift (missing / renamed {placeholders}) fails loudly
    # at import rather than silently at inference.
    _REQUIRED_PLACEHOLDERS = frozenset({"symbol", "dir_str", "horizon", "conf", "news", "factor_str"})
    _DIR_STRS = {
        "vi": {0: "giam (DOWN)", 1: "trung tinh (NEUTRAL)", 2: "tang (UP)"},
        "en": {0: "decrease (DOWN)", 1: "neutral (NEUTRAL)", 2: "increase (UP)"},
    }
    _TEMPLATES = {
        "vi": ("Tin hieu: {symbol} co kha nang {dir_str} trong {horizon} toi. "
               "Do tin cay: {conf:.2f}. "
               "Bang chung chinh: {news}. "
               "Yeu to chi phoi: {factor_str}."),
        "en": ("Signal: {symbol} likely to {dir_str} over the next {horizon}. "
               "Confidence: {conf:.2f}. "
               "Primary evidence: {news}. "
               "Dominant factors: {factor_str}."),
    }
    _UNKNOWN = {"vi": "khong ro", "en": "unknown"}

    @classmethod
    def _validate_templates(cls) -> None:
        """Raise at class-load time if any template is missing required keys.

        Session 23 P3 #13 — catches refactor accidents (e.g., renaming
        ``{conf}`` to ``{confidence}`` in one language but not the other).
        Detects placeholders via ``str.format_map`` with a tracking dict.
        """
        import string as _string
        formatter = _string.Formatter()
        required = cls._REQUIRED_PLACEHOLDERS
        for lang, tmpl in cls._TEMPLATES.items():
            found = {
                field_name
                for _, field_name, _, _ in formatter.parse(tmpl)
                if field_name
            }
            missing = required - found
            unexpected = found - required
            if missing or unexpected:
                raise ValueError(
                    f"ExplanationModule template for lang='{lang}' has "
                    f"missing={sorted(missing)} unexpected={sorted(unexpected)} "
                    f"placeholders vs required={sorted(required)}. Update "
                    f"either the template or _REQUIRED_PLACEHOLDERS."
                )
    _NONE_AVAIL = {"vi": "N/A", "en": "N/A"}

    def forward(self, alpha_tilde: torch.Tensor, p_fac: torch.Tensor,
                article_meta: list[dict], K_h: int, P: int = 3,
                direction: int = 1, confidence: float = 0.0,
                horizon: str = "1h", symbol: str = "BTC",
                lang: str = "vi") -> dict:
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

        # Eq.29: Phi_txt — template NL explanation (language-configurable)
        lang_key = lang if lang in self._TEMPLATES else "vi"
        dir_str = self._DIR_STRS[lang_key].get(direction, "?")
        unknown = self._UNKNOWN[lang_key]
        joiner = " va " if lang_key == "vi" else " and "
        factor_str = joiner.join(top_factors[:2]) if top_factors else unknown
        news_str = "; ".join(selected_news[:2]) if selected_news else self._NONE_AVAIL[lang_key]
        nl = self._TEMPLATES[lang_key].format(
            symbol=symbol, dir_str=dir_str, horizon=horizon,
            conf=confidence, news=news_str, factor_str=factor_str,
        )
        factors_text = ", ".join(top_factors) if top_factors else unknown

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
                 has_news: bool = True, K_15m: int = 3, K_1h: int = 4, K_4h: int = 5, K_24h: int = 8,
                 # Multi-timeframe bar-sequence mode for paper Eq.16. The
                 # scalar path remains available for ablations and inference
                 # compatibility.
                 use_bar_sequences: bool = False,
                 # Sprint 9 — three-way market input mode:
                 #   None    → use ``use_bar_sequences`` bool (legacy)
                 #   "scalar" → engineered 63-dim only (OLD baseline)
                 #   "bar"    → paper Eq.16 bar sequences only
                 #   "hybrid" → gated residual: u = u_scalar + gate ⊙ proj(u_bar)
                 #              with gate init=0 (starts at scalar baseline).
                 market_input_mode: Optional[str] = None,
                 bar_seq_len: int = 20,
                 bar_feat_dim: int = 10,
                 predict_volatility: bool = True,
                 predict_ret_sign: bool = True,
                 predict_ret_bin: bool = True,
                 predict_edge: bool = True):
        super().__init__()
        self.has_news    = has_news
        self.K_h_map     = {"15m": K_15m, "1h": K_1h, "4h": K_4h, "24h": K_24h}
        self.article_dim = article_dim
        # Auxiliary heads are kept in the module for checkpoint schema stability.
        # These flags only control whether their forward outputs are computed.
        self.predict_volatility = bool(predict_volatility)
        self.predict_ret_sign = bool(predict_ret_sign)
        self.predict_ret_bin = bool(predict_ret_bin)
        self.predict_edge = bool(predict_edge)

        # Horizon embedding (learned)
        self.h_emb_table = nn.Embedding(len(HORIZON_VOCAB), HORIZON_EMB_DIM)

        # Market encoder (Eq.16-19)
        mkt_dm = 64
        self.market_enc = MultiTimescaleMarketEncoder(
            market_dim=market_dim, dm=mkt_dm,
            use_bar_sequences=bool(use_bar_sequences),
            market_input_mode=market_input_mode,
            bar_seq_len=bar_seq_len,
            bar_feat_dim=bar_feat_dim,
        )
        # Canonical mode resolved by the encoder (handles legacy bool fallback).
        self.market_input_mode = self.market_enc.market_input_mode
        # Backward-compat: keep ``use_bar_sequences`` mirror in sync.
        self.use_bar_sequences = self.market_enc.use_bar_sequences

        # Query projection for attention (Eq.9): q_{s,t,h} = W_q [mbar_{s,t}; h_emb] + b_q.
        # Asset identity is represented through the market summary; the
        # paper-final model has no separate learned symbol embedding.
        self.query_proj = nn.Linear(mkt_dm + HORIZON_EMB_DIM, query_dim)

        if has_news:
            # Article encoder (Eq.8)
            # Sprint 10 anti-overfit: encoder dropout 0.2 → 0.3 (distributed
            # regularization vs single final_dropout=0.35 bottleneck).
            self.article_enc = ArticleEncoder(article_dim=article_dim, dropout=0.3)
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

        # Final dropout = 0.35 (higher than the per-module 0.1–0.2) acts as a
        # last-mile regulariser on the prediction-head input, empirically
        # closing the late-Stage-3 train/val gap without requiring a wider
        # weight_decay (which would also slow feature learning).
        self.final_dropout = nn.Dropout(0.35)  # before prediction heads

        # Prediction heads (Eq.23-25)
        self.dir_head  = nn.Linear(fused_dim, 3)  # 3-class: DOWN(0), NEUTRAL(1), UP(2)
        self.ret_head  = nn.Linear(fused_dim, 1)  # return regression (Eq.24)
        # Paper Eq.25: single linear confidence projection followed by sigmoid.
        self.conf_head = nn.Linear(fused_dim, 1)  # confidence (Eq.25)
        # Volatility-regression extension. Paper §4.1.4 stores future
        # volatility as a benchmark field, but the paper loss does not define
        # Lvol. The head remains for schema compatibility; paper-final runs
        # pass predict_volatility=False and lambda_vol=0.
        self.vol_head  = nn.Linear(fused_dim, 1)  # volatility regression extension
        # Optional return-sign auxiliary head. It outputs one raw logit for
        # P(ret > +epsilon_h) on tradeable-move samples when enabled.
        self.ret_sign_head = nn.Linear(fused_dim, 1)  # binary return-sign auxiliary
        # Optional 3-class tradability head. Bins are direction-independent and
        # based on unsigned |ret_label| vs epsilon_h thresholds:
        #   0 = no_edge, 1 = marginal_edge, 2 = strong_edge.
        # Paper-final training keeps this disabled unless lambda_ret_bin > 0.
        self.ret_bin_head = nn.Linear(fused_dim, 3)  # tradability bins
        # Optional auxiliary directional edge heads. Disabled in the paper-final
        # config unless lambda_up_edge/lambda_down_edge are enabled.
        self.up_edge_head = nn.Linear(fused_dim, 1)
        self.down_edge_head = nn.Linear(fused_dim, 1)
        # Zero-init scalar regression heads so early SmoothL1/BCE losses start
        # near the neutral prediction instead of random large logits.
        nn.init.zeros_(self.ret_head.weight)
        nn.init.zeros_(self.ret_head.bias)
        nn.init.zeros_(self.vol_head.weight)
        nn.init.zeros_(self.vol_head.bias)
        # Initial sigmoid output = 0.5, i.e. no sign preference.
        nn.init.zeros_(self.ret_sign_head.weight)
        nn.init.zeros_(self.ret_sign_head.bias)
        # Zero-init ret_bin_head so the initial softmax is uniform and carries
        # no spurious tradability preference.
        nn.init.zeros_(self.ret_bin_head.weight)
        nn.init.zeros_(self.ret_bin_head.bias)
        # Zero-init binary edge heads so the initial sigmoid is 0.5.
        nn.init.zeros_(self.up_edge_head.weight)
        nn.init.zeros_(self.up_edge_head.bias)
        nn.init.zeros_(self.down_edge_head.weight)
        nn.init.zeros_(self.down_edge_head.bias)

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
        symbol: Optional[str] = None,
        market_bars: Optional[torch.Tensor] = None,
    ) -> dict:
        """Shared market + horizon + article encoding (Eq.8-19).

        Extracts the common prefix of forward() and forward_masked() to eliminate
        code duplication. Both callers receive identical intermediate representations
        and apply their own article-weighting logic on top.

        Args:
            symbol: Accepted for call-site compatibility. The current
                    paper-final query follows Eq.9 literally and uses
                    ``[market_summary; horizon_embedding]`` only.
            market_bars: paper Eq.16 bar sequences, shape
                    (B, 5, L_δ, d_δ) where axis 1 indexes TFs in order
                    [1m, 5m, 15m, 1h, 4h]. Required when
                    ``self.use_bar_sequences`` is True, ignored otherwise
                    (market_feat scalar path used instead).

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

        # Symbol is intentionally ignored in the paper-final query.
        del symbol

        # Market encoding (Eq.16-19).
        # Sprint 9 — three modes resolved at construction time and dispatched here:
        #   "scalar" → encoder consumes (B, 63) scalar features only.
        #   "bar"    → encoder consumes 5 × (B, L, d) bar sequences only;
        #              market_bars is REQUIRED and must be 4-D (B, 5, L, d).
        #   "hybrid" → encoder consumes both scalar features AND bars (when
        #              available). When ``market_bars`` is None the encoder
        #              gracefully falls back to scalar-only (gate residual=0).
        if self.market_input_mode == "bar":
            if market_bars is None:
                raise ValueError(
                    "market_input_mode='bar' but market_bars not provided. "
                    "Pass market_bars=(B, 5, L, d) from the dataset, or "
                    "construct SAFEAlertNet with market_input_mode='scalar'/'hybrid'."
                )
            if market_bars.dim() != 4:
                raise ValueError(
                    f"market_bars must be (B, 5, L, d); got shape {tuple(market_bars.shape)}"
                )
            bars_list = [market_bars[:, i, :, :] for i in range(market_bars.shape[1])]
            z_mkt, u_stack = self.market_enc(bars_list, h_emb)  # (B, dm), (B, 5, dm)
        elif self.market_input_mode == "hybrid":
            bars_list = None
            if market_bars is not None:
                if market_bars.dim() != 4:
                    raise ValueError(
                        f"market_bars must be (B, 5, L, d); got shape {tuple(market_bars.shape)}"
                    )
                bars_list = [market_bars[:, i, :, :] for i in range(market_bars.shape[1])]
            z_mkt, u_stack = self.market_enc(market_feat, h_emb, market_bars=bars_list)
        else:  # "scalar"
            z_mkt, u_stack = self.market_enc(market_feat, h_emb)  # (B, dm), (B, 5, dm)
        if ablation == "w/o_market":
            z_mkt   = torch.zeros_like(z_mkt)
            u_stack = torch.zeros_like(u_stack)

        # Query projection (Eq.9 paper-literal) — q_{s,t,h} = W_q [m̄_{s,t}; h_emb] + b_q.
        # Asset conditioning enters through m̄_{s,t} (market summary depends on s).
        query = self.query_proj(torch.cat([z_mkt, h_emb], dim=-1))  # (B, query_dim)

        # Article encoding (Eq.8) — only when articles are present
        e_i = K_h = None
        if self.has_news and article_emb is not None and article_mask is not None:
            K_h = self.K_h_map.get(horizon, 3)
            if article_meta_vec is None:
                article_meta_vec = torch.zeros(B, article_emb.shape[1], META_DIM, device=dev)
            # Shape alignment contract: K (number of articles per sample) must
            # match across embeddings, mask, and metadata. Silently mismatched
            # K would degrade the META stream without a visible error; the
            # assertion surfaces this before it corrupts Top-K selection.
            assert article_meta_vec.shape[1] == article_emb.shape[1] == article_mask.shape[1], (
                f"K mismatch: emb.K={article_emb.shape[1]}, "
                f"mask.K={article_mask.shape[1]}, meta.K={article_meta_vec.shape[1]}"
            )
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
                ablation: Optional[str] = None,
                symbol: Optional[str] = None,
                market_bars: Optional[torch.Tensor] = None) -> dict:
        """Full forward pass (Eq.8-25).

        market_feat:      (B, M) — scalar market features (used when use_bar_sequences=False)
        horizon:          "15m" | "1h" | "4h" | "24h"
        article_emb:      (B, K, 768)
        article_mask:     (B, K) float, 1=valid, 0=padding
        article_meta_vec: (B, K, META_DIM)
        ablation:         one of None | "w/o_selective_news" | "w/o_factor" |
                          "w/o_market" | "w/o_confidence" | "w/o_horizon" |
                          "w/o_bar_sequences" (see AblationTrainer)
        symbol:           accepted for compatibility; not used by Eq.9 query.
        market_bars:      (B, 5, L_δ, d_δ) — paper Eq.16 bar sequences. Required
                          when use_bar_sequences=True, ignored otherwise.
        """
        B   = market_feat.shape[0]
        dev = market_feat.device

        enc            = self._encode_common(market_feat, horizon, article_emb,
                                             article_mask, article_meta_vec, ablation,
                                             symbol=symbol,
                                             market_bars=market_bars)
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
                # Keep e_i attached so Lfac (via p_fac_all → W_fac → e_i) can shape
                # article_enc toward factor-relevant representations.
                z_fac, p_fac_all = self.factor_mod(e_i, alpha_tilde)  # Eq.14-15

            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news  = z_news * has_any
            # Eq.22 uses z_fac in the forward fusion path. The gradient scaler
            # leaves that value unchanged while throttling direction-loss
            # backflow into factor_mod, so Lfac remains the primary factor
            # supervision signal.
            z_fac_coupled = scale_gradient(z_fac, 0.1)
            fused   = self.fusion(z_news, z_fac_coupled, z_mkt, h_emb, u_stack=u_stack)  # Eq.20-22
            attn_w  = alpha_tilde
        else:
            fused = self.mkt_only(z_mkt)

        fused      = self.final_dropout(fused)
        dir_logits = self.dir_head(fused)                             # (B, 3) Eq.23
        # Eq.24 return regression uses the same forward features as direction,
        # but its backward influence on the shared trunk is softened.
        ret_pred   = self.ret_head(fused).squeeze(-1)  # (B,)   Eq.24
        # Paper Eq.25: confidence = sigmoid(W_c · Z_fus + b_c).
        confidence = torch.sigmoid(self.conf_head(fused)).squeeze(-1) # (B,)   Eq.25
        # Volatility head extension. Disabled in paper-final config; when
        # disabled, skip the Linear+softplus work and return None.
        vol_pred = (
            F.softplus(self.vol_head(fused)).squeeze(-1)
            if self.predict_volatility
            else None
        )
        # Optional binary return-sign logit. Kept raw for BCEWithLogitsLoss.
        ret_sign_logit = (
            self.ret_sign_head(fused).squeeze(-1)
            if self.predict_ret_sign
            else None
        )
        # Optional tradability logits. When disabled, skip the head entirely and
        # return None so diagnostics can report it as disabled.
        if self.predict_ret_bin:
            ret_bin_logits     = self.ret_bin_head(scale_gradient(fused, 0.1))     # (B, 3)
            ret_bin_probs      = F.softmax(ret_bin_logits, dim=-1)                 # (B, 3)
            tradeability_score = ret_bin_probs[..., 2] - ret_bin_probs[..., 0]     # (B,) P(strong)-P(no)
        else:
            ret_bin_logits = None
            tradeability_score = None
        # Optional binary UP/DOWN edge heads. Disabled in the paper-final
        # config unless their lambdas are enabled.
        if self.predict_edge:
            edge_fused = scale_gradient(fused, 0.1)
            up_edge_logit = self.up_edge_head(edge_fused).squeeze(-1)
            down_edge_logit = self.down_edge_head(edge_fused).squeeze(-1)
            up_edge_prob = torch.sigmoid(up_edge_logit)
            down_edge_prob = torch.sigmoid(down_edge_logit)
        else:
            up_edge_logit = None
            down_edge_logit = None
            up_edge_prob = None
            down_edge_prob = None
        if ablation == "w/o_confidence":
            confidence = torch.full_like(confidence, 0.5)

        return {
            "dir_logits":     dir_logits,
            "ret_pred":       ret_pred,
            "vol_pred":       vol_pred,
            "ret_sign_logit":     ret_sign_logit,
            "ret_bin_logits":     ret_bin_logits,
            "tradeability_score": tradeability_score,
            "up_edge_logit":      up_edge_logit,
            "down_edge_logit":    down_edge_logit,
            "up_edge_prob":       up_edge_prob,
            "down_edge_prob":     down_edge_prob,
            "confidence":     confidence,
            "attn_weights":   attn_w,
            "selected_mask":  selected_mask,
            "p_fac_all":      p_fac_all,
            "soft_gates":     soft_gates,
        }

    def forward_masked(self, market_feat: torch.Tensor, horizon: str,
                       article_emb: Optional[torch.Tensor] = None,
                       article_mask: Optional[torch.Tensor] = None,
                       article_meta_vec: Optional[torch.Tensor] = None,
                       symbol: Optional[str] = None,
                       market_bars: Optional[torch.Tensor] = None) -> dict:
        """Faithfulness forward pass: predict WITHOUT the top-K selected articles (Eq.36).

        Redistributes attention weight only to UNSELECTED articles (renormalized simplex),
        giving the model's prediction "as if the selected evidence were absent".
        When all valid articles are selected (K_h ≥ n_valid), z_news is zeroed
        so the prediction degrades gracefully to market-only signal.

        Returns: {"dir_logits": (B, 3)}
        """
        # P0 #6: Force eval mode for the entire forward pass so Dropout / BN
        # stay deterministic regardless of the caller's mode. The training
        # loop used to manually wrap this call with eval()/train(), but other
        # callers (ablation scripts, faithfulness eval) easily forget — and
        # Dropout noise silently corrupts the L_faith gap signal. Guard here
        # is defence-in-depth: caller-agnostic determinism for all consumers.
        _was_training = self.training
        self.eval()
        try:
            return self._forward_masked_impl(
                market_feat, horizon, article_emb, article_mask, article_meta_vec,
                symbol=symbol,
                market_bars=market_bars,
            )
        finally:
            if _was_training:
                self.train()

    def _forward_masked_impl(self, market_feat: torch.Tensor, horizon: str,
                             article_emb: Optional[torch.Tensor] = None,
                             article_mask: Optional[torch.Tensor] = None,
                             article_meta_vec: Optional[torch.Tensor] = None,
                             symbol: Optional[str] = None,
                             market_bars: Optional[torch.Tensor] = None) -> dict:
        """Internal implementation — callers should use :meth:`forward_masked`
        so the eval-mode guard is enforced.

        Session 23 P1 #6 fix: defensive assertion catches callers that bypass
        the public :meth:`forward_masked` wrapper (e.g., debug scripts, notebooks,
        ablation runners using private API). Without this guard, a caller in
        training mode would trigger Dropout / BN in the faithfulness forward,
        corrupting the L_faith gap signal with stochastic noise. The public
        forward_masked() wraps this with eval()/train() restore; the guard
        documents that pre-condition in code.
        """
        assert not self.training, (
            "_forward_masked_impl must be called in eval mode. "
            "Use model.forward_masked(...) which enforces this contract."
        )
        enc            = self._encode_common(market_feat, horizon, article_emb,
                                             article_mask, article_meta_vec, ablation=None,
                                             symbol=symbol,
                                             market_bars=market_bars)
        h_emb          = enc["h_emb"]
        # Detach market embeddings so Lfaith gradient only flows through the
        # news/selector path. Market encoder should be trained by Ldir (full
        # forward), not by faithfulness — otherwise Lfaith would teach the
        # market encoder to ignore news co-occurrence, conflicting with fusion.
        z_mkt    = enc["z_mkt"].detach()
        u_stack  = enc["u_stack"].detach()
        # query was computed from z_mkt (pre-detach) inside _encode_common, so it
        # still carries a gradient path to market_enc. Detach here so Lfaith cannot
        # update market_enc via the sel_attn(query) → query_proj → z_mkt route.
        query    = enc["query"].detach()
        e_i, K_h = enc["e_i"], enc["K_h"]
        article_meta_vec = enc["article_meta_vec"]

        if e_i is not None:
            # Full selective attention to identify the top-K selected articles.
            # α̃ is zero on both non-top-K *and* padded slots, so α̃·(1-selmask)
            # collapses to all-zeros — which previously made forward_masked
            # degenerate to a pure market-only prediction. Instead we recompute
            # a proper softmax over the UNSELECTED articles using the raw scores
            # returned by the scorer, matching the paper's intent for L_faith
            # ("predict without the selected evidence but with the rest of the
            # article pool still visible").
            z_news, alpha_tilde, scores = self.sel_attn(
                e_i, query, article_mask, article_meta_vec, K_h
            )
            selected_mask = alpha_tilde > 0                               # (B, K) top-K
            unsel_article_mask = article_mask.bool() & ~selected_mask     # keep only non-top-K valid

            # Softmax over UNSELECTED valid articles only. Use max-offset fill so
            # the /tau divide can't overflow to -inf (previous torch.finfo().min/2
            # hack produced NaN when unsel row was all-False). nan_to_num + the
            # has_unsel gate below form defence-in-depth against the edge case.
            with torch.no_grad():
                max_valid = scores.masked_fill(~unsel_article_mask, float("-inf")).amax(
                    dim=-1, keepdim=True
                )
                max_valid = torch.where(torch.isfinite(max_valid), max_valid,
                                        torch.zeros_like(max_valid))
            fill_val = max_valid - 40.0
            scores_for_unsel = torch.where(unsel_article_mask, scores, fill_val)
            tau = self.sel_attn._tau_clamped()
            alpha_unsel = F.softmax(scores_for_unsel / tau, dim=-1)
            alpha_unsel = torch.nan_to_num(alpha_unsel, nan=0.0, posinf=0.0, neginf=0.0)
            alpha_unsel = alpha_unsel * unsel_article_mask.float()

            has_unsel = unsel_article_mask.any(dim=1, keepdim=True).float()  # (B,1)
            # When a sample has *no* unselected articles (e.g. K_h >= n_valid),
            # gracefully fall back to market-only by zeroing the article stream.
            z_news = (alpha_unsel.unsqueeze(-1) * e_i).sum(dim=1) * has_unsel
            z_fac, _ = self.factor_mod(e_i, alpha_unsel * has_unsel)

            has_any = article_mask.any(dim=-1, keepdim=True).float()
            z_news  = z_news * has_any
            # Match the full-forward factor gradient policy: keep z_fac in the
            # forward path while reducing auxiliary backflow into factor_mod.
            z_fac_coupled = scale_gradient(z_fac, 0.1)
            fused   = self.fusion(z_news, z_fac_coupled, z_mkt, h_emb, u_stack=u_stack)
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


# Session 23 P3 #13: validate ExplanationModule templates at import time.
# Raising here (before any model is constructed) ensures a refactor that
# drifts placeholder names fails loudly at `import safe_alert_net` instead
# of deep inside inference where the error traceback would be confusing.
ExplanationModule._validate_templates()
