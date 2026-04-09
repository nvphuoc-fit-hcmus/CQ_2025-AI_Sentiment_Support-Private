"""
Multi-objective loss functions for SAFE-Alert training (PDF Eq.30-37).

L = λ₁Ldir + λ₂Lret + λ₃Lfac + λ₄Lsel + λ₅Lcal + λ₆Lfaith + λ₇Lrisk

PDF Equations:
  - Ldir  (Eq.31): Direction classification   → weighted cross-entropy
  - Lret  (Eq.32): Return regression          → SmoothL1 + Pearson + bias
  - Lfac  (Eq.33): Factor prediction          → soft cross-entropy
  - Lsel  (Eq.34): Selection regularization   → (Σα̃ - K_h)² + η·H(α̃)
  - Lcal  (Eq.35): Calibration               → Brier score
  - Lfaith(Eq.36): Faithfulness               → margin loss (full vs masked)
  - Lrisk (Eq.37): Selective risk             → confidence-weighted CE + coverage
"""

import torch
import torch.nn.functional as F

# ── Numerical constants ────────────────────────────────────────────────────────
_EPS           = 1e-8        # General epsilon for division stability
_CONF_MIN      = 1e-7        # Lower bound for Lcal clamped confidence (Eq.35)
_CONF_MAX      = 1 - 1e-7   # Upper bound for Lcal clamped confidence (Eq.35)
_LRISK_CONF_MIN = 0.01       # Looser floor for Lrisk denominator (Eq.37):
                             # ensures denom >= 0.01·B, preventing grad explosion
                             # when confidence head is near-random (early training).
_LSEL_CLAMP    = 50.0        # Max absolute value for Lsel (guards term-1 overflow)
_LRET_CORR_W   = 0.2         # Weight of Pearson correlation term in Lret (not in PDF)
_LRET_BIAS_W   = 0.1         # Weight of mean-bias penalty in Lret (not in PDF)


class MultiObjectiveLoss:
    """Multi-objective loss implementing PDF Eq.30-37 exactly.

    All seven loss components are computed independently and combined via
    learnable (mutable) λ weights that the trainer updates each epoch to
    implement the 3-stage curriculum (PDF Section 4.5.1).
    """

    def __init__(
        self,
        lambda1: float = 1.0,           # Ldir   weight (Eq.31)
        lambda2: float = 0.5,           # Lret   weight (Eq.32)
        lambda3: float = 0.3,           # Lfac   weight (Eq.33)
        lambda4: float = 0.2,           # Lsel   weight (Eq.34)
        lambda5: float = 0.1,           # Lcal   weight (Eq.35)
        lambda6: float = 0.1,           # Lfaith weight (Eq.36)
        lambda7: float = 0.05,          # Lrisk  weight (Eq.37)
        K_h: int = 8,                   # Target article selection count (Lsel)
        eta: float = 0.1,               # Entropy coefficient η (Eq.34)
        faith_margin: float = 0.15,     # Margin m for Lfaith (Eq.36)
        coverage_target: float = 0.35,  # Coverage target κ for Lrisk (Eq.37)
        mu: float = 0.02,               # Coverage penalty weight μ (Eq.37)
        class_weights: torch.Tensor = None,  # Per-class weights for Ldir
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.lambda5 = lambda5
        self.lambda6 = lambda6
        self.lambda7 = lambda7
        self.K_h = K_h
        self.eta = eta
        self.faith_margin = faith_margin
        self.coverage_target = coverage_target
        self.mu = mu
        self.class_weights = class_weights

    # ── Public interface ───────────────────────────────────────────────────────

    def __call__(
        self,
        dir_logits: torch.Tensor,                    # (B, 3)
        dir_labels: torch.Tensor,                    # (B,)  ∈ {0,1,2}
        ret_pred: torch.Tensor,                      # (B,)
        ret_labels: torch.Tensor,                    # (B,)
        fac_probs: torch.Tensor,                     # (B, K, C) or (B, C)
        fac_labels: torch.Tensor,                    # (B, K, C) or (B, C)
        attn_weights: torch.Tensor,                  # (B, K)  scaled by K_h
        confidence: torch.Tensor,                    # (B,)  ∈ [0, 1]
        calibration_targets: torch.Tensor = None,   # (B,)  0/1
        masked_dir_logits: torch.Tensor = None,     # (B, 3) for Lfaith
        article_mask: torch.Tensor = None,          # (B, K) 1=real, 0=pad
    ) -> dict:
        """Compute all loss components and weighted total (PDF Eq.30).

        Returns dict with keys: loss, Ldir, Lret, Lfac, Lsel, Lcal, Lfaith, Lrisk.
        """
        device = dir_logits.device

        # ── Valid-sample mask (skip NaN/Inf batches) ──────────────────────────
        invalid = torch.isnan(dir_logits).any(dim=1) | torch.isinf(dir_logits).any(dim=1)
        if invalid.any():
            valid_mask = (~invalid).float()
            if valid_mask.sum() == 0:
                zero = torch.tensor(0.0, device=device, requires_grad=True)
                return {"loss": zero,
                        "Ldir": 0.0, "Lret": 0.0, "Lfac": 0.0, "Lsel": 0.0,
                        "Lcal": 0.0, "Lfaith": 0.0, "Lrisk": 0.0}
        else:
            valid_mask = torch.ones(len(dir_logits), device=device)

        # ── Per-sample CE (cached and reused in Lrisk) ───────────────────────
        ce_per_sample = F.cross_entropy(
            dir_logits, dir_labels, reduction='none'
        )  # (B,) — used by both Ldir and Lrisk

        # ── Ldir (Eq.31) ─────────────────────────────────────────────────────
        # Weighted CE with label smoothing to prevent overconfidence (→ stable ECE).
        Ldir = F.cross_entropy(
            dir_logits, dir_labels,
            weight=self.class_weights,
            reduction='none',
            label_smoothing=0.1,
        ) * valid_mask
        Ldir = Ldir.mean()

        # ── Lret (Eq.32) ─────────────────────────────────────────────────────
        # PDF specifies SmoothL1.  We add a Pearson term and a bias penalty to
        # prevent the degenerate solution of predicting 0 for every return
        # (which zeroes SmoothL1 but leaves Sharpe undefined).
        valid_idx = valid_mask.bool()
        Lret_l1 = (F.smooth_l1_loss(ret_pred, ret_labels, reduction='none') * valid_mask).mean()

        if valid_idx.sum() > 1:
            p, t = ret_pred[valid_idx], ret_labels[valid_idx]
            p_c, t_c = p - p.mean(), t - t.mean()
            pearson = (p_c * t_c).sum() / (p_c.norm() * t_c.norm() + _EPS)
            Lret_corr = 1.0 - pearson                          # ∈ [0, 2]
            Lret_bias = (p.mean() - t.mean()).abs()            # force unbiased predictions
        else:
            Lret_corr = torch.tensor(0.0, device=device)
            Lret_bias = torch.tensor(0.0, device=device)

        Lret = Lret_l1 + _LRET_CORR_W * Lret_corr + _LRET_BIAS_W * Lret_bias

        # ── Lfac (Eq.33) ─────────────────────────────────────────────────────
        # Soft CE: L_fac = -Σ_c ỹ_c · log(p_c).
        # Handles (B, K, C) per-article distributions by masked-averaging over K.
        Lfac = self._compute_lfac(fac_probs, fac_labels, article_mask, valid_mask, device)

        # ── Lsel (Eq.34) ─────────────────────────────────────────────────────
        # L_sel = (Σα̃_i - K_h)² + η · Σα̃_i · log(α̃_i)
        # attn_weights already scaled by K_h in trainer so term-1 ≈ 0 at convergence.
        Lsel = self._compute_lsel(attn_weights, valid_mask, device)

        # ── Lcal (Eq.35) ─────────────────────────────────────────────────────
        # Brier score: L_cal = (ĉ - I[ŷ = y])²
        if calibration_targets is not None:
            conf_c = torch.clamp(confidence, _CONF_MIN, _CONF_MAX)
            Lcal = ((conf_c - calibration_targets.float()) ** 2 * valid_mask).mean()
        else:
            Lcal = torch.tensor(0.0, device=device)

        # ── Lfaith (Eq.36) ───────────────────────────────────────────────────
        # Margin loss: L_faith = max(0, m - (p̂_full - p̂_masked))
        # When masked_dir_logits is unavailable → 0 (undefined by PDF without masked pass).
        if masked_dir_logits is not None:
            full_max   = F.softmax(dir_logits,        dim=-1).max(dim=-1)[0]  # (B,)
            masked_max = F.softmax(masked_dir_logits, dim=-1).max(dim=-1)[0]  # (B,)
            gap    = full_max - masked_max
            Lfaith = (torch.clamp(self.faith_margin - gap, min=0.0) * valid_mask).mean()
        else:
            Lfaith = torch.tensor(0.0, device=device)

        # ── Lrisk (Eq.37) ────────────────────────────────────────────────────
        # L_risk = Σĉ·ℓ / Σĉ  +  μ · max(0, κ - (1/B)·Σĉ)
        # Reuses cached ce_per_sample — no redundant forward pass.
        Lrisk = self._compute_lrisk(ce_per_sample, confidence, valid_mask)

        # ── Total (Eq.30) ─────────────────────────────────────────────────────
        total = (
            self.lambda1 * Ldir   +
            self.lambda2 * Lret   +
            self.lambda3 * Lfac   +
            self.lambda4 * Lsel   +
            self.lambda5 * Lcal   +
            self.lambda6 * Lfaith +
            self.lambda7 * Lrisk
        )

        return {
            "loss":   total,
            "Ldir":   self._to_scalar(Ldir),
            "Lret":   self._to_scalar(Lret),
            "Lfac":   self._to_scalar(Lfac),
            "Lsel":   self._to_scalar(Lsel),
            "Lcal":   self._to_scalar(Lcal),
            "Lfaith": self._to_scalar(Lfaith),
            "Lrisk":  self._to_scalar(Lrisk),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_scalar(x: torch.Tensor | float) -> float:
        """Convert a tensor or float to a Python scalar for logging."""
        return x.item() if isinstance(x, torch.Tensor) else float(x)

    def _compute_lfac(
        self,
        fac_probs: torch.Tensor,
        fac_labels: torch.Tensor,
        article_mask,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Soft cross-entropy for factor prediction (PDF Eq.33)."""
        if fac_probs.dim() == 3:  # (B, K, C)
            if article_mask is not None:
                w = article_mask.float().unsqueeze(-1)          # (B, K, 1)
                n = w.sum(dim=1).clamp(min=1.0)                 # (B, 1)
                fac_probs_avg  = (fac_probs  * w).sum(dim=1) / n  # (B, C)
                fac_labels_avg = (fac_labels * w).sum(dim=1) / n  # (B, C)
            else:
                fac_probs_avg  = fac_probs.mean(dim=1)
                fac_labels_avg = fac_labels.mean(dim=1)
            log_p = F.log_softmax(fac_probs_avg, dim=-1)
            Lfac  = -(fac_labels_avg * log_p).sum(dim=-1) * valid_mask
        elif fac_probs.dim() == 2:  # (B, C)
            if fac_labels.dim() == 1:
                Lfac = F.cross_entropy(fac_probs, fac_labels, reduction='none') * valid_mask
            else:
                log_p = F.log_softmax(fac_probs, dim=-1)
                Lfac  = -(fac_labels * log_p).sum(dim=-1) * valid_mask
        else:
            raise ValueError(f"fac_probs must be 2D or 3D, got {fac_probs.shape}")
        return Lfac.mean()

    def _compute_lsel(
        self,
        attn_weights,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Selection regularization (PDF Eq.34).

        Term 1: (Σα̃_i - K_h)² — attn_weights are pre-scaled by K_h so sum ≈ K_h.
        Term 2: η · Σα̃_i·log(α̃_i) — entropy reward; α̃ renormalized to [0,1] so
                log(α̃_i) ≤ 0, making term 2 ≤ 0 (reward for diversity).
        """
        if attn_weights is None or attn_weights.dim() < 2:
            return torch.tensor(0.0, device=device)

        alpha = attn_weights                                    # (B, K), sum ≈ K_h
        term1 = (alpha.sum(dim=1) - self.K_h) ** 2            # (B,)

        # Renormalize to proper probability simplex for entropy calculation.
        # This is necessary because log(α_i) > 0 when α_i > 1 (after K_h scaling),
        # which would invert the entropy gradient and penalise diversity.
        alpha_norm = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=_EPS)
        alpha_clamped = alpha_norm.clamp(min=_EPS)
        neg_entropy = (alpha_clamped * torch.log(alpha_clamped)).sum(dim=1)  # (B,) ≤ 0

        Lsel = ((term1 + self.eta * neg_entropy) * valid_mask).mean()
        return torch.clamp(Lsel, min=-_LSEL_CLAMP, max=_LSEL_CLAMP)

    def _compute_lrisk(
        self,
        ce_per_sample: torch.Tensor,   # (B,) cached from Ldir forward
        confidence: torch.Tensor,      # (B,)
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Selective risk + coverage penalty (PDF Eq.37).

        L_risk = Σĉ·ℓ / Σĉ  +  μ · max(0, κ - mean(ĉ))

        Confidence clamped to [0.01, 1] so denominator is always >= 0.01*B,
        preventing division instability when the model is initially uncertain.
        """
        conf = confidence.clamp(min=_LRISK_CONF_MIN)  # floor for denominator stability
        denom       = (conf * valid_mask).sum() + _EPS
        risk_term   = (ce_per_sample * conf * valid_mask).sum() / denom
        cov_penalty = torch.clamp(self.coverage_target - conf.mean(), min=0.0)
        return risk_term + self.mu * cov_penalty
