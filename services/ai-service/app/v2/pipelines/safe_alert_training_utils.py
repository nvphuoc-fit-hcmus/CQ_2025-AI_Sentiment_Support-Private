"""
Multi-objective loss functions for SAFE-Alert training (PDF Eq.30-37).

L = λ₁Ldir + λ₂Lret + λ₃Lfac + λ₄Lsel + λ₅Lcal + λ₆Lfaith + λ₇Lrisk

Implemented equations:
  - Ldir  (Eq.31): Direction classification   → cross-entropy
  - Lret  (Eq.32): Return regression          → SmoothL1 on train-scaled returns
  - Lfac  (Eq.33): Factor prediction          → soft cross-entropy
  - Lsel  (Eq.34): Selection regularization   → (Σα̃ - K_h)² + η·Σα̃ log α̃
  - Lcal  (Eq.35): Calibration                → Brier score
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
import torch.nn as nn

class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss aligned to PDF Eq.30-37."""
    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.3,
        lambda4: float = 0.2,
        lambda5: float = 0.1,
        lambda6: float = 0.1,
        lambda7: float = 0.05,
        K_h: int = 8,
        eta: float = 0.1,
        faith_margin: float = 0.15,
        coverage_target: float = 0.35,
        mu: float = 0.02,
        class_weights: torch.Tensor = None,
        learned_lambdas: bool = True,
    ):
        super().__init__()
        # Lambdas as learnable parameters
        if learned_lambdas:
            self.lambda1 = nn.Parameter(torch.tensor(lambda1, dtype=torch.float32))
            self.lambda2 = nn.Parameter(torch.tensor(lambda2, dtype=torch.float32))
            self.lambda3 = nn.Parameter(torch.tensor(lambda3, dtype=torch.float32))
            self.lambda4 = nn.Parameter(torch.tensor(lambda4, dtype=torch.float32))
            self.lambda5 = nn.Parameter(torch.tensor(lambda5, dtype=torch.float32))
            self.lambda6 = nn.Parameter(torch.tensor(lambda6, dtype=torch.float32))
            self.lambda7 = nn.Parameter(torch.tensor(lambda7, dtype=torch.float32))
        else:
            self.register_buffer('lambda1', torch.tensor(lambda1, dtype=torch.float32))
            self.register_buffer('lambda2', torch.tensor(lambda2, dtype=torch.float32))
            self.register_buffer('lambda3', torch.tensor(lambda3, dtype=torch.float32))
            self.register_buffer('lambda4', torch.tensor(lambda4, dtype=torch.float32))
            self.register_buffer('lambda5', torch.tensor(lambda5, dtype=torch.float32))
            self.register_buffer('lambda6', torch.tensor(lambda6, dtype=torch.float32))
            self.register_buffer('lambda7', torch.tensor(lambda7, dtype=torch.float32))
        self.K_h = K_h
        self.eta = eta  # Entropy weight in Lsel (PDF Eq.34): η · Σα̃_i · log(α̃_i)
        self.faith_margin = faith_margin
        self.coverage_target = coverage_target
        self.mu = mu
        self.class_weights = class_weights
        self.register_buffer('return_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('factor_label_smoothing', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('factor_label_mean_max_prob', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('factor_label_mean_entropy', torch.tensor(0.0, dtype=torch.float32))

    # ── Public interface ───────────────────────────────────────────────────────

    def __call__(
        self,
        dir_logits: torch.Tensor,                    # (B, 3)
        dir_labels: torch.Tensor,                    # (B,)  ∈ {0,1,2}
        ret_pred: torch.Tensor,                      # (B,)
        ret_labels: torch.Tensor,                    # (B,)
        fac_probs: torch.Tensor,                     # (B, K, C) or (B, C)
        fac_labels: torch.Tensor,                    # (B, K, C) or (B, C)
        attn_weights: torch.Tensor,                  # (B, K)  scaled by K_h (for entropy term)
        confidence: torch.Tensor,                    # (B,)  ∈ [0, 1]
        calibration_targets: torch.Tensor = None,   # (B,)  0/1
        masked_dir_logits: torch.Tensor = None,     # (B, 3) for Lfaith
        article_mask: torch.Tensor = None,          # (B, K) 1=real, 0=pad
        soft_gates: torch.Tensor = None,            # (B, K) σ(a_i) for Lsel sel_term
    ) -> dict:
        """Compute all loss components and weighted total (PDF Eq.30).

        Returns dict with keys: loss, Ldir, Lret, Lfac, Lsel, Lcal, Lfaith, Lrisk.
        """
        device = dir_logits.device

        # ── Valid-sample mask (skip NaN/Inf batches) ──────────────────────────
        invalid = torch.isnan(dir_logits).any(dim=1) | torch.isinf(dir_logits).any(dim=1)
        invalid |= torch.isnan(ret_pred) | torch.isinf(ret_pred)
        invalid |= torch.isnan(confidence) | torch.isinf(confidence)
        if masked_dir_logits is not None:
            invalid |= torch.isnan(masked_dir_logits).any(dim=1) | torch.isinf(masked_dir_logits).any(dim=1)

        valid_idx = ~invalid
        if valid_idx.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": zero,
                    "Ldir": 0.0, "Lret": 0.0, "Lfac": 0.0, "Lsel": 0.0,
                    "Lcal": 0.0, "Lfaith": 0.0, "Lrisk": 0.0}

        if not torch.all(valid_idx):
            dir_logits = dir_logits[valid_idx]
            dir_labels = dir_labels[valid_idx]
            ret_pred = ret_pred[valid_idx]
            ret_labels = ret_labels[valid_idx]
            fac_probs = fac_probs[valid_idx]
            fac_labels = fac_labels[valid_idx]
            attn_weights = attn_weights[valid_idx] if attn_weights is not None else None
            confidence = confidence[valid_idx]
            calibration_targets = calibration_targets[valid_idx] if calibration_targets is not None else None
            masked_dir_logits = masked_dir_logits[valid_idx] if masked_dir_logits is not None else None
            article_mask = article_mask[valid_idx] if article_mask is not None else None

        valid_mask = torch.ones(len(dir_logits), device=device)

        # ── Ldir (Eq.31): CE with sqrt-softened class weights (handles UP/DOWN/NEUTRAL imbalance)
        cw = self.class_weights.to(dir_logits.device) if self.class_weights is not None else None
        Ldir = F.cross_entropy(dir_logits, dir_labels, weight=cw, reduction='mean')

        # ── Unweighted per-sample CE for Lrisk (Eq.37) ───────────────────────
        # Lrisk measures prediction quality on abstaining samples — class weights would
        # bias confidence toward minority classes instead of measuring raw prediction quality.
        ce_per_sample = F.cross_entropy(
            dir_logits, dir_labels, reduction='none'
        )  # (B,) — used by Lrisk only

        # Lret (Eq.32): SmoothL1 on train-normalized returns for stable optimization.
        scale = self.return_scale.clamp(min=1e-6).to(device)
        Lret = (
            F.smooth_l1_loss(ret_pred / scale, ret_labels / scale, reduction='none') * valid_mask
        ).mean()

        # ── Lfac (Eq.33) ─────────────────────────────────────────────────────
        # Soft CE: L_fac = -Σ_c ỹ_c · log(p_c).
        # Handles (B, K, C) per-article distributions by masked-averaging over K.
        Lfac = self._compute_lfac(
            fac_probs, fac_labels, article_mask, valid_mask, device
        )

        # Lsel (Eq.34): (Σα̃_i - K_h)² + η·Σα̃_i·log(α̃_i)
        # sel_term uses sigmoid gates (real gradient); entropy uses softmax weights
        Lsel = self._compute_lsel_pdf(attn_weights, article_mask, valid_mask, device, soft_gates)

        # ── Lcal (Eq.35) ─────────────────────────────────────────────────────
        # Brier score: L_cal = (ĉ - I[ŷ = y])²
        if calibration_targets is not None:
            conf_c = torch.clamp(confidence, _CONF_MIN, _CONF_MAX)
            Lcal = ((conf_c - calibration_targets.float()) ** 2 * valid_mask).mean()
        else:
            Lcal = torch.tensor(0.0, device=device)

        # ── Lfaith (Eq.36) ───────────────────────────────────────────────────
        # Margin loss: L_faith = max(0, m - (p̂_full - p̂_masked))
        # Only computed for samples that actually have articles — for no-article samples
        # both full and masked predictions are market-only (gap≈0), contributing a constant
        # offset with zero gradient that distorts the mean without providing signal.
        if masked_dir_logits is not None:
            full_max   = F.softmax(dir_logits,        dim=-1).max(dim=-1)[0]  # (B,)
            masked_max = F.softmax(masked_dir_logits, dim=-1).max(dim=-1)[0]  # (B,)
            gap    = full_max - masked_max
            # Per-sample mask: only count samples with actual articles
            has_art = (
                article_mask.any(dim=1).float()
                if article_mask is not None
                else valid_mask
            )
            n_art = has_art.sum().clamp(min=1.0)
            Lfaith = (torch.clamp(self.faith_margin - gap, min=0.0) * has_art).sum() / n_art
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
        return x.item() if isinstance(x, torch.Tensor) else float(x)

    def _compute_lsel_pdf(self, attn_weights, article_mask, valid_mask, device, soft_gates=None):
        # PDF Eq.34: L_sel = (Σα̃_i - K_h)² + η * Σ α̃_i log α̃_i
        # In sparse-news samples, the effective target is min(K_h, n_valid_articles).
        if attn_weights is None and soft_gates is None:
            return torch.tensor(0.0, device=device)

        # ── sel_term ───────────────────────────────────────────────────────────
        # BUG (old): attn_weights = K_h * softmax → Σα̃ = K_h always → sel_term = 0.
        # FIX: use sigmoid(raw_scores) = soft_gates; Σσ(a_i) is NOT constant so
        # sel_term = (Σσ(a_i) - K_h)² carries a real gradient that trains the
        # attention scorer to activate approximately K_h articles per sample.
        if soft_gates is not None:
            # soft_gates (B, K): already masked by article_mask in model forward
            # Target = min(n_valid_articles, K_h) to avoid over-penalising sparse samples
            if article_mask is not None:
                target_k = article_mask.float().sum(dim=1).clamp(max=float(self.K_h))
            else:
                target_k = torch.full(
                    (soft_gates.shape[0],), float(self.K_h), device=device, dtype=soft_gates.dtype
                )
            sel_term = (soft_gates.sum(dim=1) - target_k) ** 2
        else:
            # Fallback (no soft_gates): sel_term = 0 (old structural limitation)
            if article_mask is not None:
                target_k = article_mask.float().sum(dim=1).clamp(max=float(self.K_h))
            else:
                target_k = torch.full(
                    (attn_weights.shape[0],), float(self.K_h), device=device, dtype=attn_weights.dtype
                )
            sel_term = (attn_weights.sum(dim=1) - target_k) ** 2  # = 0 always (structural)

        # ── entropy term (on normalized softmax α̃, unchanged) ─────────────────
        if attn_weights is not None:
            row_sum = attn_weights.sum(dim=1, keepdim=True)
            alpha_norm = attn_weights / row_sum.clamp(min=_EPS)
            alpha_clamped = alpha_norm.clamp(min=_EPS)
            neg_entropy = (alpha_clamped * torch.log(alpha_clamped)).sum(dim=1)
            neg_entropy = torch.where(
                row_sum.squeeze(1) > _EPS, neg_entropy, torch.zeros_like(neg_entropy)
            )
        else:
            neg_entropy = torch.zeros(sel_term.shape[0], device=device)

        lsel = ((sel_term + self.eta * neg_entropy) * valid_mask).mean()
        return torch.clamp(lsel, min=-_LSEL_CLAMP, max=_LSEL_CLAMP)

    def _compute_lfac(
        self,
        fac_probs: torch.Tensor,
        fac_labels: torch.Tensor,
        article_mask,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Soft cross-entropy for factor prediction (PDF Eq.33)."""
        smoothing = float(self.factor_label_smoothing.item())

        def _smooth_soft_labels(labels: torch.Tensor) -> torch.Tensor:
            if smoothing <= 0.0 or labels.dim() == 1:
                return labels
            num_classes = labels.shape[-1]
            uniform = 1.0 / max(num_classes, 1)
            smoothed = (1.0 - smoothing) * labels + smoothing * uniform
            return smoothed / smoothed.sum(dim=-1, keepdim=True).clamp(min=_EPS)

        if fac_probs.dim() == 3:  # (B, K, C)
            # PDF Eq.33: L_fac = -Σ_i Σ_c ỹ_c · log p_c
            # Correct: compute per-article log_softmax and CE first, then masked average.
            # Wrong (Jensen's inequality): average logits first then log_softmax gives
            # log_softmax(mean(p_i)) != mean(log_softmax(p_i)), weakening gradients by ~1/K.
            log_p = F.log_softmax(fac_probs, dim=-1)           # (B, K, C) per-article
            fac_labels_s = _smooth_soft_labels(fac_labels)     # smooth before CE
            per_article_ce = -(fac_labels_s * log_p).sum(dim=-1)  # (B, K)
            if article_mask is not None:
                w = article_mask.float()                        # (B, K)
                n = w.sum(dim=1).clamp(min=1.0)                # (B,)
                Lfac = (per_article_ce * w).sum(dim=1) / n     # (B,)
            else:
                Lfac = per_article_ce.mean(dim=1)              # (B,)
            Lfac = Lfac * valid_mask
        elif fac_probs.dim() == 2:  # (B, C)
            if fac_labels.dim() == 1:
                Lfac = F.cross_entropy(fac_probs, fac_labels, reduction='none') * valid_mask
            else:
                # fac_labels may be (B, K, C) when model is in market-only mode (p_fac_all=None).
                # Reduce to (B, C) by averaging over K before computing soft CE.
                if fac_labels.dim() == 3:
                    fac_labels = fac_labels.mean(dim=1)  # (B, C)
                fac_labels = _smooth_soft_labels(fac_labels)
                log_p = F.log_softmax(fac_probs, dim=-1)
                # PDF Eq.33: L_fac = -Σ_c ỹ_c · log p_c (soft CE, no per-sample weighting)
                Lfac  = -(fac_labels * log_p).sum(dim=-1) * valid_mask
        else:
            raise ValueError(f"fac_probs must be 2D or 3D, got {fac_probs.shape}")
        return Lfac.mean()

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
