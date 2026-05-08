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

import math

import numpy as np
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


class BalancedBatchSampler:
    """Sprint 7 — class-balanced batch sampler for direction-class stochastic stability.

    Each batch contains ceil(B/num_classes) samples from each direction class
    (e.g. B=8, classes=3 → quota [3, 3, 2]), with the "+1" rotated across batches
    so total per-class draw stays balanced over an epoch. Within each class,
    indices are shuffled per epoch (deterministic via seed+epoch). Replacement
    only kicks in when a class exhausts its samples (DOWN/NEUTRAL hit this in
    fold 1 around batch ~3k; UP doesn't).

    Rationale (Sprint 7 vs Sprint 6 Phase 8 closure):
        At B=8 with ~33/33/33 class balance under shuffle=True, the probability
        a batch is missing AT LEAST ONE direction class is

            P(missing ≥1 class) = 3·(2/3)^8 - 3·(1/3)^8 ≈ 11.7 %

        With ~3500 batch/epoch (28k samples / B=8), that's ~411 class-starved
        batches/epoch — direction head sees 2-class gradient on those steps.
        Combined with intra-fold regime heterogeneity (BTC 1h fold 1 spans
        bull 2017, crash 2018, sideways 2019, COVID 2020) the stochastic noise
        on direction-class gradients is enough to bias the head toward NEUTRAL.
        Balanced sampler ensures 3-class gradient at every step.

    Scope: train fold ONLY. Val/test loaders MUST stay shuffle=False
    (chronological per P1 #14, preserves walk-forward + embargo protocol).

    Args:
        labels: 1D array of class labels for each sample in the dataset/Subset,
                indexed by the loader's index space. -1 entries are filtered
                out (sentinel from SAFEAlertDataset.get_direction_labels for
                samples where label computation failed).
        batch_size: target batch size. Same as DataLoader's batch_size to
                preserve compute budget (batches_per_epoch ≈ N // B).
        num_classes: number of classes (default 3 for D/N/U).
        seed: RNG seed; the actual seed used per epoch is seed + epoch_count.
    """

    def __init__(self, labels, batch_size: int, num_classes: int = 3, seed: int = 42):
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.epoch = 0

        # Filter out sentinel -1 (label-compute-failed samples).
        valid_mask = labels >= 0
        valid_positions = np.nonzero(valid_mask)[0]
        valid_labels = labels[valid_mask]

        # Class indices stored in the loader's index space (post-Subset positions).
        self.class_indices = [
            valid_positions[valid_labels == c] for c in range(num_classes)
        ]
        self.class_counts = [int(len(idx)) for idx in self.class_indices]

        if any(cnt == 0 for cnt in self.class_counts):
            raise ValueError(
                f"BalancedBatchSampler: at least one class has zero samples "
                f"(class_counts={self.class_counts}). Cannot balance."
            )

        # Per-batch quota: split batch_size as evenly as possible across classes.
        # batch=8, classes=3 → [3, 3, 2]. batch=6, classes=3 → [2, 2, 2].
        base = self.batch_size // self.num_classes
        rem = self.batch_size % self.num_classes
        self.base_quota = [base + (1 if i < rem else 0) for i in range(num_classes)]

        # Match DataLoader(drop_last=False) batch count from the shuffle=True path.
        # The final sampler batch is still full-size via mild replacement.
        self.batches_per_epoch = math.ceil(sum(self.class_counts) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        shuffled = [rng.permutation(idx) for idx in self.class_indices]
        cursors = [0] * self.num_classes

        for batch_idx in range(self.batches_per_epoch):
            # Rotate which class gets the larger quota (only matters when
            # base_quota has unequal entries, e.g. batch=8 → [3,3,2]).
            rot = batch_idx % self.num_classes
            quota = [self.base_quota[(c - rot) % self.num_classes]
                     for c in range(self.num_classes)]
            batch = []
            for c in range(self.num_classes):
                need = quota[c]
                while need > 0:
                    avail = len(shuffled[c]) - cursors[c]
                    if avail >= need:
                        batch.extend(int(x) for x in
                                     shuffled[c][cursors[c]:cursors[c] + need])
                        cursors[c] += need
                        need = 0
                    else:
                        # Drain remainder, then reshuffle for replacement draw.
                        if avail > 0:
                            batch.extend(int(x) for x in shuffled[c][cursors[c]:])
                            need -= avail
                        shuffled[c] = rng.permutation(self.class_indices[c])
                        cursors[c] = 0
            # Shuffle within batch so order isn't always [class0, class0, class1, ...].
            rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss aligned to PDF Eq.30-37."""
    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 0.5,
        lambda3: float = 0.3,
        lambda4: float = 0.2,
        # Session 26 defaults aligned with train_safe_alert.py Stage-3 λ
        # (see DEVIATIONS §L11). Previously 0.1 / 0.1 / 0.05 underweighted
        # Lcal/Lfaith/Lrisk to ~1 % of total loss — decorative rather than
        # supervisory. New values target ≥5 % contribution each.
        lambda5: float = 0.25,
        lambda6: float = 0.40,
        lambda7: float = 0.10,
        K_h: int = 8,
        eta: float = 0.18,
        faith_margin: float | dict = 0.15,
        lfaith_gap_type: str = "relative",   # "relative" (engineering) | "absolute" (paper-literal Eq.36)
        coverage_target: float = 0.35,
        mu: float = 0.02,
        class_weights: torch.Tensor = None,
        # Session 21 audit fix: default lowered True → False to match paper spec.
        # Paper Eq.30 specifies FIXED λ weights per stage (curriculum design,
        # Section 4.5.1). learned_lambdas=True lets torch.optim treat each λ as
        # a trainable parameter, which would let them drift to the minimum-loss
        # configuration (trivially: push all λ→0) and break the curriculum.
        # The trainer already passes False explicitly (train_safe_alert.py:421),
        # so this default change is defence-in-depth for callers that
        # instantiate the loss directly (tests, notebooks, ablation scripts).
        learned_lambdas: bool = False,
        label_smoothing_eps: float = 0.05,   # L_dir smoothing to curb overconfidence
        focal_gamma: float = 1.0,             # focal re-weighting exponent (0 = plain CE)
        entropy_anchor_weight: float = 0.0,   # R3 #E: explicit anti-overconfidence term
        # Session 21 Fix D: default target lowered 0.95 → 0.50.
        # At 0.95·ln(3)=1.044 out of max 1.099, the anchor forced predictions
        # within 5% of uniform — the model could never be confident, post-hoc
        # temperature scaler then fitted T up to 4.2 on validation to compensate
        # (the opposite of the anchor's stated goal). 0.50·ln(3)=0.55 permits
        # max-prob ≈ 0.70 while still catching true collapse to one-hot.
        entropy_anchor_target: float = 0.50,
        # Session 22 Fix 8: logit-magnitude L2 penalty weight.
        # Adds (1e-4 by default) × mean(dir_logits²) to L_dir, penalising
        # raw logit magnitude directly. Targets the root cause of post-hoc
        # temperature drift: once logits exceed the CE-optimal magnitude
        # (~log((1-ε)/ε/(C-1)) ≈ 3.64 for ε=0.05, C=3), additional growth
        # hurts calibration without improving accuracy. Set to 0 to disable.
        logit_l2_weight: float = 1e-4,
        # Volatility-regression extension weight. Paper section 4.1.4 stores
        # future volatility as a benchmark field, but Eq.30 does not include
        # Lvol. Paper-final default is 0.0; >0 opts into the extension.
        lambda_vol: float = 0.0,
        # Sprint 3.3: keep the sign auxiliary, but mask noisy near-zero returns.
        lret_sign_weight: float = 0.1,
        lret_sign_eps_factor: float = 1.0,
        # Sprint 4 Phase 8.5 — optional ε_h override mirrored from
        # SAFEAlertDataset. None → paper canonical via
        # constants.get_epsilon_h(horizon). When set, set_horizon() uses
        # the override so the Lret_sign nonzero mask stays aligned with
        # the dataset's direction-label boundary.
        epsilon_h_override: "float | None" = None,
        # Sprint 4-D — auxiliary return-sign classification head weight.
        # 0.0 disables (default, preserves backward compat). Set >0 to
        # enable BCE on (ret_sign_logit, sign(ret_label)>0) masked to
        # samples with |ret_label| > eps_h. Pairs with the new
        # ret_sign_head added to SAFEAlertNet (Sprint 4-D Phase 8.7
        # follow-up: ret_pred regression collapsed to near-zero, so
        # this orthogonal binary head supplies clean sign-correctness
        # gradient via BCEWithLogitsLoss instead of bottlenecking
        # through the magnitude-regression Lret).
        lambda_ret_sign_cls: float = 0.0,
        # Sprint 5 — auxiliary tradability bin classification head weight.
        # 0.0 disables (default, preserves backward compat). When > 0, the
        # head's 3-class output (no_edge / marginal_edge / strong_edge) is
        # trained by CrossEntropyLoss against bin labels derived inline from
        # |ret_labels| vs ε_h thresholds. Direction-INDEPENDENT magnitude
        # classification providing a tradability signal the policy can use
        # to filter trades. Sprint 5a curriculum target: 0.10 in Stage 3
        # with linear ramp from 0.0 (S1) → 0.05 (S2) → 0.10 (S3). See
        # safe_alert_net.py ret_bin_head for matching architecture.
        lambda_ret_bin: float = 0.0,
        # Sprint 9A - auxiliary directional edge heads. Defaults keep this
        # additive wiring a no-op until the YAML explicitly enables it.
        lambda_up_edge: float = 0.0,
        lambda_down_edge: float = 0.0,
        # Session 26 — paper Eq.33 Lfac reduction mode across articles.
        #   "mean" (default): (1/N_{s,t}) · Σ_i Σ_c ỹ_{i,c} log p_{i,c}
        #     — magnitude stable as N varies (1-50 articles/candle for BTC);
        #     matches engineering convention for per-sample losses.
        #   "sum"  (paper-literal): Σ_i Σ_c ỹ_{i,c} log p_{i,c}
        #     — matches Eq.33 verbatim; scales with N_{s,t} per sample so
        #     factor loss weight varies 50× across samples, requires λ3
        #     retuning (≈ λ3 / mean(N) to preserve current gradient scale).
        # "mean" is kept as default because empirical λ3=0.3 was tuned
        # against it; "sum" is exposed for paper-strict reproduction runs.
        # See DEVIATIONS.md D16 for full rationale.
        lfac_reduction: str = "mean",
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
        self.eta = eta  # Entropy weight in Lsel (PDF Eq.34): η · Σα̃_i · log(α̃_i).
                        # Default 0.18: stronger entropy penalty concentrates
                        # attention weight on 1-2 top articles, ensuring a
                        # measurable faith gap (Del/Ins). Lower η lets the
                        # entropy term dominate and diffuses attention uniformly
                        # over K_h — selection no longer carries evidence signal.
        # faith_margin may be a scalar or a per-horizon dict. Short horizons (15m)
        # have smaller signal gaps than longer ones (24h), so a single margin
        # either over-penalises short horizons or under-constrains long ones.
        # Callers can switch the active horizon via ``set_horizon("24h")``.
        self._faith_margin_spec = faith_margin
        self._active_horizon: str | None = None
        self.faith_margin = self._resolve_faith_margin(None)
        self.ins_margin = self._resolve_ins_margin(None)
        # Lfaith gap type selector.
        # "relative" (default, engineering): (p_full - p_mask) / p_full, clipped to [0, 1].
        #     Equalises difficulty across confidence regimes — see Lfaith comment below.
        # "absolute" (paper-literal Eq.36): p_full - p_mask, clipped to [0, 1].
        #     Strict reproduction of paper text. Margins per-horizon should be retuned
        #     to absolute scale (~0.15) when this mode is selected.
        if lfaith_gap_type not in ("relative", "absolute"):
            raise ValueError(
                f"lfaith_gap_type must be 'relative' or 'absolute', got {lfaith_gap_type!r}"
            )
        self.lfaith_gap_type = lfaith_gap_type
        self.coverage_target = coverage_target
        self.mu = mu
        self.class_weights = class_weights
        # L_dir calibration helpers:
        #   label_smoothing_eps: cap target softmax at 1-eps to curb overconfidence
        #     and reduce post-hoc temperature-scaling reliance (root-cause fix
        #     for ECE drift during training).
        #   focal_gamma: (1 - p_t)^γ factor concentrating gradient on hard samples,
        #     boosting MCC which is sensitive to minority-class performance.
        self.label_smoothing_eps = float(label_smoothing_eps)
        self.focal_gamma = float(focal_gamma)
        # R3 #E + Session 21 Fix D: Entropy-anchor penalty against temperature drift.
        # History: initially introduced because Fold 1 showed temperature (post-hoc
        # Platt T) drifting from 0.65 at epoch 1 up to 2.6 by epoch 39. The first
        # implementation used target_frac=0.95 which over-corrected — Session 20
        # still observed T drifting to 4.2 because the anchor forced predictions
        # into the near-uniform region and temperature scaling compensated.
        # Session 21 lowers default target_frac to 0.50: permits moderate
        # confidence (max-prob up to ~0.70) while still penalising the collapse-
        # to-one-hot that caused the original T drift. Penalty is
        #   weight × relu(target_H − H(p̂))
        # where target_H = target_frac × log(n_classes). For 3 classes
        # log(3) ≈ 1.0986, so target 0.50 × log(3) ≈ 0.55.
        # Weight starts at 0 to preserve backward compat; tune via config.
        self.entropy_anchor_weight = float(entropy_anchor_weight)
        self.entropy_anchor_target_frac = float(entropy_anchor_target)
        # Session 22 Fix 8: see class docstring block above. Stored as float
        # so the forward path can read it cheaply without tensor allocation.
        self.logit_l2_weight = float(logit_l2_weight)
        # Volatility-extension weight. Stored as a plain float so the forward
        # path branch `if lambda_vol > 0` is a cheap Python-side short-circuit.
        self.lambda_vol = float(lambda_vol)
        self.lret_sign_weight = float(lret_sign_weight)
        self.lret_sign_eps_factor = float(lret_sign_eps_factor)
        if not (0.0 <= self.lret_sign_weight <= 1.0):
            raise ValueError(
                f"lret_sign_weight must be in [0, 1], got {self.lret_sign_weight}"
            )
        if not (0.1 <= self.lret_sign_eps_factor <= 10.0):
            raise ValueError(
                "lret_sign_eps_factor must be in [0.1, 10], "
                f"got {self.lret_sign_eps_factor}"
            )
        # Sprint 4 Phase 8.5 — store override for later use in set_horizon()
        # and as the source of self.epsilon_h. None preserves paper-canonical
        # behaviour. Range validation lives in constants.get_epsilon_h.
        self._epsilon_h_override = (
            float(epsilon_h_override) if epsilon_h_override is not None else None
        )
        # Sprint 4-D — auxiliary return-sign classification head weight.
        # Stored as plain float so the forward branch `if > 0` is a cheap
        # Python short-circuit (no tensor allocation when disabled).
        self.lambda_ret_sign_cls = float(lambda_ret_sign_cls)
        if not (0.0 <= self.lambda_ret_sign_cls <= 5.0):
            raise ValueError(
                f"lambda_ret_sign_cls must be in [0, 5], got {self.lambda_ret_sign_cls}"
            )
        # Sprint 5 — tradability bin head weight. Same Python-short-circuit
        # storage pattern as lambda_ret_sign_cls; bound check prevents the
        # auxiliary loss from accidentally dominating the total.
        self.lambda_ret_bin = float(lambda_ret_bin)
        if not (0.0 <= self.lambda_ret_bin <= 5.0):
            raise ValueError(
                f"lambda_ret_bin must be in [0, 5], got {self.lambda_ret_bin}"
            )
        # Sprint 9A - binary UP/DOWN edge head weights. Stored as floats so
        # inactive heads are a cheap Python-side no-op.
        self.lambda_up_edge = float(lambda_up_edge)
        self.lambda_down_edge = float(lambda_down_edge)
        if not (0.0 <= self.lambda_up_edge <= 5.0):
            raise ValueError(
                f"lambda_up_edge must be in [0, 5], got {self.lambda_up_edge}"
            )
        if not (0.0 <= self.lambda_down_edge <= 5.0):
            raise ValueError(
                f"lambda_down_edge must be in [0, 5], got {self.lambda_down_edge}"
            )
        # Session 26 — paper Eq.33 Lfac reduction ("mean" | "sum").
        _valid_lfac = {"mean", "sum"}
        if lfac_reduction not in _valid_lfac:
            raise ValueError(
                f"lfac_reduction must be one of {_valid_lfac}, got {lfac_reduction!r}"
            )
        self.lfac_reduction = str(lfac_reduction)

        # Session 26 — per-factor class weights for Lfac (paper Eq.33 + our
        # class-imbalance fix). Populated by trainer via set_factor_class_weights()
        # after the dataset is constructed. When None, Lfac operates with
        # uniform per-class weighting (original paper behaviour).
        self.register_buffer(
            'factor_class_weights',
            torch.ones(1, dtype=torch.float32),   # placeholder; set via setter
            persistent=False,
        )
        self._has_factor_cw = False
        self.register_buffer('return_scale', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('factor_label_smoothing', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('factor_label_mean_max_prob', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('factor_label_mean_entropy', torch.tensor(0.0, dtype=torch.float32))

    # ── Horizon-aware faith margin ─────────────────────────────────────────────

    _DEFAULT_FAITH_MARGIN_BY_HORIZON = {
        # RELATIVE deletion margin (P1 #8): after normalization, the target is
        # "removing the selected evidence must drop the predicted class
        # probability by at least X% OF ITS CURRENT VALUE". Values scale with
        # horizon because the news influence on price is longer-tailed at
        # longer horizons (more headroom for evidence to matter).
        #   15m: 8%   1h: 12%   4h: 15%   24h: 18%
        # These are achievable across confidence regimes because the margin is
        # relative — a confident sample (p_full=0.9) needs gap≥0.108, an
        # uncertain one (p_full=0.4) needs gap≥0.048.
        "15m": 0.08,
        "1h":  0.12,
        "4h":  0.15,
        "24h": 0.18,
    }
    # Insertion margin — used for (p_selected_only − p_no_articles).
    # Insertion directly measures how much the selected articles contribute
    # over a market-only baseline; the margin here must be *achievable*
    # given the news signal strength for the horizon.
    _DEFAULT_INS_MARGIN_BY_HORIZON = {
        "15m": 0.05,
        "1h":  0.08,
        "4h":  0.10,
        "24h": 0.12,
    }
    _DEFAULT_EPS_H_BY_HORIZON = {
        "15m": 0.001,
        "1h":  0.002,
        "4h":  0.005,
        "24h": 0.010,
    }

    def _resolve_faith_margin(self, horizon: str | None) -> float:
        spec = self._faith_margin_spec
        if isinstance(spec, dict):
            if horizon is not None and horizon in spec:
                return float(spec[horizon])
            # dict given but no match → fall back to 1h or mean
            if "1h" in spec:
                return float(spec["1h"])
            return float(sum(spec.values()) / max(len(spec), 1))
        return float(spec)

    def _resolve_ins_margin(self, horizon: str | None) -> float:
        tbl = self._DEFAULT_INS_MARGIN_BY_HORIZON
        if horizon is not None and horizon in tbl:
            return float(tbl[horizon])
        return float(tbl["1h"])

    def set_horizon(self, horizon: str) -> None:
        """Switch the active faith margin for the given horizon (training loop
        should call this when the horizon of the current batch changes)."""
        self._active_horizon = horizon
        self.faith_margin = self._resolve_faith_margin(horizon)
        self.ins_margin = self._resolve_ins_margin(horizon)
        # Sprint 4 Phase 8.5 — single source of truth via constants.get_epsilon_h.
        # When epsilon_h_override is set, the Lret_sign nonzero mask uses the
        # same boundary as the dataset's direction-label generator, so samples
        # in [override, paper] band are not asymmetrically supervised.
        from constants import get_epsilon_h
        self.epsilon_h = get_epsilon_h(horizon, self._epsilon_h_override)

    def set_factor_class_weights(self, weights) -> None:
        """Session 26 — install per-factor class weights for Lfac.

        Weights vector shape (C,) — typically sqrt-softened inverse-frequency
        balanced and mean-normalised (mean == 1). When installed, Lfac is
        computed as -Σ_c w_c * ỹ_c * log p_c so rare factors receive more
        gradient than they would under uniform weighting. Pass None to disable
        (returns to paper's original uniform-weighted Eq.33).

        Device handling: the replacement tensor inherits the current buffer's
        device so that if the caller has already moved the module to GPU via
        ``self.to(device)``, the class weights stay on GPU. Without this, a
        CPU numpy → cpu tensor → buffer assignment would silently demote the
        buffer back to CPU after the initial .to(device).
        """
        target_device = self.factor_class_weights.device
        if weights is None:
            self._has_factor_cw = False
            self.factor_class_weights = torch.ones(1, dtype=torch.float32, device=target_device)
            return
        w = torch.as_tensor(weights, dtype=torch.float32)
        if w.ndim != 1:
            raise ValueError(f"factor_class_weights must be 1D (C,), got {tuple(w.shape)}")
        # Register as buffer so it moves with .to(device) and .cuda()/.cpu().
        # Clone + move to current buffer's device so GPU pinning survives.
        self.factor_class_weights = w.detach().clone().to(target_device)
        self._has_factor_cw = True

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
        vol_pred: torch.Tensor = None,              # (B,) optional volatility extension
        vol_labels: torch.Tensor = None,            # (B,)  realised volatility ≥ 0
        ret_sign_logit: torch.Tensor = None,        # (B,)  Sprint 4-D auxiliary, raw logit pre-sigmoid
        ret_bin_logits: torch.Tensor = None,        # (B,3) Sprint 5 tradability bin logits
        up_edge_logit: torch.Tensor = None,         # (B,) Sprint 9A UP-vs-rest logit
        down_edge_logit: torch.Tensor = None,       # (B,) Sprint 9A DOWN-vs-rest logit
    ) -> dict:
        """Compute all loss components and weighted total (PDF Eq.30).

        Returns dict with keys: loss, Ldir, Lret, Lfac, Lsel, Lcal, Lfaith,
        Lrisk, and (when vol_pred provided) Lvol.
        """
        device = dir_logits.device

        # ── Valid-sample mask (skip NaN/Inf batches) ──────────────────────────
        invalid = torch.isnan(dir_logits).any(dim=1) | torch.isinf(dir_logits).any(dim=1)
        invalid |= torch.isnan(ret_pred) | torch.isinf(ret_pred)
        invalid |= torch.isnan(confidence) | torch.isinf(confidence)
        if ret_sign_logit is not None:
            invalid |= torch.isnan(ret_sign_logit) | torch.isinf(ret_sign_logit)
        if ret_bin_logits is not None:
            invalid |= torch.isnan(ret_bin_logits).any(dim=-1) | torch.isinf(ret_bin_logits).any(dim=-1)
        if up_edge_logit is not None:
            invalid |= torch.isnan(up_edge_logit) | torch.isinf(up_edge_logit)
        if down_edge_logit is not None:
            invalid |= torch.isnan(down_edge_logit) | torch.isinf(down_edge_logit)
        if masked_dir_logits is not None:
            invalid |= torch.isnan(masked_dir_logits).any(dim=1) | torch.isinf(masked_dir_logits).any(dim=1)

        valid_idx = ~invalid
        if valid_idx.sum() == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return {"loss": zero,
                    "Ldir": 0.0, "Lret": 0.0, "Lret_sign": 0.0,
                    "Lfac": 0.0, "Lsel": 0.0,
                    "Lcal": 0.0, "Lfaith": 0.0, "Lrisk": 0.0,
                    # Optional volatility extension: include a zeroed key so
                    # downstream unpack stays consistent on invalid batches.
                    "Lvol": 0.0,
                    "LretCls": 0.0,
                    "LretBin": 0.0,
                    "LupEdge": 0.0,
                    "LdownEdge": 0.0}

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
            # Sprint 4-D — keep ret_sign_logit aligned with the filtered batch
            # so LretCls indexes into the same sample subset as Lret/Lret_sign.
            ret_sign_logit = ret_sign_logit[valid_idx] if ret_sign_logit is not None else None
            # Sprint 5 — same alignment for ret_bin_logits.
            ret_bin_logits = ret_bin_logits[valid_idx] if ret_bin_logits is not None else None
            up_edge_logit = up_edge_logit[valid_idx] if up_edge_logit is not None else None
            down_edge_logit = down_edge_logit[valid_idx] if down_edge_logit is not None else None

        valid_mask = torch.ones(len(dir_logits), device=device)

        # ── Ldir (Eq.31): Focal-softened CE with label smoothing + class weights ─
        # Three independently-tunable regularisers cooperate on the direction
        # head. Each targets a DIFFERENT failure mode and can be disabled on
        # its own by setting the matching ctor arg to 0 for ablation:
        #
        #   (a) label_smoothing (default 0.05):
        #       Caps optimal softmax at 1-ε so the log-likelihood does not
        #       chase 1.0 and blow up logit magnitude. Primary defence against
        #       calibration drift (ECE ↑ over training).
        #
        #   (b) focal_gamma (default 1.0):
        #       (1-p_t)^γ factor down-weights already-confident samples so
        #       gradient concentrates on hard UP/DOWN minority. Primarily
        #       lifts MCC on class-imbalanced crypto direction labels.
        #
        #   (c) class_weights (sqrt-softened balanced, fit from train split):
        #       Scales the per-class CE by inverse class frequency. Addresses
        #       residual imbalance that (b) alone cannot — minority classes
        #       with *already-low* p_t are still few in number.
        #
        # The three DO overlap (all upweight rare/hard cases) but target
        # different mechanisms (target smoothing / sample reweighting /
        # class reweighting). The ablation in Table 5 reports
        # Macro-F1 + ECE for {none, LS-only, focal-only, LS+focal, full} so
        # the combined effect is attributed, not implicit. Paper Section 4.5.1
        # must mention LS=0.05 and γ=1.0 to keep the write-up faithful to code.
        cw = self.class_weights.to(dir_logits.device) if self.class_weights is not None else None
        ce_for_ldir = F.cross_entropy(
            dir_logits, dir_labels,
            weight=cw,
            label_smoothing=self.label_smoothing_eps,
            reduction='none',
        )  # (B,)
        # Session 32 BUG FIX: focal p_t was computed from weighted+smoothed CE
        # via exp(-CE), which is NOT P(correct_class) when LS>0 or cw != 1.
        #   CE_smoothed = -(1-ε)·log(p_c) - (ε/(K-1))·Σ_{≠c} log(p_j)
        #   CE_weighted = w[y]·CE_smoothed
        #   exp(-CE_weighted_smoothed) ≠ p_c
        # → focal factor (1-p_t)^γ becomes wrong → hard-sample mining fails,
        # minority class (UP/DOWN) not up-weighted → F1 stuck at NEUTRAL baseline
        # (observed 0.211 across Session 28-31 smokes).
        # Lin et al. 2017 (Focal Loss, Eq.5) defines p_t = softmax(logits)[y],
        # independent of any LS/CW transformations applied to the CE term.
        if self.focal_gamma > 0.0:
            probs = F.softmax(dir_logits, dim=-1)
            p_t = probs.gather(1, dir_labels.unsqueeze(1)).squeeze(1).clamp(
                min=1e-8, max=1.0 - 1e-6
            )
            focal_factor = (1.0 - p_t) ** self.focal_gamma
            Ldir = (focal_factor * ce_for_ldir).mean()
        else:
            Ldir = ce_for_ldir.mean()

        # R3 #E + Session 21 Fix B/D: Entropy-anchor anti-overconfidence term.
        # Computes H(p̂) = -Σ p̂_c log p̂_c and penalises deviation BELOW the
        # configured fraction of log(C). Two corrections vs. earlier versions:
        #
        #  (D) Default target_frac lowered from 0.95 → 0.50. With C=3 classes
        #      target=0.95·ln(3)=1.044 out of max ln(3)=1.099 was forcing
        #      predictions within 5% of the uniform distribution, i.e. the
        #      model was being penalised for being ANY confident. The post-hoc
        #      temperature scaler then compensated by fitting T up to 4.2 on
        #      validation — that's the "anchor is doing the opposite of
        #      intended" failure observed in Fold 1. 0.50 anchors the floor at
        #      "predictions at least 50% as diffuse as uniform", which permits
        #      max-prob ≈ 0.70 and still catches true collapse to one-hot.
        #
        #  (B) Effective weight decays linearly with λ5 once Lcal is active.
        #      Anchor and Lcal target the same failure mode (overconfidence);
        #      running both at full strength in Stage 3 creates gradient
        #      conflict because Lcal pushes conf=P(correct) (variable) while
        #      the anchor pushes H toward a fixed floor (constant). We let
        #      Lcal take over smoothly: effective_weight = base × max(0, 1 −
        #      λ5 / _ANCHOR_HANDOFF_LAMBDA5).
        #
        # Only active when weight > 0.
        if self.entropy_anchor_weight > 0.0:
            # Session 21 Fix B: hand off to Lcal as λ5 ramps up. λ5 = 0.10
            # (paper default) is the full handoff point → anchor disabled.
            _ANCHOR_HANDOFF_LAMBDA5 = 0.10
            lambda5_now = float(self.lambda5)
            handoff_decay = max(0.0, 1.0 - lambda5_now / _ANCHOR_HANDOFF_LAMBDA5)
            effective_anchor_w = self.entropy_anchor_weight * handoff_decay
            if effective_anchor_w > 0.0:
                probs = F.softmax(dir_logits, dim=-1).clamp(min=1e-8)
                entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()   # scalar tensor
                n_classes = int(dir_logits.shape[-1])
                target_H = self.entropy_anchor_target_frac * math.log(max(n_classes, 2))
                # ReLU deficit via clamp(min=0) — Python float − tensor returns a
                # tensor with gradient flowing through ``entropy`` (no extra tensor
                # allocation per batch as the previous torch.tensor() path did).
                entropy_deficit = (target_H - entropy).clamp(min=0.0)
                Ldir = Ldir + effective_anchor_w * entropy_deficit

        # Session 22 audit fix (Fix 8): logit-magnitude L2 penalty.
        # Rationale — curb temperature drift at the source:
        #   In the Fold 1 audit we saw post-hoc temperature scaling fit T→4.2
        #   on validation, indicating the model was producing logits with
        #   magnitudes far larger than needed for the label-smoothed (ε=0.05)
        #   CE target. With ε=0.05 the CE-optimal logit gap between correct
        #   and incorrect classes is log(0.95 / 0.025) ≈ 3.64 — anything
        #   beyond that is over-specialisation that post-hoc T must undo.
        #   A small L2 on dir_logits.pow(2).mean() directly penalises this
        #   over-growth, attacking the root cause of T drift rather than
        #   symptomatically fixing it post-hoc (Fix 2) or via entropy anchor
        #   (Fix B/D). Weight is deliberately tiny (1e-4): just enough to
        #   stop runaway growth without hurting CE convergence. Complementary
        #   to the Brier Lcal loss (Eq.35): Lcal aligns confidence with
        #   accuracy; this penalty limits raw logit amplitude — orthogonal.
        # Not in the paper — documented in DEVIATIONS.md as an engineering
        # regulariser. Setting ``logit_l2_weight=0.0`` disables it.
        if self.logit_l2_weight > 0.0:
            # Mean over batch × classes is O(1) regardless of batch size.
            logit_mag_penalty = dir_logits.pow(2).mean()
            Ldir = Ldir + self.logit_l2_weight * logit_mag_penalty

        # ── Unweighted per-sample CE for Lrisk (Eq.37) ───────────────────────
        # Lrisk measures prediction quality on abstaining samples — class weights
        # and focal reshaping would bias the selective-risk signal, so the raw
        # per-sample CE is kept here.
        ce_per_sample = F.cross_entropy(
            dir_logits, dir_labels, reduction='none'
        )  # (B,) — used by Lrisk only

        # Lret (Eq.32): SmoothL1 on train-normalized returns for stable optimization.
        scale = self.return_scale.clamp(min=1e-6).to(device)
        # Session 32 — clamp scaled TARGETS only (not predictions) to [-5, 5].
        # Prevents flash-crash outliers (e.g., BTC hourly |r|=0.08 scaled by
        # q75=0.006 → -13.3) from dominating the SmoothL1 linear region. SmoothL1
        # gives linear loss above |x|=1, so one outlier contributes 10-13× a typical
        # sample → Lret=6.54 spikes observed in smoke tests.
        # [-5, 5] ≈ 5× q75 ≈ 99.7th percentile. Clamping only the detached target
        # (no gradient flows through ground truth anyway) — predictions remain free
        # to grow, SmoothL1's own linear region handles large pred-target errors
        # smoothly. Paper doesn't specify this but SmoothL1 is an outlier-robust
        # choice by design; target clamping extends that robustness to the scale
        # normalization step.
        ret_labels_scaled = (ret_labels / scale).clamp(min=-5.0, max=5.0)
        ret_pred_scaled = ret_pred / scale  # free gradient, no clamp
        Lret = (
            F.smooth_l1_loss(ret_pred_scaled, ret_labels_scaled, reduction='none') * valid_mask
        ).mean()
        # Session 27 fix #14/#15: Sign-agreement auxiliary for return head.
        # Vấn đề quan sát: Lret giảm từ 5.52 → 0.31 nhưng RetCorr ≈ 0 —
        # model học "predict ~0 để minimize SmoothL1" thay vì học directional
        # signal. Thêm hinge loss trên sign(ret_pred) · sign(ret_labels):
        # khi dấu khớp → 0 loss, khi lệch dấu → penalty dương. λ = 0.1
        # nhẹ để không dominate SmoothL1 magnitude regression.
        if self.lret_sign_weight > 0.0:
            sign_pred   = torch.tanh(ret_pred / scale.clamp(min=1e-4))
            sign_target = torch.sign(ret_labels)
            hinge = torch.clamp(0.1 - sign_pred * sign_target, min=0.0)
            eps_h = float(getattr(self, "epsilon_h", self._DEFAULT_EPS_H_BY_HORIZON["1h"]))
            eps_threshold = eps_h * self.lret_sign_eps_factor
            nonzero_mask = (ret_labels.abs() > eps_threshold).float()
            mag_w = (ret_labels.abs() / scale.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
            Lret_sign = (hinge * nonzero_mask * mag_w * valid_mask).mean()
            Lret = Lret + self.lret_sign_weight * Lret_sign
        else:
            Lret_sign = torch.tensor(0.0, device=device)

        # ── LretCls (Sprint 4-D, auxiliary binary return-sign head) ──────────
        # Phase 8.7 audit confirmed ret_pred regression collapses near zero:
        # std(ret_pred) ≈ 0.0001 vs std(ret_label) ≈ 0.009 (75× too small),
        # sign_acc on |ret|>ε_h ≈ 0.50 (random). The hinge term Lret_sign
        # above shares gradient with the SmoothL1 magnitude regression and
        # cannot dominate it without crashing magnitude learning. Sprint 4-D
        # adds a SEPARATE binary classification head (ret_sign_head) trained
        # by BCE so the sign-correctness gradient does not bottleneck through
        # the regression path. Active ONLY on samples with |ret_label| > ε_h
        # so the head does not learn sign on neutral-band noise. Disabled by
        # default (lambda_ret_sign_cls = 0.0); Sprint 4-D run sets it to 0.1.
        if self.lambda_ret_sign_cls > 0.0 and ret_sign_logit is not None:
            eps_h_cls = float(
                getattr(self, "epsilon_h", self._DEFAULT_EPS_H_BY_HORIZON["1h"])
            )
            cls_mask = ((ret_labels.abs() > eps_h_cls).float() * valid_mask)
            n_cls = cls_mask.sum().clamp(min=1.0)
            target_pos = (ret_labels > 0).float()
            bce_per = F.binary_cross_entropy_with_logits(
                ret_sign_logit, target_pos, reduction="none"
            )
            LretCls = (bce_per * cls_mask).sum() / n_cls
        else:
            LretCls = torch.tensor(0.0, device=device)

        # ── LretBin (Sprint 5, auxiliary 3-class tradability bin head) ───────
        # Direction-INDEPENDENT magnitude classification on |ret_label| vs ε_h:
        #   bin 0 = no_edge       : |ret| ≤ ε_h            (noise/cost band)
        #   bin 1 = marginal_edge : ε_h < |ret| < 2·ε_h    (move present but thin)
        #   bin 2 = strong_edge   : |ret| ≥ 2·ε_h          (clear tradeable amplitude)
        # Bin labels are computed inline from ret_labels — no dataset change
        # needed (β option from S5a design discussion). CrossEntropyLoss on
        # the 3-class head logits, no class weights in v1 (distribution log
        # first to decide weighting). Disabled by default; Sprint 5a curriculum
        # will ramp 0.0 → 0.05 → 0.10 across stages. In Sprint 5a (this
        # commit), the head is trained + logged but the alert policy is
        # unchanged so model-fault and policy-fault stay separable.
        if self.lambda_ret_bin > 0.0 and ret_bin_logits is not None:
            eps_h_bin = float(
                getattr(self, "epsilon_h", self._DEFAULT_EPS_H_BY_HORIZON["1h"])
            )
            abs_ret = ret_labels.abs()
            bin_labels = torch.zeros_like(ret_labels, dtype=torch.long)
            bin_labels[abs_ret > eps_h_bin] = 1            # marginal_edge
            bin_labels[abs_ret >= 2.0 * eps_h_bin] = 2     # strong_edge
            LretBin = F.cross_entropy(
                ret_bin_logits, bin_labels, reduction="mean"
            )
        else:
            LretBin = torch.tensor(0.0, device=device)

        # Sprint 9A - auxiliary binary directional edge heads. These heads
        # ask "is this UP?" and "is this DOWN?" separately, with NEUTRAL as
        # the implicit neither-fires case. Batch-local sqrt pos_weight lightly
        # softens imbalance without letting the BCE dominate CE.
        if self.lambda_up_edge > 0.0 and up_edge_logit is not None:
            up_target = (dir_labels == 2).float()
            up_pos = up_target.sum()
            up_neg = (1.0 - up_target).sum()
            up_pos_weight = torch.sqrt(up_neg / up_pos.clamp(min=1.0)).clamp(
                min=0.25, max=4.0
            )
            LupEdge = F.binary_cross_entropy_with_logits(
                up_edge_logit, up_target, pos_weight=up_pos_weight, reduction="mean"
            )
        else:
            LupEdge = torch.tensor(0.0, device=device)

        if self.lambda_down_edge > 0.0 and down_edge_logit is not None:
            down_target = (dir_labels == 0).float()
            down_pos = down_target.sum()
            down_neg = (1.0 - down_target).sum()
            down_pos_weight = torch.sqrt(down_neg / down_pos.clamp(min=1.0)).clamp(
                min=0.25, max=4.0
            )
            LdownEdge = F.binary_cross_entropy_with_logits(
                down_edge_logit, down_target, pos_weight=down_pos_weight, reduction="mean"
            )
        else:
            LdownEdge = torch.tensor(0.0, device=device)

        # Lvol extension (disabled for paper-final runs).
        # Only active when vol_pred AND vol_labels are both provided AND the
        # lambda_vol weight is non-zero (controlled via config). Uses the same
        # train-q75 scaling as Lret so gradient magnitudes stay comparable
        # across the two regression heads. Loss is zero-tensor when inactive
        # so total loss computation stays a single expression.
        if vol_pred is not None and vol_labels is not None and float(self.lambda_vol) > 0.0:
            vol_pred_v = vol_pred
            vol_labels_v = vol_labels
            if valid_idx is not None:
                vol_pred_v = vol_pred_v[valid_idx]
                vol_labels_v = vol_labels_v[valid_idx]
            Lvol = (
                F.smooth_l1_loss(
                    vol_pred_v / scale, vol_labels_v / scale, reduction='none'
                ) * valid_mask
            ).mean()
        else:
            Lvol = torch.tensor(0.0, device=device)

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
        # Margin loss: L_faith = max(0, m - relative_gap)
        # where relative_gap = (p_full_c - p_mask_c) / p_full_c.
        #
        # Why RELATIVE gap (P1 #8 fix), not absolute:
        # With an absolute gap the achievable margin scales with p_full_c.
        # When the model is confident (p_full=0.9), max achievable gap = 0.9
        # and a margin of 0.12 is trivial. When uncertain (p_full=0.4), max
        # gap = 0.4 and the same margin is almost impossible — gradient
        # concentrates on *confident* samples which need the training signal
        # least. Normalising by p_full equalises the difficulty across
        # confidence regimes so Lfaith pushes ALL samples toward relying on
        # the selected evidence, not just the easy ones.
        #
        # Margin semantics stays: m=0.12 means "remove the selected evidence
        # should drop the predicted class probability by at least 12% *of its
        # current value*". Per-horizon margins in _DEFAULT_FAITH_MARGIN_BY_HORIZON
        # are tuned around this relative scale.
        #
        # Per-class gap still freezes to full-forward argmax (not max-prob)
        # to avoid the "class flip" bias where masking flips argmax and
        # max(p_mask) refers to a different class than max(p_full).
        if masked_dir_logits is not None:
            full_probs   = F.softmax(dir_logits,        dim=-1)   # (B, 3)
            masked_probs = F.softmax(masked_dir_logits, dim=-1)   # (B, 3)
            pred_class = dir_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            p_full_c = full_probs.gather(1, pred_class).squeeze(1)   # (B,)
            p_mask_c = masked_probs.gather(1, pred_class).squeeze(1) # (B,)
            # Lfaith gap type configurable.
            # "absolute": paper-literal Eq.36 gap = p_full - p_mask.
            # "relative": engineering improvement gap = (p_full - p_mask) / p_full.
            #     Reason: absolute gap saturates at p_full — confident samples
            #     (p_full=0.9) achieve any margin trivially; uncertain samples
            #     (p_full=0.4) cannot achieve margin > 0.4. Relative form
            #     equalises difficulty across confidence regimes.
            # Lower-clip at 0: when masking accidentally RAISES the predicted
            # class probability (gap < 0), the hinge loss should be margin
            # (as if gap=0), not margin + |gap|.
            if self.lfaith_gap_type == "absolute":
                gap = (p_full_c - p_mask_c).clamp(min=0.0, max=1.0)
            else:  # "relative" (default)
                gap = ((p_full_c - p_mask_c) / p_full_c.clamp(min=1e-3)).clamp(min=0.0, max=1.0)
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

        # Total: paper Eq.30 plus optional Lvol extension.
        # Paper Eq.30 has 7 weighted terms. Setting lambda_vol=0 keeps the
        # exact paper formula; lambda_vol>0 opts into volatility regression.
        total = (
            self.lambda1 * Ldir   +
            self.lambda2 * Lret   +
            self.lambda3 * Lfac   +
            self.lambda4 * Lsel   +
            self.lambda5 * Lcal   +
            self.lambda6 * Lfaith +
            self.lambda7 * Lrisk  +
            self.lambda_vol * Lvol   # optional volatility extension
            + self.lambda_ret_sign_cls * LretCls  # Sprint 4-D — auxiliary binary sign head
            + self.lambda_ret_bin * LretBin       # Sprint 5  — auxiliary tradability bin head
            + self.lambda_up_edge * LupEdge       # Sprint 9A - auxiliary UP-vs-rest head
            + self.lambda_down_edge * LdownEdge   # Sprint 9A - auxiliary DOWN-vs-rest head
        )

        return {
            "loss":   total,
            "Ldir":   self._to_scalar(Ldir),
            "Lret":   self._to_scalar(Lret),
            "Lret_sign": self._to_scalar(Lret_sign),
            "Lfac":   self._to_scalar(Lfac),
            "Lsel":   self._to_scalar(Lsel),
            "Lcal":   self._to_scalar(Lcal),
            "Lfaith": self._to_scalar(Lfaith),
            "Lrisk":  self._to_scalar(Lrisk),
            "Lvol":   self._to_scalar(Lvol),  # optional volatility extension
            "LretCls": self._to_scalar(LretCls),  # Sprint 4-D — auxiliary binary return-sign head
            "LretBin": self._to_scalar(LretBin),  # Sprint 5  — auxiliary tradability bin head
            "LupEdge": self._to_scalar(LupEdge),  # Sprint 9A - auxiliary UP-vs-rest head
            "LdownEdge": self._to_scalar(LdownEdge),  # Sprint 9A - auxiliary DOWN-vs-rest head
        }


    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_scalar(x: torch.Tensor | float) -> float:
        return x.item() if isinstance(x, torch.Tensor) else float(x)

    def _compute_lsel_pdf(self, attn_weights, article_mask, valid_mask, device, soft_gates=None):
        """Selection regularization (PDF Eq.34) — with a principled deviation.

        ─── As written in the PDF (Eq.34) ───────────────────────────────────
            L_sel = (Σ_i α̃_i - K_h)² + η · Σ_i α̃_i · log α̃_i

        ─── Why the PDF equation is degenerate ──────────────────────────────
        Σ_i α̃_i is a CONSTANT by construction: 1 under plain softmax or
        exactly K_h under the scaled-softmax output of SelectiveAttention.
        So (Σα̃_i - K_h)² is either a fixed nonzero constant (gives a biased
        gradient of zero w.r.t. weights) or exactly zero, and the cardinality
        term contributes NO learning signal to the scoring network. The
        paper's stated intent — "activate approximately K_h articles" — is
        then unachievable from Eq.34 alone.

        ─── Principled code deviation ───────────────────────────────────────
        We replace the cardinality operand with the raw sigmoid gates:
            sel_term = (Σ_i σ(a_i) - K_h)²
        σ(a_i) ∈ [0, 1] is NOT a simplex, so Σσ(a_i) varies per-sample and
        the MSE pushes approximately K_h gates to exceed 0.5 — exactly
        matching the paper's written intent. The entropy term keeps α̃
        (softmax-normalized top-K weights) unchanged so the sparsity
        regularisation still operates on the distribution that selects the
        top-K evidence.

            L_sel = (Σ_i σ(a_i) - K_h)²  +  η · Σ_i α̃_i · log α̃_i

        ─── Paper action required ──────────────────────────────────────────
        Update PDF Section 3.9.4 / Eq.34 to:
            L_sel = (Σ_i σ(a_i) - K_h)² + η · Σ_i α̃_i · log α̃_i
        and flag σ(a_i) as the raw sigmoid gate — see run_baselines.py and
        safe_alert_net.SelectiveAttention for how σ(a_i) is exposed via the
        'soft_gates' output. This is an INTENDED correction, not a bug.
        """
        # Both None → market-only sample (no articles) → no selection to regularize.
        if soft_gates is None and attn_weights is None:
            return torch.tensor(0.0, device=device)

        # ── cardinality term (σ(a_i) gate sum) ────────────────────────────────
        # soft_gates is None only when articles are absent or selective-news is
        # ablated; in both cases no selection loss should fire.
        if soft_gates is None:
            sel_term = torch.zeros(attn_weights.shape[0], device=device)
        else:
            # Target = min(K_h, n_valid_articles) — sparse-news samples cannot
            # and should not activate more gates than they have articles.
            if article_mask is not None:
                target_k = article_mask.float().sum(dim=1).clamp(max=float(self.K_h))
            else:
                target_k = torch.full(
                    (soft_gates.shape[0],), float(self.K_h),
                    device=device, dtype=soft_gates.dtype,
                )
            sel_term = (soft_gates.sum(dim=1) - target_k) ** 2

        # ── entropy term on α̃ (PDF Eq.34 second term) ────────────────────────
        # Normalize the entropy by max-possible-entropy log(K_h) so η has a
        # horizon-independent effect: raw Σα̃·log α̃ ∈ [-log K_h, 0] scales
        # with K_h (K_h=4 → range [-1.39, 0], K_h=8 → [-2.08, 0]). Without
        # normalization, η=0.18 pushes more weight on 24h attention (K_h=8)
        # than on 15m (K_h=3) purely due to K_h difference, not a model
        # choice. Normalization also makes η comparable across future
        # horizon additions without re-tuning.
        if attn_weights is not None:
            row_sum = attn_weights.sum(dim=1, keepdim=True)
            alpha_norm = attn_weights / row_sum.clamp(min=_EPS)
            alpha_clamped = alpha_norm.clamp(min=_EPS)
            neg_entropy = (alpha_clamped * torch.log(alpha_clamped)).sum(dim=1)
            neg_entropy = torch.where(
                row_sum.squeeze(1) > _EPS, neg_entropy, torch.zeros_like(neg_entropy)
            )
            # Per-horizon normalization: divide by log(max(K_h, 2)) so the
            # entropy term lives in a ~[-1, 0] range regardless of K_h.
            import math as _math
            k_norm = max(float(self.K_h), 2.0)
            log_kh = _math.log(k_norm) if k_norm > 1 else 1.0
            neg_entropy = neg_entropy / log_kh
        else:
            neg_entropy = torch.zeros(sel_term.shape[0], device=device)

        # neg_entropy = Σ α·log(α) ∈ [-log(K), 0] (negative value).
        # To PENALIZE diffuse attention (high H), we want higher lsel when
        # entropy is large. Since neg_entropy = -H, subtracting it adds +H:
        #   lsel = sel_term + η·H = sel_term - η·neg_entropy
        # The previous sign was wrong (+neg_entropy), which rewarded diffuse attention.
        lsel = ((sel_term - self.eta * neg_entropy) * valid_mask).mean()
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
            # PDF Eq.33: L_fac = -Σ_i Σ_c ỹ_c · log p_c (sum over articles i).
            # fac_probs is ALREADY a softmax distribution (from factor_mod.forward):
            #     p_fac = F.softmax(W_fac(articles), dim=-1)
            # So we take log(p) directly. A previous implementation called
            # F.log_softmax(fac_probs, dim=-1) — applying softmax *twice* — which
            # gave a wrong, flattened loss (CE floor ≈ 1.5 for a perfect prediction
            # instead of 0). That artificial floor is why training logs showed Lfac
            # plateauing around 1.83 instead of converging.
            #
            # Reduction across articles is controlled by ``self.lfac_reduction``
            # (D16): "mean" normalises by N_{s,t} for magnitude stability across
            # variable article counts; "sum" matches Eq.33 verbatim but scales
            # with N, requiring λ3 retuning. Default "mean".
            #
            # Session 26 — per-factor class weight w_c (shape C) applied as
            # -Σ_c w_c · ỹ_c · log p_c. Fixes the severe class imbalance
            # observed in BTC data (network_outage 0.6 % vs macro 32.8 %).
            # When self._has_factor_cw=False, reverts to uniform weighting.
            log_p = torch.log(fac_probs.clamp(min=_EPS))       # (B, K, C) per-article
            fac_labels_s = _smooth_soft_labels(fac_labels)     # smooth before CE
            if self._has_factor_cw and self.factor_class_weights.numel() == fac_probs.shape[-1]:
                w_c = self.factor_class_weights.to(log_p.device)           # (C,)
                per_article_ce = -(fac_labels_s * log_p * w_c).sum(dim=-1)  # (B, K)
            else:
                per_article_ce = -(fac_labels_s * log_p).sum(dim=-1)       # (B, K)
            if article_mask is not None:
                w = article_mask.float()                        # (B, K)
                masked_ce = per_article_ce * w                  # (B, K) — zero on pad
                if self.lfac_reduction == "sum":
                    Lfac = masked_ce.sum(dim=1)                 # (B,) paper Eq.33
                else:
                    n = w.sum(dim=1).clamp(min=1.0)             # (B,)
                    Lfac = masked_ce.sum(dim=1) / n             # (B,) engineering mean
            else:
                if self.lfac_reduction == "sum":
                    Lfac = per_article_ce.sum(dim=1)            # (B,)
                else:
                    Lfac = per_article_ce.mean(dim=1)           # (B,)
            Lfac = Lfac * valid_mask
        elif fac_probs.dim() == 2:  # (B, C)
            if fac_labels.dim() == 1:
                Lfac = F.cross_entropy(fac_probs, fac_labels, reduction='none') * valid_mask
            else:
                # fac_labels may be (B, K, C) when model is in market-only mode (p_fac_all=None).
                # Reduce to (B, C) by averaging over K before computing soft CE.
                # This loses per-article signal — warn once per process so callers
                # know they hit the degenerate branch rather than silently running it.
                if fac_labels.dim() == 3:
                    if not getattr(self, "_lfac_shape_warned", False):
                        import warnings
                        warnings.warn(
                            "[MultiObjectiveLoss] Lfac: fac_probs is (B,C) but "
                            "fac_labels is (B,K,C) — averaging labels over K. "
                            "Per-article signal lost; this path fires only in "
                            "market-only/no-news mode.",
                            RuntimeWarning, stacklevel=2,
                        )
                        self._lfac_shape_warned = True
                    fac_labels = fac_labels.mean(dim=1)  # (B, C)
                fac_labels = _smooth_soft_labels(fac_labels)
                # Same log(p) correction as the 3D branch above.
                log_p = torch.log(fac_probs.clamp(min=_EPS))
                # PDF Eq.33: L_fac = -Σ_c ỹ_c · log p_c (soft CE, per-class weighting).
                if self._has_factor_cw and self.factor_class_weights.numel() == fac_probs.shape[-1]:
                    w_c = self.factor_class_weights.to(log_p.device)      # (C,)
                    Lfac = -(fac_labels * log_p * w_c).sum(dim=-1) * valid_mask
                else:
                    Lfac = -(fac_labels * log_p).sum(dim=-1) * valid_mask
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

        Both averages are computed over VALID samples only. Previously the
        coverage penalty used ``conf.mean()`` over the full (incl. invalid)
        batch, which under-penalised coverage whenever some samples were
        filtered as NaN/Inf — the invalid samples kept their clamped-min
        confidence (0.01) and dragged the mean down, making it look like we
        were already below κ even when the valid subset exceeded κ. Masked
        averaging restores the paper's intent: the penalty fires iff the
        mean confidence ON VALID SAMPLES is below κ.
        """
        conf = confidence.clamp(min=_LRISK_CONF_MIN)  # floor for denominator stability
        valid_sum = valid_mask.sum() + _EPS
        denom       = (conf * valid_mask).sum() + _EPS
        risk_term   = (ce_per_sample * conf * valid_mask).sum() / denom
        conf_mean_valid = (conf * valid_mask).sum() / valid_sum
        cov_penalty = torch.clamp(self.coverage_target - conf_mean_valid, min=0.0)
        return risk_term + self.mu * cov_penalty
