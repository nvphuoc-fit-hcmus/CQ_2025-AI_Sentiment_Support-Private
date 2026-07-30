"""
Train SAFE-Alert model (PDF Eq.30 multi-objective loss).

Walk-forward protocol (expanding window + embargo):
train on [0:t], validate on next window, test on immediately-following window,
then shift forward and repeat.

Commands:
  python train_safe_alert.py                        # use train_config_research_best.yaml defaults
  python train_safe_alert.py --config my.yaml       # custom config
  python train_safe_alert.py --epochs 40            # override one value
  python train_safe_alert.py --walk_forward         # walk-forward CV (PDF Eq.1)
"""

import math
import os
import sys
import json
import gc
import shutil
import warnings
import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
from argparse import ArgumentParser
from typing import Any, Optional, Tuple, Dict
from torch.utils.data import Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add pipelines folder for same-directory imports
from models.safe_alert_net import SAFEAlertNet, FACTOR_CLASSES
from constants import get_epsilon_h
from safe_alert_training_utils import BalancedBatchSampler, MultiObjectiveLoss
from safe_alert_dataset import SAFEAlertDataset
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_ece_per_decile, compute_brier_score,
    compute_alert_precision, mini_backtest, compute_model_selection_score,
    compute_auc, compute_coverage_risk_auc, compute_hit_rate, search_alert_policy,
    fit_temperature_scaling, apply_temperature_to_logits,
    compute_deletion_insertion_score, compute_sufficiency, compute_factor_consistency,
)
from utils import ARTIFACT_DIR, standardize_walk_forward_artifacts

torch.manual_seed(42)
np.random.seed(42)
# CUDA determinism: seeding alone is insufficient because cuDNN autotunes
# convolution/matmul algorithms non-deterministically with ``benchmark=True``.
#
# Session 22 audit fix (Fix 4): DEFAULT is now DETERMINISTIC. Reviewer
# reproducibility is a first-class requirement for a thesis, and the prior
# default (benchmark=True for 5-15% speedup) meant any reviewer attempting
# to reproduce the headline numbers would see different results — a silent
# credibility failure. Users who need the speed boost for long ablation
# sweeps can opt out via ``SAFEALERT_FAST=1``. Legacy callers that relied
# on the old env var ``SAFEALERT_DETERMINISTIC=0`` to disable determinism
# should migrate to ``SAFEALERT_FAST=1``.
_FAST_MODE = os.environ.get("SAFEALERT_FAST", "0").strip().lower() in {"1", "true", "yes", "on"}
# Back-compat: honour an explicit SAFEALERT_DETERMINISTIC=0 as opt-out too.
_LEGACY_OFF = os.environ.get("SAFEALERT_DETERMINISTIC", "").strip().lower() in {"0", "false", "no", "off"}
_DETERMINISTIC = not (_FAST_MODE or _LEGACY_OFF)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if _DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _exc:
        # Older torch versions / unsupported kernels — deterministic cuDNN still engages.
        # Session 23 P1 #7 fix: log the cause so users can see *why* strict
        # determinism was dropped (helps triaging reproducibility gaps).
        import warnings as _w
        _w.warn(
            f"torch.use_deterministic_algorithms unavailable ({type(_exc).__name__}: {_exc}); "
            f"cuDNN deterministic mode still engaged. Results may differ slightly across "
            f"torch versions.",
            RuntimeWarning, stacklevel=2,
        )


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
    ]
    for col in candidates:
        if col in normalized.values():
            inv = {v: k for k, v in normalized.items()}
            return df.rename(columns={inv[col]: "timestamp"})
    for col_raw, col_norm in normalized.items():
        if "time" in col_norm:
            return df.rename(columns={col_raw: "timestamp"})
    return df
if not _DETERMINISTIC:
    # Fast mode: cuDNN autotunes kernels for the batch shape (faster, non-reproducible).
    torch.backends.cudnn.benchmark = True
    import warnings as _w
    _w.warn(
        "SAFEALERT_FAST=1 — cuDNN benchmark enabled. Run is NOT reproducible. "
        "Do NOT use this mode for final thesis/paper numbers.",
        RuntimeWarning, stacklevel=2,
    )


# ── P2 #21: DataLoader worker seeding for reproducibility ──────────────────────
# torch.manual_seed(42) above seeds the MAIN process only — each worker forked
# by DataLoader inherits a different torch RNG state but leaves numpy/random
# unseeded, so augmentation or order-dependent logic drifts between runs at
# num_workers>0. Supplying worker_init_fn = _seed_worker + a manual-seeded
# generator makes each worker derive its numpy/random seed deterministically
# from the main seed.
def _seed_worker(worker_id: int) -> None:
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


_LOADER_GENERATOR = torch.Generator()
_LOADER_GENERATOR.manual_seed(42)


# ── R3 #A: Stage-aware 3-phase LR scheduler ────────────────────────────────────
#
# PROBLEM with the previous `CosineAnnealingWarmRestarts(T_0=10, T_mult=2,
# eta_min=1e-7)`: for a 40-epoch run the warm restarts land at epochs 10 and 30.
# Stage 3 begins at epoch 29 (70% of 40), *one epoch before a scheduled restart*,
# when LR has already decayed almost to ``eta_min=1e-7``. The Stage-3 LR cut
# (×0.5) is then applied on top of an already-near-zero LR, so the observed
# training LR at Stage 3 entry was ~3.7e-6 — about 2% of the intended 1.5e-4.
# Result: Lcal λ5 jumps from 0.02 → 0.15 but weights literally cannot move;
# temperature stays stuck at 2.0-2.6, and best checkpoint often regresses vs
# Stage 2.
#
# SOLUTION: A deterministic 3-phase schedule that MATCHES the curriculum:
#   - Stage 1 (warmup):  linear from ``lr/10`` to ``lr``
#   - Stage 2 (learn):   cosine from ``lr`` to ``lr * stage3_lr_scale``
#                        — reaches the Stage 3 start LR smoothly, no restart
#   - Stage 3 (tune):    cosine from ``lr * stage3_lr_scale`` to ``lr * eta_frac``
#                        — meaningful floor (default 0.02 × base = 6e-6 for lr=3e-4)
#                        so gradients can still move weights during fine-tuning.
#
# This replaces BOTH the CosineAnnealingWarmRestarts AND the manual
# `_enter_stage3()` base_lrs rewrite, which were fighting each other.

class StageAwareLRScheduler:
    """Per-epoch 3-phase LR schedule tied to the curriculum fractions.

    Call ``step(epoch)`` once per epoch AFTER loss updates and optimizer.step().
    Epoch numbering matches the trainer (1-indexed).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        base_lr: float,
        stage1_end_frac: float,
        stage2_end_frac: float,
        stage3_lr_scale: float = 0.5,
        warmup_start_frac: float = 0.2,   # Session 27: 0.1 → 0.2 để tránh LR jump lớn
                                          # giữa các epoch đầu. 3e-5 → 3e-4 (10×) quá dốc
                                          # gây acc drop 0.40→0.24 ở Ep 2-3.
        eta_frac: float = 0.02,
    ) -> None:
        self.optimizer = optimizer
        self.total_epochs = int(total_epochs)
        self.base_lr = float(base_lr)
        self.stage1_end = max(1, int(total_epochs * stage1_end_frac))
        self.stage2_end = max(self.stage1_end + 1, int(total_epochs * stage2_end_frac))
        self.stage3_start_lr = self.base_lr * float(stage3_lr_scale)
        self.warmup_start_lr = self.base_lr * float(warmup_start_frac)
        self.eta_lr = self.base_lr * float(eta_frac)
        self.last_lr = self.warmup_start_lr
        # Snapshot so future `state_dict()` reads reflect initial config.
        self._initial_state = {
            "base_lr": self.base_lr,
            "stage1_end": self.stage1_end,
            "stage2_end": self.stage2_end,
            "stage3_start_lr": self.stage3_start_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "eta_lr": self.eta_lr,
        }

    def _compute_lr(self, epoch: int) -> float:
        """Return the LR that epoch ``epoch`` (1-indexed) should train with."""
        e = max(1, int(epoch))
        if e <= self.stage1_end:
            # Linear warmup
            t = (e - 1) / max(self.stage1_end - 1, 1)
            return self.warmup_start_lr + t * (self.base_lr - self.warmup_start_lr)
        if e <= self.stage2_end:
            # Cosine decay from base_lr to stage3_start_lr
            t = (e - self.stage1_end) / max(self.stage2_end - self.stage1_end, 1)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return self.stage3_start_lr + (self.base_lr - self.stage3_start_lr) * cos
        # Stage 3: cosine decay from stage3_start_lr to eta_lr
        t = (e - self.stage2_end) / max(self.total_epochs - self.stage2_end, 1)
        t = min(max(t, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        lr = self.eta_lr + (self.stage3_start_lr - self.eta_lr) * cos
        return max(self.eta_lr, lr)

    def step(self, epoch: int) -> float:
        """Write the computed LR into all param groups. Returns base LR.

        Each group's actual LR = base_lr × group["lr_mult"] (default 1.0).
        This preserves per-component multipliers (selector ×1.5, factor ×2.0,
        etc.) while the scheduler controls the shared base LR decay curve.
        """
        base_lr = self._compute_lr(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = base_lr * group.get("lr_mult", 1.0)
        self.last_lr = base_lr
        return base_lr

    def get_last_lr(self) -> list:
        """Return per-group LRs (base × lr_mult for each group)."""
        return [
            self.last_lr * g.get("lr_mult", 1.0)
            for g in self.optimizer.param_groups
        ]

    def state_dict(self) -> dict:
        return {"last_lr": self.last_lr, **self._initial_state}

    def load_state_dict(self, state: dict) -> None:
        self.last_lr = float(state.get("last_lr", self.base_lr))
        for k, v in self._initial_state.items():
            if k in state:
                setattr(self, k, state[k])


# ── Training curriculum constants ──────────────────────────────────────────────
# P2 #19: these defaults are the SOURCE OF TRUTH when no YAML config is
# absent. When present, _apply_config_overrides (called from main) overrides
# these at module level so _update_loss_weights_for_stage and _enter_stage3
# still see YAML-supplied values without a big refactor.
_STAGE1_END_FRAC  = 0.20   # Stage 1 ends at this fraction of total epochs.
_STAGE2_END_FRAC  = 0.70   # Stage 2 ends at this fraction.
                           # Stage 3 = remaining fraction; full loss active.
# Session 21 Fix A: Stage 3 warmup fraction. When Stage 3 begins, lambdas
# linearly ramp from Stage-2-end values to Stage-3 targets over this fraction
# of total epochs (clamped to 5 epochs max). Without the warmup λ5 jumped
# 0.08 → 0.15 (1.9×), λ6 0.15 → 0.20 and λ7 0.04 → 0.05 in one step, which,
# combined with the LR halve at Stage 3 entry, produced the observed
# monotone val-loss climb in Stage 3 (ep 29 → 36: 1.69 → 1.94 Fold 1).
# A 5-epoch ramp distributes the loss-surface change so the optimizer can
# track it under the decaying LR schedule.
_STAGE3_WARMUP_FRAC = 0.125   # = 5 epochs for a 40-epoch run.
_TAU_PERCENTILE   = 65     # τ seed: P(conf ≥ pX) ≈ 35% at pX=65 → matches κ=0.35.
_GAMMA_PERCENTILE = 55     # γ seed: P(max_prob ≥ pX) ≈ 45%.
_ACCUM_STEPS      = 2      # Gradient accumulation (effective_bs = bs × steps).
_ES_PATIENCE      = 4      # Session 22 audit fix (Fix 6): 7 → 4. On a 40-
                           # epoch run Stage 3 is ~11 epochs, so patience 7
                           # allowed drift for 64% of the fine-tuning stage —
                           # too lax given the observed val_loss climb in
                           # Stage 3. Patience 4 triggers ES after 4 non-
                           # improving epochs (~36% of S3), balancing against
                           # the natural noise floor of alert-metric
                           # fluctuation (~0.005 score units between folds).
_STAGE3_LR_SCALE  = 0.5    # LR multiplier when entering Stage 3.
_SWA_START_FRAC   = 0.88   # Session 22 audit fix (Fix 6 + Fix 16):
                           # Original 0.78 → 0.85 (Fix 6) → 0.88 (Fix 16).
                           # Stage 2 ends at 0.70, Stage 3 warmup runs for
                           # stage3_warmup_frac (0.125 default) more, so
                           # lambdas are steady only from frac ≈ 0.825
                           # onward. 0.85 left just 1.5% headroom which
                           # overlaps with the tail of the warmup transient
                           # (loss surface still settling). 0.88 guarantees
                           # a clean 5.5% gap so SWA averages strictly
                           # Stage-3-steady weights, addressing the "SWA
                           # averages into warmup transient" risk raised by
                           # the Session 22 forensic audit.
_ENABLE_SWA       = True
# R3 #E + Session 21 Fix D: anti-overconfidence entropy anchor defaults.
# Weight 0.05 stays; target lowered 0.95 → 0.50. Old target forced predictions
# into the 5%-of-uniform band (for C=3, max-prob ≤ ~0.40), which the post-hoc
# temperature scaler then over-corrected by fitting T up to 4.2 on validation.
# 0.50 permits max-prob ≈ 0.70 while still catching real collapse to one-hot.
# Weight further decays automatically as λ5 ramps (see
# MultiObjectiveLoss._ANCHOR_HANDOFF_LAMBDA5) so Lcal takes over smoothly.
_ENTROPY_ANCHOR_WEIGHT = 0.05
_ENTROPY_ANCHOR_TARGET = 0.50
# Session 22 Fix 8: default logit-magnitude L2 weight. See MultiObjectiveLoss
# docstring and DEVIATIONS.md section L8 for rationale.
_LOGIT_L2_WEIGHT       = 1e-4
# Volatility-regression extension weight. Paper §4.1.4 stores future
# volatility as a benchmark field but does not add Lvol to Eq.30, so the
# paper-final default is 0.0. The head remains in the model for checkpoint
# schema stability.
_LAMBDA_VOL            = 0.0

# Sprint 1 FIX 2 — articles_per_candle was previously a hard-coded 8 in the
# SAFEAlertDataset constructor and never threaded through the training
# entry point. With mean ≈ 30 articles/candle and max 737, K=8 means the
# selective news layer only sees ~27 % of the candidate pool, dominated by
# whatever ranking happens upstream. Sprint 1 wires this constant through
# _apply_config_overrides so YAML config can set ``articles_per_candle:``;
# the new default 16 doubles the candidate pool while staying inside the
# T4/P100 memory envelope (B=8, K=16, 768-d FP32 ≈ 4.5 MB / batch).
_ARTICLES_PER_CANDLE   = 16
_LOOKBACK_HOURS        = 24

# Sprint 1.5 — class weights mode. The previous code path always called
# ``compute_class_weight('balanced', ...)`` and then applied a sqrt soften +
# normalise step. For BTC 1h the sqrt mode collapses to weights of
# (1.05, 0.92, 1.03) which is effectively uniform — when focal_gamma and
# label_smoothing are both 0 (paper-literal), there is no remaining pressure
# pushing the direction head away from the trivial NEUTRAL minimum. Exposing
# this as a config flag lets us switch to the raw balanced weights without
# touching the training loop. Modes:
#   "none"     : all-ones (skip reweighting entirely).
#   "sqrt"     : legacy behaviour (balanced → sqrt → normalise).
#   "balanced" : sklearn 'balanced' normalised to mean 1, no soften.
_CLASS_WEIGHTS_MODE    = "sqrt"

# Session 26 — 7 previously-unwired YAML-configurable constants. Defaults match
# the values used as hardcoded arguments in MultiObjectiveLoss / SAFEAlertNet /
# SAFEAlertDataset constructors so behaviour is unchanged when the YAML is
# silent, but edits in the selected config now actually propagate.
_FOCAL_GAMMA         = 1.0
_LABEL_SMOOTHING_EPS = 0.05
_INGEST_DELAY_MIN    = 15
_COVERAGE_TARGET     = 0.35
_ETA_LSEL            = 0.18
_MU_LRISK            = 0.02
_FAITH_MARGIN_MAP    = {"15m": 0.08, "1h": 0.12, "4h": 0.15, "24h": 0.18}
_TOP_K_MAP           = {"15m": 3,    "1h":  4,    "4h":  5,    "24h":  8}

# Session 26 — paper Eq.33 Lfac reduction mode across articles in a window.
# "mean" (default): engineering convention, stable across variable N_{s,t}.
# "sum":  paper-literal Eq.33; requires λ3 retuning since magnitude scales
#         with number of articles per candle (1-50 for crypto news streams).
# See DEVIATIONS.md D16 for rationale. Configurable via YAML ``lfac_reduction:``.
_LFAC_REDUCTION        = "mean"
# Lfaith gap type. "relative" (default, engineering improvement) or
# "absolute" (paper-literal Eq.36). See DEVIATIONS.md §L5 for rationale.
_LFAITH_GAP_TYPE       = "relative"
_LRET_SIGN_WEIGHT      = 0.1
_LRET_SIGN_EPS_FACTOR  = 1.0
# Sprint 4 Phase 8.5 — optional ε_h override. None → paper canonical
# from constants.EPSILON_H_BY_HORIZON. When set via YAML key
# `epsilon_h_override`, the SAME value flows into BOTH the dataset
# label generator AND MultiObjectiveLoss.set_horizon — keeping
# direction-label boundary and Lret_sign nonzero mask synchronized.
_EPSILON_H_OVERRIDE    = None
# Sprint 4-D — auxiliary binary return-sign classification head weight.
# 0.0 disables (preserves backward compat); >0 enables BCE on
# (ret_sign_logit, sign(ret_label)>0) masked to |ret_label|>ε_h.
# Phase 8.7 audit closed the ret_pred regression angle (sign_acc≈0.50,
# magnitude collapsed); this orthogonal head adds dedicated capacity
# without disturbing the magnitude regression in ret_head.
_LAMBDA_RET_SIGN_CLS   = 0.0
# Sprint 5 — auxiliary tradability bin classification head weight.
# 0.0 disables (preserves backward compat); >0 enables 3-class CE on
# (ret_bin_logits, bin_label) where bin_label ∈ {0=no_edge, 1=marginal,
# 2=strong_edge} computed inline from |ret_label| vs ε_h thresholds.
# Direction-INDEPENDENT magnitude classification — answers "is this
# candle worth trading?", complementing dir_head's "long or short?".
# S5a curriculum target: see _update_loss_weights_for_stage which ramps
# 0.0 (S1) → 0.5×target (S2 end) → target (S3 steady). The YAML key
# `lambda_ret_bin` sets the S3 target value (default 0.0).
_LAMBDA_RET_BIN        = 0.0
# Optional UP/DOWN binary edge-head weights. Defaults are off so additive wiring
# does not affect paper-final runs unless explicitly enabled.
_LAMBDA_UP_EDGE        = 0.0
_LAMBDA_DOWN_EDGE      = 0.0
# Sprint 7 — Phase 8 closure follow-up. False (default) = legacy shuffle=True
# train loader. True = BalancedBatchSampler (each batch has ~ceil(B/3) samples
# per direction class, rotated). Addresses small-batch (B=8) stochastic class
# imbalance + intra-fold regime heterogeneity that drives direction-head
# NEUTRAL collapse. Train fold ONLY; val/test stay shuffle=False
# chronological per P1 #14. See sprint_7_balanced_sampler.md memory or the
# class docstring in safe_alert_training_utils.BalancedBatchSampler.
_BALANCED_BATCH_SAMPLER = False
# Session 26 — fraction of val used as the calibration slice (fits T + τ/γ).
# Remainder becomes the selection slice that reports model_score for early
# stopping. 1.0 disables (legacy full-val behaviour). 0.7 default trades 30 %
# of val sample size for an honest held-out selection measurement; smaller
# values (0.5) halve the held-out noise further at the cost of T-fit stability.
_CAL_SELECTION_SPLIT   = 0.7
# Eq.26 policy tau gate can use either raw confidence (paper path) or an
# action-aware confidence that downweights confidently-NEUTRAL predictions.
_POLICY_CONFIDENCE_SOURCE = "raw"


def _apply_config_overrides(cfg: dict) -> None:
    """P2 #19 + R3: push YAML ``curriculum`` + ``policy`` + anti-drift values
    into module-level constants used by the training/validation state
    machines. Called once from main() after the config file has been
    loaded. Keeps the original constants as fallbacks so callers without a
    YAML still work."""
    global _STAGE1_END_FRAC, _STAGE2_END_FRAC, _STAGE3_LR_SCALE, _SWA_START_FRAC
    global _ENABLE_SWA
    global _STAGE3_WARMUP_FRAC
    global _ES_PATIENCE, _ACCUM_STEPS, _TAU_PERCENTILE, _GAMMA_PERCENTILE
    global _POLICY_CONFIDENCE_SOURCE
    global _ENTROPY_ANCHOR_WEIGHT, _ENTROPY_ANCHOR_TARGET, _LOGIT_L2_WEIGHT, _LAMBDA_VOL
    global _CAL_SELECTION_SPLIT, _LFAC_REDUCTION, _LFAITH_GAP_TYPE
    global _LRET_SIGN_WEIGHT, _LRET_SIGN_EPS_FACTOR
    global _EPSILON_H_OVERRIDE
    global _LAMBDA_RET_SIGN_CLS
    global _LAMBDA_RET_BIN
    global _LAMBDA_UP_EDGE, _LAMBDA_DOWN_EDGE
    global _BALANCED_BATCH_SAMPLER
    # Session 26 — 7 additional config keys previously ignored (silent bug).
    # YAML had values but they never propagated to the runtime constants
    # below, so edits in YAML were no-ops. Now all are wired.
    global _S1_LAMBDA1, _S1_LAMBDA2, _S1_LAMBDA3, _S1_LAMBDA4
    global _S1_LAMBDA5, _S1_LAMBDA6, _S1_LAMBDA7
    global _S3_LAMBDA1, _S3_LAMBDA2, _S3_LAMBDA3, _S3_LAMBDA4
    global _S3_LAMBDA5, _S3_LAMBDA6, _S3_LAMBDA7
    global _S2_LAMBDA1, _S2_LAMBDA2, _S2_LAMBDA3, _S2_LAMBDA4
    global _S2_LAMBDA5, _S2_LAMBDA6, _S2_LAMBDA7
    global _FOCAL_GAMMA, _LABEL_SMOOTHING_EPS, _INGEST_DELAY_MIN
    global _COVERAGE_TARGET, _ETA_LSEL, _MU_LRISK
    global _FAITH_MARGIN_MAP, _TOP_K_MAP
    global _ARTICLES_PER_CANDLE, _LOOKBACK_HOURS
    global _CLASS_WEIGHTS_MODE
    curric = (cfg or {}).get("curriculum", {}) or {}
    _STAGE1_END_FRAC = float(curric.get("stage1_end_frac", _STAGE1_END_FRAC))
    _STAGE2_END_FRAC = float(curric.get("stage2_end_frac", _STAGE2_END_FRAC))
    _STAGE3_LR_SCALE = float(curric.get("stage3_lr_scale", _STAGE3_LR_SCALE))
    _STAGE3_WARMUP_FRAC = float(curric.get("stage3_warmup_frac", _STAGE3_WARMUP_FRAC))
    _ES_PATIENCE     = int(curric.get("es_patience",     _ES_PATIENCE))
    _SWA_START_FRAC  = float(curric.get("swa_start_frac", _SWA_START_FRAC))
    _ENABLE_SWA      = bool(curric.get("enable_swa", _ENABLE_SWA))
    _ACCUM_STEPS     = int((cfg or {}).get("accum_steps", _ACCUM_STEPS))
    _TAU_PERCENTILE  = int((cfg or {}).get("tau_percentile",   _TAU_PERCENTILE))
    _GAMMA_PERCENTILE = int((cfg or {}).get("gamma_percentile", _GAMMA_PERCENTILE))
    _POLICY_CONFIDENCE_SOURCE = str(
        (cfg or {}).get("policy_confidence_source", _POLICY_CONFIDENCE_SOURCE)
    ).strip().lower()
    if _POLICY_CONFIDENCE_SOURCE not in {"raw", "position"}:
        raise ValueError(
            "policy_confidence_source must be 'raw' or 'position', "
            f"got {_POLICY_CONFIDENCE_SOURCE!r}"
        )
    _ENTROPY_ANCHOR_WEIGHT = float((cfg or {}).get("entropy_anchor_weight", _ENTROPY_ANCHOR_WEIGHT))
    _ENTROPY_ANCHOR_TARGET = float((cfg or {}).get("entropy_anchor_target", _ENTROPY_ANCHOR_TARGET))
    _LOGIT_L2_WEIGHT       = float((cfg or {}).get("logit_l2_weight",       _LOGIT_L2_WEIGHT))
    _LAMBDA_VOL            = float((cfg or {}).get("lambda_vol",            _LAMBDA_VOL))
    # Session 26 — cal/sel split within val. 1.0 disables (legacy full-val).
    _CAL_SELECTION_SPLIT   = float((cfg or {}).get("cal_selection_split",   _CAL_SELECTION_SPLIT))
    # Session 26 — paper Eq.33 Lfac reduction: "mean" (default) or "sum" (D16).
    _LFAC_REDUCTION        = str((cfg or {}).get("lfac_reduction",         _LFAC_REDUCTION))
    _LFAITH_GAP_TYPE       = str((cfg or {}).get("lfaith_gap_type",        _LFAITH_GAP_TYPE))
    _LRET_SIGN_WEIGHT      = float((cfg or {}).get("lret_sign_weight",     _LRET_SIGN_WEIGHT))
    _LRET_SIGN_EPS_FACTOR  = float((cfg or {}).get("lret_sign_eps_factor", _LRET_SIGN_EPS_FACTOR))
    if not (0.0 <= _LRET_SIGN_WEIGHT <= 1.0):
        raise ValueError(f"lret_sign_weight must be in [0, 1], got {_LRET_SIGN_WEIGHT}")
    if not (0.1 <= _LRET_SIGN_EPS_FACTOR <= 10.0):
        raise ValueError(
            f"lret_sign_eps_factor must be in [0.1, 10], got {_LRET_SIGN_EPS_FACTOR}"
        )
    # Sprint 4 Phase 8.5 — ε_h override. None preserves paper canonical.
    # Validation (range, type) happens inside constants.get_epsilon_h on
    # first call; here we only normalize "null"/None and a missing key
    # to None so the constants helper sees a clean value.
    _eps_h_raw = (cfg or {}).get("epsilon_h_override", None)
    _EPSILON_H_OVERRIDE = (
        None if _eps_h_raw is None or _eps_h_raw == "" else float(_eps_h_raw)
    )
    # Sprint 4-D — auxiliary binary sign-head weight. Range validated in
    # MultiObjectiveLoss.__init__ so an off-grid YAML value crashes early.
    _LAMBDA_RET_SIGN_CLS = float(
        (cfg or {}).get("lambda_ret_sign_cls", _LAMBDA_RET_SIGN_CLS)
    )
    # Sprint 5 — auxiliary tradability bin head weight (S3 target). Curriculum
    # ramp logic in _update_loss_weights_for_stage consumes this value as the
    # Stage 3 steady target; Stages 1-2 ramp 0 → 0.5×target → target.
    _LAMBDA_RET_BIN = float(
        (cfg or {}).get("lambda_ret_bin", _LAMBDA_RET_BIN)
    )
    # Optional binary edge-head weights (S3 targets).
    _LAMBDA_UP_EDGE = float(
        (cfg or {}).get("lambda_up_edge", _LAMBDA_UP_EDGE)
    )
    _LAMBDA_DOWN_EDGE = float(
        (cfg or {}).get("lambda_down_edge", _LAMBDA_DOWN_EDGE)
    )
    # Sprint 7 — balanced batch sampler flag. Coerce common YAML truthy strings
    # ("true"/"yes"/"1") so the cell-6 markers can rely on a clean bool flow.
    _bbs_raw = (cfg or {}).get("balanced_batch_sampler", _BALANCED_BATCH_SAMPLER)
    if isinstance(_bbs_raw, str):
        _BALANCED_BATCH_SAMPLER = _bbs_raw.strip().lower() in {"true", "1", "yes", "on"}
    else:
        _BALANCED_BATCH_SAMPLER = bool(_bbs_raw)

    # Session 26 — 7 YAML keys previously ignored (silent bug). Now each YAML
    # value overrides the corresponding runtime constant.
    #
    # (a) Lambda schedule Stage 3 TARGETS from cfg.lambdas.*
    lams = (cfg or {}).get("lambdas", {}) or {}
    if lams:
        _S3_LAMBDA1 = float(lams.get("lambda1", _S3_LAMBDA1))
        _S3_LAMBDA2 = float(lams.get("lambda2", _S3_LAMBDA2))
        _S3_LAMBDA3 = float(lams.get("lambda3", _S3_LAMBDA3))
        _S3_LAMBDA4 = float(lams.get("lambda4", _S3_LAMBDA4))
        _S3_LAMBDA5 = float(lams.get("lambda5", _S3_LAMBDA5))
        _S3_LAMBDA6 = float(lams.get("lambda6", _S3_LAMBDA6))
        _S3_LAMBDA7 = float(lams.get("lambda7", _S3_LAMBDA7))
        # Rescale Stage 2 to preserve its "~60 % of Stage 3" ramp (avoid λ-shock
        # at the S2→S3 boundary when the user edits Stage 3 targets).
        _S2_LAMBDA1 = _S3_LAMBDA1
        _S2_LAMBDA2 = _S3_LAMBDA2
        _S2_LAMBDA3 = _S3_LAMBDA3
        # Phase B+C (2026-05-04) — Lsel paper-faithful S1 warmup + S2 ramp.
        # Sprint 4-R10 only zeroed S1 when YAML lambda4=0; partial-enable
        # (lambda4=0.10) left _S1_LAMBDA4=0.2 hardcoded, producing decreasing
        # S1→S2 ramp (0.2 → 0.10) that crushed direction head on ép 1
        # (NEUTRAL=1.00 stuck both ép 1+2 in Phase B+C run). Paper Eq.30
        # Stage 1 is direction warmup → all auxiliary regularizers must be 0.
        # S2 ramps from 0 to 50% of S3 (similar to lambda5/6/7 ~60% rescale).
        _S2_LAMBDA4 = 0.50 * _S3_LAMBDA4
        _S1_LAMBDA4 = 0.0
        # Rescale Stage 2 to preserve its "~60 % of Stage 3" ramp (avoid λ-shock
        # at the S2→S3 boundary when the user edits Stage 3 targets).
        # Sprint 12 note: this overrides the module-level _S2_LAMBDA5
        # fallback whenever YAML lambdas are present. With research_best
        # lambda5=0.10, effective Stage-2 Lcal is 0.064, not 0.04.
        _S2_LAMBDA5 = 0.64 * _S3_LAMBDA5
        _S2_LAMBDA6 = 0.62 * _S3_LAMBDA6
        _S2_LAMBDA7 = 0.60 * _S3_LAMBDA7
        _ETA_LSEL  = float(lams.get("eta",  _ETA_LSEL))
        _MU_LRISK  = float(lams.get("mu",   _MU_LRISK))

    # (b) Direction-head regularisers
    _FOCAL_GAMMA         = float((cfg or {}).get("focal_gamma",         _FOCAL_GAMMA))
    _LABEL_SMOOTHING_EPS = float((cfg or {}).get("label_smoothing_eps", _LABEL_SMOOTHING_EPS))

    # (c) Data pipeline
    _INGEST_DELAY_MIN    = int((cfg or {}).get("ingest_delay_minutes",  _INGEST_DELAY_MIN))
    # Sprint 1 FIX 2 — number of articles per candle handed to the selector.
    # Capped at a sane upper bound so a typo in YAML cannot blow up VRAM:
    # K=64 with B=8 already uses ~18 MB just for article_emb in FP32, beyond
    # which the article path becomes the throughput bottleneck on T4/P100.
    _ARTICLES_PER_CANDLE = int((cfg or {}).get("articles_per_candle", _ARTICLES_PER_CANDLE))
    if _ARTICLES_PER_CANDLE < 1 or _ARTICLES_PER_CANDLE > 64:
        raise ValueError(
            f"articles_per_candle must be in [1, 64], got {_ARTICLES_PER_CANDLE}"
        )
    _LOOKBACK_HOURS = int((cfg or {}).get("lookback_hours", _LOOKBACK_HOURS))
    if _LOOKBACK_HOURS < 1 or _LOOKBACK_HOURS > 720:
        raise ValueError(
            f"lookback_hours must be in [1, 720], got {_LOOKBACK_HOURS}"
        )
    # Sprint 1.5 — class weights mode (was hard-coded sqrt in _fit_train_stats).
    _CLASS_WEIGHTS_MODE = str((cfg or {}).get("class_weights_mode", _CLASS_WEIGHTS_MODE)).lower()
    if _CLASS_WEIGHTS_MODE not in {"none", "sqrt", "balanced"}:
        raise ValueError(
            f"class_weights_mode must be one of {{none, sqrt, balanced}}, "
            f"got {_CLASS_WEIGHTS_MODE!r}"
        )

    # (d) Coverage target (Eq.37 κ) — affects Lrisk penalty
    _COVERAGE_TARGET     = float((cfg or {}).get("coverage_target",     _COVERAGE_TARGET))

    # (e) Per-horizon faith margin (Eq.36 m_h). MERGE with defaults — users
    # can override a subset of horizons (e.g. only "1h": 0.20) and the rest
    # retain the paper-tuned defaults instead of going missing, which would
    # crash set_horizon() for the unspecified horizons.
    fm = (cfg or {}).get("faith_margin", {}) or {}
    if fm:
        _FAITH_MARGIN_MAP = dict(_FAITH_MARGIN_MAP)   # copy defaults
        for k, v in fm.items():
            if k in {"15m", "1h", "4h", "24h"}:
                _FAITH_MARGIN_MAP[k] = float(v)

    # (f) Per-horizon top-K (Eq.12 K_h)
    tk = (cfg or {}).get("top_k", {}) or {}
    if tk:
        _TOP_K_MAP = {
            "15m": int(tk.get("K_15m", _TOP_K_MAP.get("15m", 3))),
            "1h":  int(tk.get("K_1h",  _TOP_K_MAP.get("1h",  4))),
            "4h":  int(tk.get("K_4h",  _TOP_K_MAP.get("4h",  5))),
            "24h": int(tk.get("K_24h", _TOP_K_MAP.get("24h", 8))),
        }

    # R3 #I1: selection score weights for best-checkpoint picking. Defaults
    # match paper (F1=0.40, Sharpe=0.35, ECE=0.15, MCC=0.10). Override via
    # YAML ``selection_weights: {f1: ..., sharpe: ..., ece: ..., mcc: ...}``.
    # Example: to favour F1 over Sharpe for academic emphasis, set
    # ``{f1: 0.55, sharpe: 0.15, ece: 0.20, mcc: 0.10}``.
    sel = (cfg or {}).get("selection_weights", {}) or {}
    if sel:
        from metrics_safe_alert import set_selection_weights
        set_selection_weights(
            f1=float(sel.get("f1", 0.40)),
            sharpe=float(sel.get("sharpe", 0.35)),
            ece=float(sel.get("ece", 0.15)),
            mcc=float(sel.get("mcc", 0.10)),
            # Sprint 12 — economic gate. Default 0.0 = legacy formula
            # bit-stable. Set > 0 (e.g. 0.10-0.20) in YAML to add direct
            # PnL contribution to checkpoint selection score.
            pnl=float(sel.get("pnl", 0.0)),
        )

# ── Lambda schedule aligned to PDF defaults (Table 4) ─────────────────────────
# PDF defaults: λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.2, λ5=0.1, λ6=0.1, λ7=0.05
#
# Stage 1: direction + return + selection + small Lfac (PDF curriculum design).
# Session 21 Fix G: _S1_LAMBDA3 lifted 0.0 → 0.05 (5% of full Stage-3 value).
# Rationale — phantom-zero gradient trap:
#   With λ3=0 through Stage 1 the factor head receives NO gradient for 8 epochs
#   (20% of total). Its logits stay essentially uniform → Lfac diagnostic
#   reports ≈ log(10) = 2.30 (visible in Fold 1 log: 2.496→2.426 ep 1→8, a
#   7 pp drift due only to weight decay, not learning). At the Stage 2
#   boundary (ep 9) λ3 suddenly activates and Lfac drops from 2.426 to 1.715
#   in ONE epoch — the factor head is then forced to learn from scratch while
#   article_enc has already specialised for direction. Plateau at 1.045 (6×
#   optimum 0.17) because factor-relevant features never got co-learned into
#   article_enc during its most plastic phase.
#   Fix: a small Lfac weight in Stage 1 (0.05) means article_enc receives
#   factor-gradient from epoch 1 — not enough to disturb direction learning
#   but enough to steer the encoder toward factor-informative features. The
#   Stage-1→2 jump becomes continuous (small λ3 → mid λ3), eliminating the
#   cliff and letting the plateau move closer to the floor.
# Lfaith still 0 in Stage 1: it requires a non-random selector to produce a
# useful full-vs-masked gap; with untrained attention the signal is noise.
# Lcal/Lrisk stay 0 until Stage 2 per PDF Section 4.5.1.
_S1_LAMBDA1 = 1.0
_S1_LAMBDA2 = 0.5    # OLD verified Sharpe 0.27.
_S1_LAMBDA3 = 0.0    # Session 36 — REVERT to OLD 0.0 (paper §4.5.1: Lfac OFF in Stage 1).
                     # OLD verified Sharpe 0.27 với λ3=0 in S1. Factor head sẽ học from epoch
                     # ramp-up at S2 entry. Engineering improvement (factor class weights +
                     # log direct) đủ tăng tốc factor convergence trong S2.
_S1_LAMBDA4 = 0.2    # OLD verified — full λ4 in Stage 1.
_S1_LAMBDA5 = 0.005  # Sprint 12 — Stage 1 Lcal nudge from 0.0 to 0.005.
                     # Fixes the Sprint 4 A1 rollback root cause: best
                     # checkpoint promotes from Stage 1 where Lcal was
                     # exactly 0, so any S2/S3 bump didn't affect winning
                     # checkpoint. 0.005 is small enough to not disturb
                     # Ldir convergence (contribution ~0.001 to total loss
                     # at Lcal≈0.25 typical) but non-zero so confidence
                     # head receives gradient signal from epoch 1.
                     # Walk-forward 4h showed T*=70 clamp pervasive in
                     # Stage 2-3 — symptom of Lcal not converging during
                     # training. Earlier Lcal activation should help.
_S1_LAMBDA6 = 0.0    # OLD verified — Lfaith OFF in S1.
_S1_LAMBDA7 = 0.0    # Session 36 — REVERT to OLD 0.0.

# Stage 2 TARGET lambdas: paper adds Lfac + Lfaith.
# FIX: added small λ5=0.02 and λ7=0.01 in Stage 2 (PDF Section 4.5.1 does not
# mandate these are strictly zero in Stage 2; only that they are small compared
# to their Stage 3 values). Small Lcal prevents temperature drift (0.6→2.8
# observed without it). Small Lrisk starts teaching the confidence head to
# distinguish correct vs wrong predictions before Stage 3.
# Stage 2 linearly interpolates from S1 → S2 targets.
_S2_LAMBDA1 = 1.0
_S2_LAMBDA2 = 0.5
_S2_LAMBDA3 = 0.3    # Lfac full weight by end of Stage 2
_S2_LAMBDA4 = 0.2
# Sprint 12 runtime note: _S2_LAMBDA5 below is a fallback constant. When the
# YAML ``lambdas`` block is present, _apply_config_overrides rescales it to
# S2 ~= 64% of S3; for research_best lambda5=0.10, effective S2 Lcal is 0.064.
# Session 36 — REVERT to OLD verified Stage 2 targets.
# OLD: λ5=0.02, λ6=0.10, λ7=0.01 — these gave Sharpe 0.27 mean.
# S2→S3 ramp ratios: λ5 0.02→0.10 (5×), λ6 0.10→0.10 (=), λ7 0.01→0.05 (5×).
# Stage 3 warmup (Session 21 Fix A) smooths the 5× jumps over 5 epochs.
_S2_LAMBDA5 = 0.04   # Sprint 12 — bumped from 0.02 to 0.04. Combined with
                     # _S1_LAMBDA5=0.005 fix this should keep Lcal active
                     # through the entire training trajectory (not just S3).
                     # Previously Sprint 4 A1 bumped to 0.04 then rolled
                     # back because best promoted from S1 where Lcal=0;
                     # now S1 has λ5=0.005 so the entire trajectory has
                     # calibration pressure. Walk-forward 4h T*=70 clamp
                     # pervasive — strong evidence Lcal needs more weight
                     # earlier in training.
_S2_LAMBDA6 = 0.10   # OLD verified — Lfaith full weight by S2 end.
_S2_LAMBDA7 = 0.01   # OLD verified — Lrisk small early signal.

# Stage 3: full loss, aligned to PDF Table 4 defaults.
# Session 21 Fix C: reverted λ5 0.15 → 0.10 (PDF Table 4). The raised value was
# introduced to counter temperature drift but in practice it *created* more
# gradient conflict with the entropy anchor (they both fight overconfidence
# but push in different directions — Lcal toward conf≈P(correct), anchor
# toward uniform-ish). With Fix A (smooth S2→S3 ramp) + Fix B (anchor decays
# as λ5 rises) + Fix D (anchor target lowered), 0.10 is sufficient calibration
# pressure and removes the multi-loss fight that was driving T → 4.2 on
# validation.
# Session 26 — loss-magnitude-balanced λ schedule.
#
# Empirical loss magnitudes observed early in training (Session 21-22 logs):
#   Ldir   ≈ 1.0  (3-class CE, with LS+focal+CW)
#   Lret   ≈ 1.0  (scaled SmoothL1 — ret/q75)
#   Lfac   ≈ 1.0  (10-class CE with class weights, sqrt-softened)
#   Lsel   ≈ 0.5-1.5 (squared cardinality + normalized entropy)
#   Lcal   ≈ 0.10-0.20 (Brier — target class indicator)
#   Lfaith ≈ 0.03-0.10 (relative-gap hinge, capped at per-horizon margin)
#   Lrisk  ≈ 1.0  (weighted CE + coverage penalty)
#   Lvol   ≈ 0.5  (scaled SmoothL1 on σ/scale)
#
# Target: each λ · L contributes ≥ 5 % of Ldir's 1.0 to the total loss.
# Earlier values gave Lcal ~1 %, Lfaith ~1 % (far too weak — the confidence
# head and faithful selector were barely supervised despite "being active").
# We now boost Lcal × 2.5 and Lfaith × 2 to make the contributions visible
# in the gradient. Still below Ldir to preserve the curriculum priority.
_S3_LAMBDA1 = 1.0
_S3_LAMBDA2 = 0.5    # Session 34 — REVERT to OLD 0.5
_S3_LAMBDA3 = 0.3    # Lfac (PDF Table 4) — now effective due to factor class weights
_S3_LAMBDA4 = 0.2
# Session 34 — REVERT λ5/λ6/λ7 về OLD levels.
# OLD code (Sharpe 0.33-0.5) used 0.10/0.10/0.05. S26 boosted to 0.25/0.40/0.10
# "to make contributions visible" but this OVER-CONSTRAINED direction head:
# 3 calibration losses combined with 4 Ldir regularizers (focal+LS+anchor+L2)
# squeezed prediction confidence → policy degenerate → test Sharpe -0.73.
# Engineering improvements (per-factor CW, log direct, sigmoid Lsel, conf_head
# MLP) compensate for "decorative" concern — even at 0.10 weight, the heads
# now learn meaningful signal.
_S3_LAMBDA5 = 0.10   # Lcal — REVERT to OLD (was 0.25)
                     # (Sprint 4 A1 bumped to 0.15 then rolled back: even with
                     # full S3 active, the winning checkpoint stayed in Stage 1
                     # where Lcal=0 — the bump could not affect [BEST] promotion.
                     # Lesson: tuning λ5 alone is not the right intervention for
                     # T-clamp/NEUTRAL-collapse. See Sprint 4-B0 selection
                     # safety + future Phase 8 direction-head dynamics.)
_S3_LAMBDA6 = 0.10   # Lfaith — REVERT to OLD (was 0.40)
_S3_LAMBDA7 = 0.05   # Lrisk — REVERT to OLD (was 0.10)


class SAFEAlertTrainer:
    """Trainer for SAFE-Alert with thesis-style 3-stage curriculum."""

    def __init__(
        self,
        model: SAFEAlertNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,   # 10× default AdamW; 1e-5 was too low for a
                                      # 1-2M param model → observed val/train gap 0.51
                                      # and late-Stage-3 ECE drift 0.02→0.03. 1e-4 is
                                      # the standard AdamW value for this model scale.
        horizon: str = "1h",
        K_h: int = 8,
        grad_clip: float = 1.0,
        class_weights: torch.Tensor = None,
        policy_method: str = "grid",
        tau_percentile: int = _TAU_PERCENTILE,
        gamma_percentile: int = _GAMMA_PERCENTILE,
        policy_confidence_source: Optional[str] = None,
        ablation: Optional[str] = None,
        symbol: Optional[str] = None,   # accepted by model for call-site compatibility
    ):
        self.model = model.to(device)
        self.device = device
        self.horizon = horizon
        self.K_h = K_h
        self.grad_clip = grad_clip
        self.class_weights = class_weights
        self.policy_method = policy_method
        self.tau_percentile = int(tau_percentile)
        self.gamma_percentile = int(gamma_percentile)
        if policy_confidence_source is None:
            policy_confidence_source = _POLICY_CONFIDENCE_SOURCE
        self.policy_confidence_source = str(policy_confidence_source).strip().lower()
        if self.policy_confidence_source not in {"raw", "position"}:
            raise ValueError(
                "policy_confidence_source must be 'raw' or 'position', "
                f"got {self.policy_confidence_source!r}"
            )
        # Session 27 fix #10/#12/#13: Policy continuity via EMA smoothing.
        # Grid search tái-chạy mỗi epoch với confidence distribution thay đổi →
        # τ/γ nhảy loạn (log: 0.544 → 0.496 → 0.505 → ...). Coverage oscillate
        # 0.003 to 0.979. Fix: blend new policy 70% với prev 30% → stable.
        self._prev_tau: float | None = None
        self._prev_gamma: float | None = None
        # Session 33 P3 (Fold 1 audit fix): EMA α 0.7 → 0.5 for heavier smoothing.
        # Observed Cov range across Fold 1 epochs = [0.002, 1.000] — wild oscillation
        # due to underlying confidence distribution shifts. With α=0.7 (formula:
        # τ_new = α·τ_fresh + (1-α)·τ_prev), only 30% of previous epoch is retained,
        # not enough to dampen large per-epoch τ/γ swings. α=0.5 keeps 50% previous,
        # reducing variance of effective policy threshold across epochs while still
        # allowing real shifts to track within ~3 epochs.
        self._policy_ema_alpha: float = 0.5   # was 0.7; lower = heavier smoothing
        # Session 26 anti-overfit: split val temporally into calibration-slice
        # (fits T + τ/γ grid) and selection-slice (reports model_score for
        # early stopping). Without this the same val set that picks τ/γ also
        # scores the model, which biases both toward val-noise. Default 0.7
        # keeps 70 % of val for fitting and holds out the last 30 % as an
        # honest selection measurement. Set to 1.0 to disable (legacy behaviour).
        # Configurable via YAML ``cal_selection_split:``.
        self.cal_selection_split = float(_CAL_SELECTION_SPLIT)
        self.ablation = ablation  # e.g. "w/o_faithfulness" (PDF Section 4.6)
        # Stored for compatibility with model.forward(..., symbol=...).
        # Paper-final Eq.9 ignores symbol and uses market summary + horizon only.
        self.symbol = symbol
        # Populated by fit(dataset=...) so checkpoint saves can snapshot the
        # market scaler state. Left as None when fit() is not used (e.g. pure
        # inference from a pretrained instance) — _build_checkpoint_dict
        # handles that gracefully.
        self.dataset = None
        # Instantiate loss with learnable lambdas
        # Session 26 — use YAML-configurable values (via _apply_config_overrides)
        # instead of hardcoded literals. Stage-3 λ are the TARGET values; the
        # scheduler interpolates S1→S2→S3 each epoch and fill_() overwrites
        # the buffers, so the init values matter only for the first forward
        # pass before the scheduler runs.
        self.loss_fn = MultiObjectiveLoss(
            lambda1=_S3_LAMBDA1,
            lambda2=_S3_LAMBDA2,
            lambda3=_S3_LAMBDA3,
            lambda4=_S3_LAMBDA4,
            lambda5=_S3_LAMBDA5,
            lambda6=_S3_LAMBDA6,
            lambda7=_S3_LAMBDA7,
            K_h=K_h,
            eta=_ETA_LSEL,
            # Per-horizon RELATIVE margin map (Eq.36 + P1 #8). Configurable
            # via YAML `faith_margin:` so per-horizon tuning is not buried
            # inside source. Set per-horizon via set_horizon() below.
            faith_margin=dict(_FAITH_MARGIN_MAP),
            coverage_target=_COVERAGE_TARGET,
            mu=_MU_LRISK,
            class_weights=class_weights,
            # Sprint 3.1 — learned_lambdas=False (paper-faithful curriculum).
            # Paper Eq.30 specifies λ as fixed weights of a multi-objective
            # loss; the §4.5.1 curriculum schedule is the ONLY mechanism that
            # may change them across stages. Earlier in this codebase the flag
            # was flipped to True with a comment claiming OLD had it on (it
            # did not — OLD trainer line 173 set learned_lambdas=False); the
            # Session-35 rationale of "auto-balancing via mild drift" is
            # incorrect because total_loss = Σ λ_i·L_i with L_i > 0 has
            # ∂total/∂λ_i = L_i > 0, so AdamW always pushes λ toward zero.
            # There is no clamp / softplus on λ in MultiObjectiveLoss, so the
            # Stage-1 targets at 0.0 (λ3, λ5, λ6, λ7) drifted NEGATIVE inside
            # each epoch. A negative λ flips the sign of its loss in the
            # objective: the model is rewarded for higher Lfac, worse
            # calibration, worse selective risk. fill_() at the next epoch
            # boundary resets but the drift recurs every epoch. P0.1 in
            # TRAINING_RISK_ANALYSIS_CHECKLIST.md attributes the NEUTRAL
            # collapse, the NEUTRAL → all-UP swing, and the inflated post-hoc
            # temperature to this drift. With learned_lambdas=False the λ
            # become register_buffer tensors (no grad), so only fill_() moves
            # them — that is what the [lambda_target] / [lambda_actual] /
            # [lambda_drift] log triple now verifies every epoch.
            learned_lambdas=False,
            # L_dir regularisers (YAML label_smoothing_eps + focal_gamma).
            label_smoothing_eps=_LABEL_SMOOTHING_EPS,
            focal_gamma=_FOCAL_GAMMA,
            # R3 #E anti-drift anchor, configurable via YAML.
            entropy_anchor_weight=_ENTROPY_ANCHOR_WEIGHT,
            entropy_anchor_target=_ENTROPY_ANCHOR_TARGET,
            # Session 22 Fix 8: logit magnitude L2 penalty.
            logit_l2_weight=_LOGIT_L2_WEIGHT,
            lambda_vol=_LAMBDA_VOL,   # optional volatility extension; 0.0 for paper-final
            lfac_reduction=_LFAC_REDUCTION,   # Session 26 — paper Eq.33 reduction mode (D16)
            lfaith_gap_type=_LFAITH_GAP_TYPE, # Eq.36 gap type (relative|absolute)
            lret_sign_weight=_LRET_SIGN_WEIGHT,
            lret_sign_eps_factor=_LRET_SIGN_EPS_FACTOR,
            epsilon_h_override=_EPSILON_H_OVERRIDE,  # Sprint 4 Phase 8.5 — paired with dataset
            lambda_ret_sign_cls=_LAMBDA_RET_SIGN_CLS,  # Sprint 4-D — auxiliary BCE head weight
            lambda_ret_bin=_LAMBDA_RET_BIN,            # optional tradability bin head weight
            lambda_up_edge=_LAMBDA_UP_EDGE,            # optional UP-vs-rest edge head
            lambda_down_edge=_LAMBDA_DOWN_EDGE,        # optional DOWN-vs-rest edge head
        )
        # Bind the active horizon so L_faith uses the correct margin. This is
        # a no-op if faith_margin was passed as a scalar.
        self.loss_fn.set_horizon(self.horizon)

        # Session 26 — pin loss_fn buffers to the training device once at init.
        # Without this, every forward pass would need `self.return_scale.to(device)`
        # / `self.factor_class_weights.to(device)` / etc. to bring buffers from
        # CPU to GPU. Those per-batch `.to()` calls allocate a new tensor each
        # time (minor but measurable overhead). Pinning once covers all registered
        # buffers (lambdas, return_scale, factor_label_smoothing, factor_class_weights).
        # Plain tensor attributes (self.class_weights) are still reassigned
        # explicitly by _fit_train_stats so this is purely additive.
        self.loss_fn.to(self.device)

        # Per-component optimizer groups. Each loss targets a specific module:
        #   Lfaith → sel_attn (selector needs stronger faithfulness gradient)
        #   Lfac   → factor_mod (factor head is the slowest to converge)
        #   Lcal/Lrisk → conf_head (confidence calibration)
        # lr_mult is stored per group so StageAwareLRScheduler can scale all
        # groups proportionally while preserving relative multipliers.
        sel_params  = list(self.model.sel_attn.parameters())
        fac_params  = list(self.model.factor_mod.parameters())
        conf_params = list(self.model.conf_head.parameters())
        _special_ids = set(
            id(p) for p in sel_params + fac_params + conf_params
        )
        main_params = [
            p for p in self.model.parameters() if id(p) not in _special_ids
        ]
        # Sprint 3.1 — with learned_lambdas=False the λ are register_buffer
        # tensors, so loss_fn.parameters() yields an empty iterable. Adding an
        # empty parameter group to AdamW is technically harmless but leaves a
        # dead "lambdas" group sitting in the param-group list, which the LR
        # scheduler still iterates over. Skip the group entirely when there
        # are no learnable λ; if a future config flips learned_lambdas back
        # to True the same code path picks up the non-empty list and adds the
        # group as before.
        loss_params = [p for p in self.loss_fn.parameters() if p.requires_grad]
        param_groups = [
            {"params": main_params,  "lr": lr,        "lr_mult": 1.0, "name": "main"},
            {"params": sel_params,   "lr": lr * 1.5,  "lr_mult": 1.5, "name": "selector"},
            {"params": fac_params,   "lr": lr * 2.0,  "lr_mult": 2.0, "name": "factor"},
            {"params": conf_params,  "lr": lr * 1.5,  "lr_mult": 1.5, "name": "confidence"},
        ]
        if loss_params:
            # Lambdas use no weight_decay (they are scaling factors, not weight
            # tensors) and a small LR (5 % of base) so within-epoch drift stays
            # small relative to the curriculum-target reset performed at each
            # epoch start. Only added when learned_lambdas=True.
            param_groups.append({
                "params": loss_params, "lr": lr * 0.05, "lr_mult": 0.05,
                "weight_decay": 0.0, "name": "lambdas",
            })
        self.optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
        # Remember base LR for scheduler reconstruction in fit()
        self.base_lr = float(lr)

        # Scheduler is a PLACEHOLDER here; fit() builds the real one once total
        # ``epochs`` is known (StageAwareLRScheduler needs total_epochs).
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda _e: 1.0
        )
        self.current_epoch = 0

    @staticmethod
    def _tensor_is_finite(tensor: Optional[torch.Tensor]) -> bool:
        return tensor is None or bool(torch.isfinite(tensor).all().item())

    def _outputs_are_finite(self, outputs: Dict[str, Optional[torch.Tensor]]) -> bool:
        for key in (
            "dir_logits", "ret_pred", "ret_sign_logit", "confidence",
            "attn_weights", "p_fac_all",
        ):
            value = outputs.get(key)
            if value is not None and not self._tensor_is_finite(value):
                return False
        return True

    def _has_non_finite_gradients(self) -> bool:
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False

    def _sanitize_model_parameters(self) -> int:
        repaired = 0
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                finite_mask = torch.isfinite(param)
                if not finite_mask.all():
                    repaired += int((~finite_mask).sum().item())
                    param.data.copy_(torch.nan_to_num(param.data, nan=0.0, posinf=1e4, neginf=-1e4))
                param.data.clamp_(min=-1e4, max=1e4)

            if hasattr(self.model, "sel_attn") and hasattr(self.model.sel_attn, "tau"):
                # tau is now a raw unconstrained scalar fed through softplus in
                # _tau_clamped(). Repair: reset to 0.0 (softplus(0)+0.1 ≈ 0.79)
                # on NaN/Inf; do NOT hard-clamp since that would create a dead zone.
                tau = self.model.sel_attn.tau.data
                if not torch.isfinite(tau).all():
                    repaired += int((~torch.isfinite(tau)).sum().item())
                    tau.copy_(torch.nan_to_num(tau, nan=0.0, posinf=2.0, neginf=-2.0))
        return repaired


    def _update_loss_weights_for_stage(self, epoch: int, total_epochs: int) -> None:
        """Apply the thesis 3-stage lambda schedule with Stage 3 warmup.

        Schedule (Session 21 Fix A — smooth S2→S3 transition):
          Stage 1:            frac ∈ (0, STAGE1_END_FRAC]                — fixed S1 values
          Stage 2:            frac ∈ (STAGE1_END_FRAC, STAGE2_END_FRAC]  — S1 → S2 linear
          Stage 3 warmup:     frac ∈ (STAGE2_END_FRAC, STAGE2_END_FRAC + STAGE3_WARMUP_FRAC]
                                                                         — S2 → S3 linear
          Stage 3 steady:     frac > STAGE2_END_FRAC + STAGE3_WARMUP_FRAC — fixed S3 values

        Without the Stage-3 warmup, lambdas jumped from S2 targets to S3 targets
        in a single epoch (λ5: 0.08→0.15, etc.) which, combined with the LR halve
        at Stage 3 entry, drove monotone val-loss degradation. The warmup keeps
        the loss surface change per epoch within the range the optimizer can
        track under the decaying LR schedule.
        """
        frac = epoch / max(total_epochs, 1)
        s1 = (
            _S1_LAMBDA1, _S1_LAMBDA2, _S1_LAMBDA3, _S1_LAMBDA4,
            _S1_LAMBDA5, _S1_LAMBDA6, _S1_LAMBDA7,
        )
        s2 = (
            _S2_LAMBDA1, _S2_LAMBDA2, _S2_LAMBDA3, _S2_LAMBDA4,
            _S2_LAMBDA5, _S2_LAMBDA6, _S2_LAMBDA7,
        )
        s3 = (
            _S3_LAMBDA1, _S3_LAMBDA2, _S3_LAMBDA3, _S3_LAMBDA4,
            _S3_LAMBDA5, _S3_LAMBDA6, _S3_LAMBDA7,
        )
        # Clamp warmup length to [0, 1 - STAGE2_END_FRAC]: cannot exceed the
        # remaining post-Stage-2 budget. Also cap at 5 epochs absolute —
        # beyond that the ramp is slower than the optimizer can exploit.
        max_warmup_frac = max(1.0 - _STAGE2_END_FRAC, 1e-8)
        warmup_frac = min(max(_STAGE3_WARMUP_FRAC, 0.0), max_warmup_frac)
        if total_epochs > 0:
            warmup_frac_5ep_cap = 5.0 / total_epochs
            warmup_frac = min(warmup_frac, warmup_frac_5ep_cap)
        stage3_warmup_end_frac = _STAGE2_END_FRAC + warmup_frac

        if frac <= _STAGE1_END_FRAC:
            target = s1
        elif frac <= _STAGE2_END_FRAC:
            # Interpolate S1 -> S2: ramp Lfac (lambda3), Lfaith (lambda6),
            # and small Lcal/Lrisk. Runtime S2 targets may be rescaled from
            # YAML lambdas, so this comment intentionally avoids hard-coded
            # lambda5/lambda7 values.
            mix = (frac - _STAGE1_END_FRAC) / max(_STAGE2_END_FRAC - _STAGE1_END_FRAC, 1e-8)
            target = tuple(a + mix * (b - a) for a, b in zip(s1, s2))
        elif warmup_frac > 0 and frac <= stage3_warmup_end_frac:
            # Stage 3 warmup: linear S2 → S3 over warmup_frac of total epochs.
            mix = (frac - _STAGE2_END_FRAC) / warmup_frac
            mix = min(max(mix, 0.0), 1.0)
            target = tuple(a + mix * (b - a) for a, b in zip(s2, s3))
        else:
            target = s3

        # Ablation study overrides (PDF Section 4.6):
        # "w/o_faithfulness" — keep λ6 (Lfaith) = 0 throughout all stages.
        # "w/o_factor"       — keep λ3 (Lfac)   = 0 throughout all stages.
        if self.ablation == "w/o_faithfulness":
            target = tuple(0.0 if i == 5 else v for i, v in enumerate(target))
        elif self.ablation == "w/o_factor":
            target = tuple(0.0 if i == 2 else v for i, v in enumerate(target))

        with torch.no_grad():
            for name, value in zip(
                ["lambda1", "lambda2", "lambda3", "lambda4", "lambda5", "lambda6", "lambda7"],
                target,
            ):
                getattr(self.loss_fn, name).fill_(float(value))
        # Sprint 3.1 — snapshot the curriculum target so the per-epoch logger
        # can compare against the live ``loss_fn.lambda*`` values. Drift = 0
        # is the expected state under learned_lambdas=False; any non-zero
        # entry signals either a bug in fill_() ordering or a regression that
        # re-enabled gradient flow on the λ buffers.
        self._last_lambda_target = tuple(float(v) for v in target)

        # Sprint 5a — auxiliary tradability bin head curriculum ramp.
        # Stage 1: 0.0 (head latent, no loss contribution).
        # Stage 2 end: 0.5 × _LAMBDA_RET_BIN target (gentle ramp during S2).
        # Stage 3 warmup end / steady: _LAMBDA_RET_BIN (YAML target).
        # Same warmup-fraction logic as the 7-tuple ramp above keeps the
        # auxiliary weight increase synced with the LR schedule transition.
        # _LAMBDA_RET_BIN is a plain Python float (not a buffer), so direct
        # assignment is safe and does not require torch.no_grad().
        ret_bin_target = float(_LAMBDA_RET_BIN)
        if frac <= _STAGE1_END_FRAC:
            ret_bin_actual = 0.0
        elif frac <= _STAGE2_END_FRAC:
            mix = (frac - _STAGE1_END_FRAC) / max(_STAGE2_END_FRAC - _STAGE1_END_FRAC, 1e-8)
            ret_bin_actual = mix * 0.5 * ret_bin_target
        elif warmup_frac > 0 and frac <= stage3_warmup_end_frac:
            mix = (frac - _STAGE2_END_FRAC) / warmup_frac
            mix = min(max(mix, 0.0), 1.0)
            ret_bin_actual = (0.5 + 0.5 * mix) * ret_bin_target
        else:
            ret_bin_actual = ret_bin_target
        self.loss_fn.lambda_ret_bin = ret_bin_actual

        # Optional binary UP/DOWN edge-head curriculum. Same shape as ret_bin:
        # off in Stage 1, half-target by Stage 2 end, target in Stage 3.
        def _aux_ramp(target_value: float) -> float:
            target_value = float(target_value)
            if frac <= _STAGE1_END_FRAC:
                return 0.0
            if frac <= _STAGE2_END_FRAC:
                mix2 = (frac - _STAGE1_END_FRAC) / max(_STAGE2_END_FRAC - _STAGE1_END_FRAC, 1e-8)
                return mix2 * 0.5 * target_value
            if warmup_frac > 0 and frac <= stage3_warmup_end_frac:
                mix3 = (frac - _STAGE2_END_FRAC) / warmup_frac
                mix3 = min(max(mix3, 0.0), 1.0)
                return (0.5 + 0.5 * mix3) * target_value
            return target_value

        self.loss_fn.lambda_up_edge = _aux_ramp(_LAMBDA_UP_EDGE)
        self.loss_fn.lambda_down_edge = _aux_ramp(_LAMBDA_DOWN_EDGE)

    def _prepare_attn_weights(
        self,
        attn_w: torch.Tensor,
        article_mask: torch.Tensor,
        batch_size: int,
        num_articles: int,
    ) -> torch.Tensor:
        """Validate shape and scale attention weights for Lsel (PDF Eq.34).

        SelectiveAttention outputs softmax-normalized weights (sum≈1 per sample).
        When a sample has fewer than K_h valid articles, the effective target is
        min(K_h, n_valid_articles). Scaling to that target avoids over-penalizing
        sparse-news samples while preserving the paper's selective-evidence logic.
        """
        if attn_w is None:
            return torch.zeros(batch_size, num_articles, device=self.device)
        if attn_w.dim() == 1:
            attn_w = attn_w.unsqueeze(1)        # (B,) → (B, 1)
        elif attn_w.dim() != 2:
            raise ValueError(f"attn_weights must be 2D (B,K), got {attn_w.shape}")
        if article_mask is not None:
            effective_k = article_mask.float().sum(dim=1, keepdim=True).clamp(
                min=0.0,
                max=float(self.loss_fn.K_h),
            )
        else:
            effective_k = torch.full(
                (batch_size, 1),
                float(self.loss_fn.K_h),
                device=self.device,
                dtype=attn_w.dtype,
            )
        # Scale softmax-normalized weights so Σα̃_i ≈ min(K_h, n_valid_articles).
        row_sum = attn_w.sum(dim=1, keepdim=True)
        valid_rows = (row_sum.squeeze(1) > 1e-8) & (effective_k.squeeze(1) > 0)
        if valid_rows.any():
            attn_w = attn_w.clone()
            attn_w[valid_rows] = (
                attn_w[valid_rows] / row_sum[valid_rows] * effective_k[valid_rows]
            )
        return attn_w

    def _enter_stage3(self) -> None:
        """Mark the Stage 3 transition.

        With ``StageAwareLRScheduler`` (R3 #A), the scheduler already knows the
        Stage-2/Stage-3 boundaries and produces the correct LR for every epoch
        deterministically. This method therefore only LOGS the transition —
        no manual LR surgery needed, no base_lrs rewrite.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"[STAGE 3] Fine-tuning phase entered. "
            f"LR={current_lr:.3e} (scheduler controls decay to eta_lr)"
        )

    def _policy_confidence(
        self,
        confidence: np.ndarray,
        dir_probs: np.ndarray,
    ) -> np.ndarray:
        """Return the confidence stream used by the Eq.26 tau gate."""
        conf = np.asarray(confidence, dtype=np.float32)
        if self.policy_confidence_source == "raw":
            return conf
        probs = np.asarray(dir_probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                "position policy confidence requires dir_probs with shape (N, 3)"
            )
        return (conf * (1.0 - probs[:, 1])).astype(np.float32)

    def train_epoch(self, train_loader, accumulate_steps: int = _ACCUM_STEPS) -> Dict[str, float]:
        """Train one epoch with gradient accumulation.

        accumulate_steps=2 with batch_size=8 gives effective batch_size=16
        without additional memory overhead.
        """
        self.model.train()
        # Keep "Lvol" in the accumulator because loss_dict always reports it.
        # Paper-final runs set lambda_vol=0; lambda_vol>0 opts into the
        # volatility-regression extension.
        losses = {"loss": 0, "Ldir": 0, "Lret": 0, "Lret_sign": 0, "Lfac": 0, "Lsel": 0,
                  "Lcal": 0, "Lfaith": 0, "Lrisk": 0, "Lvol": 0,
                  "LretCls": 0, "LretBin": 0,
                  "LupEdge": 0, "LdownEdge": 0}
        count = 0
        skipped_batches = 0
        skipped_steps = 0
        # Sprint 8 diagnostic — full-epoch train PredDist accumulator. Compares
        # against val [PredDist] to distinguish (a) generalization/regime shift
        # (train balanced + val collapsed) from (b) optimization collapse (train
        # also collapsed). Per họ Sprint 7 verdict: full epoch >> last K batches
        # because end-of-epoch sampler/curriculum state biases tail. 28k int8
        # ≈ 28KB memory, trivial cost. Cleared at function start so each epoch
        # starts fresh.
        train_dir_preds_acc: list = []
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            market_feat = batch["market_features"].to(self.device)
            article_emb = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
            # Session 23 P0 #3: paper Eq.16 bar sequences — only present when
            # the dataset was built with market_bars. Use None otherwise so
            # the model's scalar path stays active.
            market_bars = batch.get("market_bars")
            if market_bars is not None:
                market_bars = market_bars.to(self.device)
            article_mask = batch["article_mask"].to(self.device)
            dir_labels = batch["direction"].to(self.device)
            ret_labels = batch["return"].to(self.device)
            fac_labels = batch["factor"].to(self.device)
            if self.current_epoch == 1 and count == 0:
                logger.debug("ret_labels: shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                             ret_labels.shape, ret_labels.min(), ret_labels.max(),
                             ret_labels.mean(), ret_labels.std())
                logger.debug("fac_labels: shape=%s min=%.4f max=%.4f row_sum=%.4f",
                             fac_labels.shape, fac_labels.min(), fac_labels.max(),
                             fac_labels.sum(dim=-1).mean())

            # Data contract sanity — runs on the FIRST batch of epoch 1 only.
            # Previously these fired on every batch via `assert`, which (a) wasted
            # cycles on the hot path and (b) silently vanished under `python -O`
            # (assertions are stripped). The first-batch gate + ValueError is the
            # senior-level pattern: validate once, fail loudly, keep the loop hot.
            if self.current_epoch == 1 and count == 0:
                if fac_labels.dim() != 3:
                    raise ValueError(
                        f"fac_labels must be 3D (B,K,C), got {fac_labels.dim()}D "
                        f"shape {fac_labels.shape}"
                    )
                if torch.isnan(fac_labels).any():
                    raise ValueError("fac_labels contains NaN")
                if torch.isinf(fac_labels).any():
                    raise ValueError("fac_labels contains Inf")
                if not ((fac_labels >= 0).all() and (fac_labels <= 1).all()):
                    raise ValueError(
                        f"fac_labels outside [0, 1]: "
                        f"min={fac_labels.min().item():.4f}, max={fac_labels.max().item():.4f}"
                    )
                if torch.isnan(market_feat).any() or torch.isinf(market_feat).any():
                    raise ValueError("market_feat contains NaN/Inf")

            # Forward pass
            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                symbol=self.symbol,
                market_bars=market_bars,  # Session 23 P0 #3: paper Eq.16
            )

            if not self._outputs_are_finite(outputs):
                skipped_batches += 1
                repaired = self._sanitize_model_parameters()
                self.optimizer.zero_grad()
                logger.warning(
                    "Epoch %d batch %d: non-finite model outputs (repaired=%d, skipped)",
                    self.current_epoch, batch_idx, repaired,
                )
                continue

            # ── Masked forward for Lfaith (Eq.36) ────────────────────────────
            # forward_masked internally forces eval() mode (see P0 #6 guard in
            # SAFEAlertNet) so Dropout noise never corrupts the faithfulness
            # gap. We no longer need to eval/train wrap the caller.
            has_articles = article_mask.sum(dim=1) > 0  # (B,) per-sample boolean
            if has_articles.any():
                # Do NOT use torch.no_grad() here. Lfaith needs gradient through
                # masked_dir_logits so the loss can push p_mask DOWN (model less
                # confident without articles). With no_grad, only p_full has gradient
                # → Lfaith pushes overconfidence instead of faithfulness (inverted signal).
                # forward_masked() handles eval-mode internally to disable Dropout.
                masked_outputs = self.model.forward_masked(
                    market_feat=market_feat,
                    horizon=self.horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta,
                    symbol=self.symbol,
                    market_bars=market_bars,  # Session 23 P0 #3
                )
                masked_dir_logits = masked_outputs["dir_logits"]  # (B, 3)

                if self.current_epoch == 1 and count == 0:
                    full_pred   = F.softmax(outputs["dir_logits"], dim=-1).max(dim=-1)[0]
                    masked_pred = F.softmax(masked_dir_logits,     dim=-1).max(dim=-1)[0]
                    logger.debug(
                        "Epoch 1 faithfulness gap (full-masked): %s",
                        (full_pred - masked_pred)[:3].tolist(),
                    )
            else:
                masked_dir_logits = None  # no articles in batch → Lfaith = 0

            if not self._tensor_is_finite(masked_dir_logits):
                skipped_batches += 1
                repaired = self._sanitize_model_parameters()
                self.optimizer.zero_grad()
                logger.warning(
                    "Epoch %d batch %d: non-finite masked logits detected (repaired=%d, skipped)",
                    self.current_epoch, batch_idx, repaired,
                )
                continue

            # Validate shape and scale for Lsel (PDF Eq.34): Σα̃_i must ≈ K_h
            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"], article_mask, market_feat.shape[0], article_emb.shape[1]
            )
            if self.current_epoch == 1 and count == 0:
                logger.debug("attn_w shape=%s sum/sample=%.4f (target K_h=%d)",
                             attn_w.shape, attn_w.sum(dim=1).mean().item(), self.loss_fn.K_h)

            # calibration_targets = correctness indicator (1=correct, 0=wrong) for Lcal (Eq.35)
            dir_preds = outputs["dir_logits"].argmax(dim=-1)
            # Sprint 8 diagnostic — accumulate train predictions for [TrainDist].
            # Cheap detach + cpu numpy conversion, called once per batch.
            train_dir_preds_acc.append(dir_preds.detach().cpu().numpy().astype(np.int8))
            dir_correct = (dir_preds == dir_labels).long()

            # Optional volatility extension: vol_labels comes from dataset
            # (B,), vol_pred from the model head. Paper-final runs keep
            # lambda_vol=0, so the term is reported but inactive.
            vol_labels = batch.get("volatility")
            if vol_labels is not None:
                vol_labels = vol_labels.to(self.device)
            loss_dict = self.loss_fn(
                dir_logits=outputs["dir_logits"],
                dir_labels=dir_labels,
                ret_pred=outputs["ret_pred"],
                ret_labels=ret_labels,
                fac_probs=outputs["p_fac_all"] if outputs["p_fac_all"] is not None else torch.zeros(market_feat.shape[0], FACTOR_CLASSES, device=self.device),
                fac_labels=fac_labels,
                attn_weights=attn_w,
                confidence=outputs["confidence"],
                calibration_targets=dir_correct,
                masked_dir_logits=masked_dir_logits,
                article_mask=article_mask,  # Pass mask so Lfac skips padding articles
                soft_gates=outputs.get("soft_gates"),  # σ(a_i) for Lsel sel_term gradient
                vol_pred=outputs.get("vol_pred"),      # optional volatility extension
                vol_labels=vol_labels,                  # optional volatility extension
                ret_sign_logit=outputs.get("ret_sign_logit"),  # optional auxiliary BCE head
                ret_bin_logits=outputs.get("ret_bin_logits"),  # optional tradability bin head
                up_edge_logit=outputs.get("up_edge_logit"),    # optional UP-vs-rest edge head
                down_edge_logit=outputs.get("down_edge_logit"),# optional DOWN-vs-rest edge head
            )

            total_loss = loss_dict["loss"]

            if self.current_epoch == 1 and count == 0:
                ret_pred_dbg = outputs["ret_pred"]
                corr = torch.corrcoef(torch.stack([ret_pred_dbg, ret_labels]))[0, 1].item()
                logger.debug("Epoch 1 return head: pred=[%.4f,%.4f] label=[%.4f,%.4f] Lret=%.4f corr=%.4f",
                             ret_pred_dbg.min(), ret_pred_dbg.max(),
                             ret_labels.min(), ret_labels.max(),
                             loss_dict["Lret"], corr)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                skipped_batches += 1
                logger.warning("Epoch %d batch %d: NaN/Inf loss — Ldir=%.4f Lret=%.4f Lfac=%.4f (skipped)",
                               self.current_epoch, batch_idx,
                               loss_dict["Ldir"], loss_dict["Lret"], loss_dict["Lfac"])
                self.optimizer.zero_grad()
                continue  # Skip this batch

            # Gradient accumulation: scale loss before backward
            (total_loss / accumulate_steps).backward()

            # Step optimizer every accumulate_steps batches (or at end of epoch)
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % accumulate_steps == 0 or is_last_batch:
                if self._has_non_finite_gradients():
                    skipped_steps += 1
                    repaired = self._sanitize_model_parameters()
                    self.optimizer.zero_grad()
                    logger.warning(
                        "Epoch %d batch %d: non-finite gradients detected (repaired=%d, step skipped)",
                        self.current_epoch, batch_idx, repaired,
                    )
                    continue
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                # Prophylactic per-step parameter sanitize removed (scanned 2-5M
                # floats every batch → ~30-60s/epoch overhead). The forward_masked
                # NaN root cause is already fixed via max-offset softmax, and the
                # pre-step gradient check above still catches any residual NaN —
                # running the scan AFTER every step was defensive but unnecessary.
                self.optimizer.zero_grad()

            # Session 26 — convert loss tensor to scalar float before accumulation.
            # loss_dict["loss"] is a tensor with grad_fn (the one we just
            # .backward()'d). The other 8 keys are already Python floats (via
            # _to_scalar). Without .item() the "loss" accumulator would hold
            # a chain of tensors across batches, pinning memory even though
            # the gradient graph is freed. .item() detaches to a Python float.
            for k in losses:
                v = loss_dict[k]
                losses[k] += v.item() if torch.is_tensor(v) else v
            count += 1

        if count == 0:
            raise RuntimeError(
                f"Training collapsed at epoch {self.current_epoch}: all batches were skipped "
                f"(invalid outputs/losses={skipped_batches}, invalid steps={skipped_steps})."
            )

        for k in losses:
            losses[k] /= max(count, 1)

        # Sprint 8 diagnostic — full-epoch train PredDist computation. Catches
        # the Sprint 7 case where val PredDist N=0.99 collapses but train side
        # may be balanced (sampler effective). Decision matrix in fold-loop log.
        if train_dir_preds_acc:
            _train_preds = np.concatenate(train_dir_preds_acc)
            _train_total = max(_train_preds.size, 1)
            _train_bins = np.bincount(_train_preds, minlength=3)
            losses["train_pred_dist_down"]    = float(_train_bins[0]) / _train_total
            losses["train_pred_dist_neutral"] = float(_train_bins[1]) / _train_total
            losses["train_pred_dist_up"]      = float(_train_bins[2]) / _train_total
            losses["train_pred_count"]        = int(_train_total)
        else:
            losses["train_pred_dist_down"]    = 0.0
            losses["train_pred_dist_neutral"] = 0.0
            losses["train_pred_dist_up"]      = 0.0
            losses["train_pred_count"]        = 0
        return losses

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        tau: Optional[float] = None,
        gamma: Optional[float] = None,
        temperature: Optional[float] = None,
        compute_explanations: bool = True,
    ) -> Dict[str, float]:
        """Validate and compute balanced forecast + alert utility metrics.

        Wrapped in ``@torch.no_grad()`` so the main forward does not allocate an
        autograd graph per batch. ``model.eval()`` below only disables dropout
        and BN stat updates — it does *not* disable gradient tracking. Without
        this decorator every validation pass was holding hundreds of MB of
        activation graph for no purpose (and risked OOM on large val splits).

        When ``compute_explanations=False`` skip the 3 extra forward passes per
        batch that only feed explanation metrics (forward_masked, selected-only,
        no-articles). This cuts validate() time ~60% and is safe to toggle —
        model selection uses the always-computed core metrics (F1, MCC, ECE,
        Sharpe); faithfulness metrics fall back to their "no change" defaults.
        """
        self.model.eval()
        val_losses = {"loss": 0, "dir_acc": 0, "count": 0}

        all_dir_preds = []
        all_dir_labels = []
        all_confidence = []
        all_ret_labels = []
        all_ret_preds = []
        all_dir_logits = []
        all_dir_probs = []
        all_regimes = []
        # Collect optional ret_bin_logits (B,3) for tradability diagnostics.
        all_ret_bin_logits = []
        # Collect optional binary edge logits for diagnostics.
        all_up_edge_logits = []
        all_down_edge_logits = []
        all_attn_sums = []
        all_attn_targets = []
        # Sprint 1 FIX 7 (corrected math) — RawAttnSum diagnostic. The forward
        # value of α̃ inside SelectiveAttention is α · 𝕀(Top-K) via STE, so it
        # is the Top-K SUBSET of softmax mass — NOT a full softmax that sums
        # to 1. With K=articles_per_candle and K_h selected slots:
        #   uniform attention   → sum ≈ K_h / K, max ≈ 1 / K
        #   concentrated attention → sum → 1.0, max → 1.0 (one article wins)
        # _prepare_attn_weights additionally rescales to the effective K_h for
        # the Lsel coverage term, making the legacy AttnSum trivially equal to
        # K_h regardless of selector quality. Tracking the unrescaled sum and
        # max is what tells us whether the scorer is actually concentrating
        # evidence on a small subset over training. Read these as RawTopKMass:
        # rising max + rising sum = selector is becoming decisive.
        all_raw_attn_sums = []
        all_raw_attn_maxes = []
        # R3 #F (extended): soft_gate diagnostics — whether the selector's raw
        # score distribution is changing across epochs. AttnSum (softmax α̃·K_h)
        # is a construction-bounded number; soft_gates.sum() is free and will
        # move when the scorer updates.
        all_soft_gate_sums = []
        all_soft_gate_maxes = []
        # Explanation / faithfulness metric accumulators (PDF Section 4.4.2)
        all_masked_dir_probs = []      # deletion: probs WITHOUT top-K articles
        all_selected_only_probs = []   # sufficiency/insertion: probs WITH ONLY top-K articles
        all_no_article_probs = []      # insertion baseline: market-only (no articles at all)
        all_factor_preds = []          # factor consistency: dominant factor per sample


        for batch in val_loader:
            market_feat = batch["market_features"].to(self.device)
            article_emb = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
            article_mask = batch["article_mask"].to(self.device)
            dir_labels = batch["direction"].to(self.device)
            fac_labels = batch["factor"].to(self.device)
            # Session 23 P0 #3: forward bar sequences if present (paper Eq.16).
            market_bars = batch.get("market_bars")
            if market_bars is not None:
                market_bars = market_bars.to(self.device)

            outputs = self.model(
                market_feat,
                horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                symbol=self.symbol,
                market_bars=market_bars,  # Session 23 P0 #3
            )

            has_articles = article_mask.any(dim=1).any()
            with torch.no_grad():
                if has_articles and compute_explanations:
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta,
                        symbol=self.symbol,
                        market_bars=market_bars,  # Session 23 P0 #3
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]
                else:
                    masked_dir_logits = None

            if compute_explanations and not self._tensor_is_finite(masked_dir_logits):
                logger.warning("Validation: skipping batch with non-finite masked logits")
                continue

            # Sprint 1 FIX 7 (corrected) — capture α̃ BEFORE _prepare_attn_weights
            # rescales to effective K_h. The forward value of α̃ is α · 𝕀(Top-K)
            # via STE, so per sample its sum is the Top-K share of softmax mass
            # (in [K_h/K, 1.0] in expectation, depending on concentration).
            raw_attn = outputs.get("attn_weights")
            if raw_attn is not None and raw_attn.dim() == 2:
                # Mask padded slots so padding indices contribute zero (forward
                # already zeros out padded α̃ via the topk-and-mask intersection,
                # but multiplying by article_mask is an extra safety net for
                # ablations that bypass the standard attention path).
                raw_attn_masked = raw_attn * article_mask.float()
                all_raw_attn_sums.append(raw_attn_masked.sum(dim=1).detach())
                all_raw_attn_maxes.append(raw_attn_masked.max(dim=1).values.detach())
            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"], article_mask, market_feat.shape[0], article_emb.shape[1]
            )
            # Accumulate GPU tensors; defer .cpu() / .numpy() until after the loop
            # to avoid CUDA-sync stalls on every batch (old pattern: 10+ .cpu()
            # calls per batch forced GPU pipeline flushes repeatedly).
            all_attn_sums.append(attn_w.sum(dim=1).detach())
            all_attn_targets.append(
                article_mask.float().sum(dim=1).clamp(max=float(self.loss_fn.K_h))
            )
            # R3 #F (extended): track soft_gates entropy to verify the selector
            # is actually learning. AttnSum is a ~constant by construction
            # (softmax-normalized α̃ × K_h = K_h for ≥K_h valid articles), so
            # it cannot reveal whether the scorer updates weights over training.
            # Soft_gates sum is NOT simplex-constrained — if it stays flat
            # across epochs the selector is frozen; if it varies, selector is
            # learning. Stored per-batch, averaged for log.
            soft_gates = outputs.get("soft_gates")
            if soft_gates is not None:
                all_soft_gate_sums.append(soft_gates.sum(dim=1).detach())
                all_soft_gate_maxes.append(soft_gates.max(dim=1).values.detach())

            dir_preds = outputs["dir_logits"].argmax(dim=-1)
            dir_correct = (dir_preds == dir_labels).long()

            loss_dict = self.loss_fn(
                dir_logits=outputs["dir_logits"],
                dir_labels=dir_labels,
                ret_pred=outputs["ret_pred"],
                ret_labels=batch["return"].to(self.device),
                fac_probs=outputs["p_fac_all"] if outputs["p_fac_all"] is not None else torch.zeros(market_feat.shape[0], FACTOR_CLASSES, device=self.device),
                fac_labels=fac_labels,
                attn_weights=attn_w,
                confidence=outputs["confidence"],
                calibration_targets=dir_correct,
                masked_dir_logits=masked_dir_logits,
                article_mask=article_mask,
                soft_gates=outputs.get("soft_gates"),
                # Optional volatility extension; inactive when lambda_vol=0.
                vol_pred=outputs.get("vol_pred"),
                vol_labels=batch.get("volatility").to(self.device)
                           if batch.get("volatility") is not None else None,
                # Sprint 4-D — auxiliary BCE head; loss_dict["LretCls"] surfaces
                # in val_losses for cell-6 inspection and future Phase 8.8 diag.
                ret_sign_logit=outputs.get("ret_sign_logit"),
                # Sprint 5 — tradability bin head; loss_dict["LretBin"] surfaces
                # in val_losses so train/val LretBin are computed identically.
                ret_bin_logits=outputs.get("ret_bin_logits"),
                up_edge_logit=outputs.get("up_edge_logit"),
                down_edge_logit=outputs.get("down_edge_logit"),
            )

            val_losses["loss"] += loss_dict["loss"].item()
            val_losses["count"] += 1

            full_probs = F.softmax(outputs["dir_logits"], dim=-1)
            all_dir_preds.append(dir_preds)
            all_dir_labels.append(dir_labels)
            all_confidence.append(outputs["confidence"])
            all_ret_labels.append(batch["return"].to(self.device))
            all_ret_preds.append(outputs["ret_pred"])
            all_dir_logits.append(outputs["dir_logits"])
            all_dir_probs.append(full_probs)
            _regime = batch.get("regime")
            if _regime is not None:
                all_regimes.append(_regime.to(self.device))
            # Sprint 5a — collect tradability bin logits for diagnostic metrics.
            # Always present in output dict (zero-init head). Skipped if missing
            # to keep this code-path robust to checkpoint schema variations.
            _rbl = outputs.get("ret_bin_logits")
            if _rbl is not None:
                all_ret_bin_logits.append(_rbl)
            _upl = outputs.get("up_edge_logit")
            if _upl is not None:
                all_up_edge_logits.append(_upl.detach())
            _dnl = outputs.get("down_edge_logit")
            if _dnl is not None:
                all_down_edge_logits.append(_dnl.detach())

            # ── Explanation metrics data collection (PDF Section 4.4.2) ──────────
            # Deletion: probs WITHOUT top-K articles.
            if masked_dir_logits is not None:
                all_masked_dir_probs.append(F.softmax(masked_dir_logits, dim=-1))
            else:
                # Either no articles in batch, or compute_explanations=False.
                # Use full probs as the fallback — deletion_drop = 0 for this batch,
                # which is the neutral baseline when the signal cannot be measured.
                all_masked_dir_probs.append(full_probs)

            # Sufficiency / insertion extras — only when explanations are requested
            # (these cost 2 extra forward passes per batch on top of forward_masked).
            sel_mask = outputs.get("selected_mask")
            if compute_explanations and sel_mask is not None and has_articles:
                with torch.no_grad():
                    sel_outputs = self.model(
                        market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=sel_mask.float(),   # only selected articles are "valid"
                        article_meta_vec=article_meta,
                        symbol=self.symbol,
                        market_bars=market_bars,  # Session 23 P0 #3
                    )
                all_selected_only_probs.append(F.softmax(sel_outputs["dir_logits"], dim=-1))
            else:
                all_selected_only_probs.append(full_probs)

            # True Insertion baseline: probs with NO articles.
            if compute_explanations:
                no_article_mask = torch.zeros_like(article_mask)
                with torch.no_grad():
                    no_art_outputs = self.model(
                        market_feat, horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=no_article_mask,
                        article_meta_vec=article_meta,
                        symbol=self.symbol,
                        market_bars=market_bars,  # Session 23 P0 #3
                    )
                all_no_article_probs.append(F.softmax(no_art_outputs["dir_logits"], dim=-1))
            else:
                all_no_article_probs.append(full_probs)

            # Factor consistency: dominant factor per sample
            if outputs["p_fac_all"] is not None and outputs["attn_weights"] is not None:
                attn_for_fac = attn_w.unsqueeze(-1)  # (B, K, 1)
                fac_per_sample = (outputs["p_fac_all"] * attn_for_fac).sum(dim=1).argmax(dim=-1)
                all_factor_preds.append(fac_per_sample)

            val_losses["dir_acc"] += (dir_preds == dir_labels).float().mean().item()

        if val_losses["count"] == 0:
            raise RuntimeError("Validation collapsed: all batches were skipped due to non-finite outputs.")

        val_losses["loss"] /= max(val_losses["count"], 1)
        val_losses["dir_acc"] /= max(val_losses["count"], 1)
        del val_losses["count"]

        # Single GPU->CPU transfer for the whole validation set. Previously
        # each batch did 10+ .cpu().numpy() calls (one per tensor), forcing a
        # CUDA sync at every iteration and serialising the GPU pipeline.
        def _cat_to_np(lst: list) -> np.ndarray:
            if not lst:
                return np.empty(0)
            return torch.cat(lst, dim=0).detach().cpu().numpy()

        all_dir_preds   = _cat_to_np(all_dir_preds)
        all_dir_labels  = _cat_to_np(all_dir_labels)
        all_confidence  = _cat_to_np(all_confidence)
        all_ret_labels  = _cat_to_np(all_ret_labels)
        all_ret_preds   = _cat_to_np(all_ret_preds)
        all_dir_logits  = _cat_to_np(all_dir_logits)
        all_dir_probs   = _cat_to_np(all_dir_probs)
        all_regimes     = (
            _cat_to_np(all_regimes).astype(np.int64)
            if all_regimes else np.empty(0, dtype=np.int64)
        )
        # Optional binary edge-head diagnostics.
        if len(all_up_edge_logits) > 0 and len(all_down_edge_logits) > 0:
            _up_logits_np = _cat_to_np(all_up_edge_logits)
            _down_logits_np = _cat_to_np(all_down_edge_logits)
            _up_probs_np = 1.0 / (1.0 + np.exp(-np.clip(_up_logits_np, -40.0, 40.0)))
            _down_probs_np = 1.0 / (1.0 + np.exp(-np.clip(_down_logits_np, -40.0, 40.0)))
            _up_true = (all_dir_labels == 2).astype(np.int64)
            _down_true = (all_dir_labels == 0).astype(np.int64)

            def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
                try:
                    if np.unique(y_true).size < 2:
                        return 0.5
                    return float(roc_auc_score(y_true, y_score))
                except ValueError:
                    return 0.5

            _up_edge_auc = _binary_auc(_up_true, _up_probs_np)
            _down_edge_auc = _binary_auc(_down_true, _down_probs_np)
            _up_pred = _up_probs_np > 0.5
            _down_pred = _down_probs_np > 0.5
            _up_edge_rate = float(_up_pred.mean())
            _down_edge_rate = float(_down_pred.mean())
            _edge_dir_cov = float((_up_pred | _down_pred).mean())
            _edge_conflict = float((_up_pred & _down_pred).mean())
            _up_edge_prec = float(_up_true[_up_pred].mean()) if _up_pred.any() else 0.0
            _down_edge_prec = float(_down_true[_down_pred].mean()) if _down_pred.any() else 0.0
        else:
            _up_edge_auc = 0.5
            _down_edge_auc = 0.5
            _up_edge_rate = 0.0
            _down_edge_rate = 0.0
            _edge_dir_cov = 0.0
            _edge_conflict = 0.0
            _up_edge_prec = 0.0
            _down_edge_prec = 0.0
        # Sprint 5a — tradability bin diagnostics. Bin labels are derived from
        # |ret_label| vs ε_h (same scheme as MultiObjectiveLoss.LretBin) so
        # train-time loss and val-time metric agree. Metrics:
        #   RetBinAcc   : overall accuracy on 3-class bin classification
        #   RetBinF1    : macro F1 across {no_edge, marginal, strong_edge}
        #   BinDist     : predicted bin distribution (no/marginal/strong)
        #   StrongEdgePrec : precision on strong_edge class (key sprint signal)
        #   DirAccOnStrongEdge : direction accuracy on samples model predicts
        #                        as strong_edge — flags risk that strong-edge
        #                        prediction conflicts with direction prediction.
        if len(all_ret_bin_logits) > 0:
            _rbl_np      = _cat_to_np(all_ret_bin_logits)        # (N, 3)
            _bin_pred    = np.argmax(_rbl_np, axis=-1)           # (N,)
            _abs_ret     = np.abs(all_ret_labels)
            _eps_h_diag  = float(get_epsilon_h(self.horizon, _EPSILON_H_OVERRIDE))
            _bin_true    = np.zeros_like(all_ret_labels, dtype=np.int64)
            _bin_true[_abs_ret > _eps_h_diag]          = 1       # marginal_edge
            _bin_true[_abs_ret >= 2.0 * _eps_h_diag]   = 2       # strong_edge
            _ret_bin_acc = float((_bin_pred == _bin_true).mean())
            _ret_bin_f1  = float(compute_macro_f1(_bin_pred, _bin_true))
            _bin_dist    = np.array([
                float((_bin_pred == 0).mean()),
                float((_bin_pred == 1).mean()),
                float((_bin_pred == 2).mean()),
            ])
            _strong_pred_mask = (_bin_pred == 2)
            _n_strong_pred    = int(_strong_pred_mask.sum())
            if _n_strong_pred > 0:
                _strong_edge_prec = float(
                    (_bin_true[_strong_pred_mask] == 2).mean()
                )
                _dir_acc_on_strong = float(
                    (all_dir_preds[_strong_pred_mask] == all_dir_labels[_strong_pred_mask]).mean()
                )
            else:
                _strong_edge_prec  = 0.0
                _dir_acc_on_strong = 0.0
        else:
            _ret_bin_acc       = 0.0
            _ret_bin_f1        = 0.0
            _bin_dist          = np.zeros(3, dtype=np.float32)
            _strong_edge_prec  = 0.0
            _dir_acc_on_strong = 0.0
            _n_strong_pred     = 0
        attn_sum_mean   = float(_cat_to_np(all_attn_sums).mean())   if all_attn_sums   else 0.0
        attn_target_mean = float(_cat_to_np(all_attn_targets).mean()) if all_attn_targets else 0.0
        # R3 #F (extended): selector-learning diagnostics
        soft_gate_sum_mean = float(_cat_to_np(all_soft_gate_sums).mean()) if all_soft_gate_sums else 0.0
        soft_gate_max_mean = float(_cat_to_np(all_soft_gate_maxes).mean()) if all_soft_gate_maxes else 0.0
        # Sprint 1 FIX 7 (corrected) — Top-K mass diagnostics, no rescale.
        # ``raw_attn_sum_mean`` is the average Top-K share of softmax mass per
        # sample, baseline K_h / K_articles (uniform). ``raw_attn_max_mean`` is
        # the average peak weight on a single selected article, baseline 1 /
        # K_articles. Both rise as the selector becomes decisive; if they stay
        # at uniform-baseline across epochs the scorer is not learning.
        raw_attn_sum_mean = float(_cat_to_np(all_raw_attn_sums).mean())  if all_raw_attn_sums  else 0.0
        raw_attn_max_mean = float(_cat_to_np(all_raw_attn_maxes).mean()) if all_raw_attn_maxes else 0.0

        # ── Explanation metrics (PDF Section 4.4.2) ────────────────────────────
        all_masked_dir_probs_np    = _cat_to_np(all_masked_dir_probs)
        all_selected_only_probs_np = _cat_to_np(all_selected_only_probs)
        all_no_article_probs_np    = _cat_to_np(all_no_article_probs)

        # Deletion (Comprehensiveness) + true Insertion with no-article baseline
        del_ins = compute_deletion_insertion_score(
            all_dir_probs, all_masked_dir_probs_np, all_dir_labels,
            no_article_probs=all_no_article_probs_np,
            selected_only_probs=all_selected_only_probs_np,
        )

        # Sufficiency: quality drop when using ONLY selected articles
        sufficiency_score = compute_sufficiency(all_dir_probs, all_selected_only_probs_np, all_dir_labels)

        # Factor consistency: split flat predictions into equal-width windows
        if len(all_factor_preds) >= 2:
            fac_all_np = _cat_to_np(all_factor_preds)
            _window_sz = min(32, max(2, len(fac_all_np) // 8))
            if len(fac_all_np) >= 2 * _window_sz:
                _n_win = len(fac_all_np) // _window_sz
                fac_windows = [fac_all_np[i * _window_sz:(i + 1) * _window_sz] for i in range(_n_win)]
                factor_consistency_score = compute_factor_consistency(fac_windows)
            else:
                factor_consistency_score = 1.0  # too few samples to compare
        else:
            factor_consistency_score = 1.0  # no factor predictions (market-only mode)

        # Session 26 — temporal cal/sel split (anti-overfit).
        # `_tune_mode`: we're actively fitting T + τ/γ on this pass.
        # When True and cal_selection_split < 1.0, split val temporally:
        #   cal_sl = first cal_frac of samples → fit T and search τ/γ here
        #   sel_sl = remaining samples → held out, scored for early stopping
        # Preserves full-val arrays (for display/diagnostics) while driving
        # model_score from an honest out-of-sample slice. Must be temporal
        # (not random) because the val set is a time series and random split
        # would leak information across the decision boundary.
        _tune_mode = (temperature is None) and (tau is None) and (gamma is None)
        _cal_frac = float(getattr(self, "cal_selection_split", 1.0))
        _N_val = len(all_dir_labels)
        # Minimum-size guards:
        #   cal ≥ 40 samples so LBFGS T-fitting is stable
        #   sel ≥ 40 samples so sel-slice metrics are not pure noise
        # If either would be below 40, disable the split and fall back to full-val.
        _use_split = (
            _tune_mode
            and 0.0 < _cal_frac < 1.0
            and int(_N_val * _cal_frac) >= 40
            and _N_val - int(_N_val * _cal_frac) >= 40
        )
        _cal_end = int(_N_val * _cal_frac) if _use_split else _N_val
        cal_sl = slice(0, _cal_end)
        sel_sl = slice(_cal_end, _N_val)

        if temperature is None:
            # Fit T on the calibration slice only when splitting; otherwise
            # fall back to full-val fit (covers small val sets + legacy mode).
            _fit_logits = all_dir_logits[cal_sl] if _use_split else all_dir_logits
            _fit_labels = all_dir_labels[cal_sl] if _use_split else all_dir_labels
            temp_fit = fit_temperature_scaling(_fit_logits, _fit_labels)
            temperature = float(temp_fit["temperature"])
            temperature_nll = float(temp_fit["temperature_nll"])
            # Session 22 Fix 12: propagate the Fix 2 suspect flag so reviewers
            # can see when post-hoc T is masking in-training Lcal failures.
            # fit_temperature_scaling emits a RuntimeWarning at fit time, but
            # the flag in the returned dict must be forwarded into val_losses
            # and then into training_metrics.json — otherwise the warning
            # would only show up in stderr and be lost after the run. The
            # unclamped T is kept too, so post-mortem analysis can tell
            # whether the clamp saved the run or was vacuous.
            temperature_suspect = bool(temp_fit.get("temperature_suspect", False))
            temperature_unclamped = float(temp_fit.get("temperature_unclamped", temperature))
            temperature_source = (
                "validation_temperature_scaling_cal_split"
                if _use_split else "validation_temperature_scaling"
            )
            # Session 26 drift check: also fit T on sel slice when splitting.
            # If the two Ts differ by > 25 %, val is non-stationary and the
            # frozen T may be inappropriate for test. Emit a suspect flag so
            # training_metrics.json captures this signal.
            if _use_split:
                try:
                    _sel_fit = fit_temperature_scaling(
                        all_dir_logits[sel_sl], all_dir_labels[sel_sl]
                    )
                    _sel_T = float(_sel_fit["temperature"])
                    _drift = abs(_sel_T - temperature) / max(temperature, 1e-3)
                    if _drift > 0.25:
                        temperature_suspect = True
                        warnings.warn(
                            f"Temperature drift across val ({temperature:.3f} cal vs "
                            f"{_sel_T:.3f} sel, {_drift*100:.0f}% gap) — val may be "
                            f"non-stationary; frozen T may be miscalibrated on test.",
                            RuntimeWarning, stacklevel=2,
                        )
                except Exception:
                    pass  # drift check is diagnostic only — fitting failure shouldn't crash val
        else:
            temperature = float(temperature)
            temperature_source = "frozen_validation_temperature"
            # Frozen case: no new fit is performed, so any suspect flag from
            # the original val-time fit cannot be recomputed here; caller may
            # inject it via kwarg in the future. Default to False for now.
            temperature_suspect = False
            temperature_unclamped = temperature
            frozen_probs = apply_temperature_to_logits(all_dir_logits, temperature)
            true_probs = np.clip(
                frozen_probs[np.arange(len(all_dir_labels)), all_dir_labels],
                1e-8,
                1.0,
            )
            temperature_nll = float(-np.log(true_probs).mean())

        calibrated_dir_probs = apply_temperature_to_logits(all_dir_logits, temperature)
        calibrated_preds = calibrated_dir_probs.argmax(axis=1)
        max_prob_all = calibrated_dir_probs.max(axis=-1)
        policy_confidence_all = self._policy_confidence(
            all_confidence, calibrated_dir_probs
        )

        ret_mae = np.mean(np.abs(all_ret_preds - all_ret_labels))
        ret_rmse = np.sqrt(np.mean((all_ret_preds - all_ret_labels) ** 2))
        ret_correlation = np.corrcoef(all_ret_preds, all_ret_labels)[0, 1] if len(all_ret_preds) > 1 else 0.0

        macro_f1 = compute_macro_f1(calibrated_preds, all_dir_labels)
        mcc = compute_mcc(calibrated_preds, all_dir_labels)
        ece = compute_ece(max_prob_all, calibrated_preds, all_dir_labels)
        # Sprint 6 Phase 7 — ECE_trade diagnostic. Global ECE measures
        # calibration over ALL predictions including NEUTRAL. Trading task only
        # cares about calibration on D/U predictions (NEUTRAL = abstain → not
        # traded → cost-gated). If ECE_global is low but ECE_trade is high,
        # calibration is "đẹp ở vùng không trade" — exactly the failure mode
        # họ Phase 7 plan suspects. Computed only when ≥10 D|U predictions
        # exist; else fallback to global ECE so logging schema stable.
        _trade_pred_mask = (calibrated_preds == 0) | (calibrated_preds == 2)
        if _trade_pred_mask.sum() >= 10:
            ece_trade = compute_ece(
                max_prob_all[_trade_pred_mask],
                calibrated_preds[_trade_pred_mask],
                all_dir_labels[_trade_pred_mask],
            )
        else:
            ece_trade = ece  # fallback when too few D|U predictions
        brier = compute_brier_score(max_prob_all, calibrated_preds, all_dir_labels)
        # Session 24: per-decile ECE breakdown (paper Section 4.4.3). Reveals
        # WHERE in the confidence spectrum the model is miscalibrated — a
        # single ECE scalar can hide systematic issues (e.g., high-conf
        # predictions are over-confident while low-conf are under-confident
        # but offset in aggregate). ece_worst_decile_gap becomes the quick
        # red-flag signal for reviewers.
        ece_decile_stats = compute_ece_per_decile(
            max_prob_all, calibrated_preds, all_dir_labels, n_deciles=10,
        )

        # Policy-objective accounting. Keep defaults explicit so frozen-policy
        # validation and grid-search validation serialize the same diagnostic
        # schema into training_metrics.json.
        policy_objective = None
        coverage_penalty = None
        under_trade_cov = 0.0
        action_bonus_policy = 0.0
        neutsh_penalty_policy = 0.0
        action_quality_penalty_policy = 0.0
        cost_volume_penalty = 0.0
        imbalance_penalty = 0.0
        cost_volume_penalty_diag = 0.0
        imbalance_penalty_diag = 0.0
        directional_imbalance = 0.0

        if tau is not None and gamma is not None:
            best_tau = float(tau)
            best_gamma = float(gamma)
            alert_prec, alert_cov, sel_risk = compute_alert_precision(
                policy_confidence_all, calibrated_preds, all_dir_labels,
                dir_probs=calibrated_dir_probs, tau=best_tau, gamma=best_gamma,
            )
            backtest_metrics = mini_backtest(
                policy_confidence_all, calibrated_preds, all_ret_labels,
                tau=best_tau, gamma=best_gamma, dir_probs=calibrated_dir_probs,
            )
            tau_seed = float(np.percentile(policy_confidence_all, self.tau_percentile))
            gamma_seed = float(np.percentile(max_prob_all, self.gamma_percentile))
            policy_source = "frozen_validation_policy"
        else:
            # Session 26 — grid search τ/γ on CAL slice only. Without this
            # split, the same val set that picks thresholds also scores the
            # model, biasing both toward val-noise. Cal-slice F1/MCC/ECE
            # drive the policy objective so the search is self-consistent.
            if _use_split:
                cal_f1  = compute_macro_f1(calibrated_preds[cal_sl], all_dir_labels[cal_sl])
                cal_mcc = compute_mcc(calibrated_preds[cal_sl], all_dir_labels[cal_sl])
                cal_ece = compute_ece(max_prob_all[cal_sl], calibrated_preds[cal_sl], all_dir_labels[cal_sl])
                policy = search_alert_policy(
                    policy_confidence_all[cal_sl], calibrated_preds[cal_sl],
                    all_dir_labels[cal_sl], all_ret_labels[cal_sl],
                    calibrated_dir_probs[cal_sl],
                    cal_f1, cal_mcc, cal_ece,
                    tau_percentile=self.tau_percentile,
                    gamma_percentile=self.gamma_percentile,
                    target_coverage=self.loss_fn.coverage_target,
                    policy_method=self.policy_method,
                )
            else:
                policy = search_alert_policy(
                    policy_confidence_all, calibrated_preds, all_dir_labels, all_ret_labels, calibrated_dir_probs,
                    macro_f1, mcc, ece,
                    tau_percentile=self.tau_percentile,
                    gamma_percentile=self.gamma_percentile,
                    target_coverage=self.loss_fn.coverage_target,
                    policy_method=self.policy_method,
                )
            best_tau = float(policy["tau"])
            best_gamma = float(policy["gamma"])
            # Session 27 fix #10/#12/#13: EMA smooth policy across epochs.
            # Without this, coverage oscillate 0.003 → 0.979 across consecutive
            # epochs because grid search picks different operating points each
            # time. EMA stabilizes both Sharpe and Alert Precision.
            if self._prev_tau is not None and self._prev_gamma is not None:
                best_tau = self._policy_ema_alpha * best_tau + (1 - self._policy_ema_alpha) * self._prev_tau
                best_gamma = self._policy_ema_alpha * best_gamma + (1 - self._policy_ema_alpha) * self._prev_gamma
            self._prev_tau = best_tau
            self._prev_gamma = best_gamma
            # Session 26 — re-evaluate frozen policy on SEL slice (held out
            # from policy fitting). These numbers are the honest reported
            # metrics that drive early stopping. Legacy path (no split)
            # uses the cal-slice policy dict directly.
            if _use_split:
                _sel_alert_prec, _sel_alert_cov, _sel_sel_risk = compute_alert_precision(
                    policy_confidence_all[sel_sl], calibrated_preds[sel_sl], all_dir_labels[sel_sl],
                    dir_probs=calibrated_dir_probs[sel_sl], tau=best_tau, gamma=best_gamma,
                )
                _sel_backtest = mini_backtest(
                    policy_confidence_all[sel_sl], calibrated_preds[sel_sl], all_ret_labels[sel_sl],
                    tau=best_tau, gamma=best_gamma, dir_probs=calibrated_dir_probs[sel_sl],
                )
                alert_prec = float(_sel_alert_prec)
                alert_cov = float(_sel_alert_cov)
                sel_risk = float(_sel_sel_risk)
                backtest_metrics = {
                    "alert_sharpe":   float(_sel_backtest["alert_sharpe"]),
                    "alert_sortino":  float(_sel_backtest["alert_sortino"]),
                    "alert_calmar":   float(_sel_backtest["alert_calmar"]),
                    "alert_max_dd":   float(_sel_backtest["alert_max_dd"]),
                    "alert_hit_rate": float(_sel_backtest["alert_hit_rate"]),
                    "alert_coverage": float(_sel_backtest["alert_coverage"]),
                    "pnl":            float(_sel_backtest.get("pnl", 0.0)),
                    # Sprint 1 Patch 1 — propagate position-only diagnostics
                    # so the SEL re-eval branch carries the same telemetry as
                    # the cal-search policy dict. Without these keys the
                    # checkpoint gate (trade_coverage, position_pnl, suppressed
                    # flag) cannot be read from training_metrics.json.
                    "trade_coverage":          float(_sel_backtest.get("trade_coverage", 0.0)),
                    "neutral_share_of_alerts": float(_sel_backtest.get(
                        "neutral_share_of_alerts",
                        _sel_backtest.get("neutral_alert_rate", 0.0),
                    )),
                    "n_position_alerts":       int(_sel_backtest.get("n_position_alerts", 0)),
                    "position_pnl":            float(_sel_backtest.get("position_pnl", 0.0)),
                    # Position-only win rate kept separate from the legacy
                    # confidence-weighted ``hit_rate`` computed below in
                    # val_losses; they answer different questions.
                    "position_hit_rate":       float(_sel_backtest.get("alert_hit_rate", 0.0)),
                    "alert_sharpe_suppressed": bool(_sel_backtest.get("alert_sharpe_suppressed", False)),
                }
                policy_source = "validation_grid_cal_sel_split"
            else:
                alert_prec = float(policy["alert_precision"])
                alert_cov = float(policy["alert_coverage"])
                sel_risk = float(policy["selective_risk"])
                backtest_metrics = {
                    "alert_sharpe":   float(policy["alert_sharpe"]),
                    "alert_sortino":  float(policy["alert_sortino"]),
                    "alert_calmar":   float(policy["alert_calmar"]),
                    "alert_max_dd":   float(policy["alert_max_dd"]),
                    "alert_hit_rate": float(policy["hit_rate"]),
                    "alert_coverage": float(policy["alert_coverage"]),
                    "pnl":            float(policy.get("pnl", 0.0)),
                    "trade_coverage":          float(policy.get("trade_coverage", 0.0)),
                    "neutral_share_of_alerts": float(policy.get(
                        "neutral_share_of_alerts",
                        policy.get("neutral_alert_rate", 0.0),
                    )),
                    "n_position_alerts":       int(policy.get("n_position_alerts", 0)),
                    "position_pnl":            float(policy.get("position_pnl", 0.0)),
                    "position_hit_rate":       float(policy.get(
                        "position_hit_rate", policy.get("hit_rate", 0.0))),
                    "alert_sharpe_suppressed": bool(policy.get("alert_sharpe_suppressed", False)),
                }
                policy_source = str(policy["policy_source"])
            tau_seed = float(policy["tau_seed"])
            gamma_seed = float(policy["gamma_seed"])
            policy_objective = float(policy["policy_objective"])
            coverage_penalty = float(policy["coverage_penalty"])
            under_trade_cov = float(policy.get("under_trade_cov", 0.0))
            action_bonus_policy = float(policy.get("action_bonus", 0.0))
            neutsh_penalty_policy = float(policy.get("neutsh_penalty_policy", 0.0))
            action_quality_penalty_policy = float(policy.get(
                "action_quality_penalty_policy", 0.0
            ))
            cost_volume_penalty = float(policy.get("cost_volume_penalty", 0.0))
            imbalance_penalty = float(policy.get("imbalance_penalty", 0.0))
            cost_volume_penalty_diag = float(policy.get("cost_volume_penalty_diag", 0.0))
            imbalance_penalty_diag = float(policy.get("imbalance_penalty_diag", 0.0))
            directional_imbalance = float(policy.get("directional_imbalance", 0.0))

        # Session 26 — report sel-slice metrics for early stopping. These are
        # the numbers the model is actually being judged on. Full-val variants
        # are kept under *_full_val keys for diagnostics / reviewer audits.
        if _use_split and _tune_mode:
            sel_macro_f1 = compute_macro_f1(calibrated_preds[sel_sl], all_dir_labels[sel_sl])
            sel_mcc = compute_mcc(calibrated_preds[sel_sl], all_dir_labels[sel_sl])
            sel_ece = compute_ece(max_prob_all[sel_sl], calibrated_preds[sel_sl], all_dir_labels[sel_sl])
            # Preserve full-val values under a diagnostic suffix before overwriting
            macro_f1_full_val = macro_f1
            mcc_full_val = mcc
            ece_full_val = ece
            macro_f1 = sel_macro_f1
            mcc = sel_mcc
            ece = sel_ece
        else:
            macro_f1_full_val = macro_f1
            mcc_full_val = mcc
            ece_full_val = ece

        # Sprint 1.5 — prediction-distribution diagnostics. Logs prove the
        # NEUTRAL collapse (or its absence) directly instead of relying on
        # F1/MCC inference. Three fields:
        #   pred_dist_*           : share of all preds that landed on each class
        #   alert_pred_dist_*     : same, restricted to the alert region
        #                           ((conf >= τ) & (max_prob >= γ))
        #   recall_down / _up     : per-class recall for the two directional
        #                           classes; NEUTRAL recall is the residual
        # Computed on the SEL slice when cal/sel split is active so the numbers
        # match the slice the early-stopping score is built from. When no split
        # is active they fall back to full validation.
        if _use_split and _tune_mode:
            _diag_preds  = calibrated_preds[sel_sl]
            _diag_labels = all_dir_labels[sel_sl]
            _diag_conf   = all_confidence[sel_sl]
            _diag_policy_conf = policy_confidence_all[sel_sl]
            _diag_probs  = calibrated_dir_probs[sel_sl]
            _diag_rets   = all_ret_labels[sel_sl]
            _diag_regimes = (
                all_regimes[sel_sl]
                if all_regimes.shape[0] == all_dir_labels.shape[0]
                else np.full(_diag_preds.shape[0], -1, dtype=np.int64)
            )
        else:
            _diag_preds  = calibrated_preds
            _diag_labels = all_dir_labels
            _diag_conf   = all_confidence
            _diag_policy_conf = policy_confidence_all
            _diag_probs  = calibrated_dir_probs
            _diag_rets   = all_ret_labels
            _diag_regimes = (
                all_regimes
                if all_regimes.shape[0] == all_dir_labels.shape[0]
                else np.full(_diag_preds.shape[0], -1, dtype=np.int64)
            )
        _n_diag = max(1, int(_diag_preds.shape[0]))
        pred_dist_down    = float((_diag_preds == 0).mean())
        pred_dist_neutral = float((_diag_preds == 1).mean())
        pred_dist_up      = float((_diag_preds == 2).mean())
        _diag_max_prob = _diag_probs.max(axis=-1)
        _diag_alert_mask = (_diag_policy_conf >= float(best_tau)) & (_diag_max_prob >= float(best_gamma))
        if _diag_alert_mask.any():
            _alert_preds = _diag_preds[_diag_alert_mask]
            alert_pred_dist_down    = float((_alert_preds == 0).mean())
            alert_pred_dist_neutral = float((_alert_preds == 1).mean())
            alert_pred_dist_up      = float((_alert_preds == 2).mean())
        else:
            alert_pred_dist_down = alert_pred_dist_neutral = alert_pred_dist_up = 0.0

        def _per_class_recall(cls: int) -> float:
            cls_mask = _diag_labels == cls
            n_cls = int(cls_mask.sum())
            if n_cls == 0:
                return 0.0
            return float((_diag_preds[cls_mask] == cls).sum() / n_cls)
        recall_down    = _per_class_recall(0)
        recall_neutral = _per_class_recall(1)
        recall_up      = _per_class_recall(2)

        # Sprint 2.1-D — confidence-by-class diagnostic. Reveals why the alert
        # region is NEUTRAL-heavy even when the head emits balanced predictions.
        # Hypothesis: conf_head is biased toward high confidence on NEUTRAL
        # outputs and low confidence on UP/DOWN, so τ-gating disproportionately
        # cuts directional alerts. Three measurements per predicted class:
        #   conf_pred_*       : mean of model confidence (sigmoid scalar) over
        #                       samples whose argmax-pred is that class
        #   maxprob_pred_*    : mean of softmax max-prob over the same samples
        #   conf_alert_*      : mean of confidence within the alert region
        #                       conditioned on predicted class
        # If conf_pred_neutral >> conf_pred_up / conf_pred_down → conf_head is
        # the suspect. If max_prob is roughly equal but conf differs → conf_head
        # decoupled from softmax entropy, which is the failure mode Lcal aims
        # to fix.
        def _mean_or_zero(arr) -> float:
            arr = np.asarray(arr)
            return float(arr.mean()) if arr.size > 0 else 0.0

        def _mean_conf_for_class(cls: int) -> tuple:
            cls_mask = _diag_preds == cls
            return (
                _mean_or_zero(_diag_conf[cls_mask]),
                _mean_or_zero(_diag_max_prob[cls_mask]),
            )

        conf_pred_down,    maxprob_pred_down    = _mean_conf_for_class(0)
        conf_pred_neutral, maxprob_pred_neutral = _mean_conf_for_class(1)
        conf_pred_up,      maxprob_pred_up      = _mean_conf_for_class(2)

        def _mean_conf_alert_for_class(cls: int) -> float:
            cls_alert = _diag_alert_mask & (_diag_preds == cls)
            return _mean_or_zero(_diag_conf[cls_alert])

        conf_alert_down    = _mean_conf_alert_for_class(0)
        conf_alert_neutral = _mean_conf_alert_for_class(1)
        conf_alert_up      = _mean_conf_alert_for_class(2)

        # Sprint 6 Phase 7 — gate-source slack diagnostic for NEUTRAL alerts.
        # Eq.26 alert: (conf >= τ) AND (max_prob >= γ). If alert region is
        # NEUTRAL-heavy, identify WHICH gate is the dominant cause:
        #   conf_neutral_alert_slack    = mean(conf - τ | NEUTRAL alert)
        #   maxprob_neutral_alert_slack = mean(max_prob - γ | NEUTRAL alert)
        # Larger slack = gate fires comfortably (redundant). Smaller slack
        # = gate barely passes (this gate is the bottleneck).
        # Interpretation:
        #   conf_slack >> maxprob_slack  → conf gate redundant, max_prob bottleneck
        #                                  → direction head NEUTRAL bias dominant
        #                                  → fix at direction head (focal/CW/LS)
        #   maxprob_slack >> conf_slack  → max_prob redundant, conf gate bottleneck
        #                                  → confidence scalar broken
        #                                  → fix at conf_head (Lcal / action-aware)
        #   both small (~0)              → gate threshold tight, marginal alerts
        #                                  → policy threshold tuning helps
        _neutral_alert_mask = _diag_alert_mask & (_diag_preds == 1)
        _n_neutral_alert = int(_neutral_alert_mask.sum())
        if _n_neutral_alert > 0:
            # Use best_tau / best_gamma (policy search result), NOT the function
            # arg tau/gamma which can be None. Same variables that defined
            # _diag_alert_mask above so slack is consistent with the gate.
            conf_neutral_alert_slack    = _mean_or_zero(_diag_policy_conf[_neutral_alert_mask] - float(best_tau))
            maxprob_neutral_alert_slack = _mean_or_zero(_diag_max_prob[_neutral_alert_mask] - float(best_gamma))
        else:
            conf_neutral_alert_slack    = 0.0
            maxprob_neutral_alert_slack = 0.0

        # Sprint 4 A1 (Phase 7) — position_confidence diagnostic.
        # confidence head emits a single scalar c ∈ [0, 1] regardless of which
        # class the model predicts. paper Eq.26 alerts on (c >= τ AND max_p >=
        # γ), so a NEUTRAL prediction with high c can still trigger an alert
        # — surfacing as NEUTRAL-heavy alert region even when direction probs
        # are well-spread. position_confidence = c * (1 - p_neutral) suppresses
        # confidence on NEUTRAL-leaning predictions: a sample where the model
        # is "confidently neutral" (c high, p_neutral high) gets near-zero
        # position_confidence, while "confidently directional" (c high,
        # p_neutral low) keeps near-c. This is a DIAGNOSTIC ONLY — Eq.26 alert
        # path remains unchanged. If pos_conf_alert_down/up are non-trivial
        # while conf_alert_down/up stay near zero, the confidence head IS
        # learning directional signal but the scalar form hides it.
        # Index 1 = NEUTRAL (verified via pred_dist_neutral = (_diag_preds == 1)).
        _p_neutral = _diag_probs[:, 1]
        _diag_position_conf = _diag_conf * (1.0 - _p_neutral)

        def _mean_pos_conf_pred(cls: int) -> float:
            cls_mask = _diag_preds == cls
            return _mean_or_zero(_diag_position_conf[cls_mask])

        def _mean_pos_conf_alert(cls: int) -> float:
            cls_alert = _diag_alert_mask & (_diag_preds == cls)
            return _mean_or_zero(_diag_position_conf[cls_alert])

        pos_conf_pred_down    = _mean_pos_conf_pred(0)
        pos_conf_pred_neutral = _mean_pos_conf_pred(1)
        pos_conf_pred_up      = _mean_pos_conf_pred(2)
        pos_conf_alert_down    = _mean_pos_conf_alert(0)
        pos_conf_alert_neutral = _mean_pos_conf_alert(1)
        pos_conf_alert_up      = _mean_pos_conf_alert(2)

        # Sprint 10 - regime-stratified validation diagnostics. Regime is a
        # dataset-emitted stratification variable (0=SIDEWAYS, 1=BULL,
        # 2=BEAR, 3=VOLATILE); it does not feed the model. Use the same eval
        # slice as [Metrics]/[PredDist] so these lines explain the reported
        # score rather than the policy-fitting cal slice.
        def _class_dist(preds: np.ndarray) -> tuple[float, float, float]:
            if preds.size == 0:
                return 0.0, 0.0, 0.0
            return (
                float((preds == 0).mean()),
                float((preds == 1).mean()),
                float((preds == 2).mean()),
            )

        def _raw_position_stats(
            preds: np.ndarray,
            returns: np.ndarray,
            alert_mask: np.ndarray,
        ) -> tuple[int, float, float]:
            pos_mask = alert_mask & (preds != 1)
            n_pos = int(pos_mask.sum())
            if n_pos == 0:
                return 0, 0.0, 0.0
            signs = np.where(preds[pos_mask] == 2, 1.0, -1.0)
            # Match mini_backtest defaults: one round-trip transaction cost.
            net_returns = signs * returns[pos_mask] - 0.001
            return n_pos, float((net_returns > 0.0).mean()), float(net_returns.sum())

        regime_metrics: Dict[str, float] = {}
        _regime_specs = (
            (0, "sideways"),
            (1, "bull"),
            (2, "bear"),
            (3, "volatile"),
        )
        _diag_max_prob_regime = _diag_probs.max(axis=-1)
        _diag_alert_mask_regime = (
            (_diag_policy_conf >= float(best_tau))
            & (_diag_max_prob_regime >= float(best_gamma))
        )
        for _rid, _rkey in _regime_specs:
            _rmask = _diag_regimes == _rid
            _prefix = f"regime_{_rkey}_"
            _n_regime = int(_rmask.sum())
            regime_metrics[f"{_prefix}n"] = _n_regime
            if _n_regime == 0:
                regime_metrics.update({
                    f"{_prefix}f1": 0.0,
                    f"{_prefix}mcc": 0.0,
                    f"{_prefix}pred_down": 0.0,
                    f"{_prefix}pred_neutral": 0.0,
                    f"{_prefix}pred_up": 0.0,
                    f"{_prefix}alert_down": 0.0,
                    f"{_prefix}alert_neutral": 0.0,
                    f"{_prefix}alert_up": 0.0,
                    f"{_prefix}alert_cov": 0.0,
                    f"{_prefix}trade_cov": 0.0,
                    f"{_prefix}neutral_share": 0.0,
                    f"{_prefix}n_pos": 0,
                    f"{_prefix}pos_hit": 0.0,
                    f"{_prefix}pos_pnl": 0.0,
                })
                continue

            _rp = _diag_preds[_rmask]
            _rl = _diag_labels[_rmask]
            _rr = _diag_rets[_rmask]
            _ra = _diag_alert_mask_regime[_rmask]
            _pred_d, _pred_n, _pred_u = _class_dist(_rp)
            if _ra.any():
                _alert_d, _alert_n, _alert_u = _class_dist(_rp[_ra])
                _neutral_share = float((_rp[_ra] == 1).mean())
            else:
                _alert_d = _alert_n = _alert_u = 0.0
                _neutral_share = 0.0
            _n_pos, _pos_hit, _pos_pnl = _raw_position_stats(_rp, _rr, _ra)
            regime_metrics.update({
                f"{_prefix}f1": float(compute_macro_f1(_rp, _rl)) if _n_regime > 1 else 0.0,
                f"{_prefix}mcc": float(compute_mcc(_rp, _rl)) if _n_regime > 1 else 0.0,
                f"{_prefix}pred_down": _pred_d,
                f"{_prefix}pred_neutral": _pred_n,
                f"{_prefix}pred_up": _pred_u,
                f"{_prefix}alert_down": _alert_d,
                f"{_prefix}alert_neutral": _alert_n,
                f"{_prefix}alert_up": _alert_u,
                f"{_prefix}alert_cov": float(_ra.mean()),
                f"{_prefix}trade_cov": float(_n_pos / max(_n_regime, 1)),
                f"{_prefix}neutral_share": _neutral_share,
                f"{_prefix}n_pos": int(_n_pos),
                f"{_prefix}pos_hit": _pos_hit,
                f"{_prefix}pos_pnl": _pos_pnl,
            })

        alert_sharpe = backtest_metrics["alert_sharpe"]
        # Sprint 1 Patch 2 — when SEL re-eval suppresses Sharpe (too few position
        # alerts to be statistically reliable), feeding 0.0 into the selection
        # score lets a checkpoint win simply because its noise was clipped, not
        # because it had real edge. Mirror search_alert_policy's treatment:
        # treat suppressed Sharpe as missing rather than as a real zero, so
        # macro_f1 / MCC / ECE drive selection until enough trades materialise.
        sel_sharpe_suppressed = bool(backtest_metrics.get("alert_sharpe_suppressed", False))
        sharpe_for_selection = 0.0 if sel_sharpe_suppressed else alert_sharpe
        # Sprint 12 — pass position_pnl so the score can include a direct
        # economic-gate term when SELECTION_WEIGHT_PNL > 0 (YAML
        # ``selection_weights.pnl``). When weight=0 (default), pnl arg is
        # accepted but ignored — score remains bit-stable with legacy 4-metric.
        _sel_pnl = backtest_metrics.get("position_pnl", backtest_metrics.get("pnl", 0.0))
        model_score = compute_model_selection_score(
            macro_f1, mcc, ece, sharpe_for_selection,
            alert_coverage=backtest_metrics["alert_coverage"],
            position_pnl=_sel_pnl,
        )
        if sel_sharpe_suppressed:
            # Sprint 3.3-C — strengthened from 0.10 to 0.20. A/B Run 1 fold 2
            # showed ep 10 (TradeCov=0, Sharpe suppressed, F1=0.397) winning
            # [BEST] over ep 2 (TradeCov=0.443, 699 PosAlerts, PosHit=0.396,
            # Sharpe=-0.174). Old 0.10 penalty was too weak: ep 10's higher
            # F1+ECE outweighed it. A 0.20 penalty makes a suppressed-Sharpe
            # checkpoint roughly cost a full F1 step (≈ 0.10 × 2 in
            # compute_model_selection_score's F1 weight), forcing selection
            # toward checkpoints that actually trade.
            model_score = float(model_score) - 0.20

        # Sprint 2.1-C — collapse-aware selection penalties. Run-1 of fold-1
        # showed [BEST] at ep 16 with PredDist (D/N/U) = 0.00/0.02/0.98 and
        # Sharpe = -0.097 — a policy that took 49 all-UP positions at a 40.8 %
        # hit rate but lost money after costs. A directional head with one
        # class effectively dead (recall_down = 0 in that run) was treated by
        # the selection score as a viable checkpoint. These penalties make the
        # selection score reject such collapses up front.
        # Threshold 0.05 on per-class recall picks up the all-NEUTRAL collapse
        # AND the all-UP collapse (both dropped recall on the abandoned classes
        # to ≈ 0). Threshold 0.90 on max(pred_dist) catches "always predict X"
        # checkpoints regardless of which class is dominant.
        recall_collapse_penalty = 0.0
        if recall_down < 0.05 or recall_up < 0.05:
            recall_collapse_penalty = 0.10
        max_pred_dist = max(pred_dist_down, pred_dist_neutral, pred_dist_up)
        dist_collapse_penalty = 0.10 if max_pred_dist > 0.90 else 0.0
        model_score = float(model_score) - recall_collapse_penalty - dist_collapse_penalty

        # Sprint 3.3-C — action-aware adjustments. The pre-3.3-C selection score
        # leaned heavily on F1/ECE (paper §4.5.1 model-selection metrics) but
        # rewarded position-only Sharpe / hit-rate only weakly through utility
        # bonuses inside compute_model_selection_score and a clipped Sharpe
        # term. Result: a checkpoint that classifies well but never enters
        # positions could still beat one that trades with a 40 %+ hit rate.
        # action_reward gives a real bump to checkpoints that (a) actually
        # enter positions on validation (TradeCov ≥ 0.05), and (b) pick the
        # right side often enough to recoup costs (PosHit thresholds at 0.40
        # and 0.50). neutsh_penalty discourages NEUTRAL-spam — checkpoints
        # whose alert region is dominated by NEUTRAL alerts (which neither
        # trade nor lose money but inflate alert_coverage statistics).
        # Sprint 3.3-D — quality-aware action reward. The pure 3.3-C version
        # rewarded any checkpoint that traded ≥ 5 % of validation, regardless
        # of whether those trades made money. A/B Run 1 fold 2 ep 2 won
        # [BEST] with TradeCov=0.443 and PosHit=0.396 even though
        # PosPnL=-0.7416 and Sharpe=-0.174 — i.e., the model traded a lot
        # but lost money. Sprint 3.3-D gates the PosHit-based tier bonuses on
        # a non-negative-quality flag (PnL ≥ 0 OR Sharpe ≥ 0) and adds an
        # action_quality_penalty that scales with negative Sharpe. The
        # baseline +0.05 for "model trades at all" stays — that signal is
        # still useful as eligibility — but trade volume alone no longer
        # earns the full 0.15 reward.
        _trade_cov = float(backtest_metrics.get("trade_coverage", 0.0))
        _pos_pnl = float(backtest_metrics.get("position_pnl", 0.0))
        _alert_sharpe = float(backtest_metrics.get("alert_sharpe", 0.0))
        _quality_ok = (_pos_pnl >= 0.0) or (_alert_sharpe >= 0.0)
        action_reward = 0.0
        if _trade_cov >= 0.05:
            action_reward += 0.05  # baseline: model actually trades
            _pos_hit = float(backtest_metrics.get("position_hit_rate", 0.0))
            # Tier bonuses gated on quality_ok: only reward hit-rate when
            # the trades are not net-negative on PnL/Sharpe.
            if _quality_ok and _pos_hit >= 0.40:
                action_reward += 0.05
            if _quality_ok and _pos_hit >= 0.50:
                action_reward += 0.05
        # Action-quality penalty for checkpoints that trade but lose money.
        # Linear ramp 0 → 0.10 across (0, 0.33] of |Sharpe|; calibrated so a
        # Sharpe of −0.17 (ep 2 fold 2 in A/B Run 1) costs ≈ 0.05, and a
        # Sharpe of −0.33 hits the cap. Together with the strengthened
        # suppression_penalty this keeps the ranking honest: trade-and-lose
        # is a worse outcome than don't-trade-at-all on the score.
        action_quality_penalty = 0.0
        if _trade_cov >= 0.05 and _alert_sharpe < 0.0:
            action_quality_penalty = min(0.10, abs(_alert_sharpe) * 0.30)
        neutsh_penalty = 0.0
        _neutsh = float(backtest_metrics.get("neutral_share_of_alerts", 0.0))
        if _neutsh > 0.70:
            # Linear ramp 0 → 0.15 across (0.70, 1.00].
            neutsh_penalty = 0.15 * min(1.0, (_neutsh - 0.70) / 0.30)
        model_score = (
            float(model_score)
            + action_reward
            - neutsh_penalty
            - action_quality_penalty
        )

        # Sprint 3.3-D — deployable_checkpoint hard gate. Independent of the
        # composite model_score (which already folds in penalties and is
        # therefore circular if used as the deploy gate). Uses observable
        # trading metrics directly so a checkpoint with strong actual trades
        # but mediocre F1/ECE is still flagged deployable, while a
        # high-F1 checkpoint that never trades or loses money is not.
        # Surface deployable_reason as a short string so the case-study
        # exporter / dashboard can show *why* a checkpoint failed the gate
        # rather than just the boolean.
        deployable_reasons: list[str] = []
        if _trade_cov < 0.05:
            deployable_reasons.append("trade_cov<0.05")
        if sel_sharpe_suppressed:
            deployable_reasons.append("sharpe_suppressed")
        if _pos_pnl < 0.0 and _alert_sharpe < 0.0:
            deployable_reasons.append("pnl_and_sharpe_negative")
        deployable_checkpoint = len(deployable_reasons) == 0
        deployable_reason = ",".join(deployable_reasons) if deployable_reasons else "ok"
        auc = compute_auc(calibrated_dir_probs, all_dir_labels)
        aurc = compute_coverage_risk_auc(max_prob_all, calibrated_preds, all_dir_labels)
        hit_rate = compute_hit_rate(policy_confidence_all, all_ret_labels, tau=best_tau)

        val_losses.update({
            "macro_f1": macro_f1,
            "mcc": mcc,
            "ece": ece,
            # Session 24: per-decile calibration breakdown (10 deciles × 4 fields
            # + worst-decile summary). Flatten into val_losses so downstream
            # JSON logging (training_metrics.json) captures them. Keys take the
            # form ece_decile_{i}_{count,conf_mean,acc,gap}.
            **ece_decile_stats,
            "brier": brier,
            # "accuracy" alias matches PDF Tables 3/5 column naming (dir_acc = accuracy)
            "accuracy": float(val_losses["dir_acc"]),
            "tau": best_tau,
            "gamma": best_gamma,
            "policy_confidence_source": self.policy_confidence_source,
            "temperature": temperature,
            "temperature_nll": temperature_nll,
            "temperature_source": temperature_source,
            # Session 22 Fix 12: surfaces the Fix 2 suspect flag to the outer
            # caller so training_metrics.json records when post-hoc T is
            # compensating for Lcal failure. Consumers (thesis write-up,
            # dashboards) should inspect temperature_suspect=True as a hint
            # that calibration is unreliable and in-training L_cal needs
            # stronger regularisation (e.g., raise λ5 or tighten entropy
            # anchor target).
            "temperature_suspect": temperature_suspect,
            "temperature_unclamped": temperature_unclamped,
            "tau_seed": tau_seed,
            "gamma_seed": gamma_seed,
            "policy_source": policy_source,
            "policy_objective": policy_objective,
            "coverage_penalty": coverage_penalty,
            # Policy-search accounting. The *_diag fields are Sprint 12 rollback
            # diagnostics: visible in logs/artifacts, intentionally not applied
            # to canonical raw policy search.
            "under_trade_cov": float(under_trade_cov),
            "action_bonus_policy": float(action_bonus_policy),
            "neutsh_penalty_policy": float(neutsh_penalty_policy),
            "action_quality_penalty_policy": float(action_quality_penalty_policy),
            "cost_volume_penalty": float(cost_volume_penalty),
            "imbalance_penalty": float(imbalance_penalty),
            "cost_volume_penalty_diag": float(cost_volume_penalty_diag),
            "imbalance_penalty_diag": float(imbalance_penalty_diag),
            "directional_imbalance": float(directional_imbalance),
            "alert_precision": alert_prec,
            "alert_coverage": alert_cov,
            "selective_risk": sel_risk,
            "alert_sharpe": float(backtest_metrics["alert_sharpe"]),
            "alert_sortino": float(backtest_metrics["alert_sortino"]),
            "alert_calmar": float(backtest_metrics["alert_calmar"]),
            "alert_max_dd": float(backtest_metrics["alert_max_dd"]),
            # pnl: cumulative net P&L of alerted positions (PDF Table 4)
            # Direction-aware + transaction cost deducted (PDF Section 4.2.4)
            "pnl": float(backtest_metrics.get("pnl", 0.0)),
            # Sprint 1 Patch 1 — surface position-only diagnostics into the
            # JSON-serialised val_losses so the checkpoint gate can read them.
            # Names are explicit:
            #   trade_coverage          : fraction of validation samples that
            #                             actually entered a position
            #   position_pnl            : PnL accrued by UP/DOWN alerts only
            #   n_position_alerts       : raw count of position-taking alerts
            #   neutral_share_of_alerts : NEUTRAL alerts / total alerts (NOT
            #                             a fraction of the dataset — see
            #                             metrics_safe_alert.py docstring)
            #   alert_sharpe_suppressed : Sharpe was clipped to 0 because
            #                             n_position_alerts < min threshold
            "trade_coverage":          float(backtest_metrics.get("trade_coverage", 0.0)),
            "position_pnl":            float(backtest_metrics.get("position_pnl", 0.0)),
            "n_position_alerts":       int(backtest_metrics.get("n_position_alerts", 0)),
            "neutral_share_of_alerts": float(backtest_metrics.get("neutral_share_of_alerts", 0.0)),
            # ``hit_rate`` (further down in this dict) is the legacy
            # confidence-weighted overall metric from compute_hit_rate; this
            # ``position_hit_rate`` is the win rate of UP/DOWN alerts only.
            "position_hit_rate":       float(backtest_metrics.get("position_hit_rate", 0.0)),
            "alert_sharpe_suppressed": bool(backtest_metrics.get("alert_sharpe_suppressed", False)),
            # Sprint 1.5 — direct evidence of NEUTRAL collapse. pred_dist_*
            # measured on the slice that drives early-stopping; alert_pred_dist_*
            # on the alert region only; recall_* per directional class. Watch
            # pred_dist_neutral above ~0.7 + recall_down / recall_up below 0.20
            # as the canonical signature of a collapsed direction head.
            "pred_dist_down":        pred_dist_down,
            "pred_dist_neutral":     pred_dist_neutral,
            "pred_dist_up":          pred_dist_up,
            "alert_pred_dist_down":    alert_pred_dist_down,
            "alert_pred_dist_neutral": alert_pred_dist_neutral,
            "alert_pred_dist_up":      alert_pred_dist_up,
            "recall_down":           recall_down,
            "recall_neutral":        recall_neutral,
            "recall_up":             recall_up,
            # Sprint 2.1-D — confidence-by-class measurements. Use these to
            # diagnose why the policy gate selects a NEUTRAL-heavy region:
            #   • conf_pred_*    : mean confidence per predicted class
            #   • maxprob_pred_* : mean softmax max-prob per predicted class
            #   • conf_alert_*   : mean confidence per predicted class within
            #                      the alert region (post-τ/γ gate)
            "conf_pred_down":      conf_pred_down,
            "conf_pred_neutral":   conf_pred_neutral,
            "conf_pred_up":        conf_pred_up,
            "maxprob_pred_down":    maxprob_pred_down,
            "maxprob_pred_neutral": maxprob_pred_neutral,
            "maxprob_pred_up":      maxprob_pred_up,
            "conf_alert_down":      conf_alert_down,
            "conf_alert_neutral":   conf_alert_neutral,
            "conf_alert_up":        conf_alert_up,
            # Sprint 4 A1 (Phase 7) — position_confidence diagnostic.
            # = confidence × (1 − p_neutral). Pred variant covers all val
            # samples grouped by argmax-pred; alert variant covers samples
            # inside the (c≥τ AND max_p≥γ) alert region. Read pattern:
            #   pos_conf_alert_down/up high while conf_alert_down/up low →
            #     confidence head DOES learn directional signal, the scalar
            #     form is just leaking via NEUTRAL p_neutral. Suggest
            #     action-aware confidence in a future sprint.
            #   pos_conf_alert_down/up also low → confidence head genuinely
            #     not learning directional signal, Lcal needs formula fix.
            "pos_conf_pred_down":     pos_conf_pred_down,
            "pos_conf_pred_neutral":  pos_conf_pred_neutral,
            "pos_conf_pred_up":       pos_conf_pred_up,
            "pos_conf_alert_down":    pos_conf_alert_down,
            "pos_conf_alert_neutral": pos_conf_alert_neutral,
            "pos_conf_alert_up":      pos_conf_alert_up,
            # Sprint 6 Phase 7 — calibration/confidence diagnostics
            "ece_trade":                  float(ece_trade),
            "conf_neutral_alert_slack":   conf_neutral_alert_slack,
            "maxprob_neutral_alert_slack": maxprob_neutral_alert_slack,
            "n_neutral_alert":            int(_n_neutral_alert),
            # Sprint 2.1-C — selection-side collapse penalties so the JSON log
            # records exactly why a checkpoint was downscored. Both fields stay
            # at 0.0 for healthy distributions and rise to 0.10 when the
            # corresponding collapse signature triggers.
            "recall_collapse_penalty": float(recall_collapse_penalty),
            "dist_collapse_penalty":   float(dist_collapse_penalty),
            # Sprint 3.3-C — action-aware adjustments to model_score:
            #   • action_reward in [0, 0.15]: bumps checkpoints that actually
            #     trade (TradeCov ≥ 0.05) and pick the right side often
            #     (PosHit ≥ 0.40 / 0.50 tiers).
            #   • neutsh_penalty in [0, 0.15]: linear ramp on
            #     neutral_share_of_alerts > 0.70.
            "action_reward":           float(action_reward),
            "neutsh_penalty":          float(neutsh_penalty),
            # Sprint 3.3-D — quality-aware additions:
            #   • action_quality_penalty in [0, 0.10]: trades-but-loses penalty.
            #   • deployable_checkpoint: independent hard gate on observable
            #     trading metrics (NOT a function of model_score).
            #   • deployable_reason: short string ("ok" or comma-separated
            #     failure reasons) for case-study / dashboard surfacing.
            "action_quality_penalty":  float(action_quality_penalty),
            "deployable_checkpoint":   bool(deployable_checkpoint),
            "deployable_reason":       str(deployable_reason),
            "model_score": model_score,
            "auc": auc,
            "aurc": aurc,
            "hit_rate": hit_rate,
            "ret_mae": ret_mae,
            "ret_rmse": ret_rmse,
            "ret_corr": ret_correlation,
            # Optional tradability-bin diagnostics. These remain zeroed when
            # lambda_ret_bin is disabled in the paper-final config.
            "ret_bin_acc":        _ret_bin_acc,
            "ret_bin_f1":         _ret_bin_f1,
            "bin_dist_no":        float(_bin_dist[0]),
            "bin_dist_marginal":  float(_bin_dist[1]),
            "bin_dist_strong":    float(_bin_dist[2]),
            "strong_edge_prec":   _strong_edge_prec,
            "dir_acc_on_strong":  _dir_acc_on_strong,
            "n_strong_pred":      int(_n_strong_pred),
            # Optional binary edge-head diagnostics. These remain zeroed when
            # lambda_up_edge/lambda_down_edge are disabled.
            "up_edge_auc":        _up_edge_auc,
            "down_edge_auc":      _down_edge_auc,
            "up_edge_rate":       _up_edge_rate,
            "down_edge_rate":     _down_edge_rate,
            "edge_dir_coverage":  _edge_dir_cov,
            "edge_conflict_rate": _edge_conflict,
            "up_edge_precision":  _up_edge_prec,
            "down_edge_precision": _down_edge_prec,
            # Sprint 10 - per-regime diagnostics, flattened as
            # regime_{sideways|bull|bear|volatile}_{metric}.
            **regime_metrics,
            "attn_sum_mean": attn_sum_mean,
            "attn_target_mean": attn_target_mean,
            "soft_gate_sum_mean": soft_gate_sum_mean,
            "soft_gate_max_mean": soft_gate_max_mean,
            "raw_attn_sum_mean": raw_attn_sum_mean,
            "raw_attn_max_mean": raw_attn_max_mean,
            # Explanation / faithfulness metrics (PDF Section 4.4.2)
            # comprehensiveness = deletion_drop: drop when selected evidence is removed (higher = more faithful)
            "comprehensiveness": del_ins["deletion_drop"],
            "deletion_drop": del_ins["deletion_drop"],    # alias for comprehensiveness
            # insertion_gain: gain from adding selected articles over market-only baseline
            "insertion_gain": del_ins["insertion_gain"],
            # sufficiency_drop: drop when using ONLY selected evidence (lower = more sufficient)
            "sufficiency_drop": sufficiency_score,
            "factor_consistency": factor_consistency_score,
            # Session 26 — cal/sel split diagnostics. macro_f1/mcc/ece/alert_*
            # above are computed on the SEL slice (held out from policy
            # fitting) when _use_split=True. These *_full_val fields keep the
            # legacy whole-val measurements for reviewer comparison.
            "cal_selection_split_used":  bool(_use_split),
            "cal_samples":               int(_cal_end if _use_split else _N_val),
            "sel_samples":               int(_N_val - _cal_end if _use_split else 0),
            "macro_f1_full_val":         float(macro_f1_full_val),
            "mcc_full_val":              float(mcc_full_val),
            "ece_full_val":              float(ece_full_val),
        })

        return val_losses


    # ── fit() helpers ──────────────────────────────────────────────────────────

    def _fit_train_stats(self, train_loader) -> None:
        """Fit all train-only statistics in a SINGLE pass over the loader.

        Combines what used to be 3 separate iterations (class weights, return
        scale, factor label sharpness) into one. Loading 90k+ samples × 8
        articles × multiple tensor streams 3× before epoch 1 was costing ~15-30
        min per fold; this collapses that to a single ~5-10 min pass.

        Sets (in-place on self / self.loss_fn):
          • self.class_weights  + loss_fn.class_weights   (PDF Eq.31 reweighting)
          • loss_fn.return_scale                         (PDF Eq.32 regression scaling)
          • loss_fn.factor_label_smoothing               (PDF Eq.33 soft-CE smoothing)
          • loss_fn.factor_label_mean_max_prob / _mean_entropy (diagnostics)
        """
        # Short-circuit class weights if the caller already provided them via
        # the trainer kwarg (idempotent behaviour preserved from previous API).
        skip_cw = self.class_weights is not None

        all_labels: list = []
        all_returns: list = []
        all_fac_max_probs: list = []
        all_fac_entropy: list = []

        for batch in train_loader:
            if not skip_cw:
                all_labels.extend(batch["direction"].cpu().numpy())
            all_returns.extend(batch["return"].cpu().numpy())

            fac = batch["factor"].float()
            mask = batch.get("article_mask")
            if mask is not None:
                valid = mask.bool()
                if not valid.any():
                    continue
                fac = fac[valid]
            else:
                fac = fac.reshape(-1, fac.shape[-1])
            if fac.numel() == 0:
                continue
            fac = fac.clamp(min=1e-8, max=1.0)
            fac = fac / fac.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            all_fac_max_probs.append(fac.max(dim=-1).values.cpu().numpy())
            all_fac_entropy.append((-(fac * torch.log(fac)).sum(dim=-1)).cpu().numpy())

        # ── Class weights (PDF Eq.31) ─────────────────────────────────────────
        if not skip_cw:
            labels_np = np.asarray(all_labels)
            # Sprint 1.5 — branch on YAML-configured mode. Previous code always
            # took the "sqrt" branch, which for BTC 1h yielded ~uniform weights
            # and could not push the direction head off the NEUTRAL minimum
            # when focal_gamma / label_smoothing were also 0.
            mode = _CLASS_WEIGHTS_MODE
            if mode == "none":
                weights = np.ones(3, dtype=np.float32)
            else:
                weights = compute_class_weight(
                    'balanced', classes=np.array([0, 1, 2]), y=labels_np
                )
                if mode == "sqrt":
                    # Legacy: sqrt-soften so balanced weights (which can hit
                    # 4.5× for rare classes) do not over-correct and tank
                    # majority recall. Halves the exponent (~4.5× → ~2.1×).
                    weights = np.sqrt(weights)
                # mode == "balanced": leave the raw balanced weights alone.
                weights = weights / weights.mean()
            self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.loss_fn.class_weights = self.class_weights.clone().detach().to(self.device)
            assert self.loss_fn.class_weights.device.type == torch.device(self.device).type, \
                f"Class weights on {self.loss_fn.class_weights.device}, model on {self.device}"
            dist = np.bincount(labels_np)
            logger.info(
                "Class weights (mode=%s): DOWN=%.4f NEUTRAL=%.4f UP=%.4f | dist=%s",
                mode, float(weights[0]), float(weights[1]), float(weights[2]), dist,
            )

        # ── Return scale (PDF Eq.32) ──────────────────────────────────────────
        # Robust train-only scale so SmoothL1 gets meaningful gradient on tiny
        # crypto returns without changing regression target semantics.
        returns_np = np.asarray(all_returns, dtype=np.float32)
        abs_returns = np.abs(returns_np)
        q75 = float(np.quantile(abs_returns, 0.75)) if abs_returns.size else 0.0
        std = float(np.std(returns_np)) if returns_np.size else 0.0
        scale = max(q75, std, 1e-4)
        self.loss_fn.return_scale.fill_(scale)
        logger.info("Return scale (train-only robust): %.6f | q75_abs=%.6f std=%.6f", scale, q75, std)

        # ── Factor label smoothing (PDF Eq.33) ────────────────────────────────
        # LLM/keyword pseudo-labels can get near one-hot (mean_max_prob > 0.95
        # on fold 3 runs). A mild uniform smoothing preserves informative
        # gradient without letting label certainty dominate the joint objective.
        if not all_fac_max_probs:
            self.loss_fn.factor_label_smoothing.fill_(0.0)
            self.loss_fn.factor_label_mean_max_prob.fill_(0.0)
            self.loss_fn.factor_label_mean_entropy.fill_(0.0)
            logger.info("Factor label stats unavailable; smoothing disabled.")
            return

        mean_max_prob = float(np.concatenate(all_fac_max_probs, axis=0).mean())
        mean_entropy  = float(np.concatenate(all_fac_entropy,   axis=0).mean())
        num_factor_classes = int(FACTOR_CLASSES)
        entropy_ratio = mean_entropy / max(np.log(max(num_factor_classes, 2)), 1e-8)

        # Session 28 — DISABLED auto-smoothing for Qwen-based labels.
        # Rationale: Qwen 2.5 7B (precompute_factor_labels.py method=qwen) với
        # 2-pass consistency filter đã cho ra factor distributions calibrated
        # (mean_max_prob=0.95 là HỢP LỆ cho Qwen decisive classification, không
        # phải "over-peaked"). 0.04 uniform smoothing trước đây làm mờ Qwen's
        # legitimate sharpness trên articles rõ ràng (institutional_inflow 82%,
        # etf_flow 73%, v.v.) mà không mang lợi ích. Auto-smoothing được thiết kế
        # cho noisy keyword-matched labels, không cần cho LLM labels.
        # Paper Eq.33 là plain soft CE — không có smoothing.
        smoothing = 0.0

        self.loss_fn.factor_label_smoothing.fill_(smoothing)
        self.loss_fn.factor_label_mean_max_prob.fill_(mean_max_prob)
        self.loss_fn.factor_label_mean_entropy.fill_(mean_entropy)
        logger.info(
            "Factor label stats (train-only): mean_max_prob=%.4f mean_entropy=%.4f | smoothing=%.3f",
            mean_max_prob, mean_entropy, smoothing,
        )

    def _log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
    ) -> None:
        """Print one-line epoch summary + λ target / actual / drift triple.

        Sprint 3.1 — the previous "active loss" line filtered out any lambda
        <= 0, which silently hid Stage-1 zero targets AND, crucially, any
        negative drift caused by learned_lambdas=True optimizing lambda
        downward. The new block prints all seven lambda unfiltered, alongside
        the curriculum target and the actual-target drift. Under
        learned_lambdas=False every drift cell must be 0.000; any non-zero
        value is a regression and should be investigated before reading any
        other metric.

        Labels use ASCII ``[lambda_target]`` / ``[lambda_actual]`` /
        ``[lambda_drift]`` so log greps work cleanly under PowerShell / cp1252
        terminals where the Greek lambda character renders as mojibake.
        """
        def _to_float(x):
            return x.item() if hasattr(x, 'item') else float(x)
        labels = ("Ldir", "Lret", "Lfac", "Lsel", "Lcal", "Lfaith", "Lrisk")
        actual = (
            _to_float(self.loss_fn.lambda1),
            _to_float(self.loss_fn.lambda2),
            _to_float(self.loss_fn.lambda3),
            _to_float(self.loss_fn.lambda4),
            _to_float(self.loss_fn.lambda5),
            _to_float(self.loss_fn.lambda6),
            _to_float(self.loss_fn.lambda7),
        )
        target = getattr(self, "_last_lambda_target", actual)
        drift = tuple(a - t for a, t in zip(actual, target))
        # R3 #F: include current LR in the epoch banner so regressions caused
        # by LR collapse (e.g. the pre-fix CosineAnnealingWarmRestarts hitting
        # eta_min at Stage 3 entry) are spotted immediately.
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_losses['loss']:.4f} val={val_losses['loss']:.4f} "
              f"acc={val_losses['dir_acc']:.3f} | lr={current_lr:.3e} | "
              f"Ldir={train_losses['Ldir']:.3f} Lret={train_losses['Lret']:.6f} "
              f"LretSign={train_losses.get('Lret_sign', 0.0):.4f} "
              f"LretCls={train_losses.get('LretCls', 0.0):.4f} "  # Sprint 4-D
              f"LretBin={train_losses.get('LretBin', 0.0):.4f} "
              f"LupEdge={train_losses.get('LupEdge', 0.0):.4f} "
              f"LdownEdge={train_losses.get('LdownEdge', 0.0):.4f} "
              f"Lfac={train_losses['Lfac']:.3f} Lsel={train_losses['Lsel']:.3f} "
              f"Lcal={train_losses['Lcal']:.3f} Lfaith={train_losses['Lfaith']:.3f} "
              f"Lrisk={train_losses['Lrisk']:.3f}")
        target_str = " ".join(f"{n}={t:+.3f}" for n, t in zip(labels, target))
        actual_str = " ".join(f"{n}={a:+.3f}" for n, a in zip(labels, actual))
        drift_str  = " ".join(f"{n}={d:+.3f}" for n, d in zip(labels, drift))
        print(f"         | [lambda_target] {target_str}")
        print(f"         | [lambda_actual] {actual_str}")
        print(f"         | [lambda_drift]  {drift_str}")
        print(
            f"         | [lambda_aux] ret_bin={float(self.loss_fn.lambda_ret_bin):+.3f} "
            f"up_edge={float(self.loss_fn.lambda_up_edge):+.3f} "
            f"down_edge={float(self.loss_fn.lambda_down_edge):+.3f}"
        )
        # Sprint 9 — hybrid market encoder gate diagnostic. Surfaces the
        # mean / max absolute value of the per-channel ``bar_gate`` and the
        # Frobenius norm of ``bar_proj`` so reviewers can see whether the
        # bar pathway is being used (gate>0) or effectively muted (gate≈0,
        # i.e. model is acting as a scalar baseline). Only printed in
        # hybrid mode; no-op for scalar/bar modes.
        try:
            _enc = getattr(getattr(self.model, "market_enc", None), "market_input_mode", None)
            if _enc == "hybrid":
                _gate = self.model.market_enc.bar_gate.detach().abs()
                _proj_w = self.model.market_enc.bar_proj.weight.detach()
                print(
                    f"         | [HybridGate] gate_abs_mean={_gate.mean().item():.4f} "
                    f"gate_abs_max={_gate.max().item():.4f} "
                    f"proj_fro={float(_proj_w.norm().item()):.3f}"
                )
        except Exception:  # noqa — diagnostic only, never fail the epoch
            pass
        # Warn loudly if any lambda went negative — under the curriculum
        # schedule the smallest legal target is 0.0, so a negative actual is
        # always a bug (gradient flow on a buffer, or a fill_() that ran on
        # the wrong tensor). Emit at WARNING so downstream log scrapers can
        # grep it.
        if any(a < 0.0 for a in actual):
            negatives = [(n, a) for n, a in zip(labels, actual) if a < 0.0]
            logger.warning(
                "Negative lambda detected at epoch %d: %s. "
                "Under learned_lambdas=False this should be impossible; "
                "check fill_() / buffer / parameter wiring.",
                epoch, ", ".join(f"{n}={a:+.4f}" for n, a in negatives),
            )

    def _log_metrics(self, val_losses: Dict[str, float]) -> None:
        """Print the [Metrics] line after early-stopping decision."""
        # Sprint 1 Patch 1 — surface position-only diagnostics in the Metrics
        # line. ``Sharpe`` shows a trailing ``*`` when alert_sharpe_suppressed is
        # True (i.e. Sharpe was clipped to 0 because there were too few trades
        # to measure reliably) so the gate reader can distinguish it from a
        # real Sharpe = 0. TradeCov / PosPnL / NeutSh round out the picture so
        # the checkpoint reviewer can see whether the policy is actually
        # trading or just publishing NEUTRAL alerts.
        sharpe_marker = "*" if val_losses.get("alert_sharpe_suppressed", False) else ""
        print(f"  [Metrics] F1={val_losses['macro_f1']:.3f} MCC={val_losses['mcc']:.3f} "
              f"ECE={val_losses['ece']:.4f} AUC={val_losses['auc']:.3f} | "
              f"Sharpe={val_losses['alert_sharpe']:.3f}{sharpe_marker} "
              f"Cov={val_losses['alert_coverage']:.3f} "
              f"TradeCov={val_losses.get('trade_coverage', 0.0):.3f} "
              f"NeutSh={val_losses.get('neutral_share_of_alerts', 0.0):.2f} "
              f"PosAlerts={val_losses.get('n_position_alerts', 0)} "
              f"PosPnL={val_losses.get('position_pnl', 0.0):.4f} "
              f"PosHit={val_losses.get('position_hit_rate', 0.0):.3f} "
              f"Prec={val_losses['alert_precision']:.3f} PnL={val_losses.get('pnl', 0.0):.4f} | "
              f"RetCorr={val_losses['ret_corr']:.4f} | Score={val_losses['model_score']:.4f}")
        print(f"           policy={val_losses['policy_source']} "
              f"conf_src={val_losses.get('policy_confidence_source', self.policy_confidence_source)} "
              f"tau={val_losses['tau']:.3f} gamma={val_losses['gamma']:.3f} "
              f"temp={val_losses['temperature']:.3f} "
              f"(seed={val_losses['tau_seed']:.3f}/{val_losses['gamma_seed']:.3f}) "
              f"Sortino={val_losses['alert_sortino']:.3f} Calmar={val_losses['alert_calmar']:.3f} "
              f"MDD={val_losses['alert_max_dd']:.3f} "
              f"AttnSum={val_losses.get('attn_sum_mean', 0.0):.3f}/"
              f"{val_losses.get('attn_target_mean', 0.0):.3f} "
              f"RawAttn(sum/max)={val_losses.get('raw_attn_sum_mean', 0.0):.3f}/"
              f"{val_losses.get('raw_attn_max_mean', 0.0):.3f} "
              f"GateSum={val_losses.get('soft_gate_sum_mean', 0.0):.2f} "
              f"GateMax={val_losses.get('soft_gate_max_mean', 0.0):.3f} "
              f"LfacSmooth={float(self.loss_fn.factor_label_smoothing.item()):.3f}")
        _policy_obj = val_losses.get("policy_objective", None)
        _policy_obj_s = "n/a" if _policy_obj is None else f"{float(_policy_obj):+.4f}"
        _deploy_s = "Y" if bool(val_losses.get("deployable_checkpoint", False)) else "N"
        print(f"           [PolicyDiag] Obj={_policy_obj_s} "
              f"CovPen={float(val_losses.get('coverage_penalty', 0.0) or 0.0):.3f} "
              f"UnderTrade={val_losses.get('under_trade_cov', 0.0):.3f} "
              f"ActBonus={val_losses.get('action_bonus_policy', 0.0):.3f} "
              f"NeutPen={val_losses.get('neutsh_penalty_policy', 0.0):.3f} "
              f"QualPen={val_losses.get('action_quality_penalty_policy', 0.0):.3f} "
              f"CostDiag={val_losses.get('cost_volume_penalty_diag', 0.0):.3f} "
              f"ImbDiag={val_losses.get('imbalance_penalty_diag', 0.0):.3f} "
              f"DirImb={val_losses.get('directional_imbalance', 0.0):.2f} "
              f"Deploy={_deploy_s}({val_losses.get('deployable_reason', 'n/a')})")
        print(f"           [Explain] Comp={val_losses.get('comprehensiveness', 0.0):.3f} "
              f"SufDrop={val_losses.get('sufficiency_drop', 0.0):.3f} "
              f"Del={val_losses.get('deletion_drop', 0.0):.3f} "
              f"Ins={val_losses.get('insertion_gain', 0.0):.3f} "
              f"FacCons={val_losses.get('factor_consistency', 0.0):.3f}")
        # Sprint 1.5 — collapse diagnostic line. PredDist shows whether the
        # head puts predictions on each class at all; AlertPredDist shows the
        # same restricted to the alert region (the one PnL is measured on);
        # Recall is the per-directional-class recall. If PredDist=NEUTRAL is
        # above ~0.70 with Recall(DOWN/UP) below ~0.20 the head has collapsed.
        print(f"           [PredDist] D/N/U="
              f"{val_losses.get('pred_dist_down', 0.0):.2f}/"
              f"{val_losses.get('pred_dist_neutral', 0.0):.2f}/"
              f"{val_losses.get('pred_dist_up', 0.0):.2f} | "
              f"AlertPredDist D/N/U="
              f"{val_losses.get('alert_pred_dist_down', 0.0):.2f}/"
              f"{val_losses.get('alert_pred_dist_neutral', 0.0):.2f}/"
              f"{val_losses.get('alert_pred_dist_up', 0.0):.2f} | "
              f"Recall D/N/U="
              f"{val_losses.get('recall_down', 0.0):.2f}/"
              f"{val_losses.get('recall_neutral', 0.0):.2f}/"
              f"{val_losses.get('recall_up', 0.0):.2f}")
        # Sprint 8 diagnostic — train-side PredDist printed adjacent to val
        # [PredDist] so 4-quadrant decision matrix can be read at-a-glance:
        #   train balanced + val collapsed → generalization/regime shift
        #   train also collapsed           → optimization issue (sampler/loss bug)
        #   train balanced + val mixed but alerts NEUTRAL → policy/confidence
        #   both mixed but PnL—            → no edge → RetHead/edge head
        # Full-epoch accumulation in train_epoch (not last-K to avoid sampler/
        # curriculum end-state bias per Sprint 7 closure feedback).
        print(f"           [TrainDist] D/N/U="
              f"{val_losses.get('train_pred_dist_down', 0.0):.2f}/"
              f"{val_losses.get('train_pred_dist_neutral', 0.0):.2f}/"
              f"{val_losses.get('train_pred_dist_up', 0.0):.2f} "
              f"(n={int(val_losses.get('train_pred_count', 0))})")
        # Sprint 2.1-D — confidence-by-class line. Read it as
        #   ConfPred  : mean confidence per predicted class
        #   MaxPPred  : mean softmax max-prob per predicted class
        #   ConfAlert : mean confidence within the alert region per pred class
        # If ConfPred(N) is much higher than ConfPred(D/U) the policy gate is
        # naturally cutting directional preds first — that points the next
        # patch at the conf_head, not at the direction head.
        print(f"           [ConfByClass] ConfPred D/N/U="
              f"{val_losses.get('conf_pred_down', 0.0):.3f}/"
              f"{val_losses.get('conf_pred_neutral', 0.0):.3f}/"
              f"{val_losses.get('conf_pred_up', 0.0):.3f} | "
              f"MaxPPred D/N/U="
              f"{val_losses.get('maxprob_pred_down', 0.0):.3f}/"
              f"{val_losses.get('maxprob_pred_neutral', 0.0):.3f}/"
              f"{val_losses.get('maxprob_pred_up', 0.0):.3f} | "
              f"ConfAlert D/N/U="
              f"{val_losses.get('conf_alert_down', 0.0):.3f}/"
              f"{val_losses.get('conf_alert_neutral', 0.0):.3f}/"
              f"{val_losses.get('conf_alert_up', 0.0):.3f}")
        # Sprint 4 A1 (Phase 7) — position_confidence diagnostic line. Kept
        # separate from main [Metrics] / [ConfByClass] to avoid line-length
        # bloat. Read alongside [ConfByClass]: if ConfAlert D/U near zero
        # but PosConf Alert D/U is non-trivial, the directional confidence
        # signal exists but is being masked by the scalar form via NEUTRAL
        # probability mass. That informs whether Sprint 4-B should redesign
        # Lcal formula or just add an action-aware confidence projection.
        print(f"           [PosConf] Pred D/N/U="
              f"{val_losses.get('pos_conf_pred_down', 0.0):.3f}/"
              f"{val_losses.get('pos_conf_pred_neutral', 0.0):.3f}/"
              f"{val_losses.get('pos_conf_pred_up', 0.0):.3f} | "
              f"Alert D/N/U="
              f"{val_losses.get('pos_conf_alert_down', 0.0):.3f}/"
              f"{val_losses.get('pos_conf_alert_neutral', 0.0):.3f}/"
              f"{val_losses.get('pos_conf_alert_up', 0.0):.3f}")
        # Sprint 6 Phase 7 — calibration/confidence gate-source diagnostic.
        #   ECE_global vs ECE_trade: if global low but trade high, calibration
        #     is "đẹp ở vùng không trade" — confidence head broken on D|U
        #   conf_slack vs maxprob_slack on NEUTRAL alerts:
        #     - conf_slack >> maxprob_slack → max_prob is bottleneck
        #       (direction head NEUTRAL bias) → fix Phase 8 focal/CW/LS
        #     - maxprob_slack >> conf_slack → conf gate bottleneck
        #       (confidence scalar broken) → fix Lcal / action-aware conf
        #     - both ~0 → policy threshold tuning helps
        print(f"           [GateSrc] ECE_global={val_losses.get('ece', 0.0):.4f} "
              f"ECE_trade={val_losses.get('ece_trade', 0.0):.4f} | "
              f"NeutAlertSlack conf={val_losses.get('conf_neutral_alert_slack', 0.0):+.3f} "
              f"maxprob={val_losses.get('maxprob_neutral_alert_slack', 0.0):+.3f} "
              f"(n={val_losses.get('n_neutral_alert', 0)})")
        # Sprint 5a — tradability bin diagnostics. Read it as:
        #   RetBinAcc/F1 : head learning on 3-class bin classification
        #   BinDist no/marginal/strong : predicted distribution
        #   StrongPrec : precision on predicted-strong samples (key signal —
        #                must be > 0.5 for tradeability_score to be useful)
        #   DirAccOnStrong : direction acc on the strong-edge subset; if
        #                    this is low while StrongPrec is high, the head
        #                    learns magnitude but direction is still wrong
        #                    on those candles → policy gate (5b) won't help.
        if float(self.loss_fn.lambda_ret_bin) > 0.0:
            print(f"           [RetBin] Acc={val_losses.get('ret_bin_acc', 0.0):.3f} "
                  f"F1={val_losses.get('ret_bin_f1', 0.0):.3f} | "
                  f"BinDist no/mg/st="
                  f"{val_losses.get('bin_dist_no', 0.0):.2f}/"
                  f"{val_losses.get('bin_dist_marginal', 0.0):.2f}/"
                  f"{val_losses.get('bin_dist_strong', 0.0):.2f} | "
                  f"StrongPrec={val_losses.get('strong_edge_prec', 0.0):.3f} "
                  f"(n={val_losses.get('n_strong_pred', 0)}) "
                  f"DirAccOnStrong={val_losses.get('dir_acc_on_strong', 0.0):.3f}")
        else:
            print("           [RetBin] disabled (lambda_ret_bin=0.000)")
        # Optional binary UP/DOWN edge-head diagnostics.
        if float(self.loss_fn.lambda_up_edge) > 0.0 or float(self.loss_fn.lambda_down_edge) > 0.0:
            print(f"           [EdgeBin] UpAUC={val_losses.get('up_edge_auc', 0.5):.3f} "
                  f"DownAUC={val_losses.get('down_edge_auc', 0.5):.3f} | "
                  f"UpRate={val_losses.get('up_edge_rate', 0.0):.3f} "
                  f"DownRate={val_losses.get('down_edge_rate', 0.0):.3f} "
                  f"DirCov={val_losses.get('edge_dir_coverage', 0.0):.3f} "
                  f"Conflict={val_losses.get('edge_conflict_rate', 0.0):.3f} | "
                  f"UpPrec={val_losses.get('up_edge_precision', 0.0):.3f} "
                  f"DownPrec={val_losses.get('down_edge_precision', 0.0):.3f}")
        else:
            print("           [EdgeBin] disabled (lambda_up_edge=lambda_down_edge=0.000)")
        # Sprint 10 - per-regime health check. These are diagnostic-only and
        # use the same validation slice as [Metrics], so a regime with positive
        # PosHit/PosPnL is a candidate for a future filter; uniformly weak
        # regimes support thesis closure rather than more architecture churn.
        for _rkey, _rname in (
            ("sideways", "SIDEWAYS"),
            ("bull", "BULL"),
            ("bear", "BEAR"),
            ("volatile", "VOLATILE"),
        ):
            _prefix = f"regime_{_rkey}_"
            print(
                f"           [RegimePerf] {_rname} "
                f"n={int(val_losses.get(_prefix + 'n', 0))} "
                f"F1={val_losses.get(_prefix + 'f1', 0.0):.3f} "
                f"MCC={val_losses.get(_prefix + 'mcc', 0.0):.3f} | "
                f"PredD/N/U="
                f"{val_losses.get(_prefix + 'pred_down', 0.0):.2f}/"
                f"{val_losses.get(_prefix + 'pred_neutral', 0.0):.2f}/"
                f"{val_losses.get(_prefix + 'pred_up', 0.0):.2f} | "
                f"AlertD/N/U="
                f"{val_losses.get(_prefix + 'alert_down', 0.0):.2f}/"
                f"{val_losses.get(_prefix + 'alert_neutral', 0.0):.2f}/"
                f"{val_losses.get(_prefix + 'alert_up', 0.0):.2f} | "
                f"Cov={val_losses.get(_prefix + 'alert_cov', 0.0):.3f} "
                f"TradeCov={val_losses.get(_prefix + 'trade_cov', 0.0):.3f} "
                f"NeutSh={val_losses.get(_prefix + 'neutral_share', 0.0):.2f} "
                f"PosAlerts={int(val_losses.get(_prefix + 'n_pos', 0))} "
                f"PosHit={val_losses.get(_prefix + 'pos_hit', 0.0):.3f} "
                f"PosPnL={val_losses.get(_prefix + 'pos_pnl', 0.0):.4f}"
            )

    def _make_selection_snapshot(
        self,
        epoch: int,
        val_losses: Dict[str, float],
    ) -> Dict[str, Any]:
        """Sprint 3.3-D-fix — capture deployability metadata at training-time
        best-checkpoint moment.

        Why this exists: after fit() returns, the outer loop loads the best
        checkpoint and calls validate() AGAIN on the val split. That
        re-validation refits temperature scaling from scratch on uncalibrated
        logits — T* commonly clamps to 2.500 / 0.500 — which lands the policy
        search on a different (often suppressed) tau/gamma than the one that
        actually won the [BEST] gate during training. As a result,
        ``deployable_reason`` surfaced via re-validation is the reason for the
        re-validated policy, not for the policy that earned the checkpoint
        its score. Snapshotting at training time is a pure bookkeeping fix —
        it does not alter scoring, action_reward formula, or the deployable
        formula itself.

        Fields tracked are exactly the ones needed for cross-fold deploy
        gating + post-hoc audit; macro_f1 / ECE / AUC etc. (which ARE
        deterministic from the weights and reproduce on re-validation) stay
        sourced from the re-validation block.
        """
        return {
            "epoch":                   int(epoch),
            "model_score":             float(val_losses.get("model_score", 0.0)),
            "trade_coverage":          float(val_losses.get("trade_coverage", 0.0)),
            "position_pnl":            float(val_losses.get("position_pnl", 0.0)),
            "position_hit_rate":       float(val_losses.get("position_hit_rate", 0.0)),
            "alert_sharpe":            float(val_losses.get("alert_sharpe", 0.0)),
            "alert_sharpe_suppressed": bool(val_losses.get("alert_sharpe_suppressed", False)),
            "neutral_share_of_alerts": float(val_losses.get("neutral_share_of_alerts", 0.0)),
            "n_position_alerts":       int(val_losses.get("n_position_alerts", 0)),
            "action_reward":           float(val_losses.get("action_reward", 0.0)),
            "action_quality_penalty":  float(val_losses.get("action_quality_penalty", 0.0)),
            "neutsh_penalty":          float(val_losses.get("neutsh_penalty", 0.0)),
            "deployable_checkpoint":   bool(val_losses.get("deployable_checkpoint", False)),
            "deployable_reason":       str(val_losses.get("deployable_reason", "n/a")),
            # Sprint 3.3-F — policy params snapshot. tau/gamma/T are the
            # thresholds the policy search landed on AT TRAINING TIME for
            # this checkpoint. Outer-loop re-validation refits T and
            # commonly clamps to 2.500 / 0.500, producing a different
            # policy than the one that earned the [BEST] gate. Routing
            # these from snapshot ensures test eval, fold_record val, and
            # the deploy policy artifact reflect the policy the
            # checkpoint actually won under, not the post-load drift.
            # Seeds and provenance fields are kept for audit (which grid
            # branch won, what objective value it scored).
            "tau":                float(val_losses.get("tau", 0.0)),
            "gamma":              float(val_losses.get("gamma", 0.0)),
            "temperature":        float(val_losses.get("temperature", 1.0)),
            "tau_seed":           float(val_losses.get("tau_seed", 0.0)),
            "gamma_seed":         float(val_losses.get("gamma_seed", 0.0)),
            "temperature_source": str(val_losses.get("temperature_source", "unknown")),
            "policy_source":      str(val_losses.get("policy_source", "unknown")),
            "policy_confidence_source": str(val_losses.get(
                "policy_confidence_source", self.policy_confidence_source
            )),
            "policy_objective":   float(val_losses.get("policy_objective", 0.0) or 0.0),
            "coverage_penalty":   float(val_losses.get("coverage_penalty", 0.0) or 0.0),
            "under_trade_cov":    float(val_losses.get("under_trade_cov", 0.0)),
            "action_bonus_policy": float(val_losses.get("action_bonus_policy", 0.0)),
            "neutsh_penalty_policy": float(val_losses.get("neutsh_penalty_policy", 0.0)),
            "action_quality_penalty_policy": float(val_losses.get(
                "action_quality_penalty_policy", 0.0
            )),
            "cost_volume_penalty": float(val_losses.get("cost_volume_penalty", 0.0)),
            "imbalance_penalty":   float(val_losses.get("imbalance_penalty", 0.0)),
            "cost_volume_penalty_diag": float(val_losses.get(
                "cost_volume_penalty_diag", 0.0
            )),
            "imbalance_penalty_diag": float(val_losses.get(
                "imbalance_penalty_diag", 0.0
            )),
            "directional_imbalance": float(val_losses.get("directional_imbalance", 0.0)),
        }

    def _try_save_best_checkpoint(
        self,
        epoch: int,
        val_losses: Dict[str, float],
        best_score: float,
        patience_counter: int,
        checkpoint_dir: Path,
        tag: str = "best",
    ) -> Tuple[float, Optional[Path], int]:
        """Save checkpoint if current score improves best; increment patience otherwise.

        Returns:
            (best_score, best_ckpt_path, patience_counter)
        """
        score = val_losses['model_score']
        if score > best_score:
            best_score = score
            patience_counter = 0
            ckpt = checkpoint_dir / f"safe_alert_{self.horizon}_{tag}_epoch{epoch}.pt"
            torch.save(self._build_checkpoint_dict(
                epoch=epoch,
                model_state=self.model.state_dict(),
                extra={
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_loss":       val_losses['loss'],
                    "val_acc":        val_losses['dir_acc'],
                    "model_score":    score,
                    "macro_f1":       val_losses['macro_f1'],
                    "ece":            val_losses['ece'],
                    "alert_sharpe":   val_losses['alert_sharpe'],
                    "temperature":    val_losses.get('temperature', 1.0),
                    "tau":            val_losses.get('tau'),
                    "gamma":          val_losses.get('gamma'),
                },
            ), ckpt)
            print(f"  [BEST] Score={score:.4f} F1={val_losses['macro_f1']:.3f} "
                  f"ECE={val_losses['ece']:.4f} Sharpe={val_losses['alert_sharpe']:.3f}")
            return best_score, ckpt, patience_counter
        else:
            return best_score, None, patience_counter + 1

    def _build_checkpoint_dict(
        self,
        epoch: int,
        model_state: Dict,
        extra: Optional[Dict] = None,
    ) -> Dict:
        """Assemble the full checkpoint payload expected by downstream loaders.

        Always includes *preprocessing state* (market scaler + return scale +
        class weights + horizon + factor ontology size) so live_infer and
        backtest can rehydrate the EXACT normalization pipeline that training
        used. Missing preprocessing state was Bug #1 in the P0 fix list: raw
        features at inference vs. z-scored features at training produced silent
        garbage predictions in production.
        """
        payload: Dict = {
            "model_state": model_state,
            "epoch":       int(epoch),
            "horizon":     self.horizon,
            "K_h":         int(self.K_h),
            # Ablation mode — logged so checkpoint consumers can verify they
            # loaded the intended variant (None = full model, "w/o_faithfulness", etc.)
            "ablation":    self.ablation,
            # Preprocessing state — required for production inference parity.
            "preprocessing_state": (
                self.dataset.get_preprocessing_state()
                if getattr(self, "dataset", None) is not None else None
            ),
            # Loss normalization state snapshotted from the active loss function.
            "return_scale":          float(self.loss_fn.return_scale.item()),
            "factor_label_smoothing": float(self.loss_fn.factor_label_smoothing.item()),
            "class_weights": (
                self.class_weights.detach().cpu().clone()
                if self.class_weights is not None else None
            ),
            # Ontology size — lets loaders verify factor module shape match.
            "n_factors":  int(FACTOR_CLASSES),
        }
        if extra:
            payload.update(extra)
        return payload

    def _finalize_training(
        self,
        checkpoint_dir: Path,
        best_ckpt_path: Optional[Path],
        best_overall_ckpt_path: Optional[Path],
        last_ckpt_path: Optional[Path],
        best_score: float,
        best_overall_score: float,
        final_epoch: int,
        final_val_loss: float,
        final_val_acc: float,
        swa_ckpt_path: Optional[Path] = None,
        swa_model_score: float = float('-inf'),
        stage3_score_std: Optional[float] = None,
        best_overall_epoch: int = 0,
        total_epochs: int = 0,
        best_overall_meta: Optional[Dict[str, Any]] = None,
        stage3_best_meta: Optional[Dict[str, Any]] = None,
        swa_best_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Path]:
        """Promote the most deploy-worthy checkpoint to FINAL and persist metrics.

        R3 #B: **Score-primary** selection, not stage-primary. Previous logic
        blindly preferred Stage 3 best even when the best_overall score was
        materially higher. In Fold 1 this meant deploying ep 32 (Score=0.3010)
        while silently discarding ep 15 (Score=0.3143) — a 4.4% Score advantage.
        The "Stage 3 is more calibrated" argument only holds if Stage 3 is
        actually trained effectively; when LR/λ dynamics bite, ignoring a
        clearly better Stage-2 checkpoint makes the final model strictly worse.

        NEW priority:
          1. SWA — when SWA validation beats everything else by the adaptive
             margin (``max(0.003, 0.5·std(stage3_scores))``). Averaging lottery-
             ticket weights only wins when the averaging pays off.
          2. best_overall — when it beats best_stage3 by at least
             ``_OVERALL_MIN_MARGIN=0.01`` (a meaningful 1% Score edge). This
             avoids Stage-2 checkpoints winning on noise while still rescuing
             the Fold 1 pattern where Stage 3 is actively worse.
          3. Stage 3 best — default when neither SWA nor overall have a
             sufficiently large advantage (keeps the paper's "prefer calibrated"
             spirit when calibration actually helps).
          4. best_overall — even without the 0.01 margin, used as fallback when
             Stage 3 never produced a best (rare, e.g. NaN crash in Stage 3).
          5. last epoch — absolute last resort.

        The margin ``_OVERALL_MIN_MARGIN`` is exposed via the curriculum config
        so it can be tuned per-experiment; set to 0 to disable and always
        prefer best_overall, or to 1.0 to disable and always prefer stage3.
        """
        final_path = checkpoint_dir / f"safe_alert_{self.horizon}_FINAL.pt"
        promoted_source = "unknown"
        # Adaptive margin: larger when Stage 3 scores were volatile.
        _FLOOR = 0.003
        if stage3_score_std is not None and stage3_score_std > 0:
            _SWA_MIN_MARGIN = max(_FLOOR, 0.5 * float(stage3_score_std))
        else:
            _SWA_MIN_MARGIN = 0.005  # legacy default when std is unavailable
        # R3 #K: stage-aware margin for best_overall. If the best_overall
        # checkpoint landed in Stage 1 (pre-Lfac/Lcal/Lrisk/Lfaith), it's
        # dangerously uncalibrated — require a LARGER 0.02 margin before
        # letting it override the properly-calibrated Stage 3 best. If it
        # landed in Stage 2 (at least Lfac/Lfaith are ramping), 0.01 is fine.
        _OVERALL_MIN_MARGIN_S2PLUS = 0.01
        _OVERALL_MIN_MARGIN_S1ONLY = 0.02
        if total_epochs > 0 and best_overall_epoch > 0:
            overall_frac = best_overall_epoch / total_epochs
            overall_margin = (
                _OVERALL_MIN_MARGIN_S1ONLY
                if overall_frac <= _STAGE1_END_FRAC
                else _OVERALL_MIN_MARGIN_S2PLUS
            )
        else:
            overall_margin = _OVERALL_MIN_MARGIN_S2PLUS  # legacy fallback

        swa_available = swa_ckpt_path is not None and swa_ckpt_path.exists()
        stage3_available = best_ckpt_path is not None and best_ckpt_path.exists()
        overall_available = best_overall_ckpt_path is not None and best_overall_ckpt_path.exists()

        # Candidate table — pick the one with the highest score above its margin.
        # This matches R3 #B intent: let the best-calibrated PERFORMING ckpt win.
        candidates = []
        if stage3_available:
            candidates.append(("stage3_best", best_score, best_ckpt_path, 0.0))
        if overall_available:
            # R3 #K: stage-aware margin (computed above).
            margin = 0.0 if not stage3_available else overall_margin
            candidates.append(("overall_best", best_overall_score, best_overall_ckpt_path, margin))
        if swa_available:
            margin = 0.0 if not stage3_available else _SWA_MIN_MARGIN
            candidates.append(("swa", swa_model_score, swa_ckpt_path, margin))

        # Compute "effective" score = raw score - margin required
        effective = [(name, score - margin, score, ckpt)
                     for (name, score, ckpt, margin) in candidates]
        # Sprint 4-B0 — deployment-selection safety. The pre-B0 policy
        # selected the highest-effective-score candidate regardless of
        # whether that candidate passed the deployable hard gate (Sprint
        # 3.3-D ``deployable_checkpoint``). Combined with Sprint 3.3-E
        # which marks the artifact ``[WARN] DO NOT DEPLOY`` when 0/N
        # folds are deployable, the file system still ended up with
        # ``FINAL.pt`` pointing at a checkpoint that traded at a loss.
        # That is a deployment risk: a downstream consumer / human reading
        # ``[OK] FINAL model = overall_best epoch2`` reads it as a
        # green-light signal even when the policy artifact says NOT
        # DEPLOYABLE. B0 fixes this at the FINAL.pt selection level:
        #   1. If at least one candidate is deployable, pick the
        #      highest-effective-score AMONG deployable candidates.
        #      This also avoids forcing Stage 3 when Stage 3 is not
        #      deployable either — we only restrict the pool, never
        #      reorder by stage.
        #   2. If 0 candidates are deployable, fall back to legacy
        #      score-primary pick BUT log [WARN] BEST-EFFORT ONLY +
        #      surface the deployable_reason. The .pt still gets written
        #      so audit / replay / ablation work — only the framing
        #      changes from "FINAL" to "BEST-EFFORT".
        # Look up each candidate's deployable flag from selection_meta
        # (captured at training time, the source of truth — see
        # Sprint 3.3-D-fix and 3.3-F for the routing chain).
        _meta_lookup = {
            "overall_best": best_overall_meta,
            "stage3_best":  stage3_best_meta,
            "swa":          swa_best_meta,
        }

        def _deployable_for(name: str) -> bool:
            m = _meta_lookup.get(name)
            return bool(m.get("deployable_checkpoint", False)) if isinstance(m, dict) else False

        def _deploy_reason_for(name: str) -> str:
            m = _meta_lookup.get(name)
            return str(m.get("deployable_reason", "n/a")) if isinstance(m, dict) else "n/a"

        if effective:
            effective.sort(key=lambda x: x[1], reverse=True)
            # Filter deployable candidates first.
            deployable_effective = [t for t in effective if _deployable_for(t[0])]
            if deployable_effective:
                promoted_source, _, promoted_raw, promoted_ckpt = deployable_effective[0]
                final_basis = "deployable_pick"
                shutil.copy(promoted_ckpt, final_path)
                print(
                    f"[OK] FINAL model = {promoted_source} (score={promoted_raw:.4f}): "
                    f"{promoted_ckpt.name} deployable=True"
                )
            else:
                # Best-effort pick — legacy highest-effective-score.
                promoted_source, _, promoted_raw, promoted_ckpt = effective[0]
                final_basis = "best_effort_no_deployable"
                shutil.copy(promoted_ckpt, final_path)
                _reason = _deploy_reason_for(promoted_source)
                print(
                    f"[WARN] FINAL checkpoint is BEST-EFFORT ONLY — NOT DEPLOYABLE\n"
                    f"       FINAL model = {promoted_source} "
                    f"(score={promoted_raw:.4f}): {promoted_ckpt.name}\n"
                    f"       reason={_reason} status=not_recommended "
                    f"(0/{len(effective)} candidates passed deployable gate)\n"
                    f"       The .pt is saved for audit/replay; "
                    f"do NOT promote to production. See artifact "
                    f"deploy_status field for downstream gating."
                )
            # Diagnostic table — every candidate with raw, effective, AND
            # deployable status. Marker shows FINAL pick + (best-effort) tag.
            for (name, eff, raw, ckpt) in effective:
                _dep = _deployable_for(name)
                if name == promoted_source:
                    marker = "← FINAL" if final_basis == "deployable_pick" else "← BEST-EFFORT FINAL"
                else:
                    marker = ""
                print(
                    f"     [candidate] {name:<14s} raw={raw:.4f} "
                    f"effective={eff:.4f} deployable={_dep} {marker}"
                )
        elif last_ckpt_path is not None and last_ckpt_path.exists():
            shutil.copy(last_ckpt_path, final_path)
            promoted_source = "last_epoch"
            final_basis = "last_epoch_fallback"
            print(f"[WARN] No best checkpoint found — FINAL = last epoch ({final_epoch})")
        else:
            final_path = last_ckpt_path  # absolute fallback
            final_basis = "no_checkpoint"

        metrics = {
            "final_epoch":         final_epoch,
            "final_val_loss":      float(final_val_loss),
            "final_val_acc":       float(final_val_acc),
            "total_epochs_trained": final_epoch,
            "model_file":          final_path.name if final_path else "unknown",
            "final_source":        promoted_source,
            # R3 #K: record which epoch (and therefore which stage) produced
            # best_overall so post-hoc analysis can see if selection needed
            # the larger 0.02 margin (Stage 1 only) or smaller 0.01 (Stage 2+).
            "best_overall_epoch":  int(best_overall_epoch),
            "best_overall_stage": (
                "S1" if (total_epochs > 0 and best_overall_epoch <= int(total_epochs * _STAGE1_END_FRAC))
                else "S2" if (total_epochs > 0 and best_overall_epoch <= int(total_epochs * _STAGE2_END_FRAC))
                else "S3"
            ),
            "note": (
                "FINAL selection (R3 #B + #K + Sprint 4-B0): score-primary with "
                "stage-aware margins, AND deployable-first when any candidate "
                "passes the hard gate. best_overall must beat stage3_best by "
                "0.02 if it landed in Stage 1 (uncalibrated), else 0.01 for "
                "Stage 2+. SWA needs max(0.003, 0.5·std) margin. Priority is "
                "by EFFECTIVE score (raw − margin) within the deployable pool; "
                "if 0 candidates are deployable, legacy score-primary pick is "
                "used but flagged BEST-EFFORT (do not deploy)."
            ),
            # Sprint 4-B0 — deploy gate at the FINAL.pt level.
            #   final_deployable: True iff the promoted candidate passed the
            #     Sprint 3.3-D hard gate (trade_cov >= 0.05 AND not
            #     sharpe-suppressed AND not pnl_and_sharpe_negative).
            #   final_selection_basis: provenance of the pick.
            #     "deployable_pick"          → at least 1 candidate deployable
            #     "best_effort_no_deployable"→ 0/N deployable, .pt saved
            #                                  for audit/replay only
            #     "last_epoch_fallback"      → no best checkpoint produced
            #     "no_checkpoint"            → no .pt at all (extreme failure)
            "final_deployable": (
                bool(_deployable_for(promoted_source))
                if promoted_source in _meta_lookup else False
            ),
            "final_selection_basis": str(final_basis),
        }
        if best_overall_ckpt_path is not None and best_overall_ckpt_path.exists():
            metrics["best_overall_model_file"] = best_overall_ckpt_path.name
            metrics["best_overall_model_score"] = float(best_overall_score)
        if swa_ckpt_path is not None and swa_ckpt_path.exists():
            metrics["swa_model_file"] = swa_ckpt_path.name

        # Sprint 3.3-D-fix — attach the training-time deploy-metadata snapshot
        # for the candidate that was actually promoted. Routes around the
        # outer-loop re-validation T-refit divergence: deployable_reason etc.
        # surfaced downstream now reflects the policy that earned the
        # checkpoint, not whatever the re-validate landed on. ``selection_meta``
        # is None only when we fell through to last_epoch / no checkpoint at
        # all; the outer loop must guard for that.
        _meta_by_source = {
            "overall_best": best_overall_meta,
            "stage3_best":  stage3_best_meta,
            "swa":          swa_best_meta,
        }
        selection_meta = _meta_by_source.get(promoted_source)
        if selection_meta is not None:
            # Copy so JSON serialisation of metrics does not bind the dict
            # held by fit() callers (defensive — outer code only reads).
            metrics["selection_meta"] = dict(selection_meta)
        else:
            metrics["selection_meta"] = None

        metrics_path = checkpoint_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[OK] Saved training metrics: {metrics_path.name}")

        return metrics, final_path

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 20,
        checkpoint_dir: Path = None,
        early_stopping_patience: int = 12,
        explain_every: int = 5,
        dataset=None,
    ) -> Tuple[Dict, Path]:
        """Train with 3-stage curriculum and stage-aware early stopping (PDF Section 4.5.1).

        Stages:
          1 (0–20%): direction + return + selection — build direction/return foundation.
          2 (20–60%): warm-up Lfac + Lfaith + small Lcal/Lrisk — smooth λ transitions.
          3 (60–100%): full loss active — early stopping on model_score.
        Model selection optimizes Score = 0.40·F1 + 0.35·Sharpe − 0.15·ECE + 0.10·MCC.

        ``explain_every``: compute full explanation metrics (Del/Ins/Suff/Comp)
        every N epochs instead of every epoch. 5 is the default trade-off —
        tracks faithfulness each week of training without paying the 3×-forward
        cost every step. Set to 1 to restore the per-epoch behavior.
        """
        checkpoint_dir = (checkpoint_dir or ARTIFACT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Keep a handle so checkpoint saves can snapshot preprocessing state
        # (market scaler mean/std/clip). Without this, production inference
        # receives RAW features while training uses z-score normalized ones
        # — a silent distribution mismatch that makes live predictions garbage.
        self.dataset = dataset

        # Fit all train-only statistics in a SINGLE pass over the loader
        # (class weights + return scale + factor label stats). Previously each
        # was a separate iteration → 3× full data load before epoch 1.
        self._fit_train_stats(train_loader)

        # Session 26 — install per-factor class weights for Lfac.
        # Fixes network_outage (0.6 % corpus) and other rare factors that
        # were otherwise ignored by the factor head. Weights are computed
        # ONCE from the full corpus at dataset init (same for every fold)
        # — they encode corpus-level label frequency, NOT fold-specific
        # label counts, so applying them does NOT leak train labels into
        # val/test. Pass-through gracefully when the dataset did not load
        # factor labels (keyword fallback path).
        _factor_cw = getattr(dataset, "factor_class_weights", None)
        if _factor_cw is not None:
            self.loss_fn.set_factor_class_weights(_factor_cw)

        # R3 #A: Rebuild the scheduler now that ``epochs`` is known. Previous
        # code used CosineAnnealingWarmRestarts whose warm restart schedule
        # collided with the 3-stage curriculum (LR near eta_min=1e-7 at Stage 3
        # entry). StageAwareLRScheduler aligns LR phases with stage boundaries
        # so Stage 3 always starts at ``base_lr × stage3_lr_scale`` (default
        # 1.5e-4) and decays smoothly to ``base_lr × eta_frac`` (default
        # 6e-6), a 25× floor improvement vs the old 1e-7.
        self.scheduler = StageAwareLRScheduler(
            self.optimizer,
            total_epochs=int(epochs),
            base_lr=self.base_lr,
            stage1_end_frac=_STAGE1_END_FRAC,
            stage2_end_frac=_STAGE2_END_FRAC,
            stage3_lr_scale=_STAGE3_LR_SCALE,
            warmup_start_frac=0.1,   # linear warmup from base_lr/10
            eta_frac=0.02,           # Stage-3 floor at base_lr/50
        )
        # Prime LR for epoch 0 so the first batch uses warmup_start_lr, not
        # whatever the placeholder LambdaLR left the param groups at.
        self.scheduler.step(epoch=1)

        # FIX: +1 ensures _enter_stage3() fires at the first epoch where Stage 3
        # lambdas are actually active. Without +1, at epoch = int(epochs*0.60),
        # frac == 0.60 which still maps to Stage 2 in _update_loss_weights_for_stage
        # (elif frac <= _STAGE2_END_FRAC), so LR was scaled 1 epoch before Stage 3
        # lambda activated — causing patience counter to start too early.
        stage3_epochs  = max(2, int(epochs * _STAGE2_END_FRAC) + 1)
        patience_counter = 0
        best_score     = float('-inf')
        best_ckpt_path = None
        best_overall_score = float('-inf')
        best_overall_ckpt_path = None
        # R3 #K: track WHICH epoch and stage produced best_overall so downstream
        # checkpoint selection can apply a stage-aware margin. A best_overall
        # from Stage 1 (no Lcal/Lfac/Lfaith/Lrisk) should need a larger margin
        # before overriding a properly-calibrated Stage 3 best.
        best_overall_epoch = 0
        last_ckpt_path = None
        final_epoch, final_val_loss, final_val_acc = 0, 0.0, 0.0

        # Sprint 3.3-D-fix — track the val_losses snapshot at the moment each
        # candidate (best_overall / stage3_best / SWA) won its gate. The outer
        # loop's re-validation refits T from scratch and lands on a different
        # tau/gamma than training time, so deployable_reason from re-validate
        # mis-attributes which checkpoint we are actually deploying. These
        # in-memory dicts get returned via _finalize_training so the outer
        # fold_record can read deployable info from the candidate that won.
        best_overall_meta: Optional[Dict[str, Any]] = None
        stage3_best_meta: Optional[Dict[str, Any]] = None
        swa_best_meta: Optional[Dict[str, Any]] = None

        # ── Stochastic Weight Averaging (SWA) ────────────────────────────────
        # Averages model parameters over the tail of training (last 15% of
        # epochs). This produces a smoother decision surface that generalises
        # better across walk-forward folds — addressing the observation that a
        # single-point Stage 3 checkpoint can have a large val→test Sharpe gap
        # under temporal shift. SWA typically recovers 1-3 F1 points and closes
        # val-test gaps. We reuse the outer CosineAnnealing scheduler for LR
        # (already keeps LR varying during SWA phase) rather than switching to
        # SWALR to avoid yet another hyperparameter.
        from torch.optim.swa_utils import AveragedModel
        # R3 #D + #J: SWA start with minimum-window safeguard.
        # `curriculum.swa_start_frac` default 0.78 gives ~9 epochs averaging for
        # a 40-epoch run. But short runs (epochs=10, 15) would give a window too
        # small for meaningful averaging: 10 × 0.22 = 2.2 epochs, can't beat a
        # single-point best. Enforce minimum ``_SWA_MIN_WINDOW=5`` epochs and
        # clamp swa_start so it can always fit. If can't fit (e.g. epochs<8),
        # DISABLE SWA entirely by setting swa_start_epoch > epochs.
        _SWA_MIN_WINDOW = 5
        computed_start = max(stage3_epochs + 1, int(epochs * _SWA_START_FRAC))
        max_start_for_window = epochs - _SWA_MIN_WINDOW + 1
        swa_start_epoch = min(computed_start, max_start_for_window)
        if not _ENABLE_SWA:
            swa_start_epoch = epochs + 1
            print("[SWA] Disabled by config")
        elif swa_start_epoch > epochs or swa_start_epoch < stage3_epochs + 1:
            # Not enough epochs for a meaningful SWA window — disable.
            swa_start_epoch = epochs + 1  # never activates
            print(f"[SWA] Disabled: epochs={epochs} too short for min window of {_SWA_MIN_WINDOW}")
        swa_model: Optional[AveragedModel] = None
        stage3_val_scores: list[float] = []  # for adaptive SWA margin

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._update_loss_weights_for_stage(epoch, epochs)

            train_losses = self.train_epoch(train_loader)
            # Run cheap validation (skip 3 extra explanation forwards) on most
            # epochs; do the full faithfulness suite only every ``explain_every``
            # steps. Always run full validation inside Stage 3 and on the final
            # epoch so early-stop / best-checkpoint decisions see the real metric.
            run_full_explain = (
                explain_every <= 1
                or (epoch % max(explain_every, 1) == 0)
                or epoch >= stage3_epochs
                or epoch == epochs
            )
            val_losses = self.validate(val_loader, compute_explanations=run_full_explain)
            # Sprint 8 diagnostic — carry train-side PredDist into the same
            # epoch metrics dict consumed by _log_metrics/checkpoint snapshots.
            # _log_metrics only receives val_losses, so keeping these keys here
            # avoids leaking train_losses into its scope.
            for _k in (
                "train_pred_dist_down",
                "train_pred_dist_neutral",
                "train_pred_dist_up",
                "train_pred_count",
            ):
                if _k in train_losses:
                    val_losses[_k] = train_losses[_k]

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._log_epoch(epoch, epochs, train_losses, val_losses)

            # Stage-aware early stopping: only active in Stage 3.
            # Stage 1-2 have near-zero Sharpe by design; tracking score there would
            # set an unreachable baseline before calibration is active.
            new_best_overall_score, overall_ckpt, _ = self._try_save_best_checkpoint(
                epoch, val_losses, best_overall_score, 0, checkpoint_dir, tag="best_overall"
            )
            if overall_ckpt:
                best_overall_ckpt_path = overall_ckpt
                best_overall_epoch = epoch  # R3 #K: record which stage produced it
                # Sprint 3.3-D-fix — snapshot deploy-relevant fields the moment
                # this checkpoint won. dict() copies so subsequent epochs do not
                # mutate the captured state.
                best_overall_meta = self._make_selection_snapshot(epoch, val_losses)
            best_overall_score = new_best_overall_score

            if epoch == stage3_epochs:
                # First epoch of Stage 3: reset counters so Stage 2 results
                # don't influence the Stage 3 baseline (fresh start).
                patience_counter = 0
                best_score = float('-inf')
                self._enter_stage3()

            if epoch >= stage3_epochs:
                best_score, new_ckpt, patience_counter = self._try_save_best_checkpoint(
                    epoch, val_losses, best_score, patience_counter, checkpoint_dir, tag="best"
                )
                if new_ckpt:
                    best_ckpt_path = new_ckpt
                    # Sprint 3.3-D-fix — snapshot the Stage 3 best at win moment.
                    stage3_best_meta = self._make_selection_snapshot(epoch, val_losses)
                # Track Stage 3 val scores so SWA can calibrate its acceptance
                # margin against observed noise rather than a hardcoded 0.005.
                score_val = val_losses.get("model_score")
                if score_val is not None and np.isfinite(score_val):
                    stage3_val_scores.append(float(score_val))
                if patience_counter >= early_stopping_patience:
                    print(f"[STOP] Early stopping at epoch {epoch} "
                          f"(no improvement for {early_stopping_patience} epochs, "
                          f"best score={best_score:.4f})")
                    self._log_metrics(val_losses)
                    break

            self._log_metrics(val_losses)

            final_epoch, final_val_loss, final_val_acc = epoch, val_losses['loss'], val_losses['dir_acc']

            # Overwrite last-epoch checkpoint every epoch (fallback for no Stage 3 best)
            last_ckpt_path = checkpoint_dir / f"safe_alert_{self.horizon}_last.pt"
            torch.save(self._build_checkpoint_dict(
                epoch=epoch,
                model_state=self.model.state_dict(),
                extra={
                    "val_loss": final_val_loss,
                    "val_acc":  final_val_acc,
                    "temperature": val_losses.get('temperature', 1.0),
                    "tau":         val_losses.get('tau'),
                    "gamma":       val_losses.get('gamma'),
                    "lr":          self.optimizer.param_groups[0]["lr"],
                },
            ), last_ckpt_path)

            # R3 #A: advance the stage-aware scheduler to the NEXT epoch's LR.
            # StageAwareLRScheduler.step() accepts an explicit epoch so we can
            # feed epoch+1 (the LR the next forward pass should use).
            if isinstance(self.scheduler, StageAwareLRScheduler):
                self.scheduler.step(epoch=epoch + 1)
            else:
                self.scheduler.step()

            # SWA update — lazy-initialize the averaged model on entry and
            # update its running average with the current parameters after
            # the LR step. Using post-step params means SWA sees the same
            # weights used for the next epoch's training.
            if epoch >= swa_start_epoch:
                if swa_model is None:
                    swa_model = AveragedModel(self.model)
                    print(f"[SWA] Weight averaging enabled from epoch {epoch} "
                          f"(last {epochs - swa_start_epoch + 1} epochs)")
                swa_model.update_parameters(self.model)

        # Finalise SWA model if any averaging happened — update BN statistics
        # (critical: averaged weights don't have correct running mean/var for
        # BN layers) and save as a dedicated checkpoint. _finalize_training
        # will consider swa_ckpt_path when promoting to FINAL.
        swa_ckpt_path: Optional[Path] = None
        swa_val_metrics: Optional[Dict] = None
        swa_model_score: float = float('-inf')
        if swa_model is not None:
            try:
                from torch.optim.swa_utils import update_bn
                # Only run BN update if any BN-like modules exist; no-op otherwise.
                has_bn = any(
                    isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
                    for m in self.model.modules()
                )
                if has_bn:
                    update_bn(train_loader, swa_model, device=self.device)
                swa_ckpt_path = checkpoint_dir / f"safe_alert_{self.horizon}_swa.pt"
                torch.save(self._build_checkpoint_dict(
                    epoch=final_epoch,
                    model_state=swa_model.module.state_dict(),
                    extra={
                        "swa_start_epoch": swa_start_epoch,
                        "total_epochs":    final_epoch,
                    },
                ), swa_ckpt_path)
                print(f"[SWA] Saved averaged model: {swa_ckpt_path.name}")

                # ── SWA validation + policy recompute ────────────────────────
                # Threshold τ/γ and temperature must be tuned on the SAME
                # model that will be deployed. Without this step, per-epoch
                # τ/γ computed on the *regular* self.model would be applied
                # to the averaged SWA weights at inference — producing a
                # coverage and calibration mismatch. Temporarily swap SWA
                # weights into self.model, validate, capture policy, then
                # restore the original weights so the rest of the function
                # (last-epoch checkpoint, etc.) sees the unchanged run state.
                print("[SWA] Validating averaged model to fit SWA-specific τ/γ/T ...")
                original_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                try:
                    self.model.load_state_dict(swa_model.module.state_dict())
                    swa_val_metrics = self.validate(val_loader)
                    swa_model_score = float(swa_val_metrics.get("model_score", float("-inf")))
                    # Sprint 3.3-D-fix — snapshot SWA's training-time deploy
                    # metadata. SWA validates inline (its own re-fit of T on
                    # averaged weights), so this snapshot still represents the
                    # policy SWA actually won under, not a stale outer-loop
                    # re-validation.
                    swa_best_meta = self._make_selection_snapshot(epoch, swa_val_metrics)
                    print(
                        f"[SWA] Validation: score={swa_model_score:.4f}, "
                        f"F1={swa_val_metrics.get('macro_f1', 0.0):.4f}, "
                        f"ECE={swa_val_metrics.get('ece', 0.0):.4f}, "
                        f"Sharpe={swa_val_metrics.get('alert_sharpe', 0.0):.4f}, "
                        f"τ={swa_val_metrics.get('tau', 0.0):.4f}, "
                        f"γ={swa_val_metrics.get('gamma', 0.0):.4f}"
                    )
                finally:
                    self.model.load_state_dict(original_state)

                # Save SWA-specific policy so downstream live inference /
                # backtest uses thresholds tuned for the averaged model.
                if swa_val_metrics is not None:
                    swa_policy = {
                        "tau":         float(swa_val_metrics.get("tau", 0.0)),
                        "gamma":       float(swa_val_metrics.get("gamma", 0.0)),
                        "temperature": float(swa_val_metrics.get("temperature", 1.0)),
                        "coverage":    float(swa_val_metrics.get("alert_coverage", 0.0)),
                        "alert_precision": float(swa_val_metrics.get("alert_precision", 0.0)),
                        "alert_sharpe":    float(swa_val_metrics.get("alert_sharpe", 0.0)),
                        "score":       swa_model_score,
                        "source":      "swa_validation",
                        "swa_start_epoch": swa_start_epoch,
                        "symbol":      getattr(self, "symbol", "BTCUSDT"),
                        "horizon":     self.horizon,
                        "note": (
                            "Thresholds tuned on SWA-averaged weights; use with "
                            "the *_swa.pt or FINAL checkpoint when FINAL source=swa."
                        ),
                    }
                    swa_policy_path = checkpoint_dir / f"safe_alert_{self.horizon}_swa_policy.json"
                    with open(swa_policy_path, "w") as f:
                        json.dump(swa_policy, f, indent=2)
                    print(f"[SWA] Saved SWA policy: {swa_policy_path.name}")
            except Exception as exc:
                print(f"[SWA] WARNING: could not finalize SWA model — {exc}")
                swa_ckpt_path = None
                swa_val_metrics = None
                swa_model_score = float('-inf')

        stage3_score_std = (
            float(np.std(stage3_val_scores, ddof=1))
            if len(stage3_val_scores) >= 2
            else None
        )
        return self._finalize_training(
            checkpoint_dir, best_ckpt_path, best_overall_ckpt_path, last_ckpt_path,
            best_score, best_overall_score, final_epoch, final_val_loss, final_val_acc,
            swa_ckpt_path=swa_ckpt_path,
            swa_model_score=swa_model_score,
            stage3_score_std=stage3_score_std,
            best_overall_epoch=best_overall_epoch,  # R3 #K: stage-aware margin input
            total_epochs=epochs,
            # Sprint 3.3-D-fix — pass training-time deploy metadata snapshots
            # so the candidate that wins promotion can attach the metadata it
            # actually earned, not a re-validation artifact.
            best_overall_meta=best_overall_meta,
            stage3_best_meta=stage3_best_meta,
            swa_best_meta=swa_best_meta,
        )


def walk_forward_split(
    dataset_len: int,
    n_folds: int = 5,
    embargo_steps: int = 24,
    min_train_frac: float = 0.40,
):
    """Generate expanding-window walk-forward train/val/test splits with embargo.

    Each fold follows the PDF protocol:
      1) Train on early segment [0:t_k)
      2) Tune on next validation window
      3) Evaluate on immediately following test window
      4) Slide forward with expanding training history

    Layout per fold k:
      train_k -> embargo -> val_k -> embargo -> test_k

    Yields:
      (train_indices, val_indices, test_indices)
    """
    if dataset_len < 200:
        return

    min_train_end = max(1, int(dataset_len * min_train_frac))
    if min_train_end >= dataset_len:
        return

    remaining = dataset_len - min_train_end
    step_span = max(1, remaining // n_folds)

    for fold in range(n_folds):
        train_end = min_train_end + fold * step_span
        if train_end <= 1:
            continue

        usable_span = step_span - 2 * embargo_steps
        if usable_span < 20:
            continue

        val_win = max(10, usable_span // 2)
        test_win = max(10, usable_span - val_win)

        val_start = train_end + embargo_steps
        val_end = val_start + val_win
        test_start = val_end + embargo_steps
        test_end = test_start + test_win

        if test_end > dataset_len:
            break

        train_indices = list(range(0, train_end))
        val_indices = list(range(val_start, val_end))
        test_indices = list(range(test_start, test_end))

        if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
            continue

        # Session 22 Fix 15: intra-fold independence assertion. The index
        # arithmetic above enforces the layout train→embargo→val→embargo→
        # test, but a silent off-by-one in a future edit could let embargo
        # shrink below the intended separation. These assertions make any
        # such regression fail immediately instead of corrupting results.
        # Also enforces val_start ≥ train_end + embargo and test_start ≥
        # val_end + embargo — the two leak gates defined by the paper
        # (Section 4.2.3).
        assert val_start - train_end >= embargo_steps, (
            f"Fold {fold}: embargo between train and val broken "
            f"(gap={val_start - train_end} < {embargo_steps})."
        )
        assert test_start - val_end >= embargo_steps, (
            f"Fold {fold}: embargo between val and test broken "
            f"(gap={test_start - val_end} < {embargo_steps})."
        )
        assert not (set(train_indices) & set(val_indices)), (
            f"Fold {fold}: train and val indices overlap."
        )
        assert not (set(val_indices) & set(test_indices)), (
            f"Fold {fold}: val and test indices overlap."
        )
        assert not (set(train_indices) & set(test_indices)), (
            f"Fold {fold}: train and test indices overlap."
        )

        yield train_indices, val_indices, test_indices


def _load_config(config_path: Path) -> dict:
    """Load YAML config file. Returns empty dict if file not found."""
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[OK] Config loaded: {config_path.name}")
        return cfg
    return {}


def main():
    script_file = Path(__file__).resolve()
    # pipelines/ -> v2/ -> app/ -> ai-service/
    service_root = script_file.parent.parent.parent.parent
    default_training_data = service_root / "training_data"
    default_training_data_v2 = default_training_data / "v2"
    default_config = script_file.parent / "train_config_research_best.yaml"

    # ── Step 1: parse --config only (before loading other args) ──────────────
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=default_config)
    pre_args, _ = pre.parse_known_args()

    cfg = _load_config(pre_args.config)
    # P2 #19: push curriculum / policy values from YAML into module-level
    # constants BEFORE the trainer is instantiated.
    _apply_config_overrides(cfg)

    # ── Step 2: full parser — CLI values override config file ─────────────────
    parser = ArgumentParser(
        description="Train SAFE-Alert. Edit train_config_research_best.yaml to avoid long CLI args."
    )
    parser.add_argument("--config",          type=Path,  default=default_config)
    parser.add_argument("--symbol",          default=cfg.get("symbol", "BTCUSDT"))
    parser.add_argument("--horizon",         default=cfg.get("horizon", "1h"), choices=["15m", "1h", "4h", "24h"])
    parser.add_argument("--data_path",       type=Path,  default=default_training_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_training_data)
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--epochs",          type=int,   default=cfg.get("epochs", 60))
    parser.add_argument("--batch_size",      type=int,   default=cfg.get("batch_size", 8))
    parser.add_argument("--lookback_hours",  type=int,   default=cfg.get("lookback_hours", _LOOKBACK_HOURS))
    parser.add_argument("--lr",              type=float, default=cfg.get("lr", 3e-4))
    parser.add_argument("--weight_decay",    type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--grad_clip",       type=float, default=cfg.get("grad_clip", 1.0))
    parser.add_argument("--walk_forward",    action="store_true",
                        default=cfg.get("walk_forward", False),
                        help="Walk-forward cross-validation (PDF Eq.1)")
    # Session 23 P0 #3: paper Eq.16 bar-sequence mode. Paper-faithful default.
    # Using argparse.BooleanOptionalAction so CLI accepts BOTH
    # ``--use_bar_sequences`` AND ``--no-use_bar_sequences`` — previous
    # action="store_true" variant silently ignored attempts to disable via CLI
    # because default=True made --use_bar_sequences a no-op, and no negation
    # flag existed. BooleanOptionalAction requires Python ≥ 3.9 (we ship 3.11+).
    parser.add_argument("--use_bar_sequences",
                        action=argparse.BooleanOptionalAction,
                        default=bool(cfg.get("use_bar_sequences", True)),
                        help="Use per-timeframe bar sequences (paper Eq.16). "
                             "Requires market_bars.npz from precompute_market_bars.py. "
                             "Falls back to scalar mode if file is missing. "
                             "Pass --no-use_bar_sequences to force scalar mode for ablation.")
    parser.add_argument("--bar_seq_len", type=int,
                        default=cfg.get("bar_seq_len", 20),
                        help="L_δ (bars per timeframe) for paper Eq.16 sequences")
    parser.add_argument("--bar_feat_dim", type=int,
                        default=cfg.get("bar_feat_dim", 10),
                        help="d_δ (features per bar) for paper Eq.16 sequences")
    # Sprint 9 — three-way market input mode. Wins over --use_bar_sequences when
    # set to a non-default value. The default "auto" defers to the legacy bool
    # so existing configs/CLI invocations behave identically.
    parser.add_argument("--market_input_mode",
                        choices=["auto", "scalar", "bar", "hybrid"],
                        default=str(cfg.get("market_input_mode", "auto")),
                        help="Layer 3 market encoder mode: "
                             "'scalar' = engineered 63-dim only (OLD baseline); "
                             "'bar' = paper Eq.16 raw bar sequences only; "
                             "'hybrid' = gated residual scalar+bar (default safe "
                             "extension — gate inits to 0 so model starts at "
                             "scalar baseline and learns to mix in bars); "
                             "'auto' = fall back to --use_bar_sequences flag.")
    parser.add_argument("--num_workers", type=int, default=cfg.get("num_workers", 2),
                        help="DataLoader worker count (default: 2).")
    parser.add_argument("--pin_memory", action="store_true",
                        default=cfg.get("pin_memory", True),
                        help="Pin memory for faster host->GPU transfer.")
    parser.add_argument("--persistent_workers", action="store_true",
                        default=cfg.get("persistent_workers", True),
                        help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--prefetch_factor", type=int, default=cfg.get("prefetch_factor", 2),
                        help="DataLoader prefetch factor (default: 2).")
    parser.add_argument("--policy_method",
                        choices=["grid", "conformal"],
                        default=cfg.get("policy_method", "grid"),
                        help="Validation-only policy selection for tau/gamma (default: grid).")
    parser.add_argument("--tau_percentile", type=int,
                        default=cfg.get("tau_percentile", _TAU_PERCENTILE),
                        help="Percentile for tau seed/grid (default: 65). Lower => higher coverage.")
    parser.add_argument("--gamma_percentile", type=int,
                        default=cfg.get("gamma_percentile", _GAMMA_PERCENTILE),
                        help="Percentile for gamma seed/grid (default: 55). Lower => higher coverage.")
    parser.add_argument("--policy_confidence_source",
                        choices=["raw", "position"],
                        default=cfg.get("policy_confidence_source", _POLICY_CONFIDENCE_SOURCE),
                        help="Confidence stream for Eq.26 tau gate: raw conf_head or "
                             "position confidence c*(1-p_neutral).")
    parser.add_argument("--n_folds",         type=int,   default=cfg.get("n_folds", 5))
    parser.add_argument("--embargo_steps",   type=int,   default=cfg.get("embargo_steps", 24),
                        help="Samples to skip between train end and val start (embargo, "
                             "default 24 = 24h for 1h data). For 15m data use 96 (=24h). "
                             "Prevents look-ahead bias.")
    parser.add_argument("--ablation", type=str, default=cfg.get("ablation", None),
                        help="Ablation study: disable a model component (PDF Section 4.6). "
                             "Choices: 'w/o_faithfulness' (λ6=0 always), "
                             "'w/o_factor' (λ3=0 always). "
                             "Default: None (full model).")
    parser.add_argument("--explain_every", type=int, default=cfg.get("explain_every", 5),
                        help="Compute full explanation metrics (Del/Ins/Suff/Comp) every "
                             "N epochs during training. 1 = every epoch (old behavior, "
                             "slow), 5 = 5× faster validation with minimal signal loss. "
                             "Final test-set evaluation always computes explanations.")

    # Device: config "auto" → detect; explicit "cuda"/"cpu" → use as-is
    _cfg_device = cfg.get("device", "auto")
    _default_device = ("cuda" if torch.cuda.is_available() else "cpu") if _cfg_device == "auto" else _cfg_device
    parser.add_argument("--device", default=_default_device)

    args = parser.parse_args()

    # Sprint 9 — reconcile --market_input_mode with --use_bar_sequences. The
    # bar-cache loading downstream is gated by ``args.use_bar_sequences``; for
    # hybrid mode we still need bars loaded, so reflect the explicit mode here.
    # When --market_input_mode="auto" we leave args.use_bar_sequences untouched
    # (legacy behaviour).
    _mim = getattr(args, "market_input_mode", "auto")
    if _mim == "scalar":
        args.use_bar_sequences = False
    elif _mim in ("bar", "hybrid"):
        args.use_bar_sequences = True
    # "auto" → keep whatever --use_bar_sequences resolved to.
    print(f"  Market input mode: {_mim} (use_bar_sequences={args.use_bar_sequences})")

    # Auto-prefer training_data/v2 when present unless user explicitly set a path.
    if (
        args.data_path == default_training_data
        and default_training_data_v2.exists()
    ):
        args.data_path = default_training_data_v2
    if (
        args.embeddings_path == default_training_data
        and default_training_data_v2.exists()
    ):
        args.embeddings_path = default_training_data_v2

    print(f"[START] Training SAFE-Alert {args.symbol} {args.horizon}")
    print(f"  Device: {args.device}")
    print(f"  Data path exists: {args.data_path.exists()} ({args.data_path})")

    # Load data (try candles_max.csv first, then others)
    candles_max = args.data_path / "candles_max.csv"
    data_path = args.data_path / f"{args.symbol.lower()}_training_dataset_v2.csv"
    aligned_path = args.data_path / "candles_aligned.csv"
    v2_symbol = args.symbol.upper()
    v2_h_path = args.data_path / f"{v2_symbol}_{args.horizon}_ohlcv.csv"

    if v2_h_path.exists():
        data_path = v2_h_path
        print(f"[OK] Using v2 candles: {data_path.name}")
    elif candles_max.exists():
        data_path = candles_max
        print(f"[OK] Using candles_max: {data_path.name}")
    elif aligned_path.exists():
        data_path = aligned_path
        print(f"[OK] Using aligned candles: {data_path.name}")
    elif not data_path.exists():
        print(f"[ERROR] Data not found: candles_max.csv, {data_path.name} or candles_aligned.csv")
        sys.exit(1)

    candle_df = pd.read_csv(data_path)
    candle_df = _ensure_timestamp_column(candle_df)
    if "timestamp" not in candle_df.columns:
        raise KeyError("timestamp")
    candle_df["timestamp"] = pd.to_datetime(candle_df["timestamp"], errors="coerce")

    # Note: previously we capped candles to 2023-12-31 to match article coverage.
    # For synchronized 2018-2025 datasets, we keep the full range.

    print(f"[OK] Loaded {len(candle_df)} candles")

    # Load embeddings (try MAX first, then others)
    emb_max = args.embeddings_path / "btcusdt_article_embeddings_max.npy"
    emb_path = args.embeddings_path / f"{args.symbol.lower()}_article_embeddings.npy"

    if emb_max.exists():
        emb_path = emb_max
        print(f"[OK] Using embeddings_max: {emb_path.name}")
    elif not emb_path.exists():
        # Try without symbol prefix
        emb_path = args.embeddings_path / "article_embeddings.npy"

    if not emb_path.exists():
        # Auto-generate embeddings if missing
        print(f"[GENERATE] Embeddings missing, generating from articles_max.csv...")
        articles_max = args.embeddings_path / "articles_max.csv"
        if articles_max.exists():
            print(f"[EMBED] Loading articles ({articles_max.name})...")
            articles_df = pd.read_csv(articles_max)

            from transformers import AutoTokenizer, AutoModel
            # Use FinBERT (financial domain) as specified in PDF Section 3
            try:
                _emb_model_name = "ProsusAI/finbert"
                tokenizer = AutoTokenizer.from_pretrained(_emb_model_name)
                model = AutoModel.from_pretrained(_emb_model_name)
                print(f"[EMBED] Using FinBERT ({_emb_model_name})")
            except Exception as _exc:
                # Session 23 P1 #7 fix: log the cause of fallback so users see
                # WHY FinBERT wasn't used (network issue, HF cache, etc.). The
                # bert-base fallback changes domain coverage, which matters for
                # reported metrics — silent fallback could misattribute bad
                # results to the model instead of the embedding swap.
                print(f"[EMBED] FinBERT load failed ({type(_exc).__name__}: {_exc})")
                _emb_model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(_emb_model_name)
                model = AutoModel.from_pretrained(_emb_model_name)
                print(f"[EMBED] Falling back to {_emb_model_name} — metrics may shift")
            device = args.device
            model = model.to(device).eval()

            embeddings_list = []
            print(f"[EMBED] Computing {len(articles_df)} embeddings...")
            for i in range(0, len(articles_df), 64):
                texts = articles_df.iloc[i:i+64]['title'].astype(str) + " " + articles_df.iloc[i:i+64]['content'].astype(str)
                texts = [str(t)[:512] for t in texts]
                inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                    embeddings_list.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())
                pct = min(i+64, len(articles_df)) / len(articles_df) * 100
                print(f"  {pct:.0f}%", end="\r", flush=True)

            embeddings = np.vstack(embeddings_list)
            emb_path = args.embeddings_path / "btcusdt_article_embeddings_max.npy"
            np.save(emb_path, embeddings)
            print(f"\n[OK] Generated: {emb_path.name} {embeddings.shape}\n")
        else:
            print(f"[ERROR] Embeddings not found and articles_max.csv not found")
            sys.exit(1)

    embeddings = np.load(emb_path)
    print(f"[OK] Loaded embeddings {embeddings.shape} from {emb_path.name}")

    # Load precomputed market features (if available) -  OPTIMIZATION
    precomputed_features = None
    precomp_path = args.embeddings_path / "features_precomputed.npy"
    if precomp_path.exists():
        precomputed_features = np.load(precomp_path)
        precomp_meta_path = args.embeddings_path / "features_precomputed.meta.json"
        precomp_usable = True
        if precomp_meta_path.exists():
            try:
                with open(precomp_meta_path, "r", encoding="utf-8") as f:
                    pre_meta = json.load(f)
                source_file = str(pre_meta.get("source_file", "") or "")
                if source_file and source_file != data_path.name:
                    print(
                        "[WARN] features_precomputed.npy was built from "
                        f"{source_file}, but current candles are {data_path.name}; "
                        "ignoring this cache to avoid cross-horizon feature misalignment."
                    )
                    precomputed_features = None
                    precomp_usable = False
                meta_rows = int(pre_meta.get("rows", -1))
                if precomp_usable and meta_rows > 0 and meta_rows != len(precomputed_features):
                    raise ValueError(
                        "features_precomputed.meta.json row count mismatch with .npy: "
                        f"meta={meta_rows}, npy={len(precomputed_features)}. Recompute precomputed features."
                    )
                ts_min_meta = pd.to_datetime(pre_meta.get("timestamp_min"), errors="coerce")
                ts_min_data = pd.to_datetime(candle_df["timestamp"].min(), errors="coerce")
                if precomp_usable and pd.notna(ts_min_meta) and pd.notna(ts_min_data) and ts_min_meta != ts_min_data:
                    print(
                        "[WARN] precompute timestamp_min differs from current candles; "
                        "consider recomputing features_precomputed.npy for strict reproducibility."
                    )
            except Exception as e:
                print(f"[WARN] Could not validate precompute metadata: {e}")
        else:
            print("[WARN] features_precomputed.meta.json not found; alignment checks are limited.")
            if args.horizon != "1h":
                print(
                    "[WARN] Non-1h horizon with metadata-less features_precomputed.npy; "
                    "ignoring cache to avoid accidental 1h->4h/24h truncation."
                )
                precomputed_features = None
                precomp_usable = False

        if precomp_usable and len(precomputed_features) != len(candle_df):
            if len(precomputed_features) > len(candle_df):
                print(
                    "[WARN] features_precomputed.npy has more rows than filtered candles; "
                    f"truncating {len(precomputed_features)} -> {len(candle_df)} to keep alignment."
                )
                precomputed_features = precomputed_features[:len(candle_df)]
            else:
                raise ValueError(
                    "features_precomputed.npy is shorter than filtered candles: "
                    f"got {len(precomputed_features)} rows, expected {len(candle_df)}. "
                    "Re-run precompute_market_features.py."
                )
        if precomp_usable:
            print(f"[OK] Loaded precomputed features {precomputed_features.shape} from {precomp_path.name}")
            print(f"     [SPEEDUP] Training will be 5-10x faster!")
        else:
            print("[WARN] Precomputed market features disabled for this run; training will compute features on the fly.")
    else:
        print(f"[WARN] Precomputed features not found: {precomp_path.name}")
        print(f"       Train will be slow (~1-2 hours per epoch on CPU)")
        print(f"       Run: python precompute_market_features.py  (one-time, ~10 min)")

    # Load articles (try articles_max.csv first, then articles.csv)
    articles_max = args.embeddings_path / "articles_max.csv"
    articles_path = args.embeddings_path / "articles.csv"

    if articles_max.exists():
        articles_path = articles_max
        print(f"[OK] Using articles_max: {articles_path.name}")
    elif not articles_path.exists():
        print(f"[ERROR] Articles not found: articles_max.csv or articles.csv (checked in embeddings_path)")
        sys.exit(1)

    articles_df = pd.read_csv(articles_path)
    print(f"[OK] Loaded {len(articles_df)} articles")

    # Load precomputed factor labels (if available) — fixes Lfac stuck at log(10)=2.303
    factor_labels_np = None
    factor_labels_path = args.embeddings_path / "article_factor_labels.npy"
    if factor_labels_path.exists():
        factor_labels_np = np.load(factor_labels_path)
        print(f"[OK] Loaded precomputed factor labels {factor_labels_np.shape} "
              f"from {factor_labels_path.name}")
    else:
        print(f"[WARN] Factor labels not found: {factor_labels_path.name}")
        print(f"       Run: python precompute_factor_labels.py")

    # Load target-based FSA entity sentiment (if available)
    # Generated by precompute_entity_sentiment.py using ProsusAI/finbert
    # Shape: (N_articles, 10) — per-factor sentiment in [-1, +1]
    entity_sentiment_np = None
    entity_sentiment_path = args.embeddings_path / "article_entity_sentiment.npy"
    if entity_sentiment_path.exists():
        entity_sentiment_np = np.load(entity_sentiment_path)
        print(f"[OK] Loaded entity sentiment (target-based FSA) {entity_sentiment_np.shape} "
              f"from {entity_sentiment_path.name}")
        print(f"     META_DIM will be 14 (4 base + 10 FSA)")
    else:
        print(f"[INFO] Entity sentiment not found — using base metadata only (META_DIM=4)")
        print(f"       Optionally run: python precompute_entity_sentiment.py")

    # Session 23 P0 #3: load per-timeframe bar sequences for paper Eq.16.
    # market_bars.npz is produced by precompute_market_bars.py and holds
    # 5 float32 arrays of shape (N_candles, L_δ, d_δ) — one per timeframe
    # in {1m, 5m, 15m, 1h, 4h}. Required when use_bar_sequences=True; when
    # the file is missing we fall back to the legacy scalar path so old
    # runs still work while users migrate to the paper-faithful pipeline.
    market_bars_dict = None
    market_bars_path = None
    if args.use_bar_sequences:
        market_bars_candidates = list(dict.fromkeys([
            args.embeddings_path / f"market_bars_{args.horizon}.npz",
            args.embeddings_path / "market_bars.npz",
        ]))
        stale_market_bars = []
        for candidate in market_bars_candidates:
            if not candidate.exists():
                continue
            with np.load(candidate) as _loaded:
                candidate_dict = {k: _loaded[k] for k in _loaded.files}
            _required_bar_keys = {f"bars_{tf}" for tf in ("1m", "5m", "15m", "1h", "4h")}
            _missing_bar_keys = _required_bar_keys - set(candidate_dict)
            if _missing_bar_keys:
                print(f"[WARN] Ignoring malformed market_bars cache {candidate.name}: "
                      f"missing {sorted(_missing_bar_keys)}")
                continue
            _shape = candidate_dict["bars_1m"].shape
            if _shape[0] != len(candle_df):
                stale_market_bars.append((candidate, _shape[0]))
                print(f"[WARN] Ignoring stale market_bars cache {candidate.name}: "
                      f"N={_shape[0]} != candles={len(candle_df)} for horizon={args.horizon}")
                continue
            if args.horizon != "1h":
                _meta_interval = candidate_dict.get("decision_interval_ns")
                if _meta_interval is None:
                    raise ValueError(
                        f"{candidate.name} was generated by an older precompute_market_bars.py "
                        f"without decision_interval metadata. Non-1h bar caches previously used "
                        f"a hard-coded +1h decision close and may be temporally misaligned. "
                        f"Regenerate it with: python app/v2/pipelines/precompute_market_bars.py "
                        f"--data-dir {args.embeddings_path} --symbol {args.symbol} "
                        f"--decision-horizon {args.horizon}"
                    )
                _ts_col = "timestamp" if "timestamp" in candle_df.columns else "datetime"
                _ts = pd.to_datetime(candle_df[_ts_col], utc=True, errors="coerce").dropna().sort_values()
                if len(_ts) >= 2:
                    _expected_ns = int(_ts.diff().dropna().median().value)
                    _actual_ns = int(np.asarray(_meta_interval).reshape(-1)[0])
                    if abs(_actual_ns - _expected_ns) > max(1, int(0.1 * _expected_ns)):
                        raise ValueError(
                            f"{candidate.name} decision_interval_ns={_actual_ns} does not match "
                            f"horizon={args.horizon} candle interval {_expected_ns}. Regenerate "
                            f"market_bars with --decision-horizon {args.horizon}."
                        )
            market_bars_path = candidate
            market_bars_dict = candidate_dict
            print(f"[OK] Loaded market_bars (paper Eq.16) from {market_bars_path.name} — "
                  f"5 TFs × {_shape[0]} candles × {_shape[1]} bars × {_shape[2]} features")
            break
        if market_bars_dict is None:
            if stale_market_bars:
                stale_desc = ", ".join(f"{p.name}:N={n}" for p, n in stale_market_bars)
                raise ValueError(
                    f"No market_bars cache matches horizon={args.horizon} "
                    f"(expected N={len(candle_df)}; stale: {stale_desc}). "
                    f"Run: python app/v2/pipelines/precompute_market_bars.py "
                    f"--data-dir {args.embeddings_path} --symbol {args.symbol} "
                    f"--decision-horizon {args.horizon}"
                )
            print(f"[WARN] use_bar_sequences=True but no horizon-compatible market_bars cache found "
                  f"at {args.embeddings_path}.")
            print(f"       Run: python app/v2/pipelines/precompute_market_bars.py "
                  f"--data-dir {args.embeddings_path} --symbol {args.symbol} "
                  f"--decision-horizon {args.horizon}")
            print(f"       Falling back to scalar mode (Contribution 3 remains simplified).")
            # Disable the flag so downstream model + dataset behave consistently.
            args.use_bar_sequences = False
            # Sprint 9 — also downgrade hybrid → scalar when bars missing. The
            # hybrid encoder's graceful-degradation path (gate residual = 0) would
            # work, but it is cleaner to construct a scalar-only model so the
            # bar sub-graph parameters are never created and the optimizer does
            # not waste capacity on dead weights.
            if getattr(args, "market_input_mode", "auto") in ("bar", "hybrid"):
                print(f"       Downgrading market_input_mode={args.market_input_mode} → 'scalar'.")
                args.market_input_mode = "scalar"

    # Session 24 — paper Section 4.1.3 TF-IDF novelty. article_novelty.npy
    # (one float32 per article in [0,1]) is produced by precompute_novelty.py
    # and replaces the rank_norm placeholder for meta[i, 3]. When absent the
    # dataset transparently falls back to rank_norm (documented deviation D4).
    article_novelty_np = None
    article_novelty_path = args.embeddings_path / "article_novelty.npy"
    if article_novelty_path.exists():
        article_novelty_np = np.load(article_novelty_path)
        print(f"[OK] Loaded article_novelty (paper Section 4.1.3 TF-IDF) "
              f"shape={article_novelty_np.shape} from {article_novelty_path.name}")
    else:
        print(f"[INFO] article_novelty.npy not found — falling back to rank_norm "
              f"for META[3] (deviation D4). Run precompute_novelty.py to enable "
              f"paper-faithful TF-IDF novelty.")

    # Create dataset with precomputed features (or None for fallback)
    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=args.horizon,
        lookback_hours=args.lookback_hours,
        articles_per_candle=_ARTICLES_PER_CANDLE,  # Sprint 1 FIX 2 — was silently
                                                   # defaulting to 8, now YAML-wired
        precomputed_features=precomputed_features,
        factor_labels=factor_labels_np,
        entity_sentiment=entity_sentiment_np,
        market_bars=market_bars_dict,   # Session 23 P0 #3: paper Eq.16
        article_novelty=article_novelty_np,  # Session 24: paper Section 4.1.3
        ingest_delay_minutes=_INGEST_DELAY_MIN,  # Session 26 — was ignored, now YAML-wired
        epsilon_h_override=_EPSILON_H_OVERRIDE,  # Sprint 4 Phase 8.5 — paired with loss_fn
    )

    # ── Walk-forward cross-validation (PDF Eq.1 + embargo) ───────────────────
    if args.walk_forward:
        n_folds       = args.n_folds
        embargo_steps = args.embargo_steps
        dataset_len   = len(dataset)

        print(f"[WALK-FORWARD] {n_folds} folds | embargo={embargo_steps} steps | "
              f"dataset={dataset_len}")
        print("[WALK-FORWARD] Protocol: train(expanding) -> val -> test (per fold)")
        print()

        # Temporary reference model to get K_h (instantiated once for logging).
        # Session 26: YAML-aware K values + explicit K_24h + n_factors for
        # parity with the real fold_model constructed inside the loop.
        _ref_model = SAFEAlertNet(
            market_dim=63, has_news=True,
            K_15m=_TOP_K_MAP["15m"], K_1h=_TOP_K_MAP["1h"],
            K_4h=_TOP_K_MAP["4h"],   K_24h=_TOP_K_MAP["24h"],
            n_factors=FACTOR_CLASSES,
            use_bar_sequences=args.use_bar_sequences,
            market_input_mode=(None if getattr(args, "market_input_mode", "auto") == "auto"
                               else args.market_input_mode),
            bar_seq_len=args.bar_seq_len,
            bar_feat_dim=args.bar_feat_dim,
        )
        K_h = _ref_model.K_h_map.get(args.horizon, 8)
        del _ref_model

        fold_metrics_list = []

        folds = list(walk_forward_split(
            dataset_len,
            n_folds=n_folds,
            embargo_steps=embargo_steps,
        ))
        for fold_idx, (train_indices, val_indices, test_indices) in enumerate(folds, start=1):
            print(f"[FOLD {fold_idx}/{len(folds)}] "
                  f"train=0-{train_indices[-1]} ({len(train_indices)} samples) | "
                  f"embargo={embargo_steps} steps | "
                  f"val={val_indices[0]}-{val_indices[-1]} "
                  f"({len(val_indices)} samples) | "
                  f"test={test_indices[0]}-{test_indices[-1]} ({len(test_indices)} samples)")

            # Session 26 — reset RNG per fold so each fold's random state is
            # reproducible independently of preceding folds. Without this, the
            # global seed (line 53-54) was set once at import time and each
            # fold's shuffle/dropout/init state depended on ALL prior folds.
            # That meant "rerun fold 5 alone" gave different results than
            # "fold 5 inside a walk-forward". Seed = 42 + fold_idx so each
            # fold has a distinct but reproducible RNG state.
            _fold_seed = 42 + fold_idx
            torch.manual_seed(_fold_seed)
            np.random.seed(_fold_seed)
            _LOADER_GENERATOR.manual_seed(_fold_seed)

            dataset.fit_market_scaler(train_indices)

            fold_train_set = Subset(dataset, train_indices)
            fold_val_set   = Subset(dataset, val_indices)
            fold_test_set  = Subset(dataset, test_indices)

            # Sprint 7 — balanced batch sampler (train fold only). Toggle via
            # YAML key `balanced_batch_sampler`. False (default) preserves
            # legacy shuffle=True behaviour; True replaces it with
            # BalancedBatchSampler that ensures every batch has ~ceil(B/3)
            # samples per direction class.
            fold_train_sampler = None
            if _BALANCED_BATCH_SAMPLER:
                _all_dir_labels = dataset.get_direction_labels()
                _train_dir_labels = _all_dir_labels[np.asarray(train_indices, dtype=np.int64)]
                fold_train_sampler = BalancedBatchSampler(
                    labels=_train_dir_labels,
                    batch_size=args.batch_size,
                    num_classes=3,
                    seed=_fold_seed,
                )
                print(
                    f"[Sampler] balanced_batch_sampler enabled (fold {fold_idx}): "
                    f"batch_size={fold_train_sampler.batch_size}, "
                    f"class_counts={fold_train_sampler.class_counts}, "
                    f"base_quota={fold_train_sampler.base_quota}, "
                    f"batches_per_epoch={len(fold_train_sampler)}",
                    flush=True,
                )

            _train_loader_kwargs = dict(
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                worker_init_fn=_seed_worker,       # P2 #21: reproducible workers
                generator=_LOADER_GENERATOR,
            )
            if fold_train_sampler is not None:
                fold_train_loader = torch.utils.data.DataLoader(
                    fold_train_set,
                    batch_sampler=fold_train_sampler,
                    **_train_loader_kwargs,
                )
            else:
                fold_train_loader = torch.utils.data.DataLoader(
                    fold_train_set,
                    batch_size=args.batch_size,
                    shuffle=True,
                    **_train_loader_kwargs,
                )
            fold_val_loader = torch.utils.data.DataLoader(
                fold_val_set,
                batch_size=args.batch_size,
                shuffle=False,   # P1 #14: time-ordered val/test evaluation
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
            fold_test_loader = torch.utils.data.DataLoader(
                fold_test_set,
                batch_size=args.batch_size,
                shuffle=False,   # P1 #14: time-ordered val/test evaluation
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )

            # Fresh model + trainer for each fold (no state bleed between folds).
            # Session 26: explicit n_factors + K_h from YAML `top_k` block
            # (previously hardcoded — users who edited YAML were ignored).
            fold_model = SAFEAlertNet(
                market_dim=63, has_news=True,
                K_15m=_TOP_K_MAP["15m"], K_1h=_TOP_K_MAP["1h"],
                K_4h=_TOP_K_MAP["4h"],   K_24h=_TOP_K_MAP["24h"],
                n_factors=FACTOR_CLASSES,
                use_bar_sequences=args.use_bar_sequences,
                market_input_mode=(None if getattr(args, "market_input_mode", "auto") == "auto"
                                   else args.market_input_mode),
                bar_seq_len=args.bar_seq_len,
                bar_feat_dim=args.bar_feat_dim,
                predict_volatility=(_LAMBDA_VOL > 0.0),
                predict_ret_sign=(_LAMBDA_RET_SIGN_CLS > 0.0),
                predict_ret_bin=(_LAMBDA_RET_BIN > 0.0),
                predict_edge=(_LAMBDA_UP_EDGE > 0.0 or _LAMBDA_DOWN_EDGE > 0.0),
            )
            fold_trainer = SAFEAlertTrainer(
                fold_model,
                device=args.device,
                lr=args.lr,
                weight_decay=args.weight_decay,
                horizon=args.horizon,
                K_h=K_h,
                grad_clip=args.grad_clip,
                policy_method=args.policy_method,
                tau_percentile=args.tau_percentile,
                gamma_percentile=args.gamma_percentile,
                policy_confidence_source=args.policy_confidence_source,
                ablation=args.ablation,
                symbol=args.symbol,
            )

            fold_ckpt_dir = args.artifact_dir / f"fold_{fold_idx}"
            fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

            fold_metrics, fold_ckpt = fold_trainer.fit(
                fold_train_loader, fold_val_loader,
                epochs=args.epochs,
                checkpoint_dir=fold_ckpt_dir,
                early_stopping_patience=_ES_PATIENCE,
                explain_every=args.explain_every,
                dataset=dataset,   # P0 #1: snapshot preprocessing state into checkpoint
            )

            # Evaluate best/final checkpoint on fold test window.
            # PyTorch 2.6 defaults torch.load(..., weights_only=True).
            # Our checkpoints include extra Python objects, so force full load
            # for trusted local artifacts generated in this training run.
            ckpt_state = torch.load(fold_ckpt, map_location=args.device, weights_only=False)
            fold_trainer.model.load_state_dict(ckpt_state["model_state"])

            # Tune Eq.26 thresholds on validation split only, then freeze for test.
            # This avoids optimistic bias from deriving thresholds on test itself.
            val_eval_metrics = fold_trainer.validate(fold_val_loader)

            # Sprint 3.3-F — extract training-time selection_meta NOW (before
            # test eval) so test_metrics can use the policy that earned the
            # checkpoint its score. Pre-3.3-F, test eval used tau/gamma/T from
            # ``val_eval_metrics`` (re-validation), where fit_temperature_scaling
            # commonly clamps T to 2.500 / 0.500 and the policy search collapses
            # to no-trade — producing test_alert_sharpe = 0.000 ± 0.000 even
            # when the [BEST] gate had Sharpe = -0.174 at training time. Using
            # the snapshot policy makes test eval honest: it answers "what
            # would ep N's policy do on test?", not "what does a fresh refit
            # on uncalibrated logits do?".
            final_meta = fold_metrics.get("selection_meta") or {}

            def _policy_meta_float(meta: dict, key: str, fallback: float,
                                   lo: float, hi: float) -> float:
                """Read ``key`` from snapshot meta with range validation.

                ``dict.get(key, fallback)`` only fires the fallback when the
                key is MISSING — not when the value is 0.0 / NaN / inf or
                outside the legal range. Snapshots can carry 0.0 from missing
                training-time fields, so a strict numeric guard is needed:
                fall back to ``fallback`` on parse failure, non-finite, or
                out-of-band values.
                """
                v = meta.get(key, None)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return float(fallback)
                if not np.isfinite(v) or v < lo or v > hi:
                    return float(fallback)
                return v

            # Range bounds: tau/gamma are sigmoid/softmax outputs in [0, 1].
            # Temperature in [0.05, 20] gives a wide margin above the
            # current clamp band (0.500–2.500) so a legitimately learned T
            # like 0.978 passes through; values outside this band almost
            # certainly mean fit failure.
            _tau = _policy_meta_float(
                final_meta, "tau",
                val_eval_metrics.get("tau", 0.7), 0.0, 1.0,
            )
            _gamma = _policy_meta_float(
                final_meta, "gamma",
                val_eval_metrics.get("gamma", 0.65), 0.0, 1.0,
            )
            _temperature = _policy_meta_float(
                final_meta, "temperature",
                val_eval_metrics.get("temperature", 1.0), 0.05, 20.0,
            )

            test_metrics = fold_trainer.validate(
                fold_test_loader,
                tau=_tau,
                gamma=_gamma,
                temperature=_temperature,
            )

            # Session 24: export HumanUsefulness case studies (paper Section 4.4.2).
            # 50 sampled test predictions with full context (articles, factors,
            # confidence, ground-truth, faithfulness gap) for qualitative review.
            # Fires once per fold so the thesis has 3× 50 = 150 case studies.
            try:
                from case_study_export import export_case_studies
                # Sprint 3.3-F — case studies use snapshot policy too. Without
                # this, prediction_logs.jsonl and test_metrics carry the
                # snapshot policy while case_studies.json carries the T-clamped
                # re-validation policy — three audit artifacts diverge in their
                # threshold provenance. Keeping all three on the same policy
                # makes downstream replay / qualitative review consistent.
                _cs_path = export_case_studies(
                    fold_trainer.model, fold_test_loader, dataset,
                    symbol=args.symbol, horizon=args.horizon,
                    n_samples=50,
                    out_path=fold_ckpt_dir / "case_studies.json",
                    tau_h=_tau,
                    gamma_h=_gamma,
                    temperature_h=_temperature,
                    policy_confidence_source=str(final_meta.get(
                        "policy_confidence_source", args.policy_confidence_source
                    )),
                )
                print(f"[OK] Case studies exported for fold {fold_idx}: {_cs_path}")
            except Exception as _cs_exc:
                # Don't fail the whole fold on a case-study glitch — the run
                # has already produced metrics; qualitative export is auxiliary.
                import warnings as _w
                _w.warn(
                    f"[fold {fold_idx}] Case-study export failed: "
                    f"{type(_cs_exc).__name__}: {_cs_exc}",
                    RuntimeWarning, stacklevel=2,
                )

            # Session 25: export full per-sample prediction logs (paper
            # Section 4.1.1 third data source — AI Service / Notification /
            # Core prediction logs). Unlike case studies (50 sampled rows),
            # this dump contains EVERY test-set prediction in JSONL with the
            # same schema the production Notification Service would emit,
            # enabling audit / replay / retraining signal workflows.
            try:
                from prediction_logs import export_prediction_logs
                # Sprint 3.3-F — prediction logs use snapshot policy (same
                # tau/gamma/T as test eval and fold_record). Without this,
                # the audit JSONL would carry the re-validated policy while
                # test_metrics carries the snapshot policy — a divergence
                # that would silently make audit replay fail to reproduce
                # the cross-fold test_alert_sharpe numbers.
                _pl_path = export_prediction_logs(
                    fold_trainer.model, fold_test_loader, dataset,
                    symbol=args.symbol, horizon=args.horizon,
                    fold=fold_idx,
                    out_path=fold_ckpt_dir / "prediction_logs.jsonl",
                    tau_h=_tau,
                    gamma_h=_gamma,
                    temperature_h=_temperature,
                    policy_confidence_source=str(final_meta.get(
                        "policy_confidence_source", args.policy_confidence_source
                    )),
                )
                print(f"[OK] Prediction logs exported for fold {fold_idx}: {_pl_path}")
            except Exception as _pl_exc:
                import warnings as _w
                _w.warn(
                    f"[fold {fold_idx}] Prediction-logs export failed: "
                    f"{type(_pl_exc).__name__}: {_pl_exc}",
                    RuntimeWarning, stacklevel=2,
                )

            # Sprint 3.3-D-fix + 3.3-F — fold_record reads from final_meta
            # (extracted earlier, before test_metrics). selection_meta carries
            # the exact numbers from the moment the [BEST] gate fired during
            # training. Trade / sharpe / deploy / policy fields below source
            # from final_meta; macro_f1 / ECE / AUC / attention / explanations
            # stay sourced from val_eval_metrics because they ARE deterministic
            # from the weights and reproduce identically on re-validation.
            # ``final_meta`` is already a dict (extracted with ``or {}`` above).

            fold_record = {
                "fold": fold_idx,
                "train_range": [train_indices[0], train_indices[-1]],
                "val_range": [val_indices[0], val_indices[-1]],
                "test_range": [test_indices[0], test_indices[-1]],
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "test_size": len(test_indices),
                "fit": fold_metrics,
                "val": {
                    "final_val_loss": float(fold_metrics.get("final_val_loss", 0.0)),
                    "final_val_acc": float(fold_metrics.get("final_val_acc", 0.0)),
                    # Sprint 3.3-F — tau/gamma/temperature routed from snapshot
                    # via _policy_meta_float (range-validated). These three
                    # flow into utils.standardize_walk_forward_artifacts
                    # (deploy policy artifact) via best_val.get("tau"), so
                    # routing them correctly here is what fixes the policy
                    # JSON consumer side, not just the fold_record itself.
                    "tau": _tau,
                    "gamma": _gamma,
                    "temperature": _temperature,
                    "tau_seed": float(final_meta.get("tau_seed", val_eval_metrics.get("tau_seed", 0.0))),
                    "gamma_seed": float(final_meta.get("gamma_seed", val_eval_metrics.get("gamma_seed", 0.0))),
                    "temperature_nll": float(val_eval_metrics.get("temperature_nll", 0.0)),
                    "temperature_source": str(final_meta.get("temperature_source", val_eval_metrics.get("temperature_source", "unknown"))),
                    # Session 22 Fix 12: bubble the suspect flag into fold
                    # metrics JSON. Downstream aggregation + thesis dashboards
                    # can scan for ``temperature_suspect=True`` across folds
                    # to identify which runs have unreliable post-hoc
                    # calibration.
                    "temperature_suspect": bool(val_eval_metrics.get("temperature_suspect", False)),
                    "temperature_unclamped": float(val_eval_metrics.get("temperature_unclamped", val_eval_metrics.get("temperature", 1.0))),
                    # Sprint 3.3-F — policy provenance routed from snapshot.
                    # policy_source tells consumers WHICH grid branch won
                    # (e.g., "validation_grid_cal_sel_split" vs a fallback
                    # like "highest-trade-coverage"); policy_objective is
                    # the objective value at that policy. Both are audit
                    # signals: a deployable=True artifact whose policy_source
                    # is a fallback branch deserves extra scrutiny.
                    "policy_source": str(final_meta.get("policy_source", val_eval_metrics.get("policy_source", "unknown"))),
                    "policy_confidence_source": str(final_meta.get(
                        "policy_confidence_source",
                        val_eval_metrics.get("policy_confidence_source", args.policy_confidence_source),
                    )),
                    "policy_objective": float(final_meta.get("policy_objective", val_eval_metrics.get("policy_objective", 0.0) or 0.0)),
                    "coverage_penalty": float(final_meta.get("coverage_penalty", val_eval_metrics.get("coverage_penalty", 0.0) or 0.0)),
                    "under_trade_cov": float(final_meta.get("under_trade_cov", val_eval_metrics.get("under_trade_cov", 0.0))),
                    "action_bonus_policy": float(final_meta.get("action_bonus_policy", val_eval_metrics.get("action_bonus_policy", 0.0))),
                    "neutsh_penalty_policy": float(final_meta.get("neutsh_penalty_policy", val_eval_metrics.get("neutsh_penalty_policy", 0.0))),
                    "action_quality_penalty_policy": float(final_meta.get(
                        "action_quality_penalty_policy",
                        val_eval_metrics.get("action_quality_penalty_policy", 0.0),
                    )),
                    "cost_volume_penalty": float(final_meta.get("cost_volume_penalty", val_eval_metrics.get("cost_volume_penalty", 0.0))),
                    "imbalance_penalty": float(final_meta.get("imbalance_penalty", val_eval_metrics.get("imbalance_penalty", 0.0))),
                    "cost_volume_penalty_diag": float(final_meta.get(
                        "cost_volume_penalty_diag",
                        val_eval_metrics.get("cost_volume_penalty_diag", 0.0),
                    )),
                    "imbalance_penalty_diag": float(final_meta.get(
                        "imbalance_penalty_diag",
                        val_eval_metrics.get("imbalance_penalty_diag", 0.0),
                    )),
                    "directional_imbalance": float(final_meta.get("directional_imbalance", val_eval_metrics.get("directional_imbalance", 0.0))),
                    "macro_f1": float(val_eval_metrics.get("macro_f1", 0.0)),
                    "mcc": float(val_eval_metrics.get("mcc", 0.0)),
                    "ece": float(val_eval_metrics.get("ece", 0.0)),
                    "alert_precision": float(val_eval_metrics.get("alert_precision", 0.0)),
                    "alert_coverage": float(val_eval_metrics.get("alert_coverage", 0.0)),
                    # Sprint 3.3-D-fix — alert_sharpe routed from snapshot;
                    # sortino/calmar/max_dd are not in snapshot so they stay
                    # from re-validation. (They are companion metrics whose
                    # divergence from snapshot.alert_sharpe in pathological
                    # T-clamp runs is itself diagnostic — keeping them
                    # surfaces that divergence rather than hiding it.)
                    "alert_sharpe": float(final_meta.get("alert_sharpe", val_eval_metrics.get("alert_sharpe", 0.0))),
                    "alert_sortino": float(val_eval_metrics.get("alert_sortino", 0.0)),
                    "alert_calmar": float(val_eval_metrics.get("alert_calmar", 0.0)),
                    "alert_max_dd": float(val_eval_metrics.get("alert_max_dd", 0.0)),
                    # Sprint 3.3-D-fix — model_score routed from snapshot.
                    # The training-time score is the one the [BEST] gate used;
                    # the re-validation score is misleading when re-fit T
                    # collapses the policy.
                    "model_score": float(final_meta.get("model_score", val_eval_metrics.get("model_score", 0.0))),
                    "attn_sum_mean": float(val_eval_metrics.get("attn_sum_mean", 0.0)),
                    "attn_target_mean": float(val_eval_metrics.get("attn_target_mean", 0.0)),
                    "comprehensiveness": float(val_eval_metrics.get("comprehensiveness", 0.0)),
                    "deletion_drop": float(val_eval_metrics.get("deletion_drop", 0.0)),
                    "insertion_gain": float(val_eval_metrics.get("insertion_gain", 0.0)),
                    "sufficiency_drop": float(val_eval_metrics.get("sufficiency_drop", 0.0)),
                    "factor_consistency": float(val_eval_metrics.get("factor_consistency", 0.0)),
                    # Sprint 3.3-D-fix — quality-aware action + position +
                    # deploy fields ALL routed from snapshot (the candidate
                    # that won promotion), not from re-validation.
                    "action_reward":           float(final_meta.get("action_reward", 0.0)),
                    "action_quality_penalty":  float(final_meta.get("action_quality_penalty", 0.0)),
                    "neutsh_penalty":          float(final_meta.get("neutsh_penalty", 0.0)),
                    "trade_coverage":          float(final_meta.get("trade_coverage", 0.0)),
                    "position_pnl":            float(final_meta.get("position_pnl", 0.0)),
                    "position_hit_rate":       float(final_meta.get("position_hit_rate", 0.0)),
                    "n_position_alerts":       int(final_meta.get("n_position_alerts", 0)),
                    "neutral_share_of_alerts": float(final_meta.get("neutral_share_of_alerts", 0.0)),
                    "alert_sharpe_suppressed": bool(final_meta.get("alert_sharpe_suppressed", False)),
                    "deployable_checkpoint":   bool(final_meta.get("deployable_checkpoint", False)),
                    "deployable_reason":       str(final_meta.get("deployable_reason", "n/a")),
                    "selected_epoch":          int(final_meta.get("epoch", 0)),
                },
                # Sprint 3.3-D-fix — top-level deploy gate snapshot for the
                # cross-fold report builder. Now reads from final_meta (the
                # candidate that won promotion) instead of val_eval_metrics
                # (which is the re-validation, post T-refit, often stale).
                "deployable_checkpoint":   bool(final_meta.get("deployable_checkpoint", False)),
                "deployable_reason":       str(final_meta.get("deployable_reason", "n/a")),
                "test": {
                    "loss": float(test_metrics.get("loss", 0.0)),
                    "dir_acc": float(test_metrics.get("dir_acc", 0.0)),
                    "accuracy": float(test_metrics.get("accuracy", 0.0)),
                    "macro_f1": float(test_metrics.get("macro_f1", 0.0)),
                    "mcc": float(test_metrics.get("mcc", 0.0)),
                    "ece": float(test_metrics.get("ece", 0.0)),
                    "auc": float(test_metrics.get("auc", 0.0)),
                    "alert_precision": float(test_metrics.get("alert_precision", 0.0)),
                    "alert_coverage": float(test_metrics.get("alert_coverage", 0.0)),
                    "alert_sharpe": float(test_metrics.get("alert_sharpe", 0.0)),
                    "alert_sortino": float(test_metrics.get("alert_sortino", 0.0)),
                    "alert_calmar": float(test_metrics.get("alert_calmar", 0.0)),
                    "alert_max_dd": float(test_metrics.get("alert_max_dd", 0.0)),
                    "pnl": float(test_metrics.get("pnl", 0.0)),
                    "model_score": float(test_metrics.get("model_score", 0.0)),
                    "tau": float(test_metrics.get("tau", 0.0)),
                    "gamma": float(test_metrics.get("gamma", 0.0)),
                    "temperature": float(test_metrics.get("temperature", 1.0)),
                    "policy_source": str(test_metrics.get("policy_source", "unknown")),
                    "policy_confidence_source": str(test_metrics.get(
                        "policy_confidence_source", args.policy_confidence_source
                    )),
                    "comprehensiveness": float(test_metrics.get("comprehensiveness", 0.0)),
                    "deletion_drop": float(test_metrics.get("deletion_drop", 0.0)),
                    "insertion_gain": float(test_metrics.get("insertion_gain", 0.0)),
                    "sufficiency_drop": float(test_metrics.get("sufficiency_drop", 0.0)),
                    "factor_consistency": float(test_metrics.get("factor_consistency", 0.0)),
                },
            }
            fold_metrics_list.append(fold_record)

            print(f"[FOLD {fold_idx}/{len(folds)}] Done — "
                  f"val_loss={fold_metrics['final_val_loss']:.4f} "
                  f"val_acc={fold_metrics['final_val_acc']:.3f} | "
                  f"test_f1={test_metrics['macro_f1']:.3f} "
                  f"test_sharpe={test_metrics['alert_sharpe']:.3f}")
            print()

            # Free GPU memory between folds
            del fold_model, fold_trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Aggregate metrics across folds ────────────────────────────────────
        # Session 22 Fix 10: now reports mean AND sample standard deviation
        # (ddof=1) per metric. Previously only mean was reported — reviewers
        # could not assess whether a 2-4% improvement between models is within
        # fold-to-fold noise or represents a real effect. Sample std (not
        # population std) matches convention for K-fold CV: each fold is an
        # observation from the same underlying distribution. For very small
        # K (≤2) std is undefined so we emit NaN; K=3 is the minimum that
        # gives a usable std estimate.
        print("=" * 60)
        print(f"WALK-FORWARD SUMMARY ({len(fold_metrics_list)} folds, mean ± std)")
        print("=" * 60)

        def _mean_std(values: list[float]) -> tuple[float, float]:
            """Returns (mean, sample_std). std=nan if fewer than 2 values."""
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                return (0.0, float("nan"))
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size >= 2 else float("nan")
            return (mean, std)

        val_loss_mean,  val_loss_std  = _mean_std([m["val"]["final_val_loss"] for m in fold_metrics_list])
        val_acc_mean,   val_acc_std   = _mean_std([m["val"]["final_val_acc"]  for m in fold_metrics_list])
        test_f1_mean,   test_f1_std   = _mean_std([m["test"]["macro_f1"]        for m in fold_metrics_list])
        test_mcc_mean,  test_mcc_std  = _mean_std([m["test"]["mcc"]             for m in fold_metrics_list])
        test_ece_mean,  test_ece_std  = _mean_std([m["test"]["ece"]             for m in fold_metrics_list])
        test_auc_mean,  test_auc_std  = _mean_std([m["test"]["auc"]             for m in fold_metrics_list])
        test_prec_mean, test_prec_std = _mean_std([m["test"]["alert_precision"] for m in fold_metrics_list])
        test_cov_mean,  test_cov_std  = _mean_std([m["test"]["alert_coverage"]  for m in fold_metrics_list])
        test_sharpe_mean, test_sharpe_std = _mean_std([m["test"]["alert_sharpe"] for m in fold_metrics_list])
        test_sortino_mean, test_sortino_std = _mean_std([m["test"]["alert_sortino"] for m in fold_metrics_list])
        test_calmar_mean, test_calmar_std = _mean_std([m["test"]["alert_calmar"] for m in fold_metrics_list])
        test_max_dd_mean, test_max_dd_std = _mean_std([m["test"]["alert_max_dd"] for m in fold_metrics_list])
        test_score_mean, test_score_std = _mean_std([m["test"]["model_score"] for m in fold_metrics_list])
        test_pnl_mean, test_pnl_std = _mean_std([m["test"]["pnl"] for m in fold_metrics_list])
        test_comp_mean, test_comp_std = _mean_std([m["test"]["comprehensiveness"] for m in fold_metrics_list])
        test_suf_drop_mean, test_suf_drop_std = _mean_std([m["test"]["sufficiency_drop"] for m in fold_metrics_list])
        test_fac_cons_mean, test_fac_cons_std = _mean_std([m["test"]["factor_consistency"] for m in fold_metrics_list])

        def _fmt(mean: float, std: float) -> str:
            return f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}  (n<2 no std)"

        print(f"  val_loss:              {_fmt(val_loss_mean, val_loss_std)}")
        print(f"  val_acc:               {_fmt(val_acc_mean, val_acc_std)}")
        print(f"  test_f1:               {_fmt(test_f1_mean, test_f1_std)}")
        print(f"  test_mcc:              {_fmt(test_mcc_mean, test_mcc_std)}")
        print(f"  test_ece:              {_fmt(test_ece_mean, test_ece_std)}")
        print(f"  test_auc:              {_fmt(test_auc_mean, test_auc_std)}")
        print(f"  test_alert_precision:  {_fmt(test_prec_mean, test_prec_std)}")
        print(f"  test_alert_coverage:   {_fmt(test_cov_mean, test_cov_std)}")
        print(f"  test_alert_sharpe:     {_fmt(test_sharpe_mean, test_sharpe_std)}")
        print(f"  test_alert_sortino:    {_fmt(test_sortino_mean, test_sortino_std)}")
        print(f"  test_alert_calmar:     {_fmt(test_calmar_mean, test_calmar_std)}")
        print(f"  test_alert_max_dd:     {_fmt(test_max_dd_mean, test_max_dd_std)}")
        print(f"  test_pnl:              {_fmt(test_pnl_mean, test_pnl_std)}")
        print(f"  test_model_score:      {_fmt(test_score_mean, test_score_std)}")
        print(f"  test_comprehensiveness:{_fmt(test_comp_mean, test_comp_std)}")
        print(f"  test_sufficiency_drop: {_fmt(test_suf_drop_mean, test_suf_drop_std)}")
        print(f"  test_factor_consistency:{_fmt(test_fac_cons_mean, test_fac_cons_std)}")

        agg = {
            "final_val_loss":          val_loss_mean,        "final_val_loss_std":          val_loss_std,
            "final_val_acc":           val_acc_mean,         "final_val_acc_std":           val_acc_std,
            "test_macro_f1":           test_f1_mean,         "test_macro_f1_std":           test_f1_std,
            "test_mcc":                test_mcc_mean,        "test_mcc_std":                test_mcc_std,
            "test_ece":                test_ece_mean,        "test_ece_std":                test_ece_std,
            "test_auc":                test_auc_mean,        "test_auc_std":                test_auc_std,
            "test_alert_precision":    test_prec_mean,       "test_alert_precision_std":    test_prec_std,
            "test_alert_coverage":     test_cov_mean,        "test_alert_coverage_std":     test_cov_std,
            "test_alert_sharpe":       test_sharpe_mean,     "test_alert_sharpe_std":       test_sharpe_std,
            "test_alert_sortino":      test_sortino_mean,    "test_alert_sortino_std":      test_sortino_std,
            "test_alert_calmar":       test_calmar_mean,     "test_alert_calmar_std":       test_calmar_std,
            "test_alert_max_dd":       test_max_dd_mean,     "test_alert_max_dd_std":       test_max_dd_std,
            "test_pnl":                test_pnl_mean,        "test_pnl_std":                test_pnl_std,
            "test_model_score":        test_score_mean,      "test_model_score_std":        test_score_std,
            "test_comprehensiveness":  test_comp_mean,       "test_comprehensiveness_std":  test_comp_std,
            "test_sufficiency_drop":   test_suf_drop_mean,   "test_sufficiency_drop_std":   test_suf_drop_std,
            "test_factor_consistency": test_fac_cons_mean,   "test_factor_consistency_std": test_fac_cons_std,
        }

        # Persist fold results
        wf_results = {
            "n_folds":      n_folds,
            "dataset_len":  dataset_len,
            "protocol":     "expanding_window_train_val_test_with_embargo",
            "embargo_steps": embargo_steps,
            "epochs_per_fold": args.epochs,
            "fold_metrics": fold_metrics_list,
            "averages":     agg,
        }
        wf_results_path = args.artifact_dir / "walk_forward_results.json"
        with open(wf_results_path, "w") as f:
            json.dump(wf_results, f, indent=2)
        print(f"\n[OK] Fold results saved: {wf_results_path}")
        summary = standardize_walk_forward_artifacts(
            wf_results_path, args.symbol, args.horizon, artifact_dir=args.artifact_dir
        )
        if summary is not None:
            deploy_fold = summary.get("selected_deployment_fold")
            policy = summary.get("deployment_policy", {})
            # Sprint 3.3-E — promote the deployable status into the same
            # log line as the policy file so anyone reading the run output
            # sees the deploy gate verdict together with the thresholds
            # being written. Pre-3.3-E this line implied "policy saved =
            # safe to deploy", which was wrong when 0/N folds passed the
            # gate. Use [WARN] tag instead of [OK] when not deployable so
            # log scrapers / CI grep can hard-fail on it.
            _deployable_artifact = bool(policy.get("deployable", False))
            _deploy_status = str(policy.get("deploy_status", "unknown"))
            _deploy_n_pass = int(policy.get("deployable_n_passed", 0))
            _deploy_n_total = int(policy.get("deployable_n_total", 0))
            _tag = "[OK]" if _deployable_artifact else "[WARN]"
            _suffix = (
                f" deployable=True ({_deploy_n_pass}/{_deploy_n_total} folds passed gate)"
                if _deployable_artifact
                else f" deployable=False status={_deploy_status} ({_deploy_n_pass}/{_deploy_n_total} folds passed gate) — DO NOT DEPLOY"
            )
            print(
                f"{_tag} Standardized policy/report saved "
                f"(deploy_fold={deploy_fold}, tau={policy.get('tau')}, gamma={policy.get('gamma')}){_suffix}"
            )
            # Sprint 3.3-D-fix — report deployable status across folds. Reads
            # the deployable_checkpoint / deployable_reason fields routed from
            # the training-time selection_meta (the candidate that won
            # promotion), NOT from the outer-loop re-validation. Pre-fix
            # versions of this warning showed the re-validation reason which
            # often mis-attributed (e.g., a checkpoint with TradeCov=0.443 at
            # training time would surface trade_cov<0.05 because re-fit T
            # collapsed the policy).
            _deploy_states = []
            for _fm in fold_metrics_list:
                _flag = _fm.get("deployable_checkpoint")
                _reason = _fm.get("deployable_reason", "n/a")
                if _flag is not None:
                    _deploy_states.append((_fm.get("fold"), bool(_flag), str(_reason)))
            if _deploy_states:
                _ok_count = sum(1 for _, _ok, _ in _deploy_states if _ok)
                if _ok_count == 0:
                    print(
                        f"[WARN] No fold produced a deployable checkpoint "
                        f"(0/{len(_deploy_states)} folds passed)."
                    )
                    for fid, ok, reason in _deploy_states:
                        print(f"        fold={fid}: deployable={ok} reason={reason}")
                    print(
                        "        Production deploy NOT recommended. "
                        "Investigate trade quality / calibration / data labels."
                    )
                else:
                    print(
                        f"[OK] Deployable checkpoints: {_ok_count}/{len(_deploy_states)} folds."
                    )

    else:
        # ── Fixed 70/15/15 temporal split (original behaviour, unchanged) ─────
        train_size = int(0.7 * len(dataset))
        val_size   = int(0.15 * len(dataset))
        test_size  = len(dataset) - train_size - val_size

        train_set = Subset(dataset, range(0, train_size))
        val_set   = Subset(dataset, range(train_size, train_size + val_size))
        test_set  = Subset(dataset, range(train_size + val_size, len(dataset)))
        dataset.fit_market_scaler(range(0, train_size))

        # Sprint 7 — balanced batch sampler (fixed-split mode). Same toggle/
        # rationale as walk-forward path; patched here to avoid behaviour
        # divergence between modes when the YAML flag is on.
        _train_sampler = None
        if _BALANCED_BATCH_SAMPLER:
            _all_dir_labels = dataset.get_direction_labels()
            _fixed_train_indices = np.arange(0, train_size, dtype=np.int64)
            _train_dir_labels = _all_dir_labels[_fixed_train_indices]
            _train_sampler = BalancedBatchSampler(
                labels=_train_dir_labels,
                batch_size=args.batch_size,
                num_classes=3,
                seed=42,
            )
            print(
                f"[Sampler] balanced_batch_sampler enabled (fixed-split): "
                f"batch_size={_train_sampler.batch_size}, "
                f"class_counts={_train_sampler.class_counts}, "
                f"base_quota={_train_sampler.base_quota}, "
                f"batches_per_epoch={len(_train_sampler)}",
                flush=True,
            )

        _train_loader_kwargs = dict(
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            worker_init_fn=_seed_worker,       # P2 #21: reproducible workers
            generator=_LOADER_GENERATOR,
        )
        if _train_sampler is not None:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_sampler=_train_sampler,
                **_train_loader_kwargs,
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                **_train_loader_kwargs,
            )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,   # P1 #14: time-ordered val/test evaluation
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,   # P1 #14: explicit — never shuffle val/test (time-ordered evaluation)
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )

        print(f"[OK] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # Session 26: K_h from YAML `top_k` block (previously hardcoded 3/4/5/8).
        # YAML-editable per-horizon top-K (paper §3.3.2, Eq.12).
        model = SAFEAlertNet(
            market_dim=63, has_news=True,
            K_15m=_TOP_K_MAP["15m"], K_1h=_TOP_K_MAP["1h"],
            K_4h=_TOP_K_MAP["4h"],   K_24h=_TOP_K_MAP["24h"],
            n_factors=FACTOR_CLASSES,
            use_bar_sequences=args.use_bar_sequences,
            market_input_mode=(None if getattr(args, "market_input_mode", "auto") == "auto"
                               else args.market_input_mode),
            bar_seq_len=args.bar_seq_len,
            bar_feat_dim=args.bar_feat_dim,
            predict_volatility=(_LAMBDA_VOL > 0.0),
            predict_ret_sign=(_LAMBDA_RET_SIGN_CLS > 0.0),
            predict_ret_bin=(_LAMBDA_RET_BIN > 0.0),
            predict_edge=(_LAMBDA_UP_EDGE > 0.0 or _LAMBDA_DOWN_EDGE > 0.0),
        )
        K_h = model.K_h_map.get(args.horizon, 8)  # derive from model to avoid mismatch
        print(f"[OK] Model initialized (K_15m=3, K_1h=4, K_4h=5, K_24h=8) - selective article selection")
        print(f"[OK] K_h={K_h} derived from model.K_h_map for horizon={args.horizon}")

        trainer = SAFEAlertTrainer(
            model, device=args.device, lr=args.lr, weight_decay=args.weight_decay,
            horizon=args.horizon, K_h=K_h, grad_clip=args.grad_clip,
            policy_method=args.policy_method,
            tau_percentile=args.tau_percentile,
            gamma_percentile=args.gamma_percentile,
            policy_confidence_source=args.policy_confidence_source,
            ablation=args.ablation,
            symbol=args.symbol,
        )
        metrics, best_ckpt = trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs,
            checkpoint_dir=args.artifact_dir,
            early_stopping_patience=_ES_PATIENCE,
            explain_every=args.explain_every,
            dataset=dataset,   # P0 #1: snapshot preprocessing state into checkpoint
        )

        print(f"\n[COMPLETE] Training complete!")
        print(f"  Final val loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final val acc: {metrics['final_val_acc']:.3f}")
        print(f"  Final epoch: {metrics['final_epoch']}")
        print(f"  Final model: {best_ckpt.name}")

        # ── P0 #3: Held-out TEST evaluation with frozen val thresholds ────────
        # Previously this branch built test_loader but never evaluated it (old
        # comment: `# noqa: F841`). That silently collapsed the 70/15/15 split
        # to 70/15 and any number reported as "test" was actually val. Now we
        # tune τ/γ/T on val, freeze them, and measure the test split exactly
        # once — matching the walk-forward path and PDF Section 4.2.3 protocol.
        print(f"\n[TEST] Loading FINAL checkpoint for held-out evaluation ...")
        final_path = args.artifact_dir / f"safe_alert_{args.horizon}_FINAL.pt"
        _ckpt_path = final_path if final_path.exists() else best_ckpt
        _ckpt_state = torch.load(_ckpt_path, map_location=args.device, weights_only=False)
        trainer.model.load_state_dict(_ckpt_state["model_state"])

        print(f"[TEST] Tuning τ/γ/T on val split (frozen for test) ...")
        val_eval = trainer.validate(val_loader)
        print(f"[TEST] Evaluating on held-out test split with "
              f"τ={val_eval['tau']:.4f} γ={val_eval['gamma']:.4f} T={val_eval['temperature']:.3f} ...")
        test_eval = trainer.validate(
            test_loader,
            tau=val_eval["tau"],
            gamma=val_eval["gamma"],
            temperature=val_eval["temperature"],
        )
        print("\n[TEST RESULTS]")
        print(f"  Macro-F1:       {test_eval['macro_f1']:.4f}")
        print(f"  MCC:            {test_eval['mcc']:.4f}")
        print(f"  AUC:            {test_eval.get('auc', 0.0):.4f}")
        print(f"  ECE:            {test_eval['ece']:.4f}")
        print(f"  Alert Sharpe:   {test_eval['alert_sharpe']:.4f}")
        print(f"  Alert Coverage: {test_eval['alert_coverage']:.4f}")
        print(f"  Alert Precision:{test_eval['alert_precision']:.4f}")
        print(f"  Alert Max DD:   {test_eval['alert_max_dd']:.4f}")
        print(f"  PnL:            {test_eval.get('pnl', 0.0):.4f}")

        test_report_path = args.artifact_dir / f"test_report_{args.horizon}.json"
        with open(test_report_path, "w") as f:
            json.dump({
                "val_frozen_policy": {
                    "tau":         float(val_eval["tau"]),
                    "gamma":       float(val_eval["gamma"]),
                    "temperature": float(val_eval["temperature"]),
                    "policy_confidence_source": str(
                        val_eval.get("policy_confidence_source", args.policy_confidence_source)
                    ),
                },
                "test_metrics": {k: float(v) for k, v in test_eval.items()
                                 if isinstance(v, (int, float))},
                "checkpoint":  _ckpt_path.name,
                "symbol":      args.symbol,
                "horizon":     args.horizon,
            }, f, indent=2)
        print(f"[TEST] Saved report → {test_report_path.name}")


if __name__ == "__main__":
    main()
