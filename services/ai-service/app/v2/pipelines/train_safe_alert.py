"""
Train SAFE-Alert model (PDF Eq.30 multi-objective loss).

Walk-forward protocol (expanding window + embargo):
train on [0:t], validate on next window, test on immediately-following window,
then shift forward and repeat.

Commands:
  python train_safe_alert.py                        # use train_config.yaml defaults
  python train_safe_alert.py --config my.yaml       # custom config
  python train_safe_alert.py --epochs 40            # override one value
  python train_safe_alert.py --walk_forward         # walk-forward CV (PDF Eq.1)
"""

import os
import sys
import json
import gc
import shutil
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional, Tuple, Dict
from torch.utils.data import Subset
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))  # Add pipelines folder for same-directory imports
from models.safe_alert_net import SAFEAlertNet, FACTOR_CLASSES
from safe_alert_training_utils import MultiObjectiveLoss
from safe_alert_dataset import SAFEAlertDataset
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_brier_score,
    compute_alert_precision, mini_backtest, compute_model_selection_score,
    compute_auc, compute_coverage_risk_auc, compute_hit_rate, search_alert_policy,
    fit_temperature_scaling, apply_temperature_to_logits,
    compute_deletion_insertion_score, compute_sufficiency, compute_factor_consistency,
)
from utils import ARTIFACT_DIR, standardize_walk_forward_artifacts

torch.manual_seed(42)
np.random.seed(42)


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
torch.backends.cudnn.benchmark = True

# ── Training curriculum constants ──────────────────────────────────────────────
_STAGE1_END_FRAC  = 0.20   # Stage 1 ends after this fraction of total epochs
                            # Reduced from 0.30 → 0.20: factor head cold start fix.
                            # Stage 2 now gets 40% of epochs (vs 30%) so Lfac has
                            # enough time to recover from zero-gradient Stage 1.
_STAGE2_END_FRAC  = 0.60   # Stage 2 ends after this fraction of total epochs
_TAU_PERCENTILE   = 65     # Alert threshold τ seed: P(conf≥p65)≈35% → near 35% coverage target
_GAMMA_PERCENTILE = 55     # Alert threshold γ seed: P(max_prob≥p55)≈45% → AND-gate ≈ 35%×45%≈16%+grid search
_ACCUM_STEPS      = 2      # Gradient accumulation steps (effective_bs = bs * steps)
_ES_PATIENCE      = 12     # Early-stopping patience (Stage 3 only)
_STAGE3_LR_SCALE  = 0.5    # Lower LR in fine-tuning stage to reduce late-stage drift

# ── Lambda schedule aligned to PDF defaults (Table 4) ─────────────────────────
# PDF defaults: λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.2, λ5=0.1, λ6=0.1, λ7=0.05
#
# Stage 1: direction + return + selection only (PDF curriculum design).
# Lfac, Lfaith are OFF — builds direction/return foundation first.
# Lcal/Lrisk stay OFF in Stage 1 per PDF Section 4.5.1.
_S1_LAMBDA1 = 1.0
_S1_LAMBDA2 = 0.5
_S1_LAMBDA3 = 0.0    # Lfac   — OFF Stage 1 (per PDF Section 4.5.1)
_S1_LAMBDA4 = 0.2    # Lsel   — ON Stage 1 (trains selection attention)
_S1_LAMBDA5 = 0.0    # Lcal   — OFF Stage 1
_S1_LAMBDA6 = 0.0    # Lfaith — OFF Stage 1 (per PDF Section 4.5.1)
_S1_LAMBDA7 = 0.0    # Lrisk  — OFF Stage 1

# Stage 2 TARGET lambdas: paper adds Lfac + Lfaith.
# FIX: added small λ5=0.02 and λ7=0.01 in Stage 2 (PDF Section 4.5.1 does not
# mandate these are strictly zero in Stage 2; only that they are small compared
# to their Stage 3 values). Small Lcal prevents temperature drift (0.6→2.8
# observed without it). Small Lrisk starts teaching the confidence head to
# distinguish correct vs wrong predictions before Stage 3.
# Stage 2 linearly interpolates from S1 → S2 targets.
_S2_LAMBDA1 = 1.0
_S2_LAMBDA2 = 0.5
_S2_LAMBDA3 = 0.3    # Lfac   — full weight by end of Stage 2
_S2_LAMBDA4 = 0.2
_S2_LAMBDA5 = 0.02   # Lcal   — small value (was 0.0); prevents temperature drift
_S2_LAMBDA6 = 0.1    # Lfaith — full weight by end of Stage 2
_S2_LAMBDA7 = 0.01   # Lrisk  — tiny value (was 0.0); early coverage signal

# Stage 3: full loss, aligned to PDF Table 4 defaults.
_S3_LAMBDA1 = 1.0
_S3_LAMBDA2 = 0.5
_S3_LAMBDA3 = 0.3    # Lfac   — PDF Table 4
_S3_LAMBDA4 = 0.2
_S3_LAMBDA5 = 0.1    # Lcal   — PDF Table 4 (full calibration)
_S3_LAMBDA6 = 0.1    # Lfaith — PDF Table 4
_S3_LAMBDA7 = 0.05   # Lrisk  — PDF Table 4 (full selective risk)


class SAFEAlertTrainer:
    """Trainer for SAFE-Alert with thesis-style 3-stage curriculum."""

    def __init__(
        self,
        model: SAFEAlertNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        horizon: str = "1h",
        K_h: int = 8,
        grad_clip: float = 1.0,
        class_weights: torch.Tensor = None,
        policy_method: str = "grid",
        tau_percentile: int = _TAU_PERCENTILE,
        gamma_percentile: int = _GAMMA_PERCENTILE,
        ablation: Optional[str] = None,
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
        self.ablation = ablation  # e.g. "w/o_faithfulness" (PDF Section 4.6)
        # Instantiate loss with learnable lambdas
        self.loss_fn = MultiObjectiveLoss(
            lambda1=1.0,
            lambda2=0.5,
            lambda3=0.3,
            lambda4=0.2,
            lambda5=0.1,
            lambda6=0.1,
            lambda7=0.05,
            K_h=K_h,
            eta=0.1,
            faith_margin=0.15,
            coverage_target=0.35,
            mu=0.02,
            class_weights=class_weights,
            learned_lambdas=False,
        )

        # Optimizer: include both model and lambda parameters
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.loss_fn.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )
        self.current_epoch = 0

    @staticmethod
    def _tensor_is_finite(tensor: Optional[torch.Tensor]) -> bool:
        return tensor is None or bool(torch.isfinite(tensor).all().item())

    def _outputs_are_finite(self, outputs: Dict[str, Optional[torch.Tensor]]) -> bool:
        for key in ("dir_logits", "ret_pred", "confidence", "attn_weights", "p_fac_all"):
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
                tau = self.model.sel_attn.tau.data
                if not torch.isfinite(tau).all():
                    repaired += int((~torch.isfinite(tau)).sum().item())
                tau.copy_(torch.nan_to_num(tau, nan=1.0, posinf=5.0, neginf=0.1))
                tau.clamp_(min=0.1, max=5.0)
        return repaired

    def _log_factor_head_gradients(self, batch_idx: int) -> None:
        """Log factor-module gradient norms once per epoch for debugging."""
        factor_grad_lines = []
        for name, param in self.model.named_parameters():
            if "factor_mod" not in name:
                continue
            if param.grad is None:
                factor_grad_lines.append(f"{name}=NO_GRAD")
            else:
                factor_grad_lines.append(f"{name}={param.grad.norm().item():.6f}")
        if factor_grad_lines:
            logger.info(
                "Epoch %d batch %d factor-gradients | %s",
                self.current_epoch,
                batch_idx,
                " ; ".join(factor_grad_lines),
            )

    def _update_loss_weights_for_stage(self, epoch: int, total_epochs: int) -> None:
        """Apply the thesis 3-stage lambda schedule."""
        frac = epoch / max(total_epochs, 1)
        if frac <= _STAGE1_END_FRAC:
            target = (
                _S1_LAMBDA1, _S1_LAMBDA2, _S1_LAMBDA3, _S1_LAMBDA4,
                _S1_LAMBDA5, _S1_LAMBDA6, _S1_LAMBDA7,
            )
        elif frac <= _STAGE2_END_FRAC:
            # Interpolate S1 → S2: ramp Lfac (λ3), Lfaith (λ6), and small Lcal/Lrisk.
            # λ5 ramps 0→0.02 and λ7 ramps 0→0.01 to prevent temperature drift and
            # give confidence head early calibration signal before Stage 3.
            mix = (frac - _STAGE1_END_FRAC) / max(_STAGE2_END_FRAC - _STAGE1_END_FRAC, 1e-8)
            s1 = (
                _S1_LAMBDA1, _S1_LAMBDA2, _S1_LAMBDA3, _S1_LAMBDA4,
                _S1_LAMBDA5, _S1_LAMBDA6, _S1_LAMBDA7,
            )
            s2 = (
                _S2_LAMBDA1, _S2_LAMBDA2, _S2_LAMBDA3, _S2_LAMBDA4,
                _S2_LAMBDA5, _S2_LAMBDA6, _S2_LAMBDA7,
            )
            target = tuple(a + mix * (b - a) for a, b in zip(s1, s2))
        else:
            target = (
                _S3_LAMBDA1, _S3_LAMBDA2, _S3_LAMBDA3, _S3_LAMBDA4,
                _S3_LAMBDA5, _S3_LAMBDA6, _S3_LAMBDA7,
            )

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
        """Lower LR for the thesis fine-tuning stage.

        IMPORTANT: CosineAnnealingWarmRestarts recomputes lr from base_lrs on each
        scheduler.step(), so we must update base_lrs as well — otherwise the manual
        group["lr"] *= scale is overwritten on the very next scheduler.step() call.
        """
        for group in self.optimizer.param_groups:
            group["lr"] *= _STAGE3_LR_SCALE
        # Persist the scale into the scheduler's base LR so cosine restarts use
        # the reduced LR as the new ceiling (not the original pre-Stage-3 LR).
        if hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [
                base_lr * _STAGE3_LR_SCALE for base_lr in self.scheduler.base_lrs
            ]
        print(
            f"[STAGE 3] Fine-tuning LR scale x{_STAGE3_LR_SCALE:.2f} -> "
            f"{self.optimizer.param_groups[0]['lr']:.6g} "
            f"(scheduler base_lrs updated)"
        )

    def train_epoch(self, train_loader, accumulate_steps: int = _ACCUM_STEPS) -> Dict[str, float]:
        """Train one epoch with gradient accumulation.

        accumulate_steps=2 with batch_size=8 gives effective batch_size=16
        without additional memory overhead.
        """
        self.model.train()
        losses = {"loss": 0, "Ldir": 0, "Lret": 0, "Lfac": 0, "Lsel": 0, "Lcal": 0, "Lfaith": 0, "Lrisk": 0}
        count = 0
        skipped_batches = 0
        skipped_steps = 0
        logged_factor_grads = False
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            market_feat = batch["market_features"].to(self.device)
            article_emb = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
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

            # CRITICAL: fac_labels must be (B,K,C) with values in [0,1] summing to 1
            assert fac_labels.dim() == 3, f"fac_labels must be 3D (B,K,C), got {fac_labels.dim()}D shape {fac_labels.shape}"
            assert not torch.isnan(fac_labels).any(), "fac_labels contains NaN!"
            assert not torch.isinf(fac_labels).any(), "fac_labels contains Inf!"
            assert (fac_labels >= 0).all() and (fac_labels <= 1).all(), f"fac_labels values out of [0,1] range: min={fac_labels.min():.4f}, max={fac_labels.max():.4f}"

            assert not torch.isnan(market_feat).any(), "NaN in market_features"
            assert not torch.isinf(market_feat).any(), "Inf in market_features"

            # Forward pass
            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
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
            # Requirements:
            #   1. eval() mode → deterministic (no Dropout noise on faithfulness gap).
            #   2. any() condition → compute for batches where ≥1 sample has articles;
            #      per-sample masking in the loss handles no-article samples correctly.
            has_articles = article_mask.sum(dim=1) > 0  # (B,) per-sample boolean
            if has_articles.any():
                self.model.eval()
                with torch.no_grad():
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta,
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]  # (B, 3)
                self.model.train()

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
            dir_correct = (dir_preds == dir_labels).long()

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
                if not logged_factor_grads:
                    self._log_factor_head_gradients(batch_idx)
                    logged_factor_grads = True
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                repaired = self._sanitize_model_parameters()
                if repaired:
                    logger.warning(
                        "Epoch %d batch %d: sanitized %d non-finite/overflow parameters after optimizer step",
                        self.current_epoch, batch_idx, repaired,
                    )
                self.optimizer.zero_grad()

            for k in losses:
                losses[k] += loss_dict[k]
            count += 1

        if count == 0:
            raise RuntimeError(
                f"Training collapsed at epoch {self.current_epoch}: all batches were skipped "
                f"(invalid outputs/losses={skipped_batches}, invalid steps={skipped_steps})."
            )

        for k in losses:
            losses[k] /= max(count, 1)
        return losses

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        tau: Optional[float] = None,
        gamma: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, float]:
        """Validate and compute balanced forecast + alert utility metrics."""
        self.model.eval()
        val_losses = {"loss": 0, "dir_acc": 0, "count": 0}

        all_dir_preds = []
        all_dir_labels = []
        all_confidence = []
        all_ret_labels = []
        all_ret_preds = []
        all_dir_logits = []
        all_dir_probs = []
        all_attn_sums = []
        all_attn_targets = []
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

            outputs = self.model(
                market_feat,
                horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
            )

            has_articles = article_mask.any(dim=1).any()
            with torch.no_grad():
                if has_articles:
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta,
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]
                else:
                    masked_dir_logits = None

            if not self._tensor_is_finite(masked_dir_logits):
                logger.warning("Validation: skipping batch with non-finite masked logits")
                continue

            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"], article_mask, market_feat.shape[0], article_emb.shape[1]
            )
            all_attn_sums.append(attn_w.sum(dim=1).detach().cpu().numpy())
            all_attn_targets.append(
                article_mask.float().sum(dim=1).clamp(max=float(self.loss_fn.K_h)).cpu().numpy()
            )

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
            )

            val_losses["loss"] += loss_dict["loss"].item()
            val_losses["count"] += 1

            all_dir_preds.append(dir_preds.cpu().numpy())
            all_dir_labels.append(dir_labels.cpu().numpy())
            all_confidence.append(outputs["confidence"].cpu().numpy())
            all_ret_labels.append(batch["return"].cpu().numpy())
            all_ret_preds.append(outputs["ret_pred"].cpu().numpy())
            all_dir_logits.append(outputs["dir_logits"].cpu().numpy())
            all_dir_probs.append(F.softmax(outputs["dir_logits"], dim=-1).cpu().numpy())

            # ── Explanation metrics data collection (PDF Section 4.4.2) ──────────
            # Deletion: probs WITHOUT top-K articles (already computed as masked_outputs)
            if masked_dir_logits is not None:
                all_masked_dir_probs.append(F.softmax(masked_dir_logits, dim=-1).cpu().numpy())
            else:
                # No articles → masked == full (deletion_drop = 0 for this batch)
                all_masked_dir_probs.append(F.softmax(outputs["dir_logits"], dim=-1).cpu().numpy())

            # Sufficiency: probs WITH ONLY the selected top-K articles
            sel_mask = outputs.get("selected_mask")
            if sel_mask is not None and has_articles:
                with torch.no_grad():
                    sel_outputs = self.model(
                        market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=sel_mask.float(),   # only selected articles are "valid"
                        article_meta_vec=article_meta,
                    )
                all_selected_only_probs.append(F.softmax(sel_outputs["dir_logits"], dim=-1).cpu().numpy())
            else:
                # No selection info → treat as full (sufficiency = 1.0 contribution)
                all_selected_only_probs.append(F.softmax(outputs["dir_logits"], dim=-1).cpu().numpy())

            # True Insertion baseline: probs with NO articles.
            # All-zeros mask → has_any=0 → z_news=0, z_fac=0 → fusion([0,0,z_mkt,h_emb]).
            # Both no_article and selected_only use the same CrossAttentionFusion path,
            # so insertion_gain = selected_conf - no_art_conf is a pure article contribution
            # measured through the same sub-network (PDF Section 4.4.2).
            no_article_mask = torch.zeros_like(article_mask)
            with torch.no_grad():
                no_art_outputs = self.model(
                    market_feat, horizon=self.horizon,
                    article_emb=article_emb,
                    article_mask=no_article_mask,
                    article_meta_vec=article_meta,
                )
            all_no_article_probs.append(F.softmax(no_art_outputs["dir_logits"], dim=-1).cpu().numpy())

            # Factor consistency: dominant factor per sample
            # Use attention-weighted sum (Eq.15: z_fac = Σ alpha_tilde_i U p_fac_i)
            # so only selected articles contribute, matching PDF intent.
            if outputs["p_fac_all"] is not None and outputs["attn_weights"] is not None:
                # attn_w already prepared above: (B, K), alpha_tilde (top-K gated)
                # p_fac_all: (B, K, C) → attention-weighted → (B, C) → argmax → (B,)
                attn_for_fac = attn_w.unsqueeze(-1)  # (B, K, 1)
                fac_per_sample = (outputs["p_fac_all"] * attn_for_fac).sum(dim=1).argmax(dim=-1)
                all_factor_preds.append(fac_per_sample.cpu().numpy())

            val_losses["dir_acc"] += (dir_preds == dir_labels).float().mean().item()

        if val_losses["count"] == 0:
            raise RuntimeError("Validation collapsed: all batches were skipped due to non-finite outputs.")

        val_losses["loss"] /= max(val_losses["count"], 1)
        val_losses["dir_acc"] /= max(val_losses["count"], 1)
        del val_losses["count"]

        all_dir_preds = np.concatenate(all_dir_preds, axis=0)
        all_dir_labels = np.concatenate(all_dir_labels, axis=0)
        all_confidence = np.concatenate(all_confidence, axis=0)
        all_ret_labels = np.concatenate(all_ret_labels, axis=0)
        all_ret_preds = np.concatenate(all_ret_preds, axis=0)
        all_dir_logits = np.concatenate(all_dir_logits, axis=0)
        all_dir_probs = np.concatenate(all_dir_probs, axis=0)
        attn_sum_mean = float(np.concatenate(all_attn_sums, axis=0).mean()) if all_attn_sums else 0.0
        attn_target_mean = float(np.concatenate(all_attn_targets, axis=0).mean()) if all_attn_targets else 0.0

        # ── Explanation metrics (PDF Section 4.4.2) ────────────────────────────
        # Concatenate all per-batch probs
        all_masked_dir_probs_np    = np.concatenate(all_masked_dir_probs, axis=0)
        all_selected_only_probs_np = np.concatenate(all_selected_only_probs, axis=0)
        all_no_article_probs_np    = np.concatenate(all_no_article_probs, axis=0)

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
            fac_all_np = np.concatenate(all_factor_preds, axis=0)
            _window_sz = min(32, max(2, len(fac_all_np) // 8))
            if len(fac_all_np) >= 2 * _window_sz:
                _n_win = len(fac_all_np) // _window_sz
                fac_windows = [fac_all_np[i * _window_sz:(i + 1) * _window_sz] for i in range(_n_win)]
                factor_consistency_score = compute_factor_consistency(fac_windows)
            else:
                factor_consistency_score = 1.0  # too few samples to compare
        else:
            factor_consistency_score = 1.0  # no factor predictions (market-only mode)

        if temperature is None:
            temp_fit = fit_temperature_scaling(all_dir_logits, all_dir_labels)
            temperature = float(temp_fit["temperature"])
            temperature_nll = float(temp_fit["temperature_nll"])
            temperature_source = "validation_temperature_scaling"
        else:
            temperature = float(temperature)
            temperature_source = "frozen_validation_temperature"
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

        ret_mae = np.mean(np.abs(all_ret_preds - all_ret_labels))
        ret_rmse = np.sqrt(np.mean((all_ret_preds - all_ret_labels) ** 2))
        ret_correlation = np.corrcoef(all_ret_preds, all_ret_labels)[0, 1] if len(all_ret_preds) > 1 else 0.0

        macro_f1 = compute_macro_f1(calibrated_preds, all_dir_labels)
        mcc = compute_mcc(calibrated_preds, all_dir_labels)
        ece = compute_ece(max_prob_all, calibrated_preds, all_dir_labels)
        brier = compute_brier_score(max_prob_all, calibrated_preds, all_dir_labels)

        if tau is not None and gamma is not None:
            best_tau = float(tau)
            best_gamma = float(gamma)
            alert_prec, alert_cov, sel_risk = compute_alert_precision(
                all_confidence, calibrated_preds, all_dir_labels,
                dir_probs=calibrated_dir_probs, tau=best_tau, gamma=best_gamma,
            )
            backtest_metrics = mini_backtest(
                all_confidence, calibrated_preds, all_ret_labels,
                tau=best_tau, gamma=best_gamma, dir_probs=calibrated_dir_probs,
            )
            tau_seed = float(np.percentile(all_confidence, self.tau_percentile))
            gamma_seed = float(np.percentile(max_prob_all, self.gamma_percentile))
            policy_source = "frozen_validation_policy"
            policy_objective = None
            coverage_penalty = None
        else:
            policy = search_alert_policy(
                all_confidence, calibrated_preds, all_dir_labels, all_ret_labels, calibrated_dir_probs,
                macro_f1, mcc, ece,
                tau_percentile=self.tau_percentile,
                gamma_percentile=self.gamma_percentile,
                target_coverage=self.loss_fn.coverage_target,
                policy_method=self.policy_method,
            )
            best_tau = float(policy["tau"])
            best_gamma = float(policy["gamma"])
            alert_prec = float(policy["alert_precision"])
            alert_cov = float(policy["alert_coverage"])
            sel_risk = float(policy["selective_risk"])
            backtest_metrics = {
                "alert_sharpe": float(policy["alert_sharpe"]),
                "alert_sortino": float(policy["alert_sortino"]),
                "alert_calmar": float(policy["alert_calmar"]),
                "alert_max_dd": float(policy["alert_max_dd"]),
                "alert_hit_rate": float(policy["hit_rate"]),
                "alert_coverage": float(policy["alert_coverage"]),
                "pnl": float(policy.get("pnl", 0.0)),  # direction-aware cumulative P&L (PDF Table 4)
            }
            tau_seed = float(policy["tau_seed"])
            gamma_seed = float(policy["gamma_seed"])
            policy_source = str(policy["policy_source"])
            policy_objective = float(policy["policy_objective"])
            coverage_penalty = float(policy["coverage_penalty"])

        alert_sharpe = backtest_metrics["alert_sharpe"]
        model_score = compute_model_selection_score(macro_f1, mcc, ece, alert_sharpe)
        auc = compute_auc(calibrated_dir_probs, all_dir_labels)
        aurc = compute_coverage_risk_auc(max_prob_all, calibrated_preds, all_dir_labels)
        hit_rate = compute_hit_rate(all_confidence, all_ret_labels, tau=best_tau)

        val_losses.update({
            "macro_f1": macro_f1,
            "mcc": mcc,
            "ece": ece,
            "brier": brier,
            # "accuracy" alias matches PDF Tables 3/5 column naming (dir_acc = accuracy)
            "accuracy": float(val_losses["dir_acc"]),
            "tau": best_tau,
            "gamma": best_gamma,
            "temperature": temperature,
            "temperature_nll": temperature_nll,
            "temperature_source": temperature_source,
            "tau_seed": tau_seed,
            "gamma_seed": gamma_seed,
            "policy_source": policy_source,
            "policy_objective": policy_objective,
            "coverage_penalty": coverage_penalty,
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
            "model_score": model_score,
            "auc": auc,
            "aurc": aurc,
            "hit_rate": hit_rate,
            "ret_mae": ret_mae,
            "ret_rmse": ret_rmse,
            "ret_corr": ret_correlation,
            "attn_sum_mean": attn_sum_mean,
            "attn_target_mean": attn_target_mean,
            # Explanation / faithfulness metrics (PDF Section 4.4.2)
            # comprehensiveness = deletion_drop: drop when selected evidence is removed (higher = more faithful)
            "comprehensiveness": del_ins["deletion_drop"],
            "deletion_drop": del_ins["deletion_drop"],    # alias for comprehensiveness
            # insertion_gain: gain from adding selected articles over market-only baseline
            "insertion_gain": del_ins["insertion_gain"],
            # sufficiency_drop: drop when using ONLY selected evidence (lower = more sufficient)
            "sufficiency_drop": sufficiency_score,
            "factor_consistency": factor_consistency_score,
        })

        return val_losses


    # ── fit() helpers ──────────────────────────────────────────────────────────

    def _compute_class_weights(self, train_loader) -> None:
        """Compute sqrt-softened class weights from training labels (idempotent)."""
        if self.class_weights is not None:
            return

        all_labels = []
        for batch in train_loader:
            all_labels.extend(batch["direction"].cpu().numpy())
        all_labels = np.array(all_labels)

        # Balanced weights, then sqrt-soften to reduce extreme ratios (~4.5x → ~2.1x)
        weights = compute_class_weight('balanced', classes=np.array([0, 1, 2]), y=all_labels)
        weights = np.sqrt(weights)
        weights /= weights.mean()   # normalize to mean=1

        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.loss_fn.class_weights = self.class_weights.clone().detach().to(self.device)
        assert self.loss_fn.class_weights.device.type == torch.device(self.device).type, \
            f"Class weights on {self.loss_fn.class_weights.device}, model on {self.device}"

        dist = np.bincount(all_labels)
        logger.info("Class weights (sqrt-softened): DOWN=%.4f NEUTRAL=%.4f UP=%.4f | dist=%s",
                    weights[0], weights[1], weights[2], dist)

    def _compute_return_scale(self, train_loader) -> None:
        """Fit a robust train-only scale for return regression.

        This keeps Eq.32 as SmoothL1 regression, but rescales tiny crypto returns so the
        auxiliary return head contributes meaningful gradient signal without changing the
        predicted target semantics.
        """
        all_returns = []
        for batch in train_loader:
            all_returns.extend(batch["return"].cpu().numpy())

        all_returns = np.asarray(all_returns, dtype=np.float32)
        abs_returns = np.abs(all_returns)
        q75 = float(np.quantile(abs_returns, 0.75)) if abs_returns.size else 0.0
        std = float(np.std(all_returns)) if all_returns.size else 0.0
        scale = max(q75, std, 1e-4)
        self.loss_fn.return_scale.fill_(scale)
        logger.info("Return scale (train-only robust): %.6f | q75_abs=%.6f std=%.6f", scale, q75, std)

    def _compute_factor_label_stats(self, train_loader) -> None:
        """Estimate train-only factor label sharpness and apply mild smoothing if needed.

        Mistral/LLM pseudo-labels can become near one-hot for most articles. A very
        small amount of soft-label smoothing keeps Eq.33 informative without letting
        pseudo-label certainty dominate the joint objective.
        """
        all_max_probs = []
        all_entropy = []

        for batch in train_loader:
            fac = batch["factor"].float()
            mask = batch.get("article_mask")
            if mask is not None:
                valid = mask.bool()
                if valid.any():
                    fac = fac[valid]
                else:
                    continue
            else:
                fac = fac.reshape(-1, fac.shape[-1])

            if fac.numel() == 0:
                continue
            fac = fac.clamp(min=1e-8, max=1.0)
            fac = fac / fac.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            all_max_probs.append(fac.max(dim=-1).values.cpu().numpy())
            all_entropy.append((-(fac * torch.log(fac)).sum(dim=-1)).cpu().numpy())

        if not all_max_probs:
            self.loss_fn.factor_label_smoothing.fill_(0.0)
            self.loss_fn.factor_label_mean_max_prob.fill_(0.0)
            self.loss_fn.factor_label_mean_entropy.fill_(0.0)
            logger.info("Factor label stats unavailable; smoothing disabled.")
            return

        mean_max_prob = float(np.concatenate(all_max_probs, axis=0).mean())
        mean_entropy = float(np.concatenate(all_entropy, axis=0).mean())
        num_factor_classes = int(FACTOR_CLASSES)
        entropy_ratio = mean_entropy / max(np.log(max(num_factor_classes, 2)), 1e-8)

        if mean_max_prob >= 0.95 or entropy_ratio <= 0.10:
            smoothing = 0.04
        elif mean_max_prob >= 0.88 or entropy_ratio <= 0.18:
            # Lowered from 0.90 → 0.88: fold 1 had mean_max_prob=0.893 and got
            # smoothing=0.000 while fold 2 (0.903) got smoothing=0.020 — inconsistent
            # behaviour on the same data distribution. Both deserve mild smoothing.
            smoothing = 0.02
        else:
            smoothing = 0.0

        self.loss_fn.factor_label_smoothing.fill_(smoothing)
        self.loss_fn.factor_label_mean_max_prob.fill_(mean_max_prob)
        self.loss_fn.factor_label_mean_entropy.fill_(mean_entropy)
        logger.info(
            "Factor label stats (train-only): mean_max_prob=%.4f mean_entropy=%.4f "
            "| smoothing=%.3f",
            mean_max_prob,
            mean_entropy,
            smoothing,
        )

    def _log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
    ) -> None:
        """Print one-line epoch summary + active loss components."""
        def _to_float(x):
            return x.item() if hasattr(x, 'item') else float(x)
        active = " + ".join(
            f"{n}x{_to_float(w):.2f}" for n, w in [
                ("Ldir", self.loss_fn.lambda1), ("Lret", self.loss_fn.lambda2),
                ("Lfac", self.loss_fn.lambda3), ("Lsel", self.loss_fn.lambda4),
                ("Lcal", self.loss_fn.lambda5), ("Lfaith", self.loss_fn.lambda6),
                ("Lrisk", self.loss_fn.lambda7),
            ] if _to_float(w) > 0
        )
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_losses['loss']:.4f} val={val_losses['loss']:.4f} "
              f"acc={val_losses['dir_acc']:.3f} | "
              f"Ldir={train_losses['Ldir']:.3f} Lret={train_losses['Lret']:.6f} "
              f"Lfac={train_losses['Lfac']:.3f} Lsel={train_losses['Lsel']:.3f} "
              f"Lcal={train_losses['Lcal']:.3f} Lfaith={train_losses['Lfaith']:.3f} "
              f"Lrisk={train_losses['Lrisk']:.3f}")
        print(f"         | [{active}]")

    def _log_metrics(self, val_losses: Dict[str, float]) -> None:
        """Print the [Metrics] line after early-stopping decision."""
        print(f"  [Metrics] F1={val_losses['macro_f1']:.3f} MCC={val_losses['mcc']:.3f} "
              f"ECE={val_losses['ece']:.4f} AUC={val_losses['auc']:.3f} | "
              f"Sharpe={val_losses['alert_sharpe']:.3f} Cov={val_losses['alert_coverage']:.3f} "
              f"Prec={val_losses['alert_precision']:.3f} PnL={val_losses.get('pnl', 0.0):.4f} | "
              f"RetCorr={val_losses['ret_corr']:.4f} | Score={val_losses['model_score']:.4f}")
        print(f"           policy={val_losses['policy_source']} "
              f"tau={val_losses['tau']:.3f} gamma={val_losses['gamma']:.3f} "
              f"temp={val_losses['temperature']:.3f} "
              f"(seed={val_losses['tau_seed']:.3f}/{val_losses['gamma_seed']:.3f}) "
              f"Sortino={val_losses['alert_sortino']:.3f} Calmar={val_losses['alert_calmar']:.3f} "
              f"MDD={val_losses['alert_max_dd']:.3f} "
              f"AttnSum={val_losses.get('attn_sum_mean', 0.0):.3f}/"
              f"{val_losses.get('attn_target_mean', 0.0):.3f} "
              f"LfacSmooth={float(self.loss_fn.factor_label_smoothing.item()):.3f}")
        print(f"           [Explain] Comp={val_losses.get('comprehensiveness', 0.0):.3f} "
              f"SufDrop={val_losses.get('sufficiency_drop', 0.0):.3f} "
              f"Del={val_losses.get('deletion_drop', 0.0):.3f} "
              f"Ins={val_losses.get('insertion_gain', 0.0):.3f} "
              f"FacCons={val_losses.get('factor_consistency', 0.0):.3f}")

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
            torch.save({
                "model_state":    self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch":          epoch,
                "val_loss":       val_losses['loss'],
                "val_acc":        val_losses['dir_acc'],
                "model_score":    score,
                "macro_f1":       val_losses['macro_f1'],
                "ece":            val_losses['ece'],
                "alert_sharpe":   val_losses['alert_sharpe'],
            }, ckpt)
            print(f"  [BEST] Score={score:.4f} F1={val_losses['macro_f1']:.3f} "
                  f"ECE={val_losses['ece']:.4f} Sharpe={val_losses['alert_sharpe']:.3f}")
            return best_score, ckpt, patience_counter
        else:
            return best_score, None, patience_counter + 1

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
    ) -> Tuple[Dict, Path]:
        """Copy best (or last) checkpoint to FINAL and persist training metrics JSON."""
        final_path = checkpoint_dir / f"safe_alert_{self.horizon}_FINAL.pt"

        # FINAL model is always the best Stage 3 checkpoint (PDF Section 4.5.1).
        # Stage 3 has all loss components active (Lcal+Lfaith+Lrisk) — it is the only
        # fully calibrated and risk-aware model. If Stage 3 underperforms Stage 2, the
        # solution is more epochs, not using an under-calibrated Stage 2 model as FINAL.
        if best_ckpt_path is not None and best_ckpt_path.exists():
            shutil.copy(best_ckpt_path, final_path)
            print(f"[OK] FINAL model = best Stage 3 (score={best_score:.4f}): {best_ckpt_path.name}")
            if best_overall_score > best_score + 0.02:
                print(f"[INFO] best_overall score={best_overall_score:.4f} "
                      f"(Stage 1/2) was higher — consider running more epochs.")
        elif last_ckpt_path is not None and last_ckpt_path.exists():
            shutil.copy(last_ckpt_path, final_path)
            print(f"[WARN] No Stage 3 best found — FINAL = last epoch ({final_epoch})")
        else:
            final_path = last_ckpt_path  # absolute fallback

        metrics = {
            "final_epoch":         final_epoch,
            "final_val_loss":      float(final_val_loss),
            "final_val_acc":       float(final_val_acc),
            "total_epochs_trained": final_epoch,
            "model_file":          final_path.name if final_path else "unknown",
            "note": "FINAL model from Stage 3 optimizes Sharpe (Lrisk), not accuracy!",
        }
        if best_overall_ckpt_path is not None and best_overall_ckpt_path.exists():
            metrics["best_overall_model_file"] = best_overall_ckpt_path.name
            metrics["best_overall_model_score"] = float(best_overall_score)
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
    ) -> Tuple[Dict, Path]:
        """Train with 3-stage curriculum and stage-aware early stopping (PDF Section 4.5.1).

        Stages:
          1 (0–20%): direction + return + selection — build direction/return foundation.
          2 (20–60%): warm-up Lfac + Lfaith + small Lcal/Lrisk — smooth λ transitions.
          3 (60–100%): full loss active — early stopping on model_score.
        Model selection optimizes Score = 0.40·F1 + 0.35·Sharpe − 0.15·ECE + 0.10·MCC.
        """
        checkpoint_dir = (checkpoint_dir or ARTIFACT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._compute_class_weights(train_loader)
        self._compute_return_scale(train_loader)
        self._compute_factor_label_stats(train_loader)

        # Re-init scheduler for this run (T_0 matches __init__ default; T_mult=2 doubles cycle)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )

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
        last_ckpt_path = None
        final_epoch, final_val_loss, final_val_acc = 0, 0.0, 0.0

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._update_loss_weights_for_stage(epoch, epochs)

            train_losses = self.train_epoch(train_loader)
            val_losses   = self.validate(val_loader)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._log_epoch(epoch, epochs, train_losses, val_losses)

            # Stage-aware early stopping: only active in Stage 3.
            # Stage 1-2 have near-zero Sharpe by design; tracking score there would
            # set an unreachable baseline before calibration is active.
            best_overall_score, overall_ckpt, _ = self._try_save_best_checkpoint(
                epoch, val_losses, best_overall_score, 0, checkpoint_dir, tag="best_overall"
            )
            if overall_ckpt:
                best_overall_ckpt_path = overall_ckpt

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
            torch.save({"model_state": self.model.state_dict(), "epoch": epoch,
                        "val_loss": final_val_loss, "val_acc": final_val_acc}, last_ckpt_path)

            self.scheduler.step()

        return self._finalize_training(
            checkpoint_dir, best_ckpt_path, best_overall_ckpt_path, last_ckpt_path,
            best_score, best_overall_score, final_epoch, final_val_loss, final_val_acc,
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
    default_config = script_file.parent / "train_config.yaml"

    # ── Step 1: parse --config only (before loading other args) ──────────────
    pre = ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=default_config)
    pre_args, _ = pre.parse_known_args()

    cfg = _load_config(pre_args.config)

    # ── Step 2: full parser — CLI values override config file ─────────────────
    parser = ArgumentParser(
        description="Train SAFE-Alert. Edit train_config.yaml to avoid long CLI args."
    )
    parser.add_argument("--config",          type=Path,  default=default_config)
    parser.add_argument("--symbol",          default=cfg.get("symbol", "BTCUSDT"))
    parser.add_argument("--horizon",         default=cfg.get("horizon", "1h"), choices=["15m", "1h", "4h", "24h"])
    parser.add_argument("--data_path",       type=Path,  default=default_training_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_training_data)
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--epochs",          type=int,   default=cfg.get("epochs", 60))
    parser.add_argument("--batch_size",      type=int,   default=cfg.get("batch_size", 8))
    parser.add_argument("--lr",              type=float, default=cfg.get("lr", 3e-4))
    parser.add_argument("--grad_clip",       type=float, default=cfg.get("grad_clip", 1.0))
    parser.add_argument("--walk_forward",    action="store_true",
                        default=cfg.get("walk_forward", False),
                        help="Walk-forward cross-validation (PDF Eq.1)")
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

    # Device: config "auto" → detect; explicit "cuda"/"cpu" → use as-is
    _cfg_device = cfg.get("device", "auto")
    _default_device = ("cuda" if torch.cuda.is_available() else "cpu") if _cfg_device == "auto" else _cfg_device
    parser.add_argument("--device", default=_default_device)

    args = parser.parse_args()

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
            except Exception:
                _emb_model_name = "bert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(_emb_model_name)
                model = AutoModel.from_pretrained(_emb_model_name)
                print(f"[EMBED] FinBERT unavailable, using {_emb_model_name}")
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
        if precomp_meta_path.exists():
            try:
                with open(precomp_meta_path, "r", encoding="utf-8") as f:
                    pre_meta = json.load(f)
                meta_rows = int(pre_meta.get("rows", -1))
                if meta_rows > 0 and meta_rows != len(precomputed_features):
                    raise ValueError(
                        "features_precomputed.meta.json row count mismatch with .npy: "
                        f"meta={meta_rows}, npy={len(precomputed_features)}. Recompute precomputed features."
                    )
                ts_min_meta = pd.to_datetime(pre_meta.get("timestamp_min"), errors="coerce")
                ts_min_data = pd.to_datetime(candle_df["timestamp"].min(), errors="coerce")
                if pd.notna(ts_min_meta) and pd.notna(ts_min_data) and ts_min_meta != ts_min_data:
                    print(
                        "[WARN] precompute timestamp_min differs from current candles; "
                        "consider recomputing features_precomputed.npy for strict reproducibility."
                    )
            except Exception as e:
                print(f"[WARN] Could not validate precompute metadata: {e}")
        else:
            print("[WARN] features_precomputed.meta.json not found; alignment checks are limited.")

        if len(precomputed_features) != len(candle_df):
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
        print(f"[OK] Loaded precomputed features {precomputed_features.shape} from {precomp_path.name}")
        print(f"     [SPEEDUP] Training will be 5-10x faster!")
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

    # Create dataset with precomputed features (or None for fallback)
    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=args.horizon,
        precomputed_features=precomputed_features,
        factor_labels=factor_labels_np,
        entity_sentiment=entity_sentiment_np,
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

        # Temporary reference model to get K_h (instantiated once for logging)
        _ref_model = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
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

            dataset.fit_market_scaler(train_indices)

            fold_train_set = Subset(dataset, train_indices)
            fold_val_set   = Subset(dataset, val_indices)
            fold_test_set  = Subset(dataset, test_indices)

            fold_train_loader = torch.utils.data.DataLoader(
                fold_train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
            fold_val_loader = torch.utils.data.DataLoader(
                fold_val_set,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )
            fold_test_loader = torch.utils.data.DataLoader(
                fold_test_set,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory and args.device.startswith("cuda"),
                persistent_workers=args.persistent_workers and args.num_workers > 0,
                prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            )

            # Fresh model + trainer for each fold (no state bleed between folds)
            fold_model = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
            fold_trainer = SAFEAlertTrainer(
                fold_model,
                device=args.device,
                lr=args.lr,
                horizon=args.horizon,
                K_h=K_h,
                grad_clip=args.grad_clip,
                policy_method=args.policy_method,
                tau_percentile=args.tau_percentile,
                gamma_percentile=args.gamma_percentile,
                ablation=args.ablation,
            )

            fold_ckpt_dir = args.artifact_dir / f"fold_{fold_idx}"
            fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

            fold_metrics, fold_ckpt = fold_trainer.fit(
                fold_train_loader, fold_val_loader,
                epochs=args.epochs,
                checkpoint_dir=fold_ckpt_dir,
                early_stopping_patience=_ES_PATIENCE,
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
            test_metrics = fold_trainer.validate(
                fold_test_loader,
                tau=val_eval_metrics["tau"],
                gamma=val_eval_metrics["gamma"],
                temperature=val_eval_metrics["temperature"],
            )

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
                    "tau": float(val_eval_metrics.get("tau", 0.0)),
                    "gamma": float(val_eval_metrics.get("gamma", 0.0)),
                    "tau_seed": float(val_eval_metrics.get("tau_seed", 0.0)),
                    "gamma_seed": float(val_eval_metrics.get("gamma_seed", 0.0)),
                    "temperature": float(val_eval_metrics.get("temperature", 1.0)),
                    "temperature_nll": float(val_eval_metrics.get("temperature_nll", 0.0)),
                    "temperature_source": str(val_eval_metrics.get("temperature_source", "unknown")),
                    "policy_source": str(val_eval_metrics.get("policy_source", "unknown")),
                    "policy_objective": float(val_eval_metrics.get("policy_objective", 0.0) or 0.0),
                    "macro_f1": float(val_eval_metrics.get("macro_f1", 0.0)),
                    "mcc": float(val_eval_metrics.get("mcc", 0.0)),
                    "ece": float(val_eval_metrics.get("ece", 0.0)),
                    "alert_precision": float(val_eval_metrics.get("alert_precision", 0.0)),
                    "alert_coverage": float(val_eval_metrics.get("alert_coverage", 0.0)),
                    "alert_sharpe": float(val_eval_metrics.get("alert_sharpe", 0.0)),
                    "alert_sortino": float(val_eval_metrics.get("alert_sortino", 0.0)),
                    "alert_calmar": float(val_eval_metrics.get("alert_calmar", 0.0)),
                    "alert_max_dd": float(val_eval_metrics.get("alert_max_dd", 0.0)),
                    "model_score": float(val_eval_metrics.get("model_score", 0.0)),
                    "attn_sum_mean": float(val_eval_metrics.get("attn_sum_mean", 0.0)),
                    "attn_target_mean": float(val_eval_metrics.get("attn_target_mean", 0.0)),
                    "comprehensiveness": float(val_eval_metrics.get("comprehensiveness", 0.0)),
                    "deletion_drop": float(val_eval_metrics.get("deletion_drop", 0.0)),
                    "insertion_gain": float(val_eval_metrics.get("insertion_gain", 0.0)),
                    "sufficiency_drop": float(val_eval_metrics.get("sufficiency_drop", 0.0)),
                    "factor_consistency": float(val_eval_metrics.get("factor_consistency", 0.0)),
                },
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
        print("=" * 60)
        print(f"WALK-FORWARD SUMMARY ({len(fold_metrics_list)} folds)")
        print("=" * 60)

        val_loss_avg = float(np.mean([m["val"]["final_val_loss"] for m in fold_metrics_list]))
        val_acc_avg = float(np.mean([m["val"]["final_val_acc"] for m in fold_metrics_list]))
        test_f1_avg = float(np.mean([m["test"]["macro_f1"] for m in fold_metrics_list]))
        test_mcc_avg = float(np.mean([m["test"]["mcc"] for m in fold_metrics_list]))
        test_ece_avg = float(np.mean([m["test"]["ece"] for m in fold_metrics_list]))
        test_auc_avg = float(np.mean([m["test"]["auc"] for m in fold_metrics_list]))
        test_prec_avg = float(np.mean([m["test"]["alert_precision"] for m in fold_metrics_list]))
        test_cov_avg = float(np.mean([m["test"]["alert_coverage"] for m in fold_metrics_list]))
        test_sharpe_avg = float(np.mean([m["test"]["alert_sharpe"] for m in fold_metrics_list]))
        test_sortino_avg = float(np.mean([m["test"]["alert_sortino"] for m in fold_metrics_list]))
        test_calmar_avg = float(np.mean([m["test"]["alert_calmar"] for m in fold_metrics_list]))
        test_max_dd_avg = float(np.mean([m["test"]["alert_max_dd"] for m in fold_metrics_list]))
        test_score_avg = float(np.mean([m["test"]["model_score"] for m in fold_metrics_list]))
        test_pnl_avg = float(np.mean([m["test"]["pnl"] for m in fold_metrics_list]))
        test_comp_avg = float(np.mean([m["test"]["comprehensiveness"] for m in fold_metrics_list]))
        test_suf_drop_avg = float(np.mean([m["test"]["sufficiency_drop"] for m in fold_metrics_list]))
        test_fac_cons_avg = float(np.mean([m["test"]["factor_consistency"] for m in fold_metrics_list]))

        print(f"  avg val_loss: {val_loss_avg:.4f}")
        print(f"  avg val_acc: {val_acc_avg:.4f}")
        print(f"  avg test_f1: {test_f1_avg:.4f}")
        print(f"  avg test_mcc: {test_mcc_avg:.4f}")
        print(f"  avg test_ece: {test_ece_avg:.4f}")
        print(f"  avg test_auc: {test_auc_avg:.4f}")
        print(f"  avg test_alert_precision: {test_prec_avg:.4f}")
        print(f"  avg test_alert_coverage: {test_cov_avg:.4f}")
        print(f"  avg test_alert_sharpe: {test_sharpe_avg:.4f}")
        print(f"  avg test_alert_sortino: {test_sortino_avg:.4f}")
        print(f"  avg test_alert_calmar: {test_calmar_avg:.4f}")
        print(f"  avg test_alert_max_dd: {test_max_dd_avg:.4f}")
        print(f"  avg test_pnl: {test_pnl_avg:.4f}")
        print(f"  avg test_model_score: {test_score_avg:.4f}")
        print(f"  avg test_comprehensiveness: {test_comp_avg:.4f}")
        print(f"  avg test_sufficiency_drop: {test_suf_drop_avg:.4f}")
        print(f"  avg test_factor_consistency: {test_fac_cons_avg:.4f}")

        agg = {
            "final_val_loss": val_loss_avg,
            "final_val_acc": val_acc_avg,
            "test_macro_f1": test_f1_avg,
            "test_mcc": test_mcc_avg,
            "test_ece": test_ece_avg,
            "test_auc": test_auc_avg,
            "test_alert_precision": test_prec_avg,
            "test_alert_coverage": test_cov_avg,
            "test_alert_sharpe": test_sharpe_avg,
            "test_alert_sortino": test_sortino_avg,
            "test_alert_calmar": test_calmar_avg,
            "test_alert_max_dd": test_max_dd_avg,
            "test_pnl": test_pnl_avg,
            "test_model_score": test_score_avg,
            "test_comprehensiveness": test_comp_avg,
            "test_sufficiency_drop": test_suf_drop_avg,
            "test_factor_consistency": test_fac_cons_avg,
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
            print(
                f"[OK] Standardized policy/report saved "
                f"(deploy_fold={deploy_fold}, tau={policy.get('tau')}, gamma={policy.get('gamma')})"
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

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.startswith("cuda"),
            persistent_workers=args.persistent_workers and args.num_workers > 0,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        )  # noqa: F841

        print(f"[OK] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # K_15m=3, K_1h=4, K_4h=5, K_24h=8: per-horizon top-K (selective evidence, PDF Section 3)
        model = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
        K_h = model.K_h_map.get(args.horizon, 8)  # derive from model to avoid mismatch
        print(f"[OK] Model initialized (K_15m=3, K_1h=4, K_4h=5, K_24h=8) - selective article selection")
        print(f"[OK] K_h={K_h} derived from model.K_h_map for horizon={args.horizon}")

        trainer = SAFEAlertTrainer(
            model, device=args.device, lr=args.lr,
            horizon=args.horizon, K_h=K_h, grad_clip=args.grad_clip,
            policy_method=args.policy_method,
            tau_percentile=args.tau_percentile,
            gamma_percentile=args.gamma_percentile,
            ablation=args.ablation,
        )
        metrics, best_ckpt = trainer.fit(
            train_loader, val_loader,
            epochs=args.epochs,
            checkpoint_dir=args.artifact_dir,
            early_stopping_patience=_ES_PATIENCE,
        )

        print(f"\n[COMPLETE] Training complete!")
        print(f"  Final val loss: {metrics['final_val_loss']:.4f}")
        print(f"  Final val acc: {metrics['final_val_acc']:.3f}")
        print(f"  Final epoch: {metrics['final_epoch']}")
        print(f"  Final model: {best_ckpt.name}")


if __name__ == "__main__":
    main()
