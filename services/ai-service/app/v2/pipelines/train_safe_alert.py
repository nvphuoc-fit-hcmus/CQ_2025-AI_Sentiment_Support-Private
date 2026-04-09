"""
Train SAFE-Alert model (PDF Eq.30 multi-objective loss).

Walk-forward training: train on [0:t], validate on [t:t+val_win], test on [t+val_win:t+val_win+test_win]
Shift window forward, repeat until end of data.

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
    compute_auc, compute_coverage_risk_auc, compute_hit_rate,
)
from utils import ARTIFACT_DIR

torch.manual_seed(42)
np.random.seed(42)

# ── Training curriculum constants ──────────────────────────────────────────────
_STAGE1_END_FRAC  = 0.30   # Stage 1 ends after this fraction of total epochs
_STAGE2_END_FRAC  = 0.50   # Stage 2 ends after this fraction of total epochs
_TAU_PERCENTILE   = 70     # Alert threshold τ: top-(100-70)=30% confidence
_GAMMA_PERCENTILE = 60     # Alert threshold γ: top-(100-60)=40% max_prob
_ACCUM_STEPS      = 2      # Gradient accumulation steps (effective_bs = bs * steps)
_ES_PATIENCE      = 12     # Early-stopping patience (Stage 3 only)

# ── Lambda weight schedule (deliberate deviations from PDF defaults) ───────────
# PDF defaults: λ1=1.0, λ2=0.5, λ3=0.3, λ4=0.2, λ5=0.1, λ6=0.1, λ7=0.05
# Deviations are justified in comments; must be documented in thesis Section 4.5.1.
_S1_LAMBDA1 = 2.0   # Ldir 2× (PDF: 1.0) — amplify direction signal in Stage 1
_S1_LAMBDA2 = 1.0   # Lret 2× (PDF: 0.5) — prevent return head collapse
_S1_LAMBDA4 = 0.2   # Lsel start value (matches PDF default)

_S3_LAMBDA1 = 1.0   # Ldir — restore to PDF default in Stage 3
_S3_LAMBDA2 = 1.0   # Lret — keep strong (2× PDF) throughout
_S3_LAMBDA3 = 0.3   # Lfac — matches PDF default
_S3_LAMBDA4 = 0.3   # Lsel — slightly above PDF (0.2) after Stage 2 ramp
_S3_LAMBDA5 = 0.40  # Lcal — 4× PDF (0.1) for ECE < 0.08 target
_S3_LAMBDA6 = 0.25  # Lfaith — 2.5× PDF (0.1) after Stage 2 ramp
_S3_LAMBDA7 = 0.20  # Lrisk — 4× PDF (0.05) to drive Sharpe in Stage 3


class SAFEAlertTrainer:
    """Trainer for SAFE-Alert model with 3-stage training strategy (PDF Section 4.5.1)."""

    def __init__(
        self,
        model: SAFEAlertNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        horizon: str = "1h",
        K_h: int = 8,
        grad_clip: float = 1.0,
        class_weights: torch.Tensor = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.horizon = horizon
        self.K_h = K_h
        self.grad_clip = grad_clip
        self.class_weights = class_weights
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss_fn = MultiObjectiveLoss(
            lambda1=1.0,     # Ldir weight (Eq.31)
            lambda2=0.5,     # Lret weight (Eq.32)
            lambda3=0.3,     # Lfac weight (Eq.33)
            lambda4=0.2,     # Lsel weight (Eq.34)
            lambda5=0.1,     # Lcal weight (Eq.35)
            lambda6=0.1,     # Lfaith weight (Eq.36)
            lambda7=0.05,    # Lrisk weight (Eq.37)
            K_h=K_h,         #  Target # of selected articles (for Lsel)
            eta=0.1,         #  Entropy weight in Lsel (Eq.34)
            faith_margin=0.15,       # Margin m for Lfaith (Eq.36)
            coverage_target=0.35,    # Coverage target κ for Lrisk (Eq.37)
            mu=0.02,                 # Coverage penalty weight μ (Eq.37)
            class_weights=class_weights,
        )

        # CosineAnnealingLR: smooth monotonic decay over all epochs.
        # WarmRestarts was causing LR spikes at stage transitions (T_0=10 → spike at epoch 11
        # right when Stage 2 activates Lfac+Lfaith), leading to loss instability.
        # T_max set at fit() time (reinit) so it matches --epochs argument.
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30, eta_min=1e-6
        )
        self.current_epoch = 0 
    def _prepare_attn_weights(
        self,
        attn_w: torch.Tensor,
        batch_size: int,
        num_articles: int,
    ) -> torch.Tensor:
        """Validate shape and scale attention weights for Lsel (PDF Eq.34).

        SelectiveAttention outputs softmax-normalized weights (sum≈1 per sample).
        Lsel term-1 is (Σα̃_i - K_h)², so weights must be scaled to sum≈K_h.
        """
        if attn_w is None:
            return torch.zeros(batch_size, num_articles, device=self.device)
        if attn_w.dim() == 1:
            attn_w = attn_w.unsqueeze(1)        # (B,) → (B, 1)
        elif attn_w.dim() != 2:
            raise ValueError(f"attn_weights must be 2D (B,K), got {attn_w.shape}")
        # Scale softmax-normalized weights so Σα̃_i ≈ K_h (PDF Eq.34 requirement)
        if attn_w.sum(dim=1).mean() < 2.0:
            attn_w = attn_w * self.loss_fn.K_h
        return attn_w

    def train_epoch(self, train_loader, accumulate_steps: int = _ACCUM_STEPS) -> Dict[str, float]:
        """Train one epoch with gradient accumulation.

        accumulate_steps=2 with batch_size=8 gives effective batch_size=16
        without additional memory overhead.
        """
        self.model.train()
        losses = {"loss": 0, "Ldir": 0, "Lret": 0, "Lfac": 0, "Lsel": 0, "Lcal": 0, "Lfaith": 0, "Lrisk": 0}
        count = 0
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
            outputs = self.model(market_feat, horizon=self.horizon,
                               article_emb=article_emb,
                               article_mask=article_mask,
                               article_meta_vec=article_meta)

            # Masked forward for Lfaith (Eq.36): skip when batch has no articles
            with torch.no_grad():
                # Check if batch has any articles
                has_articles = article_mask.sum(dim=1) > 0  # (B,)
                if has_articles.all():
                    # All samples have articles → compute masked forward
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]  # (B, 3)

                    if self.current_epoch == 1 and count == 0:
                        full_pred = F.softmax(outputs["dir_logits"], dim=-1).max(dim=-1)[0]
                        masked_pred = F.softmax(masked_dir_logits, dim=-1).max(dim=-1)[0]
                        logger.debug("article importance gap (full-masked): %s", (full_pred - masked_pred)[:3])
                else:
                    # Some samples have no articles → fallback to normal logits
                    masked_dir_logits = outputs["dir_logits"]  # Use normal prediction as fallback

            # Validate shape and scale for Lsel (PDF Eq.34): Σα̃_i must ≈ K_h
            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"], market_feat.shape[0], article_emb.shape[1]
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
                logger.warning("Epoch %d batch %d: NaN/Inf loss — Ldir=%.4f Lret=%.4f Lfac=%.4f (skipped)",
                               self.current_epoch, batch_idx,
                               loss_dict["Ldir"], loss_dict["Lret"], loss_dict["Lfac"])
                continue  # Skip this batch

            # Gradient accumulation: scale loss before backward
            (total_loss / accumulate_steps).backward()

            # Step optimizer every accumulate_steps batches (or at end of epoch)
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if (batch_idx + 1) % accumulate_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            for k in losses:
                losses[k] += loss_dict[k]
            count += 1

        for k in losses:
            losses[k] /= max(count, 1)
        return losses

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate and compute multi-metrics (Macro-F1, MCC, ECE, Alert Sharpe)."""
        self.model.eval()
        val_losses = {"loss": 0, "dir_acc": 0, "count": 0}

        # Accumulate for metric computation
        all_dir_preds = []
        all_dir_labels = []
        all_confidence = []
        all_ret_labels = []
        all_ret_preds = []
        all_dir_probs = []  # (N, 3) softmax probs for AUC

        for batch in val_loader:
            market_feat = batch["market_features"].to(self.device)
            article_emb = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
            article_mask = batch["article_mask"].to(self.device)
            dir_labels = batch["direction"].to(self.device)
            fac_labels = batch["factor"].to(self.device)

            outputs = self.model(market_feat, horizon=self.horizon,
                               article_emb=article_emb,
                               article_mask=article_mask,
                               article_meta_vec=article_meta)

            # Only compute masked forward when at least one sample has articles.
            # Skipping it when no articles avoids wasted compute and ensures
            # Lfaith=0 for no-article batches (no selected articles to mask out).
            has_articles = article_mask.any(dim=1).any()  # scalar bool
            with torch.no_grad():
                if has_articles:
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]  # (B, 3)
                else:
                    masked_dir_logits = None  # Lfaith=0 for no-article batches

            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"], market_feat.shape[0], article_emb.shape[1]
            )

            # calibration_targets = correctness indicator for Lcal (Eq.35)
            dir_preds = outputs["dir_logits"].argmax(dim=-1)
            dir_correct = (dir_preds == dir_labels).long()  # 1 if correct, 0 if wrong

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
                article_mask=article_mask,  # Pass mask so Lfac skips padding articles
            )

            # Accumulate loss
            val_losses["loss"] += loss_dict["loss"].item()
            val_losses["count"] += 1

            all_dir_preds.append(dir_preds.cpu().numpy())
            all_dir_labels.append(dir_labels.cpu().numpy())
            all_confidence.append(outputs["confidence"].cpu().numpy())
            all_ret_labels.append(batch["return"].cpu().numpy())
            all_ret_preds.append(outputs["ret_pred"].cpu().numpy())
            all_dir_probs.append(F.softmax(outputs["dir_logits"], dim=-1).cpu().numpy())

            # Direction accuracy
            dir_acc = (dir_preds == dir_labels).float().mean().item()
            val_losses["dir_acc"] += dir_acc

        # Finalize loss/acc
        val_losses["loss"] /= max(val_losses["count"], 1)
        val_losses["dir_acc"] /= max(val_losses["count"], 1)
        del val_losses["count"]

        all_dir_preds = np.concatenate(all_dir_preds, axis=0)
        all_dir_labels = np.concatenate(all_dir_labels, axis=0)
        all_confidence = np.concatenate(all_confidence, axis=0)
        all_ret_labels = np.concatenate(all_ret_labels, axis=0)
        all_ret_preds = np.concatenate(all_ret_preds, axis=0)
        all_dir_probs = np.concatenate(all_dir_probs, axis=0)  # (N, 3)

        # Return prediction metrics
        ret_mae = np.mean(np.abs(all_ret_preds - all_ret_labels))
        ret_rmse = np.sqrt(np.mean((all_ret_preds - all_ret_labels) ** 2))
        ret_correlation = np.corrcoef(all_ret_preds, all_ret_labels)[0, 1] if len(all_ret_preds) > 1 else 0.0

        # Macro-F1 & MCC
        macro_f1 = compute_macro_f1(all_dir_preds, all_dir_labels)
        mcc = compute_mcc(all_dir_preds, all_dir_labels)

        # Calibration
        ece = compute_ece(all_confidence, all_dir_preds, all_dir_labels)
        brier = compute_brier_score(all_confidence, all_dir_preds, all_dir_labels)

        # Fixed percentile thresholds (PDF Eq.26): τ selects top-30% confidence,
        # γ selects top-40% max_prob → AlertCov ~0.20-0.25 with higher precision.
        _max_prob_all = all_dir_probs.max(axis=-1)  # (N,)
        best_tau   = float(np.percentile(all_confidence, _TAU_PERCENTILE))
        best_gamma = float(np.percentile(_max_prob_all,  _GAMMA_PERCENTILE))

        # Alerting metrics with stable thresholds
        alert_prec, alert_cov, sel_risk = compute_alert_precision(
            all_confidence, all_dir_preds, all_dir_labels,
            dir_probs=all_dir_probs,  # real softmax probs for γ threshold (no data leakage)
            tau=best_tau, gamma=best_gamma,
        )

        # Mini backtest with optimized thresholds
        backtest_metrics = mini_backtest(all_confidence, all_dir_preds, all_ret_labels,
                                        tau=best_tau, gamma=best_gamma,
                                        dir_probs=all_dir_probs)
        alert_sharpe = backtest_metrics["alert_sharpe"]

        model_score = compute_model_selection_score(macro_f1, mcc, ece, alert_sharpe)
        auc = compute_auc(all_dir_probs, all_dir_labels)
        aurc = compute_coverage_risk_auc(all_confidence, all_dir_preds, all_dir_labels)
        hit_rate = compute_hit_rate(all_confidence, all_ret_labels)

        val_losses.update({
            "macro_f1":       macro_f1,
            "mcc":            mcc,
            "ece":            ece,
            "brier":          brier,
            "alert_precision": alert_prec,
            "alert_coverage": alert_cov,
            "selective_risk": sel_risk,
            "alert_sharpe":   alert_sharpe,
            "model_score":    model_score,
            "auc":            auc,
            "aurc":           aurc,
            "hit_rate":       hit_rate,
            "ret_mae":        ret_mae,
            "ret_rmse":       ret_rmse,
            "ret_corr":       ret_correlation,
        })

        return val_losses

    def _update_loss_weights_for_stage(self, epoch: int, total_epochs: int):
        """3-stage curriculum (PDF Section 4.5.1) with smooth λ transitions.

        Stage 1 (0–30%): Direction + Return + Selection — build foundation.
        Stage 2 (30–50%): Gradually add Factor + Faithfulness; λ1 ramps 2→1,
                          λ4 ramps 0.2→0.3 so Stage 3 starts with no jump.
        Stage 3 (50–100%): All components fully active.
        """
        stage1_epochs = max(1, int(total_epochs * _STAGE1_END_FRAC))
        stage2_epochs = max(2, int(total_epochs * _STAGE2_END_FRAC))

        if epoch <= stage1_epochs:
            # Stage 1: foundation — direction, return, light selection
            self.loss_fn.lambda1 = _S1_LAMBDA1   # Ldir amplified
            self.loss_fn.lambda2 = _S1_LAMBDA2   # Lret strong from epoch 1
            self.loss_fn.lambda3 = 0.0            # Lfac disabled
            self.loss_fn.lambda4 = _S1_LAMBDA4   # Lsel light pressure
            self.loss_fn.lambda5 = 0.0            # Lcal disabled
            self.loss_fn.lambda6 = 0.0            # Lfaith disabled
            self.loss_fn.lambda7 = 0.0            # Lrisk disabled
            stage_name = "STAGE 1"

        elif epoch <= stage2_epochs:
            # Stage 2: warm-up Lfac + Lfaith; smoothly interpolate λ1 and λ4
            # so Stage 3 starts with no abrupt jumps.
            p = (epoch - stage1_epochs) / max(stage2_epochs - stage1_epochs, 1)
            self.loss_fn.lambda1 = _S1_LAMBDA1 + (_S3_LAMBDA1 - _S1_LAMBDA1) * p  # 2→1
            self.loss_fn.lambda2 = _S1_LAMBDA2                                      # constant
            self.loss_fn.lambda3 = _S3_LAMBDA3 * p                                  # 0→0.3
            self.loss_fn.lambda4 = _S1_LAMBDA4 + (_S3_LAMBDA4 - _S1_LAMBDA4) * p  # 0.2→0.3
            self.loss_fn.lambda5 = 0.0
            self.loss_fn.lambda6 = _S3_LAMBDA6 * p                                  # 0→0.25
            self.loss_fn.lambda7 = 0.0
            stage_name = "STAGE 2"

        else:
            # Stage 3: all components at target values (continuous from Stage 2 end)
            self.loss_fn.lambda1 = _S3_LAMBDA1
            self.loss_fn.lambda2 = _S3_LAMBDA2
            self.loss_fn.lambda3 = _S3_LAMBDA3
            self.loss_fn.lambda4 = _S3_LAMBDA4
            self.loss_fn.lambda5 = _S3_LAMBDA5
            self.loss_fn.lambda6 = _S3_LAMBDA6
            self.loss_fn.lambda7 = _S3_LAMBDA7
            stage_name = "STAGE 3"

        logger.info("Epoch %d/%d [%s] lambdas: dir=%.2f ret=%.2f fac=%.2f sel=%.2f cal=%.2f faith=%.2f risk=%.2f",
                    epoch, total_epochs, stage_name,
                    self.loss_fn.lambda1, self.loss_fn.lambda2, self.loss_fn.lambda3,
                    self.loss_fn.lambda4, self.loss_fn.lambda5, self.loss_fn.lambda6,
                    self.loss_fn.lambda7)

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

    def _log_epoch(
        self,
        epoch: int,
        epochs: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
    ) -> None:
        """Print one-line epoch summary + active loss components."""
        active = " + ".join(
            f"{n}x{w:.2f}" for n, w in [
                ("Ldir", self.loss_fn.lambda1), ("Lret", self.loss_fn.lambda2),
                ("Lfac", self.loss_fn.lambda3), ("Lsel", self.loss_fn.lambda4),
                ("Lcal", self.loss_fn.lambda5), ("Lfaith", self.loss_fn.lambda6),
                ("Lrisk", self.loss_fn.lambda7),
            ] if w > 0
        )
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"train={train_losses['loss']:.4f} val={val_losses['loss']:.4f} "
              f"acc={val_losses['dir_acc']:.3f} | "
              f"Ldir={train_losses['Ldir']:.3f} Lret={train_losses['Lret']:.3f} "
              f"Lfac={train_losses['Lfac']:.3f} Lcal={train_losses['Lcal']:.3f}")
        print(f"         | [{active}]")

    def _log_metrics(self, val_losses: Dict[str, float]) -> None:
        """Print the [Metrics] line after early-stopping decision."""
        print(f"  [Metrics] F1={val_losses['macro_f1']:.3f} MCC={val_losses['mcc']:.3f} "
              f"ECE={val_losses['ece']:.4f} AUC={val_losses['auc']:.3f} | "
              f"Sharpe={val_losses['alert_sharpe']:.3f} Cov={val_losses['alert_coverage']:.3f} "
              f"Prec={val_losses['alert_precision']:.3f} | "
              f"RetCorr={val_losses['ret_corr']:.4f} | Score={val_losses['model_score']:.4f}")

    def _try_save_best_checkpoint(
        self,
        epoch: int,
        val_losses: Dict[str, float],
        best_score: float,
        patience_counter: int,
        checkpoint_dir: Path,
    ) -> Tuple[float, Optional[Path], int]:
        """Save checkpoint if current score improves best; increment patience otherwise.

        Returns:
            (best_score, best_ckpt_path, patience_counter)
        """
        score = val_losses['model_score']
        if score > best_score:
            best_score = score
            patience_counter = 0
            ckpt = checkpoint_dir / f"safe_alert_{self.horizon}_best_epoch{epoch}.pt"
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
        last_ckpt_path: Optional[Path],
        best_score: float,
        final_epoch: int,
        final_val_loss: float,
        final_val_acc: float,
    ) -> Tuple[Dict, Path]:
        """Copy best (or last) checkpoint to FINAL and persist training metrics JSON."""
        final_path = checkpoint_dir / f"safe_alert_{self.horizon}_FINAL.pt"

        if best_ckpt_path is not None and best_ckpt_path.exists():
            shutil.copy(best_ckpt_path, final_path)
            print(f"[OK] FINAL model = best Stage 3 (score={best_score:.4f}): {best_ckpt_path.name}")
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
          1 (0–30%): direction + return + selection — build foundation.
          2 (30–50%): warm-up factor + faithfulness — smooth λ transitions.
          3 (50–100%): all components — early stopping on model_score.
        Model selection optimizes Score = 0.40·F1 + 0.35·Sharpe − 0.15·ECE + 0.10·MCC.
        """
        checkpoint_dir = (checkpoint_dir or ARTIFACT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._compute_class_weights(train_loader)

        # Re-init scheduler with correct T_max (placeholder T_max=30 set in __init__)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        stage3_epochs  = max(2, int(epochs * _STAGE2_END_FRAC))
        patience_counter = 0
        best_score     = float('-inf')
        best_ckpt_path = None
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
            if epoch == stage3_epochs:
                # First epoch of Stage 3: reset counters so Stage 2 results
                # don't influence the Stage 3 baseline (fresh start).
                patience_counter = 0
                best_score = float('-inf')

            if epoch >= stage3_epochs:
                best_score, new_ckpt, patience_counter = self._try_save_best_checkpoint(
                    epoch, val_losses, best_score, patience_counter, checkpoint_dir
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
            checkpoint_dir, best_ckpt_path, last_ckpt_path,
            best_score, final_epoch, final_val_loss, final_val_acc,
        )


def walk_forward_split(dataset_len: int, n_folds: int = 5, test_frac: float = 0.15):
    """Generate walk-forward (expanding window) train/val index splits.

    Implements PDF Eq.1: train on [0:t], validate on [t:t+val_win], repeat.

    The last `test_frac` of data is held out as a fixed test set and is
    never used during training or fold selection.

    Args:
        dataset_len: Total number of samples in the dataset.
        n_folds:     Number of walk-forward folds (default 5).
        test_frac:   Fraction of data reserved as a fixed hold-out test set.

    Yields:
        (train_indices, val_indices): index lists for each fold.
            train_indices always starts from 0 (expanding window).
            val_indices slides forward by one val_window per fold.
    """
    # Fixed test set: last test_frac of all data — never touched during training
    trainval_end = int(dataset_len * (1.0 - test_frac))  # e.g. 85% boundary

    # Each validation window covers (trainval_end / n_folds) samples
    val_win = trainval_end // n_folds

    for fold in range(n_folds):
        val_start = val_win * fold          # first val index for this fold
        val_end   = val_win * (fold + 1)    # exclusive end of val window

        # Clamp to trainval_end to avoid bleeding into the test set
        val_end = min(val_end, trainval_end)

        # Training always starts from index 0 (expanding window)
        train_end = val_start

        # Need at least 1 training sample
        if train_end < 1:
            continue

        train_indices = list(range(0, train_end))
        val_indices   = list(range(val_start, val_end))

        yield train_indices, val_indices


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
    parser.add_argument("--horizon",         default=cfg.get("horizon", "1h"), choices=["1h", "4h", "24h"])
    parser.add_argument("--data_path",       type=Path,  default=default_training_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_training_data)
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--epochs",          type=int,   default=cfg.get("epochs", 60))
    parser.add_argument("--batch_size",      type=int,   default=cfg.get("batch_size", 8))
    parser.add_argument("--lr",              type=float, default=cfg.get("lr", 0.001))
    parser.add_argument("--grad_clip",       type=float, default=cfg.get("grad_clip", 1.0))
    parser.add_argument("--walk_forward",    action="store_true",
                        default=cfg.get("walk_forward", False),
                        help="Walk-forward cross-validation (PDF Eq.1)")
    parser.add_argument("--n_folds",         type=int,   default=cfg.get("n_folds", 5))

    # Device: config "auto" → detect; explicit "cuda"/"cpu" → use as-is
    _cfg_device = cfg.get("device", "auto")
    _default_device = ("cuda" if torch.cuda.is_available() else "cpu") if _cfg_device == "auto" else _cfg_device
    parser.add_argument("--device", default=_default_device)

    args = parser.parse_args()

    print(f"[START] Training SAFE-Alert {args.symbol} {args.horizon}")
    print(f"  Device: {args.device}")
    print(f"  Data path exists: {args.data_path.exists()} ({args.data_path.name}/)")

    # Load data (try candles_max.csv first, then others)
    candles_max = args.data_path / "candles_max.csv"
    data_path = args.data_path / f"{args.symbol.lower()}_training_dataset_v2.csv"
    aligned_path = args.data_path / "candles_aligned.csv"

    if candles_max.exists():
        data_path = candles_max
        print(f"[OK] Using candles_max: {data_path.name}")
    elif aligned_path.exists():
        data_path = aligned_path
        print(f"[OK] Using aligned candles: {data_path.name}")
    elif not data_path.exists():
        print(f"[ERROR] Data not found: candles_max.csv, {data_path.name} or candles_aligned.csv")
        sys.exit(1)

    candle_df = pd.read_csv(data_path)
    candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'])

    # Restrict to article coverage period (2020-05 to 2023-12-19).
    # 2024-2026 has zero articles — walk-forward test folds there give NaN metrics.
    # Keep 2020-2022 for market encoder diversity (bull/bear regimes).
    _article_end = pd.Timestamp("2023-12-31")
    n_before = len(candle_df)
    candle_df = candle_df[candle_df['timestamp'] <= _article_end].reset_index(drop=True)
    if len(candle_df) < n_before:
        print(f"[OK] Restricted candles to <= {_article_end.date()}: "
              f"{n_before} -> {len(candle_df)} ({len(candle_df)/n_before*100:.1f}%)")

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

    # ── Walk-forward cross-validation (PDF Eq.1) ──────────────────────────────
    if args.walk_forward:
        n_folds    = args.n_folds
        test_frac  = 0.15
        dataset_len = len(dataset)

        # Fixed test set: last 15% — printed for reference, never used in training
        test_start = int(dataset_len * (1.0 - test_frac))
        print(f"[WALK-FORWARD] {n_folds} folds | dataset={dataset_len} | "
              f"fixed test set: indices {test_start}-{dataset_len-1} "
              f"({dataset_len - test_start} samples, NEVER used in training)")
        print()

        # Temporary reference model to get K_h (instantiated once for logging)
        _ref_model = SAFEAlertNet(market_dim=63, has_news=True, K_1h=4, K_4h=5, K_24h=8)
        K_h = _ref_model.K_h_map.get(args.horizon, 8)
        del _ref_model

        fold_metrics_list = []  # collect val metrics per fold

        folds = list(walk_forward_split(dataset_len, n_folds=n_folds, test_frac=test_frac))
        for fold_idx, (train_indices, val_indices) in enumerate(folds, start=1):
            print(f"[FOLD {fold_idx}/{n_folds}] train=0-{train_indices[-1]} "
                  f"({len(train_indices)} samples) | "
                  f"val={val_indices[0]}-{val_indices[-1]} "
                  f"({len(val_indices)} samples)")

            fold_train_set = Subset(dataset, train_indices)
            fold_val_set   = Subset(dataset, val_indices)

            fold_train_loader = torch.utils.data.DataLoader(
                fold_train_set, batch_size=args.batch_size, shuffle=True
            )
            fold_val_loader = torch.utils.data.DataLoader(
                fold_val_set, batch_size=args.batch_size
            )

            # Fresh model + trainer for each fold (no state bleed between folds)
            fold_model = SAFEAlertNet(market_dim=63, has_news=True, K_1h=4, K_4h=5, K_24h=8)
            fold_trainer = SAFEAlertTrainer(
                fold_model,
                device=args.device,
                lr=args.lr,
                horizon=args.horizon,
                K_h=K_h,
                grad_clip=args.grad_clip,
            )

            fold_ckpt_dir = args.artifact_dir / f"fold_{fold_idx}"
            fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

            fold_metrics, fold_ckpt = fold_trainer.fit(
                fold_train_loader, fold_val_loader,
                epochs=args.epochs,
                checkpoint_dir=fold_ckpt_dir,
                early_stopping_patience=_ES_PATIENCE,
            )
            fold_metrics_list.append(fold_metrics)

            print(f"[FOLD {fold_idx}/{n_folds}] Done — "
                  f"val_loss={fold_metrics['final_val_loss']:.4f} "
                  f"val_acc={fold_metrics['final_val_acc']:.3f}")
            print()

            # Free GPU memory between folds
            del fold_model, fold_trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ── Aggregate metrics across folds ────────────────────────────────────
        # Each fold_metrics dict only contains the _finalize_training keys.
        # Richer per-epoch val metrics are not directly available here, so we
        # report what _finalize_training guarantees (final_val_loss, final_val_acc).
        print("=" * 60)
        print(f"WALK-FORWARD SUMMARY ({n_folds} folds)")
        print("=" * 60)

        summary_keys = ["final_val_loss", "final_val_acc", "final_epoch",
                        "total_epochs_trained"]
        agg: Dict[str, float] = {}
        for k in summary_keys:
            vals = [m[k] for m in fold_metrics_list if k in m]
            if vals:
                agg[k] = float(np.mean(vals))

        for k, v in agg.items():
            print(f"  avg {k}: {v:.4f}")

        # Persist fold results
        wf_results = {
            "n_folds":      n_folds,
            "dataset_len":  dataset_len,
            "test_start":   test_start,
            "epochs_per_fold": args.epochs,
            "fold_metrics": fold_metrics_list,
            "averages":     agg,
        }
        wf_results_path = args.artifact_dir / "walk_forward_results.json"
        with open(wf_results_path, "w") as f:
            json.dump(wf_results, f, indent=2)
        print(f"\n[OK] Fold results saved: {wf_results_path}")

    else:
        # ── Fixed 70/15/15 temporal split (original behaviour, unchanged) ─────
        train_size = int(0.7 * len(dataset))
        val_size   = int(0.15 * len(dataset))
        test_size  = len(dataset) - train_size - val_size

        train_set = Subset(dataset, range(0, train_size))
        val_set   = Subset(dataset, range(train_size, train_size + val_size))
        test_set  = Subset(dataset, range(train_size + val_size, len(dataset)))

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size)  # noqa: F841

        print(f"[OK] Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

        # K_1h=4, K_4h=5: select top-K articles (selective evidence, PDF Section 3)
        model = SAFEAlertNet(market_dim=63, has_news=True, K_1h=4, K_4h=5, K_24h=8)
        K_h = model.K_h_map.get(args.horizon, 8)  # derive from model to avoid mismatch
        print(f"[OK] Model initialized (K_1h=4, K_4h=5, K_24h=8) - selective article selection")
        print(f"[OK] K_h={K_h} derived from model.K_h_map for horizon={args.horizon}")

        trainer = SAFEAlertTrainer(
            model, device=args.device, lr=args.lr,
            horizon=args.horizon, K_h=K_h, grad_clip=args.grad_clip
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
