"""
Ablation study for SAFE-Alert (PDF Section 5.2).

8 ablation variants are tested:
  1. w/o_selective_news  -- uniform average over ALL articles, no Top-K gating
  2. w/o_factor          -- z_fac = zeros (factor module disabled)
  3. w/o_market          -- z_mkt = zeros (market encoder disabled)
  4. w/o_confidence      -- fixed confidence = 0.5 (confidence head disabled)
  5. w/o_faithfulness    -- lambda6 = 0 throughout training (Lfaith disabled)
  6. w/o_horizon         -- h_emb = zeros (horizon conditioning disabled)
  7. w/o_lrisk           -- lambda7 = 0 throughout training (Lrisk disabled)
  8. w/o_bar_sequences   -- scalar market encoder instead of Eq.16 bar sequences

Usage:
  python run_ablation.py --symbol BTCUSDT --horizon 1h --epochs 30
"""

import sys
import json
import gc
import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Path setup — identical to train_safe_alert.py
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))   # v2/
sys.path.insert(0, str(_SCRIPT_DIR))           # pipelines/

from models.safe_alert_net import SAFEAlertNet
from safe_alert_dataset import SAFEAlertDataset
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_auc,
    compute_alert_precision, mini_backtest, compute_model_selection_score,
    fit_temperature_scaling, apply_temperature_to_logits,
)
from train_safe_alert import SAFEAlertTrainer, _ES_PATIENCE
# Session 26 — import the train_safe_alert module itself so we can access
# _TOP_K_MAP / _apply_config_overrides via module attributes at USE time,
# not at IMPORT time. Python's `from module import name` captures the
# object at import — if `name` is later rebound via global assignment in
# the module, the imported alias still points to the old object.
# `import module` + `module.name` reads fresh every access.
import train_safe_alert as _train
from models.safe_alert_net import FACTOR_CLASSES as _FACTOR_CLASSES
from utils import ARTIFACT_DIR

torch.manual_seed(42)
np.random.seed(42)

# ── Ablation variant definitions ───────────────────────────────────────────────
# Each entry: (variant_name, description, ablation_flag_or_None, lambda6_override, lambda7_override)
# ablation_flag is passed to model.forward(ablation=...) inside a patched trainer.
# lambda6_override: if not None, clamps lambda6 to this value every epoch.
# lambda7_override: if not None, clamps lambda7 to this value every epoch.
ABLATION_VARIANTS = [
    ("w/o_selective_news",
     "Uniform average over all articles (no Top-K gating)",
     "w/o_selective_news", None, None),
    ("w/o_factor",
     "Factor module disabled (z_fac = 0)",
     "w/o_factor", None, None),
    ("w/o_market",
     "Market encoder disabled (z_mkt = 0, news-only prediction)",
     "w/o_market", None, None),
    ("w/o_confidence",
     "Confidence head disabled (fixed conf = 0.5)",
     "w/o_confidence", None, None),
    ("w/o_faithfulness",
     "Faithfulness loss disabled (lambda6 = 0 throughout)",
     None, 0.0, None),
    ("w/o_horizon",
     "Horizon conditioning disabled (h_emb = 0)",
     "w/o_horizon", None, None),
    ("w/o_lrisk",
     "Selective risk loss disabled (lambda7 = 0 throughout)",
     None, None, 0.0),
    # Session 23 P0 #3: Contribution 3 ablation — trains a SEPARATE model
    # with use_bar_sequences=False so the market encoder degenerates to the
    # legacy scalar path (63-dim indicators instead of (L, d) bar sequences
    # per timeframe per paper Eq.16). Comparing main vs this ablation
    # quantifies the empirical lift from paper-faithful temporal modeling.
    # Mechanism: when AblationTrainer sees ablation_flag="w/o_bar_sequences",
    # the model constructor at ``_use_bar_sequences`` guard below receives
    # False and builds with the legacy scalar encoder. The ablation_flag
    # string is NOT consumed at forward-time — the switch is purely
    # construction-time (different model, same config elsewhere).
    ("w/o_bar_sequences",
     "Bar-sequence market encoder disabled (legacy 63-dim scalar path, paper Eq.16 simplification)",
     "w/o_bar_sequences", None, None),
]


# ── Ablation-aware subclass of SAFEAlertTrainer ────────────────────────────────

class AblationTrainer(SAFEAlertTrainer):
    """Extends SAFEAlertTrainer to inject the ablation flag into every forward pass.

    Three mechanisms:
    - ablation_flag: string passed to model.forward(ablation=...) — modifies the
                     model's computation graph without touching weight tensors.
    - lambda6_force: if set, overrides loss_fn.lambda6 to this value every epoch
                     so the Lfaith term is permanently disabled (w/o_faithfulness).
    - lambda7_force: if set, overrides loss_fn.lambda7 to this value every epoch
                     so the Lrisk term is permanently disabled (w/o_lrisk).
    """

    def __init__(self, *args, ablation_flag: Optional[str] = None,
                 lambda6_force: Optional[float] = None,
                 lambda7_force: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ablation_flag  = ablation_flag
        self.lambda6_force  = lambda6_force
        self.lambda7_force  = lambda7_force

    # Override _update_loss_weights_for_stage to enforce lambda6_force / lambda7_force
    def _update_loss_weights_for_stage(self, epoch: int, total_epochs: int):
        super()._update_loss_weights_for_stage(epoch, total_epochs)
        if self.lambda6_force is not None:
            with torch.no_grad():
                self.loss_fn.lambda6.fill_(float(self.lambda6_force))
        if self.lambda7_force is not None:
            with torch.no_grad():
                self.loss_fn.lambda7.fill_(float(self.lambda7_force))

    # Override train_epoch to pass ablation flag to every forward call
    def train_epoch(self, train_loader, accumulate_steps=2) -> Dict[str, float]:
        import torch.nn.functional as F
        from models.safe_alert_net import FACTOR_CLASSES

        self.model.train()
        # Include "Lvol" key because loss_dict always reports it. Paper-final
        # runs set lambda_vol=0; lambda_vol>0 opts into the volatility
        # extension.
        losses = {
            "loss": 0, "Ldir": 0, "Lret": 0, "Lfac": 0,
            "Lsel": 0, "Lcal": 0, "Lfaith": 0, "Lrisk": 0,
            "Lvol": 0,
        }
        count = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            market_feat  = batch["market_features"].to(self.device)
            article_emb  = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
            article_mask = batch["article_mask"].to(self.device)
            dir_labels   = batch["direction"].to(self.device)
            ret_labels   = batch["return"].to(self.device)
            fac_labels   = batch["factor"].to(self.device)
            # Session 23 P0 #3: paper Eq.16 bar sequences when dataset emits them.
            market_bars  = batch.get("market_bars")
            if market_bars is not None:
                market_bars = market_bars.to(self.device)

            # Forward with ablation flag
            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                ablation=self.ablation_flag,
                symbol=self.symbol,
                market_bars=market_bars,   # Session 23 P0 #3
            )

            # Faithfulness masked pass — skip for w/o_faithfulness (lambda6=0)
            has_articles = article_mask.sum(dim=1) > 0
            with torch.no_grad():
                if has_articles.all() and self.loss_fn.lambda6 > 0:
                    masked_outputs = self.model.forward_masked(
                        market_feat=market_feat,
                        horizon=self.horizon,
                        article_emb=article_emb,
                        article_mask=article_mask,
                        article_meta_vec=article_meta,
                        symbol=self.symbol,
                        market_bars=market_bars,   # Session 23 P0 #3
                    )
                    masked_dir_logits = masked_outputs["dir_logits"]
                else:
                    masked_dir_logits = outputs["dir_logits"]

            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"],
                article_mask,
                market_feat.shape[0],
                article_emb.shape[1],
            )

            dir_preds  = outputs["dir_logits"].argmax(dim=-1)
            dir_correct = (dir_preds == dir_labels).long()

            fac_probs = (
                outputs["p_fac_all"]
                if outputs["p_fac_all"] is not None
                else torch.zeros(market_feat.shape[0], FACTOR_CLASSES, device=self.device)
            )

            # Optional volatility extension: forward vol labels from dataset
            # to loss when present; inactive for paper-final lambda_vol=0.
            vol_labels_t = batch.get("volatility")
            if vol_labels_t is not None:
                vol_labels_t = vol_labels_t.to(self.device)
            loss_dict = self.loss_fn(
                dir_logits=outputs["dir_logits"],
                dir_labels=dir_labels,
                ret_pred=outputs["ret_pred"],
                ret_labels=ret_labels,
                fac_probs=fac_probs,
                fac_labels=fac_labels,
                attn_weights=attn_w,
                confidence=outputs["confidence"],
                calibration_targets=dir_correct,
                masked_dir_logits=masked_dir_logits,
                article_mask=article_mask,
                soft_gates=outputs.get("soft_gates"),  # Session 20 Fix 7
                vol_pred=outputs.get("vol_pred"),       # optional volatility extension
                vol_labels=vol_labels_t,                # optional volatility extension
            )

            total_loss = loss_dict["loss"]
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            (total_loss / accumulate_steps).backward()

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

    # Override validate to pass ablation flag
    @torch.no_grad()
    def validate(
        self,
        val_loader,
        tau: Optional[float] = None,
        gamma: Optional[float] = None,
        temperature: Optional[float] = None,
        compute_explanations: bool = True,
    ) -> Dict[str, float]:
        import torch.nn.functional as F

        self.model.eval()
        val_losses = {"loss": 0, "dir_acc": 0, "count": 0}

        all_dir_preds  = []
        all_dir_labels = []
        all_confidence = []
        all_ret_labels = []
        all_ret_preds  = []
        all_dir_probs  = []
        all_dir_logits = []

        for batch in val_loader:
            market_feat  = batch["market_features"].to(self.device)
            article_emb  = batch["article_embeddings"].to(self.device)
            article_meta = batch["article_metadata"].to(self.device)
            article_mask = batch["article_mask"].to(self.device)
            dir_labels   = batch["direction"].to(self.device)
            # Session 23 P0 #3 + Session 24: thread bar sequences and
            # volatility labels when the dataset emits them. Without this
            # threading the ablation validate would crash under
            # use_bar_sequences=True (model raises on missing market_bars)
            # and silently drop the Lvol auxiliary loss.
            market_bars  = batch.get("market_bars")
            if market_bars is not None:
                market_bars = market_bars.to(self.device)
            vol_labels   = batch.get("volatility")
            if vol_labels is not None:
                vol_labels = vol_labels.to(self.device)

            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                ablation=self.ablation_flag,
                symbol=self.symbol,
                market_bars=market_bars,    # Session 23 P0 #3
            )

            has_articles = article_mask.any(dim=1).any()
            if has_articles and compute_explanations and self.loss_fn.lambda6 > 0:
                masked_outputs = self.model.forward_masked(
                    market_feat=market_feat,
                    horizon=self.horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta,
                    symbol=self.symbol,
                    market_bars=market_bars,    # Session 23 P0 #3
                )
                masked_dir_logits = masked_outputs["dir_logits"]
            else:
                masked_dir_logits = None

            attn_w = self._prepare_attn_weights(
                outputs["attn_weights"],
                article_mask,
                market_feat.shape[0],
                article_emb.shape[1],
            )

            dir_preds   = outputs["dir_logits"].argmax(dim=-1)
            dir_correct = (dir_preds == dir_labels).long()

            from models.safe_alert_net import FACTOR_CLASSES
            fac_probs = (
                outputs["p_fac_all"]
                if outputs["p_fac_all"] is not None
                else torch.zeros(market_feat.shape[0], FACTOR_CLASSES, device=self.device)
            )

            loss_dict = self.loss_fn(
                dir_logits=outputs["dir_logits"],
                dir_labels=dir_labels,
                ret_pred=outputs["ret_pred"],
                ret_labels=batch["return"].to(self.device),
                fac_probs=fac_probs,
                fac_labels=batch["factor"].to(self.device),
                attn_weights=attn_w,
                confidence=outputs["confidence"],
                calibration_targets=dir_correct,
                masked_dir_logits=masked_dir_logits,
                article_mask=article_mask,
                soft_gates=outputs.get("soft_gates"),  # Session 20 Fix 7
                vol_pred=outputs.get("vol_pred"),       # optional volatility extension
                vol_labels=vol_labels,                   # optional volatility extension
            )

            val_losses["loss"]    += loss_dict["loss"].item()
            val_losses["count"]   += 1
            val_losses["dir_acc"] += (dir_preds == dir_labels).float().mean().item()

            all_dir_preds.append(dir_preds.cpu().numpy())
            all_dir_labels.append(dir_labels.cpu().numpy())
            all_confidence.append(outputs["confidence"].cpu().numpy())
            all_ret_labels.append(batch["return"].cpu().numpy())
            all_ret_preds.append(outputs["ret_pred"].cpu().numpy())
            all_dir_probs.append(F.softmax(outputs["dir_logits"], dim=-1).cpu().numpy())
            all_dir_logits.append(outputs["dir_logits"].cpu().numpy())

        n = max(val_losses["count"], 1)
        val_losses["loss"]    /= n
        val_losses["dir_acc"] /= n
        del val_losses["count"]

        all_dir_preds  = np.concatenate(all_dir_preds)
        all_dir_labels = np.concatenate(all_dir_labels)
        all_confidence = np.concatenate(all_confidence)
        all_ret_labels = np.concatenate(all_ret_labels)
        all_ret_preds  = np.concatenate(all_ret_preds)
        all_dir_logits = np.concatenate(all_dir_logits)

        # Temperature scaling — fit on this split's logits (same protocol as main trainer)
        if temperature is None:
            temp_fit = fit_temperature_scaling(all_dir_logits, all_dir_labels)
            temperature = float(temp_fit["temperature"])
        else:
            temperature = float(temperature)
        all_dir_probs = apply_temperature_to_logits(all_dir_logits, temperature)

        from train_safe_alert import _TAU_PERCENTILE, _GAMMA_PERCENTILE
        _max_prob = all_dir_probs.max(axis=-1)
        best_tau   = float(tau) if tau is not None else float(np.percentile(all_confidence, _TAU_PERCENTILE))
        best_gamma = float(gamma) if gamma is not None else float(np.percentile(_max_prob, _GAMMA_PERCENTILE))

        macro_f1 = compute_macro_f1(all_dir_probs.argmax(axis=1), all_dir_labels)
        mcc      = compute_mcc(all_dir_probs.argmax(axis=1), all_dir_labels)
        ece      = compute_ece(_max_prob, all_dir_probs.argmax(axis=1), all_dir_labels)
        auc      = compute_auc(all_dir_probs, all_dir_labels)

        cal_preds = all_dir_probs.argmax(axis=1)
        alert_prec, alert_cov, _ = compute_alert_precision(
            all_confidence, cal_preds, all_dir_labels,
            dir_probs=all_dir_probs, tau=best_tau, gamma=best_gamma,
        )
        backtest = mini_backtest(
            all_confidence, cal_preds, all_ret_labels,
            tau=best_tau, gamma=best_gamma, dir_probs=all_dir_probs,
        )
        alert_sharpe = backtest["alert_sharpe"]
        alert_sortino = backtest.get("alert_sortino", 0.0)
        alert_calmar = backtest.get("alert_calmar", 0.0)
        alert_max_dd = backtest.get("alert_max_dd", 0.0)
        score = compute_model_selection_score(
            macro_f1, mcc, ece, alert_sharpe,
            alert_coverage=backtest.get("alert_coverage", 0.0),
            position_pnl=backtest.get("position_pnl", backtest.get("pnl", 0.0)),
        )
        all_dir_preds = cal_preds

        ret_corr = (
            float(np.corrcoef(all_ret_preds, all_ret_labels)[0, 1])
            if len(all_ret_preds) > 1 else 0.0
        )

        val_losses.update({
            "macro_f1":       macro_f1,
            "mcc":            mcc,
            "ece":            ece,
            "auc":            auc,
            "alert_precision": alert_prec,
            "alert_coverage": alert_cov,
            "alert_sharpe":   alert_sharpe,
            "alert_sortino":  alert_sortino,
            "alert_calmar":   alert_calmar,
            "alert_max_dd":   alert_max_dd,
            "model_score":    score,
            "ret_corr":       ret_corr,
            "policy_source":  "validation_percentile",
            "tau":            best_tau,
            "gamma":          best_gamma,
            "temperature":    temperature,
            "tau_seed":       best_tau,
            "gamma_seed":     best_gamma,
            "attn_sum_mean":  0.0,
            "attn_target_mean": 0.0,
        })
        return val_losses


# ── Data loading (mirrors train_safe_alert.py main()) ─────────────────────────

def load_data(args):
    """Load candles, embeddings, articles, and precomputed features."""
    script_file   = Path(__file__).resolve()
    service_root  = script_file.parent.parent.parent.parent

    default_data = service_root / "training_data"
    default_data_v2 = default_data / "v2"
    data_path_base = args.data_path if args.data_path else default_data
    emb_path_base  = args.embeddings_path if args.embeddings_path else default_data
    if args.data_path is None and default_data_v2.exists():
        data_path_base = default_data_v2
    if args.embeddings_path is None and default_data_v2.exists():
        emb_path_base = default_data_v2

    # Candles
    symbol_upper = args.symbol.upper()
    horizon_label = args.horizon
    for candidate in [
        data_path_base / f"{symbol_upper}_{horizon_label}_ohlcv.csv",
        data_path_base / "candles_max.csv",
        data_path_base / "candles_aligned.csv",
        data_path_base / f"{args.symbol.lower()}_training_dataset_v2.csv",
    ]:
        if candidate.exists():
            candle_df = pd.read_csv(candidate)
            print(f"[OK] Candles: {candidate.name} ({len(candle_df)} rows)")
            break
    else:
        logger.error("No candle CSV found in %s", data_path_base)
        sys.exit(1)

    # Embeddings
    embeddings = None
    for candidate in [
        emb_path_base / "btcusdt_article_embeddings_max.npy",
        emb_path_base / f"{args.symbol.lower()}_article_embeddings.npy",
        emb_path_base / "article_embeddings.npy",
    ]:
        if candidate.exists():
            embeddings = np.load(candidate)
            print(f"[OK] Embeddings: {candidate.name} {embeddings.shape}")
            break
    if embeddings is None:
        logger.error("No embeddings .npy found in %s", emb_path_base)
        sys.exit(1)

    # Articles
    articles_df = None
    for candidate in [
        emb_path_base / "articles_max.csv",
        emb_path_base / "articles.csv",
    ]:
        if candidate.exists():
            articles_df = pd.read_csv(candidate)
            print(f"[OK] Articles: {candidate.name} ({len(articles_df)} rows)")
            break
    if articles_df is None:
        logger.error("No articles CSV found in %s", emb_path_base)
        sys.exit(1)

    # Precomputed features (optional)
    precomp = None
    precomp_path = emb_path_base / "features_precomputed.npy"
    if precomp_path.exists():
        precomp = np.load(precomp_path)
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
        print(f"[OK] Precomputed features: {precomp_path.name} {precomp.shape}")

    # Precomputed factor labels (optional)
    factor_labels = None
    factor_path = emb_path_base / "article_factor_labels.npy"
    if factor_path.exists():
        factor_labels = np.load(factor_path)
        print(f"[OK] Factor labels: {factor_path.name} {factor_labels.shape}")

    # Precomputed entity sentiment (optional, but required for META_DIM=14)
    entity_sentiment = None
    entity_path = emb_path_base / "article_entity_sentiment.npy"
    if entity_path.exists():
        entity_sentiment = np.load(entity_path)
        print(f"[OK] Entity sentiment: {entity_path.name} {entity_sentiment.shape}")

    # Session 23 P0 #3 parity: load horizon-compatible bar sequences for
    # paper Eq.16 whenever the full/most ablation variants use sequence mode.
    market_bars = None
    if getattr(args, "use_bar_sequences", True):
        stale_market_bars = []
        for candidate in [
            emb_path_base / f"market_bars_{args.horizon}.npz",
            emb_path_base / "market_bars.npz",
        ]:
            if not candidate.exists():
                continue
            with np.load(candidate) as loaded:
                candidate_dict = {k: loaded[k] for k in loaded.files}
            required_bar_keys = {f"bars_{tf}" for tf in ("1m", "5m", "15m", "1h", "4h")}
            missing_bar_keys = required_bar_keys - set(candidate_dict)
            if missing_bar_keys:
                print(f"[WARN] Ignoring malformed market_bars cache {candidate.name}: "
                      f"missing {sorted(missing_bar_keys)}")
                continue
            shape = candidate_dict["bars_1m"].shape
            if shape[0] != len(candle_df):
                stale_market_bars.append((candidate, shape[0]))
                print(f"[WARN] Ignoring stale market_bars cache {candidate.name}: "
                      f"N={shape[0]} != candles={len(candle_df)} for horizon={args.horizon}")
                continue
            if args.horizon != "1h":
                meta_interval = candidate_dict.get("decision_interval_ns")
                if meta_interval is None:
                    raise ValueError(
                        f"{candidate.name} was generated by an older precompute_market_bars.py "
                        f"without decision_interval metadata. Non-1h bar caches previously used "
                        f"a hard-coded +1h decision close and may be temporally misaligned. "
                        f"Regenerate it with --decision-horizon {args.horizon}."
                    )
                ts_col = "timestamp" if "timestamp" in candle_df.columns else "datetime"
                ts = pd.to_datetime(candle_df[ts_col], utc=True, errors="coerce").dropna().sort_values()
                if len(ts) >= 2:
                    expected_ns = int(ts.diff().dropna().median().value)
                    actual_ns = int(np.asarray(meta_interval).reshape(-1)[0])
                    if abs(actual_ns - expected_ns) > max(1, int(0.1 * expected_ns)):
                        raise ValueError(
                            f"{candidate.name} decision_interval_ns={actual_ns} does not match "
                            f"horizon={args.horizon} candle interval {expected_ns}. Regenerate "
                            f"market_bars with --decision-horizon {args.horizon}."
                        )
            market_bars = candidate_dict
            print(f"[OK] Market bars: {candidate.name} — "
                  f"5 TFs x {shape[0]} candles x {shape[1]} bars x {shape[2]} features")
            break
        if market_bars is None:
            if stale_market_bars:
                stale_desc = ", ".join(f"{p.name}:N={n}" for p, n in stale_market_bars)
                raise ValueError(
                    f"No market_bars cache matches horizon={args.horizon} "
                    f"(expected N={len(candle_df)}; stale: {stale_desc}). "
                    f"Run precompute_market_bars.py with --decision-horizon {args.horizon}."
                )
            print(f"[WARN] use_bar_sequences=True but no market_bars cache found in {emb_path_base}; "
                  "falling back to scalar market features.")
            args.use_bar_sequences = False

    return candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment, market_bars


def build_loaders(candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment,
                  market_bars, args, horizon):
    """Build SAFEAlertDataset and temporal train/val/test DataLoaders.

    Returns (train_loader, val_loader, test_loader, dataset) — the dataset
    handle is needed by the caller to pass into trainer.fit(dataset=...) so
    checkpoints carry the preprocessing state (P0 #1 fix applied to ablation).
    """
    from torch.utils.data import Subset, DataLoader

    import train_safe_alert as _ts_ref
    _eps_override = getattr(_ts_ref, '_EPSILON_H_OVERRIDE', None)
    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=horizon,
        precomputed_features=precomp,
        factor_labels=factor_labels,
        entity_sentiment=entity_sentiment,
        market_bars=market_bars if getattr(args, "use_bar_sequences", True) else None,
        epsilon_h_override=_eps_override,  # match canonical training config
    )

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_start = train_size + val_size

    # P0 #2 parity: ablation MUST z-score on the train split only so every
    # variant (full model + ablation variants) sees identical preprocessing.
    # Without this, variants trained on raw features while the full model
    # (when run via train_safe_alert.py) trained on z-scored → unfair
    # ablation comparison that exaggerated the contribution of whichever
    # module was present in the normalized condition.
    dataset.fit_market_scaler(range(0, train_size))

    train_set = Subset(dataset, range(0, train_size))
    val_set   = Subset(dataset, range(train_size, train_size + val_size))
    test_set  = Subset(dataset, range(test_start, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[OK] Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)} samples")
    print(f"[OK] Market scaler fit on train split (shared across all ablation variants).")
    return train_loader, val_loader, test_loader, dataset


# ── Run a single variant ───────────────────────────────────────────────────────

def run_variant(variant_name: str, ablation_flag: Optional[str],
                lambda6_force: Optional[float], lambda7_force: Optional[float],
                train_loader, val_loader, test_loader, args, dataset=None) -> Dict:
    """Train one ablation variant and return its final val/test metrics.

    dataset: Pass-through to trainer.fit(dataset=...) so the per-variant
    checkpoint carries the shared preprocessing state (P0 #1 parity).
    """
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name}")
    print(f"{'='*60}")

    # Session 23 P0 #3: the w/o_bar_sequences variant trains a SEPARATE
    # model with use_bar_sequences=False so the market encoder uses the
    # legacy scalar path. All other variants use the paper-faithful
    # sequence mode (default True) so the ablation isolates one module
    # at a time rather than comparing against a partially-crippled baseline.
    _use_bar_sequences = (ablation_flag != "w/o_bar_sequences") and getattr(
        args, "use_bar_sequences", True
    )

    # Session 26: use YAML top_k (imported from train_safe_alert._TOP_K_MAP)
    # + explicit K_24h + n_factors to match train_safe_alert.py consistently.
    model = SAFEAlertNet(
        market_dim=63, has_news=True,
        K_15m=_train._TOP_K_MAP["15m"], K_1h=_train._TOP_K_MAP["1h"],
        K_4h=_train._TOP_K_MAP["4h"],   K_24h=_train._TOP_K_MAP["24h"],
        n_factors=_FACTOR_CLASSES,
        use_bar_sequences=_use_bar_sequences,
        bar_seq_len=getattr(args, "bar_seq_len", 20),
        bar_feat_dim=getattr(args, "bar_feat_dim", 10),
    )
    K_h   = model.K_h_map.get(args.horizon, 8)

    trainer = AblationTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        horizon=args.horizon,
        K_h=K_h,
        grad_clip=args.grad_clip,
        ablation_flag=ablation_flag,
        lambda6_force=lambda6_force,
        lambda7_force=lambda7_force,
    )

    # Suppress checkpoint saving for ablation runs (use temp dir)
    ablation_dir = args.artifact_dir / "ablation_runs" / variant_name.replace("/", "_")
    ablation_dir.mkdir(parents=True, exist_ok=True)

    final_metrics, _ = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        checkpoint_dir=ablation_dir,
        early_stopping_patience=max(5, _ES_PATIENCE // 2),
        dataset=dataset,   # P0 #1: preprocessing state in every ablation checkpoint
    )

    # Tune τ/γ/T on val, FREEZE for test (previously test refit its own threshold,
    # producing optimistic test metrics that bounced up for every variant —
    # masking the true effect of ablating a module). Proper protocol matches
    # train_safe_alert.py walk-forward evaluator.
    final_val = trainer.validate(val_loader)
    final_test = trainer.validate(
        test_loader,
        tau=final_val["tau"],
        gamma=final_val["gamma"],
        temperature=final_val["temperature"],
    )
    final_test.update({f"val_{k}": v for k, v in final_val.items()})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test


# ── Full model (no ablation) ───────────────────────────────────────────────────

def run_full_model(train_loader, val_loader, test_loader, args, dataset=None) -> Dict:
    """Train the full SAFE-Alert model (all modules enabled)."""
    print(f"\n{'='*60}")
    print(f"  FULL MODEL (all modules enabled)")
    print(f"{'='*60}")

    # Session 26: YAML-aware K_h + explicit K_24h + n_factors for parity
    # with train_safe_alert.py's model construction.
    model = SAFEAlertNet(
        market_dim=63, has_news=True,
        K_15m=_train._TOP_K_MAP["15m"], K_1h=_train._TOP_K_MAP["1h"],
        K_4h=_train._TOP_K_MAP["4h"],   K_24h=_train._TOP_K_MAP["24h"],
        n_factors=_FACTOR_CLASSES,
        # Session 23 P0 #3: full model uses paper-faithful bar sequences
        # (Eq.16) when available; ablation 'w/o_bar_sequences' is handled
        # in the other branch of this file.
        use_bar_sequences=getattr(args, "use_bar_sequences", True),
        bar_seq_len=getattr(args, "bar_seq_len", 20),
        bar_feat_dim=getattr(args, "bar_feat_dim", 10),
    )
    K_h   = model.K_h_map.get(args.horizon, 8)

    trainer = SAFEAlertTrainer(
        model=model,
        device=args.device,
        lr=args.lr,
        horizon=args.horizon,
        K_h=K_h,
        grad_clip=args.grad_clip,
    )

    full_dir = args.artifact_dir / "ablation_runs" / "full_model"
    full_dir.mkdir(parents=True, exist_ok=True)

    trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        checkpoint_dir=full_dir,
        early_stopping_patience=max(5, _ES_PATIENCE // 2),
        dataset=dataset,   # P0 #1 parity with ablation variants
    )

    # Frozen val thresholds for test evaluation — same protocol as variants.
    final_val = trainer.validate(val_loader)
    final_test = trainer.validate(
        test_loader,
        tau=final_val["tau"],
        gamma=final_val["gamma"],
        temperature=final_val["temperature"],
    )
    final_test.update({f"val_{k}": v for k, v in final_val.items()})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test


# ── Print summary table ────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Dict]):
    """Print a formatted comparison table of all variants."""
    header_cols = ["Variant", "F1", "MCC", "ECE", "Sharpe", "Score", "RetCorr", "AlertCov"]
    col_w = [26, 7, 7, 7, 8, 8, 9, 9]

    def _fmt(s, w): return str(s).ljust(w)[:w]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    header = "|" + "|".join(_fmt(h, w) for h, w in zip(header_cols, col_w)) + "|"

    print("\n" + sep)
    print(header)
    print(sep)

    for variant, metrics in results.items():
        row = [
            variant,
            f"{metrics.get('macro_f1', 0):.4f}",
            f"{metrics.get('mcc', 0):.4f}",
            f"{metrics.get('ece', 0):.4f}",
            f"{metrics.get('alert_sharpe', 0):.4f}",
            f"{metrics.get('model_score', 0):.4f}",
            f"{metrics.get('ret_corr', 0):.4f}",
            f"{metrics.get('alert_coverage', 0):.4f}",
        ]
        print("|" + "|".join(_fmt(v, w) for v, w in zip(row, col_w)) + "|")

    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="SAFE-Alert ablation study (PDF Section 5.2)")
    parser.add_argument("--symbol",          default="BTCUSDT")
    parser.add_argument("--horizon",         default="1h", choices=["15m", "1h", "4h", "24h"])
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--output",          type=Path,  default=None,
                        help="Path to save ablation_results.json (default: artifact_dir/ablation_results.json)")

    script_file  = Path(__file__).resolve()
    service_root = script_file.parent.parent.parent.parent
    default_data = service_root / "training_data"
    parser.add_argument("--walk_forward",    action="store_true", default=False)
    parser.add_argument("--n_folds",         type=int,   default=4)
    parser.add_argument("--embargo_steps",   type=int,   default=24)
    parser.add_argument("--data_path",       type=Path,  default=default_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_data)
    parser.add_argument("--use_bar_sequences", dest="use_bar_sequences",
                        action="store_true", default=None,
                        help="Use paper Eq.16 bar-sequence market encoder.")
    parser.add_argument("--no-use_bar_sequences", dest="use_bar_sequences",
                        action="store_false",
                        help="Force legacy scalar market encoder for debugging.")

    # Which variants to run (default: all)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variant names to run (default: all 7 + full model)")
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "train_config_research_best.yaml",
                        help="YAML config (same as train_safe_alert.py). Ablation "
                             "honours the same hyperparameters so results are "
                             "directly comparable to the full-model run.")
    args = parser.parse_args()

    # Session 26 — apply YAML overrides to train_safe_alert module-level
    # constants so _TOP_K_MAP / λ schedule / faith_margin / etc. used by
    # SAFEAlertTrainer + AblationTrainer here match the full trainer.
    # Without this, run_ablation.py would silently use module defaults
    # even when user edited YAML.
    try:
        import yaml as _yaml
        if args.config and args.config.exists():
            with open(args.config) as _f:
                _cfg = _yaml.safe_load(_f) or {}
            _train._apply_config_overrides(_cfg)
            if args.use_bar_sequences is None:
                args.use_bar_sequences = bool(_cfg.get("use_bar_sequences", True))
            print(f"[CONFIG] Loaded {args.config}")
        else:
            _train._apply_config_overrides({})   # uses defaults
            if args.use_bar_sequences is None:
                args.use_bar_sequences = True
    except Exception as _e:
        print(f"[CONFIG WARN] Could not apply config overrides: {_e}")
        if args.use_bar_sequences is None:
            args.use_bar_sequences = True

    output_path = args.output or (args.artifact_dir / "ablation_results.json")

    print(f"[START] SAFE-Alert Ablation Study")
    print(f"  Symbol : {args.symbol}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Device : {args.device}")

    # Load shared data (loaded once, reused across all variants).
    # Dataset handle is returned so every trainer.fit() can snapshot its
    # scaler/return-scale state into the variant checkpoint (P0 #1 parity).
    candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment, market_bars = load_data(args)
    requested = set(args.variants) if args.variants else None

    def _run_all_variants(train_loader, val_loader, test_loader, dataset):
        res = {}
        if requested is None or "full_model" in requested:
            res["full_model"] = run_full_model(train_loader, val_loader, test_loader, args, dataset=dataset)
        for (vname, desc, aflag, l6, l7) in ABLATION_VARIANTS:
            if requested is not None and vname not in requested:
                continue
            print(f"\n[INFO] {desc}")
            res[vname] = run_variant(vname, aflag, l6, l7, train_loader, val_loader, test_loader, args, dataset=dataset)
        return res

    if args.walk_forward:
        from train_safe_alert import walk_forward_split
        from torch.utils.data import Subset, DataLoader as _DL
        _, _, _, dataset = build_loaders(candle_df, embeddings, articles_df, precomp,
                                         factor_labels, entity_sentiment, market_bars, args, args.horizon)
        folds = list(walk_forward_split(len(dataset), n_folds=args.n_folds, embargo_steps=args.embargo_steps))
        print(f"[WALK-FORWARD] {args.n_folds} folds | embargo={args.embargo_steps}")

        fold_results_all: Dict[str, list] = {}
        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
            print(f"\n{'='*60}\n  FOLD {fold_idx}/{len(folds)}\n{'='*60}")
            dataset.fit_market_scaler(train_idx)
            tr = _DL(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True,  num_workers=0)
            vl = _DL(Subset(dataset, val_idx),   batch_size=args.batch_size, shuffle=False, num_workers=0)
            te = _DL(Subset(dataset, test_idx),  batch_size=args.batch_size, shuffle=False, num_workers=0)
            fold_res = _run_all_variants(tr, vl, te, dataset)
            for name, metrics in fold_res.items():
                fold_results_all.setdefault(name, []).append(metrics)

        results: Dict[str, Dict] = {}
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
        train_loader, val_loader, test_loader, dataset = build_loaders(
            candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment,
            market_bars, args, args.horizon
        )
        results = _run_all_variants(train_loader, val_loader, test_loader, dataset)

    # --- Print summary ---
    print_summary_table(results)

    # --- Save JSON ---
    # Convert numpy floats to Python floats for JSON serialisation
    def _to_serialisable(d):
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in d.items()}

    json_results = {k: _to_serialisable(v) for k, v in results.items()}
    json_results["_meta"] = {
        "symbol":   args.symbol,
        "horizon":  args.horizon,
        "epochs":   args.epochs,
        "device":   args.device,
        "protocol": f"walk_forward_{args.n_folds}fold" if args.walk_forward else "single_split_70_15_15",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Ablation results saved: {output_path}")


if __name__ == "__main__":
    main()
