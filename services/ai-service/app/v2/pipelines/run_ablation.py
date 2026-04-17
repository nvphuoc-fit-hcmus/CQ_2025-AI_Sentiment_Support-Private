"""
Ablation study for SAFE-Alert (PDF Section 5.2).

7 ablation variants are tested (Table 5 in paper):
  1. w/o_selective_news  -- uniform average over ALL articles, no Top-K gating
  2. w/o_factor          -- z_fac = zeros (factor module disabled)
  3. w/o_market          -- z_mkt = zeros (market encoder disabled)
  4. w/o_confidence      -- fixed confidence = 0.5 (confidence head disabled)
  5. w/o_faithfulness    -- lambda6 = 0 throughout training (Lfaith disabled)
  6. w/o_horizon         -- h_emb = zeros (horizon conditioning disabled)
  7. w/o_lrisk           -- lambda7 = 0 throughout training (Lrisk disabled)

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
        losses = {
            "loss": 0, "Ldir": 0, "Lret": 0, "Lfac": 0,
            "Lsel": 0, "Lcal": 0, "Lfaith": 0, "Lrisk": 0
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

            # Forward with ablation flag
            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                ablation=self.ablation_flag,
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
    def validate(self, val_loader) -> Dict[str, float]:
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

            outputs = self.model(
                market_feat, horizon=self.horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                ablation=self.ablation_flag,
            )

            has_articles = article_mask.any(dim=1).any()
            if has_articles and self.loss_fn.lambda6 > 0:
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
        temp_fit  = fit_temperature_scaling(all_dir_logits, all_dir_labels)
        temperature = float(temp_fit["temperature"])
        all_dir_probs = apply_temperature_to_logits(all_dir_logits, temperature)

        from train_safe_alert import _TAU_PERCENTILE, _GAMMA_PERCENTILE
        _max_prob = all_dir_probs.max(axis=-1)
        best_tau   = float(np.percentile(all_confidence, _TAU_PERCENTILE))
        best_gamma = float(np.percentile(_max_prob, _GAMMA_PERCENTILE))

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
        score = compute_model_selection_score(macro_f1, mcc, ece, alert_sharpe)
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

    return candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment


def build_loaders(candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment,
                  args, horizon):
    """Build SAFEAlertDataset and temporal train/val/test DataLoaders."""
    from torch.utils.data import Subset, DataLoader

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
    )

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))
    test_start = train_size + val_size

    train_set = Subset(dataset, range(0, train_size))
    val_set   = Subset(dataset, range(train_size, train_size + val_size))
    test_set  = Subset(dataset, range(test_start, len(dataset)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    print(f"[OK] Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)} samples")
    return train_loader, val_loader, test_loader


# ── Run a single variant ───────────────────────────────────────────────────────

def run_variant(variant_name: str, ablation_flag: Optional[str],
                lambda6_force: Optional[float], lambda7_force: Optional[float],
                train_loader, val_loader, test_loader, args) -> Dict:
    """Train one ablation variant and return its final val/test metrics."""
    print(f"\n{'='*60}")
    print(f"  VARIANT: {variant_name}")
    print(f"{'='*60}")

    model = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
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
    )

    # Collect final validation and test metrics
    final_val = trainer.validate(val_loader)
    final_test = trainer.validate(test_loader)
    final_test.update({f"val_{k}": v for k, v in final_val.items()})

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_test


# ── Full model (no ablation) ───────────────────────────────────────────────────

def run_full_model(train_loader, val_loader, test_loader, args) -> Dict:
    """Train the full SAFE-Alert model (all modules enabled)."""
    print(f"\n{'='*60}")
    print(f"  FULL MODEL (all modules enabled)")
    print(f"{'='*60}")

    model = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
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
    )

    final_val = trainer.validate(val_loader)
    final_test = trainer.validate(test_loader)
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
    parser.add_argument("--data_path",       type=Path,  default=default_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_data)

    # Which variants to run (default: all)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variant names to run (default: all 7 + full model)")
    args = parser.parse_args()

    output_path = args.output or (args.artifact_dir / "ablation_results.json")

    print(f"[START] SAFE-Alert Ablation Study")
    print(f"  Symbol : {args.symbol}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Device : {args.device}")

    # Load shared data (loaded once, reused across all variants)
    candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment = load_data(args)
    train_loader, val_loader, test_loader = build_loaders(
        candle_df, embeddings, articles_df, precomp, factor_labels, entity_sentiment,
        args, args.horizon
    )

    results: Dict[str, Dict] = {}

    # Determine which variants to run
    requested = set(args.variants) if args.variants else None

    # --- Full model first (baseline reference) ---
    if requested is None or "full_model" in requested:
        results["full_model"] = run_full_model(train_loader, val_loader, test_loader, args)

    # --- Ablation variants ---
    for (variant_name, description, ablation_flag, lambda6_force, lambda7_force) in ABLATION_VARIANTS:
        if requested is not None and variant_name not in requested:
            continue
        print(f"\n[INFO] {description}")
        results[variant_name] = run_variant(
            variant_name, ablation_flag, lambda6_force, lambda7_force,
            train_loader, val_loader, test_loader, args
        )

    # --- Print summary ---
    print_summary_table(results)

    # --- Save JSON ---
    # Convert numpy floats to Python floats for JSON serialisation
    def _to_serialisable(d):
        return {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in d.items()}

    json_results = {k: _to_serialisable(v) for k, v in results.items()}
    json_results["_meta"] = {
        "symbol":  args.symbol,
        "horizon": args.horizon,
        "epochs":  args.epochs,
        "device":  args.device,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[OK] Ablation results saved: {output_path}")


if __name__ == "__main__":
    main()
