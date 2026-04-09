"""
Baseline comparison for SAFE-Alert (PDF Section 5.1 / related work).

Four baselines:
  1. market_only       -- 2-layer MLP on 63-dim market features, no news
  2. all_news_fusion   -- average ALL article embeddings + market MLP (no selection)
  3. always_alert      -- trivial: always fires alert, no training
  4. raw_prob_threshold -- market-only model + raw softmax threshold (no conf head)

All baselines use the same train/val split as SAFE-Alert.
Metrics are computed with the same functions from metrics_safe_alert.py.

Usage:
  python run_baselines.py --symbol BTCUSDT --horizon 1h --epochs 30
  python run_baselines.py --baselines market_only always_alert
"""

import sys
import json
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))

from safe_alert_dataset import SAFEAlertDataset
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_brier_score,
    compute_alert_precision, mini_backtest, compute_model_selection_score,
    compute_auc,
)
from utils import ARTIFACT_DIR

torch.manual_seed(42)
np.random.seed(42)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
_MARKET_DIM   = 63
_NEWS_EMB_DIM = 768
_HIDDEN       = 128
_DROPOUT      = 0.3
_N_CLASSES    = 3
_ACCUM_STEPS  = 2


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE MODEL DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

class MarketOnlyMLP(nn.Module):
    """Baseline 1 & 4: 2-layer MLP on 63-dim market features.

    No news, no attention, no confidence head.
    Used directly as Baseline 1 (market_only) and as the backbone for
    Baseline 4 (raw_prob_threshold) which replaces the conf head with
    max(softmax) thresholding at inference.
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(market_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)  # learned confidence (used by Baseline 1)

    def forward(self, market_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(market_feat)
        dir_logits = self.dir_head(h)             # (B, 3)
        confidence = torch.sigmoid(self.conf_head(h)).squeeze(-1)  # (B,)
        return {"dir_logits": dir_logits, "confidence": confidence}


class AllNewsFusionMLP(nn.Module):
    """Baseline 2: Average ALL article embeddings (no Top-K, no attention).

    Concatenate averaged news embedding with market features -> MLP -> direction.
    Tests whether selective attention adds value over naive all-news averaging.
    """
    def __init__(self, market_dim: int = _MARKET_DIM,
                 news_emb_dim: int = _NEWS_EMB_DIM,
                 hidden: int = _HIDDEN, n_classes: int = _N_CLASSES,
                 dropout: float = _DROPOUT):
        super().__init__()
        # Project news embedding to a smaller dimension first (avoid huge concat)
        self.news_proj = nn.Sequential(
            nn.Linear(news_emb_dim, 128),
            nn.GELU(),
        )
        fused_dim = market_dim + 128
        self.net = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dir_head  = nn.Linear(hidden, n_classes)
        self.conf_head = nn.Linear(hidden, 1)

    def forward(self, market_feat: torch.Tensor,
                article_emb: torch.Tensor,
                article_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        market_feat:  (B, 63)
        article_emb:  (B, K, 768)
        article_mask: (B, K) float — 1 for real, 0 for padding
        """
        # Average ALL valid article embeddings (uniform, no selection)
        mask_f  = article_mask.float().unsqueeze(-1)          # (B, K, 1)
        denom   = mask_f.sum(dim=1).clamp(min=1e-8)           # (B, 1)
        avg_emb = (article_emb * mask_f).sum(dim=1) / denom   # (B, 768)

        # Project news and concatenate with market
        news_h = self.news_proj(avg_emb)                       # (B, 128)
        fused  = torch.cat([market_feat, news_h], dim=-1)      # (B, 63+128)

        h = self.net(fused)
        dir_logits = self.dir_head(h)
        confidence = torch.sigmoid(self.conf_head(h)).squeeze(-1)
        return {"dir_logits": dir_logits, "confidence": confidence}


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _train_epoch_market_only(model: MarketOnlyMLP, loader, optimizer,
                              device: str, accum: int = _ACCUM_STEPS) -> float:
    """One training epoch for the market-only baseline."""
    model.train()
    total_loss, count = 0.0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        market_feat = batch["market_features"].to(device)
        dir_labels  = batch["direction"].to(device)

        out   = model(market_feat)
        loss  = F.cross_entropy(out["dir_logits"], dir_labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        (loss / accum).backward()

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        count      += 1

    return total_loss / max(count, 1)


def _train_epoch_all_news(model: AllNewsFusionMLP, loader, optimizer,
                           device: str, accum: int = _ACCUM_STEPS) -> float:
    """One training epoch for the all-news-fusion baseline."""
    model.train()
    total_loss, count = 0.0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        market_feat  = batch["market_features"].to(device)
        article_emb  = batch["article_embeddings"].to(device)
        article_mask = batch["article_mask"].to(device)
        dir_labels   = batch["direction"].to(device)

        out  = model(market_feat, article_emb, article_mask)
        loss = F.cross_entropy(out["dir_logits"], dir_labels)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        (loss / accum).backward()

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        count      += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate_market_only(model: MarketOnlyMLP, loader,
                           device: str) -> Tuple[np.ndarray, ...]:
    """Return (preds, labels, confidence, ret_labels, dir_probs) arrays."""
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []

    for batch in loader:
        market_feat = batch["market_features"].to(device)
        dir_labels  = batch["direction"]
        ret_labels  = batch["return"]

        out       = model(market_feat)
        probs     = F.softmax(out["dir_logits"], dim=-1)
        dir_preds = probs.argmax(dim=-1)

        preds_l.append(dir_preds.cpu().numpy())
        labels_l.append(dir_labels.numpy())
        conf_l.append(out["confidence"].cpu().numpy())
        ret_l.append(ret_labels.numpy())
        probs_l.append(probs.cpu().numpy())

    return (
        np.concatenate(preds_l),
        np.concatenate(labels_l),
        np.concatenate(conf_l),
        np.concatenate(ret_l),
        np.concatenate(probs_l),
    )


@torch.no_grad()
def _evaluate_all_news(model: AllNewsFusionMLP, loader,
                        device: str) -> Tuple[np.ndarray, ...]:
    """Return (preds, labels, confidence, ret_labels, dir_probs) arrays."""
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []

    for batch in loader:
        market_feat  = batch["market_features"].to(device)
        article_emb  = batch["article_embeddings"].to(device)
        article_mask = batch["article_mask"].to(device)
        dir_labels   = batch["direction"]
        ret_labels   = batch["return"]

        out       = model(market_feat, article_emb, article_mask)
        probs     = F.softmax(out["dir_logits"], dim=-1)
        dir_preds = probs.argmax(dim=-1)

        preds_l.append(dir_preds.cpu().numpy())
        labels_l.append(dir_labels.numpy())
        conf_l.append(out["confidence"].cpu().numpy())
        ret_l.append(ret_labels.numpy())
        probs_l.append(probs.cpu().numpy())

    return (
        np.concatenate(preds_l),
        np.concatenate(labels_l),
        np.concatenate(conf_l),
        np.concatenate(ret_l),
        np.concatenate(probs_l),
    )


def _compute_metrics(preds, labels, confidence, ret_labels,
                     dir_probs=None) -> Dict:
    """Compute the full metric set using the same functions as SAFEAlertTrainer."""
    _max_prob = dir_probs.max(axis=-1) if dir_probs is not None else confidence
    best_tau   = float(np.percentile(confidence, 70))
    best_gamma = float(np.percentile(_max_prob, 60))

    macro_f1 = compute_macro_f1(preds, labels)
    mcc      = compute_mcc(preds, labels)
    ece      = compute_ece(confidence, preds, labels)
    brier    = compute_brier_score(confidence, preds, labels)

    alert_prec, alert_cov, _ = compute_alert_precision(
        confidence, preds, labels,
        dir_probs=dir_probs, tau=best_tau, gamma=best_gamma,
    )
    backtest = mini_backtest(
        confidence, preds, ret_labels,
        tau=best_tau, gamma=best_gamma, dir_probs=dir_probs,
    )
    alert_sharpe = backtest["alert_sharpe"]
    score = compute_model_selection_score(macro_f1, mcc, ece, alert_sharpe)
    auc   = compute_auc(dir_probs, labels) if dir_probs is not None else 0.5

    return {
        "macro_f1":       macro_f1,
        "mcc":            mcc,
        "ece":            ece,
        "brier":          brier,
        "alert_precision": alert_prec,
        "alert_coverage": alert_cov,
        "alert_sharpe":   alert_sharpe,
        "model_score":    score,
        "auc":            auc,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE RUNNERS
# ═════════════════════════════════════════════════════════════════════════════

def run_market_only(train_loader, val_loader, args) -> Dict:
    """Baseline 1: 2-layer MLP on market features only, with learned confidence head."""
    print(f"\n{'='*60}")
    print(f"  BASELINE 1: Market-Only MLP")
    print(f"{'='*60}")

    model = MarketOnlyMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    preds, labels, confidence, ret_labels, dir_probs = _evaluate_market_only(
        model, val_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f} MCC={metrics['mcc']:.4f} "
          f"ECE={metrics['ece']:.4f} Sharpe={metrics['alert_sharpe']:.4f} "
          f"Score={metrics['model_score']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def run_all_news_fusion(train_loader, val_loader, args) -> Dict:
    """Baseline 2: Average ALL articles (no selection) + market features."""
    print(f"\n{'='*60}")
    print(f"  BASELINE 2: All-News Fusion (uniform average, no Top-K)")
    print(f"{'='*60}")

    model = AllNewsFusionMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_all_news(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    preds, labels, confidence, ret_labels, dir_probs = _evaluate_all_news(
        model, val_loader, args.device
    )
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f} MCC={metrics['mcc']:.4f} "
          f"ECE={metrics['ece']:.4f} Sharpe={metrics['alert_sharpe']:.4f} "
          f"Score={metrics['model_score']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


@torch.no_grad()
def run_always_alert(val_loader, args) -> Dict:
    """Baseline 3: Always fire alert (conf=1.0, direction=majority class).

    No training. Tests the lower bound: how well does a naive always-alert
    strategy perform vs the learned model?

    Direction prediction: always predicts the majority class (class 1 = NEUTRAL
    in most crypto datasets).
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 3: Always-Alert (trivial, no training)")
    print(f"{'='*60}")

    all_labels, all_ret_labels = [], []

    for batch in val_loader:
        all_labels.append(batch["direction"].numpy())
        all_ret_labels.append(batch["return"].numpy())

    labels     = np.concatenate(all_labels)
    ret_labels = np.concatenate(all_ret_labels)
    N = len(labels)

    # Majority class prediction
    majority_class = int(np.bincount(labels.astype(np.int64)).argmax())
    preds      = np.full(N, majority_class, dtype=np.int64)
    confidence = np.ones(N, dtype=np.float32)  # always alert → confidence = 1.0

    # Uniform dir_probs (model has no probability estimate)
    dir_probs = np.zeros((N, 3), dtype=np.float32)
    dir_probs[:, majority_class] = 1.0

    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  Majority class: {majority_class} | N={N}")
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f} MCC={metrics['mcc']:.4f} "
          f"ECE={metrics['ece']:.4f} Sharpe={metrics['alert_sharpe']:.4f} "
          f"Score={metrics['model_score']:.4f}")

    return metrics


def run_raw_prob_threshold(train_loader, val_loader, args) -> Dict:
    """Baseline 4: Market-Only model with raw max(softmax) threshold as confidence.

    Trains the same MarketOnlyMLP but at inference uses max(softmax) directly
    as the alert confidence (no learned confidence head). Tests whether the
    dedicated confidence head adds value over raw probability thresholding.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE 4: Raw-Probability Threshold (no confidence head)")
    print(f"{'='*60}")

    model = MarketOnlyMLP().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch_market_only(model, train_loader, optimizer, args.device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | train_loss={loss:.4f}")

    # Evaluate: use max(softmax) as confidence proxy (not the learned conf head)
    model.eval()
    preds_l, labels_l, conf_l, ret_l, probs_l = [], [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            market_feat = batch["market_features"].to(args.device)
            dir_labels  = batch["direction"]
            ret_labels  = batch["return"]

            out   = model(market_feat)
            probs = F.softmax(out["dir_logits"], dim=-1)
            dir_preds = probs.argmax(dim=-1)

            # Confidence = max softmax probability (raw, no confidence head)
            raw_confidence = probs.max(dim=-1).values

            preds_l.append(dir_preds.cpu().numpy())
            labels_l.append(dir_labels.numpy())
            conf_l.append(raw_confidence.cpu().numpy())
            ret_l.append(ret_labels.numpy())
            probs_l.append(probs.cpu().numpy())

    preds      = np.concatenate(preds_l)
    labels     = np.concatenate(labels_l)
    confidence = np.concatenate(conf_l)
    ret_labels = np.concatenate(ret_l)
    dir_probs  = np.concatenate(probs_l)

    # For raw-prob threshold: confidence == max_prob, so gamma threshold is same as tau.
    # We still use percentile thresholds for fair comparison.
    metrics = _compute_metrics(preds, labels, confidence, ret_labels, dir_probs)
    print(f"  [RESULT] F1={metrics['macro_f1']:.4f} MCC={metrics['mcc']:.4f} "
          f"ECE={metrics['ece']:.4f} Sharpe={metrics['alert_sharpe']:.4f} "
          f"Score={metrics['model_score']:.4f}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING (mirrors train_safe_alert.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_data(args):
    script_file   = Path(__file__).resolve()
    service_root  = script_file.parent.parent.parent.parent

    data_root = args.data_path or (service_root / "training_data")
    emb_root  = args.embeddings_path or (service_root / "training_data")

    # Candles
    for candidate in [
        data_root / "candles_max.csv",
        data_root / "candles_aligned.csv",
        data_root / f"{args.symbol.lower()}_training_dataset_v2.csv",
    ]:
        if candidate.exists():
            candle_df = pd.read_csv(candidate)
            print(f"[OK] Candles: {candidate.name} ({len(candle_df)} rows)")
            break
    else:
        logger.error("No candle CSV found in %s", data_root)
        sys.exit(1)

    # Embeddings
    embeddings = None
    for candidate in [
        emb_root / "btcusdt_article_embeddings_max.npy",
        emb_root / f"{args.symbol.lower()}_article_embeddings.npy",
        emb_root / "article_embeddings.npy",
    ]:
        if candidate.exists():
            embeddings = np.load(candidate)
            print(f"[OK] Embeddings: {candidate.name} {embeddings.shape}")
            break
    if embeddings is None:
        logger.error("No embeddings .npy found in %s", emb_root)
        sys.exit(1)

    # Articles
    articles_df = None
    for candidate in [
        emb_root / "articles_max.csv",
        emb_root / "articles.csv",
    ]:
        if candidate.exists():
            articles_df = pd.read_csv(candidate)
            print(f"[OK] Articles: {candidate.name} ({len(articles_df)} rows)")
            break
    if articles_df is None:
        logger.error("No articles CSV found in %s", emb_root)
        sys.exit(1)

    # Precomputed features (optional)
    precomp = None
    pp = emb_root / "features_precomputed.npy"
    if pp.exists():
        precomp = np.load(pp)
        print(f"[OK] Precomputed features: {pp.name} {precomp.shape}")

    # Factor labels (optional)
    factor_labels = None
    fp = emb_root / "article_factor_labels.npy"
    if fp.exists():
        factor_labels = np.load(fp)
        print(f"[OK] Factor labels: {fp.name} {factor_labels.shape}")

    return candle_df, embeddings, articles_df, precomp, factor_labels


def build_loaders(candle_df, embeddings, articles_df, precomp, factor_labels, args):
    from torch.utils.data import Subset, DataLoader

    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=args.horizon,
        precomputed_features=precomp,
        factor_labels=factor_labels,
    )

    train_size = int(0.7 * len(dataset))
    val_size   = int(0.15 * len(dataset))

    train_set = Subset(dataset, range(0, train_size))
    val_set   = Subset(dataset, range(train_size, train_size + val_size))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)

    print(f"[OK] Train={len(train_set)}, Val={len(val_set)} samples")
    return train_loader, val_loader


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_summary_table(results: Dict[str, Dict]):
    header_cols = ["Baseline", "F1", "MCC", "ECE", "Sharpe", "Score", "AlertCov", "AUC"]
    col_w       = [28, 7, 7, 7, 8, 8, 9, 7]

    def _fmt(s, w): return str(s).ljust(w)[:w]

    sep    = "+" + "+".join("-" * w for w in col_w) + "+"
    header = "|" + "|".join(_fmt(h, w) for h, w in zip(header_cols, col_w)) + "|"

    print("\n" + sep)
    print(header)
    print(sep)

    for name, metrics in results.items():
        row = [
            name,
            f"{metrics.get('macro_f1', 0):.4f}",
            f"{metrics.get('mcc', 0):.4f}",
            f"{metrics.get('ece', 0):.4f}",
            f"{metrics.get('alert_sharpe', 0):.4f}",
            f"{metrics.get('model_score', 0):.4f}",
            f"{metrics.get('alert_coverage', 0):.4f}",
            f"{metrics.get('auc', 0):.4f}",
        ]
        print("|" + "|".join(_fmt(v, w) for v, w in zip(row, col_w)) + "|")

    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

_ALL_BASELINES = ["market_only", "all_news_fusion", "always_alert", "raw_prob_threshold"]


def main():
    parser = ArgumentParser(description="SAFE-Alert baseline comparison")
    parser.add_argument("--symbol",          default="BTCUSDT")
    parser.add_argument("--horizon",         default="1h", choices=["1h", "4h"])
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--artifact_dir",    type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--output",          type=Path,  default=None,
                        help="Path for baseline_results.json (default: artifact_dir/baseline_results.json)")
    parser.add_argument("--baselines",       nargs="*",  default=None,
                        choices=_ALL_BASELINES,
                        help=f"Which baselines to run (default: all). Choices: {_ALL_BASELINES}")

    script_file  = Path(__file__).resolve()
    service_root = script_file.parent.parent.parent.parent
    default_data = service_root / "training_data"
    parser.add_argument("--data_path",       type=Path,  default=default_data)
    parser.add_argument("--embeddings_path", type=Path,  default=default_data)

    args = parser.parse_args()
    output_path = args.output or (args.artifact_dir / "baseline_results.json")

    to_run = args.baselines if args.baselines else _ALL_BASELINES

    print(f"[START] SAFE-Alert Baseline Study")
    print(f"  Symbol    : {args.symbol}")
    print(f"  Horizon   : {args.horizon}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Device    : {args.device}")
    print(f"  Baselines : {to_run}")

    candle_df, embeddings, articles_df, precomp, factor_labels = load_data(args)
    train_loader, val_loader = build_loaders(
        candle_df, embeddings, articles_df, precomp, factor_labels, args
    )

    results: Dict[str, Dict] = {}

    if "market_only" in to_run:
        results["market_only"] = run_market_only(train_loader, val_loader, args)

    if "all_news_fusion" in to_run:
        results["all_news_fusion"] = run_all_news_fusion(train_loader, val_loader, args)

    if "always_alert" in to_run:
        results["always_alert"] = run_always_alert(val_loader, args)

    if "raw_prob_threshold" in to_run:
        results["raw_prob_threshold"] = run_raw_prob_threshold(
            train_loader, val_loader, args
        )

    # Summary
    print_summary_table(results)

    # Save JSON
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
    print(f"\n[OK] Baseline results saved: {output_path}")


if __name__ == "__main__":
    main()
