"""
Evaluate all 4 fold checkpoints on their test splits.
Produces a complete walk_forward_summary.json with test metrics for all folds.

Usage:
    python eval_all_folds.py
"""
import sys, os, json, warnings
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'app' / 'v2' / 'pipelines'))
sys.path.insert(0, str(Path(__file__).parent / 'app' / 'v2'))
sys.path.insert(0, str(Path(__file__).parent / 'app'))

from train_safe_alert import (
    SAFEAlertDataset, walk_forward_split, _apply_config_overrides
)
from models.safe_alert_net import SAFEAlertNet
from metrics_safe_alert import (
    compute_macro_f1, compute_mcc, compute_ece, compute_brier_score,
    compute_alert_precision, mini_backtest, compute_auc,
    compute_deletion_insertion_score, compute_sufficiency, compute_factor_consistency,
)
import yaml
import pandas as pd
from torch.utils.data import DataLoader, Subset

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).parent / 'artifacts'
DATA_DIR     = Path(__file__).parent / 'training_data' / 'v2'
CONFIG_FILE  = Path(__file__).parent / 'app' / 'v2' / 'pipelines' / 'train_config_research_best.yaml'
OUTPUT_FILE  = ARTIFACT_DIR / 'walk_forward_summary_4folds.json'

DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS   = 4
BATCH     = 32

print(f'Device: {DEVICE}')
print(f'Artifact dir: {ARTIFACT_DIR}')

# ── Load config ───────────────────────────────────────────────────────────────
with open(CONFIG_FILE) as f:
    cfg = yaml.safe_load(f)
_apply_config_overrides(cfg)

# ── Load dataset ──────────────────────────────────────────────────────────────
print('Loading dataset...')
candle_df   = pd.read_csv(DATA_DIR / 'BTCUSDT_1h_ohlcv.csv')
# Normalize column name to 'timestamp'
if 'timestamp' not in candle_df.columns:
    for col in ['datetime', 'date', 'time', 'open_time']:
        if col in candle_df.columns:
            candle_df = candle_df.rename(columns={col: 'timestamp'})
            break
candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'], errors='coerce')
articles_df = pd.read_csv(DATA_DIR / 'articles_max.csv')
embeddings  = np.load(DATA_DIR / 'btcusdt_article_embeddings_max.npy')
precomp     = np.load(DATA_DIR / 'features_precomputed.npy')
factor_lbl  = np.load(DATA_DIR / 'article_factor_labels.npy')
entity_sent = np.load(DATA_DIR / 'article_entity_sentiment.npy')
novelty     = np.load(DATA_DIR / 'article_novelty.npy')

try:
    market_bars = np.load(DATA_DIR / 'market_bars.npz')
except Exception:
    market_bars = None
    print('[WARN] market_bars.npz not found, using scalar mode')

full_dataset = SAFEAlertDataset(
    candle_df=candle_df,
    article_embeddings=embeddings,
    article_meta=articles_df,
    article_to_candle={},
    symbol='BTCUSDT',
    horizon='1h',
    precomputed_features=precomp,
    factor_labels=factor_lbl,
    entity_sentiment=entity_sent,
    article_novelty=novelty,
    market_bars=market_bars,
    epsilon_h_override=0.0015,   # Must match training config
)
print(f'Dataset: {len(full_dataset)} samples')

# ── Walk-forward splits ───────────────────────────────────────────────────────
folds = list(walk_forward_split(
    len(full_dataset),
    n_folds=N_FOLDS,
    embargo_steps=24,
))

# ── Evaluate each fold ────────────────────────────────────────────────────────
all_results = []

for fold_idx, (train_idx, val_idx, test_idx) in enumerate(folds, start=1):
    ckpt_path = ARTIFACT_DIR / f'fold_{fold_idx}' / 'safe_alert_1h_FINAL.pt'
    if not ckpt_path.exists():
        print(f'[FOLD {fold_idx}] FINAL.pt not found — skip')
        continue

    meta_path = ARTIFACT_DIR / f'fold_{fold_idx}' / 'training_metrics.json'
    with open(meta_path) as f:
        meta = json.load(f)

    policy = meta['selection_meta']
    tau, gamma, temp = policy['tau'], policy['gamma'], policy['temperature']

    print(f'\n[FOLD {fold_idx}/{N_FOLDS}] test={test_idx[0]}-{test_idx[-1]} '
          f'({len(test_idx)} samples) | checkpoint: epoch {meta["best_overall_epoch"]}')

    # Load model with CORRECT architecture (must match training config)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    n_factors = ckpt.get('n_factors', 10)
    model = SAFEAlertNet(
        market_dim=63,
        has_news=True,
        market_input_mode='hybrid',  # Must match train_config_research_best.yaml
        bar_seq_len=20,
        bar_feat_dim=10,
        n_factors=n_factors,
    )
    state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'  [WARN] Missing keys: {missing[:3]}...')
    model.to(DEVICE).eval()

    # Apply market bar scaler from checkpoint
    ps = ckpt.get('preprocessing_state', {})
    bar_mean = ps.get('market_bar_mean')
    bar_std  = ps.get('market_bar_std')
    bar_clip = ps.get('market_bar_clip', 8.0)
    if bar_mean is not None and bar_std is not None:
        full_dataset.market_bar_mean = bar_mean.to('cpu') if hasattr(bar_mean, 'to') else torch.tensor(bar_mean)
        full_dataset.market_bar_std  = bar_std.to('cpu')  if hasattr(bar_std,  'to') else torch.tensor(bar_std)
        full_dataset.market_bar_clip = float(bar_clip)
        print(f'  Bar scaler applied (shape: {bar_mean.shape})')
    else:
        print(f'  [WARN] No bar scaler in checkpoint — bars unscaled')

    # Apply market feature scaler
    feat_mean = ps.get('market_feature_mean')
    feat_std  = ps.get('market_feature_std')
    feat_clip = ps.get('market_feature_clip', 8.0)
    if feat_mean is not None and feat_std is not None:
        full_dataset.market_feature_mean = feat_mean.to('cpu') if hasattr(feat_mean, 'to') else torch.tensor(feat_mean)
        full_dataset.market_feature_std  = feat_std.to('cpu')  if hasattr(feat_std,  'to') else torch.tensor(feat_std)
        full_dataset.market_feature_clip = float(feat_clip)
        print(f'  Market feature scaler applied')

    # Test loader
    test_subset = Subset(full_dataset, test_idx)
    test_loader = DataLoader(test_subset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Inference
    all_dir_logits, all_labels, all_conf, all_ret_labels = [], [], [], []
    all_attn, all_pfac = [], []

    with torch.no_grad():
        for batch in test_loader:
            mf   = batch['market_features'].to(DEVICE)
            ae   = batch.get('article_embeddings')
            am   = batch.get('article_mask')
            amv  = batch.get('article_metadata')
            mb   = batch.get('market_bars')

            if ae is not None:
                ae = ae.to(DEVICE)
            if am is not None:
                am = am.to(DEVICE)
            if amv is not None:
                amv = amv.to(DEVICE)
            if mb is not None:
                mb = mb.to(DEVICE)

            out = model(market_feat=mf, horizon='1h',
                        article_emb=ae, article_mask=am,
                        article_meta_vec=amv, market_bars=mb)

            import torch.nn.functional as F
            # Apply temperature scaling
            logits_scaled = out['dir_logits'] / max(temp, 0.1)
            all_dir_logits.append(logits_scaled.cpu())
            all_conf.append(out['confidence'].cpu())
            all_labels.append(batch['direction'].cpu())
            all_ret_labels.append(batch.get('return', torch.zeros(mf.shape[0])).cpu())
            if out.get('attn_weights') is not None:
                all_attn.append(out['attn_weights'].cpu())
            if out.get('p_fac_all') is not None:
                all_pfac.append(out['p_fac_all'].cpu())

    logits  = torch.cat(all_dir_logits)
    conf    = torch.cat(all_conf)
    labels  = torch.cat(all_labels).long()
    ret_lbl = torch.cat(all_ret_labels)

    import torch.nn.functional as F
    probs = F.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1)

    # Alert decisions
    max_prob = probs.max(dim=-1).values
    alerts   = ((conf >= tau) & (max_prob >= gamma)).long()

    # Metrics
    preds_np  = preds.numpy()
    labels_np = labels.numpy()
    conf_np   = conf.numpy()
    probs_np  = probs.numpy()
    ret_np    = ret_lbl.numpy()
    alerts_np = alerts.numpy()

    valid = labels_np >= 0
    f1  = compute_macro_f1(preds_np[valid], labels_np[valid])
    mcc = compute_mcc(preds_np[valid], labels_np[valid])
    ece = compute_ece(conf_np[valid], preds_np[valid], labels_np[valid])
    auc = compute_auc(probs_np[valid], labels_np[valid])
    prec, cov, _ = compute_alert_precision(conf_np, preds_np, labels_np,
                                           dir_probs=probs_np, tau=tau, gamma=gamma)

    bt = mini_backtest(conf_np, preds_np, ret_np,
                       tau=tau, gamma=gamma, dir_probs=probs_np,
                       transaction_cost=0.001, slippage=0.0)

    result = {
        'fold': fold_idx,
        'test_size': len(test_idx),
        'test_range': [int(test_idx[0]), int(test_idx[-1])],
        'macro_f1': float(f1),
        'mcc': float(mcc),
        'ece': float(ece),
        'auc': float(auc),
        'alert_precision': float(prec),
        'alert_coverage': float(cov),
        'alert_sharpe': float(bt.get('alert_sharpe', 0)),
        'alert_sortino': float(bt.get('alert_sortino', 0)),
        'alert_max_dd': float(bt.get('alert_max_dd', 0)),
        'pnl': float(bt.get('pnl', 0)),
        'position_pnl': float(bt.get('position_pnl', 0)),
        'position_hit_rate': float(bt.get('alert_hit_rate', 0)),
        'tau': tau,
        'gamma': gamma,
        'temperature': temp,
        'checkpoint_epoch': meta['best_overall_epoch'],
        'deployable': meta['final_deployable'],
    }
    all_results.append(result)
    print(f'  F1={f1:.3f}  MCC={mcc:.3f}  ECE={ece:.4f}  '
          f'Sharpe={bt.get("alert_sharpe",0):.3f}  '
          f'Prec={prec:.3f}  Cov={cov:.3f}  PnL={bt.get("pnl",0):.3f}')

# ── Summary ───────────────────────────────────────────────────────────────────
if all_results:
    keys = ['macro_f1','mcc','ece','auc','alert_precision','alert_coverage',
            'alert_sharpe','alert_sortino','alert_max_dd','pnl','position_pnl','position_hit_rate']
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results if k in r]
        summary[f'{k}_mean'] = float(np.mean(vals))
        summary[f'{k}_std']  = float(np.std(vals, ddof=1) if len(vals) > 1 else 0)

    output = {
        'n_folds': len(all_results),
        'folds': all_results,
        'summary': summary,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\n=== WALK-FORWARD SUMMARY ({len(all_results)} folds) ===')
    print(f"  F1:       {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}")
    print(f"  MCC:      {summary['mcc_mean']:.3f} ± {summary['mcc_std']:.3f}")
    print(f"  ECE:      {summary['ece_mean']:.4f} ± {summary['ece_std']:.4f}")
    print(f"  Sharpe:   {summary['alert_sharpe_mean']:.3f} ± {summary['alert_sharpe_std']:.3f}")
    print(f"  Prec:     {summary['alert_precision_mean']:.3f} ± {summary['alert_precision_std']:.3f}")
    print(f"  PnL:      {summary['pnl_mean']:.3f} ± {summary['pnl_std']:.3f}")
    print(f'\nSaved: {OUTPUT_FILE}')
