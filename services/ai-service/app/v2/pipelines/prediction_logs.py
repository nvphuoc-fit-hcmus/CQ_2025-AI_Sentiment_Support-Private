"""Prediction + explanation logs export — paper Section 4.1.1 third data source.

Session 25: paper Section 4.1.1 lists three raw data streams:
  1. News stream (articles from crawler)
  2. Market stream (OHLCV feeds)
  3. **Prediction + system logs** (AI Service / Notification / Core / Backtest)

The third stream is a PRODUCTION artefact — once the system is live, every
prediction, explanation, and alert decision is persisted and feeds
downstream analysis (audit, backtest regeneration, retrain signal).

Historically our offline training pipeline did not emit this stream, so
the paper's "3-source" claim was a documented deviation (out-of-scope for
offline runs). This module closes that gap: after every fold's final
evaluation we export ALL test-set predictions in the same schema the
production Notification Service would emit, giving reviewers an
end-to-end paper trail and enabling replay / audit workflows.

The log schema is deliberately stable and identical to what
``NotificationService`` and ``CoreService`` produce in live deployment,
so thesis figures / reviewer tools can consume either source
interchangeably.

Schema (v1) — one JSONL line per test sample:
    {
        "ts_decision":       ISO-8601 UTC of the decision mark,
        "symbol":            "BTCUSDT",
        "horizon":           "1h",
        "direction":         "UP" | "NEUTRAL" | "DOWN",
        "direction_probs":   {"DOWN": p0, "NEUTRAL": p1, "UP": p2},
        "return_pred_pct":   float (predicted %),
        "return_actual_pct": float (ground truth %),
        "volatility_pred":   float (volatility extension output; 0.0 when disabled),
        "volatility_actual": float,
        "confidence":        float ∈ [0, 1],
        "max_prob":          float ∈ [0, 1] (calibrated by frozen T),
        "alert":             bool (Eq.26 dual-threshold decision),
        "alert_tau_h":       float,
        "alert_gamma_h":     float,
        "temperature_h":     float (post-hoc T, frozen from val),
        "selected_news_idx": [int]   (top-K article indices in the sample),
        "top_factors":       [{"name": str, "prob": float}],
        "faithfulness_gap":  float   (relative, paper Eq.36 semantics),
        "fold":              int,
    }

Usage:
    from prediction_logs import export_prediction_logs
    export_prediction_logs(
        model, test_loader, dataset,
        symbol="BTCUSDT", horizon="1h", fold=1,
        tau_h=..., gamma_h=..., temperature_h=...,
        out_path=fold_dir / "prediction_logs.jsonl",
    )

The output JSONL (not JSON array) is append-friendly and works with
line-by-line streaming — matches production log-pipeline idioms.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F


_DIR_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


def _softmax_row(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-3)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@torch.no_grad()
def export_prediction_logs(
    model,
    test_loader,
    dataset,
    *,
    symbol: str = "BTCUSDT",
    horizon: str = "1h",
    fold: int = 1,
    out_path: Path | str,
    tau_h: float = 0.7,
    gamma_h: float = 0.65,
    temperature_h: float = 1.0,
    policy_confidence_source: str = "raw",
    factor_names: Optional[List[str]] = None,
    top_p_factors: int = 3,
) -> Path:
    """Run every test sample through the model and persist one log row per.

    Emits JSONL (newline-delimited JSON) for cheap streaming consumption.
    Handles bar sequences + volatility prediction when the model was
    trained with them (Sessions 23-24 features).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if factor_names is None:
        try:
            from models.safe_alert_net import FACTOR_NAMES as _DEFAULTS
            factor_names = list(_DEFAULTS)
        except Exception:
            factor_names = [f"factor_{i}" for i in range(10)]

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device

    # Candle timestamps (decision marks) for schema field ts_decision.
    # Session 26 BUG FIX: the walk-forward test_loader wraps a
    # torch.utils.data.Subset(dataset, test_indices), so the N-th sample
    # produced by the loader corresponds to dataset position
    # ``test_indices[N]``, NOT position N in dataset.valid_idx. The old
    # code indexed valid_idx directly with the loader stream counter,
    # which emitted timestamps for the FIRST candles of the training
    # data instead of the test window — every prediction_logs.jsonl
    # was silently mis-timestamped.
    #
    # Fix: detect Subset and resolve via its .indices mapping. Falls
    # back to direct indexing when the loader wraps the raw dataset.
    _subset = getattr(test_loader, "dataset", None)
    _subset_indices = getattr(_subset, "indices", None)   # list | None

    def _resolve_ts(loader_pos: int) -> Optional[str]:
        try:
            if _subset_indices is not None:
                dataset_pos = int(_subset_indices[int(loader_pos)])
            else:
                dataset_pos = int(loader_pos)
            candle_idx = dataset.valid_idx[dataset_pos]
            ts = dataset.candle_df.iloc[int(candle_idx)]["datetime"]
            return str(ts)
        except Exception:
            return None

    lines_written = 0
    with open(out_path, "w", encoding="utf-8") as fp:
        global_sample_idx = 0
        for batch in test_loader:
            market_feat  = batch["market_features"].to(device)
            article_emb  = batch["article_embeddings"].to(device)
            article_meta = batch["article_metadata"].to(device)
            article_mask = batch["article_mask"].to(device)
            dir_labels   = batch["direction"].to(device)
            ret_labels   = batch["return"].to(device)
            vol_labels   = batch.get("volatility")
            if vol_labels is not None:
                vol_labels = vol_labels.to(device)
            market_bars  = batch.get("market_bars")
            if market_bars is not None:
                market_bars = market_bars.to(device)

            outputs = model(
                market_feat, horizon=horizon,
                article_emb=article_emb,
                article_mask=article_mask,
                article_meta_vec=article_meta,
                symbol=symbol,
                market_bars=market_bars,
            )
            masked_outputs = model.forward_masked(
                market_feat=market_feat, horizon=horizon,
                article_emb=article_emb, article_mask=article_mask,
                article_meta_vec=article_meta,
                symbol=symbol, market_bars=market_bars,
            )

            dir_logits    = outputs["dir_logits"].detach().cpu().numpy()
            ret_preds     = outputs["ret_pred"].detach().cpu().numpy().reshape(-1)
            confidence    = outputs["confidence"].detach().cpu().numpy().reshape(-1)
            vol_preds_t   = outputs.get("vol_pred")
            vol_preds     = (vol_preds_t.detach().cpu().numpy().reshape(-1)
                             if vol_preds_t is not None
                             else np.zeros(dir_logits.shape[0], dtype=np.float32))
            masked_logits = masked_outputs["dir_logits"].detach().cpu().numpy()
            sel_mask_t    = outputs.get("selected_mask")
            sel_mask      = (sel_mask_t.detach().cpu().numpy()
                             if sel_mask_t is not None
                             else np.zeros((dir_logits.shape[0], article_emb.shape[1]), dtype=bool))
            p_fac_all_t   = outputs.get("p_fac_all")
            attn_w_t      = outputs.get("attn_weights")
            p_fac_all     = (p_fac_all_t.detach().cpu().numpy() if p_fac_all_t is not None
                             else np.zeros((dir_logits.shape[0], article_emb.shape[1], len(factor_names))))
            attn_w        = (attn_w_t.detach().cpu().numpy() if attn_w_t is not None
                             else np.zeros((dir_logits.shape[0], article_emb.shape[1])))

            dir_labels_np = dir_labels.cpu().numpy()
            ret_labels_np = ret_labels.cpu().numpy()
            vol_labels_np = (vol_labels.cpu().numpy() if vol_labels is not None
                             else np.zeros(dir_logits.shape[0], dtype=np.float32))
            # Session 25: market regime stratification tag (paper 4.1.4).
            regime_t = batch.get("regime")
            regime_np = (regime_t.cpu().numpy() if regime_t is not None
                         else np.zeros(dir_logits.shape[0], dtype=np.int8))
            _REGIME_NAMES = {0: "SIDEWAYS", 1: "BULL", 2: "BEAR", 3: "VOLATILE"}

            B = dir_logits.shape[0]
            for b in range(B):
                probs_full = _softmax_row(dir_logits[b], temperature_h)
                probs_mask = _softmax_row(masked_logits[b], temperature_h)
                pred_class = int(probs_full.argmax())
                max_prob   = float(probs_full.max())
                conf       = float(confidence[b])
                policy_conf = conf
                if policy_confidence_source == "position":
                    policy_conf = conf * (1.0 - float(probs_full[1]))
                alert      = bool(policy_conf >= tau_h and max_prob >= gamma_h)

                p_full_c = float(probs_full[pred_class])
                p_mask_c = float(probs_mask[pred_class])
                rel_gap  = (p_full_c - p_mask_c) / max(p_full_c, 1e-3)

                # Weighted factor distribution over attended articles
                fac_weighted = (attn_w[b][:, None] * p_fac_all[b]).sum(axis=0)
                fac_weighted = fac_weighted / max(fac_weighted.sum(), 1e-8)
                top_idx = np.argsort(-fac_weighted)[:top_p_factors]
                top_facts = [
                    {"name": factor_names[int(i)], "prob": float(fac_weighted[int(i)])}
                    for i in top_idx
                ]

                row = {
                    "ts_decision":       _resolve_ts(global_sample_idx),
                    "symbol":            symbol,
                    "horizon":           horizon,
                    "direction":         _DIR_NAMES.get(pred_class, str(pred_class)),
                    "direction_probs": {
                        _DIR_NAMES[k]: float(probs_full[k]) for k in range(len(probs_full))
                    },
                    "direction_actual":  _DIR_NAMES.get(int(dir_labels_np[b]), str(int(dir_labels_np[b]))),
                    "return_pred_pct":   float(ret_preds[b]),
                    "return_actual_pct": float(ret_labels_np[b]),
                    "volatility_pred":   float(vol_preds[b]),
                    "volatility_actual": float(vol_labels_np[b]),
                    "confidence":        conf,
                    "policy_confidence": float(policy_conf),
                    "policy_confidence_source": str(policy_confidence_source),
                    "max_prob":          max_prob,
                    "alert":             alert,
                    "alert_tau_h":       float(tau_h),
                    "alert_gamma_h":     float(gamma_h),
                    "temperature_h":     float(temperature_h),
                    "selected_news_idx": [int(j) for j in np.where(sel_mask[b])[0].tolist()],
                    "top_factors":       top_facts,
                    "faithfulness_gap":  float(rel_gap),
                    "fold":              int(fold),
                    # Session 25 — paper 4.1.4 regime stratification
                    "regime":            _REGIME_NAMES.get(int(regime_np[b]), "UNKNOWN"),
                }
                fp.write(json.dumps(row, ensure_ascii=False))
                fp.write("\n")
                lines_written += 1
                global_sample_idx += 1

    if was_training:
        model.train()
    return out_path
