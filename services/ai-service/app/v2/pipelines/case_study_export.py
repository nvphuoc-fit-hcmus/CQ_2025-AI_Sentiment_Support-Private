"""Human-usefulness case-study export — paper Section 4.4.2.

Session 24: paper Section 4.4.2 lists **Human Usefulness** among explanation
metrics. This is a qualitative metric — a human (typically domain expert or
thesis advisor) reads a sample of predictions plus explanations and judges
whether the system's reasoning is useful for a trading decision.

This module generates that sample: N randomly-chosen test predictions
serialised to JSON with enough context (selected articles, top factors,
direction + confidence + ground truth, alert decision, faithfulness masked
probability) that a reviewer can audit quality in one pass. Output feeds
directly into the thesis Section 5.3 (qualitative case studies) and
supports the paper's HumanUsefulness metric.

Usage (invoked from train_safe_alert.py after final fold):

    from case_study_export import export_case_studies
    export_case_studies(
        model, test_loader, dataset,
        symbol="BTCUSDT", horizon="1h",
        n_samples=50, out_path=fold_dir / "case_studies.json",
        tau_h=val_eval_metrics["tau"], gamma_h=val_eval_metrics["gamma"],
        temperature_h=val_eval_metrics["temperature"],
    )

The JSON schema is stable so downstream tooling (thesis figures, reviewer
dashboards) can parse without changes across runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# Paper Eq.29 natural-language explanation generator.
# Template-based, deterministic, no LLM dependency. See explanation_generator.py
# for design rationale (faithfulness by construction, audit-friendly).
try:
    from explanation_generator import generate_explanation as _generate_explanation
except ImportError:  # invoked as a package
    from .explanation_generator import generate_explanation as _generate_explanation


# Direction labels — mirror the training label space (paper Eq.51).
_DIR_NAMES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def _get_article_titles(dataset, sample_idx: int, selected_mask: np.ndarray,
                       max_titles: int = 5,
                       subset_indices: Optional[list] = None) -> List[str]:
    """Return titles of the top-K selected articles for a given sample.

    Session 26 BUG FIX: when caller passes a loader that wraps a
    ``torch.utils.data.Subset(dataset, subset_indices)`` (walk-forward
    fold's test_loader does), ``sample_idx`` is the position WITHIN the
    subset's stream — NOT a direct index into ``dataset.valid_idx``. We
    remap through ``subset_indices[sample_idx]`` first so the resolved
    candle belongs to the test window, not the first candle of the
    training data.

    Relies on the dataset's internal article lookup. Gracefully falls back
    to placeholder strings if the lookup fails (e.g., test fold articles
    not loaded into memory).
    """
    try:
        if subset_indices is not None:
            dataset_pos = int(subset_indices[int(sample_idx)])
        else:
            dataset_pos = int(sample_idx)
        candle_idx = dataset.valid_idx[dataset_pos]
        article_indices = dataset.candle_to_articles.get(int(candle_idx), [])
        titles = []
        for art_i in article_indices[:max_titles * 4]:   # over-scan; filter by mask
            # Mask alignment: article_indices ordering matches the (K, ...)
            # tensor order consumed by the model. selected_mask is in that
            # same ordering.
            pos = len(titles)
            if pos >= len(selected_mask):
                break
            if bool(selected_mask[pos]):
                title = dataset.article_meta.iloc[art_i].get("title", "<no title>")
                titles.append(str(title)[:200])   # truncate ultra-long titles
                if len(titles) >= max_titles:
                    break
        return titles
    except Exception:
        return [f"<article lookup failed for sample {sample_idx}>"]


@torch.no_grad()
def export_case_studies(
    model,
    test_loader,
    dataset,
    *,
    symbol: str = "BTCUSDT",
    horizon: str = "1h",
    n_samples: int = 50,
    out_path: Path | str,
    tau_h: float = 0.7,
    gamma_h: float = 0.65,
    temperature_h: float = 1.0,
    policy_confidence_source: str = "raw",
    factor_names: Optional[List[str]] = None,
    seed: int = 42,
) -> Path:
    """Sample ``n_samples`` test predictions and export as JSON.

    Each case-study record captures everything a human needs to judge the
    system's reasoning: context, selected evidence, predicted factors,
    direction + confidence, alert decision, ground-truth return, and the
    faithfulness signal (prob drop when masking selected articles).
    """
    # ── Paths + seeding ─────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Lazy-load factor names from the dataset if not provided.
    if factor_names is None:
        try:
            from models.safe_alert_net import FACTOR_NAMES as _DEFAULT_FACTOR_NAMES
            factor_names = list(_DEFAULT_FACTOR_NAMES)
        except Exception:
            factor_names = [f"factor_{i}" for i in range(10)]

    # Session 26 — detect Subset wrapper to fix sample_idx → dataset_pos
    # mapping. When test_loader wraps Subset(dataset, test_indices), the
    # N-th stream position corresponds to dataset.valid_idx[test_indices[N]],
    # NOT dataset.valid_idx[N]. Pull the indices up front so every
    # _get_article_titles call can remap.
    _loader_ds = getattr(test_loader, "dataset", None)
    _subset_indices = getattr(_loader_ds, "indices", None)  # list | None

    # ── Run inference and collect per-sample records ───────────────────────
    model_was_training = model.training
    model.eval()
    records: list[dict] = []
    device = next(model.parameters()).device
    global_sample_idx = 0

    for batch in test_loader:
        if len(records) >= n_samples:
            break
        market_feat  = batch["market_features"].to(device)
        article_emb  = batch["article_embeddings"].to(device)
        article_meta = batch["article_metadata"].to(device)
        article_mask = batch["article_mask"].to(device)
        dir_labels   = batch["direction"].to(device)
        ret_labels   = batch["return"].to(device)
        # Session 23 P0 #3: include bar sequences when available
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

        # Do NOT squeeze scalar-prone axes — for batch_size=1 .squeeze()
        # collapses (B,) → () which then breaks `confidence[b]` indexing.
        # reshape(-1) gives a 1-D view that keeps indexing valid for any B.
        dir_logits  = outputs["dir_logits"].detach().cpu().numpy()
        ret_preds   = outputs["ret_pred"].detach().cpu().numpy().reshape(-1)
        confidence  = outputs["confidence"].detach().cpu().numpy().reshape(-1)
        sel_mask_t  = outputs.get("selected_mask")
        sel_mask    = (sel_mask_t.detach().cpu().numpy() if sel_mask_t is not None
                       else np.zeros((dir_logits.shape[0], article_emb.shape[1]), dtype=bool))
        p_fac_all_t = outputs.get("p_fac_all")
        p_fac_all   = (p_fac_all_t.detach().cpu().numpy() if p_fac_all_t is not None
                       else np.zeros((dir_logits.shape[0], article_emb.shape[1], len(factor_names))))
        attn_w_t    = outputs.get("attn_weights")
        attn_w      = (attn_w_t.detach().cpu().numpy() if attn_w_t is not None
                       else np.zeros((dir_logits.shape[0], article_emb.shape[1])))
        masked_logits = masked_outputs["dir_logits"].detach().cpu().numpy()

        dir_labels_np = dir_labels.cpu().numpy()
        ret_labels_np = ret_labels.cpu().numpy()

        B = dir_logits.shape[0]
        for b in range(B):
            # Respect total-sample cap; random-sample within stream to reduce
            # selection bias (head of test set isn't representative of whole).
            if len(records) >= n_samples:
                break
            if rng.random() > max(0.05, n_samples / max(len(test_loader.dataset), n_samples)):
                global_sample_idx += 1
                continue

            # Temperature-calibrated direction probs (paper Section 4.4.3
            # post-hoc T applied to logits).
            probs_full = _softmax_1d(dir_logits[b] / max(temperature_h, 1e-3))
            probs_mask = _softmax_1d(masked_logits[b] / max(temperature_h, 1e-3))
            pred_class = int(probs_full.argmax())
            pred_conf  = float(confidence[b])
            max_prob   = float(probs_full.max())
            policy_conf = pred_conf
            if policy_confidence_source == "position":
                policy_conf = pred_conf * (1.0 - float(probs_full[1]))
            alert      = bool(policy_conf >= tau_h and max_prob >= gamma_h)

            # Faithfulness signal — paper Eq.36 relative gap
            p_full_c = float(probs_full[pred_class])
            p_mask_c = float(probs_mask[pred_class])
            rel_gap  = (p_full_c - p_mask_c) / max(p_full_c, 1e-3)

            # Weighted factor distribution over top-K articles for this sample.
            # Zero-out padding rows then average with attention weights.
            fac_probs = p_fac_all[b]                      # (K, C)
            attn     = attn_w[b]                           # (K,)
            weighted_fac = (attn[:, None] * fac_probs).sum(axis=0)
            weighted_fac = weighted_fac / max(weighted_fac.sum(), 1e-8)
            # Top-3 factors by weighted prob
            top_fac_idx = np.argsort(-weighted_fac)[:3]
            top_factors = [
                {"name": factor_names[int(i)], "prob": float(weighted_fac[int(i)])}
                for i in top_fac_idx
            ]

            selected_titles = _get_article_titles(
                dataset, global_sample_idx, sel_mask[b], max_titles=5,
                subset_indices=_subset_indices,   # Session 26 fix: remap through Subset
            )

            records.append({
                "sample_index": int(global_sample_idx),
                "symbol": symbol,
                "horizon": horizon,
                # Prediction
                "prediction": {
                    "direction": _DIR_NAMES.get(pred_class, str(pred_class)),
                    "direction_probs": {
                        _DIR_NAMES[k]: float(probs_full[k])
                        for k in range(len(probs_full))
                    },
                    "confidence": pred_conf,
                    "policy_confidence": float(policy_conf),
                    "policy_confidence_source": str(policy_confidence_source),
                    # ret_preds is guaranteed 1-D after reshape(-1) above.
                    "return_pred_pct": float(ret_preds[b]),
                },
                # Ground truth (for reviewer to check correctness)
                "ground_truth": {
                    "direction": _DIR_NAMES.get(int(dir_labels_np[b]), str(int(dir_labels_np[b]))),
                    "return_pct": float(ret_labels_np[b]),
                },
                # Evidence / explanation
                "selected_articles": selected_titles,
                "top_factors": top_factors,
                # Alert + calibration
                "alert_decision": {
                    "fired": alert,
                    "tau_h": float(tau_h),
                    "gamma_h": float(gamma_h),
                    "temperature": float(temperature_h),
                    "max_prob": max_prob,
                    "policy_confidence": float(policy_conf),
                    "policy_confidence_source": str(policy_confidence_source),
                },
                # Faithfulness (paper Eq.36)
                "faithfulness": {
                    "p_full_pred_class": p_full_c,
                    "p_mask_pred_class": p_mask_c,
                    "relative_gap": float(rel_gap),
                },
                # Paper Eq.29 natural-language explanation
                # (Φ_txt template, faithful by construction).
                "nl_explanation": _generate_explanation(
                    direction=pred_class,
                    confidence=pred_conf,
                    horizon=horizon,
                    top_articles=[
                        {"title": t} for t in selected_titles
                    ],
                    top_factors=[
                        (rec["name"], rec["prob"]) for rec in top_factors
                    ],
                    alert_decision=alert,
                    return_pred=float(ret_preds[b]),
                    asset=symbol.replace("USDT", "").replace("USD", ""),
                ),
                # Reviewer scratch field (fill in manually)
                "human_rating": None,
                "human_comment": None,
            })
            global_sample_idx += 1

    if model_was_training:
        model.train()

    # ── Persist ────────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": 1,
                "description": "SAFE-Alert case studies for paper Section 4.4.2 HumanUsefulness review",
                "n_samples": len(records),
                "symbol": symbol,
                "horizon": horizon,
                "tau_h": float(tau_h),
                "gamma_h": float(gamma_h),
                "temperature_h": float(temperature_h),
                "policy_confidence_source": str(policy_confidence_source),
                "seed": int(seed),
                "records": records,
            },
            f, indent=2, ensure_ascii=False,
        )
    return out_path
