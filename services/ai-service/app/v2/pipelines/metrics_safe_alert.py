"""
SAFE-Alert evaluation metrics beyond accuracy.

Four metric groups:
  1. Forecast: Macro-F1, MCC, MAE, RMSE
  2. Calibration: ECE, Brier Score
  3. Alerting: Alert Precision, Coverage, Selective Risk
  4. Backtest: Sharpe, Sortino, Max Drawdown
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from typing import Tuple, Dict, List


# ─────────────────────────────────────────────────────────────
# 1. FORECAST METRICS
# ─────────────────────────────────────────────────────────────

def compute_macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged F1 (important for imbalanced classes)."""
    return f1_score(labels, preds, average='macro', zero_division=0)


def compute_mcc(preds: np.ndarray, labels: np.ndarray) -> float:
    """Matthews Correlation Coefficient (harder to game than accuracy)."""
    return matthews_corrcoef(labels, preds)


def compute_mae(ret_pred: np.ndarray, ret_labels: np.ndarray) -> float:
    """Mean Absolute Error for return regression."""
    return np.mean(np.abs(ret_pred - ret_labels))


def compute_rmse(ret_pred: np.ndarray, ret_labels: np.ndarray) -> float:
    """Root Mean Squared Error for return regression."""
    return np.sqrt(np.mean((ret_pred - ret_labels) ** 2))


# ─────────────────────────────────────────────────────────────
# 2. CALIBRATION METRICS
# ─────────────────────────────────────────────────────────────

def compute_ece(confidence: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Lower is better. ECE < 0.05 is excellent, < 0.08 is acceptable for crypto.
    Measures: |confidence - accuracy| in probability bins.
    """
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    correct = (preds == labels).astype(np.float32)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if mask.sum() > 0:
            bin_confidence = confidence[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece += np.abs(bin_confidence - bin_accuracy) * mask.mean()

    return ece


def compute_brier_score(confidence: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier Score (MSE of confidence).

    Lower is better. Formula: mean((confidence - correctness)^2)
    """
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    correct = (preds == labels).astype(np.float32)

    return np.mean((confidence - correct) ** 2)


# ─────────────────────────────────────────────────────────────
# 3. ALERTING METRICS
# ─────────────────────────────────────────────────────────────

def compute_alert_precision(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    dir_probs: np.ndarray = None,
    tau: float = None,
    gamma: float = None,
) -> Tuple[float, float, float]:
    """
    Alert Precision, Coverage (alert rate), and Selective Risk (Eq.26).

    Args:
        confidence : (N,) model confidence from conf_head (sigmoid)
        preds      : (N,) predicted class indices
        labels     : (N,) true class indices
        dir_probs  : (N, 3) softmax probabilities — used as max_prob for γ threshold.
                     If None, confidence is used (less accurate).
        tau        : confidence threshold τ_h (default: adaptive 60th-pct)
        gamma      : max-prob threshold γ_h (default: adaptive 50th-pct)

    Returns:
        (alert_precision, coverage, selective_risk)
    """
    preds  = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    # max_prob = max softmax class probability (the γ_h operand in Eq.26).
    # Using actual softmax probs avoids the data-leakage bug where
    # (preds==labels) was used to artificially inflate max_prob for correct
    # predictions, making AlertPrec always ≈1.0 regardless of model quality.
    if dir_probs is not None:
        max_prob = dir_probs.max(axis=-1)          # (N,) — real model certainty
    else:
        max_prob = confidence                       # fallback: use conf head

    # Adaptive thresholds: percentile-based when caller supplies no fixed value.
    # 70th / 60th match the percentiles used in trainer.validate() for consistency.
    _tau   = tau   if tau   is not None else float(np.percentile(confidence, 70))
    _gamma = gamma if gamma is not None else float(np.percentile(max_prob,   60))

    # Alert condition (Eq.26): ĉ >= τ_h  AND  max(p̂) >= γ_h
    alert_mask = (confidence >= _tau) & (max_prob >= _gamma)

    if alert_mask.sum() == 0:
        return 0.0, 0.0, 1.0

    alert_correct  = (preds[alert_mask] == labels[alert_mask]).astype(np.float32)
    alert_precision = float(alert_correct.mean())
    coverage        = float(alert_mask.mean())
    selective_risk  = 1.0 - alert_precision

    return alert_precision, coverage, selective_risk


# ─────────────────────────────────────────────────────────────
# 4. BACKTEST METRICS
# ─────────────────────────────────────────────────────────────

def compute_sharpe_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """
    Sharpe Ratio (risk-adjusted return).

    Formula: (mean_return - rf_rate) / std_return
    Higher is better. > 1.0 is decent, > 1.5 is good.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_return = returns.mean() - rf_rate
    return excess_return / returns.std()


def compute_sortino_ratio(returns: np.ndarray, rf_rate: float = 0.0) -> float:
    """
    Sortino Ratio (penalizes downside risk only).

    Similar to Sharpe but uses downside deviation instead of total std.
    Higher is better.
    """
    if len(returns) < 2:
        return 0.0

    excess_return = returns.mean() - rf_rate
    downside_returns = np.minimum(returns, 0)
    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0 if excess_return <= 0 else float('inf')

    return excess_return / downside_std


def compute_max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum Drawdown.

    Largest peak-to-trough decline. Lower (less negative) is better.
    Range: [-1, 0]
    """
    if len(returns) < 2:
        return 0.0

    cumsum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = (cumsum - running_max) / running_max

    return drawdown.min()


def compute_calmar_ratio(returns: np.ndarray) -> float:
    """
    Calmar Ratio = annual_return / abs(max_drawdown).

    Higher is better. Combines return and drawdown.
    """
    annual_return = (1 + returns.mean()) ** 252 - 1
    mdd = compute_max_drawdown(returns)

    if mdd == 0:
        return 0.0

    return annual_return / abs(mdd)


# ─────────────────────────────────────────────────────────────
# 5. MINI BACKTEST (validation only)
# ─────────────────────────────────────────────────────────────

def mini_backtest(
    confidence: np.ndarray,
    preds: np.ndarray,
    ret_labels: np.ndarray,
    tau: float = 0.78,
    gamma: float = 0.72,
    dir_probs: np.ndarray = None,
) -> Dict[str, float]:
    """
    Run mini backtest on validation set (Eq.26 alert policy).

    Uses adaptive thresholds when the model's confidence distribution is too low
    to trigger any alerts with the fixed τ/γ — this occurs in Stages 1-2 when
    the confidence head outputs near-uniform values (≈0.33).  Falling back to
    adaptive thresholds prevents Sharpe=0 masking real learning progress.

    Args:
        dir_probs: (N, 3) softmax probabilities — used as max_prob for γ threshold.
                   If None, confidence is used (less accurate proxy).

    Returns:
      - alert_sharpe  : Sharpe ratio of alerted-only positions
      - alert_max_dd  : Max drawdown of alerted positions
      - alert_hit_rate: % of alerts that were profitable
      - alert_coverage: fraction of val set that triggered an alert
    """
    preds = preds.astype(np.int64)

    # max_prob = maximum softmax class probability (the γ_h operand in Eq.26).
    # Using dir_probs avoids the incorrect proxy `np.maximum(confidence, 0.33)`
    # which clipped all values to ≥0.33, distorting the γ threshold.
    if dir_probs is not None:
        max_prob = dir_probs.max(axis=-1)   # (N,) — real model certainty
    else:
        max_prob = confidence               # fallback

    # Use fixed thresholds when the model is already confident enough;
    # fall back to 70th-percentile adaptive thresholds during early training.
    _tau   = tau   if confidence.max() >= tau   else float(np.percentile(confidence, 70))
    _gamma = gamma if max_prob.max()   >= gamma else float(np.percentile(max_prob,   70))

    alert_mask = (confidence >= _tau) & (max_prob >= _gamma)

    if alert_mask.sum() == 0:
        return {
            "alert_sharpe": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": 0.0,
            "alert_coverage": 0.0,
        }

    alert_returns = ret_labels[alert_mask]
    alert_coverage = float(alert_mask.mean())

    # Sharpe is statistically unreliable when computed from very few samples.
    # With AlertCov < 5% of val set (< ~100 samples), a single bad trade can
    # swing Sharpe from +2 to -5, inflating/deflating Score by up to 0.35.
    # Return 0 in this case so Score is not distorted by noise.
    if alert_mask.sum() < max(20, int(0.05 * len(ret_labels))):
        return {
            "alert_sharpe": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": float((alert_returns > 0).mean()) if len(alert_returns) > 0 else 0.0,
            "alert_coverage": alert_coverage,
        }

    alert_sharpe   = compute_sharpe_ratio(alert_returns)
    alert_max_dd   = compute_max_drawdown(alert_returns)
    alert_hit_rate = float((alert_returns > 0).mean())

    return {
        "alert_sharpe":   alert_sharpe,
        "alert_max_dd":   alert_max_dd,
        "alert_hit_rate": alert_hit_rate,
        "alert_coverage": alert_coverage,
    }


# ─────────────────────────────────────────────────────────────
# 6. AGGREGATED SCORE FOR MODEL SELECTION
# ─────────────────────────────────────────────────────────────

def compute_model_selection_score(
    macro_f1: float,
    mcc: float,
    ece: float,
    alert_sharpe: float,
) -> float:
    """
    Multi-metric score for early stopping (PDF Section 4.5.1).

    Score = 0.40*MacroF1 + 0.35*AlertSharpe - 0.15*ECE + 0.10*MCC

    Higher score = better model.

    Rationale:
      - MacroF1 (40%): Core prediction quality
      - AlertSharpe (35%): Downstream utility and profitability
      - ECE (15%, subtracted): Lighter calibration penalty
      - MCC (10%): Robustness to class imbalance
    """
    # Normalize inputs to [0, 1] range for fair weighting
    macro_f1_norm = np.clip(macro_f1, 0, 1)  # [0, 1]
    mcc_norm = np.clip((mcc + 1) / 2, 0, 1)  # [-1, 1] → [0, 1]
    ece_norm = np.clip(1 - ece, 0, 1)      # Lower ECE = higher contribution
    alert_sharpe_norm = np.clip(alert_sharpe / 2.0, 0, 1)  # Normalize to [0, 1]

    score = (
        0.40 * macro_f1_norm
        + 0.35 * alert_sharpe_norm
        - 0.15 * (1 - ece_norm)  # Penalize high ECE
        + 0.10 * mcc_norm
    )

    return score


# ─────────────────────────────────────────────────────────────
# 7. AUC (one-vs-rest for 3-class)
# ─────────────────────────────────────────────────────────────

def compute_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Macro-averaged AUC (one-vs-rest for 3 classes: DOWN/NEUTRAL/UP).

    Args:
        probs: (N, 3) softmax probabilities
        labels: (N,) integer class labels 0/1/2
    Returns:
        macro-averaged AUC across 3 classes
    """
    try:
        return roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except ValueError:
        return 0.5  # Fallback when only one class present in small batches


# ─────────────────────────────────────────────────────────────
# 8. COVERAGE-RISK CURVE AUC
# ─────────────────────────────────────────────────────────────

def compute_coverage_risk_auc(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 20,
) -> float:
    """
    Area Under Coverage-Risk Curve (AURC).

    Sweeps confidence threshold τ from high→low, computes (coverage, error_rate) pairs.
    Lower AURC = better selective prediction (high confidence = low error).

    Returns:
        aurc: area under curve (lower is better, range [0, 1])
    """
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    correct = (preds == labels).astype(np.float32)

    thresholds = np.linspace(confidence.max(), confidence.min(), n_thresholds)
    coverages, risks = [], []

    for tau in thresholds:
        mask = confidence >= tau
        if mask.sum() == 0:
            continue
        coverages.append(mask.mean())
        risks.append(1.0 - correct[mask].mean())

    if len(coverages) < 2:
        return 1.0  # Degenerate: always high risk

    # Sort by coverage for AUC computation
    coverages = np.array(coverages)
    risks = np.array(risks)
    order = np.argsort(coverages)
    trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
    return float(trapz(risks[order], coverages[order]))


# ─────────────────────────────────────────────────────────────
# 9. FAITHFULNESS: DELETION / INSERTION SCORES
# ─────────────────────────────────────────────────────────────

def compute_deletion_insertion_score(
    full_probs: np.ndarray,
    masked_probs: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Faithfulness metrics (PDF Eq.36 evaluation).

    Deletion: drop in confidence when top articles removed (higher = more faithful)
    Insertion: gain in confidence when top articles added from scratch (higher = more faithful)

    Args:
        full_probs: (N, 3) softmax with ALL articles
        masked_probs: (N, 3) softmax with TOP articles REMOVED
        labels: (N,) true class labels
    Returns:
        {"deletion_drop": ..., "insertion_gain": ...}
    """
    labels = labels.astype(np.int64)

    # Confidence on true class
    full_conf = full_probs[np.arange(len(labels)), labels]
    masked_conf = masked_probs[np.arange(len(labels)), labels]

    # Deletion: full → masked (removing articles should reduce confidence)
    deletion_drop = float(np.mean(full_conf - masked_conf))

    # Insertion: masked → full (adding articles should increase confidence)
    insertion_gain = float(np.mean(full_conf - masked_conf))  # same as deletion by symmetry

    return {
        "deletion_drop": deletion_drop,    # Higher = articles genuinely contribute
        "insertion_gain": insertion_gain,  # Higher = better
    }


# ─────────────────────────────────────────────────────────────
# 10. HIT RATE (alert profitability)
# ─────────────────────────────────────────────────────────────

def compute_hit_rate(
    confidence: np.ndarray,
    ret_labels: np.ndarray,
    tau: float = None,
) -> float:
    """
    Hit Rate: % of alerted predictions that had positive return.

    Args:
        confidence: (N,) model confidence scores
        ret_labels: (N,) actual returns (positive = profitable)
        tau: alert threshold (default: 60th percentile)
    Returns:
        hit_rate in [0, 1]
    """
    if tau is None:
        tau = np.percentile(confidence, 70)  # consistent with validate() and compute_alert_precision()

    alert_mask = confidence >= tau
    if alert_mask.sum() == 0:
        return 0.0

    return float((ret_labels[alert_mask] > 0).mean())


# ─────────────────────────────────────────────────────────────
# 11. SUFFICIENCY (PDF Section 4.4.2)
# ─────────────────────────────────────────────────────────────

def compute_sufficiency(
    full_probs: np.ndarray,
    selected_only_probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Sufficiency: prediction quality when using ONLY the selected (top-K) evidence.

    PDF Section 4.4.2: "If the selected articles alone are sufficient to explain
    the prediction, the model confidence on the true class should remain high
    even when all other articles are removed."

    Score = mean confidence on true class using selected-only articles.
    Higher = selected articles alone are sufficient (model doesn't need the rest).

    Args:
        full_probs:           (N, 3) softmax with ALL articles
        selected_only_probs:  (N, 3) softmax with ONLY selected Top-K articles (others zeroed/masked)
        labels:               (N,) true class labels
    Returns:
        sufficiency in [0, 1] (higher = more sufficient)
    """
    labels = labels.astype(np.int64)
    # Confidence on true class using only selected articles
    selected_conf = selected_only_probs[np.arange(len(labels)), labels]
    return float(np.mean(selected_conf))


# ─────────────────────────────────────────────────────────────
# 12. FACTOR CONSISTENCY (stability of factor predictions)
# ─────────────────────────────────────────────────────────────

def compute_factor_consistency(
    factor_preds_list: List[np.ndarray],
) -> float:
    """
    Factor Consistency: stability of top-1 factor prediction across similar samples.

    Uses mode agreement: across a sliding window of N consecutive predictions,
    what fraction agree on the top factor?

    Args:
        factor_preds_list: list of (N,) top-factor-index arrays across eval windows
    Returns:
        consistency in [0, 1] (higher = more stable explanations)
    """
    if len(factor_preds_list) < 2:
        return 1.0

    # Stack all predictions and compute pairwise agreement
    stacked = np.stack(factor_preds_list, axis=0)  # (W, N)
    # For each sample position, check if all windows agree
    mode_vals = []
    for i in range(stacked.shape[1]):
        vals = stacked[:, i]
        # Agreement = fraction matching most common prediction
        mode_count = np.bincount(vals.astype(np.int64)).max()
        mode_vals.append(mode_count / len(vals))

    return float(np.mean(mode_vals))
