"""
SAFE-Alert evaluation metrics beyond accuracy.

Four metric groups:
  1. Forecast: Macro-F1, MCC, MAE, RMSE
  2. Calibration: ECE, Brier Score
  3. Alerting: Alert Precision, Coverage, Selective Risk
  4. Backtest: Sharpe, Sortino, Calmar, Max Drawdown
"""

import warnings
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
        tau        : confidence threshold τ_h (default: adaptive 70th-pct)
        gamma      : max-prob threshold γ_h (default: adaptive 60th-pct)

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
    # Semi-deviation (lower partial moment): sqrt(mean(min(r, 0)²)).
    # Zeros (from positive returns) correctly contribute 0 to the mean-squared sum.
    # Previous bug: np.std(np.minimum(r, 0)) computes std of the zero-padded sequence,
    # which inflates denominator denominator (zeros pull mean toward 0, shrinking std),
    # producing a Sortino that is ~2x larger than the standard semi-deviation formula.
    downside_std = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))

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
    Annualization uses 8760 = 365 * 24 hours (BTC trades 24/7, not 252 stock-market days).
    Uses simple (arithmetic) annualization — compound formula (1+r)^8760 gives astronomically
    large values for hourly crypto returns where mean_return ~ 0.5%, making logged Calmar
    unreadable and clipping the utility_bonus contribution to ±0.5 for almost all policies.
    """
    annual_return = returns.mean() * 8760  # arithmetic annualization for hourly BTC
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
    transaction_cost: float = 0.001,
) -> Dict[str, float]:
    """
    Run mini backtest on validation set (Eq.26 alert policy).

    Uses adaptive thresholds when the model's confidence distribution is too low
    to trigger any alerts with the fixed τ/γ — this occurs in Stages 1-2 when
    the confidence head outputs near-uniform values (≈0.33).  Falling back to
    adaptive thresholds prevents Sharpe=0 masking real learning progress.

    Args:
        dir_probs:        (N, 3) softmax probabilities — used as max_prob for γ threshold.
                          If None, confidence is used (less accurate proxy).
        transaction_cost: Fixed per-trade cost deducted from each net return
                          (PDF Section 4.2.4). Default 0.001 = 0.1% per trade.

    Returns:
      - alert_sharpe  : Sharpe ratio of direction-aware net returns after costs
      - alert_max_dd  : Max drawdown of net returns
      - alert_hit_rate: % of alerts with positive net return after costs
      - alert_coverage: fraction of val set that triggered an alert
      - pnl           : Cumulative net P&L over the alert window (PDF Table 4)
    """
    preds = preds.astype(np.int64)

    # max_prob = maximum softmax class probability (the γ_h operand in Eq.26).
    # Using dir_probs avoids the incorrect proxy `np.maximum(confidence, 0.33)`
    # which clipped all values to ≥0.33, distorting the γ threshold.
    if dir_probs is not None:
        max_prob = dir_probs.max(axis=-1)   # (N,) — real model certainty
    else:
        max_prob = confidence               # fallback

    # Always use the fixed thresholds supplied by the caller (from search_alert_policy).
    # Adaptive fallback was removed: it caused cross-epoch Sharpe values to be computed
    # on different τ distributions, distorting early-stopping model selection scores.
    # Sharpe=0 in Stages 1-2 is correct — early stopping is inactive there anyway.
    _tau   = float(tau)
    _gamma = float(gamma)

    alert_mask = (confidence >= _tau) & (max_prob >= _gamma)

    if alert_mask.sum() == 0:
        return {
            "alert_sharpe": 0.0,
            "alert_sortino": 0.0,
            "alert_calmar": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": 0.0,
            "alert_coverage": 0.0,
            "pnl": 0.0,
        }

    alert_coverage = float(alert_mask.mean())

    # Direction-aware net returns (PDF Section 4.2.4):
    # UP pred (class=2) → LONG  → profit when price rises  → sign = +1
    # DOWN pred (class=0) → SHORT → profit when price falls → sign = -1
    # NEUTRAL pred (class=1) → no position (should not be alerted in practice)
    price_ret = ret_labels[alert_mask]
    trading_sign = np.where(preds[alert_mask] == 2, 1.0,
                   np.where(preds[alert_mask] == 0, -1.0, 0.0))
    net_returns = trading_sign * price_ret - transaction_cost

    # Sharpe is statistically unreliable when computed from very few samples.
    # With AlertCov < 5% of val set (< ~100 samples), a single bad trade can
    # swing Sharpe from +2 to -5, inflating/deflating Score by up to 0.35.
    # Return 0 in this case so Score is not distorted by noise.
    if alert_mask.sum() < max(20, int(0.05 * len(ret_labels))):
        return {
            "alert_sharpe": 0.0,
            "alert_sortino": 0.0,
            "alert_calmar": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": float((net_returns > 0).mean()) if len(net_returns) > 0 else 0.0,
            "alert_coverage": alert_coverage,
            "pnl": float(np.sum(net_returns)),
        }

    alert_sharpe   = compute_sharpe_ratio(net_returns)
    alert_max_dd   = compute_max_drawdown(net_returns)
    alert_hit_rate = float((net_returns > 0).mean())
    pnl            = float(np.sum(net_returns))

    return {
        "alert_sharpe":   alert_sharpe,
        "alert_sortino":  compute_sortino_ratio(net_returns),
        "alert_calmar":   compute_calmar_ratio(net_returns),
        "alert_max_dd":   alert_max_dd,
        "alert_hit_rate": alert_hit_rate,
        "alert_coverage": alert_coverage,
        "pnl":            pnl,
    }


def evaluate_alert_policy(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    ret_labels: np.ndarray,
    dir_probs: np.ndarray,
    tau: float,
    gamma: float,
) -> Dict[str, float]:
    """Evaluate one Eq.26 alert policy on a fixed validation split."""
    alert_prec, alert_cov, sel_risk = compute_alert_precision(
        confidence,
        preds,
        labels,
        dir_probs=dir_probs,
        tau=tau,
        gamma=gamma,
    )
    backtest = mini_backtest(
        confidence,
        preds,
        ret_labels,
        tau=tau,
        gamma=gamma,
        dir_probs=dir_probs,
    )
    return {
        "tau": float(tau),
        "gamma": float(gamma),
        "alert_precision": float(alert_prec),
        "alert_coverage": float(alert_cov),
        "selective_risk": float(sel_risk),
        "alert_sharpe": float(backtest["alert_sharpe"]),
        "alert_sortino": float(backtest["alert_sortino"]),
        "alert_calmar": float(backtest["alert_calmar"]),
        "alert_max_dd": float(backtest["alert_max_dd"]),
        "hit_rate": float(backtest["alert_hit_rate"]),
        "pnl": float(backtest.get("pnl", 0.0)),
    }


def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to logits."""
    temp = max(float(temperature), 1e-3)
    scaled = logits / temp
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-8, None)


def fit_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
    grid: List[float] | None = None,
) -> Dict[str, float]:
    """Fit a scalar temperature on validation logits only via small grid search."""
    if grid is None:
        # Extended to 6.0: in Stage 3 the model has been observed reaching T=2.8–3.5
        # (overconfident logits from Ldir pressure).  Capping at 3.5 would silently
        # leave the model miscalibrated if the true optimal T > 3.5.
        grid = [0.60, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0]

    labels = labels.astype(np.int64)
    best_temp = 1.0
    best_nll = float("inf")

    for temp in grid:
        probs = apply_temperature_to_logits(logits, temp)
        true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0)
        nll = float(-np.log(true_probs).mean())
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)

    return {"temperature": best_temp, "temperature_nll": best_nll}


def search_alert_policy(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    ret_labels: np.ndarray,
    dir_probs: np.ndarray,
    macro_f1: float,
    mcc: float,
    ece: float,
    tau_percentile: int = 70,
    gamma_percentile: int = 60,
    target_coverage: float = 0.35,
    policy_method: str = "grid",
) -> Dict[str, float]:
    """Grid-search a narrow validation-only Eq.26 policy around percentile seeds.

    This keeps the paper's confidence-aware alerting mechanism intact while making
    threshold selection less brittle than a single percentile pass.
    The search is intentionally biased toward thresholds at-or-above the thesis
    seeds so we do not inflate coverage at the cost of selective quality.
    """
    conf = np.asarray(confidence, dtype=np.float32)
    probs = np.asarray(dir_probs, dtype=np.float32)
    max_prob = probs.max(axis=-1)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    if policy_method not in {"grid", "conformal"}:
        raise ValueError(f"Unknown policy_method: {policy_method}")

    def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        scores = np.sort(scores.astype(np.float32))
        n = len(scores)
        if n == 0:
            return float(np.percentile(conf, tau_percentile))
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        k = min(max(k, 0), n - 1)
        return float(scores[k])

    if policy_method == "conformal":
        # Use correct predictions as calibration set for conformal thresholds.
        correct_mask = preds == labels
        conf_cal = conf[correct_mask]
        maxprob_cal = max_prob[correct_mask]
        # Fallback to all samples if calibration set too small.
        if conf_cal.shape[0] < max(20, int(0.05 * len(conf))):
            conf_cal = conf
            maxprob_cal = max_prob
        alpha = float(max(1e-3, 1.0 - target_coverage))
        tau = _conformal_quantile(conf_cal, alpha)
        gamma = _conformal_quantile(maxprob_cal, alpha)
        policy = evaluate_alert_policy(
            conf, preds, labels, ret_labels, probs, tau, gamma
        )
        score = compute_model_selection_score(
            macro_f1,
            mcc,
            ece,
            policy["alert_sharpe"],
        )
        min_alert_cov = max(0.05, target_coverage * 0.20)
        # Linear penalty on *any* under-coverage (replaces coverage_gap + under_coverage_soft).
        # Weight 0.50 means 12% coverage → penalty=0.115, which dominates a 0.2 Sharpe
        # advantage from keeping only 12% of trades (utility_bonus Δ ≈ 0.04).
        under_coverage = max(0.0, target_coverage - policy["alert_coverage"])
        over_coverage = max(0.0, policy["alert_coverage"] - (target_coverage + 0.05))
        under_coverage_hard = max(0.0, min_alert_cov - policy["alert_coverage"])
        coverage_penalty = (
            0.70 * under_coverage +
            0.20 * over_coverage +
            0.60 * under_coverage_hard
        )
        utility_bonus = (
            0.04 * policy["alert_precision"] +
            0.03 * policy["hit_rate"] +
            0.20 * float(np.clip(policy["alert_sharpe"], -0.50, 0.50)) +
            0.05 * float(np.clip(policy["alert_sortino"], -0.50, 0.80)) +
            0.04 * float(np.clip(policy["alert_calmar"], -0.50, 1.00))
        )
        negative_sharpe_penalty = 0.18 * max(0.0, -float(policy["alert_sharpe"]))
        objective = score + utility_bonus - coverage_penalty - negative_sharpe_penalty
        return {
            **policy,
            "model_score": float(score),
            "policy_objective": float(objective),
            "coverage_penalty": float(coverage_penalty),
            "utility_bonus": float(utility_bonus),
            "negative_sharpe_penalty": float(negative_sharpe_penalty),
            "policy_source": "validation_conformal",
            "tau_seed": float(np.percentile(conf, tau_percentile)),
            "gamma_seed": float(np.percentile(max_prob, gamma_percentile)),
        }

    tau_seed = float(np.percentile(conf, tau_percentile))
    gamma_seed = float(np.percentile(max_prob, gamma_percentile))

    # Expand percentiles around the seed to reduce brittleness while staying val-only.
    # Safety anchors (20/35/50) ensure at least some very lenient thresholds are always
    # included so the compliant pool (Cov >= MIN_ALERT_COV) stays non-empty even when
    # the model is highly confident and all high-percentile grid points give low coverage.
    tau_percentiles   = [20, 35, 50, max(1, tau_percentile - 5), tau_percentile, 75, 80, 85, 90]
    gamma_percentiles = [20, 35, 50, max(1, gamma_percentile - 5), gamma_percentile, 65, 70, 75, 80]
    tau_grid = np.unique(np.percentile(conf, tau_percentiles).astype(np.float32))
    gamma_grid = np.unique(np.percentile(max_prob, gamma_percentiles).astype(np.float32))

    if tau_seed not in tau_grid:
        tau_grid = np.unique(np.append(tau_grid, tau_seed))
    if gamma_seed not in gamma_grid:
        gamma_grid = np.unique(np.append(gamma_grid, gamma_seed))

    # Hard minimum coverage floor: reject policies that alert < MIN_ALERT_COV of samples.
    # Without this, the grid finds degenerate τ/γ pairs that alert only 2-5% of samples
    # with inflated Sharpe/Calmar/Sortino (tiny MDD from abstaining), which gives
    # unrealistically high utility_bonus that overwhelms the coverage_penalty.
    MIN_ALERT_COV = max(0.05, target_coverage * 0.20)   # ≥ 5% and ≥ 20% of target

    # Collect ALL candidate policies, then separate into compliant vs fallback.
    # Previous approach had a bug: the `best is None` check let the very first
    # evaluated policy bypass the floor when it was the only one evaluated before
    # best was set — meaning if ALL grid points have Cov < MIN_ALERT_COV, the
    # first-evaluated (not highest-coverage) policy would win.
    compliant: list[tuple[float, Dict[str, float]]] = []   # (objective, policy_dict)
    fallback:  list[tuple[float, Dict[str, float]]] = []   # (coverage, policy_dict) for tie-break

    def _compute_policy_objective(policy: Dict[str, float]) -> tuple[Dict[str, float], float]:
        score = compute_model_selection_score(
            macro_f1,
            mcc,
            ece,
            policy["alert_sharpe"],
        )
        under_coverage = max(0.0, target_coverage - policy["alert_coverage"])
        over_coverage = max(0.0, policy["alert_coverage"] - (target_coverage + 0.05))
        under_coverage_hard = max(0.0, MIN_ALERT_COV - policy["alert_coverage"])
        coverage_penalty = (
            0.70 * under_coverage +
            0.20 * over_coverage +
            0.60 * under_coverage_hard
        )
        utility_bonus = (
            0.04 * policy["alert_precision"] +
            0.03 * policy["hit_rate"] +
            0.20 * float(np.clip(policy["alert_sharpe"], -0.50, 0.50)) +
            0.05 * float(np.clip(policy["alert_sortino"], -0.50, 0.80)) +
            0.04 * float(np.clip(policy["alert_calmar"], -0.50, 1.00))
        )
        negative_sharpe_penalty = 0.18 * max(0.0, -float(policy["alert_sharpe"]))
        objective = score + utility_bonus - coverage_penalty - negative_sharpe_penalty
        enriched = {
            **policy,
            "model_score": float(score),
            "policy_objective": float(objective),
            "coverage_penalty": float(coverage_penalty),
            "utility_bonus": float(utility_bonus),
            "negative_sharpe_penalty": float(negative_sharpe_penalty),
            "policy_source": "validation_grid_search",
            "tau_seed": float(tau_seed),
            "gamma_seed": float(gamma_seed),
        }
        return enriched, objective

    for tau in tau_grid.tolist():
        for gamma in gamma_grid.tolist():
            policy = evaluate_alert_policy(
                conf, preds, labels, ret_labels, probs, float(tau), float(gamma)
            )
            enriched, objective = _compute_policy_objective(policy)
            if policy["alert_coverage"] >= MIN_ALERT_COV:
                compliant.append((objective, enriched))
            else:
                # Fallback: sort by coverage (highest = least degenerate)
                fallback.append((policy["alert_coverage"], enriched))

    if compliant:
        # Pick the policy with the highest objective among coverage-compliant candidates.
        _, best = max(compliant, key=lambda x: x[0])
    else:
        # No compliant policy found — pick the highest-coverage fallback to minimise
        # degeneracy. This can happen when the model is very confident and even the
        # most lenient τ/γ pair alerts fewer than MIN_ALERT_COV samples.
        _, best = max(fallback, key=lambda x: x[0])
        warnings.warn(
            f"search_alert_policy: no grid point met MIN_ALERT_COV={MIN_ALERT_COV:.3f}; "
            f"falling back to highest-coverage policy "
            f"(cov={best['alert_coverage']:.3f}). "
            f"Consider reducing tau_percentile/gamma_percentile.",
            RuntimeWarning,
            stacklevel=2,
        )

    assert best is not None
    return best


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
    # Normalize Sharpe symmetrically so negative utility penalizes score.
    # This keeps the PDF intent (utility should matter) while bounding influence.
    alert_sharpe_norm = np.clip(alert_sharpe / 2.0, -1, 1)

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
    no_article_probs: np.ndarray = None,
    selected_only_probs: np.ndarray = None,
) -> Dict[str, float]:
    """
    Faithfulness metrics (PDF Section 4.4.2 / Eq.36 evaluation).

    Deletion (= Comprehensiveness): drop in confidence when top-K selected articles
        are removed. Higher = more faithful (selected articles genuinely contributed).

    Insertion: gain in confidence when selected articles are added to an empty
        (market-only) context. Higher = selected articles add real informational value.
        True insertion requires a no-article baseline forward pass; when unavailable
        (no_article_probs=None) it falls back to deletion_drop for backward compat.

    Args:
        full_probs:          (N, 3) softmax with ALL articles
        masked_probs:        (N, 3) softmax with TOP-K articles REMOVED
        labels:              (N,) true class labels
        no_article_probs:    (N, 3) softmax with NO articles (market-only baseline).
                             When provided, enables the true insertion formula
                             (PDF Section 4.4.2): gain = selected_conf - no_art_conf.
        selected_only_probs: (N, 3) softmax with ONLY selected top-K articles.
                             Required together with no_article_probs for true insertion.
    Returns:
        {"deletion_drop": ..., "insertion_gain": ...}
    """
    labels = labels.astype(np.int64)
    idx = np.arange(len(labels))

    # Confidence on true class with and without top-K articles
    full_conf   = full_probs[idx, labels]    # (N,) all articles present
    masked_conf = masked_probs[idx, labels]  # (N,) top articles removed

    # Deletion / Comprehensiveness: full → masked.
    # Higher = removing top articles hurts more = they genuinely contributed.
    deletion_drop = float(np.mean(np.maximum(full_conf - masked_conf, 0.0)))

    # True Insertion (PDF Section 4.4.2): gain = selected_conf - no_art_conf
    # measures how much selected articles improve over market-only predictions.
    # This requires both a no-article baseline AND selected-only predictions.
    if no_article_probs is not None and selected_only_probs is not None:
        no_art_conf  = no_article_probs[idx, labels]     # (N,) market-only
        sel_conf     = selected_only_probs[idx, labels]  # (N,) selected only
        insertion_gain = float(np.mean(np.maximum(sel_conf - no_art_conf, 0.0)))
    else:
        # Fallback: use deletion_drop (backward compat when no-article pass is unavailable)
        insertion_gain = deletion_drop

    return {
        "deletion_drop": deletion_drop,    # Higher = articles genuinely contribute (faithfulness)
        "insertion_gain": insertion_gain,  # Higher = selected articles add real value vs market-only
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
        tau: alert threshold (default: 70th percentile)
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
    Sufficiency: drop in prediction quality when using ONLY selected (top-K) evidence.

    PDF Section 4.4.2: "giữ lại selected evidence rồi đo mức giảm chất lượng dự báo"
    = keep selected evidence, then measure the quality DECREASE vs full evidence.

    Formula: mean(full_conf - selected_only_conf) on true class.
    Lower value = smaller drop = selected articles alone are sufficient (good).
    Higher value = large drop = selected articles are NOT sufficient by themselves.

    Args:
        full_probs:           (N, 3) softmax with ALL articles
        selected_only_probs:  (N, 3) softmax with ONLY selected Top-K articles (others zeroed/masked)
        labels:               (N,) true class labels
    Returns:
        sufficiency_drop >= 0 (lower = more sufficient; 0 = selected articles fully sufficient)
    """
    labels = labels.astype(np.int64)
    full_conf     = full_probs[np.arange(len(labels)), labels]          # (N,) confidence with all articles
    selected_conf = selected_only_probs[np.arange(len(labels)), labels]  # (N,) confidence with selected only
    # Clamp to 0 to avoid negative drops from numerical noise
    return float(np.mean(np.maximum(full_conf - selected_conf, 0.0)))


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
