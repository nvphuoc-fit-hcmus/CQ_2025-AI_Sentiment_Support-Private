"""
SAFE-Alert evaluation metrics beyond accuracy.

Four metric groups:
  1. Forecast: Macro-F1, MCC, MAE, RMSE
  2. Calibration: ECE, Brier Score
  3. Alerting: Alert Precision, Coverage, Selective Risk
  4. Backtest: Sharpe, Sortino, Calmar, Max Drawdown
"""

import math
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from typing import Tuple, Dict, List, Optional


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
        # Last bin uses <= so confidence=1.0 is included (< 1.0 would miss it).
        upper = (confidence <= bins[i+1]) if i == n_bins - 1 else (confidence < bins[i+1])
        mask = (confidence >= bins[i]) & upper
        if mask.sum() > 0:
            bin_confidence = confidence[mask].mean()
            bin_accuracy = correct[mask].mean()
            ece += np.abs(bin_confidence - bin_accuracy) * mask.mean()

    return ece


def compute_ece_per_decile(
    confidence: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    n_deciles: int = 10,
) -> Dict[str, float]:
    """Per-decile calibration breakdown (Session 24 — paper Section 4.4.3).

    A single ECE scalar can hide where the model is mis-calibrated. Paper
    Section 4.4.3 calls for calibration diagnostics; this function returns
    per-confidence-decile stats so reviewers can see which confidence region
    is most miscalibrated:

        decile 0 (conf ∈ [0.0, 0.1)) → low-confidence predictions
        ...
        decile 9 (conf ∈ [0.9, 1.0]) → high-confidence predictions

    Each decile gets:
      - `count`     : number of samples
      - `conf_mean` : mean predicted confidence
      - `acc`       : fraction correct
      - `gap`       : |conf_mean − acc|    (miscalibration in this decile)

    Return dict keys: ``ece_decile_{i}_{field}`` for i in [0, n_deciles-1].
    An additional ``ece_worst_decile_gap`` flags the decile with largest
    miscalibration, useful for quick review signal.
    """
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)
    correct = (preds == labels).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_deciles + 1)

    out: Dict[str, float] = {}
    worst_gap = 0.0
    worst_decile = -1
    for i in range(n_deciles):
        upper = (confidence <= bins[i + 1]) if i == n_deciles - 1 else (confidence < bins[i + 1])
        mask = (confidence >= bins[i]) & upper
        n = int(mask.sum())
        if n > 0:
            conf_mean = float(confidence[mask].mean())
            acc = float(correct[mask].mean())
            gap = abs(conf_mean - acc)
        else:
            conf_mean = 0.0
            acc = 0.0
            gap = 0.0
        out[f"ece_decile_{i}_count"] = n
        out[f"ece_decile_{i}_conf_mean"] = conf_mean
        out[f"ece_decile_{i}_acc"] = acc
        out[f"ece_decile_{i}_gap"] = gap
        if n > 0 and gap > worst_gap:
            worst_gap = gap
            worst_decile = i
    out["ece_worst_decile"] = worst_decile
    out["ece_worst_decile_gap"] = worst_gap
    return out


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
    # Note: NEUTRAL predictions ARE included for policy search — Early epochs
    # model predicts ~all NEUTRAL, filtering them kills grid search coverage.
    # The "action-less" alerts contribute 0 return (trading_sign=0 in backtest)
    # so they don't inflate Sharpe anyway, and NEUTRAL precision reflects
    # model's classification quality honestly.
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
    if len(returns) < 2 or returns.std() < 1e-6:
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
    # Semi-deviation (lower partial moment): sqrt(mean(min(r - rf, 0)²)).
    # Uses excess returns (r - rf_rate) in the denominator, consistent with the
    # numerator. When rf_rate=0 this is identical to sqrt(mean(min(r,0)²)).
    downside_std = float(np.sqrt(np.mean(np.minimum(returns - rf_rate, 0.0) ** 2)))

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
    slippage: float = 0.0,
    spread_penalty: float = 0.0,
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
        transaction_cost: Fixed commission/fee per trade (PDF §4.2.4 first bullet).
                          Default 0.001 = 10 bps (typical crypto spot maker+taker).
        slippage:         Session 26 — PDF §4.2.4 second bullet. Expected price
                          impact per leg (market order fill vs. quoted mid).
                          Separate from transaction_cost so reviewers can tune
                          each cost source independently. Default 0.0 because
                          historical OHLCV doesn't expose realised slippage;
                          set e.g. 0.0003 (3 bps) for BTC market orders at
                          $5-10 M notional, 0.001 (10 bps) for smaller alts.
        spread_penalty:   Session 25 — paper §4.2.4 third bullet "optional
                          spread penalty if using full microstructure data".
                          Default 0.0 because standard OHLCV feeds do not
                          carry bid-ask spread; set to e.g. 0.0005 (5 bps)
                          when microstructure data is wired in to reflect
                          the realistic entry-exit half-spread cost.

        Total round-trip cost per trade = transaction_cost + slippage + spread_penalty.

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

    # NEUTRAL preds NOT filtered from alert_mask (filtering them breaks policy
    # grid search when model predicts mostly NEUTRAL). To avoid charging trading
    # cost on alerts the model declines to take a position on, we now multiply
    # the cost terms by ``has_position`` (1 for UP/DOWN preds, 0 for NEUTRAL).
    # Paper §4.2.4 cost model is paid only when entering/exiting a position;
    # an alert with trading_sign=0 corresponds to "publish signal, abstain
    # from trading", which has neither P&L nor execution cost.
    alert_mask = (confidence >= _tau) & (max_prob >= _gamma)

    if alert_mask.sum() == 0:
        return {
            "alert_sharpe": 0.0,
            "alert_sortino": 0.0,
            "alert_calmar": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": 0.0,
            "alert_coverage": 0.0,
            "trade_coverage": 0.0,
            "neutral_share_of_alerts": 0.0,
            "neutral_alert_rate": 0.0,  # legacy alias
            "n_position_alerts": 0,
            "position_pnl": 0.0,
            "pnl": 0.0,
            # Zero-alert is the strongest form of "Sharpe is unmeasurable" —
            # always flag suppressed so search_alert_policy routes this to
            # fallback instead of treating Sharpe = 0 as a real edge of zero.
            "alert_sharpe_suppressed": True,
        }

    alert_coverage = float(alert_mask.mean())

    # Direction-aware net returns (PDF Section 4.2.4):
    # UP pred (class=2) → LONG  → profit when price rises  → sign = +1
    # DOWN pred (class=0) → SHORT → profit when price falls → sign = -1
    # NEUTRAL pred (class=1) → no position (should not be alerted in practice)
    price_ret = ret_labels[alert_mask]
    alert_preds = preds[alert_mask]
    trading_sign = np.where(alert_preds == 2, 1.0,
                   np.where(alert_preds == 0, -1.0, 0.0))
    has_position = (trading_sign != 0.0).astype(np.float32)
    # Paper §4.2.4 three-component cost model (Session 26 separates slippage
    # from transaction_cost; previously they were implicitly combined):
    #   - transaction_cost : commission / exchange fee
    #   - slippage         : market-impact on fill vs. quoted mid
    #   - spread_penalty   : half-spread (requires microstructure data)
    # All three default to 0 or conservative fee only; callers override per
    # their execution model. Kept as separate args so a thesis reviewer can
    # read each component's contribution to reported Sharpe.
    # Sprint 1 FIX 1B: gate cost on has_position so NEUTRAL alerts (sign=0)
    # contribute exactly 0 to net_returns instead of -cost. Prior behaviour
    # silently bled fees on every NEUTRAL alert, distorting Sharpe and PnL
    # whenever the model predicted NEUTRAL within the alert region.
    cost_per_trade = transaction_cost + slippage + spread_penalty
    net_returns = trading_sign * price_ret - has_position * cost_per_trade
    # Position-only diagnostics: separate trade coverage (alerts that actually
    # take a position) from total alert coverage. trade_coverage is the
    # right input to coverage floors / target enforcement after FIX 1B,
    # because alert_coverage can be inflated by NEUTRAL alerts that never
    # enter the market.
    n_position_alerts = int(has_position.sum())
    trade_coverage = float(has_position.sum() / len(ret_labels))
    # Sprint 1 Patch 4 — name disambiguation. This is a SHARE OF ALERTS, not
    # a share of the dataset: it answers "of the alerts the policy fires, what
    # fraction are NEUTRAL". Two consumers care about it: (a) checkpoint-gate
    # readers who want to confirm the policy isn't dominated by NEUTRAL spam,
    # (b) policy designers tuning τ/γ to push the alert region toward
    # directional preds. The legacy field name ``neutral_alert_rate`` is kept
    # for backward compatibility but the canonical key is now
    # ``neutral_share_of_alerts``.
    neutral_share_of_alerts = float((alert_preds == 1).mean()) if alert_mask.any() else 0.0
    position_pnl = float(np.sum(net_returns[has_position.astype(bool)])) \
        if n_position_alerts > 0 else 0.0

    # Sharpe is statistically unreliable when computed from very few POSITION
    # alerts. FIX 3: the threshold is now applied to n_position_alerts (not raw
    # alert count) — a policy that triggers 200 NEUTRAL alerts and 5 position
    # alerts has effectively 5 trades for Sharpe, not 205. The suppressed flag
    # propagates to search_alert_policy so model selection cannot prefer a
    # zero-Sharpe-by-suppression policy over a real low-Sharpe one.
    min_required = max(20, int(0.05 * len(ret_labels)))
    suppressed = n_position_alerts < min_required
    if suppressed:
        return {
            "alert_sharpe": 0.0,
            "alert_sortino": 0.0,
            "alert_calmar": 0.0,
            "alert_max_dd": 0.0,
            "alert_hit_rate": 0.0,
            "alert_coverage": alert_coverage,
            "trade_coverage": trade_coverage,
            "neutral_share_of_alerts": neutral_share_of_alerts,
            "neutral_alert_rate": neutral_share_of_alerts,  # legacy alias
            "n_position_alerts": n_position_alerts,
            "position_pnl": position_pnl,
            "pnl": float(np.sum(net_returns)),
            "alert_sharpe_suppressed": True,
        }

    # Position-only stream is the right basis for Sharpe/Sortino/Calmar/hit_rate
    # after FIX 1B: NEUTRAL alerts contribute exact zeros that would dilute
    # both numerator (mean) and denominator (std) of Sharpe, biasing it toward 0
    # and masking the actual edge of the position-taking policy.
    position_returns = net_returns[has_position.astype(bool)]
    alert_sharpe   = compute_sharpe_ratio(position_returns)
    alert_max_dd   = compute_max_drawdown(position_returns)
    alert_hit_rate = float((position_returns > 0).mean())
    pnl            = float(np.sum(net_returns))

    return {
        "alert_sharpe":   alert_sharpe,
        "alert_sortino":  compute_sortino_ratio(position_returns),
        "alert_calmar":   compute_calmar_ratio(position_returns),
        "alert_max_dd":   alert_max_dd,
        "alert_hit_rate": alert_hit_rate,
        "alert_coverage": alert_coverage,
        "trade_coverage": trade_coverage,
        "neutral_share_of_alerts": neutral_share_of_alerts,
        "neutral_alert_rate": neutral_share_of_alerts,  # legacy alias
        "n_position_alerts": n_position_alerts,
        "position_pnl": position_pnl,
        "pnl":            pnl,
        "alert_sharpe_suppressed": False,
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
    confidence = np.asarray(confidence)
    preds = np.asarray(preds)
    if dir_probs is not None:
        dir_probs = np.asarray(dir_probs)
        max_prob = dir_probs.max(axis=-1)
    else:
        # Defensive fallback for direct callers. search_alert_policy always
        # passes class probabilities, but compute_alert_precision/mini_backtest
        # support confidence-only gating and this wrapper should match them.
        max_prob = confidence
    alert_mask = (confidence >= tau) & (max_prob >= gamma)
    if alert_mask.any():
        alert_preds = preds[alert_mask]
        alert_pred_dist_down = float((alert_preds == 0).mean())
        alert_pred_dist_neutral = float((alert_preds == 1).mean())
        alert_pred_dist_up = float((alert_preds == 2).mean())
    else:
        alert_pred_dist_down = alert_pred_dist_neutral = alert_pred_dist_up = 0.0

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
        # Sprint 1 FIX 1B + FIX 3 — propagate position-only diagnostics and
        # the suppressed-Sharpe flag so search_alert_policy and the model-
        # selection score can distinguish "real Sharpe = 0" (no edge) from
        # "Sharpe set to 0 because there were too few position alerts to
        # measure reliably". Without this, a low-coverage policy could win
        # on objective just because its noisy Sharpe was clipped to 0.
        "trade_coverage":          float(backtest.get("trade_coverage", 0.0)),
        # Canonical name + legacy alias kept for backward compatibility with
        # any reader still pulling "neutral_alert_rate" (e.g. older notebooks).
        "neutral_share_of_alerts": float(backtest.get("neutral_share_of_alerts", 0.0)),
        "neutral_alert_rate":      float(backtest.get("neutral_share_of_alerts", 0.0)),
        "n_position_alerts":       int(backtest.get("n_position_alerts", 0)),
        "position_pnl":            float(backtest.get("position_pnl", 0.0)),
        # Position-only hit rate kept under a distinct key so it never collides
        # with the legacy hit_rate metric val_losses computes via
        # compute_hit_rate (which is a confidence-weighted overall metric, not
        # a per-position win rate).
        "position_hit_rate":       float(backtest.get("alert_hit_rate", 0.0)),
        "alert_sharpe_suppressed": bool(backtest.get("alert_sharpe_suppressed", False)),
        # Sprint 12 Fix 2: directional imbalance penalty needs the alert-region
        # prediction distribution at policy-search time, before trainer-side
        # diagnostics are assembled.
        "alert_pred_dist_down":    alert_pred_dist_down,
        "alert_pred_dist_neutral": alert_pred_dist_neutral,
        "alert_pred_dist_up":      alert_pred_dist_up,
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
    *,
    use_lbfgs: bool = True,
    # Session 22 audit fix (Fix 2): narrowed bounds from (0.05, 10.0) → (0.5, 2.5).
    # Old bounds let T reach 10 (near-uniform softmax) or 0.05 (near-one-hot),
    # both indicate catastrophic training failure — not a real calibration
    # fix. Allowing wide bounds meant the LBFGS could silently "rescue" an
    # uncalibrated model at test time by fitting huge |T|, masking failures
    # in Lcal (paper's Eq.35) that should have been caught. The new bounds
    # [0.5, 2.5] cover genuine mild mis-calibration (±60% scaling) but hard-
    # clamp pathological values so reviewers see the real calibration quality.
    # Callers observing T_star at either boundary should treat the model as
    # uncalibrated and investigate (see warning emitted below).
    bounds: tuple = (0.5, 2.5),
) -> Dict[str, float]:
    """Fit a scalar temperature on validation logits (Guo et al., 2017).

    P1 #12: gradient-based LBFGS optimisation is the default path instead of
    grid search. LBFGS finds T* to machine precision in O(20-50) iterations
    and is still < 1 s for typical validation sizes.

    Session 22 audit fix (Fix 2): bounds narrowed [0.5, 2.5] and a warning is
    emitted when T_star lands within 5% of either bound — this flags the
    calibration as suspect and prevents the post-hoc scaler from silently
    masking Lcal failures. The paper (Eq.35) claims calibration is learned
    in-training via Lcal; if T_star → boundary, Lcal did not converge.

    Args:
        logits: (N, C) float array.
        labels: (N,) int array.
        grid:   Optional grid — used for the legacy path when ``use_lbfgs=False``.
        use_lbfgs: If True (default), use LBFGS on the log-temperature to
                   avoid constrained optimisation. Falls back to grid on
                   numerical error.
        bounds: (T_min, T_max) hard floor/ceiling applied after optimisation.

    Returns: {"temperature": T*, "temperature_nll": NLL@T*, "temperature_suspect": bool}
    """
    labels = labels.astype(np.int64)
    logits_f = logits.astype(np.float32)

    def _nll_at(temp: float) -> float:
        probs = apply_temperature_to_logits(logits_f, temp)
        true_probs = np.clip(probs[np.arange(len(labels)), labels], 1e-8, 1.0)
        return float(-np.log(true_probs).mean())

    # R4 Q2: Guard against NaN/Inf in input logits before LBFGS.
    # An upstream NaN from the model forward (e.g. very early training with
    # extreme initialization) would propagate through LBFGS closure → log_T
    # becomes NaN → final T is NaN → downstream softmax is NaN → silent
    # calibration failure. Sanitise first; if any non-finite remains, skip
    # LBFGS entirely and return T=1.0 (neutral) so training continues.
    if not np.isfinite(logits_f).all():
        n_bad = int(np.sum(~np.isfinite(logits_f)))
        import warnings as _w
        _w.warn(
            f"fit_temperature_scaling: {n_bad} non-finite values in logits; "
            f"skipping LBFGS and returning T=1.0 (identity).",
            RuntimeWarning, stacklevel=2,
        )
        return {"temperature": 1.0, "temperature_nll": float("nan")}

    if use_lbfgs:
        try:
            logits_t = torch.from_numpy(logits_f)
            labels_t = torch.from_numpy(labels)
            # Parametrise by log(T) so positivity is automatic and the
            # optimiser sees an unconstrained scalar (standard trick).
            log_T = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
            opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50,
                                    tolerance_grad=1e-6, tolerance_change=1e-8)

            def closure():
                opt.zero_grad()
                T = log_T.exp()
                nll = F.cross_entropy(logits_t / T, labels_t)
                nll.backward()
                return nll

            opt.step(closure)
            T_star = float(log_T.detach().exp().item())
            # R4 Q2: verify finite before returning; NaN log_T would escape
            # the pre-check if numerical explosion happened inside LBFGS.
            if not math.isfinite(T_star):
                raise ValueError(f"LBFGS produced non-finite T: {T_star}")
            T_unclamped = T_star
            # Session 22 audit fix: Clamp to narrow bounds (paper Eq.35 says
            # Lcal handles calibration; post-hoc should only do fine-tuning).
            T_star = max(bounds[0], min(bounds[1], T_star))
            # Emit warning when the fit lands at/near the boundary — signals
            # that Lcal did not converge and the model is uncalibrated. The
            # `temperature_suspect` flag is returned so callers (validation
            # logging) can surface this condition.
            suspect = (
                T_unclamped <= bounds[0] * 1.05
                or T_unclamped >= bounds[1] * 0.95
                or T_unclamped > 1.5  # moderate drift still worth a note
            )
            if suspect:
                import warnings as _w
                _w.warn(
                    f"Temperature scaling T*={T_unclamped:.3f} (clamped to "
                    f"{T_star:.3f}) — outside [0.95, 1.5] suggests Lcal did "
                    f"not achieve in-training calibration; post-hoc scaling "
                    f"is now masking the gap. Investigate Lcal convergence.",
                    RuntimeWarning, stacklevel=2,
                )
            return {
                "temperature": T_star,
                "temperature_nll": _nll_at(T_star),
                "temperature_suspect": bool(suspect),
                "temperature_unclamped": T_unclamped,
            }
        except Exception as exc:
            # Graceful fallback to grid on any numerical issue.
            import warnings as _w
            _w.warn(f"LBFGS temperature fit failed ({exc}); falling back to grid.",
                    RuntimeWarning, stacklevel=2)

    # Legacy grid path (still available via use_lbfgs=False or on fallback).
    if grid is None:
        grid = [0.60, 0.75, 0.85, 0.95, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
    best_temp = 1.0
    best_nll = float("inf")
    for temp in grid:
        nll = _nll_at(temp)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp)
    best_temp = max(bounds[0], min(bounds[1], best_temp))
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
    """Calibrated threshold selection for Eq.26 alert gate (PDF Section 3.7.1).

    ── Problem ──────────────────────────────────────────────────────────────────
    Paper Eq.26 gates alerts on two thresholds:
        A^(h) = 1  iff  ĉ ≥ τ_h  AND  max(p̂) ≥ γ_h
    but leaves τ_h, γ_h unspecified. A single percentile pass (e.g. τ=p85(conf))
    is brittle: if ĉ and max(p̂) are weakly correlated, the induced alert rate
    can be far below the paper's target coverage κ=0.35, yielding high Sharpe
    from selective abstention (tiny MDD, few trades) that would NOT hold at the
    intended operational coverage.

    ── Algorithm ────────────────────────────────────────────────────────────────
    We run a coverage-constrained grid search on validation only.

    (1) Seed grid around user percentiles τ_seed = p65(conf), γ_seed = p55(maxp),
        plus safety anchors at p{20, 35, 50} to guarantee at least one lenient
        pair survives when the model is highly confident overall.

    (2) For each (τ, γ) in the grid, evaluate_alert_policy(·) produces
        {coverage, precision, hit_rate, sharpe, sortino, calmar}. We then score:

            objective(τ, γ) = S(τ, γ) + U(τ, γ) - P(τ, γ) - N(τ, γ)

        where
            S — model selection score (macro_F1, MCC, ECE, alert_sharpe blend)
            U — utility bonus (paper Table 4 metrics, clipped):
                  U = 0.04·Precision + 0.03·HitRate
                      + 0.20·clip(Sharpe,  -0.5, 0.5)
                      + 0.05·clip(Sortino, -0.5, 0.8)
                      + 0.04·clip(Calmar,  -0.5, 1.0)
            P — coverage penalty enforcing κ=0.35:
                  P = 0.70 · under_κ  +  0.20 · over_κ_5pct  +  0.60 · under_hard
                    under_κ         = max(0, κ - coverage)
                    over_κ_5pct     = max(0, coverage - (κ + 0.05))  [wider tolerance above]
                    under_hard      = max(0, MIN_ALERT_COV - coverage)
            N — asymmetric negative-Sharpe penalty:  N = 0.18 · max(0, -Sharpe)

    (3) Hard floor:  MIN_ALERT_COV = max(0.05, 0.20·κ) rejects any policy that
        alerts < 5 % of samples. Without this floor, Sharpe is inflated by
        extreme abstention and drowns out the under-coverage penalty.

    (4) Compliant-first tie-break: if at least one grid point satisfies
        coverage ≥ MIN_ALERT_COV we pick the best objective among the compliant
        pool; otherwise we fall back to the highest-coverage policy so the
        system degrades gracefully on unusually sparse predictions.

    ── Penalty weight rationale (0.70 / 0.20 / 0.60) ───────────────────────────
    At coverage = 12 % (κ = 0.35):
        under_κ = 0.23  →  P_under = 0.70 · 0.23 = 0.161
    A 12 % alert policy typically wins ~0.20 Sharpe over a 35 % policy
    (fewer, cleaner trades). Mapping through U that equals:
        ΔU ≈ 0.20 · 0.20 = 0.040
    Setting weight 0.70 on under_coverage guarantees P_under ≫ ΔU, so the
    objective cannot be won by degenerate low-coverage points. The weights
    on over_coverage (0.20) and under_hard (0.60) were chosen so policies
    over-covering by 5 % are still preferred to policies below the hard
    floor, while symmetric ±5 % tolerance around κ incurs zero penalty.

    ── Conformal mode ──────────────────────────────────────────────────────────
    When ``policy_method="conformal"`` we replace the grid with conformal
    quantiles calibrated on the correctly-classified subset at level α = 1-κ.
    The same objective is applied. The conformal path is offered for the
    thesis's "conformal prediction" future-work claim (Section 5.4.5) but is
    not the default; the grid search is the reference path.

    ── Returned dict ───────────────────────────────────────────────────────────
    In addition to all fields from evaluate_alert_policy(·), this returns:
        model_score, policy_objective, coverage_penalty, utility_bonus,
        negative_sharpe_penalty, policy_source, tau_seed, gamma_seed.
    """
    conf = np.asarray(confidence, dtype=np.float32)
    probs = np.asarray(dir_probs, dtype=np.float32)
    max_prob = probs.max(axis=-1)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)

    # Session 26: conformal path removed. DEVIATIONS.md §M3 documented that the
    # conformal branch fit τ/γ only on correctly-classified samples — biasing
    # thresholds toward the easy subset and inflating alert precision / Sharpe
    # by 5-15 %. We refused to keep a known-biased code path callable by a
    # config flag. If a future reviewer needs proper conformal calibration,
    # implement one that does NOT condition on correctness (e.g. split-conformal
    # on a held-out calibration slice of validation).
    if policy_method != "grid":
        raise ValueError(
            f"Unsupported policy_method={policy_method!r}. "
            "Only 'grid' is supported (conformal path removed in Session 26 — "
            "see DEVIATIONS.md §M3 for why)."
        )

    tau_seed = float(np.percentile(conf, tau_percentile))
    gamma_seed = float(np.percentile(max_prob, gamma_percentile))

    # Session 22 audit fix (Fix 3): policy grid reduced 9×9=81 → 4×4=16
    # candidates (closer to 15 after dedup). The original 81-point grid was
    # effectively 81 hyperparameter configurations fit on the validation set,
    # contributing ~5-10% inflation to reported alert_sharpe / alert_precision
    # through val-fit. The narrower grid keeps the seed (paper-specified
    # percentile) plus one low, one mid, one high anchor — enough to cover
    # the threshold landscape without over-fitting validation.
    # Anchors kept:
    #   - tau/gamma seeds: the paper's coverage-calibrated percentile
    #   - 35th pct:  lenient lower anchor for low-confidence regimes
    #   - 50th pct:  median safety anchor
    #   - 80th pct:  strict upper anchor
    # This shrinks the effective search space by 5×, reducing val-overfit risk
    # while still finding a good (τ, γ) for the target coverage.
    tau_percentiles   = [35, 50, tau_percentile, 80]
    gamma_percentiles = [35, 50, gamma_percentile, 80]
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
    # Sprint 2.1-B — trade-coverage floor. The alert-coverage floor alone can be
    # satisfied by NEUTRAL alerts that never enter a position; in fold-1 ep 19
    # we observed alert_coverage = 0.45 with trade_coverage = 0.017 (a 26×
    # gap). Selection still picked it because Sharpe was suppressed and the
    # alert_coverage floor was nominally met. Adding a parallel trade_coverage
    # floor forces the policy to actually take positions before it can win the
    # compliant bracket. The ratio is loose (0.20 × κ = 7 % of validation when
    # κ = 0.35) so we don't over-constrain early training when the head is
    # still learning to predict UP / DOWN at all; once the head is genuinely
    # signalling, this is the easier floor to clear, not the harder one.
    MIN_TRADE_COV = max(0.03, target_coverage * 0.20)   # ≥ 3% and ≥ 20% of target

    # Collect ALL candidate policies, then separate into compliant vs fallback.
    # Previous approach had a bug: the `best is None` check let the very first
    # evaluated policy bypass the floor when it was the only one evaluated before
    # best was set — meaning if ALL grid points have Cov < MIN_ALERT_COV, the
    # first-evaluated (not highest-coverage) policy would win.
    compliant: list[tuple[float, Dict[str, float]]] = []   # (objective, policy_dict)
    fallback:  list[tuple[float, int, float, float, float, Dict[str, float]]] = []
    alert_cov_fallback: list[tuple[float, Dict[str, float]]] = []

    def _compute_policy_objective(policy: Dict[str, float]) -> tuple[Dict[str, float], float]:
        # Sprint 1 FIX 3 — when Sharpe was suppressed (too few position alerts
        # for reliable statistics) the value is 0.0 by convention, not because
        # the policy actually achieved zero edge. Treat that as missing data
        # rather than as a real Sharpe = 0: drop the Sharpe-derived utility,
        # keep coverage / hit-rate / precision contributions. Combined with the
        # compliant filter below this prevents a low-coverage policy with
        # suppressed Sharpe from beating a real lower-Sharpe policy on
        # objective just because its noisy Sharpe was clipped to zero.
        sharpe_suppressed = bool(policy.get("alert_sharpe_suppressed", False))
        sharpe_for_score = 0.0 if sharpe_suppressed else policy["alert_sharpe"]
        # Sprint 12 — thread position_pnl through so score includes economic
        # gate when SELECTION_WEIGHT_PNL > 0. Legacy bit-stable when weight=0.
        _policy_pnl = policy.get("position_pnl", policy.get("pnl", 0.0))
        score = compute_model_selection_score(
            macro_f1,
            mcc,
            ece,
            sharpe_for_score,
            alert_coverage=policy["alert_coverage"],
            position_pnl=_policy_pnl,
        )
        under_coverage = max(0.0, target_coverage - policy["alert_coverage"])
        over_coverage = max(0.0, policy["alert_coverage"] - (target_coverage + 0.05))
        under_coverage_hard = max(0.0, MIN_ALERT_COV - policy["alert_coverage"])
        # Sprint 2.1-B — trade-coverage shortfall acts as an additional penalty
        # on policies that hit alert_coverage but skip the actual position
        # taking. Quadratic in the gap so a near-zero trade_coverage hurts more
        # than a marginal shortfall, encouraging the search to drift toward
        # τ/γ pairs where directional preds dominate the alert region.
        trade_cov_now = float(policy.get("trade_coverage", 0.0))
        under_trade_cov = max(0.0, MIN_TRADE_COV - trade_cov_now)
        coverage_penalty = (
            0.70 * under_coverage +
            0.20 * over_coverage +
            0.60 * under_coverage_hard +
            0.80 * under_trade_cov
        )
        if sharpe_suppressed:
            utility_bonus = (
                0.04 * policy["alert_precision"] +
                0.03 * policy["hit_rate"]
            )
            negative_sharpe_penalty = 0.0
            # Sprint 3.3-C — strengthened from 0.10 to 0.20 for parity with
            # train_safe_alert.py model_score. Suppressed-Sharpe checkpoints
            # need a meaningful penalty to lose against measurable-Sharpe
            # alternatives; old 0.10 was below the F1+ECE noise budget.
            suppression_penalty = 0.20
        else:
            utility_bonus = (
                0.04 * policy["alert_precision"] +
                0.03 * policy["hit_rate"] +
                0.20 * float(np.clip(policy["alert_sharpe"], -0.50, 0.50)) +
                0.05 * float(np.clip(policy["alert_sortino"], -0.50, 0.80)) +
                0.04 * float(np.clip(policy["alert_calmar"], -0.50, 1.00))
            )
            negative_sharpe_penalty = 0.18 * max(0.0, -float(policy["alert_sharpe"]))
            suppression_penalty = 0.0
        # Sprint 3.3-C — action-aware policy objective. Mirror train_safe_alert.py
        # model_score so search_alert_policy returns a policy whose [BEST] pick
        # is consistent with what the trainer's checkpoint selection would
        # promote. action_bonus rewards real trading (TradeCov ≥ 0.05) plus
        # PosHit tiers; neutsh_penalty_policy discourages NEUTRAL-spam alerts
        # that inflate alert_coverage without producing positions.
        # Sprint 3.3-D — quality-aware tier gating + action_quality_penalty_policy
        # mirror of the trainer-side change. Two adjustments vs 3.3-C:
        #   1. PosHit tier bonuses only fire when the alert region's PnL or
        #      Sharpe is non-negative (gate: trade-and-lose can't earn the
        #      tier reward, only the baseline +0.05 for trading at all).
        #   2. action_quality_penalty_policy adds a small negative-Sharpe
        #      adjustment specifically for action-heavy losing policies,
        #      capped at 0.05 (vs trainer's 0.10) because the policy
        #      objective already includes negative_sharpe_penalty (0.18 ×
        #      |Sharpe|). Stacking both at 0.10 each would double-punish.
        _trade_cov_p = float(policy.get("trade_coverage", 0.0))
        _pos_pnl_p = float(policy.get("position_pnl", 0.0))
        _alert_sharpe_p = float(policy.get("alert_sharpe", 0.0))
        _quality_ok_p = (_pos_pnl_p >= 0.0) or (_alert_sharpe_p >= 0.0)
        action_bonus = 0.0
        if _trade_cov_p >= 0.05:
            action_bonus += 0.05  # baseline: policy actually takes positions
            _pos_hit_p = float(policy.get("position_hit_rate", policy.get("hit_rate", 0.0)))
            if _quality_ok_p and _pos_hit_p >= 0.40:
                action_bonus += 0.05
            if _quality_ok_p and _pos_hit_p >= 0.50:
                action_bonus += 0.05
        action_quality_penalty_policy = 0.0
        if _trade_cov_p >= 0.05 and _alert_sharpe_p < 0.0:
            # Cap 0.05 (half of trainer's 0.10) because negative_sharpe_penalty
            # already accounts for some of the same signal in this layer.
            action_quality_penalty_policy = min(0.05, abs(_alert_sharpe_p) * 0.15)
        neutsh_penalty_policy = 0.0
        _neutsh_p = float(policy.get(
            "neutral_share_of_alerts", policy.get("neutral_alert_rate", 0.0)
        ))
        if _neutsh_p > 0.70:
            neutsh_penalty_policy = 0.15 * min(1.0, (_neutsh_p - 0.70) / 0.30)
        # Sprint 12 diagnostic — cost-volume risk. Position canary failure showed
        # that high-coverage policies suffer cost erosion: TradeCov=0.62 with
        # 0.001/trade cost = $0.62 cost overhead per epoch unit. Sharpe is
        # already net-of-cost via mini_backtest. Keep the would-be penalty as
        # telemetry only after Sprint 12 rollback; do not hide this inside
        # policy search.
        cost_volume_penalty_diag = 0.0
        if _trade_cov_p > 0.30:
            # Quadratic ramp when TradeCov > 0.30: 0 (at 0.30) to 0.10 (at 0.60).
            _excess_cov = min(1.0, (_trade_cov_p - 0.30) / 0.30)
            cost_volume_penalty_diag = 0.10 * (_excess_cov ** 2)
        # Sprint 12 diagnostic — directional imbalance risk. Policies that fire
        # all UP or all DOWN alerts (e.g. AlertPredDist D/N/U=0.00/0.00/1.00)
        # capture only one side of the market and produce regime-fragile PnL.
        # Compute the would-be penalty from alert distribution for auditing.
        _alert_d = float(policy.get("alert_pred_dist_down", policy.get("alert_dist_down", 0.0)))
        _alert_u = float(policy.get("alert_pred_dist_up", policy.get("alert_dist_up", 0.0)))
        directional_imbalance = abs(_alert_u - _alert_d)
        imbalance_penalty_diag = 0.0
        if _trade_cov_p >= 0.05 and directional_imbalance > 0.70:
            # Diagnostic value: 0 at 0.70, ramps to 0.10 at 1.00.
            imbalance_penalty_diag = 0.10 * min(1.0, (directional_imbalance - 0.70) / 0.30)
        # Sprint 12 rollback (2026-05-04): keep these diagnostics visible, but
        # do not apply them in policy search. The active penalty layer selected
        # high-volume/one-sided checkpoints in fold 2 and degraded test PnL.
        # Any future Sprint 12b should enforce this as a deploy hard gate or
        # explicit model-selection term, not a hidden policy-objective tweak.
        cost_volume_penalty = 0.0
        imbalance_penalty = 0.0
        objective = (
            score
            + utility_bonus
            - coverage_penalty
            - negative_sharpe_penalty
            - suppression_penalty
            + action_bonus
            - neutsh_penalty_policy
            - action_quality_penalty_policy
            - cost_volume_penalty
            - imbalance_penalty
        )
        enriched = {
            **policy,
            "model_score": float(score),
            "policy_objective": float(objective),
            "coverage_penalty": float(coverage_penalty),
            "utility_bonus": float(utility_bonus),
            "negative_sharpe_penalty": float(negative_sharpe_penalty),
            "suppression_penalty": float(suppression_penalty),
            # Sprint 2.1-B — surface the trade-coverage shortfall so the
            # trainer can log how much the policy lost to under-trading.
            "under_trade_cov": float(under_trade_cov),
            # Sprint 3.3-C — surface action-aware adjustments for traceability.
            "action_bonus":           float(action_bonus),
            "neutsh_penalty_policy":  float(neutsh_penalty_policy),
            # Sprint 3.3-D — quality-aware penalty for action-heavy losing
            # policies. Capped at 0.05 to avoid stacking with the existing
            # negative_sharpe_penalty.
            "action_quality_penalty_policy": float(action_quality_penalty_policy),
            # Sprint 12 rollback: applied penalties are zero. The *_diag fields
            # expose what the penalty would have been, useful for Sprint 12b
            # hard-gate design without altering canonical raw policy search.
            "cost_volume_penalty":  float(cost_volume_penalty),
            "imbalance_penalty":    float(imbalance_penalty),
            "cost_volume_penalty_diag":  float(cost_volume_penalty_diag),
            "imbalance_penalty_diag":    float(imbalance_penalty_diag),
            "directional_imbalance": float(directional_imbalance),
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
            # Sprint 1 FIX 3 + Sprint 2.1-B — a policy enters the compliant
            # bracket only if (a) Sharpe is measurable (not suppressed by
            # too few position alerts), (b) alert_coverage clears the legacy
            # floor, AND (c) trade_coverage clears its own floor. The third
            # condition is the new one: it prevents NEUTRAL-heavy policies
            # from winning just because they alert on plenty of samples
            # without ever entering a position.
            sharpe_suppressed = bool(policy.get("alert_sharpe_suppressed", False))
            trade_cov_ok = float(policy.get("trade_coverage", 0.0)) >= MIN_TRADE_COV
            if (
                not sharpe_suppressed
                and policy["alert_coverage"] >= MIN_ALERT_COV
                and trade_cov_ok
            ):
                compliant.append((objective, enriched))
            else:
                trade_cov = float(enriched.get("trade_coverage", 0.0))
                n_pos_alerts = int(enriched.get("n_position_alerts", 0))
                pos_pnl = float(enriched.get("position_pnl", 0.0))
                alert_cov = float(enriched["alert_coverage"])
                fallback.append((trade_cov, n_pos_alerts, float(objective), pos_pnl, alert_cov, enriched))
                alert_cov_fallback.append((alert_cov, enriched))

    if compliant:
        # Pick the policy with the highest objective among coverage-compliant candidates.
        _, best = max(compliant, key=lambda x: x[0])
    else:
        # No compliant policy found — pick the highest-coverage fallback to minimise
        # degeneracy. This can happen when the model is very confident and even the
        # most lenient τ/γ pair alerts fewer than MIN_ALERT_COV samples.
        best_trade = max(fallback, key=lambda x: (x[0], x[1], x[2], x[3])) if fallback else None
        if best_trade is not None and best_trade[0] > 0.0:
            trade_cov, n_pos_alerts, objective, pos_pnl, _alert_cov, best = best_trade
            warnings.warn(
                f"search_alert_policy: no grid point met both MIN_ALERT_COV="
                f"{MIN_ALERT_COV:.3f} and MIN_TRADE_COV={MIN_TRADE_COV:.3f} with a "
                f"measurable Sharpe; falling back to highest-trade-coverage policy "
                f"(trade_cov={trade_cov:.3f}, pos_alerts={n_pos_alerts}, "
                f"objective={objective:.4f}, pos_pnl={pos_pnl:.4f}, "
                f"alert_cov={best['alert_coverage']:.3f}). "
                f"Consider reducing tau_percentile/gamma_percentile or check whether "
                f"the head is collapsing to NEUTRAL (see [PredDist] / [ConfByClass] logs).",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            _, best = max(alert_cov_fallback, key=lambda x: x[0])
            warnings.warn(
                f"search_alert_policy: no grid point met both MIN_ALERT_COV="
                f"{MIN_ALERT_COV:.3f} and MIN_TRADE_COV={MIN_TRADE_COV:.3f} with a "
                f"measurable Sharpe, and no fallback candidate had trade_coverage > 0; "
                f"falling back to alert_cov last resort "
                f"(alert_cov={best['alert_coverage']:.3f}, "
                f"trade_cov={float(best.get('trade_coverage', 0.0)):.3f}). "
                f"Consider reducing tau_percentile/gamma_percentile or check whether "
                f"the head is collapsing to NEUTRAL (see [PredDist] / [ConfByClass] logs).",
                RuntimeWarning,
                stacklevel=2,
            )

    assert best is not None
    return best


# ─────────────────────────────────────────────────────────────
# 6. AGGREGATED SCORE FOR MODEL SELECTION
# ─────────────────────────────────────────────────────────────

# R3 #I1: Model-score weights exposed as module-level constants so YAML can
# override them via ``_apply_config_overrides`` in train_safe_alert.py. Previous
# hardcoded weights (0.40/0.35/0.15/0.10) match paper Section 4.5.1 but prevent
# users from biasing selection toward F1 (academic) vs Sharpe (trading). In Fold
# 1 log, best F1=0.472 was at ep 12 but ep 15 (F1=0.451, Sharpe=0.426) won the
# best_overall slot because Sharpe weight 0.35 rewarded the higher Sharpe.
SELECTION_WEIGHT_F1     = 0.40
SELECTION_WEIGHT_SHARPE = 0.35
SELECTION_WEIGHT_ECE    = 0.15
SELECTION_WEIGHT_MCC    = 0.10
# Sprint 12 — economic gate. Default 0.0 preserves legacy 4-metric formula
# bit-stable. Set > 0 (e.g. 0.10-0.20) to add direct PnL contribution to
# checkpoint selection. Motivated by walk-forward fold 2 ép 11 (PosPnL=+0.246,
# Score=0.214) losing to ép 12 (PosPnL=+0.012, Score=0.290) — score
# formula undervaluing direct economic outcome.
SELECTION_WEIGHT_PNL    = 0.0


def set_selection_weights(
    f1: float,
    sharpe: float,
    ece: float,
    mcc: float,
    pnl: float = 0.0,
) -> None:
    """R3 #I1: configure model_selection_score weights at module level.

    Called once from train_safe_alert._apply_config_overrides() after YAML load.
    Weights do NOT need to sum to 1 — the score is not a probability.

    Sprint 12 — added optional ``pnl`` weight. Default 0.0 preserves legacy
    4-metric formula bit-stable; set > 0 to enable direct PnL contribution.
    """
    global SELECTION_WEIGHT_F1, SELECTION_WEIGHT_SHARPE
    global SELECTION_WEIGHT_ECE, SELECTION_WEIGHT_MCC
    global SELECTION_WEIGHT_PNL
    SELECTION_WEIGHT_F1     = float(f1)
    SELECTION_WEIGHT_SHARPE = float(sharpe)
    SELECTION_WEIGHT_ECE    = float(ece)
    SELECTION_WEIGHT_MCC    = float(mcc)
    SELECTION_WEIGHT_PNL    = float(pnl)


def compute_model_selection_score(
    macro_f1: float,
    mcc: float,
    ece: float,
    alert_sharpe: float,
    alert_coverage: float = 0.35,
    position_pnl: Optional[float] = None,
    pnl_scale: float = 0.5,
) -> float:
    """
    Multi-metric score for early stopping (PDF Section 4.5.1 + R3 #I1).

    Score = w_f1·MacroF1 + w_sharpe·AlertSharpe_norm - w_ece·(1-ECE_norm) + w_mcc·MCC_norm
          + w_pnl·PnL_norm - coverage_floor_penalty(alert_coverage)

    Default weights: w_f1=0.40, w_sharpe=0.35, w_ece=0.15, w_mcc=0.10
    (paper Section 4.5.1). Override via YAML ``selection_weights:`` or at
    runtime via ``set_selection_weights()``.

    Sprint 12 — economic gate addition:
        ``position_pnl`` (optional) adds a PnL contribution normalized by
        ``pnl_scale`` (default 0.5 cumulative PnL units). When provided
        and finite, contributes ``SELECTION_WEIGHT_PNL × clip(pnl/pnl_scale, -1, 1)``.
        When None (default), the legacy 4-metric formula is preserved
        bit-stable for backward compat. Motivated by walk-forward fold 2
        ép 12 winning over ép 11 despite ép 11 having 20× better PnL —
        score formula was undervaluing direct economic outcome.

    Session 26 — degenerate-alert guard:
        When alert_coverage < 5 %, mini_backtest returns alert_sharpe = 0 (by
        design — Sharpe on < 20 trades is noise). Without a coverage-floor
        penalty, a model that fires ZERO alerts could still win early-stopping
        on F1 + MCC alone (~0.13 baseline). This checkpoint would then go to
        production and alert on nothing. We therefore subtract a hard penalty
        proportional to how far below 5 % coverage drops, strong enough to
        push a degenerate model below any reasonably-trained model.

    Higher score = better model.
    """
    # Guard against NaN/Inf inputs — any non-finite metric (e.g. from an
    # empty validation split or a degenerate model) must not propagate to
    # the score and corrupt early-stopping / checkpoint-selection logic.
    def _safe(v: float, default: float) -> float:
        return float(v) if math.isfinite(float(v)) else default

    macro_f1       = _safe(macro_f1,       0.0)
    mcc            = _safe(mcc,            0.0)
    ece            = _safe(ece,            1.0)   # worst-case ECE = 1.0
    alert_sharpe   = _safe(alert_sharpe,   0.0)
    alert_coverage = _safe(alert_coverage, 0.0)

    # Normalize inputs to [0, 1] range for fair weighting
    macro_f1_norm = np.clip(macro_f1, 0, 1)  # [0, 1]
    mcc_norm = np.clip((mcc + 1) / 2, 0, 1)  # [-1, 1] → [0, 1]
    ece_norm = np.clip(1 - ece, 0, 1)      # Lower ECE = higher contribution
    # Normalize Sharpe symmetrically so negative utility penalizes score.
    alert_sharpe_norm = np.clip(alert_sharpe / 2.0, -1, 1)

    score = (
        SELECTION_WEIGHT_F1     * macro_f1_norm
        + SELECTION_WEIGHT_SHARPE * alert_sharpe_norm
        - SELECTION_WEIGHT_ECE    * (1 - ece_norm)
        + SELECTION_WEIGHT_MCC    * mcc_norm
    )

    # Sprint 12 — direct PnL contribution. Active only when position_pnl is
    # provided AND SELECTION_WEIGHT_PNL > 0 (default 0.0 keeps legacy formula).
    # Normalized by pnl_scale (default 0.5 cumulative PnL units) and clipped
    # symmetrically. A positive PnL at or above pnl_scale contributes the full
    # weight; deeply
    # negative PnL (below -pnl_scale) penalizes the same magnitude. Combined
    # with the existing alert_sharpe term, this rewards checkpoints that
    # produce real economic edge (not just well-calibrated abstention).
    if position_pnl is not None and SELECTION_WEIGHT_PNL > 0.0:
        pnl = _safe(position_pnl, 0.0)
        pnl_norm = float(np.clip(pnl / max(float(pnl_scale), 1e-6), -1.0, 1.0))
        score += SELECTION_WEIGHT_PNL * pnl_norm

    # Coverage-floor penalty: linear ramp from 0 (at coverage ≥ 5 %) to 0.30
    # (at coverage = 0). A value of 0.30 is larger than the total positive
    # contribution from F1 + MCC (0.40 + 0.10 = 0.50 max), guaranteeing a
    # model with zero alerts cannot outscore a properly-alerting model with
    # similar F1. Floor of 5 % matches mini_backtest's minimum-sample gate.
    COVERAGE_FLOOR = 0.05
    if alert_coverage < COVERAGE_FLOOR:
        gap = (COVERAGE_FLOOR - alert_coverage) / COVERAGE_FLOOR   # ∈ [0, 1]
        score -= 0.30 * float(gap)

    # Session 33 FIX (Fold 1 audit): degenerate-Sharpe penalty.
    # Observed on Fold 1 ep 23: F1=0.321, Sharpe=EXACTLY 0, Cov=0.154 (>5%
    # floor so coverage penalty doesn't fire). model_score = 0.1753 won
    # against ep 18 (F1=0.190, Sharpe=+0.050, score=0.1337) because F1's
    # 0.04 contribution beat Sharpe's 0.009 contribution. The selected
    # checkpoint then produced TEST Sharpe=-0.730 (catastrophic).
    # Sharpe=0 with non-zero coverage signals one of:
    #   (a) std(returns)<1e-6 → policy makes near-identical predictions
    #       on every alert (degenerate one-class output);
    #   (b) policy returned all-zero or perfectly cancelling trades;
    #   (c) hit threshold for "few alerts" guard inside mini_backtest.
    # In every case the policy is unlikely to generalise. Penalty 0.05 is
    # large enough to flip ep 18 vs ep 23 in the observed log (0.1753-0.05
    # = 0.1253 < 0.1337) but small enough to keep slightly-degenerate
    # policies from being aggressively excluded. Only fires when coverage
    # is above the floor (otherwise the existing coverage-floor penalty
    # already disqualifies the model).
    if abs(alert_sharpe) < 1e-6 and alert_coverage >= COVERAGE_FLOOR:
        score -= 0.05

    return float(score)


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
