"""
SAFE-Alert Component 4 — Confidence-Aware Alerting with Abstention (Eq.26).

OFFICIAL IMPLEMENTATION: decide_alert_multimodal_eq26

  A_h = 1  iff  signal_h != HOLD  AND  confidence_h >= tau_h  AND  max_prob_h >= gamma_h

Final policy:
  - no active horizon       → abstain
  - conflicting active dirs → abstain
  - one active              → medium alert
  - both active + agree     → high alert
"""
from __future__ import annotations


def decide_alert_multimodal_eq26(
    *,
    signal_1h: str,
    confidence_1h: float,
    max_prob_1h: float,
    tau_1h: float,
    gamma_1h: float,
    signal_4h: str,
    confidence_4h: float,
    max_prob_4h: float,
    tau_4h: float,
    gamma_4h: float,
) -> dict:
    """
    [OFFICIAL] SAFE-Alert Eq.26 multi-horizon alert decision.

    THESIS REQUIREMENT: All alert decisions must use this function (Eq.26).

    Implements equation 26 from thesis:
      A_h = 1 iff signal_h != HOLD AND confidence_h >= tau_h AND max_prob_h >= gamma_h

    Final policy:
      - no active horizon      -> abstain
      - conflicting active dirs-> abstain
      - one active             -> medium alert
      - both active + agree    -> high alert

    Args:
        signal_1h, confidence_1h: 1h horizon
        max_prob_1h: Max probability from 1h prediction
        tau_1h, gamma_1h: Thresholds for 1h (confidence and max_prob)
        signal_4h, confidence_4h: 4h horizon
        max_prob_4h: Max probability from 4h prediction
        tau_4h, gamma_4h: Thresholds for 4h
    """
    act_1h = (signal_1h != "HOLD") and (confidence_1h >= tau_1h) and (max_prob_1h >= gamma_1h)
    act_4h = (signal_4h != "HOLD") and (confidence_4h >= tau_4h) and (max_prob_4h >= gamma_4h)

    if not act_1h and not act_4h:
        return {
            "alert": False,
            "level": "abstain",
            "reason": "No horizon passes Eq.26 thresholds",
            "signal_60m": signal_1h,
            "signal_240m": signal_4h,
            "tau_1h": round(float(tau_1h), 4),
            "gamma_1h": round(float(gamma_1h), 4),
            "tau_4h": round(float(tau_4h), 4),
            "gamma_4h": round(float(gamma_4h), 4),
        }

    if act_1h and act_4h and signal_1h != signal_4h:
        return {
            "alert": False,
            "level": "abstain",
            "reason": f"Conflicting active horizons: 1h={signal_1h}, 4h={signal_4h}",
            "signal_60m": signal_1h,
            "signal_240m": signal_4h,
            "tau_1h": round(float(tau_1h), 4),
            "gamma_1h": round(float(gamma_1h), 4),
            "tau_4h": round(float(tau_4h), 4),
            "gamma_4h": round(float(gamma_4h), 4),
        }

    consensus = signal_1h if act_1h else signal_4h
    level = "high" if (act_1h and act_4h) else "medium"

    return {
        "alert": True,
        "level": level,
        "reason": (
            f"Eq.26 active horizons: 1h={act_1h} (conf={confidence_1h:.3f}, p={max_prob_1h:.3f}), "
            f"4h={act_4h} (conf={confidence_4h:.3f}, p={max_prob_4h:.3f})"
        ),
        "signal": consensus,
        "signal_60m": signal_1h,
        "signal_240m": signal_4h,
        "tau_1h": round(float(tau_1h), 4),
        "gamma_1h": round(float(gamma_1h), 4),
        "tau_4h": round(float(tau_4h), 4),
        "gamma_4h": round(float(gamma_4h), 4),
    }
