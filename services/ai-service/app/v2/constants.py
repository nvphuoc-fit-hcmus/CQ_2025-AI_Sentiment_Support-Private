"""Shared hyperparameter and constant definitions for SAFE-Alert v2.

Session 23 P2 #8 fix: previously each file had its own hardcoded constants
(LABEL_TEMP=0.3 vs 0.5, lookback_hours, K_h, etc.) which caused subtle bugs
when values drifted between files. All paper-relevant constants live here
with rationale citations so reviewers can audit in one place.

Values here are DEFAULTS — callers that accept a config (YAML or CLI) can
still override. Never import a value from another pipeline file if it
could live here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dataset windowing (Section 4.1.3 + 4.2.2)
# ─────────────────────────────────────────────────────────────────────────────
# News window W_n: how many hours back from a decision mark t we consider
# articles. Paper Section 4.1.3 says "past 24h" for 1h horizon; shorter
# horizons use tighter windows but 24h is a reasonable default.
LOOKBACK_HOURS_DEFAULT = 24

# Articles per candle (K in paper). Paper Section 3.3.2 specifies per-horizon
# K_h; this is the maximum K stored in the tensor. Dataset pads to this size.
ARTICLES_PER_CANDLE_DEFAULT = 8

# Ingest delay (paper Section 4.1.3 "temporal consistency"):
# published_at + delay < candle_time required for article to be visible.
INGEST_DELAY_MINUTES_DEFAULT = 15

# Market feature window W_m: number of past candles used to extract the
# multi-timeframe market features (Section 4.2.2). 250 hours ≈ 10 days
# of 1h bars, sufficient for 4h-indicator windows with headroom.
MARKET_LOOKBACK_CANDLES = 250

# Paper ε_h per horizon — neutral band for Eq.51 direction labels.
# Must match paper Table 2 exactly.
EPSILON_H_BY_HORIZON = {
    "15m": 0.001,
    "1h":  0.002,
    "4h":  0.005,
    "24h": 0.010,
}


def get_epsilon_h(horizon: str, override=None) -> float:
    """Resolve ε_h for a given horizon, with optional YAML override.

    Sprint 4 Phase 8.5 — single source of truth for ε_h. The label
    generator (safe_alert_dataset._get_direction_label) and the
    Lret_sign nonzero mask (MultiObjectiveLoss) MUST agree on the
    same value or supervision becomes asymmetric on samples in
    [override, paper] band. Both sites call this helper instead of
    keeping their own hardcoded copy.

    Args:
        horizon: one of '15m', '1h', '4h', '24h'.
        override: float or None. When float, replaces the paper value
            after range validation in [0.0001, 0.05] (paper-plausible
            crypto bounds). When None, paper-canonical is returned.

    Raises:
        KeyError: if horizon not in EPSILON_H_BY_HORIZON.
        ValueError: if override is out of [0.0001, 0.05].
    """
    paper = EPSILON_H_BY_HORIZON[horizon]  # KeyError on unknown horizon
    if override is None:
        return float(paper)
    eps = float(override)
    if not (1e-4 <= eps <= 5e-2):
        raise ValueError(
            f"epsilon_h_override must be in [0.0001, 0.05], got {eps}"
        )
    return eps

# K_h per horizon (paper Section 3.3.2 Top-K selection).
K_H_BY_HORIZON = {
    "15m": 3,
    "1h":  4,
    "4h":  5,
    "24h": 8,
}

# ─────────────────────────────────────────────────────────────────────────────
# Factor ontology (Section 3.4 + 4.5.3)
# ─────────────────────────────────────────────────────────────────────────────
# Softmax temperature for keyword-fallback factor labels. Must match the
# precompute script to avoid distribution shift when Qwen-precomputed
# labels are missing for a small tail of articles. Session 23 P0 #1.
FACTOR_LABEL_TEMP = 0.5

# Min confidence for factor pseudo-labels (articles with lower max-prob are
# treated as noise and given uniform distribution). Paper Section 4.1.5.
MIN_FACTOR_CONFIDENCE = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# Article metadata construction
# ─────────────────────────────────────────────────────────────────────────────
# Content length normalization ceiling. Articles longer than this are
# treated as "long" (length_norm → 1.0). Heuristic from crypto news
# median length; not paper-specified.
MAX_CONTENT_LENGTH_NORM = 500

# Fallback source credibility for sources not in the learned frequency map.
SOURCE_CREDIBILITY_UNKNOWN = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Loss / training (Section 3.9 + 4.5.1)
# ─────────────────────────────────────────────────────────────────────────────
# Confidence clamp in Lrisk: numerical floor prevents denom collapse when
# confidence head is near-random early in training. Not paper-specified.
LRISK_CONFIDENCE_MIN = 0.01

# Lsel clamp: prevents numerical overflow in the cardinality term when
# sigmoid gates are saturated on a long article list.
LSEL_CLAMP = 50.0

# Entropy anchor handoff point (Session 21 Fix B): effective anchor weight
# decays linearly with λ5 and reaches 0 at this threshold. 0.10 matches the
# paper Table 4 Stage 3 default for λ5 (Lcal).
ANCHOR_HANDOFF_LAMBDA5 = 0.10

# Logit L2 default weight (Session 22 Fix 8). 1e-4 is enough to curb
# runaway logit magnitudes without competing with CE convergence.
LOGIT_L2_DEFAULT_WEIGHT = 1e-4

# Feature zero-padding strict threshold (Session 23 P0 #2): fraction of
# market-feature dims that may be zero-padded in non-precomputed mode
# before strict mode raises RuntimeError.
STRICT_FEATURE_PAD_THRESHOLD = 0.15
