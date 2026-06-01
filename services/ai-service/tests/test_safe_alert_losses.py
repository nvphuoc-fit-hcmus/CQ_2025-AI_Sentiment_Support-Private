"""Unit tests for SAFE-Alert MultiObjectiveLoss (PDF Eq.30-37).

Each loss component has its own test verifying:
  (a) the expected value in a known-answer scenario (sanity boundary),
  (b) gradient exists w.r.t. inputs that should drive it,
  (c) corner cases (NaN/Inf inputs, zero articles, class flip).

Run:
    pytest services/ai-service/tests/test_safe_alert_losses.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

# ── Import path setup (matches train_safe_alert.py layout) ──────────
_THIS = Path(__file__).resolve()
_AI = _THIS.parent.parent  # services/ai-service/
_V2 = _AI / "app" / "v2"
for p in (str(_AI), str(_V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipelines.safe_alert_training_utils import MultiObjectiveLoss  # noqa: E402
from models.safe_alert_net import FACTOR_CLASSES  # noqa: E402


# ── Shared fixtures ─────────────────────────────────────────────────

@pytest.fixture
def loss_fn():
    """Default-config loss, learned_lambdas=False so lambdas stay constant."""
    return MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, eta=0.18,
        faith_margin=0.12,
        coverage_target=0.35, mu=0.02,
        learned_lambdas=False,
    )


def _make_batch(B: int = 4, K: int = 8, C: int = FACTOR_CLASSES, correct: bool = True):
    """Build a minimal (dir_logits, dir_labels, ret, fac_probs, fac_labels, ...) tuple."""
    torch.manual_seed(0)
    dir_labels = torch.tensor([0, 1, 2, 1][:B])
    if correct:
        dir_logits = F.one_hot(dir_labels, num_classes=3).float() * 10.0
    else:
        # Flip labels to guarantee wrong predictions.
        dir_logits = F.one_hot((dir_labels + 1) % 3, num_classes=3).float() * 10.0
    ret_pred = torch.randn(B) * 0.01
    ret_labels = ret_pred.clone() + torch.randn(B) * 0.001
    fac_probs = torch.randn(B, K, C).softmax(dim=-1)
    fac_labels = torch.randn(B, K, C).softmax(dim=-1)
    attn = torch.full((B, K), 1.0 / K)   # uniform — sum = 1
    mask = torch.ones(B, K)
    conf = torch.full((B,), 0.5)
    cal_targets = torch.ones(B, dtype=torch.long) if correct else torch.zeros(B, dtype=torch.long)
    return dict(
        dir_logits=dir_logits, dir_labels=dir_labels,
        ret_pred=ret_pred, ret_labels=ret_labels,
        fac_probs=fac_probs, fac_labels=fac_labels,
        attn_weights=attn, article_mask=mask,
        confidence=conf, calibration_targets=cal_targets,
    )


# ── Ldir (Eq.31) ────────────────────────────────────────────────────

def test_ldir_near_zero_when_perfect(loss_fn):
    b = _make_batch(correct=True)
    out = loss_fn(**b)
    # label_smoothing=0.05 puts a floor on the achievable CE; it is ~0.2,
    # not 0. Verify the loss is small relative to a random-prediction baseline.
    assert 0.0 < out["Ldir"] < 0.5, f"Ldir={out['Ldir']}"


def test_ldir_high_when_wrong(loss_fn):
    bad = _make_batch(correct=False)
    out = loss_fn(**bad)
    assert out["Ldir"] > 1.0, f"Ldir={out['Ldir']} should be high for wrong predictions"


# ── Lret (Eq.32) ────────────────────────────────────────────────────

def test_lret_zero_on_identity(loss_fn):
    b = _make_batch()
    b["ret_labels"] = b["ret_pred"].clone()
    out = loss_fn(**b)
    assert out["Lret"] < 1e-5, f"Lret={out['Lret']} should be ~0 when pred==label"


# ── Lfac (Eq.33) ────────────────────────────────────────────────────

def test_lfac_near_zero_on_identity(loss_fn):
    b = _make_batch()
    # Set labels == probs (as distributions). Soft-CE is then the entropy,
    # which is non-zero but bounded by log(C). Verify the loss is FINITE.
    b["fac_labels"] = b["fac_probs"].clone()
    out = loss_fn(**b)
    assert 0.0 < out["Lfac"] < math.log(FACTOR_CLASSES) + 0.1


def test_lfac_decreases_with_agreement(loss_fn):
    """Aligned distributions should yield lower Lfac than misaligned ones."""
    b_aligned = _make_batch()
    b_aligned["fac_labels"] = b_aligned["fac_probs"].clone()
    b_misaligned = _make_batch()
    # Flip the argmax of labels so they disagree with probs.
    B, K, C = b_misaligned["fac_probs"].shape
    shifted = torch.roll(b_misaligned["fac_probs"], shifts=1, dims=-1)
    b_misaligned["fac_labels"] = shifted
    assert loss_fn(**b_aligned)["Lfac"] < loss_fn(**b_misaligned)["Lfac"]


# ── Lsel (Eq.34) ────────────────────────────────────────────────────

def test_lsel_uses_soft_gates_when_provided(loss_fn):
    """When soft_gates sum exactly to K_h, cardinality term is 0."""
    b = _make_batch(B=2, K=8)
    soft_gates = torch.zeros(2, 8)
    # Put exactly K_h=4 gates at 1.0, rest at 0.
    soft_gates[:, :4] = 1.0
    b["soft_gates"] = soft_gates
    out = loss_fn(**b)
    # Lsel = 0 (cardinality) + η·normalised-entropy(α̃). α̃ uniform → entropy > 0
    # but sign is negative (Σα·log α < 0), so Lsel < 0 is expected.
    assert abs(out["Lsel"]) < 1.0


def test_lsel_penalises_too_many_gates(loss_fn):
    """All gates open → cardinality term = (K - K_h)² = 16."""
    b = _make_batch(B=2, K=8)
    b["soft_gates"] = torch.ones(2, 8)     # 8 gates open, K_h=4
    out_many = loss_fn(**b)
    b["soft_gates"] = torch.zeros(2, 8)
    b["soft_gates"][:, :4] = 1.0            # exactly K_h gates open
    out_ok = loss_fn(**b)
    assert out_many["Lsel"] > out_ok["Lsel"] + 1.0


# ── Lcal (Eq.35 Brier) ──────────────────────────────────────────────

def test_lcal_zero_for_perfect_calibration(loss_fn):
    """ĉ=1 on correct samples, ĉ=0 on wrong → Brier=0."""
    b = _make_batch(correct=True)
    b["confidence"] = torch.ones(4)
    b["calibration_targets"] = torch.ones(4, dtype=torch.long)
    out = loss_fn(**b)
    assert out["Lcal"] < 1e-4


def test_lcal_maximum_for_inverted_calibration(loss_fn):
    """ĉ=0 on correct samples → Brier = 1 per sample."""
    b = _make_batch(correct=True)
    b["confidence"] = torch.zeros(4)
    b["calibration_targets"] = torch.ones(4, dtype=torch.long)
    out = loss_fn(**b)
    assert out["Lcal"] > 0.9


# ── Lfaith (Eq.36, relative gap after P1 #8) ────────────────────────

def test_lfaith_zero_when_full_dominates_masked(loss_fn):
    """p_full=0.9, p_mask=0.1 → relative_gap=8/9 ≈ 0.89 ≫ margin 0.12."""
    b = _make_batch()
    # Force masked logits to produce a very different class distribution.
    masked = torch.zeros_like(b["dir_logits"])
    masked[:, (b["dir_labels"] + 1) % 3] = 5.0   # wrong class dominates
    b["masked_dir_logits"] = masked
    out = loss_fn(**b)
    assert out["Lfaith"] < 0.05


def test_lfaith_hits_margin_when_no_gap(loss_fn):
    """p_full == p_mask → relative_gap=0 → loss = margin (0.12)."""
    b = _make_batch()
    b["masked_dir_logits"] = b["dir_logits"].clone()   # identical forward
    out = loss_fn(**b)
    # margin 0.12 exactly when gap==0
    assert abs(out["Lfaith"] - 0.12) < 1e-3


# ── Lrisk (Eq.37) ───────────────────────────────────────────────────

def test_lrisk_lower_when_confident_samples_are_correct(loss_fn):
    """High confidence on correct samples → numerator small → Lrisk small."""
    b_good = _make_batch(correct=True)
    b_good["confidence"] = torch.full((4,), 0.9)
    b_bad = _make_batch(correct=False)
    b_bad["confidence"] = torch.full((4,), 0.9)
    assert loss_fn(**b_good)["Lrisk"] < loss_fn(**b_bad)["Lrisk"]


def test_lrisk_coverage_penalty_active_when_below_target(loss_fn):
    """mean(ĉ)=0.1 < κ=0.35 → μ·(κ-mean)=0.005 added on top of risk term."""
    b = _make_batch(correct=True)
    b["confidence"] = torch.full((4,), 0.1)
    out = loss_fn(**b)
    # risk term ~0 for perfect preds; penalty alone = 0.02 * (0.35 - 0.1) = 0.005
    assert out["Lrisk"] >= 0.004


# ── Gradient sanity ─────────────────────────────────────────────────

def test_total_loss_has_gradient_through_logits(loss_fn):
    b = _make_batch()
    b["dir_logits"] = b["dir_logits"].detach().clone().requires_grad_(True)
    b["fac_probs"] = b["fac_probs"].detach().clone().requires_grad_(True)
    b["confidence"] = b["confidence"].detach().clone().requires_grad_(True)
    out = loss_fn(**b)
    out["loss"].backward()
    assert b["dir_logits"].grad is not None
    assert torch.isfinite(b["dir_logits"].grad).all()


# ── NaN/Inf robustness ──────────────────────────────────────────────

def test_nan_in_logits_is_masked(loss_fn):
    b = _make_batch()
    b["dir_logits"][0, 0] = float("nan")
    out = loss_fn(**b)
    # Loss should be finite — the invalid row is dropped internally.
    assert torch.isfinite(out["loss"])


# ── Entropy anchor (R3 #E anti-overconfidence) ──────────────────────

def test_entropy_anchor_fires_on_overconfident_preds():
    """Overconfident (one-hot-like) predictions should incur anchor penalty."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
        entropy_anchor_weight=0.1,
        entropy_anchor_target=0.95,
    )
    b = _make_batch(correct=True)
    # One-hot logits have H(p)≈0 — deep below the 0.95·log(3)≈1.044 target
    out_anchored = loss_fn(**b)

    loss_fn_no_anchor = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
        entropy_anchor_weight=0.0,
    )
    out_plain = loss_fn_no_anchor(**_make_batch(correct=True))

    # Anchor must raise Ldir when predictions are overconfident
    assert out_anchored["Ldir"] > out_plain["Ldir"]


def test_entropy_anchor_silent_on_uniform_preds():
    """Uniform predictions already satisfy the anchor target → no penalty."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
        entropy_anchor_weight=0.1,
        entropy_anchor_target=0.95,
    )
    b = _make_batch(correct=True)
    b["dir_logits"] = torch.zeros_like(b["dir_logits"])  # uniform → H = log(3)

    loss_fn_no_anchor = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
        entropy_anchor_weight=0.0,
    )
    b2 = _make_batch(correct=True)
    b2["dir_logits"] = torch.zeros_like(b2["dir_logits"])

    # With entropy at maximum, anchor deficit = 0 → Ldir identical
    out_anchored = loss_fn(**b)
    out_plain = loss_fn_no_anchor(**b2)
    assert out_anchored["Ldir"] == pytest.approx(out_plain["Ldir"], rel=1e-4)


# ── Lfac log(p) correctness (bug fix: log_softmax on softmax output) ───

def test_lfac_near_zero_for_perfect_prediction():
    """When p_fac matches one-hot labels, Lfac should be near 0 (not stuck at ~1.5)."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=1.0, lambda4=0.0,
        lambda5=0.0, lambda6=0.0, lambda7=0.0,
        K_h=4, learned_lambdas=False,
    )
    # Perfect prediction: fac_probs concentrated on argmax of labels
    b = _make_batch(B=4, K=8)
    # One-hot labels per article (pick factor 0 for all)
    fac_labels = torch.zeros(4, 8, FACTOR_CLASSES)
    fac_labels[..., 0] = 1.0
    # Model predicts the correct factor with p=0.99 (near-perfect)
    fac_probs = torch.full((4, 8, FACTOR_CLASSES), 0.01 / (FACTOR_CLASSES - 1))
    fac_probs[..., 0] = 0.99
    b["fac_probs"] = fac_probs
    b["fac_labels"] = fac_labels
    out = loss_fn(**b)
    # With correct log(p): Lfac ≈ -0.99*log(0.99) - (C-1)*0.01*log(0.01/(C-1)) ≈ 0.056
    # With bug log_softmax(softmax): Lfac would be ≈ 1.5 (artificial floor)
    assert out["Lfac"] < 0.15, (
        f"Lfac={out['Lfac']:.3f} too high for near-perfect prediction — "
        "log_softmax(softmax) bug may have regressed"
    )


def test_lfac_decreases_toward_zero_with_better_alignment():
    """Monotonicity: better alignment → lower Lfac."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=1.0, lambda4=0.0,
        lambda5=0.0, lambda6=0.0, lambda7=0.0,
        K_h=4, learned_lambdas=False,
    )
    b = _make_batch(B=4, K=8)
    # One-hot labels
    fac_labels = torch.zeros(4, 8, FACTOR_CLASSES)
    fac_labels[..., 0] = 1.0
    b["fac_labels"] = fac_labels

    # Uniform prediction
    b["fac_probs"] = torch.full((4, 8, FACTOR_CLASSES), 1.0 / FACTOR_CLASSES)
    lfac_uniform = loss_fn(**b)["Lfac"]

    # Partially correct (0.5 on true factor)
    probs_mid = torch.full((4, 8, FACTOR_CLASSES), 0.5 / (FACTOR_CLASSES - 1))
    probs_mid[..., 0] = 0.5
    b["fac_probs"] = probs_mid
    lfac_mid = loss_fn(**b)["Lfac"]

    # Mostly correct (0.9 on true factor)
    probs_hi = torch.full((4, 8, FACTOR_CLASSES), 0.1 / (FACTOR_CLASSES - 1))
    probs_hi[..., 0] = 0.9
    b["fac_probs"] = probs_hi
    lfac_hi = loss_fn(**b)["Lfac"]

    assert lfac_uniform > lfac_mid > lfac_hi, (
        f"Lfac not monotonic: uniform={lfac_uniform:.3f}, "
        f"mid={lfac_mid:.3f}, hi={lfac_hi:.3f}"
    )
    # Uniform should give ≈ log(C) = log(10) ≈ 2.303
    assert abs(lfac_uniform - math.log(FACTOR_CLASSES)) < 0.1


# ── Lfaith relative margin (P1 #8) ──────────────────────────────────

def test_lfaith_zero_when_gap_exceeds_margin():
    """Relative gap above margin → no faithfulness penalty."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
    )
    b = _make_batch(correct=True)
    # p_full ≈ 1.0 on argmax (one-hot logits × 10), p_mask near uniform
    b["masked_dir_logits"] = torch.zeros_like(b["dir_logits"])
    out = loss_fn(**b)
    # rel_gap ≈ (1 - 0.33) / 1.0 = 0.67 >> 0.12 → Lfaith should be 0
    assert out["Lfaith"] == pytest.approx(0.0, abs=1e-5)


def test_lfaith_positive_when_mask_unchanged():
    """If masking doesn't change prediction, rel_gap=0 < margin → loss active."""
    loss_fn = MultiObjectiveLoss(
        lambda1=1.0, lambda2=0.5, lambda3=0.3, lambda4=0.2,
        lambda5=0.1, lambda6=0.1, lambda7=0.05,
        K_h=4, faith_margin=0.12, coverage_target=0.35,
        learned_lambdas=False,
    )
    b = _make_batch(correct=True)
    b["masked_dir_logits"] = b["dir_logits"].clone()  # identical → rel_gap=0
    out = loss_fn(**b)
    assert out["Lfaith"] == pytest.approx(0.12, rel=1e-3)
