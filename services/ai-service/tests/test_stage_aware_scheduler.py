"""Unit tests for StageAwareLRScheduler (R3 #A).

Validates the fix for the Fold 1 LR-collapse bug where CosineAnnealingWarmRestarts
decayed to ~1e-7 at Stage 3 entry (observed actual 3.7e-6), making Stage 3 λ₅ jump
0.02→0.15 unable to push temperature back to 1.0.

Runs:
    cd services/ai-service
    pytest tests/test_stage_aware_scheduler.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

_AI = Path(__file__).resolve().parent.parent
for p in (str(_AI), str(_AI / "app" / "v2")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipelines.train_safe_alert import StageAwareLRScheduler  # noqa: E402


def _make_opt(lr: float = 3e-4) -> torch.optim.Optimizer:
    """Dummy optimizer holding one parameter, for LR inspection."""
    p = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.AdamW([p], lr=lr)


@pytest.fixture
def sched_40ep():
    """40-epoch schedule matching Fold 1 config."""
    opt = _make_opt(3e-4)
    return StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=3e-4,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
        stage3_lr_scale=0.5, warmup_start_frac=0.1, eta_frac=0.02,
    ), opt


# ── Stage-boundary correctness ──────────────────────────────────────────────

def test_stage_boundaries_40_epochs(sched_40ep):
    sched, _ = sched_40ep
    assert sched.stage1_end == 8   # 40 × 0.20
    assert sched.stage2_end == 28  # 40 × 0.70
    assert sched.stage3_start_lr == 3e-4 * 0.5
    assert sched.warmup_start_lr == 3e-4 * 0.1
    assert sched.eta_lr == 3e-4 * 0.02


# ── Warmup phase (Stage 1) ──────────────────────────────────────────────────

def test_warmup_starts_at_base_lr_over_10(sched_40ep):
    sched, _ = sched_40ep
    assert sched._compute_lr(1) == pytest.approx(3e-5)  # base / 10


def test_warmup_reaches_base_at_stage1_end(sched_40ep):
    sched, _ = sched_40ep
    assert sched._compute_lr(8) == pytest.approx(3e-4, rel=1e-4)


def test_warmup_monotonically_increasing(sched_40ep):
    sched, _ = sched_40ep
    prev = -1
    for ep in range(1, 9):
        lr = sched._compute_lr(ep)
        assert lr > prev, f"Warmup not monotonic at ep {ep}"
        prev = lr


# ── Stage 2 (cosine decay base → stage3_start) ──────────────────────────────

def test_stage2_monotonically_decreasing(sched_40ep):
    sched, _ = sched_40ep
    prev = 1e9
    for ep in range(9, 29):
        lr = sched._compute_lr(ep)
        assert lr < prev, f"Stage 2 LR not monotonic at ep {ep} (lr={lr})"
        prev = lr


def test_stage2_ends_at_stage3_start_lr(sched_40ep):
    sched, _ = sched_40ep
    assert sched._compute_lr(28) == pytest.approx(1.5e-4, rel=1e-3)


# ── Stage 3 (cosine decay stage3_start → eta) ───────────────────────────────

def test_stage3_starts_at_stage3_start_lr(sched_40ep):
    """Regression test for Fold 1 bug: Stage 3 entry was 3.7e-6 instead of 1.5e-4."""
    sched, _ = sched_40ep
    ep29_lr = sched._compute_lr(29)
    # Must be ≥ 1.4e-4 (95% of stage3_start) — the OLD bug gave 3.7e-6 = 2% of target
    assert ep29_lr >= 1.4e-4, f"Stage 3 LR collapsed: got {ep29_lr}, need >= 1.4e-4"
    assert ep29_lr <= 1.5e-4, f"Stage 3 LR too high: got {ep29_lr}, should start <= 1.5e-4"


def test_stage3_ends_at_eta_lr(sched_40ep):
    sched, _ = sched_40ep
    assert sched._compute_lr(40) == pytest.approx(6e-6, rel=1e-3)


def test_stage3_floor_not_below_eta(sched_40ep):
    sched, _ = sched_40ep
    # At epoch 40 or beyond, LR clamps to eta_lr
    assert sched._compute_lr(40) >= 6e-6
    assert sched._compute_lr(100) >= 6e-6  # clamp working even past total_epochs


# ── No LR collapse across any transition ────────────────────────────────────

def test_no_sudden_lr_drop_between_stages(sched_40ep):
    """Guard against the CosineAnnealingWarmRestarts bug where restarts
    caused 100× LR drops at stage boundaries.
    """
    sched, _ = sched_40ep
    for ep in range(1, 40):
        lr_prev = sched._compute_lr(ep)
        lr_curr = sched._compute_lr(ep + 1)
        ratio = lr_curr / lr_prev
        # Allow up to 15% decrease per epoch (normal cosine)
        assert ratio >= 0.85, \
            f"Sharp LR drop at ep {ep}→{ep+1}: {lr_prev:.3e}→{lr_curr:.3e} (ratio={ratio:.3f})"


# ── step() side effect: writes to optimizer ─────────────────────────────────

def test_step_updates_param_groups(sched_40ep):
    sched, opt = sched_40ep
    sched.step(epoch=1)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-5)
    sched.step(epoch=8)
    assert opt.param_groups[0]["lr"] == pytest.approx(3e-4, rel=1e-4)
    sched.step(epoch=29)
    assert opt.param_groups[0]["lr"] >= 1.4e-4


def test_get_last_lr_reflects_latest_step(sched_40ep):
    sched, _ = sched_40ep
    sched.step(epoch=15)
    assert sched.get_last_lr()[0] == sched.last_lr


# ── Edge cases ──────────────────────────────────────────────────────────────

def test_tiny_epoch_budget_does_not_crash():
    """epochs=5 is below any sane training; must not divide by zero."""
    opt = _make_opt(1e-3)
    sched = StageAwareLRScheduler(
        opt, total_epochs=5, base_lr=1e-3,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
    )
    for ep in range(1, 6):
        lr = sched._compute_lr(ep)
        assert lr > 0 and lr < 1.0


def test_stage1_frac_zero_uses_min_one():
    """stage1_end_frac=0 with 40 epochs should floor stage1_end to 1, not 0."""
    opt = _make_opt(1e-3)
    sched = StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=1e-3,
        stage1_end_frac=0.0, stage2_end_frac=0.5,
    )
    assert sched.stage1_end >= 1
    assert sched.stage2_end > sched.stage1_end


def test_state_dict_roundtrip():
    opt = _make_opt(1e-3)
    sched = StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=1e-3,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
    )
    sched.step(epoch=15)
    snap = sched.state_dict()
    assert "last_lr" in snap
    # Restore
    sched.load_state_dict(snap)
    assert sched.last_lr == snap["last_lr"]


# ── Per-group lr_mult (per-component optimizer groups) ─────────────────────

def _make_opt_with_groups(base_lr: float = 3e-4) -> torch.optim.Optimizer:
    """Optimizer with 4 groups matching trainer config: main/selector/factor/conf."""
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(4)]
    return torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": base_lr,        "lr_mult": 1.0, "name": "main"},
            {"params": [params[1]], "lr": base_lr * 1.5,  "lr_mult": 1.5, "name": "selector"},
            {"params": [params[2]], "lr": base_lr * 2.0,  "lr_mult": 2.0, "name": "factor"},
            {"params": [params[3]], "lr": base_lr * 1.5,  "lr_mult": 1.5, "name": "confidence"},
        ]
    )


def test_lr_mult_preserved_across_warmup():
    opt = _make_opt_with_groups(3e-4)
    sched = StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=3e-4,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
    )
    base = sched.step(epoch=2)  # early Stage 1 (warmup region)
    lrs = [g["lr"] for g in opt.param_groups]
    assert lrs[0] == pytest.approx(base * 1.0)
    assert lrs[1] == pytest.approx(base * 1.5)
    assert lrs[2] == pytest.approx(base * 2.0)
    assert lrs[3] == pytest.approx(base * 1.5)


def test_lr_mult_preserved_at_stage3():
    opt = _make_opt_with_groups(3e-4)
    sched = StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=3e-4,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
        stage3_lr_scale=0.5,
    )
    base = sched.step(epoch=30)  # Stage 3
    # Base at Stage 3 should be >= eta_lr (6e-6), not collapsed
    assert base >= 5e-6
    factor_lr = opt.param_groups[2]["lr"]
    # factor head must train at 2× base to offset gradient interference
    assert factor_lr == pytest.approx(base * 2.0)


def test_get_last_lr_returns_per_group():
    opt = _make_opt_with_groups(3e-4)
    sched = StageAwareLRScheduler(
        opt, total_epochs=40, base_lr=3e-4,
        stage1_end_frac=0.20, stage2_end_frac=0.70,
    )
    sched.step(epoch=10)
    per_group = sched.get_last_lr()
    assert len(per_group) == 4
    # Ratios must reflect configured multipliers
    assert per_group[1] / per_group[0] == pytest.approx(1.5)
    assert per_group[2] / per_group[0] == pytest.approx(2.0)
