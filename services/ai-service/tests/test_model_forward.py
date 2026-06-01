"""Unit tests for SAFEAlertNet core forward paths — Session 23 P2 #11 & #12.

Targets:
  (P2 #11) scale_gradient: identity in forward, scaled in backward. This is
    critical for Session 22 Fix 1 (factor-grounded coupling) — a regression
    silently breaking the gradient scale would revert Contribution 2 to
    vestigial.

  (P2 #12) forward_masked: eval-mode assertion guard + shape handling +
    proper behaviour when all articles are masked (market-only fallback).
    These properties underpin L_faith (Eq.36) — a bypass would corrupt the
    faithfulness gap signal with dropout noise.

Tests avoid requiring the full dataset pipeline; they build minimal tensors
directly. Run with:
    python -m pytest tests/test_model_forward.py -v
"""
from __future__ import annotations

import pytest
import torch

# Allow running from ai-service root OR from tests/ without install
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "app"))

from v2.models.safe_alert_net import (  # noqa: E402
    SAFEAlertNet,
    scale_gradient,
    _GradScale,
    ExplanationModule,
    BarSequenceEncoder,
    MultiTimescaleMarketEncoder,
)


# ─────────────────────────────────────────────────────────────────────────────
# P2 #11: scale_gradient forward + backward correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestScaleGradient:
    """Backward gradient scaling — Session 22 Fix 1."""

    def test_forward_is_identity(self):
        """scale_gradient must not change the forward value (Eq.22 z_fac contribution)."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = scale_gradient(x, scale=0.1)
        assert torch.equal(x.detach(), y.detach()), (
            f"scale_gradient changed forward values: {x.data} != {y.data}"
        )

    def test_backward_scales_gradient(self):
        """Gradient through scale_gradient must be multiplied by `scale`."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = scale_gradient(x, scale=0.1)
        loss = (y * y).sum()  # dL/dy = 2y, dL/dx = 0.1 · 2y = 0.2y
        loss.backward()
        expected = 0.1 * 2.0 * torch.tensor([1.0, 2.0, 3.0])  # [0.2, 0.4, 0.6]
        assert torch.allclose(x.grad, expected, atol=1e-6), (
            f"Gradient scaling wrong: got {x.grad.tolist()}, expected {expected.tolist()}"
        )

    def test_backward_with_scale_one_is_identity(self):
        """scale=1.0 → gradient unchanged (regression test)."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = scale_gradient(x, scale=1.0)
        loss = (y * y).sum()
        loss.backward()
        expected = 2.0 * torch.tensor([1.0, 2.0, 3.0])  # [2, 4, 6]
        assert torch.allclose(x.grad, expected, atol=1e-6)

    def test_backward_with_scale_zero_kills_gradient(self):
        """scale=0 → no gradient flows (equivalent to .detach() at this hop)."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = scale_gradient(x, scale=0.0)
        loss = (y * y).sum()
        loss.backward()
        assert torch.allclose(x.grad, torch.zeros(3), atol=1e-8), (
            f"scale=0 should zero gradient, got {x.grad.tolist()}"
        )

    def test_multidim_tensor(self):
        """Works for 2D/3D tensors (matches z_fac shape (B, d_f))."""
        x = torch.randn(4, 8, requires_grad=True)
        y = scale_gradient(x, scale=0.1)
        assert y.shape == x.shape
        loss = y.sum()
        loss.backward()
        # dL/dx should be 0.1 · ones_like(x)
        assert torch.allclose(x.grad, 0.1 * torch.ones_like(x), atol=1e-6)

    def test_gradscale_function_directly(self):
        """_GradScale.apply is the underlying autograd Function."""
        x = torch.tensor([1.0], requires_grad=True)
        y = _GradScale.apply(x, 0.25)
        y.backward()
        assert torch.allclose(x.grad, torch.tensor([0.25]), atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# P2 #12: forward_masked eval-mode + shape handling
# ─────────────────────────────────────────────────────────────────────────────

class TestForwardMasked:
    """forward_masked contract — Session 22 Fix + Session 23 P1 #6."""

    @pytest.fixture
    def model(self):
        """Small model for fast tests."""
        # K_15m=3, K_1h=4, K_4h=5, K_24h=8 matches paper
        m = SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5, K_24h=8)
        m.eval()
        return m

    @pytest.fixture
    def batch(self):
        B, K = 2, 6
        return {
            "market_feat": torch.randn(B, 63),
            "article_emb": torch.randn(B, K, 768),
            "article_mask": torch.ones(B, K).float(),
            "article_meta_vec": torch.randn(B, K, 14),
        }

    def test_forward_masked_shape(self, model, batch):
        """Returns a dict with dir_logits shape (B, 3)."""
        out = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=batch["article_mask"],
            article_meta_vec=batch["article_meta_vec"],
        )
        assert "dir_logits" in out
        assert out["dir_logits"].shape == (2, 3)
        assert torch.isfinite(out["dir_logits"]).all()

    def test_forward_masked_forces_eval_mode(self, model, batch):
        """Even if caller is in train() mode, internal eval() guard kicks in.

        Session 22 P0 #6 + Session 23 P1 #6 — prevents dropout noise from
        corrupting the Lfaith gap signal.
        """
        model.train()  # switch to training mode
        out = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=batch["article_mask"],
            article_meta_vec=batch["article_meta_vec"],
        )
        assert out["dir_logits"].shape == (2, 3)
        # After call, training mode should be restored
        assert model.training, "forward_masked should restore training mode on exit"

    def test_private_impl_asserts_eval_mode(self, model, batch):
        """_forward_masked_impl raises when called in training mode (P1 #6 guard)."""
        model.train()
        with pytest.raises(AssertionError, match="eval mode"):
            model._forward_masked_impl(
                batch["market_feat"], horizon="1h",
                article_emb=batch["article_emb"],
                article_mask=batch["article_mask"],
                article_meta_vec=batch["article_meta_vec"],
            )

    def test_forward_masked_all_masked(self, model, batch):
        """When article_mask is all zeros, falls back to market-only gracefully."""
        zero_mask = torch.zeros_like(batch["article_mask"])
        out = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=zero_mask,
            article_meta_vec=batch["article_meta_vec"],
        )
        assert out["dir_logits"].shape == (2, 3)
        assert torch.isfinite(out["dir_logits"]).all()

    def test_forward_masked_is_differentiable(self, model, batch):
        """Output must be differentiable so L_faith backward works."""
        out = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=batch["article_mask"],
            article_meta_vec=batch["article_meta_vec"],
        )
        loss = out["dir_logits"].sum()
        # Should not raise — masked forward preserves gradient graph
        loss.backward()

    def test_forward_masked_different_horizons(self, model, batch):
        """K_h varies by horizon; masked forward handles each correctly."""
        for h in ["15m", "1h", "4h", "24h"]:
            out = model.forward_masked(
                batch["market_feat"], horizon=h,
                article_emb=batch["article_emb"],
                article_mask=batch["article_mask"],
                article_meta_vec=batch["article_meta_vec"],
            )
            assert out["dir_logits"].shape == (2, 3), f"horizon={h} gave wrong shape"

    def test_forward_masked_accepts_symbol_compat_arg(self, model, batch):
        """The paper-final Eq.9 query ignores symbol but keeps the argument."""
        model.eval()
        out_btc = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=batch["article_mask"],
            article_meta_vec=batch["article_meta_vec"],
            symbol="BTCUSDT",
        )
        out_unk = model.forward_masked(
            batch["market_feat"], horizon="1h",
            article_emb=batch["article_emb"],
            article_mask=batch["article_mask"],
            article_meta_vec=batch["article_meta_vec"],
            symbol=None,
        )
        assert torch.allclose(out_btc["dir_logits"], out_unk["dir_logits"])


# ─────────────────────────────────────────────────────────────────────────────
# Factor-pathway gradient coupling (integration test for Session 22 Fix 1)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorGradientCoupling:
    """Verifies the Fix 1 intent: Ldir gradient reaches factor_mod at 10%."""

    def test_ldir_gradient_reaches_factor_mod(self):
        """With scale_gradient(z_fac, 0.1), Ldir must backprop into W_fac."""
        model = SAFEAlertNet(market_dim=63, has_news=True)
        model.eval()
        B, K = 2, 4
        out = model(
            torch.randn(B, 63), horizon="1h",
            article_emb=torch.randn(B, K, 768),
            article_mask=torch.ones(B, K),
            article_meta_vec=torch.randn(B, K, 14),
        )
        loss = out["dir_logits"].sum()  # only direction loss
        loss.backward()
        # W_fac should have non-zero gradient (the whole point of Fix 1)
        assert model.factor_mod.W_fac.weight.grad is not None, (
            "factor_mod.W_fac has no gradient — scale_gradient may be broken"
        )
        grad_norm = model.factor_mod.W_fac.weight.grad.norm().item()
        assert grad_norm > 0.0, (
            f"factor_mod W_fac gradient norm = {grad_norm}, expected > 0"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Template validation (P3 #13)
# ─────────────────────────────────────────────────────────────────────────────

class TestExplanationTemplates:
    def test_templates_validate_at_import(self):
        """_validate_templates was called at import; re-call must still pass."""
        ExplanationModule._validate_templates()  # must not raise

    def test_templates_fail_on_missing_placeholder(self, monkeypatch):
        """Simulate a refactor that drops a required placeholder."""
        broken = dict(ExplanationModule._TEMPLATES)
        broken["en"] = "Signal: {symbol} — no other fields"
        monkeypatch.setattr(ExplanationModule, "_TEMPLATES", broken)
        with pytest.raises(ValueError, match="missing=\\["):
            ExplanationModule._validate_templates()


# ─────────────────────────────────────────────────────────────────────────────
# P0 #3: BarSequenceEncoder + MultiTimescaleMarketEncoder sequence mode
# ─────────────────────────────────────────────────────────────────────────────

class TestBarSequenceEncoder:
    """Paper Eq.16 literal implementation (Session 23 P0 #3)."""

    def test_forward_shape(self):
        """Input (B, L, d) → output (B, out_dim)."""
        enc = BarSequenceEncoder(seq_len=20, feat_dim=10, out_dim=64)
        x = torch.randn(4, 20, 10)
        y = enc(x)
        assert y.shape == (4, 64)
        assert torch.isfinite(y).all()

    def test_rejects_wrong_shape(self):
        """Clear error if caller passes 2D tensor by mistake."""
        enc = BarSequenceEncoder(seq_len=20, feat_dim=10, out_dim=64)
        with pytest.raises(ValueError, match=r"expects \(B, L, d\)"):
            enc(torch.randn(4, 10))      # missing L axis

    def test_differentiable(self):
        """Gradient flows through Conv + pool + linear."""
        enc = BarSequenceEncoder(seq_len=20, feat_dim=10, out_dim=64)
        x = torch.randn(4, 20, 10, requires_grad=True)
        y = enc(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_seq_len(self):
        """Adaptive pool handles any L at inference (but init sets training L)."""
        enc = BarSequenceEncoder(seq_len=20, feat_dim=10, out_dim=64)
        # Conv kernels are seq-length-agnostic; adaptive pool absorbs L.
        for L in [15, 20, 25]:
            y = enc(torch.randn(2, L, 10))
            assert y.shape == (2, 64), f"seq_len={L} gave {y.shape}"


class TestFullModelBarSequencesE2E:
    """End-to-end forward through SAFEAlertNet(use_bar_sequences=True).

    Session 23 P0 #3: verifies the paper-faithful Eq.16 path works all the
    way from market_bars input through direction / return / confidence heads.
    """

    @pytest.fixture
    def bar_model(self):
        m = SAFEAlertNet(
            market_dim=63, has_news=True,
            use_bar_sequences=True, bar_seq_len=20, bar_feat_dim=10,
        )
        m.eval()
        return m

    @pytest.fixture
    def bar_batch(self):
        B, K, L, d = 3, 6, 20, 10
        return {
            "market_feat": torch.randn(B, 63),
            "market_bars": torch.randn(B, 5, L, d),
            "article_emb": torch.randn(B, K, 768),
            "article_mask": torch.ones(B, K),
            "article_meta_vec": torch.randn(B, K, 14),
        }

    def test_full_forward_with_bars(self, bar_model, bar_batch):
        out = bar_model(
            bar_batch["market_feat"], horizon="1h",
            article_emb=bar_batch["article_emb"],
            article_mask=bar_batch["article_mask"],
            article_meta_vec=bar_batch["article_meta_vec"],
            market_bars=bar_batch["market_bars"],
        )
        assert out["dir_logits"].shape == (3, 3)
        assert out["ret_pred"].shape == (3,)
        assert out["confidence"].shape == (3,)
        assert torch.isfinite(out["dir_logits"]).all()

    def test_forward_masked_with_bars(self, bar_model, bar_batch):
        out = bar_model.forward_masked(
            bar_batch["market_feat"], horizon="1h",
            article_emb=bar_batch["article_emb"],
            article_mask=bar_batch["article_mask"],
            article_meta_vec=bar_batch["article_meta_vec"],
            market_bars=bar_batch["market_bars"],
        )
        assert out["dir_logits"].shape == (3, 3)

    def test_missing_bars_raises(self, bar_model, bar_batch):
        """use_bar_sequences=True + market_bars=None must raise."""
        with pytest.raises(ValueError, match="market_bars not provided"):
            bar_model(
                bar_batch["market_feat"], horizon="1h",
                article_emb=bar_batch["article_emb"],
                article_mask=bar_batch["article_mask"],
                article_meta_vec=bar_batch["article_meta_vec"],
                market_bars=None,
            )

    def test_wrong_bars_shape_raises(self, bar_model, bar_batch):
        """3D market_bars (missing TF axis) must raise."""
        bad_bars = bar_batch["market_bars"][:, 0]   # (B, L, d) — missing TF axis
        with pytest.raises(ValueError, match=r"\(B, 5, L, d\)"):
            bar_model(
                bar_batch["market_feat"], horizon="1h",
                article_emb=bar_batch["article_emb"],
                article_mask=bar_batch["article_mask"],
                article_meta_vec=bar_batch["article_meta_vec"],
                market_bars=bad_bars,
            )

    def test_scalar_model_ignores_bars(self):
        """use_bar_sequences=False + market_bars provided → bars ignored."""
        scalar_model = SAFEAlertNet(
            market_dim=63, has_news=True,
            use_bar_sequences=False,
        )
        scalar_model.eval()
        out = scalar_model(
            torch.randn(2, 63), horizon="1h",
            article_emb=torch.randn(2, 4, 768),
            article_mask=torch.ones(2, 4),
            article_meta_vec=torch.randn(2, 4, 14),
            market_bars=torch.randn(2, 5, 20, 10),  # should be ignored
        )
        assert out["dir_logits"].shape == (2, 3)


class TestMultiTimescaleSequenceMode:
    """MultiTimescaleMarketEncoder in use_bar_sequences=True mode."""

    def test_sequence_mode_forward(self):
        """Accepts list of 5 × (B, L, d) bar tensors, produces (B, dm) + (B, 5, dm)."""
        enc = MultiTimescaleMarketEncoder(
            market_dim=63, n_timeframes=5, dm=64,
            use_bar_sequences=True,
            bar_seq_len=20, bar_feat_dim=10,
        )
        bars = [torch.randn(4, 20, 10) for _ in range(5)]
        h_emb = torch.randn(4, 16)
        Z_mkt, u_stack = enc(bars, h_emb)
        assert Z_mkt.shape == (4, 64)
        assert u_stack.shape == (4, 5, 64)

    def test_sequence_mode_rejects_scalar_tensor(self):
        """use_bar_sequences=True with (B, 63) scalar input must raise."""
        enc = MultiTimescaleMarketEncoder(
            market_dim=63, n_timeframes=5, dm=64,
            use_bar_sequences=True,
        )
        h_emb = torch.randn(4, 16)
        with pytest.raises(ValueError, match="list of 5 bar-sequence"):
            enc(torch.randn(4, 63), h_emb)

    def test_scalar_mode_still_works(self):
        """Backward compat: default mode still accepts (B, 63) scalar."""
        enc = MultiTimescaleMarketEncoder(market_dim=63, n_timeframes=5, dm=64)
        h_emb = torch.randn(4, 16)
        Z_mkt, u_stack = enc(torch.randn(4, 63), h_emb)
        assert Z_mkt.shape == (4, 64)
        assert u_stack.shape == (4, 5, 64)

    def test_seq_mode_learnable_params_differ(self):
        """Sequence encoder has more params than scalar MLP (Conv layers)."""
        scalar_enc = MultiTimescaleMarketEncoder(market_dim=63, n_timeframes=5, dm=64)
        seq_enc    = MultiTimescaleMarketEncoder(
            market_dim=63, n_timeframes=5, dm=64,
            use_bar_sequences=True,
        )
        n_scalar = sum(p.numel() for p in scalar_enc.parameters())
        n_seq    = sum(p.numel() for p in seq_enc.parameters())
        # Bar-sequence encoder has two Conv1d layers per TF → more params
        assert n_seq > n_scalar, (
            f"Sequence mode should have more params "
            f"(got seq={n_seq} <= scalar={n_scalar})"
        )
