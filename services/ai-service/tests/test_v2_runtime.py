import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from app.v2.nlp.news_selector import select_news_for_horizon
from app.v2.models.safe_alert_net import SAFEAlertNet
from app.v2.pipelines import live_infer
from app.v2.pipelines.metrics_safe_alert import (
    search_alert_policy,
    fit_temperature_scaling,
    apply_temperature_to_logits,
)
from app.v2.pipelines.precompute_factor_labels import compute_factor_probs_for_article
from app.v2.pipelines.train_safe_alert import SAFEAlertTrainer
from app.v2.pipelines.safe_alert_training_utils import MultiObjectiveLoss
from app.v2.pipelines.utils import standardize_walk_forward_artifacts


class _StubEmbedder:
    def encode(self, texts):
        return np.ones((len(texts), 768), dtype=np.float32)


class TestV2Runtime(unittest.TestCase):
    def _make_trainer(self):
        model = SAFEAlertNet(market_dim=63, has_news=True, K_1h=4, K_4h=5)
        return SAFEAlertTrainer(model, device="cpu", lr=1e-3, horizon="1h", K_h=4)

    def test_news_selector_excludes_current_timestamp(self):
        now = pd.Timestamp("2026-04-11T10:00:00Z")
        df = pd.DataFrame(
            [
                {"title": "old", "published_at": now - pd.Timedelta(minutes=10), "source": "coindesk"},
                {"title": "at_now", "published_at": now, "source": "coindesk"},
            ]
        )
        out = select_news_for_horizon(df, now, "1h")
        self.assertEqual(out["title"].tolist(), ["old"])

    def test_standardize_walk_forward_uses_validation_policy(self):
        wf = {
            "fold_metrics": [
                {
                    "fold": 1,
                    "val": {
                        "tau": 0.4,
                        "gamma": 0.5,
                        "temperature": 1.2,
                        "model_score": 0.8,
                        "alert_precision": 0.6,
                        "alert_coverage": 0.3,
                        "alert_sharpe": 0.02,
                    },
                    "test": {"model_score": 0.1, "alert_precision": 0.1, "alert_coverage": 0.1, "alert_sharpe": -1.0},
                },
                {
                    "fold": 2,
                    "val": {
                        "tau": 0.7,
                        "gamma": 0.8,
                        "model_score": 0.2,
                        "alert_precision": 0.9,
                        "alert_coverage": 0.2,
                        "alert_sharpe": 0.5,
                    },
                    "test": {"model_score": 0.9, "alert_precision": 0.9, "alert_coverage": 0.2, "alert_sharpe": 0.5},
                },
            ],
            "averages": {},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wf_path = root / "walk_forward_results.json"
            wf_path.write_text(json.dumps(wf), encoding="utf-8")
            summary = standardize_walk_forward_artifacts(wf_path, "BTCUSDT", "1h", artifact_dir=root)
            self.assertEqual(summary["selected_deployment_fold"], 1)
            self.assertAlmostEqual(summary["deployment_policy"]["tau"], 0.4)
            self.assertAlmostEqual(summary["deployment_policy"]["gamma"], 0.5)
            self.assertAlmostEqual(summary["deployment_policy"]["temperature"], 1.2)
            policy_path = root / "safe_alert_btcusdt_1h_policy.json"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(policy["temperature"], 1.2)

    def test_build_article_tensors_returns_14d_meta(self):
        news = pd.DataFrame(
            [
                {
                    "title": "BlackRock ETF inflow",
                    "content": "BlackRock ETF inflow improves bitcoin outlook.",
                    "source": "coindesk",
                    "published_at": pd.Timestamp("2026-04-11T09:30:00Z"),
                    "sentiment_score": 0.6,
                },
                {
                    "title": "Exchange hack risk",
                    "content": "Exchange hack raises risk for bitcoin traders.",
                    "source": "reuters",
                    "published_at": pd.Timestamp("2026-04-11T09:00:00Z"),
                    "sentiment_score": -0.7,
                },
            ]
        )
        with mock.patch.object(live_infer, "_get_runtime_embedder", return_value=_StubEmbedder()):
            tensors = live_infer._build_article_tensors(
                news, "1h", pd.Timestamp("2026-04-11T10:00:00Z"), "BTCUSDT"
            )
        self.assertIsNotNone(tensors)
        emb, meta, meta_dicts = tensors
        self.assertEqual(tuple(emb.shape), (2, 768))
        self.assertEqual(tuple(meta.shape), (2, 14))
        self.assertEqual(len(meta_dicts), 2)

    def test_loss_filters_invalid_samples_without_poisoning_batch(self):
        loss_fn = MultiObjectiveLoss(K_h=4, learned_lambdas=False)
        dir_logits = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [float("nan"), 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        dir_labels = torch.tensor([2, 1], dtype=torch.long)
        ret_pred = torch.tensor([0.01, 0.02], dtype=torch.float32)
        ret_labels = torch.tensor([0.02, 0.01], dtype=torch.float32)
        fac_probs = torch.full((2, 4, 10), 0.1, dtype=torch.float32)
        fac_labels = torch.full((2, 4, 10), 0.1, dtype=torch.float32)
        attn_weights = torch.ones((2, 4), dtype=torch.float32)
        confidence = torch.tensor([0.7, 0.8], dtype=torch.float32)
        calibration_targets = torch.tensor([1, 0], dtype=torch.long)
        masked_dir_logits = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [float("nan"), 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        article_mask = torch.ones((2, 4), dtype=torch.float32)

        out = loss_fn(
            dir_logits=dir_logits,
            dir_labels=dir_labels,
            ret_pred=ret_pred,
            ret_labels=ret_labels,
            fac_probs=fac_probs,
            fac_labels=fac_labels,
            attn_weights=attn_weights,
            confidence=confidence,
            calibration_targets=calibration_targets,
            masked_dir_logits=masked_dir_logits,
            article_mask=article_mask,
        )

        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_policy_search_uses_validation_grid_and_prefers_better_objective(self):
        confidence = np.array([0.95, 0.90, 0.85, 0.60, 0.55, 0.52], dtype=np.float32)
        preds = np.array([2, 2, 0, 1, 1, 0], dtype=np.int64)
        labels = np.array([2, 2, 0, 1, 0, 0], dtype=np.int64)
        ret_labels = np.array([0.03, 0.02, 0.01, -0.01, -0.02, 0.015], dtype=np.float32)
        dir_probs = np.array(
            [
                [0.02, 0.03, 0.95],
                [0.03, 0.06, 0.91],
                [0.84, 0.08, 0.08],
                [0.14, 0.70, 0.16],
                [0.25, 0.55, 0.20],
                [0.70, 0.15, 0.15],
            ],
            dtype=np.float32,
        )

        policy = search_alert_policy(
            confidence,
            preds,
            labels,
            ret_labels,
            dir_probs,
            macro_f1=0.4,
            mcc=0.15,
            ece=0.05,
        )

        self.assertEqual(policy["policy_source"], "validation_grid_search")
        self.assertGreaterEqual(policy["tau"], policy["tau_seed"] - 0.2)
        self.assertGreaterEqual(policy["gamma"], policy["gamma_seed"] - 0.2)
        self.assertIn("policy_objective", policy)

    def test_temperature_scaling_fits_positive_temperature(self):
        logits = np.array(
            [
                [3.0, 0.2, -0.1],
                [2.8, 0.1, -0.2],
                [0.3, 2.1, 0.1],
                [0.1, 0.3, 2.2],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 2], dtype=np.int64)
        fit = fit_temperature_scaling(logits, labels)
        probs = apply_temperature_to_logits(logits, fit["temperature"])
        self.assertGreater(fit["temperature"], 0.0)
        self.assertTrue(np.isfinite(fit["temperature_nll"]))
        self.assertEqual(probs.shape, logits.shape)

    def test_stage3_lr_drop_halves_learning_rate(self):
        trainer = self._make_trainer()
        before = trainer.optimizer.param_groups[0]["lr"]
        trainer._enter_stage3()
        after = trainer.optimizer.param_groups[0]["lr"]
        self.assertAlmostEqual(after, before * 0.5, places=10)

    def test_prepare_attn_weights_scales_sum_close_to_k_h(self):
        trainer = self._make_trainer()
        attn = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        prepared = trainer._prepare_attn_weights(attn, mask, batch_size=1, num_articles=4)
        self.assertAlmostEqual(float(prepared.sum().item()), 4.0, places=5)

    def test_prepare_attn_weights_renormalizes_topk_gated_mass_to_k_h(self):
        trainer = self._make_trainer()
        attn = torch.tensor([[0.55, 0.25, 0.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        prepared = trainer._prepare_attn_weights(attn, mask, batch_size=1, num_articles=4)
        self.assertAlmostEqual(float(prepared.sum().item()), 4.0, places=5)
        self.assertAlmostEqual(float(prepared[0, 0].item() / prepared[0, 1].item()), 0.55 / 0.25, places=5)

    def test_prepare_attn_weights_respects_sparse_article_target(self):
        trainer = self._make_trainer()
        attn = torch.tensor([[0.60, 0.40, 0.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
        prepared = trainer._prepare_attn_weights(attn, mask, batch_size=1, num_articles=4)
        self.assertAlmostEqual(float(prepared.sum().item()), 2.0, places=5)

    def test_return_scale_fits_from_loader_only(self):
        trainer = self._make_trainer()
        train_loader = DataLoader(
            [
                {"return": torch.tensor(0.01, dtype=torch.float32)},
                {"return": torch.tensor(0.02, dtype=torch.float32)},
                {"return": torch.tensor(-0.01, dtype=torch.float32)},
            ],
            batch_size=3,
        )
        trainer._compute_return_scale(train_loader)
        self.assertGreater(float(trainer.loss_fn.return_scale.item()), 0.0)
        self.assertLess(float(trainer.loss_fn.return_scale.item()), 0.05)

    def test_factor_label_stats_enable_small_smoothing_for_overconfident_labels(self):
        trainer = self._make_trainer()
        factor = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        train_loader = DataLoader(
            [{"factor": factor[0], "article_mask": torch.tensor([1, 1], dtype=torch.bool)},
             {"factor": factor[1], "article_mask": torch.tensor([1, 1], dtype=torch.bool)}],
            batch_size=2,
        )
        trainer._compute_factor_label_stats(train_loader)
        self.assertGreater(float(trainer.loss_fn.factor_label_smoothing.item()), 0.0)
        self.assertGreater(float(trainer.loss_fn.factor_label_mean_max_prob.item()), 0.9)

    def test_live_entity_sentiment_is_factor_weighted_not_broadcast(self):
        title = "ETF inflow improves adoption"
        content = "Regulation risk and mining costs remain in focus."
        scores = live_infer._estimate_entity_sentiment(title, content, "coindesk")
        self.assertEqual(scores.shape[0], len(live_infer.FACTOR_NAMES))
        non_zero = np.flatnonzero(scores)
        self.assertGreaterEqual(len(non_zero), 1)
        self.assertLessEqual(float(np.abs(scores).sum()), 1.0)
        if len(non_zero) > 1:
            self.assertFalse(np.allclose(scores[non_zero], scores[non_zero[0]]))

    def test_factor_probs_helper_returns_normalized_distribution(self):
        probs, matched = compute_factor_probs_for_article(
            "ETF inflow surges",
            "Institutional fund demand rises after approval.",
            "coindesk",
            deterministic_prior=True,
        )
        self.assertTrue(matched)
        self.assertEqual(probs.shape[0], len(live_infer.FACTOR_NAMES))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertGreater(float(probs.max()), 0.15)

    def test_live_factor_distribution_uses_source_prior_when_no_keyword_match(self):
        probs = live_infer._infer_factor_distribution("General market update", "No obvious factor terms here.", "reuters")
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)
        self.assertGreater(float(probs.max()), float(1.0 / len(probs)))


if __name__ == "__main__":
    unittest.main()
