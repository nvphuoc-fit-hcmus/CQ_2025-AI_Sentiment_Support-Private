"""
Faithfulness evaluation for SAFE-Alert (PDF Section 4.4.2).

This script evaluates whether selected news truly affects prediction outputs.
It uses the current project interfaces from SAFEAlertNet and SAFEAlertDataset.

Metrics:
- Occlusion gap: max_prob(full) - max_prob(masked)
- Fidelity ratio: fraction of samples with gap > margin
- Direction change ratio after masking
- Insertion lift with synthetic positive article
- Attention-gap correlation (proxy fidelity alignment)

Usage:
  python app/v2/pipelines/eval_faithfulness_sec4p4p2.py \
    --model_path app/v2/artifacts/v2/safe_alert_BTCUSDT_1h_best.pt \
    --symbol BTCUSDT --horizon 1h --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

# Align imports with train_safe_alert.py behavior.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from models.safe_alert_net import SAFEAlertNet
from safe_alert_dataset import SAFEAlertDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FaithfulnessEvaluator:
    def __init__(self, model: SAFEAlertNet, horizon: str, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.horizon = horizon
        self.device = device
        self.model.eval()

    @staticmethod
    def _max_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1).max(dim=-1)[0]

    def _forward(self, market_feat, article_emb, article_mask, article_meta):
        return self.model(
            market_feat=market_feat,
            horizon=self.horizon,
            article_emb=article_emb,
            article_mask=article_mask,
            article_meta_vec=article_meta,
        )

    def _selected_mask(self, attn_weights: torch.Tensor, article_mask: torch.Tensor) -> torch.Tensor:
        k_h = self.model.K_h_map.get(self.horizon, 3)
        scores = attn_weights.masked_fill(~article_mask.bool(), float("-inf"))
        topk = min(k_h, scores.shape[1])
        if topk <= 0:
            return article_mask.bool()
        top_idx = scores.topk(topk, dim=1).indices
        keep = torch.zeros_like(article_mask, dtype=torch.bool)
        keep.scatter_(1, top_idx, True)
        return keep & article_mask.bool()

    def occlusion_test(self, test_loader: DataLoader, margin: float = 0.15, num_batches: int = 100) -> dict:
        gaps = []
        dir_changes = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break

                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)

                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                out_masked = self.model.forward_masked(
                    market_feat=market_feat,
                    horizon=self.horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta,
                )

                full_max = self._max_prob_from_logits(out_full["dir_logits"])
                masked_max = self._max_prob_from_logits(out_masked["dir_logits"])
                gap = (full_max - masked_max).cpu().numpy()
                gaps.extend(gap)

                dir_full = out_full["dir_logits"].argmax(dim=-1)
                dir_masked = out_masked["dir_logits"].argmax(dim=-1)
                dir_changes.extend((dir_full != dir_masked).float().cpu().numpy())

        gaps = np.asarray(gaps, dtype=np.float32)
        dir_changes = np.asarray(dir_changes, dtype=np.float32)
        if gaps.size == 0:
            return {
                "test_name": "occlusion",
                "num_samples": 0,
                "gap_mean": 0.0,
                "gap_std": 0.0,
                "gap_min": 0.0,
                "gap_max": 0.0,
                "gap_median": 0.0,
                "fidelity_ratio": 0.0,
                "direction_change_ratio": 0.0,
                "margin": margin,
            }

        return {
            "test_name": "occlusion",
            "num_samples": int(gaps.size),
            "gap_mean": float(gaps.mean()),
            "gap_std": float(gaps.std()),
            "gap_min": float(gaps.min()),
            "gap_max": float(gaps.max()),
            "gap_median": float(np.median(gaps)),
            "fidelity_ratio": float((gaps > margin).mean()),
            "direction_change_ratio": float(dir_changes.mean()) if dir_changes.size else 0.0,
            "margin": margin,
        }

    def insertion_test(self, test_loader: DataLoader, num_batches: int = 50) -> dict:
        lifts = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break

                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)

                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                full_max = self._max_prob_from_logits(out_full["dir_logits"])

                # Insert one synthetic positive article token per sample.
                synth_emb = article_emb.mean(dim=1, keepdim=True) + 0.05
                synth_meta = article_meta.mean(dim=1, keepdim=True)
                synth_mask = torch.ones(article_emb.shape[0], 1, dtype=article_mask.dtype, device=self.device)

                emb_with_synth = torch.cat([article_emb, synth_emb], dim=1)
                meta_with_synth = torch.cat([article_meta, synth_meta], dim=1)
                mask_with_synth = torch.cat([article_mask, synth_mask], dim=1)

                out_synth = self._forward(market_feat, emb_with_synth, mask_with_synth, meta_with_synth)
                synth_max = self._max_prob_from_logits(out_synth["dir_logits"])
                lifts.extend((synth_max - full_max).cpu().numpy())

        lifts = np.asarray(lifts, dtype=np.float32)
        if lifts.size == 0:
            return {
                "test_name": "insertion",
                "num_samples": 0,
                "insertion_lift_mean": 0.0,
                "insertion_lift_std": 0.0,
                "insertion_lift_median": 0.0,
                "insertion_ratio_positive": 0.0,
            }

        return {
            "test_name": "insertion",
            "num_samples": int(lifts.size),
            "insertion_lift_mean": float(lifts.mean()),
            "insertion_lift_std": float(lifts.std()),
            "insertion_lift_median": float(np.median(lifts)),
            "insertion_ratio_positive": float((lifts > 0).mean()),
        }

    def fidelity_metric(self, test_loader: DataLoader, num_batches: int = 100) -> dict:
        attn_values = []
        gap_values = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break

                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)

                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                out_masked = self.model.forward_masked(
                    market_feat=market_feat,
                    horizon=self.horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta,
                )

                attn = out_full.get("attn_weights")
                if attn is None:
                    continue

                full_max = self._max_prob_from_logits(out_full["dir_logits"]).cpu().numpy()
                masked_max = self._max_prob_from_logits(out_masked["dir_logits"]).cpu().numpy()
                gap = full_max - masked_max

                attn_np = attn.cpu().numpy()
                attn_values.extend(attn_np.flatten())
                gap_values.extend(np.repeat(gap, attn_np.shape[1]))

        attn_values = np.asarray(attn_values, dtype=np.float32)
        gap_values = np.asarray(gap_values, dtype=np.float32)
        if attn_values.size < 2:
            corr = 0.0
        else:
            corr = float(np.corrcoef(attn_values, gap_values)[0, 1])
            if np.isnan(corr):
                corr = 0.0

        return {
            "test_name": "fidelity_metric",
            "num_samples": int(attn_values.size),
            "fidelity_correlation": corr,
            "fidelity_pvalue": None,
        }

    def sufficiency_test(self, test_loader: DataLoader, num_batches: int = 100) -> dict:
        drops = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)
                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                attn = out_full.get("attn_weights")
                if attn is None:
                    continue
                keep_mask = self._selected_mask(attn, article_mask)
                out_keep = self._forward(market_feat, article_emb, keep_mask, article_meta)
                drops.extend((self._max_prob_from_logits(out_full["dir_logits"]) -
                              self._max_prob_from_logits(out_keep["dir_logits"])).cpu().numpy())
        drops = np.asarray(drops, dtype=np.float32)
        return {
            "test_name": "sufficiency",
            "num_samples": int(drops.size),
            "confidence_drop_mean": float(drops.mean()) if drops.size else 0.0,
            "confidence_drop_median": float(np.median(drops)) if drops.size else 0.0,
        }

    def comprehensiveness_test(self, test_loader: DataLoader, num_batches: int = 100) -> dict:
        drops = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)
                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                attn = out_full.get("attn_weights")
                if attn is None:
                    continue
                keep_mask = article_mask.bool() & ~self._selected_mask(attn, article_mask)
                out_drop = self._forward(market_feat, article_emb, keep_mask, article_meta)
                drops.extend((self._max_prob_from_logits(out_full["dir_logits"]) -
                              self._max_prob_from_logits(out_drop["dir_logits"])).cpu().numpy())
        drops = np.asarray(drops, dtype=np.float32)
        return {
            "test_name": "comprehensiveness",
            "num_samples": int(drops.size),
            "confidence_drop_mean": float(drops.mean()) if drops.size else 0.0,
            "confidence_drop_median": float(np.median(drops)) if drops.size else 0.0,
        }

    def factor_consistency_test(self, test_loader: DataLoader, num_batches: int = 100, top_p: int = 3) -> dict:
        scores = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= num_batches:
                    break
                market_feat = batch["market_features"].to(self.device)
                article_emb = batch["article_embeddings"].to(self.device)
                article_meta = batch["article_metadata"].to(self.device)
                article_mask = batch["article_mask"].to(self.device)
                out_full = self._forward(market_feat, article_emb, article_mask, article_meta)
                if out_full.get("attn_weights") is None or out_full.get("p_fac_all") is None:
                    continue
                noisy_market = market_feat + torch.randn_like(market_feat) * 0.01
                out_noisy = self._forward(noisy_market, article_emb, article_mask, article_meta)
                if out_noisy.get("attn_weights") is None or out_noisy.get("p_fac_all") is None:
                    continue

                for attn_a, p_fac_a, attn_b, p_fac_b in zip(
                    out_full["attn_weights"], out_full["p_fac_all"],
                    out_noisy["attn_weights"], out_noisy["p_fac_all"],
                ):
                    top_a = set(((attn_a.unsqueeze(-1) * p_fac_a).sum(dim=0)).topk(top_p).indices.tolist())
                    top_b = set(((attn_b.unsqueeze(-1) * p_fac_b).sum(dim=0)).topk(top_p).indices.tolist())
                    union = len(top_a | top_b)
                    scores.append(len(top_a & top_b) / union if union else 1.0)

        scores = np.asarray(scores, dtype=np.float32)
        return {
            "test_name": "factor_consistency",
            "num_samples": int(scores.size),
            "jaccard_mean": float(scores.mean()) if scores.size else 0.0,
            "jaccard_median": float(np.median(scores)) if scores.size else 0.0,
        }


def _load_checkpoint(model: SAFEAlertNet, model_path: Path, device: str) -> None:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            return
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            return
    # Raw state_dict fallback.
    model.load_state_dict(ckpt)


def _build_eval_loader(args: argparse.Namespace) -> DataLoader:
    candle_df = pd.read_csv(args.candles_csv)
    candle_df["timestamp"] = pd.to_datetime(candle_df["timestamp"], errors="coerce")
    if args.max_candle_ts:
        candle_df = candle_df[candle_df["timestamp"] <= pd.Timestamp(args.max_candle_ts)].reset_index(drop=True)

    articles_df = pd.read_csv(args.articles_csv)
    embeddings = np.load(args.embeddings_npy)

    precomputed_features = np.load(args.features_npy) if args.features_npy.exists() else None
    if precomputed_features is not None:
        if len(precomputed_features) < len(candle_df):
            raise ValueError(
                "features_precomputed.npy is shorter than filtered candles: "
                f"got {len(precomputed_features)}, expected at least {len(candle_df)}"
            )
        if len(precomputed_features) > len(candle_df):
            logger.warning(
                "features_precomputed.npy has extra rows (%d > %d); truncating to align with filtered candles",
                len(precomputed_features),
                len(candle_df),
            )
            precomputed_features = precomputed_features[:len(candle_df)]
    factor_labels = np.load(args.factor_labels_npy) if args.factor_labels_npy.exists() else None
    entity_sentiment = np.load(args.entity_sentiment_npy) if args.entity_sentiment_npy.exists() else None

    dataset = SAFEAlertDataset(
        candle_df=candle_df,
        article_embeddings=embeddings,
        article_meta=articles_df,
        article_to_candle={},
        symbol=args.symbol,
        horizon=args.horizon,
        precomputed_features=precomputed_features,
        factor_labels=factor_labels,
        entity_sentiment=entity_sentiment,
    )

    n = len(dataset)
    start = int(n * args.eval_tail_ratio)
    start = min(max(0, start), max(n - 1, 0))
    subset = Subset(dataset, list(range(start, n)))
    logger.info("Eval subset size: %d (tail ratio start=%.2f)", len(subset), args.eval_tail_ratio)
    return DataLoader(subset, batch_size=args.batch_size, shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithfulness evaluation for SAFE-Alert (Section 4.4.2)")
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--horizon", type=str, default="1h", choices=["1h", "4h", "24h"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--eval_tail_ratio", type=float, default=0.8)
    parser.add_argument("--max_candle_ts", type=str, default="")
    parser.add_argument("--output_json", type=Path, default=Path("app/v2/artifacts/v2/faithfulness_eval_sec4p4p2.json"))

    parser.add_argument("--candles_csv", type=Path, default=Path("training_data/candles_max.csv"))
    parser.add_argument("--articles_csv", type=Path, default=Path("training_data/articles_max.csv"))
    parser.add_argument("--embeddings_npy", type=Path, default=Path("training_data/btcusdt_article_embeddings_max.npy"))
    parser.add_argument("--features_npy", type=Path, default=Path("training_data/features_precomputed.npy"))
    parser.add_argument("--factor_labels_npy", type=Path, default=Path("training_data/article_factor_labels.npy"))
    parser.add_argument("--entity_sentiment_npy", type=Path, default=Path("training_data/article_entity_sentiment.npy"))

    args = parser.parse_args()
    default_data = Path("training_data")
    default_data_v2 = default_data / "v2"
    if default_data_v2.exists():
        # Auto-prefer v2 paths when user keeps defaults.
        if args.candles_csv == default_data / "candles_max.csv":
            v2_candles = default_data_v2 / f"{args.symbol.upper()}_{args.horizon}_ohlcv.csv"
            args.candles_csv = v2_candles if v2_candles.exists() else args.candles_csv
        if args.articles_csv == default_data / "articles_max.csv":
            args.articles_csv = default_data_v2 / "articles_max.csv"
        if args.embeddings_npy == default_data / "btcusdt_article_embeddings_max.npy":
            args.embeddings_npy = default_data_v2 / "btcusdt_article_embeddings_max.npy"
        if args.features_npy == default_data / "features_precomputed.npy":
            args.features_npy = default_data_v2 / "features_precomputed.npy"
        if args.factor_labels_npy == default_data / "article_factor_labels.npy":
            args.factor_labels_npy = default_data_v2 / "article_factor_labels.npy"
        if args.entity_sentiment_npy == default_data / "article_entity_sentiment.npy":
            args.entity_sentiment_npy = default_data_v2 / "article_entity_sentiment.npy"

    model = SAFEAlertNet(market_dim=63, has_news=True)
    _load_checkpoint(model, args.model_path, args.device)

    test_loader = _build_eval_loader(args)
    evaluator = FaithfulnessEvaluator(model, horizon=args.horizon, device=args.device)

    logger.info("Running occlusion test...")
    occlusion = evaluator.occlusion_test(test_loader, margin=args.margin, num_batches=args.num_batches)
    logger.info("Running insertion test...")
    insertion = evaluator.insertion_test(test_loader, num_batches=max(1, args.num_batches // 2))
    logger.info("Running sufficiency test...")
    sufficiency = evaluator.sufficiency_test(test_loader, num_batches=args.num_batches)
    logger.info("Running comprehensiveness test...")
    comprehensiveness = evaluator.comprehensiveness_test(test_loader, num_batches=args.num_batches)
    logger.info("Running factor consistency test...")
    factor_consistency = evaluator.factor_consistency_test(test_loader, num_batches=max(1, args.num_batches // 2))
    logger.info("Running fidelity correlation test...")
    fidelity = evaluator.fidelity_metric(test_loader, num_batches=args.num_batches)

    results = {
        "occlusion_test": occlusion,
        "insertion_test": insertion,
        "sufficiency_test": sufficiency,
        "comprehensiveness_test": comprehensiveness,
        "factor_consistency_test": factor_consistency,
        "fidelity_metric": fidelity,
        "horizon": args.horizon,
        "symbol": args.symbol,
        "model_path": str(args.model_path),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Faithfulness evaluation complete")
    print(f"Output: {args.output_json}")
    print(f"Occlusion gap mean: {occlusion['gap_mean']:.4f}")
    print(f"Occlusion fidelity ratio (> {args.margin}): {occlusion['fidelity_ratio']:.2%}")
    print(f"Direction change ratio: {occlusion['direction_change_ratio']:.2%}")
    print(f"Insertion lift mean: {insertion['insertion_lift_mean']:.4f}")
    print(f"Sufficiency drop mean: {sufficiency['confidence_drop_mean']:.4f}")
    print(f"Comprehensiveness drop mean: {comprehensiveness['confidence_drop_mean']:.4f}")
    print(f"Factor consistency Jaccard mean: {factor_consistency['jaccard_mean']:.4f}")
    print(f"Attention-gap correlation: {fidelity['fidelity_correlation']:.4f}")


if __name__ == "__main__":
    main()
