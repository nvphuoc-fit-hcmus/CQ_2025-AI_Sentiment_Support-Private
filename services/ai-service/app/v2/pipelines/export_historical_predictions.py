"""Export timestamped SAFE-Alert predictions with batched real-model inference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SERVICE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVICE_ROOT))

from app.v2.pipelines.backtest_safe_alert import (
    _resolve_data_dir,
    _select_candle_file,
    _softmax,
    load_model,
)
from app.v2.pipelines.safe_alert_dataset import SAFEAlertDataset


def load_dataset(data_dir: Path, symbol: str, horizon: str, model):
    candle_path = _select_candle_file(data_dir, symbol, horizon)
    candles = pd.read_csv(candle_path)
    if "timestamp" not in candles and "datetime" in candles:
        candles = candles.rename(columns={"datetime": "timestamp"})
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    articles = pd.read_csv(data_dir / "articles_max.csv")
    if "timestamp" not in articles and "published_at" in articles:
        articles = articles.rename(columns={"published_at": "timestamp"})
    articles["timestamp"] = pd.to_datetime(articles["timestamp"], utc=True)

    features_path = data_dir / f"features_precomputed_{horizon}.npy"
    if not features_path.exists():
        features_path = data_dir / "features_precomputed.npy"
    features = np.load(features_path) if features_path.exists() else None
    if features is not None and features.shape != (len(candles), 63):
        features = None

    bars = None
    if model.use_bar_sequences:
        bars_path = data_dir / f"market_bars_{horizon}.npz"
        if not bars_path.exists():
            bars_path = data_dir / "market_bars.npz"
        with np.load(bars_path) as npz:
            bars = {key: npz[key] for key in npz.files}

    def optional_npy(name):
        path = data_dir / name
        return np.load(path) if path.exists() else None

    dataset = SAFEAlertDataset(
        candle_df=candles,
        article_embeddings=np.load(data_dir / "btcusdt_article_embeddings_max.npy"),
        article_meta=articles,
        article_to_candle={},
        symbol=symbol,
        horizon=horizon,
        articles_per_candle=32,
        precomputed_features=features,
        factor_labels=optional_npy("article_factor_labels.npy"),
        entity_sentiment=optional_npy("article_entity_sentiment.npy"),
        article_novelty=optional_npy("article_novelty.npy"),
        market_bars=bars,
    )
    preproc = model._preprocessing_state or {}
    dataset.load_preprocessing_state(preproc)
    return candles, dataset


def export(args):
    data_dir = _resolve_data_dir(args.data_dir)
    model = load_model(args.artifact_dir, args.symbol, args.horizon, device="cpu")
    if model is None:
        raise RuntimeError("Checkpoint not found")
    candles, dataset = load_dataset(data_dir, args.symbol, args.horizon, model)

    n_test = int(len(dataset) * args.test_fraction)
    indices = list(range(len(dataset) - n_test, len(dataset)))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    temperature = float(getattr(model, "_ckpt_temperature", 1.0) or 1.0)
    rows = []
    offset = 0
    model.eval()
    with torch.inference_mode():
        for batch_no, batch in enumerate(loader, start=1):
            market_bars = batch.get("market_bars")
            output = model(
                market_feat=batch["market_features"],
                horizon=args.horizon,
                article_emb=batch["article_embeddings"],
                article_mask=batch["article_mask"],
                article_meta_vec=batch["article_metadata"],
                market_bars=market_bars,
                symbol=args.symbol,
            )
            logits = output["dir_logits"]
            probabilities = _softmax(logits.numpy() / max(temperature, 1e-6))
            directions = logits.argmax(dim=1).numpy()
            confidence = output.get("confidence", logits.softmax(dim=1).max(dim=1).values)
            confidence = confidence.reshape(-1).numpy()
            returns = batch["return"].reshape(-1).numpy()
            for local_idx, direction_idx in enumerate(directions):
                dataset_idx = indices[offset + local_idx]
                candle_idx = dataset.valid_idx[dataset_idx]
                probs = probabilities[local_idx]
                rows.append({
                    "time": pd.Timestamp(candles.iloc[candle_idx]["timestamp"]).isoformat(),
                    "symbol": args.symbol.upper(),
                    "horizon": args.horizon,
                    "direction": ("DOWN", "NEUTRAL", "UP")[int(direction_idx)],
                    "confidence": float(confidence[local_idx]),
                    "max_probability": float(probs.max()),
                    "probabilities": {
                        "DOWN": float(probs[0]),
                        "NEUTRAL": float(probs[1]),
                        "UP": float(probs[2]),
                    },
                    "return": float(returns[local_idx]),
                })
            offset += len(directions)
            if batch_no % 25 == 0:
                print(f"[{args.horizon}] {offset}/{len(indices)} predictions", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] Exported {len(rows)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--horizon", choices=["1h", "4h"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    export(parser.parse_args())
