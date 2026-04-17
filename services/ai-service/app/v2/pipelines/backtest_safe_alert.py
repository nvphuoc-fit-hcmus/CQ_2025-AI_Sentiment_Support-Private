"""
Backtest SAFE-Alert trained model & generate Bảng 3/4/5 comparison tables.

Implements walk-forward validation on test set:
- Test set: 6,959 samples (15% of 46,387)
- Metrics: Sharpe, Sortino, Calmar, max_dd, win_rate, PnL
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
import json
from collections import defaultdict

# Use relative paths from script location (portable)
SCRIPT_FILE = Path(__file__).resolve()  # Absolute path
SCRIPT_DIR = SCRIPT_FILE.parent         # pipelines/
V2_DIR = SCRIPT_DIR.parent              # v2/
APP_DIR = V2_DIR.parent                 # app/
WORK_DIR = APP_DIR.parent               # services/ai-service/
DEFAULT_ARTIFACT_DIR = WORK_DIR / "artifacts/v2"
DEFAULT_DATA_DIR = WORK_DIR / "training_data"
DEFAULT_DATA_DIR_V2 = DEFAULT_DATA_DIR / "v2"

# Constants per PDF
INITIAL_CAPITAL = 10000  # Initial portfolio
TRANSACTION_COST = 0.001  # 0.1% per trade (PDF Section 4.2.4)
SLIPPAGE = 0.0005         # 0.05% slippage per trade (PDF Section 4.2.4)
SHARPE_RF = 0.02          # Risk-free rate 2%


def _find_checkpoint(artifact_dir: Path, symbol: str, horizon: str) -> Optional[Path]:
    symbol = symbol.lower()
    horizon = horizon.lower()

    candidates = []
    candidates.append(artifact_dir / f"safe_alert_{symbol}_{horizon}_FINAL.pt")
    candidates.append(artifact_dir / f"safe_alert_{horizon}_FINAL.pt")
    candidates.extend(sorted(artifact_dir.glob(f"safe_alert_{horizon}_best_epoch*.pt")))
    candidates.extend(sorted(artifact_dir.glob("safe_alert_*_best_epoch*.pt")))
    candidates.extend(sorted(artifact_dir.glob(f"safe_alert_{symbol}_{horizon}_stage3_epoch*.pt")))
    candidates.extend(sorted(artifact_dir.glob("safe_alert_*_stage3_epoch*.pt")))
    candidates.extend(sorted(artifact_dir.glob("safe_alert_*.pt")))

    for path in candidates:
        if path.exists():
            return path
    return None


def load_model(artifact_dir: Path, symbol: str, horizon: str, device: str = "cpu"):
    """Load best trained model from the chosen artifact directory."""
    import sys
    sys.path.insert(0, str(WORK_DIR))

    from app.v2.models.safe_alert_net import SAFEAlertNet

    ckpt_path = _find_checkpoint(artifact_dir, symbol, horizon)
    if ckpt_path is None:
        print(f"[ERROR] No checkpoints found in {artifact_dir}")
        return None
    print(f"[OK] Loading checkpoint: {ckpt_path.name}")

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.name}")
        return None

    # Load weights
    # PyTorch >=2.6 defaults to weights_only=True. Use weights_only=False for trusted local checkpoints.
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)

    # CRITICAL: architecture must EXACTLY match training config in train_safe_alert.py main()
    # train uses: SAFEAlertNet(market_dim=63, has_news=True, K_15m=3, K_1h=4, K_4h=5)
    # article_dim=128 (internal encoding dim, NOT FinBERT 768 — that's input size)
    model = SAFEAlertNet(
        market_dim=63,
        article_dim=128,    # internal encoding dim (matches training default)
        has_news=True,
        K_15m=3,            # matches train_safe_alert.py main() K_15m=3
        K_1h=4,             # matches train_safe_alert.py main() K_1h=4
        K_4h=5,             # matches train_safe_alert.py main() K_4h=5
    )

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"[OK] Model loaded ({ckpt_path.name})")
    return model


def compute_metrics(returns: np.ndarray, daily_returns: np.ndarray = None):
    """Compute backtest metrics per PDF specs."""

    total_return = np.prod(1 + returns) - 1 if len(returns) > 0 else 0
    annual_return = (1 + total_return) ** (252 / max(len(returns), 1)) - 1

    # Volatility
    std_return = np.std(returns) if len(returns) > 0 else 0
    annual_vol = std_return * np.sqrt(252)

    # Sharpe
    sharpe = (annual_return - SHARPE_RF) / (annual_vol + 1e-6) if annual_vol > 0 else 0

    # Sortino (downside deviation)
    downside = np.minimum(returns, 0)
    downside_std = np.std(downside) if len(downside) > 0 else 0
    downside_vol = downside_std * np.sqrt(252)
    sortino = (annual_return - SHARPE_RF) / (downside_vol + 1e-6) if downside_vol > 0 else 0

    # Cumulative returns & max drawdown
    cumret = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumret)
    drawdown = (cumret - running_max) / running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

    # Calmar (annual return / abs(max_dd))
    calmar = annual_return / (abs(max_dd) + 1e-6)

    # Win rate
    win_rate = np.mean(returns > 0) if len(returns) > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
    }


def _load_policy(artifact_dir: Path, symbol: str, horizon: str, policy_json: Optional[Path]) -> Optional[dict]:
    if policy_json is not None and policy_json.exists():
        with open(policy_json, "r") as f:
            return json.load(f)

    # Try common policy filenames in artifact_dir
    candidates = [
        artifact_dir / f"safe_alert_policy_{symbol.lower()}_{horizon}.json",
        artifact_dir / f"safe_alert_{symbol.lower()}_{horizon}_policy.json",
        artifact_dir / "safe_alert_policy_btcusdt_1h.json",
        artifact_dir / "safe_alert_btcusdt_1h_policy.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def simulate_trades(signals: np.ndarray, returns: np.ndarray):
    """
    Simulate trading based on signals.

    signals: (N,) array of {1=BUY, 0=HOLD, -1=SELL}
    returns: (N,) array of next-candle returns
    """
    position = 0  # 0=flat, 1=long
    pnl_list = []

    for i, sig in enumerate(signals):
        ret = returns[i] if i < len(returns) else 0

        # Entry signal
        if sig == 1 and position == 0:
            position = 1
            pnl_list.append(-(TRANSACTION_COST + SLIPPAGE))  # Entry cost + slippage
        # Exit signal
        elif sig == -1 and position == 1:
            position = 0
            pnl_list.append(-(TRANSACTION_COST + SLIPPAGE))  # Exit cost + slippage

        # Mark-to-market P&L while holding a long position
        if position == 1:
            pnl_list.append(ret)
        elif sig == 0:
            pnl_list.append(0.0)

    return np.array(pnl_list)


def _resolve_data_dir(data_dir: Path) -> Path:
    if data_dir == DEFAULT_DATA_DIR and DEFAULT_DATA_DIR_V2.exists():
        return DEFAULT_DATA_DIR_V2
    return data_dir


def _select_candle_file(data_dir: Path, symbol: str, horizon: str) -> Path:
    symbol_upper = symbol.upper()
    if horizon.lower() == "1h":
        v2 = data_dir / f"{symbol_upper}_1h_ohlcv.csv"
        if v2.exists():
            return v2
    if horizon.lower() == "4h":
        v2 = data_dir / f"{symbol_upper}_4h_ohlcv.csv"
        if v2.exists():
            return v2
    # Fallbacks
    candles_max = data_dir / "candles_max.csv"
    if candles_max.exists():
        return candles_max
    return data_dir / f"{symbol.lower()}_training_dataset_v2.csv"


def backtest(artifact_dir: Path, data_dir: Path, symbol: str, horizon: str, policy_json: Optional[Path] = None):
    """Run backtest on test set with REAL model predictions."""

    print("\n" + "="*70)
    print("SAFE-ALERT BACKTEST: Real Model Inference on Test Set")
    print("="*70)

    device = "cpu"
    model = load_model(artifact_dir, symbol, horizon, device=device)
    if model is None:
        raise RuntimeError("[ERROR] Failed to load model. Cannot backtest without model.")

    data_dir = _resolve_data_dir(data_dir)

    # Load test data
    print("\n[*] Loading test set data...")
    import sys
    sys.path.insert(0, str(WORK_DIR / "app"))  # FIXED: Add "app" to path
    from v2.pipelines.safe_alert_dataset import SAFEAlertDataset  # FIXED: Correct class name

    # Load data files
    candle_path = _select_candle_file(data_dir, symbol, horizon)
    print(f"[*] Loading candles from {candle_path}...")
    candle_df = pd.read_csv(candle_path)
    candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'])

    print(f"[*] Loading embeddings...")
    article_embeddings = np.load(data_dir / "btcusdt_article_embeddings_max.npy")

    print(f"[*] Loading articles metadata...")
    articles_df = pd.read_csv(data_dir / "articles_max.csv")
    articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'])

    factor_labels = None
    factor_path = data_dir / "article_factor_labels.npy"
    if factor_path.exists():
        print("[*] Loading factor labels...")
        factor_labels = np.load(factor_path)

    entity_sentiment = None
    entity_path = data_dir / "article_entity_sentiment.npy"
    if entity_path.exists():
        print("[*] Loading entity sentiment (FSA)...")
        entity_sentiment = np.load(entity_path)

    # Create dataset to get test split (FIXED: correct constructor)
    try:
        dataset = SAFEAlertDataset(
            candle_df=candle_df,
            article_embeddings=article_embeddings,
            article_meta=articles_df,
            article_to_candle={},
            factor_labels=factor_labels,
            entity_sentiment=entity_sentiment,
            symbol=symbol,
            horizon=horizon,
            articles_per_candle=8,
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load dataset: {e}. Ensure training data exists.")

    # Get test set (last 15% of data, temporal split)
    n_total = len(dataset)
    n_test = int(n_total * 0.15)
    test_start_idx = n_total - n_test

    print(f"[OK] Dataset size: {n_total}, test set: {n_test} samples")

    policy = _load_policy(artifact_dir, symbol, horizon, policy_json)
    if policy:
        print(f"[OK] Loaded policy: tau={policy.get('tau')} gamma={policy.get('gamma')} "
              f"temp={policy.get('temperature', 1.0)}")
    else:
        print("[WARN] No policy JSON found; falling back to always-alert signals.")

    # Run model inference on test set
    print("\n[*] Running model inference on test set...")
    model.eval()
    all_predictions = []
    all_confidence = []
    all_max_prob = []
    all_returns = []

    with torch.no_grad():
        for i in range(test_start_idx, n_total):
            try:
                sample = dataset[i]

                # Extract return (actual market return from next candle)
                ret = float(sample['return'].item() if isinstance(sample['return'], torch.Tensor) else sample['return'])
                all_returns.append(ret)

                # Run model forward pass with correct arguments
                # market_features: (63,) → (1, 63)
                # article_embeddings: (K, 768) → (1, K, 768)
                # article_metadata: (K, 4) → (1, K, 4)
                # article_mask: (K,) → (1, K)
                market_feat = sample['market_features'].unsqueeze(0).to(device)  # (1, 63)
                article_emb = sample['article_embeddings'].unsqueeze(0).to(device)  # (1, K, 768)
                article_mask = sample['article_mask'].unsqueeze(0).to(device)  # (1, K)
                article_meta = sample['article_metadata'].unsqueeze(0).to(device)  # (1, K, 4)

                # Get prediction
                pred_dict = model(
                    market_feat=market_feat,
                    horizon=horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta
                )

                dir_logits = pred_dict['dir_logits']  # Shape: (1, 3)
                direction_pred = torch.argmax(dir_logits, dim=1).item()  # 0=DOWN, 1=NEUTRAL, 2=UP
                confidence = float(pred_dict.get("confidence", torch.tensor([0.0])).view(-1)[0].item())

                temperature = float(policy.get("temperature", 1.0)) if policy else 1.0
                logits_np = dir_logits.detach().cpu().numpy() / max(temperature, 1e-6)
                probs = _softmax(logits_np)[0]
                max_prob = float(np.max(probs))

                all_predictions.append(direction_pred)
                all_confidence.append(confidence)
                all_max_prob.append(max_prob)

            except Exception as e:
                print(f"[WARN] Sample {i} failed: {e}, skipping")
                continue

    if len(all_predictions) < n_test * 0.8:  # At least 80% successful
        raise RuntimeError(
            f"[ERROR] Model inference failed on {n_test - len(all_predictions)} samples. "
            f"Cannot generate valid backtest results."
        )

    all_predictions = np.array(all_predictions)
    all_confidence = np.array(all_confidence)
    all_max_prob = np.array(all_max_prob)
    all_returns = np.array(all_returns[:len(all_predictions)])  # Align lengths

    print(f"[OK] Generated {len(all_predictions)} predictions")

    # Convert model classes to trading signals with policy thresholds (Eq.26)
    if policy:
        tau = float(policy.get("tau", 1.0))
        gamma = float(policy.get("gamma", 1.0))
        alert_mask = (all_confidence >= tau) & (all_max_prob >= gamma)
        signals = np.where(alert_mask, np.where(all_predictions == 2, 1, np.where(all_predictions == 0, -1, 0)), 0)
        print(f"[OK] Alert coverage: {alert_mask.mean():.4f} (tau={tau:.4f}, gamma={gamma:.4f})")
    else:
        signals = np.where(all_predictions == 2, 1, np.where(all_predictions == 0, -1, 0))

    # Strategy PnL with real returns
    strategy_pnl = simulate_trades(signals, all_returns)

    # Buy-hold baseline with same real returns
    buyhold_pnl = all_returns

    # Compute metrics on REAL data
    strategy_metrics = compute_metrics(strategy_pnl)
    buyhold_metrics = compute_metrics(buyhold_pnl)

    print("\n[STRATEGY METRICS] (Real Model Predictions)")
    for k, v in strategy_metrics.items():
        print(f"  {k:15s}: {v:10.4f}")

    print("\n[BUY-HOLD BASELINE] (Same Real Returns)")
    for k, v in buyhold_metrics.items():
        print(f"  {k:15s}: {v:10.4f}")

    # Table 3: Comparison
    table3 = pd.DataFrame({
        "SAFE-Alert": strategy_metrics,
        "Buy-Hold": buyhold_metrics,
    })

    # Tables 4 & 5: Mark as ABLATION REQUIRED (cannot generate without separate models)
    print("\n[NOTE] Tables 4 & 5 require separate ablated models.")
    print("[NOTE] To generate ablation tables, use: python run_ablation.py")

    table4 = pd.DataFrame({
        "Note": ["Ablation study requires training models with components removed"],
        "Status": ["Pending - use run_ablation.py"]
    })

    table5 = pd.DataFrame({
        "Note": ["Multi-timeframe study requires separate 4h model"],
        "Status": ["Pending - train 4h model separately"]
    })

    # Save tables
    output_dir = artifact_dir / "backtest_results"
    output_dir.mkdir(parents=True, exist_ok=True)  # ✅ Added parents=True

    table3.to_csv(output_dir / "table3_comparison.csv")
    table4.to_csv(output_dir / "table4_ablation.csv")
    table5.to_csv(output_dir / "table5_timeframes.csv")

    print("\n[OK] Real backtest results saved:")
    print(f"  - table3_comparison.csv (REAL model predictions)")
    print(f"  - table4_ablation.csv (awaiting ablation runs)")
    print(f"  - table5_timeframes.csv (awaiting 4h model)")

    # Save metrics JSON
    metrics_json = {
        "test_size": len(all_predictions),
        "strategy_metrics": {k: float(v) for k, v in strategy_metrics.items()},
        "buyhold_metrics": {k: float(v) for k, v in buyhold_metrics.items()},
        "data_source": "Real model inference on test set",
        "signal_distribution": {
            "UP": int(np.sum(all_predictions == 2)),
            "NEUTRAL": int(np.sum(all_predictions == 1)),
            "DOWN": int(np.sum(all_predictions == 0)),
        }
    }

    with open(artifact_dir / "backtest_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\n  - {artifact_dir}/backtest_metrics.json")

    print("\n[COMPLETE] Real backtest done!")
    return table3, table4, table5


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Backtest SAFE-Alert on test split (real inference)")
    parser.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT_DIR,
                        help="Artifacts directory containing trained checkpoints")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Training data directory (candles/articles/embeddings)")
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Symbol used for checkpoint name resolution")
    parser.add_argument("--horizon", type=str, default="1h",
                        help="Horizon used for checkpoint name resolution")
    parser.add_argument("--policy_json", type=Path, default=None,
                        help="Optional policy JSON with tau/gamma/temperature")

    args = parser.parse_args()
    backtest(args.artifact_dir, args.data_dir, args.symbol, args.horizon, args.policy_json)
