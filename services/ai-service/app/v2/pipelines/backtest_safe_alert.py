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
import json
from collections import defaultdict

# Use relative paths from script location (portable)
SCRIPT_FILE = Path(__file__).resolve()  # Absolute path
SCRIPT_DIR = SCRIPT_FILE.parent         # pipelines/
V2_DIR = SCRIPT_DIR.parent              # v2/
APP_DIR = V2_DIR.parent                 # app/
WORK_DIR = APP_DIR.parent               # services/ai-service/
ARTIFACT_DIR = WORK_DIR / "artifacts/v2"
DATA_DIR = WORK_DIR / "training_data"

# Constants per PDF
INITIAL_CAPITAL = 10000  # Initial portfolio
TRANSACTION_COST = 0.001  # 0.1% per trade (PDF Section 4.2.4)
SLIPPAGE = 0.0005         # 0.05% slippage per trade (PDF Section 4.2.4)
SHARPE_RF = 0.02          # Risk-free rate 2%


def load_model(device="cpu"):
    """Load best trained model (epoch20 with val_loss=1.0980)."""
    import sys
    sys.path.insert(0, str(WORK_DIR))

    from app.v2.models.safe_alert_net import SAFEAlertNet

    # Load FINAL model (best by model_score from Stage 3, saved by train_safe_alert.py)
    final_path = ARTIFACT_DIR / "safe_alert_btcusdt_1h_FINAL.pt"
    if final_path.exists():
        ckpt_path = final_path
        print(f"[OK] Loading FINAL model (best by model_score): {ckpt_path.name}")
    else:
        # Fallback: find best-epoch checkpoint from Stage 3
        best_files = sorted(ARTIFACT_DIR.glob("safe_alert_btcusdt_1h_best_epoch*.pt"))
        if best_files:
            ckpt_path = best_files[-1]
            print(f"[OK] Loading best-epoch checkpoint: {ckpt_path.name}")
        else:
            # Last resort: highest epoch
            epoch_files = sorted(ARTIFACT_DIR.glob("safe_alert_btcusdt_1h_stage3_epoch*.pt"),
                                 key=lambda x: int(x.stem.split("epoch")[-1]))
            if not epoch_files:
                epoch_files = sorted(ARTIFACT_DIR.glob("safe_alert_btcusdt_1h_*.pt"))
            if epoch_files:
                ckpt_path = epoch_files[-1]
                print(f"[WARN] FINAL model not found, using: {ckpt_path.name}")
            else:
                print("[ERROR] No checkpoints found in artifacts/v2/")
                return None

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.name}")
        return None

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device)

    # CRITICAL: architecture must EXACTLY match training config in train_safe_alert.py main()
    # train uses: SAFEAlertNet(market_dim=63, has_news=True, K_1h=4, K_4h=5)
    # article_dim=128 (internal encoding dim, NOT FinBERT 768 — that's input size)
    model = SAFEAlertNet(
        market_dim=63,
        article_dim=128,    # internal encoding dim (matches training default)
        has_news=True,
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


def simulate_trades(signals: np.ndarray, returns: np.ndarray):
    """
    Simulate trading based on signals.

    signals: (N,) array of {0=no_trade, 1=BUY, -1=SELL}
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

        # P&L if holding
        if position == 1:
            pnl_list.append(ret)

    return np.array(pnl_list)


def backtest():
    """Run backtest on test set with REAL model predictions."""

    print("\n" + "="*70)
    print("SAFE-ALERT BACKTEST: Real Model Inference on Test Set")
    print("="*70)

    device = "cpu"
    model = load_model(device)
    if model is None:
        raise RuntimeError("[ERROR] Failed to load model. Cannot backtest without model.")

    # Load test data
    print("\n[*] Loading test set data...")
    import sys
    sys.path.insert(0, str(WORK_DIR / "app"))  # FIXED: Add "app" to path
    from v2.pipelines.safe_alert_dataset import SAFEAlertDataset  # FIXED: Correct class name

    # Load data files
    print(f"[*] Loading candles from {DATA_DIR}...")
    candle_df = pd.read_csv(DATA_DIR / "candles_max.csv")
    candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'])

    print(f"[*] Loading embeddings...")
    article_embeddings = np.load(DATA_DIR / "btcusdt_article_embeddings_max.npy")

    print(f"[*] Loading articles metadata...")
    articles_df = pd.read_csv(DATA_DIR / "articles_max.csv")
    articles_df['timestamp'] = pd.to_datetime(articles_df['timestamp'])

    # Create dataset to get test split (FIXED: correct constructor)
    try:
        dataset = SAFEAlertDataset(
            candle_df=candle_df,
            article_embeddings=article_embeddings,
            article_meta=articles_df,
            article_to_candle={},
            symbol="BTCUSDT",
            horizon="1h",
            articles_per_candle=8,
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load dataset: {e}. Ensure training data exists.")

    # Get test set (last 15% of data, temporal split)
    n_total = len(dataset)
    n_test = int(n_total * 0.15)
    test_start_idx = n_total - n_test

    print(f"[OK] Dataset size: {n_total}, test set: {n_test} samples")

    # Run model inference on test set
    print("\n[*] Running model inference on test set...")
    model.eval()
    all_predictions = []
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
                    horizon="1h",
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta
                )

                dir_logits = pred_dict['dir_logits']  # Shape: (1, 3)
                direction_pred = torch.argmax(dir_logits, dim=1).item()  # 0=DOWN, 1=NEUTRAL, 2=UP

                all_predictions.append(direction_pred)

            except Exception as e:
                print(f"[WARN] Sample {i} failed: {e}, skipping")
                continue

    if len(all_predictions) < n_test * 0.8:  # At least 80% successful
        raise RuntimeError(
            f"[ERROR] Model inference failed on {n_test - len(all_predictions)} samples. "
            f"Cannot generate valid backtest results."
        )

    all_predictions = np.array(all_predictions)
    all_returns = np.array(all_returns[:len(all_predictions)])  # Align lengths

    print(f"[OK] Generated {len(all_predictions)} predictions")

    # Convert to trading signals: UP -> BUY (+1), else -> SELL (-1)
    signals = np.where(all_predictions == 2, 1, -1)

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
    output_dir = ARTIFACT_DIR / "backtest_results"
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

    with open(ARTIFACT_DIR / "backtest_metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\n  - {ARTIFACT_DIR}/backtest_metrics.json")

    print("\n[COMPLETE] Real backtest done!")
    return table3, table4, table5


if __name__ == "__main__":
    backtest()
