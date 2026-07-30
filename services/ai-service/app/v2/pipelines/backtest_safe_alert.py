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

# Annualization for 1-hour crypto candles: 24 × 365 = 8,760 periods/year.
# Previously this module used 252 (equities trading-day convention), which
# understated annualized Sharpe/Sortino by √(8760/252) ≈ 5.9× for the
# hourly-returns stream on which we actually run backtests.
PERIODS_PER_YEAR = 8760


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
    candidates.extend(sorted(artifact_dir.glob(f"fold_*/safe_alert_{horizon}_FINAL.pt")))
    candidates.extend(sorted(artifact_dir.glob(f"fold_*/safe_alert_{horizon}_best_epoch*.pt")))

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
    # Session 26: explicit K_24h + n_factors to prevent silent drift from
    # default changes. K values match the training-time YAML top_k block.
    # If the checkpoint was trained with custom K, load via the checkpoint's
    # `K_h` field (single horizon value) — future work: save full K_h_map.
    from app.v2.models.safe_alert_net import FACTOR_CLASSES as _FCL
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    has_bar_encoder = any(str(k).startswith("market_enc.bar_encoders.") for k in state)
    model = SAFEAlertNet(
        market_dim=63,
        article_dim=128,    # internal encoding dim (matches training default)
        has_news=True,
        K_15m=3, K_1h=4, K_4h=5, K_24h=8,
        n_factors=_FCL,
        use_bar_sequences=has_bar_encoder,
        market_input_mode="hybrid" if has_bar_encoder else "scalar",
        bar_seq_len=20,
        bar_feat_dim=10,
    )

    # strict=False keeps backtests tolerant to checkpoint schema drift while
    # still loading every matching layer. Any mismatch is logged below.
    if "model_state" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        import warnings as _w
        _w.warn(
            f"load_state_dict had missing={list(missing)[:5]} unexpected={list(unexpected)[:5]} "
            "(showing first 5). Checkpoint schema may differ from current model.",
            RuntimeWarning, stacklevel=2,
        )

    model.to(device)
    model.eval()

    # P0 #1: stash preprocessing state on the model so the backtest loop can
    # apply the same z-score normalization the training pipeline used. Prior
    # to this fix backtest silently fed raw features into a model trained on
    # normalized features, invalidating every metric.
    preproc = checkpoint.get("preprocessing_state") if isinstance(checkpoint, dict) else None
    if preproc is not None and preproc.get("market_feature_mean") is not None:
        model._market_feature_mean = torch.as_tensor(preproc["market_feature_mean"]).float().to(device)
        model._market_feature_std  = torch.as_tensor(preproc["market_feature_std"]).float().to(device)
        model._market_feature_clip = float(preproc.get("market_feature_clip", 8.0))
        print(f"[OK] Loaded preprocessing state from checkpoint "
              f"(clip={model._market_feature_clip}, mean_norm={model._market_feature_mean.abs().mean():.4f})")
    else:
        model._market_feature_mean = None
        model._market_feature_std  = None
        model._market_feature_clip = 8.0
        import warnings as _warnings
        _warnings.warn(
            f"[backtest] Checkpoint {ckpt_path.name} has NO preprocessing_state "
            f"(pre-P0#1 artifact). Backtest will feed RAW features to a model "
            f"trained on z-score normalized features. Retrain to refresh.",
            RuntimeWarning, stacklevel=2,
        )
    model._ckpt_temperature = float(
        checkpoint.get("temperature", 1.0) if isinstance(checkpoint, dict) else 1.0
    )
    model._preprocessing_state = preproc

    print(f"[OK] Model loaded ({ckpt_path.name})")
    return model


def compute_metrics(returns: np.ndarray, daily_returns: np.ndarray = None):
    """Compute backtest metrics per PDF specs.

    Fixes applied here (all three affected every reported backtest number):
      1. Sortino downside-deviation formula now matches the Sortino & Price (1994)
         definition (sqrt of mean squared negative returns), not std() of a
         zero-padded negative sequence which zeroed-out the mean and shrank the
         denominator by ~√2 — inflating Sortino.
      2. Annualisation factor switched from 252 (equities trading-day convention)
         to ``PERIODS_PER_YEAR`` = 8,760 — correct for the hourly crypto returns
         stream this module is actually fed.
      3. ``np.std`` defaults to population std (ddof=0); explicit ``ddof=1``
         gives the sample std expected in finance (Bessel correction).
    """
    if len(returns) == 0:
        return {
            "total_return": 0.0, "annual_return": 0.0, "annual_vol": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "calmar": 0.0,
            "win_rate": 0.0,
        }

    total_return  = float(np.prod(1.0 + returns) - 1.0)
    annual_return = (1.0 + total_return) ** (PERIODS_PER_YEAR / max(len(returns), 1)) - 1.0

    # Volatility (sample std with Bessel correction when sample is non-trivial).
    std_return = float(np.std(returns, ddof=1 if len(returns) > 1 else 0))
    annual_vol = std_return * np.sqrt(PERIODS_PER_YEAR)

    # Sharpe
    sharpe = (annual_return - SHARPE_RF) / (annual_vol + 1e-6) if annual_vol > 0 else 0.0

    # Sortino — use the √(E[min(r,0)²]) definition (Sortino & Price 1994).
    # ``np.std(np.minimum(returns, 0))`` is WRONG: the zero-pad inflates the
    # mean toward 0 and the resulting std is √2 smaller than the true downside
    # deviation, giving an optimistic Sortino.
    neg = np.minimum(returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(neg * neg)))
    downside_vol = downside_dev * np.sqrt(PERIODS_PER_YEAR)
    sortino = (annual_return - SHARPE_RF) / (downside_vol + 1e-6) if downside_vol > 0 else 0.0

    # Cumulative returns & max drawdown
    cumret = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumret)
    drawdown = (cumret - running_max) / running_max
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

    # Calmar (annual return / abs(max_dd))
    calmar = annual_return / (abs(max_dd) + 1e-6)

    # Win rate (exclude zero-return periods so flat HOLD days don't dilute).
    nonzero = returns[returns != 0.0]
    win_rate = float(np.mean(nonzero > 0)) if len(nonzero) > 0 else 0.0

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


def walk_forward_metrics(returns: np.ndarray, n_windows: int = 5) -> dict:
    """Walk-forward (rolling-window) performance evaluation per PDF Section 4.2.3.

    Splits the PnL stream into n_windows non-overlapping consecutive chunks,
    computes standard metrics on each chunk, and reports mean±std across
    chunks plus the per-window table. This addresses the paper's requirement
    for temporal robustness evaluation — a single-window Sharpe can be
    inflated by a favourable regime, while mean±std across windows reveals
    whether performance is stable over time.

    Non-overlapping windows preserve the independence assumption used for the
    Diebold–Mariano-style std; overlapping rolling windows would bias it.

    Args:
        returns: 1-D array of per-period PnL values.
        n_windows: Number of equal-size chunks (default 5 — 20% per chunk).

    Returns:
        dict with keys:
          - ``per_window``: list of dicts (one compute_metrics() output per chunk)
          - ``summary``: {metric_mean, metric_std} for each metric
          - ``n_windows``, ``window_size``
    """
    if len(returns) == 0 or n_windows <= 0:
        return {"per_window": [], "summary": {}, "n_windows": 0, "window_size": 0}

    # Minimum window length — if data is too short, reduce n_windows rather
    # than producing degenerate single-sample windows.
    effective_windows = min(n_windows, max(1, len(returns) // 20))
    window_size = len(returns) // effective_windows

    per_window = []
    for i in range(effective_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < effective_windows - 1 else len(returns)
        chunk = returns[start:end]
        metrics = compute_metrics(chunk)
        metrics["window_idx"] = i
        metrics["window_start"] = int(start)
        metrics["window_end"] = int(end)
        per_window.append(metrics)

    # Aggregate mean / std across windows for each scalar metric.
    if per_window:
        metric_keys = [k for k in per_window[0].keys()
                       if k not in {"window_idx", "window_start", "window_end"}]
        summary = {}
        for k in metric_keys:
            values = np.array([m[k] for m in per_window], dtype=np.float64)
            summary[f"{k}_mean"] = float(np.mean(values))
            summary[f"{k}_std"] = float(np.std(values))
    else:
        summary = {}

    return {
        "per_window": per_window,
        "summary": summary,
        "n_windows": effective_windows,
        "window_size": int(window_size),
    }


def _validate_policy(policy: dict, source: str) -> dict:
    """Sanity-check a policy JSON. Rejects out-of-range τ/γ/T — a corrupt policy
    silently produces either zero alerts (τ>1) or all alerts (τ<0), either of
    which invalidates the entire backtest."""
    if not isinstance(policy, dict):
        raise ValueError(f"Policy from {source} must be a JSON object, got {type(policy).__name__}")
    tau = float(policy.get("tau", policy.get("tau_h", 0.5)))
    gamma = float(policy.get("gamma", policy.get("gamma_h", 0.5)))
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"Policy {source}: tau={tau} outside [0,1] — corrupt policy file.")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"Policy {source}: gamma={gamma} outside [0,1] — corrupt policy file.")
    T = float(policy.get("temperature", 1.0))
    if not (0.05 <= T <= 20.0):
        raise ValueError(f"Policy {source}: temperature={T} outside [0.05,20] — corrupt policy file.")
    return policy


def _load_policy(artifact_dir: Path, symbol: str, horizon: str, policy_json: Optional[Path]) -> Optional[dict]:
    if policy_json is not None and policy_json.exists():
        with open(policy_json, "r") as f:
            return _validate_policy(json.load(f), str(policy_json))

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
                return _validate_policy(json.load(f), str(path))
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def simulate_trades(signals: np.ndarray, returns: np.ndarray, allow_short: bool = True):
    """PnL simulator with consistent long+short semantics.

    signals: (N,) {+1=LONG signal, 0=HOLD, -1=SHORT signal}
    returns: (N,) next-period market returns aligned with signals
    allow_short:
        • True  (default) — opens SHORT on sig=-1, earning -ret while short.
                             Matches metrics_safe_alert.mini_backtest, which is
                             what early stopping / policy search use. This
                             keeps validation and final backtest on the SAME
                             strategy (was Bug #4 before fix).
        • False — long-only fallback: sig=-1 CLOSES any open long but never
                  opens a short. Useful for spot-only deployments.

    Returns:
        pnl: (N,) per-step net PnL = market return earned that step by the
             active position minus entry/exit transaction costs when the
             position changed at the end of the step.

    Sign / cost invariants enforced:
      • len(pnl) == len(signals)                        (no off-by-one)
      • transaction cost is applied ONCE per open and ONCE per close
      • mark-to-market on step i uses returns[i] (return over the step)
      • long position earns +ret; short position earns -ret
    """
    signals = np.asarray(signals)
    returns = np.asarray(returns)
    if signals.shape != returns.shape:
        raise ValueError(
            f"simulate_trades: signals.shape={signals.shape} != returns.shape={returns.shape} "
            f"— callers must align these before simulation (len must match)."
        )
    if signals.ndim != 1:
        raise ValueError(f"simulate_trades: signals must be 1-D, got {signals.ndim}-D.")

    # position: 0 = flat, +1 = long, -1 = short
    position = 0
    pnl = np.zeros(len(signals), dtype=np.float64)
    cost = TRANSACTION_COST + SLIPPAGE

    for i, sig in enumerate(signals):
        ret = float(returns[i]) if i < len(returns) else 0.0
        step_pnl = 0.0

        # Mark-to-market: PnL earned during step i by whatever position was
        # held from step i-1's close. LONG earns +ret, SHORT earns -ret.
        if position != 0:
            step_pnl += position * ret

        # Act on the signal at END of this step. An entry or close charges
        # one cost event. A reversal (long→short or short→long) = close + open
        # = 2 × cost, which is physically accurate.
        if sig == 1:
            if position == 0:
                step_pnl -= cost
                position = 1
            elif position == -1:
                step_pnl -= 2.0 * cost  # close short + open long
                position = 1
            # position == 1 already → no-op
        elif sig == -1:
            if allow_short:
                if position == 0:
                    step_pnl -= cost
                    position = -1
                elif position == 1:
                    step_pnl -= 2.0 * cost  # close long + open short
                    position = -1
                # position == -1 already → no-op
            else:
                # Long-only fallback: sig=-1 only CLOSES an open long.
                if position == 1:
                    step_pnl -= cost
                    position = 0
        # sig == 0 → HOLD: no position change, already marked-to-market above.

        pnl[i] = step_pnl

    assert len(pnl) == len(signals), (
        f"simulate_trades internal invariant broken: len(pnl)={len(pnl)} "
        f"!= len(signals)={len(signals)}"
    )
    return pnl


def _resolve_data_dir(data_dir: Path) -> Path:
    if data_dir == DEFAULT_DATA_DIR and DEFAULT_DATA_DIR_V2.exists():
        return DEFAULT_DATA_DIR_V2
    return data_dir


def _select_candle_file(data_dir: Path, symbol: str, horizon: str) -> Path:
    """Pick the candle CSV for (symbol, horizon). If multiple viable files
    exist, prefer the horizon-specific OHLCV and log the ambiguity so users
    know which one was chosen. Raise if none of the candidates exist."""
    symbol_upper = symbol.upper()
    preferred_by_horizon = {
        "1h":  data_dir / f"{symbol_upper}_1h_ohlcv.csv",
        "4h":  data_dir / f"{symbol_upper}_4h_ohlcv.csv",
        "15m": data_dir / f"{symbol_upper}_15m_ohlcv.csv",
        "24h": data_dir / f"{symbol_upper}_1d_ohlcv.csv",
    }
    fallback_max = data_dir / "candles_max.csv"
    fallback_training = data_dir / f"{symbol.lower()}_training_dataset_v2.csv"

    ordered = [preferred_by_horizon.get(horizon.lower()), fallback_max, fallback_training]
    ordered = [p for p in ordered if p is not None]
    existing = [p for p in ordered if p.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No candle file found for symbol={symbol} horizon={horizon} in {data_dir}. "
            f"Tried: {[str(p.name) for p in ordered]}"
        )
    chosen = existing[0]
    if len(existing) > 1:
        others = [p.name for p in existing[1:]]
        print(f"[WARN] _select_candle_file: multiple candidates exist {others}, "
              f"using highest-priority: {chosen.name}", flush=True)
    return chosen


def backtest(artifact_dir: Path, data_dir: Path, symbol: str, horizon: str,
             policy_json: Optional[Path] = None, allow_short: bool = True,
             export_predictions: Optional[Path] = None):
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
    if "timestamp" not in candle_df.columns and "datetime" in candle_df.columns:
        candle_df = candle_df.rename(columns={"datetime": "timestamp"})
    candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'])

    print(f"[*] Loading embeddings...")
    article_embeddings = np.load(data_dir / "btcusdt_article_embeddings_max.npy")

    print(f"[*] Loading articles metadata...")
    articles_df = pd.read_csv(data_dir / "articles_max.csv")
    if "timestamp" not in articles_df.columns and "published_at" in articles_df.columns:
        articles_df = articles_df.rename(columns={"published_at": "timestamp"})
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

    precomputed_features = None
    features_path = data_dir / f"features_precomputed_{horizon}.npy"
    if not features_path.exists():
        features_path = data_dir / "features_precomputed.npy"
    if features_path.exists():
        candidate_features = np.load(features_path, mmap_mode="r")
        if candidate_features.shape == (len(candle_df), 63):
            precomputed_features = candidate_features
            print(f"[*] Loading precomputed market features from {features_path.name}...")

    market_bars = None
    if getattr(model, "use_bar_sequences", False):
        bars_path = data_dir / f"market_bars_{horizon}.npz"
        if not bars_path.exists():
            bars_path = data_dir / "market_bars.npz"
        if not bars_path.exists():
            raise RuntimeError(
                f"[ERROR] Checkpoint requires hybrid bar sequences but {bars_path.name} is missing."
            )
        with np.load(bars_path) as bars_npz:
            candidate_bars = {key: bars_npz[key] for key in bars_npz.files}
        first_bars = next(iter(candidate_bars.values()))
        if first_bars.shape[0] != len(candle_df):
            raise RuntimeError(
                f"[ERROR] {bars_path.name} has N={first_bars.shape[0]}, "
                f"but {candle_path.name} has N={len(candle_df)}."
            )
        market_bars = candidate_bars
        print(f"[*] Loading hybrid market bars from {bars_path.name}...")

    article_novelty = None
    novelty_path = data_dir / "article_novelty.npy"
    if novelty_path.exists():
        article_novelty = np.load(novelty_path, mmap_mode="r")

    # Create dataset to get test split (FIXED: correct constructor)
    try:
        dataset = SAFEAlertDataset(
            candle_df=candle_df,
            article_embeddings=article_embeddings,
            article_meta=articles_df,
            article_to_candle={},
            factor_labels=factor_labels,
            entity_sentiment=entity_sentiment,
            precomputed_features=precomputed_features,
            market_bars=market_bars,
            article_novelty=article_novelty,
            symbol=symbol,
            horizon=horizon,
            articles_per_candle=32,
        )
    except Exception as e:
        raise RuntimeError(f"[ERROR] Cannot load dataset: {e}. Ensure training data exists.")

    # P0 #1: rehydrate training preprocessing state onto the dataset so that
    # Dataset._normalize_market_features mirrors what training did — test
    # samples are z-scored using the exact (mean, std, clip) the trainer fit.
    if hasattr(model, "_market_feature_mean") and model._market_feature_mean is not None:
        preprocessing_state = {
            "market_feature_mean": model._market_feature_mean.cpu(),
            "market_feature_std":  model._market_feature_std.cpu(),
            "market_feature_clip": model._market_feature_clip,
        }
        checkpoint_preproc = getattr(model, "_preprocessing_state", None) or {}
        for key in ("market_bar_mean", "market_bar_std", "market_bar_clip"):
            if checkpoint_preproc.get(key) is not None:
                preprocessing_state[key] = checkpoint_preproc[key]
        dataset.load_preprocessing_state(preprocessing_state)
        print("[OK] Dataset preprocessing state rehydrated from checkpoint.")
    else:
        print("[WARN] Dataset has no scaler — backtest may use raw features (distribution mismatch).")

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
    prediction_rows = []

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
                market_bars_sample = sample.get("market_bars")
                if market_bars_sample is not None:
                    market_bars_sample = market_bars_sample.unsqueeze(0).to(device)

                # Get prediction. Symbol is forwarded for API compatibility;
                # the paper-final model does not use it in Eq.9.
                pred_dict = model(
                    market_feat=market_feat,
                    horizon=horizon,
                    article_emb=article_emb,
                    article_mask=article_mask,
                    article_meta_vec=article_meta,
                    symbol=symbol,
                    market_bars=market_bars_sample,
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
                candle_idx = dataset.valid_idx[i]
                prediction_rows.append({
                    "time": pd.Timestamp(candle_df.iloc[candle_idx]["timestamp"]).isoformat(),
                    "symbol": symbol.upper(),
                    "horizon": horizon,
                    "direction": ("DOWN", "NEUTRAL", "UP")[direction_pred],
                    "confidence": confidence,
                    "max_probability": max_prob,
                    "probabilities": {
                        "DOWN": float(probs[0]),
                        "NEUTRAL": float(probs[1]),
                        "UP": float(probs[2]),
                    },
                    "return": ret,
                })

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

    if export_predictions is not None:
        export_predictions.parent.mkdir(parents=True, exist_ok=True)
        with export_predictions.open("w", encoding="utf-8") as handle:
            for row in prediction_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[OK] Exported historical predictions to {export_predictions}")

    # Convert model classes to trading signals with policy thresholds (Eq.26)
    if policy:
        tau = float(policy.get("tau", 1.0))
        gamma = float(policy.get("gamma", 1.0))
        alert_mask = (all_confidence >= tau) & (all_max_prob >= gamma)
        signals = np.where(alert_mask, np.where(all_predictions == 2, 1, np.where(all_predictions == 0, -1, 0)), 0)
        print(f"[OK] Alert coverage: {alert_mask.mean():.4f} (tau={tau:.4f}, gamma={gamma:.4f})")
    else:
        signals = np.where(all_predictions == 2, 1, np.where(all_predictions == 0, -1, 0))

    # Strategy PnL with real returns. allow_short MUST match the convention
    # used by metrics_safe_alert.mini_backtest (which trains threshold policies
    # with LONG+SHORT on UP/DOWN) — otherwise the deployed strategy evaluates
    # on different economics than the τ/γ search was optimized for (was Bug #4).
    strategy_pnl = simulate_trades(signals, all_returns, allow_short=allow_short)
    print(f"[OK] Strategy mode: {'long+short' if allow_short else 'long-only'}")

    # Buy-hold baseline with same real returns
    buyhold_pnl = all_returns

    # Compute metrics on REAL data — full-window (single aggregate view).
    strategy_metrics = compute_metrics(strategy_pnl)
    buyhold_metrics = compute_metrics(buyhold_pnl)

    print("\n[STRATEGY METRICS] (Real Model Predictions)")
    for k, v in strategy_metrics.items():
        print(f"  {k:15s}: {v:10.4f}")

    print("\n[BUY-HOLD BASELINE] (Same Real Returns)")
    for k, v in buyhold_metrics.items():
        print(f"  {k:15s}: {v:10.4f}")

    # Walk-forward (PDF Section 4.2.3): split test set into rolling chunks and
    # report mean±std. Addresses the reviewer concern that a single-window
    # Sharpe can be inflated by a favourable regime.
    strategy_wf = walk_forward_metrics(strategy_pnl, n_windows=5)
    buyhold_wf = walk_forward_metrics(buyhold_pnl, n_windows=5)

    print(f"\n[WALK-FORWARD] Strategy — {strategy_wf['n_windows']} non-overlapping windows "
          f"of ~{strategy_wf['window_size']} samples each")
    for key in ("sharpe", "sortino", "calmar", "max_dd", "win_rate", "total_return"):
        mean = strategy_wf["summary"].get(f"{key}_mean", 0.0)
        std = strategy_wf["summary"].get(f"{key}_std", 0.0)
        print(f"  {key:15s}: {mean:+.4f} ± {std:.4f}")

    print(f"\n[WALK-FORWARD] Buy-Hold — {buyhold_wf['n_windows']} windows")
    for key in ("sharpe", "sortino", "calmar", "max_dd", "win_rate", "total_return"):
        mean = buyhold_wf["summary"].get(f"{key}_mean", 0.0)
        std = buyhold_wf["summary"].get(f"{key}_std", 0.0)
        print(f"  {key:15s}: {mean:+.4f} ± {std:.4f}")

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

    # Save metrics JSON — full-window + walk-forward (PDF Section 4.2.3).
    metrics_json = {
        "test_size": len(all_predictions),
        "strategy_metrics": {k: float(v) for k, v in strategy_metrics.items()},
        "buyhold_metrics": {k: float(v) for k, v in buyhold_metrics.items()},
        "strategy_walk_forward": strategy_wf,
        "buyhold_walk_forward": buyhold_wf,
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
    parser.add_argument("--long_only", action="store_true",
                        help="Long-only strategy (spot). Default: long+short, "
                             "matches mini_backtest used during τ/γ search.")

    parser.add_argument("--export_predictions", type=Path, default=None,
                        help="Optional JSONL path for timestamped historical predictions")
    args = parser.parse_args()
    backtest(args.artifact_dir, args.data_dir, args.symbol, args.horizon,
             args.policy_json, allow_short=not args.long_only,
             export_predictions=args.export_predictions)
