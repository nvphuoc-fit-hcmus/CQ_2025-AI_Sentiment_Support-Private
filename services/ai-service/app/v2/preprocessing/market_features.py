"""
Market feature engineering for V2 pipeline.

Inspired by crypto-price-forecasting research (sections 4-6):
- Section 6 (Feature Importance): Technical indicators are #1 driver (37% total_gain for BTC)
  Key indicators: AO, MACD, EMA, BB, RSI, Stoch RSI, VWAP, NVI
- Section 4 (EDA): Lag features improve prediction — research used lags 0-13 days
  For hourly candles: lags 1-5 candles are equivalent
- Section 5 (Time Series Models): Features must be stationary (log-returns, not raw price)

✅ FIX #13: 63-Dimension Feature Breakdown (PDF Eq.16-19)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Per-Timeframe Features (12 indicators × 5 timeframes = 60 dims):

Timeframe 1 (1m):    [RSI_14, MACD, MACD_hist, BB_pos, Stoch_RSI, EMA_cross, VWAP_dev, log_ret, 5ret, Vol_spike, MA20, Volatility]
Timeframe 2 (5m):    [RSI_14, MACD, MACD_hist, BB_pos, Stoch_RSI, EMA_cross, VWAP_dev, log_ret, 5ret, Vol_spike, MA20, Volatility]
Timeframe 3 (15m):   [RSI_14, MACD, MACD_hist, BB_pos, Stoch_RSI, EMA_cross, VWAP_dev, log_ret, 5ret, Vol_spike, MA20, Volatility]
Timeframe 4 (1h):    [RSI_14, MACD, MACD_hist, BB_pos, Stoch_RSI, EMA_cross, VWAP_dev, log_ret, 5ret, Vol_spike, MA20, Volatility]
Timeframe 5 (4h):    [RSI_14, MACD, MACD_hist, BB_pos, Stoch_RSI, EMA_cross, VWAP_dev, log_ret, 5ret, Vol_spike, MA20, Volatility]

Cross-Timeframe Aggregate Features (3 dims):
  - RSI_aggregate:   mean(RSI_14 across 5 timeframes)
  - Momentum:        4h_trend - 1h_trend (relative momentum)
  - Volume_trend:    slope of volume over last 5 candles

Total: 5×12 + 3 = 63 features ✅

Multi-Timeframe Enhancement (Eq.16-19):
- Implements PDF spec: 5 timeframes (1m, 5m, 15m, 1h, 4h)
- Resamples from 1m candles (base timeframe from Kafka)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# MULTI-TIMEFRAME RESAMPLING (Eq.16-19 Support)
# ──────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.

    Assumes df is sorted by timestamp with uniform spacing.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        target_timeframe: "1m", "5m", "15m", "1h", "4h", etc.

    Returns:
        Resampled DataFrame
    """
    # Map timeframe to pandas frequency
    # BUG FIX: Use newest pandas lowercase format
    freq_map = {
        "1m": "1min",   # 1 minute
        "5m": "5min",   # 5 minutes
        "15m": "15min", # 15 minutes
        "1h": "1h",     # 1 hour (lowercase!)
        "4h": "4h",     # 4 hours (lowercase!)
        "1d": "1d",     # 1 day (lowercase!)
    }

    if target_timeframe not in freq_map:
        return df

    freq = freq_map[target_timeframe]

    # Set timestamp as index
    df_indexed = df.copy()
    if "timestamp" in df_indexed.columns:
        df_indexed["timestamp"] = pd.to_datetime(df_indexed["timestamp"], unit="ms", errors="coerce")
        df_indexed = df_indexed.set_index("timestamp")
    elif df_indexed.index.name != "timestamp":
        df_indexed.index = pd.to_datetime(df_indexed.index, unit="ms", errors="coerce")

    # Resample OHLCV
    resampled = pd.DataFrame()
    resampled["open"] = df_indexed["open"].resample(freq).first()
    resampled["high"] = df_indexed["high"].resample(freq).max()
    resampled["low"] = df_indexed["low"].resample(freq).min()
    resampled["close"] = df_indexed["close"].resample(freq).last()
    resampled["volume"] = df_indexed["volume"].resample(freq).sum()

    # Reset index
    resampled = resampled.reset_index()
    resampled = resampled.dropna()

    return resampled


def build_multiframe_features(df_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Eq.16-19: Build market features for 5 timeframes.

    Takes base 1m candles and resamples + engineers features at each timeframe.

    Args:
        df_1m: DataFrame with 1m OHLCV data (min 250 rows for 250 minutes)

    Returns:
        dict: {
            "1m": features_df,
            "5m": features_df,
            "15m": features_df,
            "1h": features_df,
            "4h": features_df,
        }
    """
    timeframes = ["1m", "5m", "15m", "1h", "4h"]
    result = {}

    for tf in timeframes:
        if tf == "1m":
            df_tf = df_1m.copy()
        else:
            df_tf = resample_ohlcv(df_1m, tf)

        # Add technical indicators
        df_tf = add_technical_indicators(df_tf)
        df_tf = add_return_features(df_tf)

        result[tf] = df_tf.dropna()

    return result


# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    width = (upper - lower) / mid.replace(0, np.nan)
    # position within band: 0 = at lower, 1 = at upper
    pos = (close - lower) / (upper - lower).replace(0, np.nan)
    return upper, mid, lower, width, pos


def _stoch_rsi(close: pd.Series, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    rsi = _rsi(close, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    return (rsi - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD, Bollinger Bands, EMA, SMA, Stoch RSI.
    These are the top-performing features from Section 6 feature importance analysis.
    """
    out = df.copy()
    close = out["close"]
    volume = out["volume"]

    # RSI 14 — high importance (btc_indicator_RSI_d_3 in causal vars)
    out["rsi_14"] = _rsi(close, 14)

    # MACD — consistently top 5 (btc_indicator_MACD_d_0 through _d_5)
    macd, macd_sig, macd_hist = _macd(close)
    out["macd"] = macd
    out["macd_signal"] = macd_sig
    out["macd_hist"] = macd_hist

    # Bollinger Bands — BB width & position (btc_indicator_BBM/BBW)
    bb_upper, bb_mid, bb_lower, bb_width, bb_pos = _bollinger(close)
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["bb_pos"] = bb_pos

    # EMA — btc_indicator_EMA_d_2 through _d_9
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()
    out["ema_cross"] = out["ema_9"] - out["ema_21"]  # crossover signal

    # SMA
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()

    # Stoch RSI — btc_indicator_Stoch_RSI_d_0 and _d_3
    out["stoch_rsi"] = _stoch_rsi(close)

    # VWAP (approximation using rolling window) — btc_indicator_VWAP_d_0
    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap_20"] = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    out["vwap_deviation"] = (close - out["vwap_20"]) / out["vwap_20"].replace(0, np.nan)

    return out


# ──────────────────────────────────────────────────────────────
# RETURN & VOLUME FEATURES
# ──────────────────────────────────────────────────────────────

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Price returns at multiple horizons + volume features.
    log_return is stationary (important for Section 5 models).
    """
    out = df.copy()
    close = out["close"]
    volume = out["volume"]

    # Log returns — stationary, used in reference repo as target variable
    out["log_return_1"] = np.log(close / close.shift(1))
    out["log_return_3"] = np.log(close / close.shift(3))
    out["log_return_5"] = np.log(close / close.shift(5))

    # Pct returns
    out["return_1"] = close.pct_change(1)
    out["return_5"] = close.pct_change(5)
    out["return_15"] = close.pct_change(15)

    # Rolling stats
    out["ma_5"] = close.rolling(5).mean()
    out["ma_10"] = close.rolling(10).mean()
    out["ma_20"] = close.rolling(20).mean()
    out["std_5"] = close.rolling(5).std()
    out["std_15"] = close.rolling(15).std()

    # Price range (intrabar volatility)
    out["range"] = (out["high"] - out["low"]) / close.replace(0, np.nan)

    # Volume features — Exchange volume data is 4th most important (Section 6)
    out["volume_ma_10"] = volume.rolling(10).mean()
    out["volume_spike"] = volume / out["volume_ma_10"].replace(0, np.nan)
    out["volume_log"] = np.log1p(volume)

    return out


# ──────────────────────────────────────────────────────────────
# LAG FEATURES
# ──────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """
    Add lagged versions of specified columns.

    Directly inspired by lag_functions.py from reference repo (Section 4).
    Research found lags 0-13 days significant — for 60-min candles we use lags 1-5.

    Example: return_1_lag1 = return_1 shifted by 1 candle (1 hour ago)
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


# ──────────────────────────────────────────────────────────────
# COMBINED PIPELINE
# ──────────────────────────────────────────────────────────────

#: Columns to lag — selected based on Section 6 feature importance:
#: Technical indicators > NLP > Volume > Price
DEFAULT_LAG_COLS = [
    "log_return_1",   # most stationary return
    "return_1",
    "volume_spike",
    "rsi_14",         # top TA indicator
    "macd_hist",      # MACD histogram (more stationary than MACD itself)
    "bb_width",       # volatility
    "stoch_rsi",
]

DEFAULT_LAGS = [1, 2, 3, 5]  # equivalent to 1h, 2h, 3h, 5h lookback


def _compute_tf_features(df: pd.DataFrame, rsi_p: int, macd_fast: int, macd_slow: int,
                          macd_sig: int, bb_p: int, ema_s: int, ema_l: int,
                          stoch_p: int, ret_n: int, ma_p: int, std_p: int) -> np.ndarray:
    """
    Compute 12 indicators for one temporal scale using specified periods.
    Returns (12,) float32 array. All NaN → 0.0.

    Features (match key_features layout in old code):
      [0]  RSI                [1]  MACD_hist
      [2]  BB_pos             [3]  StochRSI
      [4]  EMA_cross          [5]  VWAP_deviation
      [6]  log_return_1       [7]  return_N
      [8]  volume_spike       [9]  range (intrabar)
      [10] MA_P               [11] std_P
    """
    close  = df["close"]
    volume = df["volume"]
    high   = df["high"]
    low    = df["low"]

    def _safe(v):
        return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    # [0] RSI
    rsi = _rsi(close, rsi_p).iloc[-1]

    # [1] MACD histogram
    ema_f = close.ewm(span=macd_fast, adjust=False).mean()
    ema_s_ser = close.ewm(span=macd_slow, adjust=False).mean()
    macd_line = ema_f - ema_s_ser
    sig_line  = macd_line.ewm(span=macd_sig, adjust=False).mean()
    macd_h    = (macd_line - sig_line).iloc[-1]

    # [2] Bollinger position
    mid = close.rolling(bb_p).mean()
    std = close.rolling(bb_p).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    denom = (upper - lower).replace(0, np.nan)
    bb_pos = ((close - lower) / denom).iloc[-1]

    # [3] StochRSI
    rsi_s = _rsi(close, rsi_p)
    rsi_min = rsi_s.rolling(stoch_p).min()
    rsi_max = rsi_s.rolling(stoch_p).max()
    stoch_rsi = ((rsi_s - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan)).iloc[-1]

    # [4] EMA cross (short - long)
    ema_cross = (close.ewm(span=ema_s, adjust=False).mean()
                 - close.ewm(span=ema_l, adjust=False).mean()).iloc[-1]
    # Normalise by close price so it's dimensionless
    ema_cross_norm = ema_cross / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

    # [5] VWAP deviation
    typical = (high + low + close) / 3.0
    vwap = ((typical * volume).rolling(ma_p).sum()
            / volume.rolling(ma_p).sum().replace(0, np.nan))
    vwap_dev = ((close - vwap) / vwap.replace(0, np.nan)).iloc[-1]

    # [6] log_return_1
    log_ret = np.log(close / close.shift(1)).iloc[-1]

    # [7] return_N
    ret_n_val = close.pct_change(ret_n).iloc[-1]

    # [8] volume_spike
    vol_ma = volume.rolling(ma_p).mean().replace(0, np.nan)
    vol_spike = (volume / vol_ma).iloc[-1]

    # [9] intrabar range
    rng = ((high - low) / close.replace(0, np.nan)).iloc[-1]

    # [10] MA_P (relative to last close, dimensionless)
    ma_val = close.rolling(ma_p).mean().iloc[-1]
    ma_rel = (ma_val / close.iloc[-1] - 1.0) if close.iloc[-1] != 0 else 0.0

    # [11] rolling std_P (normalised)
    std_val = close.rolling(std_p).std().iloc[-1]
    std_norm = std_val / close.iloc[-1] if close.iloc[-1] != 0 else 0.0

    return np.array([
        _safe(rsi), _safe(macd_h), _safe(bb_pos), _safe(stoch_rsi),
        _safe(ema_cross_norm), _safe(vwap_dev), _safe(log_ret), _safe(ret_n_val),
        _safe(vol_spike), _safe(rng), _safe(ma_rel), _safe(std_norm),
    ], dtype=np.float32)


# Five temporal scales for 1h-base candles (Eq.16-19).
# Each row: (rsi_p, macd_fast, macd_slow, macd_sig, bb_p, ema_s, ema_l, stoch_p, ret_n, ma_p, std_p)
# Scale 0 ≈ 4h,  Scale 1 ≈ 12h,  Scale 2 ≈ 24h,  Scale 3 ≈ 72h,  Scale 4 ≈ 168h
_TF_PARAMS = [
    (4,   3,  6,  2,  5,  3,  6,  4,  1,  5,  4),   # very short (4h)
    (8,   6, 13,  4, 10,  5, 10,  8,  3, 10,  8),   # short (12h)
    (14, 12, 26,  9, 20,  9, 21, 14,  6, 20, 15),   # standard (24h)
    (21, 16, 34,  9, 30, 13, 34, 21, 12, 30, 21),   # medium (72h)
    (34, 26, 52,  9, 50, 21, 55, 34, 24, 50, 34),   # long (168h)
]
_MIN_ROWS_PER_TF = [6, 15, 28, 40, 60]  # minimum rows needed per scale


def extract_multiframe_market_features(df_1h: pd.DataFrame) -> np.ndarray:
    """
    Eq.16-19: Extract true 5-scale market features from 1h-base candles.

    Input data is 1h candles (NOT 1m), so resampling to sub-hour timeframes
    is meaningless — all sub-1h resamples return identical data.

    Fix: compute the same 12 indicators with 5 different period scales
    (short→long), each representing a distinct temporal resolution:
      Scale 0 ≈  4h context  (periods: RSI_4,  MACD 3/6/2)
      Scale 1 ≈ 12h context  (periods: RSI_8,  MACD 6/13/4)
      Scale 2 ≈ 24h context  (periods: RSI_14, MACD 12/26/9)  ← standard
      Scale 3 ≈ 72h context  (periods: RSI_21, MACD 16/34/9)
      Scale 4 ≈ 168h context (periods: RSI_34, MACD 26/52/9)

    Returns: np.ndarray of shape (63,)
      - 5 scales × 12 indicators = 60 features
      - 3 cross-scale aggregate features = 3 features
      - Total: 63 features, all real data, no zero-padding
    """
    result = []
    scale_rsi = []  # collect RSI values for cross-scale aggregate

    for scale_idx, params in enumerate(_TF_PARAMS):
        min_rows = _MIN_ROWS_PER_TF[scale_idx]
        if len(df_1h) < min_rows:
            result.append(np.zeros(12, dtype=np.float32))
            scale_rsi.append(50.0)
            continue

        feats = _compute_tf_features(df_1h, *params)
        result.append(feats)
        scale_rsi.append(float(feats[0]))  # RSI is index 0

    # Cross-scale aggregate features (3 dims = indices 60-62)
    # [60] Mean RSI across all 5 scales
    agg_rsi = float(np.mean(scale_rsi))

    # [61] Momentum: RSI long-scale minus RSI short-scale (trend strength)
    agg_momentum = float(scale_rsi[-1] - scale_rsi[0])  # scale4 - scale0

    # [62] Volatility ratio: std_short / std_long (normalised)
    std_short = float(result[0][11])  # scale0 std
    std_long  = float(result[4][11])  # scale4 std
    agg_vol_ratio = std_short / (std_long + 1e-8)
    # Clamp to reasonable range to prevent outliers
    agg_vol_ratio = float(np.clip(agg_vol_ratio, 0.0, 10.0))

    result.append(np.array([agg_rsi, agg_momentum, agg_vol_ratio], dtype=np.float32))

    features = np.concatenate(result, dtype=np.float32)  # (63,)
    return features[:63].astype(np.float32)


# ──────────────────────────────────────────────────────────────
# MARKET REGIME
# ──────────────────────────────────────────────────────────────

# True multi-timeframe override (1m -> 5m/15m/1h/4h) while keeping 1h fallback.
_BASE_TF_PARAMS = (14, 12, 26, 9, 20, 9, 21, 14, 5, 20, 20)


def _extract_multiframe_from_1h(df_1h: pd.DataFrame) -> np.ndarray:
    """Fallback multiscale features from 1h candles using scaled indicator periods."""
    result = []
    scale_rsi = []

    for scale_idx, params in enumerate(_TF_PARAMS):
        min_rows = _MIN_ROWS_PER_TF[scale_idx]
        if len(df_1h) < min_rows:
            result.append(np.zeros(12, dtype=np.float32))
            scale_rsi.append(50.0)
            continue

        feats = _compute_tf_features(df_1h, *params)
        result.append(feats)
        scale_rsi.append(float(feats[0]))

    agg_rsi = float(np.mean(scale_rsi))
    agg_momentum = float(scale_rsi[-1] - scale_rsi[0])
    std_short = float(result[0][11])
    std_long  = float(result[4][11])
    agg_vol_ratio = std_short / (std_long + 1e-8)
    agg_vol_ratio = float(np.clip(agg_vol_ratio, 0.0, 10.0))

    result.append(np.array([agg_rsi, agg_momentum, agg_vol_ratio], dtype=np.float32))
    features = np.concatenate(result, dtype=np.float32)
    return features[:63].astype(np.float32)


def _extract_multiframe_from_frames(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    """True multi-timeframe features using separate candles per timeframe."""
    result = []
    scale_rsi = []
    for tf in ["1m", "5m", "15m", "1h", "4h"]:
        df_tf = frames.get(tf)
        if df_tf is None or len(df_tf) < 30:
            result.append(np.zeros(12, dtype=np.float32))
            scale_rsi.append(50.0)
            continue
        feats = _compute_tf_features(df_tf, *_BASE_TF_PARAMS)
        result.append(feats)
        scale_rsi.append(float(feats[0]))

    agg_rsi = float(np.mean(scale_rsi))
    agg_momentum = float(scale_rsi[-1] - scale_rsi[0])
    std_short = float(result[0][11])
    std_long  = float(result[4][11])
    agg_vol_ratio = std_short / (std_long + 1e-8)
    agg_vol_ratio = float(np.clip(agg_vol_ratio, 0.0, 10.0))

    result.append(np.array([agg_rsi, agg_momentum, agg_vol_ratio], dtype=np.float32))
    features = np.concatenate(result, dtype=np.float32)
    return features[:63].astype(np.float32)


def _looks_like_1m(df: pd.DataFrame) -> bool:
    if len(df) < 3:
        return False
    ts = df["timestamp"] if "timestamp" in df.columns else df.index
    ts = pd.to_datetime(ts, errors="coerce")
    diffs = ts.diff().dropna()
    if diffs.empty:
        return False
    median_minutes = diffs.median().total_seconds() / 60.0
    return 0.5 <= median_minutes <= 2.0


def extract_multiframe_market_features(df: pd.DataFrame) -> np.ndarray:
    """
    True multi-timeframe features when 1m candles are provided.
    Falls back to scaled-period 1h features when only 1h data is available.
    """
    if _looks_like_1m(df):
        frames = {
            "1m": df.copy(),
            "5m": resample_ohlcv(df, "5m"),
            "15m": resample_ohlcv(df, "15m"),
            "1h": resample_ohlcv(df, "1h"),
            "4h": resample_ohlcv(df, "4h"),
        }
        return _extract_multiframe_from_frames(frames)
    return _extract_multiframe_from_1h(df)


def extract_multiframe_market_features_from_frames(frames: dict[str, pd.DataFrame]) -> np.ndarray:
    """Public wrapper for true multi-timeframe features from pre-sliced frames."""
    return _extract_multiframe_from_frames(frames)


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market regime: trending up / trending down / sideways.

    Uses:
      - ADX (Average Directional Index) > 25 → trending
      - EMA50 vs EMA200 → direction
      - Regime encoded as 3 binary features (no ordinal assumption)

    regime_trending_up   = 1 if ADX>25 and EMA50>EMA200
    regime_trending_down = 1 if ADX>25 and EMA50<EMA200
    regime_sideways      = 1 if ADX<=25
    """
    out = df.copy()
    high  = out["high"]
    low   = out["low"]
    close = out["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move   = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    period = 14
    atr    = pd.Series(tr).rolling(period).mean()
    plus_di  = 100 * pd.Series(plus_dm).rolling(period).mean()  / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(period).mean()

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    trending    = adx > 25
    trend_up    = trending & (ema50 > ema200)
    trend_down  = trending & (ema50 < ema200)
    sideways    = ~trending

    out["regime_trending_up"]   = trend_up.astype(int)
    out["regime_trending_down"] = trend_down.astype(int)
    out["regime_sideways"]      = sideways.astype(int)
    out["adx"]                  = adx

    return out


# ──────────────────────────────────────────────────────────────
# BTC CROSS-COIN FEATURES
# ──────────────────────────────────────────────────────────────

BTC_CROSS_COLS = [
    "btc_return_1",
    "btc_log_return_1",
    "btc_volume_spike",
    "btc_rsi_14",
    "btc_macd_hist",
    "btc_regime_trending_up",
    "btc_regime_trending_down",
    "btc_regime_sideways",
]


def add_btc_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BTC features vào dataset của altcoin theo timestamp.

    btc_df phải đã qua build_market_features() + add_market_regime().
    Chỉ lấy các cột quan trọng nhất của BTC làm cross-coin signal.

    BTC thường dẫn dắt altcoin 0-2h → dùng BTC data tại cùng timestamp (lag=0).
    """
    btc_cols = {
        "timestamp":           "timestamp",
        "return_1":            "btc_return_1",
        "log_return_1":        "btc_log_return_1",
        "volume_spike":        "btc_volume_spike",
        "rsi_14":              "btc_rsi_14",
        "macd_hist":           "btc_macd_hist",
        "regime_trending_up":  "btc_regime_trending_up",
        "regime_trending_down":"btc_regime_trending_down",
        "regime_sideways":     "btc_regime_sideways",
    }

    # Chỉ lấy các cột có trong btc_df
    available = {k: v for k, v in btc_cols.items() if k in btc_df.columns}
    btc_subset = btc_df[list(available.keys())].rename(columns=available)

    out = df.merge(btc_subset, on="timestamp", how="left")

    # Fill NaN (timestamp không khớp) bằng 0
    for col in BTC_CROSS_COLS:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    return out


def build_market_features(df: pd.DataFrame, add_lags: bool = True) -> pd.DataFrame:
    """
    Full market feature engineering pipeline.

    Steps:
    1. Price returns + volume features
    2. Technical indicators (RSI, MACD, BB, EMA, Stoch RSI, VWAP)
    3. Lag features for top indicators (lags 1-5 candles)
    4. Drop NaN rows from rolling/lag windows

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        add_lags: Whether to add lag features (disable for inference on single row)

    Returns:
        DataFrame with all engineered features, NaN rows dropped.
    """
    out = add_return_features(df)
    out = add_technical_indicators(out)
    out = add_market_regime(out)

    if add_lags:
        out = add_lag_features(out, cols=DEFAULT_LAG_COLS, lags=DEFAULT_LAGS)

    out = out.dropna().reset_index(drop=True)
    return out
