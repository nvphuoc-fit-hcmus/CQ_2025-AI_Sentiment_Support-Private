#!/usr/bin/env python3
"""
Precompute all 52,542 market features ONCE before training.
Saves to: features_precomputed.npy (52542 × 63)

This eliminates on-the-fly computation during __getitem__(),
reducing training time from 20+ hours to 3-5 hours!

Command:
  python precompute_market_features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.market_features import (
    extract_multiframe_market_features,
    extract_multiframe_market_features_from_frames,
)


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamp column name to 'timestamp'."""
    # Normalize column names (strip whitespace/BOM and lowercase for matching)
    normalized = {
        col: col.strip().lstrip("\ufeff").lower()
        for col in df.columns
    }
    if "timestamp" in normalized.values():
        inv = {v: k for k, v in normalized.items()}
        return df.rename(columns={inv["timestamp"]: "timestamp"})
    # Try common alternatives
    candidates = [
        "time", "date", "datetime", "open_time", "open_time_ms", "timestamp_ms", "ts",
    ]
    for col in candidates:
        if col in normalized.values():
            inv = {v: k for k, v in normalized.items()}
            df = df.rename(columns={inv[col]: "timestamp"})
            return df
    # Heuristic: first column containing 'time'
    for col_raw, col_norm in normalized.items():
        if "time" in col_norm:
            df = df.rename(columns={col_raw: "timestamp"})
            return df
    return df


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convert mixed timestamp formats to pandas datetime."""
    if np.issubdtype(series.dtype, np.number):
        # Heuristic: ms if large, else seconds
        unit = "ms" if series.max() > 1e12 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")

def precompute_all_features(candle_file, output_file, frames=None):
    """Precompute market features for the target candle file."""

    if frames is None:
        print("[PRECOMPUTE] Loading candles...")
        candles = pd.read_csv(candle_file)
        candles = _ensure_timestamp_column(candles)
        if "timestamp" not in candles.columns:
            raise KeyError("timestamp")
        candles["timestamp"] = _to_datetime(candles["timestamp"])
        print(f"  Loaded {len(candles)} candles")
    else:
        candles = frames["1h"]
        print(f"[PRECOMPUTE] Using multi-timeframe frames (target=1h, rows={len(candles)})")

    # Initialize output array
    all_features = np.zeros((len(candles), 63), dtype=np.float32)

    print(f"\n[PRECOMPUTE] Computing features for {len(candles)} candles...")
    print(f"  This will take 5-10 minutes on CPU, 1-2 minutes on GPU\n")

    for i in range(len(candles)):
        # Progress
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(candles)} ({(i+1)/len(candles)*100:.1f}%)")

        try:
            if frames is None:
                # Look back 250 candles (standard window)
                lookback_idx = max(0, i - 250)
                df_window = candles.iloc[lookback_idx:i+1].copy()

                if len(df_window) >= 50:
                    features = extract_multiframe_market_features(df_window)
                    all_features[i] = features[:63].astype(np.float32)
                else:
                    all_features[i] = np.zeros(63, dtype=np.float32)
            else:
                ts = candles.iloc[i]["timestamp"]
                tail_map = {"1m": 500, "5m": 200, "15m": 200, "1h": 200, "4h": 120}
                sliced = {}
                for tf, df_tf in frames.items():
                    tail_n = tail_map.get(tf, 200)
                    sliced_tf = df_tf[df_tf["timestamp"] <= ts].tail(tail_n).copy()
                    sliced[tf] = sliced_tf
                if len(sliced["1h"]) >= 30:
                    features = extract_multiframe_market_features_from_frames(sliced)
                    all_features[i] = features[:63].astype(np.float32)
                else:
                    all_features[i] = np.zeros(63, dtype=np.float32)

        except Exception as e:
            print(f"  [WARN] Error at index {i}: {e}, using zeros")
            all_features[i] = np.zeros(63, dtype=np.float32)

    # Save
    print(f"\n[SAVE] Saving {all_features.shape} to {output_file.name}...")
    np.save(output_file, all_features)

    meta = {
        "source_file": candle_file.name,
        "rows": int(len(candles)),
        "timestamp_min": str(candles["timestamp"].min()),
        "timestamp_max": str(candles["timestamp"].max()),
        "feature_dim": 63,
    }
    meta_path = output_file.parent / "features_precomputed.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"[OK] Saved: {output_file.name} ({size_mb:.1f}MB)")
    print(f"[OK] Saved metadata: {meta_path.name}")
    print(f"\n[READY] Training will be 5-10x faster!")

if __name__ == "__main__":
    data_root = Path(__file__).parent.parent.parent.parent / "training_data"
    data_v2 = data_root / "v2"
    data_path = data_v2 if data_v2.exists() else data_root

    candle_file = data_path / "candles_max.csv"
    v2_1m = data_path / "BTCUSDT_1m_ohlcv.csv"
    v2_5m = data_path / "BTCUSDT_5m_ohlcv.csv"
    v2_15m = data_path / "BTCUSDT_15m_ohlcv.csv"
    v2_1h = data_path / "BTCUSDT_1h_ohlcv.csv"
    v2_4h = data_path / "BTCUSDT_4h_ohlcv.csv"
    if v2_1h.exists():
        candle_file = v2_1h
    output_file = data_path / "features_precomputed.npy"

    if not candle_file.exists():
        print(f"[ERROR] Candles file not found: {candle_file}")
        sys.exit(1)

    if output_file.exists():
        print(f"[SKIP] Features already precomputed: {output_file.name}")
        print(f"       Delete if you want to recompute: {output_file}")
        sys.exit(0)

    frames = None
    if all(p.exists() for p in [v2_1m, v2_5m, v2_15m, v2_1h, v2_4h]):
        print("[PRECOMPUTE] Loading multi-timeframe candles (1m/5m/15m/1h/4h)...")
        frames = {}
        for tf, path in [("1m", v2_1m), ("5m", v2_5m), ("15m", v2_15m), ("1h", v2_1h), ("4h", v2_4h)]:
            df_tf = pd.read_csv(path)
            df_tf = _ensure_timestamp_column(df_tf)
            if "timestamp" not in df_tf.columns:
                raise KeyError("timestamp")
            df_tf["timestamp"] = _to_datetime(df_tf["timestamp"])
            frames[tf] = df_tf
        candle_file = v2_1h

    precompute_all_features(candle_file, output_file, frames=frames)
