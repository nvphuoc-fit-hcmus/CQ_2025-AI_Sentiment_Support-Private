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

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.market_features import extract_multiframe_market_features

def precompute_all_features(candle_file, output_file):
    """Precompute all 52,542 candles' features."""

    print("[PRECOMPUTE] Loading candles...")
    candles = pd.read_csv(candle_file)
    candles['timestamp'] = pd.to_datetime(candles['timestamp'])
    print(f"  Loaded {len(candles)} candles")

    # Initialize output array
    all_features = np.zeros((len(candles), 63), dtype=np.float32)

    print(f"\n[PRECOMPUTE] Computing features for {len(candles)} candles...")
    print(f"  This will take 5-10 minutes on CPU, 1-2 minutes on GPU\n")

    for i in range(len(candles)):
        # Progress
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(candles)} ({(i+1)/len(candles)*100:.1f}%)")

        try:
            # Look back 250 candles (standard window)
            lookback_idx = max(0, i - 250)
            df_window = candles.iloc[lookback_idx:i+1].copy()

            if len(df_window) >= 50:
                # Extract 63-dim features
                features = extract_multiframe_market_features(df_window)
                all_features[i] = features[:63].astype(np.float32)
            else:
                # Not enough data, use zeros (rare at start)
                all_features[i] = np.zeros(63, dtype=np.float32)

        except Exception as e:
            print(f"  [WARN] Error at index {i}: {e}, using zeros")
            all_features[i] = np.zeros(63, dtype=np.float32)

    # Save
    print(f"\n[SAVE] Saving {all_features.shape} to {output_file.name}...")
    np.save(output_file, all_features)

    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"[OK] Saved: {output_file.name} ({size_mb:.1f}MB)")
    print(f"\n[READY] Training will be 5-10x faster!")

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent.parent / "training_data"

    candle_file = data_path / "candles_max.csv"
    output_file = data_path / "features_precomputed.npy"

    if not candle_file.exists():
        print(f"[ERROR] Candles file not found: {candle_file}")
        sys.exit(1)

    if output_file.exists():
        print(f"[SKIP] Features already precomputed: {output_file.name}")
        print(f"       Delete if you want to recompute: {output_file}")
        sys.exit(0)

    precompute_all_features(candle_file, output_file)
