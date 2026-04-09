#!/usr/bin/env python3
"""
Fresh training script - clean slate, no old checkpoints interfering.
Deletes old epochs, trains with maximum dataset (8,722 articles).
"""

import sys
import shutil
from pathlib import Path
import subprocess

# Paths
SCRIPT_DIR = Path(__file__).parent
ARTIFACT_DIR = SCRIPT_DIR.parent.parent.parent / "artifacts/v2"

print("=" * 70)
print("CLEAN TRAINING INITIALIZATION")
print("=" * 70)

# Clean up old checkpoints
print("\n[CLEANUP] Removing old checkpoints...")
for pattern in ["safe_alert_btcusdt_1h_epoch*.pt"]:
    for f in ARTIFACT_DIR.glob(pattern):
        f.unlink()
        print(f"  Deleted: {f.name}")

print(f"\n[TRAINING] Starting fresh training with maximum dataset...")
print(f"  Device: CPU (training will be slower)")
print(f"  Dataset: 8,722 articles + 52,542 candles")
print(f"  Architecture: 63 market features, 768-dim article embeddings")
print(f"  Epochs: 20, Batch size: 32, LR: 0.001")
print(f"  Patience: 5 (early stopping)")
print("")

# Run training
cmd = [
    sys.executable,
    "train_safe_alert.py",
    "--symbol", "BTCUSDT",
    "--horizon", "1h",
    "--epochs", "20",
    "--lr", "0.001"
]

result = subprocess.run(cmd, cwd=SCRIPT_DIR)
sys.exit(result.returncode)
