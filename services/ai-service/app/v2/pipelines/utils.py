"""
Utility functions shared across SAFE-Alert pipelines.
"""
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Paths
CURRENT_FILE = Path(__file__).resolve()
PIPELINES_DIR = CURRENT_FILE.parent
V2_DIR = PIPELINES_DIR.parent
APP_DIR = V2_DIR.parent
SERVICE_ROOT = APP_DIR.parent

ARTIFACT_DIR = SERVICE_ROOT / "artifacts" / "v2"
DATA_PATH = SERVICE_ROOT / "training_data" / "processed" / "btcusdt_training_dataset_v2.csv"
OUTPUT_PATH = ARTIFACT_DIR / "latest_signal.json"


def load_safe_alert_policy(symbol: str, horizon: str) -> dict:
    """Load SAFE-Alert decision thresholds (tau_h, gamma_h, etc.) from policy JSON.

    Args:
        symbol: e.g., "BTCUSDT"
        horizon: "1h" or "4h"

    Returns:
        dict with keys: "tau" (high threshold), "gamma" (medium threshold), etc.
    """
    policy_path = ARTIFACT_DIR / f"safe_alert_{symbol.lower()}_{horizon}_policy.json"

    # Default policy if file doesn't exist
    default_policy = {
        "tau": 0.80,      # High alert threshold
        "gamma": 0.50,    # Medium alert threshold
        "symbol": symbol,
        "horizon": horizon,
    }

    if not policy_path.exists():
        logger.warning(f"Policy file not found: {policy_path}. Using defaults: {default_policy}")
        return default_policy

    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
        logger.info(f"Loaded policy from {policy_path}")
        return policy
    except Exception as e:
        logger.error(f"Failed to load policy: {e}. Using defaults.")
        return default_policy
