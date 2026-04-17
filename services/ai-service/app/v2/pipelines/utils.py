"""
Utility functions shared across SAFE-Alert pipelines.
"""
import json
import logging
from pathlib import Path
from typing import Any

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

DEFAULT_POLICY_MAP = {
    "15m": {"tau": 0.80, "gamma": 0.75, "temperature": 1.0},
    "1h": {"tau": 0.78, "gamma": 0.72, "temperature": 1.0},
    "4h": {"tau": 0.75, "gamma": 0.70, "temperature": 1.0},
    "24h": {"tau": 0.72, "gamma": 0.68, "temperature": 1.0},
}


def _default_policy(symbol: str, horizon: str) -> dict[str, Any]:
    base = DEFAULT_POLICY_MAP.get(horizon, DEFAULT_POLICY_MAP["1h"])
    return {
        "tau": float(base["tau"]),
        "gamma": float(base["gamma"]),
        "temperature": float(base.get("temperature", 1.0)),
        "symbol": symbol.upper(),
        "horizon": horizon,
        "source": "default_threshold_map",
    }


def _policy_candidate_paths(symbol: str, horizon: str, artifact_dir: Path | None = None) -> list[Path]:
    sym = symbol.lower()
    artifact_dir = artifact_dir or ARTIFACT_DIR
    return [
        artifact_dir / f"safe_alert_{sym}_{horizon}_policy.json",
        artifact_dir / f"safe_alert_policy_{sym}_{horizon}.json",
    ]


def load_safe_alert_policy(symbol: str, horizon: str) -> dict:
    """Load SAFE-Alert decision thresholds (tau_h, gamma_h, etc.) from policy JSON.

    Args:
        symbol: e.g., "BTCUSDT"
        horizon: "1h" or "4h"

    Returns:
        dict with keys: "tau" (high threshold), "gamma" (medium threshold), etc.
    """
    default_policy = _default_policy(symbol, horizon)
    for policy_path in _policy_candidate_paths(symbol, horizon):
        if not policy_path.exists():
            continue
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy = json.load(f)
            logger.info("Loaded policy from %s", policy_path)
            return policy
        except Exception as e:
            logger.error("Failed to load policy %s: %s", policy_path, e)

    logger.warning("Policy file not found for %s/%s. Using defaults: %s", symbol, horizon, default_policy)
    return default_policy


def save_safe_alert_policy(
    policy: dict[str, Any],
    symbol: str,
    horizon: str,
    artifact_dir: Path | None = None,
) -> list[Path]:
    """Persist canonical + legacy policy filenames for compatibility."""
    policy = dict(policy)
    policy["symbol"] = symbol.upper()
    policy["horizon"] = horizon
    artifact_dir = artifact_dir or ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in _policy_candidate_paths(symbol, horizon, artifact_dir):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(policy, f, indent=2, ensure_ascii=False)
        written.append(path)
    return written


def standardize_walk_forward_artifacts(
    walk_forward_path: Path,
    symbol: str,
    horizon: str,
    artifact_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Create canonical deploy-policy and summary artifacts from walk-forward results."""
    artifact_dir = artifact_dir or ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not walk_forward_path.exists():
        logger.warning("Walk-forward artifact missing: %s", walk_forward_path)
        return None

    with open(walk_forward_path, "r", encoding="utf-8") as f:
        wf = json.load(f)

    folds = wf.get("fold_metrics", [])
    if not folds:
        logger.warning("No fold metrics in %s", walk_forward_path)
        return None

    def _score(rec: dict[str, Any]) -> float:
        val = rec.get("val", {})
        if "model_score" in val:
            return float(val.get("model_score", float("-inf")))
        acc = float(val.get("final_val_acc", 0.0))
        loss = float(val.get("final_val_loss", 0.0))
        return acc - 0.05 * loss

    best = max(folds, key=_score)
    best_val = best.get("val", {})
    best_test = best.get("test", {})

    policy = {
        "tau": float(best_val.get("tau", _default_policy(symbol, horizon)["tau"])),
        "gamma": float(best_val.get("gamma", _default_policy(symbol, horizon)["gamma"])),
        "temperature": float(best_val.get("temperature", _default_policy(symbol, horizon)["temperature"])),
        "tau_seed": float(best_val.get("tau_seed", _default_policy(symbol, horizon)["tau"])),
        "gamma_seed": float(best_val.get("gamma_seed", _default_policy(symbol, horizon)["gamma"])),
        "temperature_nll": float(best_val.get("temperature_nll", 0.0)),
        "temperature_source": str(best_val.get("temperature_source", "validation_temperature_scaling")),
        "coverage": float(best_val.get("alert_coverage", best_test.get("alert_coverage", 0.0))),
        "alert_precision": float(best_val.get("alert_precision", best_test.get("alert_precision", 0.0))),
        "val_sharpe_proxy": float(best_val.get("alert_sharpe", best_test.get("alert_sharpe", 0.0))),
        "policy_source": str(best_val.get("policy_source", "validation_grid_search")),
        "policy_objective": float(best_val.get("policy_objective", best_val.get("model_score", _score(best)))),
        "target_coverage": 0.35,
        "score": float(best_val.get("model_score", _score(best))),
        "deploy_fold": int(best.get("fold", -1)),
        "mode": "walk_forward_best_fold",
        "source_artifact": str(walk_forward_path),
        "note": "Thresholds frozen from validation split of best walk-forward fold.",
    }
    save_safe_alert_policy(policy, symbol, horizon, artifact_dir=artifact_dir)

    summary = {
        "source": str(walk_forward_path),
        "selected_deployment_fold": int(best.get("fold", -1)),
        "deployment_policy": policy,
        "metric_groups": {
            "forecast": {
                "macro_f1": float(best_test.get("macro_f1", 0.0)),
                "mcc": float(best_test.get("mcc", 0.0)),
                "auc": float(best_test.get("auc", 0.0)),
            },
            "alert": {
                "alert_precision": float(best_test.get("alert_precision", 0.0)),
                "alert_coverage": float(best_test.get("alert_coverage", 0.0)),
                "tau": float(best_test.get("tau", policy["tau"])),
                "gamma": float(best_test.get("gamma", policy["gamma"])),
                "temperature": float(best_test.get("temperature", policy["temperature"])),
            },
            "utility": {
                "alert_sharpe": float(best_test.get("alert_sharpe", 0.0)),
                "alert_sortino": float(best_test.get("alert_sortino", 0.0)),
                "alert_calmar": float(best_test.get("alert_calmar", 0.0)),
                "alert_max_dd": float(best_test.get("alert_max_dd", 0.0)),
                "model_score": float(best_test.get("model_score", 0.0)),
            },
        },
        "folds": [
            {"fold": int(rec.get("fold", -1)), **rec.get("test", {})}
            for rec in folds
        ],
        "averages": wf.get("averages", {}),
    }
    with open(artifact_dir / "table3_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
