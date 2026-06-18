"""Generate paper-ready Tables 3, 4, 5 from JSON result files.

Reads:
  - baseline_results.json  (or individual result_*.json files from training_data/)
  - ablation_results.json  (or individual result_w_o_*.json files)
  - faithfulness_results.json

Writes:
  - table3_baselines.csv / table3_baselines.md
  - table4_ablation.csv  / table4_ablation.md
  - table5_faithfulness.csv / table5_faithfulness.md

Usage:
  # From canonical run output:
  python app/v2/pipelines/generate_paper_tables.py \\
      --baseline_json results/baseline_results.json \\
      --ablation_json results/ablation_results.json \\
      --faithfulness_json results/faithfulness_results.json \\
      --output_dir results/paper_tables

  # From individual per-baseline files (training_data/ablation + baseline/):
  python app/v2/pipelines/generate_paper_tables.py \\
      --individual_dir training_data/ablation_plus_baseline \\
      --output_dir results/paper_tables
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
sys.path.insert(0, str(_SCRIPT_DIR))


# ── Display names for models ───────────────────────────────────────────────────
_BASELINE_DISPLAY = {
    "market_only":              "Market-Only MLP",
    "lightgbm_market_only":     "LightGBM (Market-Only)",
    "all_news_fusion":          "All-News Fusion (Avg)",
    "always_alert":             "Always Alert",
    "raw_prob_threshold":       "Raw Prob Threshold",
    "sentiment_market":         "Sentiment + Market",
    "nsm_style":                "NSM-Style",
    "llm_factor":               "LLM-Factor",
    "sep_style":                "SEP-Style",
    "finin_style":              "FinIn-Style",
    "interleaved":              "Interleaved (Article×Market)",
    "temperature_scaled":       "Temp. Scaled",
    "selective_forecasting":    "Selective Forecasting",
    "current_price_predictor":  "Current Price Predictor",
    "all_news_llm_expl":        "All-News + LLM Expl.",
    "random_k_evidence":        "Random-K Evidence",
    "most_recent_k_evidence":   "Most-Recent-K Evidence",
    "SAFE-Alert":               "SAFE-Alert (Ours)",
}

_ABLATION_DISPLAY = {
    "full_model":             "SAFE-Alert Full Model",
    "w/o_selective_news":     "w/o Selective News",
    "w/o_factor":             "w/o Factor Module",
    "w/o_market":             "w/o Market Encoder",
    "w/o_confidence":         "w/o Confidence Head",
    "w/o_faithfulness":       "w/o Faithfulness Loss",
    "w/o_horizon":            "w/o Multi-Timescale",
    "w/o_lrisk":              "w/o Lrisk",
    "w/o_bar_sequences":      "w/o Bar Sequences",
}

# Columns for Table 3 (baselines)
_T3_COLS = [
    ("macro_f1",       "Macro-F1", ".4f"),
    ("mcc",            "MCC",      ".4f"),
    ("auc",            "AUC",      ".4f"),
    ("ece",            "ECE",      ".4f"),
    ("alert_precision","Alert-Prec", ".4f"),
    ("alert_coverage", "Alert-Cov",  ".4f"),
    ("alert_sharpe",   "Sharpe",   ".4f"),
    ("model_score",    "Score",    ".4f"),
]

# Columns for Table 4 (ablation) — superset includes faithfulness metrics
_T4_COLS = [
    ("macro_f1",           "Macro-F1",   ".4f"),
    ("mcc",                "MCC",        ".4f"),
    ("auc",                "AUC",        ".4f"),
    ("ece",                "ECE",        ".4f"),
    ("alert_sharpe",       "Sharpe",     ".4f"),
    ("alert_coverage",     "Alert-Cov",  ".4f"),
    ("alert_max_dd",       "Max-DD",     ".4f"),
    ("comprehensiveness",  "Comp.",      ".4f"),
    ("insertion_gain",     "Ins.",       ".4f"),
    ("sufficiency_drop",   "Suff.",      ".4f"),
    ("factor_consistency", "Fact-Cons.", ".4f"),
]

# Columns for Table 5 (faithfulness)
_T5_ROWS = [
    ("occlusion_test",         "gap_mean",              "Occlusion Gap (mean)"),
    ("occlusion_test",         "fidelity_ratio",        "Fidelity Ratio (> margin)"),
    ("occlusion_test",         "direction_change_ratio","Direction Change Ratio"),
    ("insertion_test",         "insertion_lift_mean",   "Insertion Lift (mean)"),
    ("sufficiency_test",       "confidence_drop_mean",  "Sufficiency Drop (mean)"),
    ("comprehensiveness_test", "confidence_drop_mean",  "Comprehensiveness Drop (mean)"),
    ("factor_consistency_test","jaccard_mean",          "Factor Consistency (Jaccard)"),
    ("fidelity_metric",        "fidelity_correlation",  "Attn-Gap Correlation"),
    ("random_k_comparison",    "selector_lift",         "Selector Lift vs Random-K"),
    ("random_k_comparison",    "selector_better_ratio", "Selector Better Ratio vs Random-K"),
    ("most_recent_k_comparison","selector_lift",        "Selector Lift vs Most-Recent-K"),
    ("most_recent_k_comparison","selector_better_ratio","Selector Better Ratio vs Most-Recent-K"),
]


# ── Loaders ────────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_baseline_results(
    baseline_json: Optional[Path],
    individual_dir: Optional[Path],
) -> dict[str, dict]:
    """Return {model_name: metrics_dict}. _meta key is stripped."""
    if baseline_json and baseline_json.exists():
        raw = _load_json(baseline_json)
        return {k: v for k, v in raw.items() if k != "_meta"}

    if individual_dir and individual_dir.exists():
        results: dict[str, dict] = {}
        # Files named result_<model>.json or result_<model>.json
        for path in sorted(individual_dir.glob("result_*.json")):
            raw = _load_json(path)
            for key, val in raw.items():
                if key == "_meta":
                    continue
                # Skip ablation keys (w_o_ prefix)
                if not key.startswith("w/o_") and key != "full_model":
                    results[key] = val
        return results

    return {}


def load_ablation_results(
    ablation_json: Optional[Path],
    individual_dir: Optional[Path],
) -> dict[str, dict]:
    """Return {variant_name: metrics_dict}."""
    if ablation_json and ablation_json.exists():
        raw = _load_json(ablation_json)
        return {k: v for k, v in raw.items() if k != "_meta"}

    if individual_dir and individual_dir.exists():
        results: dict[str, dict] = {}
        ablation_keys = {"full_model", "w/o_selective_news", "w/o_factor",
                         "w/o_market", "w/o_confidence", "w/o_faithfulness",
                         "w/o_horizon", "w/o_lrisk", "w/o_bar_sequences"}
        # Match both naming conventions
        name_map = {
            "result_full_model.json":        "full_model",
            "result_w_o_selective_news.json": "w/o_selective_news",
            "result_w_o_factor.json":         "w/o_factor",
            "result_w_o_market.json":         "w/o_market",
            "result_w_o_confidence.json":     "w/o_confidence",
            "result_w_o_faithfulness.json":   "w/o_faithfulness",
            "result_w_o_horizon.json":        "w/o_horizon",
            "result_w_o_lrisk.json":          "w/o_lrisk",
            "result_w_o_bar_sequences.json":  "w/o_bar_sequences",
        }
        for path in sorted(individual_dir.glob("result_*.json")):
            canonical = name_map.get(path.name)
            if canonical is None:
                # Try direct key from file contents
                raw = _load_json(path)
                for k, v in raw.items():
                    if k in ablation_keys:
                        results[k] = v
            else:
                raw = _load_json(path)
                for k, v in raw.items():
                    if k == canonical or k in ablation_keys:
                        results[canonical] = v
        return results

    return {}


# ── Formatters ─────────────────────────────────────────────────────────────────
def _fmt(val, fmt: str) -> str:
    if val is None:
        return "—"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    hdr = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    lines = [hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(lines)


def _csv_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [",".join(headers)]
    for row in rows:
        lines.append(",".join(str(c).replace(",", ";") for c in row))
    return "\n".join(lines)


# ── Table 3: Baselines ─────────────────────────────────────────────────────────
def build_table3(baseline_results: dict[str, dict]) -> tuple[list[str], list[list[str]]]:
    headers = ["Model"] + [col[1] for col in _T3_COLS]
    rows = []
    # Canonical order: defined in _BASELINE_DISPLAY, then any extras
    order = list(_BASELINE_DISPLAY.keys())
    present = set(baseline_results.keys())
    sorted_keys = [k for k in order if k in present] + \
                  [k for k in sorted(present) if k not in set(order)]
    for key in sorted_keys:
        metrics = baseline_results[key]
        row = [_BASELINE_DISPLAY.get(key, key)]
        for col_key, _, fmt in _T3_COLS:
            row.append(_fmt(metrics.get(col_key), fmt))
        rows.append(row)
    return headers, rows


# ── Table 4: Ablation ──────────────────────────────────────────────────────────
def build_table4(ablation_results: dict[str, dict]) -> tuple[list[str], list[list[str]]]:
    headers = ["Variant"] + [col[1] for col in _T4_COLS]
    rows = []
    order = list(_ABLATION_DISPLAY.keys())
    present = set(ablation_results.keys())
    sorted_keys = [k for k in order if k in present] + \
                  [k for k in sorted(present) if k not in set(order)]
    for key in sorted_keys:
        metrics = ablation_results[key]
        row = [_ABLATION_DISPLAY.get(key, key)]
        for col_key, _, fmt in _T4_COLS:
            row.append(_fmt(metrics.get(col_key), fmt))
        rows.append(row)
    return headers, rows


# ── Table 5: Faithfulness ──────────────────────────────────────────────────────
def build_table5(faith_results: dict) -> tuple[list[str], list[list[str]]]:
    headers = ["Metric", "Value"]
    rows = []
    for section, key, label in _T5_ROWS:
        section_data = faith_results.get(section, {})
        val = section_data.get(key)
        rows.append([label, _fmt(val, ".4f")])
    return headers, rows


# ── Writer ─────────────────────────────────────────────────────────────────────
def _write_table(output_dir: Path, stem: str,
                 headers: list[str], rows: list[list[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path  = output_dir / f"{stem}.md"
    csv_path = output_dir / f"{stem}.csv"
    md_path.write_text(_markdown_table(headers, rows), encoding="utf-8")
    csv_path.write_text(_csv_table(headers, rows), encoding="utf-8")
    print(f"  [OK] {md_path.name}  {csv_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    script_dir   = Path(__file__).resolve().parent
    service_root = script_dir.parent.parent.parent.parent
    default_data = service_root / "training_data" / "ablation + baseline"
    # Also check the variant path used on Linux server (no space/plus in name)
    if not default_data.exists():
        for alt in ["ablation+baseline", "ablation_baseline", "ablation_plus_baseline"]:
            candidate = service_root / "training_data" / alt
            if candidate.exists():
                default_data = candidate
                break

    parser = argparse.ArgumentParser(
        description="Generate paper Tables 3/4/5 from SAFE-Alert experiment JSON results"
    )
    # Combined JSON files (from canonical SLURM run)
    parser.add_argument("--baseline_json",    type=Path, default=None,
                        help="Combined baselines JSON (baseline_results.json)")
    parser.add_argument("--ablation_json",    type=Path, default=None,
                        help="Combined ablation JSON (ablation_results.json)")
    parser.add_argument("--faithfulness_json",type=Path, default=None,
                        help="Faithfulness eval JSON (faithfulness_results.json)")

    # Alternative: individual per-model files
    parser.add_argument("--individual_dir",   type=Path, default=default_data,
                        help="Directory with individual result_*.json files "
                             "(default: training_data/ablation + baseline/)")

    parser.add_argument("--output_dir",       type=Path,
                        default=service_root / "results" / "paper_tables",
                        help="Output directory for CSV + Markdown tables")
    args = parser.parse_args()

    print("[LOAD] Reading result files...")

    baseline_results = load_baseline_results(args.baseline_json, args.individual_dir)
    ablation_results = load_ablation_results(args.ablation_json, args.individual_dir)
    faith_results    = {}
    if args.faithfulness_json and args.faithfulness_json.exists():
        faith_results = _load_json(args.faithfulness_json)

    if not baseline_results:
        print("[WARN] No baseline results found. Table 3 will be empty.")
    if not ablation_results:
        print("[WARN] No ablation results found. Table 4 will be empty.")
    if not faith_results:
        print("[WARN] No faithfulness results found. Table 5 will be empty.")

    print(f"  Baselines loaded: {list(baseline_results.keys())}")
    print(f"  Ablation loaded:  {list(ablation_results.keys())}")
    print(f"  Faithfulness keys: {[k for k in faith_results if not k.startswith('_')]}")

    print(f"\n[BUILD] Generating tables -> {args.output_dir}")

    h3, r3 = build_table3(baseline_results)
    _write_table(args.output_dir, "table3_baselines", h3, r3)

    h4, r4 = build_table4(ablation_results)
    _write_table(args.output_dir, "table4_ablation", h4, r4)

    h5, r5 = build_table5(faith_results)
    _write_table(args.output_dir, "table5_faithfulness", h5, r5)

    print(f"\n[DONE] Tables saved to {args.output_dir}")
    if baseline_results:
        print(f"\n--- Table 3 Preview (first 5 rows) ---")
        for row in r3[:5]:
            print("  " + " | ".join(f"{c:<22s}" if i == 0 else f"{c:>8s}"
                                     for i, c in enumerate(row)))
    if ablation_results:
        print(f"\n--- Table 4 Preview (all rows) ---")
        for row in r4:
            print("  " + " | ".join(f"{c:<28s}" if i == 0 else f"{c:>8s}"
                                     for i, c in enumerate(row)))


if __name__ == "__main__":
    main()
