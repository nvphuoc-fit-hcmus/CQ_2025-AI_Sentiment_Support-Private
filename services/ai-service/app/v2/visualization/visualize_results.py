"""
Thesis visualization: SAFE-Alert walk-forward results.

Usage (from ai-service root):
    python app/v2/visualization/visualize_results.py
    python app/v2/visualization/visualize_results.py --results artifacts/colab_run/walk_forward_results.json
    python app/v2/visualization/visualize_results.py --show

Outputs to: artifacts/colab_run/figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
AI_ROOT      = SCRIPT_DIR.parent.parent.parent          # services/ai-service
DEFAULT_JSON = AI_ROOT / "artifacts/colab_run/walk_forward_results.json"
DEFAULT_FIGS = AI_ROOT / "artifacts/colab_run/figures"

# ── Style ──────────────────────────────────────────────────────────────────────
FOLD_COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
AVG_COLOR    = "#2d2d2d"
TARGET_COLOR = "#e74c3c"

STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
}


def _save(fig: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"  Saved → {path.name}")
    if show:
        plt.show()
    plt.close(fig)


# ── Figure 1: Walk-forward test metrics per fold ───────────────────────────────
def fig_walkforward_metrics(folds: list[dict], avgs: dict, out: Path, show: bool) -> None:
    metrics = [
        ("macro_f1",        "Macro F1",        None,  0, 1),
        ("auc",             "AUC-ROC",         None,  0, 1),
        ("mcc",             "MCC",             None, -1, 1),
        ("ece",             "ECE ↓",           None,  0, 0.15),
        ("alert_sharpe",    "Alert Sharpe",    None, -0.1, 0.6),
        ("alert_coverage",  "Alert Coverage",  0.35,  0, 0.6),
    ]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("SAFE-Alert — Walk-forward Test Metrics (5 Folds)", fontsize=15, fontweight="bold", y=1.01)
        axes = axes.flatten()

        fold_nums = [f["fold"] for f in folds]
        x = np.arange(len(fold_nums))
        width = 0.6

        for ax, (key, label, target, ymin, ymax) in zip(axes, metrics):
            vals = [f[key] for f in folds]
            avg  = avgs.get(f"test_{key}", np.mean(vals))

            bars = ax.bar(x, vals, width=width,
                          color=FOLD_COLORS[:len(folds)], alpha=0.85, edgecolor="white")

            # value labels
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (ymax - ymin) * 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)

            # avg line
            ax.axhline(avg, color=AVG_COLOR, linewidth=1.8, linestyle="--", label=f"Avg={avg:.3f}")

            # target line if specified
            if target is not None:
                ax.axhline(target, color=TARGET_COLOR, linewidth=1.4, linestyle=":", label=f"Target={target}")

            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
            ax.set_ylim(ymin, ymax)
            ax.legend(fontsize=8, loc="lower right")

        fig.tight_layout()
        _save(fig, out / "fig1_walkforward_metrics.png", show)


# ── Figure 2: Financial performance ────────────────────────────────────────────
def fig_financial(folds: list[dict], avgs: dict, out: Path, show: bool) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        fig.suptitle("SAFE-Alert — Financial Performance per Fold", fontsize=14, fontweight="bold")

        panels = [
            ("pnl",             "PnL (×Initial Capital)", None, None),
            ("alert_sharpe",    "Alert Sharpe Ratio",     None, None),
            ("alert_sortino",   "Alert Sortino Ratio",    None, None),
            ("alert_max_dd",    "Max Drawdown",           None, None),
        ]

        fold_nums = [f["fold"] for f in folds]
        x = np.arange(len(fold_nums))
        width = 0.55

        for ax, (key, label, ymin, ymax) in zip(axes, panels):
            vals = [f[key] for f in folds]
            avg  = avgs.get(f"test_{key}", np.mean(vals))

            colors = [FOLD_COLORS[i] for i in range(len(folds))]
            bars   = ax.bar(x, vals, width=width, color=colors, alpha=0.85, edgecolor="white")

            for bar, v in zip(bars, vals):
                offset = 0.05 if v >= 0 else -0.05
                va = "bottom" if v >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        f"{v:.3f}", ha="center", va=va, fontsize=9)

            ax.axhline(avg, color=AVG_COLOR, linewidth=1.8, linestyle="--", label=f"Avg={avg:.3f}")
            ax.axhline(0,   color="gray",    linewidth=0.8)

            ax.set_title(label)
            ax.set_xticks(x)
            ax.set_xticklabels([f"F{n}" for n in fold_nums])
            ax.legend(fontsize=8)

        fig.tight_layout()
        _save(fig, out / "fig2_financial_performance.png", show)


# ── Figure 3: Radar chart ──────────────────────────────────────────────────────
def fig_radar(folds: list[dict], avgs: dict, out: Path, show: bool) -> None:
    # Metrics normalized 0→1 (higher = better)
    def norm_ece(v):   return max(0, 1 - v / 0.15)
    def norm_mdd(v):   return max(0, 1 + v)          # MDD is negative
    def norm_sharpe(v): return max(0, min(1, v / 0.5))

    categories = ["Macro F1", "AUC-ROC", "MCC", "Calibration\n(1−ECE)", "Alert Sharpe", "Coverage\n≈35%"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    def fold_vals(f):
        cov_score = 1 - abs(f["alert_coverage"] - 0.35) / 0.35
        return [
            f["macro_f1"],
            f["auc"] - 0.5,                              # normalize from 0.5
            (f["mcc"] + 1) / 2,
            norm_ece(f["ece"]),
            norm_sharpe(f["alert_sharpe"]),
            max(0, cov_score),
        ]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw={"polar": True})
        fig.suptitle("SAFE-Alert — Per-fold Radar (Normalised)", fontsize=13, fontweight="bold")

        for i, f in enumerate(folds):
            vals = fold_vals(f)
            vals += vals[:1]
            ax.plot(angles, vals, color=FOLD_COLORS[i], linewidth=1.5, alpha=0.7)
            ax.fill(angles, vals, color=FOLD_COLORS[i], alpha=0.08)

        # Average
        avg_vals = [np.mean([fold_vals(f)[j] for f in folds]) for j in range(N)]
        avg_vals += avg_vals[:1]
        ax.plot(angles, avg_vals, color=AVG_COLOR, linewidth=2.5, linestyle="--")
        ax.fill(angles, avg_vals, color=AVG_COLOR, alpha=0.12)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8)

        legend_handles = [mpatches.Patch(color=FOLD_COLORS[i], alpha=0.7, label=f"Fold {f['fold']}") for i, f in enumerate(folds)]
        legend_handles.append(mpatches.Patch(color=AVG_COLOR, alpha=0.5, label="Average"))
        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        fig.tight_layout()
        _save(fig, out / "fig3_radar_chart.png", show)


# ── Figure 4: Policy parameters per fold ──────────────────────────────────────
def fig_policy(folds: list[dict], out: Path, show: bool) -> None:
    fold_nums = [f["fold"] for f in folds]
    taus      = [f["tau"]         for f in folds]
    gammas    = [f["gamma"]       for f in folds]
    temps     = [f["temperature"] for f in folds]
    coverages = [f["alert_coverage"] for f in folds]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        fig.suptitle("SAFE-Alert — Policy Parameters per Fold (Frozen from Val)", fontsize=13, fontweight="bold")

        x = np.arange(len(fold_nums))
        w = 0.5

        # τ and γ together
        ax = axes[0]
        ax.bar(x - 0.18, taus,   width=0.35, color="#4C72B0", alpha=0.85, label="τ (conf threshold)")
        ax.bar(x + 0.18, gammas, width=0.35, color="#DD8452", alpha=0.85, label="γ (max_prob threshold)")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_ylim(0, 1); ax.set_title("Alert Thresholds τ & γ")
        ax.legend(fontsize=9)

        # Temperature
        ax = axes[1]
        bars = ax.bar(x, temps, width=w, color=FOLD_COLORS[:len(folds)], alpha=0.85)
        for bar, v in zip(bars, temps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", label="T=1 (no scaling)")
        ax.set_xticks(x); ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_title("Temperature Scaling"); ax.legend(fontsize=9)

        # Coverage vs target
        ax = axes[2]
        bars = ax.bar(x, coverages, width=w, color=FOLD_COLORS[:len(folds)], alpha=0.85)
        for bar, v in zip(bars, coverages):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=10)
        ax.axhline(0.35, color=TARGET_COLOR, linewidth=1.8, linestyle="--", label="Target 35%")
        ax.set_ylim(0, 0.6); ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_title("Alert Coverage (Test)"); ax.legend(fontsize=9)

        fig.tight_layout()
        _save(fig, out / "fig4_policy_params.png", show)


# ── Figure 5: Faithfulness / Explainability ────────────────────────────────────
def fig_faithfulness(folds: list[dict], avgs: dict, out: Path, show: bool) -> None:
    fold_nums = [f["fold"] for f in folds]
    x = np.arange(len(fold_nums))
    w = 0.18

    comp   = [f["comprehensiveness"]  for f in folds]
    ins    = [f["insertion_gain"]     for f in folds]
    suf    = [f["sufficiency_drop"]   for f in folds]
    fcons  = [f["factor_consistency"] for f in folds]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("SAFE-Alert — Faithfulness & Explainability Metrics (Test)", fontsize=13, fontweight="bold")

        # Occlusion metrics
        ax1.bar(x - w, comp, width=w, color="#4C72B0", alpha=0.85, label="Comprehensiveness")
        ax1.bar(x,     ins,  width=w, color="#55A868", alpha=0.85, label="Insertion Gain")
        ax1.bar(x + w, suf,  width=w, color="#DD8452", alpha=0.85, label="Sufficiency Drop")

        avg_comp = avgs.get("test_comprehensiveness", np.mean(comp))
        avg_ins  = np.mean(ins)
        avg_suf  = avgs.get("test_sufficiency_drop", np.mean(suf))
        ax1.axhline(avg_comp, color="#4C72B0", linewidth=1.2, linestyle="--", alpha=0.6)
        ax1.axhline(avg_ins,  color="#55A868", linewidth=1.2, linestyle="--", alpha=0.6)
        ax1.axhline(avg_suf,  color="#DD8452", linewidth=1.2, linestyle="--", alpha=0.6)

        ax1.set_xticks(x); ax1.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax1.set_title("Occlusion-based Faithfulness"); ax1.legend(fontsize=9)
        ax1.set_ylabel("Score (higher = more faithful)")

        # Factor consistency
        bars = ax2.bar(x, fcons, width=0.5, color=FOLD_COLORS[:len(folds)], alpha=0.85)
        for bar, v in zip(bars, fcons):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
        avg_fc = avgs.get("test_factor_consistency", np.mean(fcons))
        ax2.axhline(avg_fc, color=AVG_COLOR, linewidth=1.8, linestyle="--", label=f"Avg={avg_fc:.3f}")
        ax2.set_xticks(x); ax2.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax2.set_title("Factor Consistency"); ax2.legend(fontsize=9)
        ax2.set_ylabel("Consistency Score")
        ax2.set_ylim(0, 1)

        fig.tight_layout()
        _save(fig, out / "fig5_faithfulness.png", show)


# ── Figure 6: Val vs Test score comparison ─────────────────────────────────────
def fig_val_test_score(wf_data: dict, out: Path, show: bool) -> None:
    fold_metrics = wf_data["fold_metrics"]
    folds_test   = wf_data["fold_metrics"]

    val_scores  = [f["val"]["model_score"]  for f in fold_metrics]
    test_scores = [f["test"]["model_score"] for f in fold_metrics]
    fold_nums   = [f["fold"] for f in fold_metrics]
    x = np.arange(len(fold_nums))
    w = 0.3

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle("SAFE-Alert — Validation vs Test Model Score", fontsize=13, fontweight="bold")

        ax.bar(x - w / 2, val_scores,  width=w, color="#4C72B0", alpha=0.85, label="Validation Score")
        ax.bar(x + w / 2, test_scores, width=w, color="#55A868", alpha=0.85, label="Test Score")

        for i, (v, t) in enumerate(zip(val_scores, test_scores)):
            ax.text(i - w / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            ax.text(i + w / 2, t + 0.003, f"{t:.3f}", ha="center", va="bottom", fontsize=9)

        ax.axhline(np.mean(val_scores),  color="#4C72B0", linewidth=1.5, linestyle="--",
                   alpha=0.7, label=f"Val avg={np.mean(val_scores):.3f}")
        ax.axhline(np.mean(test_scores), color="#55A868", linewidth=1.5, linestyle="--",
                   alpha=0.7, label=f"Test avg={np.mean(test_scores):.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {n}" for n in fold_nums])
        ax.set_ylabel("Model Score (0.40·F1 + 0.35·Sharpe/2 − 0.15·ECE + 0.10·MCC)")
        ax.set_ylim(0, 0.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        _save(fig, out / "fig6_val_vs_test_score.png", show)


# ── Figure 7: Summary table as figure ─────────────────────────────────────────
def fig_summary_table(folds: list[dict], avgs: dict, out: Path, show: bool) -> None:
    cols   = ["Fold", "F1", "AUC", "MCC", "ECE", "Sharpe", "Coverage", "MDD", "PnL", "Score"]
    rows   = []
    for f in folds:
        rows.append([
            f"Fold {f['fold']}",
            f"{f['macro_f1']:.3f}",
            f"{f['auc']:.3f}",
            f"{f['mcc']:.3f}",
            f"{f['ece']:.3f}",
            f"{f['alert_sharpe']:.3f}",
            f"{f['alert_coverage']:.1%}",
            f"{f['alert_max_dd']:.3f}",
            f"{f['pnl']:.2f}×",
            f"{f['model_score']:.3f}",
        ])
    rows.append([
        "Average",
        f"{avgs['test_macro_f1']:.3f}",
        f"{avgs['test_auc']:.3f}",
        f"{avgs['test_mcc']:.3f}",
        f"{avgs['test_ece']:.3f}",
        f"{avgs['test_alert_sharpe']:.3f}",
        f"{avgs['test_alert_coverage']:.1%}",
        f"{avgs['test_alert_max_dd']:.3f}",
        f"{avgs['test_pnl']:.2f}×",
        f"{avgs['test_model_score']:.3f}",
    ])

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(14, 3.5))
        fig.suptitle("SAFE-Alert — Walk-forward Test Results Summary", fontsize=13, fontweight="bold")
        ax.axis("off")

        tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.8)

        # Header style
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#2d3436")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Average row style
        for j in range(len(cols)):
            tbl[len(folds) + 1, j].set_facecolor("#dfe6e9")
            tbl[len(folds) + 1, j].set_text_props(fontweight="bold")

        # Fold 3 highlight (weak fold)
        for j in range(len(cols)):
            tbl[3, j].set_facecolor("#ffeaa7")

        fig.tight_layout()
        _save(fig, out / "fig7_summary_table.png", show)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output",  type=Path, default=DEFAULT_FIGS)
    parser.add_argument("--show",    action="store_true", help="Show plots interactively")
    args = parser.parse_args()

    print(f"Loading: {args.results}")
    with open(args.results) as f:
        wf = json.load(f)

    folds = wf["fold_metrics"]
    # Build flat fold test dicts with all fields accessible at top level
    flat_folds = []
    for fold_entry in folds:
        d = {"fold": fold_entry["fold"]}
        d.update(fold_entry["test"])
        flat_folds.append(d)

    avgs = wf["averages"]
    out  = args.output
    show = args.show

    print(f"Output:  {out}\n")
    fig_walkforward_metrics(flat_folds, avgs, out, show)
    fig_financial(flat_folds, avgs, out, show)
    fig_radar(flat_folds, avgs, out, show)
    fig_policy(flat_folds, out, show)
    fig_faithfulness(flat_folds, avgs, out, show)
    fig_val_test_score(wf, out, show)
    fig_summary_table(flat_folds, avgs, out, show)

    print(f"\nDone — {len(list(out.glob('*.png')))} figures saved to {out}")


if __name__ == "__main__":
    main()
