"""
Visualize per-epoch training curves from training log text files.

The training log (stdout) contains lines like:
  Epoch  13/60 | train=1.04 val=0.97 acc=0.607 | ...
  [Metrics] F1=0.573 MCC=0.364 ECE=0.017 AUC=0.769 | Sharpe=0.123 Cov=0.393 ...

Usage:
    # Redirect training output to a log file first:
    python train_safe_alert.py ... 2>&1 | tee training_log.txt

    # Then visualize:
    python app/v2/visualization/visualize_training_curves.py --log training_log.txt
    python app/v2/visualization/visualize_training_curves.py --log training_log.txt --fold 4
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AI_ROOT    = SCRIPT_DIR.parent.parent.parent
DEFAULT_LOG = AI_ROOT / "training_log.txt"
DEFAULT_OUT = AI_ROOT / "artifacts/colab_run/figures"

STYLE = {
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
}

# Regex patterns
RE_FOLD  = re.compile(r"\[FOLD (\d+)/(\d+)\]")
RE_EPOCH = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*train=([\d.]+)\s+val=([\d.]+)\s+acc=([\d.]+)"
)
RE_METRICS = re.compile(
    r"\[Metrics\].*?F1=([\d.]+).*?MCC=([\d.-]+).*?ECE=([\d.]+).*?AUC=([\d.]+)"
    r".*?\|\s*Sharpe=([\d.-]+).*?Cov=([\d.]+)"
)
RE_STAGE2 = re.compile(r"\[Ldirx1\.00.*?Lfacx")
RE_STAGE3 = re.compile(r"\[STAGE 3\]")
RE_BEST   = re.compile(r"\[BEST\]\s+Score=([\d.]+)")


def parse_log(log_path: Path) -> dict[int, dict]:
    """Returns {fold_num: {metric_name: [epoch_values]}}."""
    folds: dict[int, dict] = {}
    current_fold = None
    pending_epoch = None

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        m = RE_FOLD.search(line)
        if m:
            current_fold = int(m.group(1))
            folds[current_fold] = {k: [] for k in [
                "epoch", "train_loss", "val_loss", "val_acc",
                "f1", "mcc", "ece", "auc", "sharpe", "coverage",
                "stage", "is_best",
            ]}
            continue

        if current_fold is None:
            continue

        m = RE_EPOCH.search(line)
        if m:
            pending_epoch = {
                "epoch":      int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss":   float(m.group(3)),
                "val_acc":    float(m.group(4)),
            }
            # Determine stage from lambda line (appears just after epoch header)
            if "Lfacx" in line:
                pending_epoch["stage"] = 2 if "Lcalx0.0" in line or "Lcalx0.01" in line else 3
            else:
                pending_epoch["stage"] = 1
            pending_epoch["is_best"] = False
            continue

        if pending_epoch and RE_STAGE3.search(line):
            pending_epoch["stage"] = 3

        if pending_epoch and RE_BEST.search(line):
            pending_epoch["is_best"] = True

        m = RE_METRICS.search(line)
        if m and pending_epoch:
            pending_epoch.update({
                "f1":       float(m.group(1)),
                "mcc":      float(m.group(2)),
                "ece":      float(m.group(3)),
                "auc":      float(m.group(4)),
                "sharpe":   float(m.group(5)),
                "coverage": float(m.group(6)),
            })
            for k in ["epoch", "train_loss", "val_loss", "val_acc",
                      "f1", "mcc", "ece", "auc", "sharpe", "coverage",
                      "stage", "is_best"]:
                folds[current_fold][k].append(pending_epoch.get(k, None))
            pending_epoch = None

    return folds


def plot_fold(fold_num: int, data: dict, out: Path, show: bool) -> None:
    epochs   = data["epoch"]
    stages   = data["stage"]
    is_best  = data["is_best"]

    if not epochs:
        print(f"  No data for Fold {fold_num}, skipping.")
        return

    stage_colors = {1: "#e8f4f8", 2: "#fef9e7", 3: "#fdecea"}

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"SAFE-Alert — Training Curves (Fold {fold_num})", fontsize=14, fontweight="bold")
        axes = axes.flatten()

        panels = [
            ("train_loss", "val_loss", "Train vs Val Loss", "Loss"),
            ("val_acc",    None,       "Val Accuracy",      "Accuracy"),
            ("f1",         None,       "Val F1",            "F1"),
            ("sharpe",     None,       "Alert Sharpe",      "Sharpe"),
            ("coverage",   None,       "Alert Coverage",    "Coverage"),
            ("ece",        None,       "ECE (Calibration)", "ECE"),
        ]

        for ax, (key1, key2, title, ylabel) in zip(axes, panels):
            # Stage background shading
            prev_s, prev_e = stages[0] if stages else 1, epochs[0] if epochs else 1
            for i, (e, s) in enumerate(zip(epochs, stages)):
                if s != prev_s or i == len(epochs) - 1:
                    ax.axvspan(prev_e - 0.5, e + 0.5, alpha=0.25,
                               color=stage_colors.get(prev_s, "white"), zorder=0)
                    prev_s, prev_e = s, e

            y1 = [v for v in data[key1] if v is not None]
            ep = [e for e, v in zip(epochs, data[key1]) if v is not None]
            ax.plot(ep, y1, color="#2980b9", linewidth=1.8, label=key1)

            if key2 and data.get(key2):
                y2 = [v for v in data[key2] if v is not None]
                ax.plot(epochs, y2, color="#e74c3c", linewidth=1.8, linestyle="--", label=key2)

            # Mark best epochs
            best_eps = [e for e, b, v in zip(epochs, is_best, data[key1]) if b and v is not None]
            best_vs  = [v for b, v in zip(is_best, data[key1]) if b and v is not None]
            if best_eps:
                ax.scatter(best_eps, best_vs, color="gold", s=60, zorder=5, label="Best Score")

            # Coverage target
            if key1 == "coverage":
                ax.axhline(0.35, color="#e74c3c", linewidth=1.2, linestyle=":", label="Target 35%")

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            if key2 or best_eps or key1 == "coverage":
                ax.legend(fontsize=8)

            # Stage labels
            for stage_n, stage_label, color in [(1, "S1", "#2980b9"), (2, "S2", "#f39c12"), (3, "S3", "#e74c3c")]:
                stage_eps = [e for e, s in zip(epochs, stages) if s == stage_n]
                if stage_eps:
                    mid = stage_eps[len(stage_eps) // 2]
                    ax.text(mid, ax.get_ylim()[1] * 0.97, stage_label,
                            ha="center", va="top", fontsize=8, color=color, alpha=0.7)

        fig.tight_layout()
        path = out / f"fig_training_fold{fold_num}.png"
        _save(fig, path, show)


def _save(fig, path, show):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"  Saved → {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",  type=Path, default=DEFAULT_LOG)
    parser.add_argument("--out",  type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fold", type=int,  default=None, help="Specific fold only")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Log file not found: {args.log}")
        print("Redirect training output to a file: python train_safe_alert.py ... 2>&1 | tee training_log.txt")
        return

    print(f"Parsing: {args.log}")
    folds = parse_log(args.log)
    print(f"Found folds: {sorted(folds.keys())}")

    targets = [args.fold] if args.fold else sorted(folds.keys())
    for fold_num in targets:
        if fold_num in folds:
            plot_fold(fold_num, folds[fold_num], args.out, args.show)
        else:
            print(f"  Fold {fold_num} not found in log.")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
