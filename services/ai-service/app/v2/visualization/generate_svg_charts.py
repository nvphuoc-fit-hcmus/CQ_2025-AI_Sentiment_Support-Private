"""
Pure Python SVG chart generator — zero external dependencies.
Chạy được kể cả khi numpy/matplotlib bị hỏng.

Usage:
    python app/v2/visualization/generate_svg_charts.py
"""

import os
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent.parent / "artifacts/colab_run/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Data từ walk_forward_results.json ─────────────────────────────────────────
FOLDS = [
    {"fold":1,  "f1":0.4542,"auc":0.6474,"mcc":0.2039,"ece":0.0114,
     "sharpe":0.3660,"coverage":0.3304,"mdd":-0.1001,"pnl":5.253,
     "score":0.3042,"sortino":0.8312,"precision":0.5892,
     "tau":0.529,"gamma":0.504,"temp":2.8,
     "comp":0.0648,"ins":0.0804,"suf":0.0199,"fcons":0.6236},
    {"fold":2,  "f1":0.5384,"auc":0.7190,"mcc":0.3090,"ece":0.0223,
     "sharpe":0.3861,"coverage":0.3748,"mdd":-0.0709,"pnl":5.054,
     "score":0.3450,"sortino":0.8880,"precision":0.6525,
     "tau":0.504,"gamma":0.547,"temp":1.8,
     "comp":0.0654,"ins":0.0736,"suf":0.0135,"fcons":0.5551},
    {"fold":3,  "f1":0.5697,"auc":0.7615,"mcc":0.3455,"ece":0.0239,
     "sharpe":0.0430,"coverage":0.4039,"mdd":-0.2421,"pnl":0.303,
     "score":0.2991,"sortino":0.1421,"precision":0.8074,
     "tau":0.497,"gamma":0.677,"temp":1.1,
     "comp":0.0615,"ins":0.0686,"suf":0.0128,"fcons":0.3168},
    {"fold":4,  "f1":0.5507,"auc":0.7383,"mcc":0.3247,"ece":0.0341,
     "sharpe":0.3876,"coverage":0.3368,"mdd":-0.0484,"pnl":3.562,
     "score":0.3492,"sortino":1.2567,"precision":0.7054,
     "tau":0.455,"gamma":0.631,"temp":1.25,
     "comp":0.0556,"ins":0.0658,"suf":0.0119,"fcons":0.3461},
    {"fold":5,  "f1":0.5631,"auc":0.7455,"mcc":0.3379,"ece":0.0420,
     "sharpe":0.1644,"coverage":0.3833,"mdd":-0.1071,"pnl":1.191,
     "score":0.3146,"sortino":0.4012,"precision":0.7206,
     "tau":0.493,"gamma":0.581,"temp":1.8,
     "comp":0.0445,"ins":0.0546,"suf":0.0183,"fcons":0.3051},
]
AVGS = {
    "f1":0.5352,"auc":0.7223,"mcc":0.3042,"ece":0.0267,
    "sharpe":0.2694,"coverage":0.3658,"mdd":-0.1137,"pnl":3.073,
    "score":0.3224,"sortino":0.7038,"precision":0.6950,
    "comp":0.0583,"suf":0.0153,"fcons":0.4293,
}

C = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]  # fold colors
AVG_C = "#2d3436";  TGT_C = "#e74c3c"

# ── SVG primitives ─────────────────────────────────────────────────────────────
def _e(tag, content="", **attrs):
    a = " ".join(f'{k.replace("_","-")}="{v}"' for k, v in attrs.items())
    if content:
        return f"<{tag} {a}>{content}</{tag}>"
    return f"<{tag} {a}/>"

def svg_wrap(elements, w, h):
    body = "\n  ".join(elements)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" font-family="Arial,sans-serif">\n  '
            f'{body}\n</svg>')

# ── Core chart builder ────────────────────────────────────────────────────────
class Chart:
    def __init__(self, w=860, h=480, title="",
                 ml=80, mr=25, mt=55, mb=70):
        self.W, self.H = w, h
        self.title = title
        self.ml, self.mr, self.mt, self.mb = ml, mr, mt, mb
        self.pw = w - ml - mr
        self.ph = h - mt - mb
        self.x0 = ml
        self.yb = h - mb   # y bottom (axis)
        self.yt = mt        # y top
        self.els = []

    def vy(self, v, ymin, ymax):
        r = (v - ymin) / max(ymax - ymin, 1e-12)
        return self.yb - r * self.ph

    def _bg_grid(self, ymin, ymax, n=5):
        # background
        self.els.append(_e("rect", x=self.x0, y=self.yt,
                           width=self.pw, height=self.ph,
                           fill="#f8f9fa", stroke="#dee2e6"))
        # grid + y-labels
        for i in range(n + 1):
            v = ymin + i * (ymax - ymin) / n
            gy = self.vy(v, ymin, ymax)
            self.els.append(_e("line", x1=self.x0, y1=f"{gy:.1f}",
                               x2=self.x0+self.pw, y2=f"{gy:.1f}",
                               stroke="#dee2e6", stroke_width="1"))
            self.els.append(_e("text", f"{v:.2f}",
                               x=self.x0-6, y=f"{gy+4:.1f}",
                               text_anchor="end", font_size="10", fill="#555"))

    def _axes(self):
        self.els.append(_e("line", x1=self.x0, y1=self.yb,
                           x2=self.x0+self.pw, y2=self.yb,
                           stroke="#333", stroke_width="1.5"))
        self.els.append(_e("line", x1=self.x0, y1=self.yt,
                           x2=self.x0, y2=self.yb,
                           stroke="#333", stroke_width="1.5"))

    def _title(self):
        self.els.append(_e("text", self.title,
                           x=f"{self.W/2:.0f}", y="36",
                           text_anchor="middle", font_size="15",
                           font_weight="bold", fill="#222"))

    def _hline(self, v, ymin, ymax, color, dash, label="", right_label=True):
        y = self.vy(v, ymin, ymax)
        self.els.append(_e("line", x1=self.x0, y1=f"{y:.1f}",
                           x2=self.x0+self.pw, y2=f"{y:.1f}",
                           stroke=color, stroke_width="2",
                           stroke_dasharray=dash))
        if label:
            lx = self.x0 + self.pw + 4 if right_label else self.x0 + 4
            self.els.append(_e("text", label, x=lx, y=f"{y-4:.1f}",
                               font_size="10", fill=color))

    def _bar(self, bx, val, bw, ymin, ymax, color, show_label=True):
        zero_y = self.vy(max(ymin, min(0, ymax)), ymin, ymax)
        val_y  = self.vy(max(ymin, min(val, ymax)), ymin, ymax)
        rect_y = min(zero_y, val_y)
        rect_h = abs(zero_y - val_y)
        if rect_h < 1:
            rect_h = 1
        self.els.append(_e("rect", x=f"{bx:.1f}", y=f"{rect_y:.1f}",
                           width=f"{bw:.1f}", height=f"{rect_h:.1f}",
                           fill=color, opacity="0.88", rx="2"))
        if show_label:
            label_y = val_y - 5 if val >= 0 else val_y + rect_h + 12
            self.els.append(_e("text", f"{val:.3f}",
                               x=f"{bx+bw/2:.1f}", y=f"{label_y:.1f}",
                               text_anchor="middle", font_size="9", fill="#222"))

    def grouped_bar(self, cat_labels, series, series_labels, series_colors,
                    ymin=0, ymax=1, avg=None, target=None, target_label="",
                    ylabel=""):
        self._title()
        self._bg_grid(ymin, ymax)
        n_cats = len(cat_labels)
        n_ser  = len(series)
        grp_w  = self.pw / n_cats
        pad    = grp_w * 0.12
        bar_w  = (grp_w - 2 * pad) / max(n_ser, 1)

        for si, (vals, color) in enumerate(zip(series, series_colors)):
            for ci, val in enumerate(vals):
                bx = self.x0 + ci * grp_w + pad + si * bar_w
                self._bar(bx, val, bar_w - 1, ymin, ymax, color,
                          show_label=(n_ser == 1))

        # x-labels
        for ci, lbl in enumerate(cat_labels):
            cx = self.x0 + (ci + 0.5) * grp_w
            self.els.append(_e("text", lbl, x=f"{cx:.0f}",
                               y=f"{self.yb+18}", text_anchor="middle",
                               font_size="11", fill="#333"))

        if avg is not None:
            self._hline(avg, ymin, ymax, AVG_C, "6,3", f"Avg={avg:.3f}")
        if target is not None:
            self._hline(target, ymin, ymax, TGT_C, "4,3", target_label, right_label=False)

        self._axes()

        # legend (multi-series only)
        if n_ser > 1:
            lx = self.x0 + self.pw - n_ser * 110 + 10
            for si, (lbl, color) in enumerate(zip(series_labels, series_colors)):
                self.els.append(_e("rect", x=lx + si*110, y=self.yt-22,
                                   width="12", height="12", fill=color,
                                   opacity="0.88", rx="2"))
                self.els.append(_e("text", lbl, x=lx+si*110+16, y=self.yt-11,
                                   font_size="10", fill="#333"))

        if ylabel:
            mx = self.x0 - 55
            my = self.yt + self.ph // 2
            self.els.append(_e("text", ylabel,
                               x=mx, y=my, text_anchor="middle",
                               font_size="11", fill="#555",
                               transform=f"rotate(-90,{mx},{my})"))

    def to_svg(self):
        return svg_wrap(self.els, self.W, self.H)


# ── Individual charts ──────────────────────────────────────────────────────────
cats = [f"Fold {f['fold']}" for f in FOLDS]

def make_classification():
    c = Chart(title="Classification Metrics per Fold (Test Set)")
    c.grouped_bar(cats,
        [[f["f1"] for f in FOLDS],
         [f["auc"] for f in FOLDS],
         [f["mcc"] for f in FOLDS]],
        ["Macro F1", "AUC-ROC", "MCC"],
        [C[0], C[1], C[2]],
        ymin=0, ymax=0.9)
    # per-series avg labels manual
    for si, (key, color) in enumerate([("f1",C[0]),("auc",C[1]),("mcc",C[2])]):
        avg = AVGS[key]
        y = c.vy(avg, 0, 0.9)
        grp_w = c.pw / len(cats)
        bar_w = (grp_w - grp_w*0.24) / 3
        # Draw avg markers as small triangles on y-axis
    return c.to_svg()

def make_sharpe_sortino():
    c = Chart(title="Alert Sharpe & Sortino Ratio per Fold (Test Set)")
    c.grouped_bar(cats,
        [[f["sharpe"] for f in FOLDS],
         [f["sortino"] for f in FOLDS]],
        ["Sharpe Ratio", "Sortino Ratio"],
        [C[0], C[3]],
        ymin=0, ymax=1.5,
        avg=AVGS["sharpe"])
    return c.to_svg()

def make_coverage():
    c = Chart(title="Alert Coverage per Fold (Test Set) — Target: 35%")
    c.grouped_bar(cats,
        [[f["coverage"] for f in FOLDS]],
        ["Coverage"], [C[1]],
        ymin=0, ymax=0.6,
        avg=AVGS["coverage"],
        target=0.35, target_label="Target 35%")
    return c.to_svg()

def make_pnl():
    c = Chart(title="Portfolio PnL per Fold (×Initial Capital, Test Set)")
    c.grouped_bar(cats,
        [[f["pnl"] for f in FOLDS]],
        ["PnL"], [C[2]],
        ymin=0, ymax=6.0,
        avg=AVGS["pnl"])
    return c.to_svg()

def make_mdd():
    c = Chart(title="Maximum Drawdown per Fold (Test Set, lower = better)")
    c.grouped_bar(cats,
        [[f["mdd"] for f in FOLDS]],
        ["Max Drawdown"], [C[3]],
        ymin=-0.35, ymax=0.05,
        avg=AVGS["mdd"])
    return c.to_svg()

def make_ece():
    c = Chart(title="ECE Calibration Error per Fold (lower = better)")
    c.grouped_bar(cats,
        [[f["ece"] for f in FOLDS]],
        ["ECE"], [C[4]],
        ymin=0, ymax=0.06,
        avg=AVGS["ece"])
    return c.to_svg()

def make_policy():
    c = Chart(title="Alert Policy Parameters per Fold (τ, γ, Temperature)")
    c.grouped_bar(cats,
        [[f["tau"]   for f in FOLDS],
         [f["gamma"] for f in FOLDS],
         [f["temp"]  for f in FOLDS]],
        ["τ (conf threshold)", "γ (max_prob threshold)", "Temperature"],
        [C[0], C[1], C[3]],
        ymin=0, ymax=3.5)
    return c.to_svg()

def make_faithfulness():
    c = Chart(title="Faithfulness Metrics per Fold (Test Set)")
    c.grouped_bar(cats,
        [[f["comp"]  for f in FOLDS],
         [f["ins"]   for f in FOLDS],
         [f["fcons"] for f in FOLDS]],
        ["Comprehensiveness", "Insertion Gain", "Factor Consistency"],
        [C[0], C[2], C[1]],
        ymin=0, ymax=0.75,
        avg=AVGS["fcons"])
    return c.to_svg()

def make_precision():
    c = Chart(title="Alert Precision per Fold (Test Set)")
    c.grouped_bar(cats,
        [[f["precision"] for f in FOLDS]],
        ["Alert Precision"], [C[0]],
        ymin=0, ymax=1.0,
        avg=AVGS["precision"])
    return c.to_svg()

def make_score():
    c = Chart(title="Model Score per Fold — Val vs Test")
    val_scores  = [0.2919, 0.3579, 0.3360, 0.3406, 0.3551]
    test_scores = [f["score"] for f in FOLDS]
    c.grouped_bar(cats,
        [val_scores, test_scores],
        ["Validation Score", "Test Score"],
        [C[0], C[2]],
        ymin=0, ymax=0.45)
    return c.to_svg()

def make_summary_table():
    """SVG table with all test metrics."""
    W, H = 1000, 320
    els = []

    headers = ["Fold","F1","AUC","MCC","ECE","Sharpe","Coverage","MDD","PnL","Score"]
    rows = []
    for f in FOLDS:
        rows.append([
            f"Fold {f['fold']}",
            f"{f['f1']:.3f}", f"{f['auc']:.3f}", f"{f['mcc']:.3f}",
            f"{f['ece']:.3f}", f"{f['sharpe']:.3f}",
            f"{f['coverage']:.1%}", f"{f['mdd']:.3f}",
            f"{f['pnl']:.2f}×", f"{f['score']:.3f}",
        ])
    rows.append([
        "Average",
        f"{AVGS['f1']:.3f}", f"{AVGS['auc']:.3f}", f"{AVGS['mcc']:.3f}",
        f"{AVGS['ece']:.3f}", f"{AVGS['sharpe']:.3f}",
        f"{AVGS['coverage']:.1%}", f"{AVGS['mdd']:.3f}",
        f"{AVGS['pnl']:.2f}×", f"{AVGS['score']:.3f}",
    ])

    els.append(_e("text", "SAFE-Alert — Walk-forward Test Results Summary",
                  x="500", y="28", text_anchor="middle",
                  font_size="15", font_weight="bold", fill="#222"))

    col_w = [80] + [85] * 9
    col_x = [10]
    for w in col_w[:-1]:
        col_x.append(col_x[-1] + w)

    row_h = 34
    y_start = 48

    for ri, row in enumerate([headers] + rows):
        ry = y_start + ri * row_h
        is_header = ri == 0
        is_avg    = ri == len(rows)
        is_fold3  = ri == 3

        bg = "#2d3436" if is_header else ("#dfe6e9" if is_avg else ("#ffeaa7" if is_fold3 else ("#ffffff" if ri % 2 == 0 else "#f8f9fa")))
        txt_color = "#ffffff" if is_header else "#222"
        font_w = "bold" if (is_header or is_avg) else "normal"

        els.append(_e("rect", x=col_x[0], y=ry, width=sum(col_w), height=row_h,
                      fill=bg, stroke="#dee2e6"))

        for ci, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
            els.append(_e("text", cell,
                          x=cx + cw//2, y=ry + 22,
                          text_anchor="middle", font_size="11",
                          fill=txt_color, font_weight=font_w))

    # Note about fold 3
    els.append(_e("text", "* Fold 3 (highlighted) underperforms due to high-gamma policy regime in this time period.",
                  x="10", y=y_start + (len(rows)+1)*row_h + 15,
                  font_size="9", fill="#666", font_style="italic"))

    return svg_wrap(els, W, H)


# ── Radar chart (pure SVG math) ───────────────────────────────────────────────
def make_radar():
    W, H = 600, 580
    cx, cy, R = 300, 300, 200
    labels = ["Macro F1", "AUC-ROC", "MCC", "Calibration\n(1−ECE)", "Sharpe", "Coverage\n≈35%"]
    N = len(labels)
    angles = [math.pi/2 - 2*math.pi*i/N for i in range(N)]  # start at top

    def polar(r, a):
        return cx + r * math.cos(a), cy - r * math.sin(a)

    def fold_scores(f):
        cov_score = max(0, 1 - abs(f["coverage"] - 0.35) / 0.35)
        return [
            f["f1"],
            f["auc"] - 0.5,
            (f["mcc"] + 1) / 2,
            max(0, 1 - f["ece"] / 0.10),
            max(0, min(1, f["sharpe"] / 0.5)),
            cov_score,
        ]

    els = []
    els.append(_e("text", "SAFE-Alert — Per-fold Profile (Normalised)",
                  x=W//2, y="28", text_anchor="middle",
                  font_size="14", font_weight="bold", fill="#222"))

    # Grid rings
    for level in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{polar(level*R, a)[0]:.1f},{polar(level*R, a)[1]:.1f}"
                       for a in angles)
        pts += f" {polar(level*R, angles[0])[0]:.1f},{polar(level*R, angles[0])[1]:.1f}"
        els.append(_e("polyline", points=pts, fill="none",
                      stroke="#dee2e6", stroke_width="1"))
        lx, ly = polar(level*R, angles[0])
        els.append(_e("text", f"{level:.2f}", x=f"{lx+4:.0f}", y=f"{ly-2:.0f}",
                      font_size="8", fill="#aaa"))

    # Spokes
    for a in angles:
        x1, y1 = polar(0, a)
        x2, y2 = polar(R, a)
        els.append(_e("line", x1=f"{x1:.1f}", y1=f"{y1:.1f}",
                      x2=f"{x2:.1f}", y2=f"{y2:.1f}",
                      stroke="#dee2e6", stroke_width="1"))

    # Axis labels
    for i, (lbl, a) in enumerate(zip(labels, angles)):
        lx, ly = polar(R + 28, a)
        for j, part in enumerate(lbl.split("\n")):
            els.append(_e("text", part,
                          x=f"{lx:.0f}", y=f"{ly + j*13:.0f}",
                          text_anchor="middle", font_size="11", fill="#333"))

    # Fold polygons
    for fi, (f, color) in enumerate(zip(FOLDS, C)):
        scores = fold_scores(f)
        pts = " ".join(f"{polar(s*R, a)[0]:.1f},{polar(s*R, a)[1]:.1f}"
                       for s, a in zip(scores, angles))
        pts += f" {polar(scores[0]*R, angles[0])[0]:.1f},{polar(scores[0]*R, angles[0])[1]:.1f}"
        els.append(_e("polyline", points=pts, fill=color,
                      fill_opacity="0.08", stroke=color,
                      stroke_width="1.8", stroke_opacity="0.8"))

    # Average polygon
    n_folds = len(FOLDS)
    avg_scores = [sum(fold_scores(f)[i] for f in FOLDS) / n_folds for i in range(N)]
    pts = " ".join(f"{polar(s*R, a)[0]:.1f},{polar(s*R, a)[1]:.1f}"
                   for s, a in zip(avg_scores, angles))
    pts += f" {polar(avg_scores[0]*R, angles[0])[0]:.1f},{polar(avg_scores[0]*R, angles[0])[1]:.1f}"
    els.append(_e("polyline", points=pts, fill=AVG_C,
                  fill_opacity="0.12", stroke=AVG_C,
                  stroke_width="2.5", stroke_dasharray="5,3"))

    # Legend
    leg_y = H - 90
    all_labels = [f"Fold {f['fold']}" for f in FOLDS] + ["Average"]
    all_colors = C + [AVG_C]
    for i, (lbl, col) in enumerate(zip(all_labels, all_colors)):
        lx = 20 + (i % 3) * 180
        ly = leg_y + (i // 3) * 22
        els.append(_e("rect", x=lx, y=ly-10, width="14", height="14",
                      fill=col, opacity="0.85", rx="2"))
        els.append(_e("text", lbl, x=lx+18, y=ly+1,
                      font_size="11", fill="#333"))

    return svg_wrap(els, W, H)


# ── Generate all ───────────────────────────────────────────────────────────────
charts = [
    ("fig1_classification_metrics",  make_classification),
    ("fig2_sharpe_sortino",          make_sharpe_sortino),
    ("fig3_coverage",                make_coverage),
    ("fig4_pnl",                     make_pnl),
    ("fig5_max_drawdown",            make_mdd),
    ("fig6_ece_calibration",         make_ece),
    ("fig7_policy_params",           make_policy),
    ("fig8_faithfulness",            make_faithfulness),
    ("fig9_alert_precision",         make_precision),
    ("fig10_val_vs_test_score",      make_score),
    ("fig11_radar_chart",            make_radar),
    ("fig12_summary_table",          make_summary_table),
]

if __name__ == "__main__":
    print(f"Output: {OUT}\n")
    for name, fn in charts:
        svg = fn()
        path = OUT / f"{name}.svg"
        path.write_text(svg, encoding="utf-8")
        print(f"  Saved → {name}.svg")
    print(f"\nDone! {len(charts)} SVG files saved.")
    print("Open any .svg file in Chrome/Edge/Firefox to view.")
