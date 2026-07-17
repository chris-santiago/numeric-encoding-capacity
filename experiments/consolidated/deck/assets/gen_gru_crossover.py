# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the encoder crossover in the affine-read GRU (absolute PR-AUC by shape).

log (reference) vs fixed PLE vs learned projection, with the oracle ceiling.
Data: deck_data.json['gru_prauc'] (ConsolidatedFlow, K=6, 8-seed mean).
Run:  uv run --python 3.12 gen_gru_crossover.py
"""
from _svglib import SVG, BLUE, RED, GREEN, GRAY, LGRAY, load_data

D = load_data()
G = D["gru_prauc"]
SHAPES = ["log-lin", "curved", "smooth", "sharp", "sharp-off"]
WINNER = ["log", "projection", "projection", "PLE", "PLE"]
WIN_COLOR = {"log": BLUE, "projection": GREEN, "PLE": RED}
BARS = [("log", BLUE), ("ple", RED), ("projection", GREEN)]

W, H = 1040, 575
X0, X1, Y0, TOP = 78, 1015, 470, 74
VMAX = 0.66
PLOT_H = Y0 - TOP
GW = (X1 - X0) / len(SHAPES)
BW = 40


def y(v):
    return Y0 - (v / VMAX) * PLOT_H


s = SVG(W, H)
s.text(W / 2, 32, "Affine-read GRU: the winning encoder crosses over by risk shape", 25, BLUE, weight="bold")

for g in [0.0, 0.15, 0.30, 0.45, 0.60]:
    s.line(X0, y(g), X1, y(g), stroke="#e5e7eb", w=1)
    s.text(X0 - 10, y(g) + 4, f"{g:.2f}", 15, GRAY, anchor="end")
s.line(X0, Y0, X1, Y0, stroke=GRAY, w=1.5)
s.text(30, TOP + PLOT_H / 2, "PR-AUC (8-seed mean)", 16, GRAY, rotate=-90)

for i, sh in enumerate(SHAPES):
    cx = X0 + (i + 0.5) * GW
    start = cx - (len(BARS) * BW + (len(BARS) - 1) * 6) / 2
    for j, (enc, col) in enumerate(BARS):
        bx = start + j * (BW + 6)
        v = G[sh][enc]
        s.rect(bx, y(v), BW, Y0 - y(v), fill=col, stroke="none", rx=3)
        s.text(bx + BW / 2, y(v) - 7, f"{v:.2f}", 13, col, weight="bold")
    # oracle ceiling tick across the group
    ov = G[sh]["oracle"]
    s.line(start - 6, y(ov), start + len(BARS) * BW + (len(BARS) - 1) * 6 + 6, y(ov),
           stroke=LGRAY, w=3, dash="6 4")
    s.text(cx, y(ov) - 6, f"oracle {ov:.2f}", 12, "#9aa0a6")
    # shape + winner
    s.text(cx, Y0 + 24, sh, 17, "#222", weight="bold")
    s.text(cx, Y0 + 47, f"→ {WINNER[i]}", 15, WIN_COLOR[WINNER[i]], weight="bold")

# legend
lx = X0 + 2
for enc, col, lab in [("log", BLUE, "log (reference scalar)"), ("ple", RED, "ple (fixed quantile basis)"),
                      ("projection", GREEN, "projection (learned per-feature)")]:
    s.rect(lx, TOP - 8, 15, 15, fill=col, stroke="none", rx=2)
    s.text(lx + 21, TOP + 4, lab, 14, "#222", anchor="start")
    lx += 24 + len(lab) * 7.4

s.text(W / 2, H - 12,
       "Fixed PLE towers where risk is sharp; learned projection leads where it is smooth/curved; log wins only log-linear.   "
       "Source: ConsolidatedFlow/" + D["run_id"] + ", K=6.",
       14, GRAY, style="italic")
s.save("fig_gru_crossover.svg")
