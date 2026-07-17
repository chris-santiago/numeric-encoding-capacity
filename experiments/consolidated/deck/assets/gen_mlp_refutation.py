# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the refutation. Free-nonlinearity MLP, log vs ple PR-AUC by risk shape.

Data: deck_data.json['mlp_prauc'] (ConsolidatedFlow, K=6, 8-seed mean).
Run:  uv run --python 3.12 gen_mlp_refutation.py
"""
from _svglib import SVG, BLUE, RED, GRAY, GREEN, load_data

D = load_data()
M = D["mlp_prauc"]
SHAPES = ["log-lin", "curved", "smooth", "sharp", "sharp-off"]
READ = ["redundant", "~redundant", "PLE hurts", "DECISIVE", "small"]
READ_COLOR = [GRAY, GRAY, RED, GREEN, GRAY]

W, H = 1040, 570
X0, X1, Y0, TOP = 78, 1015, 478, 70
VMAX = 0.66
PLOT_H = Y0 - TOP
GW = (X1 - X0) / len(SHAPES)
BW = 62


def y(v):
    return Y0 - (v / VMAX) * PLOT_H


s = SVG(W, H)
s.text(W / 2, 34, "Free-nonlinearity MLP: PLE is redundant — except on the sharp band", 25, BLUE, weight="bold")

# y grid + axis
for g in [0.0, 0.15, 0.30, 0.45, 0.60]:
    s.line(X0, y(g), X1, y(g), stroke="#e5e7eb", w=1)
    s.text(X0 - 10, y(g) + 4, f"{g:.2f}", 15, GRAY, anchor="end")
s.line(X0, Y0, X1, Y0, stroke=GRAY, w=1.5)
s.text(30, TOP + PLOT_H / 2, "PR-AUC (8-seed mean)", 16, GRAY, rotate=-90)

for i, sh in enumerate(SHAPES):
    cx = X0 + (i + 0.5) * GW
    lv, pv = M[sh]["log"], M[sh]["ple"]
    # log bar (left), ple bar (right)
    lx, px = cx - BW - 6, cx + 6
    s.rect(lx, y(lv), BW, Y0 - y(lv), fill=BLUE, stroke="none", rx=3)
    s.rect(px, y(pv), BW, Y0 - y(pv), fill=RED, stroke="none", rx=3)
    s.text(lx + BW / 2, y(lv) - 8, f"{lv:.2f}", 15, BLUE, weight="bold")
    s.text(px + BW / 2, y(pv) - 8, f"{pv:.2f}", 15, RED, weight="bold")
    # shape label + reading
    s.text(cx, Y0 + 24, sh, 17, "#222", weight="bold")
    s.text(cx, Y0 + 46, READ[i], 15, READ_COLOR[i], weight="bold" if i in (2, 3) else "normal", style="italic")

# highlight the sharp gain
sx = X0 + (3 + 0.5) * GW
s.arrow(sx - 42, y(M["sharp"]["ple"]) + 12, sx - 42, y(M["sharp"]["log"]) - 6, stroke=GREEN, w=2.5)
s.text(sx + 96, (y(M["sharp"]["ple"]) + y(M["sharp"]["log"])) / 2, "+0.39", 22, GREEN, weight="bold", anchor="middle")

# legend
s.rect(X0 + 4, TOP - 6, 16, 16, fill=BLUE, stroke="none", rx=2)
s.text(X0 + 26, TOP + 7, "mlp_log (reference)", 15, "#222", anchor="start")
s.rect(X0 + 210, TOP - 6, 16, 16, fill=RED, stroke="none", rx=2)
s.text(X0 + 232, TOP + 7, "mlp_ple (fixed basis)", 15, "#222", anchor="start")

s.text(W / 2, H - 12,
       "A free per-step nonlinearity does NOT make encoding redundant — localization does.   "
       "Source: ConsolidatedFlow/" + D["run_id"] + ", mlp arms, K=6.",
       14, GRAY, style="italic")
s.save("fig_mlp_refutation.svg")
