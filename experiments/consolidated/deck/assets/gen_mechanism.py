# /// script
# requires-python = ">=3.10"
# ///
"""Figure: one account, two obstacles — the 2x2 that fits all three architectures.

Conceptual synthesis (the mechanism section of REPORT.md); cell verdicts echo
the run's raw gaps. Not itself run output.
Run:  uv run --python 3.12 gen_mechanism.py
"""
from _svglib import SVG, BLUE, RED, GREEN, GRAY, ORANGE, FILL_RED, FILL_GREEN, FILL_BLUE

W, H = 1050, 566
s = SVG(W, H)
s.text(W / 2, 34, "One account, two obstacles", 26, BLUE, weight="bold")

# obstacle definitions
s.rect(30, 54, 495, 78, fill=FILL_BLUE, stroke=BLUE, sw=1.5, rx=8)
s.text(46, 80, "Obstacle 1 — can't FORM the shape", 17, BLUE, weight="bold", anchor="start")
s.text(46, 104, "affine read: per-step class = span(e).", 14, "#333", anchor="start")
s.text(46, 122, "A richer basis widens it. Any non-log-linear shape.", 14, "#333", anchor="start")

s.rect(W - 30 - 495, 54, 495, 78, fill=FILL_RED, stroke=RED, sw=1.5, rx=8)
s.text(W - 30 - 479, 80, "Obstacle 2 — can't FIND the shape", 17, RED, weight="bold", anchor="start")
s.text(W - 30 - 479, 104, "SGD cannot locate a σ≈0.15 bump from a scalar.", 14, "#333", anchor="start")
s.text(W - 30 - 479, 122, "A fixed quantile basis hands it over. Sharp only.", 14, "#333", anchor="start")

# 2x2 matrix
GX, GY = 300, 190
CW, CH = 340, 132
COLS = ["smooth / curved target", "sharp (localized) target"]
ROWS = ["affine-read\n(GRU, static)", "free-nonlinearity\n(MLP)"]
# cell = (help?, encoder, obstacle note, fill, textcolor)
CELLS = [
    [("basis helps", "learned projection", "Obstacle 1", FILL_GREEN, GREEN),
     ("basis helps", "fixed PLE", "Obstacles 1 + 2", FILL_RED, RED)],
    [("redundant / harmful", "— (MLP forms it)", "neither", "#f3f4f6", GRAY),
     ("basis helps", "fixed PLE  (+0.39)", "Obstacle 2", FILL_RED, RED)],
]

# column headers
for j, c in enumerate(COLS):
    s.text(GX + CW / 2 + j * CW, GY - 14, c, 17, "#222", weight="bold")
# row headers + cells
for i in range(2):
    ry = GY + i * CH
    for li, line in enumerate(ROWS[i].split("\n")):
        s.text(GX - 16, ry + CH / 2 - 8 + li * 20, line, 15, "#222", anchor="end", weight="bold")
    for j in range(2):
        rx = GX + j * CW
        helpv, enc, obs, fill, tcol = CELLS[i][j]
        s.rect(rx, ry, CW - 12, CH - 12, fill=fill, stroke=tcol, sw=2, rx=10)
        s.text(rx + (CW - 12) / 2, ry + 36, helpv, 18, tcol, weight="bold")
        s.text(rx + (CW - 12) / 2, ry + 66, enc, 15, "#222", mono=True)
        s.text(rx + (CW - 12) / 2, ry + 96, obs, 14, GRAY, style="italic")

s.text(W / 2, H - 42,
       "A basis helps when the model cannot FORM the shape, or cannot FIND it by SGD.",
       19, ORANGE, weight="bold")
s.text(W / 2, H - 16,
       "Sharp non-monotonicity trips both — which is why it is the largest, most universal lever (it even helps the MLP).",
       15, GRAY, style="italic")
s.save("fig_mechanism.svg")
