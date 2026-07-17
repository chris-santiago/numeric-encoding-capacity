# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the two ways a model reads a per-step numeric feature.

Schematic contrasting the affine-read models (GRU / static logistic head) with
the free-nonlinearity MLP. Definitions from flow.py (_GRU affine input vs the
per-step MLP). Not run output.
Run:  uv run --python 3.12 gen_read_modes.py
"""
from _svglib import SVG, BLUE, GREEN, GRAY, ORANGE, FILL_BLUE, FILL_GREEN

W, H = 1050, 452
s = SVG(W, H)
s.text(W / 2, 36, "Two ways a model reads a per-step numeric feature", 26, BLUE, weight="bold")

PY, PH = 84, 250
LW = 486
LX, RX = 26, W - 26 - LW


def pipeline(x0, title, tcol, fill, steps, verdict, vcol):
    s.rect(x0, PY, LW, PH, fill=fill, stroke=tcol, sw=2, rx=12)
    s.text(x0 + LW / 2, PY + 34, title, 21, tcol, weight="bold")
    n = len(steps)
    bw, gap = 92, 26
    total = n * bw + (n - 1) * gap
    bx = x0 + (LW - total) / 2
    cy = PY + 120
    for i, (lab, sub) in enumerate(steps):
        s.rect(bx, cy - 34, bw, 68, fill="#fff", stroke=GRAY, sw=1.5, rx=8)
        s.text(bx + bw / 2, cy - 4, lab, 15, "#222", weight="bold", mono=True)
        if sub:
            s.text(bx + bw / 2, cy + 18, sub, 12, GRAY)
        if i < n - 1:
            s.arrow(bx + bw + 4, cy, bx + bw + gap - 4, cy, stroke=GRAY, w=2)
        bx += bw + gap
    s.text(x0 + LW / 2, PY + PH - 22, verdict, 14, vcol, weight="bold")


pipeline(LX, "Affine-read  ·  GRU, static logistic head", BLUE, FILL_BLUE,
         [("x_t", "value"), ("e(x_t)", "encode"), ("W · e", "linear"), ("σ / tanh", "FIXED gate")],
         "per-step class = span(e): only shapes the basis spans", BLUE)
pipeline(RX, "Free-nonlinearity  ·  per-step MLP", GREEN, FILL_GREEN,
         [("x_t", "value"), ("Linear", "→ReLU"), ("Linear", "width 64"), ("f(x_t)", "ANY shape")],
         "represents any 1-D shape — if SGD can find it", GREEN)

s.text(W / 2, H - 30,
       "The old account: a basis only helps the affine-read model. The catch it missed: 'can represent' ≠ 'can find'.",
       17, ORANGE, weight="bold")
s.text(W / 2, H - 8, "Source: flow.py _GRU (affine input) vs the per-step MLP arm.", 13, GRAY, style="italic")
s.save("fig_read_modes.svg")
