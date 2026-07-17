# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the five encoders — a fixed transform / basis vs a learned expansion.

Schematic (definitions from flow.py encode() / _GRU); not run output.
Run:  uv run --python 3.12 gen_encoders.py
"""
from _svglib import SVG, BLUE, RED, GREEN, PURPLE, GRAY, FILL_BLUE, FILL_RED, FILL_GREEN

CARDS = [
    ("raw", GRAY, "#f3f4f6", "standardized value", "no transform", "dim 1"),
    ("log", BLUE, FILL_BLUE, "standardized log1p(x)", "the reference scalar", "dim 1"),
    ("ple", RED, FILL_RED, "8 quantile bins on the", "log coordinate (FIXED)", "dim 8"),
    ("projection", GREEN, FILL_GREEN, "Linear(1→8)→ReLU", "per feature (LEARNED)", "dim 8"),
    ("dense", PURPLE, "#efe6fb", "joint Linear→ReLU over", "all features (LEARNED)", "dim h"),
]

W, H = 1050, 460
CW, CH, TOP = 186, 168, 150
GAP = (W - len(CARDS) * CW) / (len(CARDS) + 1)

s = SVG(W, H)
s.text(W / 2, 36, "Five encoders of one numeric feature", 26, BLUE, weight="bold")
s.text(W / 2, 66, "all consume the same log scalar — they differ in whether the nonlinearity is fixed or learned, and where it lives",
       16, "#444")

# group bands
fixed_x0 = GAP
fixed_x1 = GAP * 3 + CW * 3
learn_x0 = fixed_x1 + GAP
learn_x1 = W - GAP
s.text((fixed_x0 + fixed_x1) / 2, TOP - 22, "FIXED · applied before the model", 16, GRAY, weight="bold")
s.text((learn_x0 + learn_x1) / 2, TOP - 22, "LEARNED · inside the model", 16, GRAY, weight="bold")
s.line(learn_x0 - GAP / 2, TOP - 34, learn_x0 - GAP / 2, TOP + CH + 30, stroke="#d1d5db", w=1.5, dash="5 4")

for i, (name, col, fill, l1, l2, dim) in enumerate(CARDS):
    cx = GAP + i * CW + (i) * 0  # cards flush with equal gaps
    x = GAP * (i + 1) + CW * i
    s.rect(x, TOP, CW, CH, fill=fill, stroke=col, sw=2, rx=10)
    s.text(x + CW / 2, TOP + 34, name, 22, col, weight="bold", mono=True)
    s.line(x + 16, TOP + 46, x + CW - 16, TOP + 46, stroke=col, w=1)
    s.text(x + CW / 2, TOP + 74, l1, 14, "#333")
    s.text(x + CW / 2, TOP + 96, l2, 14, "#333")
    s.text(x + CW / 2, TOP + 134, dim, 15, col, weight="bold", mono=True)

s.text(W / 2, TOP + CH + 66, "PLE and projection are dimension-matched (8) on purpose: the fixed-vs-learned comparison is fair.",
       16, "#222", weight="bold")
s.text(W / 2, H - 14, "Source: flow.py encode() and _GRU embedding modes.", 14, GRAY, style="italic")
s.save("fig_encoders.svg")
