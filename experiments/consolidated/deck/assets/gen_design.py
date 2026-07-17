# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the consolidated design — one crossed flow, dual positive-controlled.

Schematic of the factors and controls (HYPOTHESIS.md / flow.py). Not run output.
Run:  uv run --python 3.12 gen_design.py
"""
from _svglib import SVG, BLUE, RED, GREEN, GRAY, ORANGE, FILL_BLUE, FILL_GREEN, FILL_ORANGE

W, H = 1050, 452
s = SVG(W, H)
s.text(W / 2, 36, "One controlled synthetic flow, fully crossed", 26, BLUE, weight="bold")

# left: factors
FX, FW = 30, 300
factors = [
    ("3 architectures", "static · GRU · MLP"),
    ("5 risk shapes", "log-lin · curved · smooth · 2 sharp"),
    ("5 encoders", "raw · log · ple · projection · dense"),
    ("multiplicity K", "K ∈ {1, 6}"),
    ("8 seeds", "data draw + init, re-trained"),
]
fy = 78
for name, sub in factors:
    s.rect(FX, fy, FW, 52, fill=FILL_BLUE, stroke=BLUE, sw=1.5, rx=8)
    s.text(FX + 14, fy + 22, name, 16, BLUE, weight="bold", anchor="start")
    s.text(FX + 14, fy + 42, sub, 13, "#333", anchor="start", mono=True)
    s.arrow(FX + FW + 6, fy + 26, FX + FW + 54, fy + 26, stroke=GRAY, w=2)
    fy += 66

# center
CX, CW = FX + FW + 60, 226
s.rect(CX, 150, CW, 150, fill="#fff", stroke="#111", sw=2, rx=12)
s.text(CX + CW / 2, 196, "ConsolidatedFlow", 19, "#111", weight="bold", mono=True)
s.text(CX + CW / 2, 224, "temperature-calibrated", 14, GRAY)
s.text(CX + CW / 2, 246, "seed-paired 95% CI", 14, GRAY)
s.text(CX + CW / 2, 268, "Holm across the family", 14, GRAY)
s.arrow(CX + CW + 6, 225, CX + CW + 54, 225, stroke=GRAY, w=2)

# right: metric + controls
RX, RW = CX + CW + 60, W - (CX + CW + 60) - 30
s.rect(RX, 92, RW, 60, fill=FILL_GREEN, stroke=GREEN, sw=1.5, rx=8)
s.text(RX + RW / 2, 116, "Metric: PR-AUC", 16, GREEN, weight="bold")
s.text(RX + RW / 2, 138, "raw_gap (deploy) + dc_lift", 13, "#333")

s.rect(RX, 166, RW, 78, fill=FILL_ORANGE, stroke=ORANGE, sw=1.5, rx=8)
s.text(RX + RW / 2, 190, "Two positive controls", 15, ORANGE, weight="bold")
s.text(RX + RW / 2, 210, "PLE must detect sharp in the", 13, "#333")
s.text(RX + RW / 2, 226, "static path AND the GRU (hard-halt)", 13, "#333")

s.rect(RX, 258, RW, 52, fill="#f3f4f6", stroke=GRAY, sw=1.5, rx=8)
s.text(RX + RW / 2, 280, "Negative control", 15, GRAY, weight="bold")
s.text(RX + RW / 2, 300, "log_linear: no arm beats log", 13, "#333")

s.text(W / 2, H - 40,
       "Same signal held fixed across architectures — so what varies is the model and the shape, not the task.",
       17, "#222", weight="bold")
s.text(W / 2, H - 14,
       "Cross-architecture is not a single-variable manipulation; the within-model controls (GRU projection arm, MLP sharp) carry the cleanest weight.",
       13, GRAY, style="italic")
s.save("fig_design.svg")
