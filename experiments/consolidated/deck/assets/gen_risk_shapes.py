# /// script
# requires-python = ">=3.10"
# ///
"""Figure: the five risk shapes (latent log-odds vs standardized-log value s).

These are the DGP's risk functions over the standardized log coordinate s
(make_data, sigma=0.15) — the conceptual definition of each condition, not run
output. Colours mark which encoder wins that shape in the GRU.
Run:  uv run --python 3.12 gen_risk_shapes.py
"""
import math
from _svglib import SVG, BLUE, RED, GREEN, GRAY

PANELS = [
    ("log-linear", "f(s) = s", lambda s: s, BLUE, "log wins"),
    ("monotone-curved", "f(s) = s³", lambda s: s ** 3, GREEN, "projection"),
    ("smooth non-monotone", "f(s) = s²", lambda s: s ** 2, GREEN, "projection"),
    ("sharp (mode)", "exp(−s²/2σ²)", lambda s: math.exp(-(s ** 2) / (2 * 0.15 ** 2)), RED, "PLE"),
    ("sharp (off)", "exp(−(s−1.5)²/2σ²)", lambda s: math.exp(-((s - 1.5) ** 2) / (2 * 0.15 ** 2)), RED, "PLE"),
]

W, H = 1050, 360
N = len(PANELS)
PAD = 16
PW = (W - PAD * (N + 1)) / N
PTOP, PH = 96, 176
SMIN, SMAX = -2.5, 2.5

s = SVG(W, H)
s.text(W / 2, 34, "Five risk shapes: how fraud log-odds bends with the feature value", 24, BLUE, weight="bold")
s.text(W / 2, 62, "s = standardized log value (x-axis) · risk = latent log-odds (y-axis) · sharp = a localized band",
       16, "#444")

xs = [SMIN + i * (SMAX - SMIN) / 120 for i in range(121)]
for k, (name, formula, f, col, win) in enumerate(PANELS):
    px = PAD + k * (PW + PAD)
    s.rect(px, PTOP, PW, PH, fill="#fbfcfd", stroke="#e5e7eb", sw=1.2, rx=6)
    ys = [f(x) for x in xs]
    lo, hi = min(ys), max(ys)
    rng = (hi - lo) or 1.0

    def sx(x, px=px):
        return px + 12 + (x - SMIN) / (SMAX - SMIN) * (PW - 24)

    def sy(v, lo=lo, rng=rng):
        return PTOP + PH - 16 - (v - lo) / rng * (PH - 40)

    # zero baseline for value axis
    s.line(sx(SMIN), sy(0) if lo <= 0 <= hi else PTOP + PH - 16,
           sx(SMAX), sy(0) if lo <= 0 <= hi else PTOP + PH - 16, stroke="#e5e7eb", w=1)
    s.polyline([(sx(x), sy(f(x))) for x in xs], stroke=col, w=3)
    s.text(px + PW / 2, PTOP - 10, name, 15, "#222", weight="bold")
    s.text(px + PW / 2, PTOP + PH + 24, formula, 15, GRAY, mono=True)
    s.text(px + PW / 2, PTOP + PH + 46, f"→ {win}", 14, col, weight="bold")

s.text(W / 2, H - 8,
       "Only the sharp band concentrates risk in a narrow range of the value — the shape a fixed quantile basis resolves for free.",
       14, GRAY, style="italic")
s.save("fig_risk_shapes.svg")
