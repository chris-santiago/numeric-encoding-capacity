# /// script
# requires-python = ">=3.10"
# ///
"""Figure: read the deployment gap, not the estimand.

Per illustrative GRU arm: raw_gap with its 95% CI (green = real, grey = n.s.)
vs the deficit-corrected dc_lift (blue). Their gap is the deficit add-back.
Data: deck_data.json['addback'] (ConsolidatedFlow, K=6).
Run:  uv run --python 3.12 gen_addback.py
"""
from _svglib import SVG, BLUE, GREEN, GRAY, RED, load_data

D = load_data()
ROWS = sorted(D["addback"], key=lambda r: r["raw_gap"])

W, H = 1040, 560
LX, X0, X1, TOP = 40, 275, 1002, 96
XMIN, XMAX = -0.08, 0.45
ROW_H = (470 - TOP) / len(ROWS)


def x(v):
    return X0 + (v - XMIN) / (XMAX - XMIN) * (X1 - X0)


s = SVG(W, H)
s.text(W / 2, 32, "Read the deployment gap, not the estimand", 25, BLUE, weight="bold")
s.text(W / 2, 60, "raw_gap (coloured CI) is the deployment quantity · dc_lift (blue) inflates it by the deficit add-back",
       17, "#444")

# x grid
for gx in [-0.05, 0.0, 0.1, 0.2, 0.3, 0.4]:
    s.line(x(gx), TOP - 6, x(gx), 476, stroke="#eceef1", w=1)
    s.text(x(gx), 496, f"{gx:+.2f}".replace("+0.00", "0"), 14, GRAY)
s.line(x(0.0), TOP + 4, x(0.0), 476, stroke="#111", w=1.6)  # zero line (starts below legend)
s.text((X0 + X1) / 2, 520, "lift over log (PR-AUC)", 15, GRAY)

for i, r in enumerate(ROWS):
    cy = TOP + (i + 0.5) * ROW_H
    sig = r["raw_gap_sig"]
    col = GREEN if sig else GRAY
    caution = (r["enc"] == "raw")
    # label
    s.text(X0 - 16, cy + 5, r["label"], 16, RED if caution else "#222",
           anchor="end", weight="bold" if caution else "normal")
    # add-back connector raw_gap -> dc_lift
    s.line(x(r["raw_gap"]), cy, x(r["dc_lift"]), cy, stroke="#c9ccd1", w=1.2, dash="3 3")
    # raw_gap CI rule + point
    s.line(x(r["raw_gap_lo"]), cy, x(r["raw_gap_hi"]), cy, stroke=col, w=5)
    s.circle(x(r["raw_gap"]), cy, 6, fill=col)
    # dc_lift point
    s.circle(x(r["dc_lift"]), cy, 6, fill=BLUE, stroke="#fff", sw=1.5)
    # numbers
    s.text(x(r["raw_gap_hi"]) + 12, cy - 6, f"raw {r['raw_gap']:+.2f}", 13, col, anchor="start")
    s.text(x(r["dc_lift"]) + 12, cy + 15, f"dc {r['dc_lift']:+.2f}", 13, BLUE, anchor="start")

# caution callout on the raw·sharp row (first, lowest raw_gap)
s.text(W / 2, 540,
       "'raw · sharp': dc_lift +0.16 (Holm-sig) on a raw_gap of +0.00 — pure add-back, zero deployment value.   "
       "Source: ConsolidatedFlow/" + D["run_id"] + ".",
       14, GRAY, style="italic")

# legend
s.circle(X0 + 6, 76, 6, fill=GREEN)
s.text(X0 + 18, 81, "raw_gap, CI excludes 0", 13, "#222", anchor="start")
s.circle(X0 + 240, 76, 6, fill=GRAY)
s.text(X0 + 252, 81, "raw_gap, n.s.", 13, "#222", anchor="start")
s.circle(X0 + 400, 76, 6, fill=BLUE, stroke="#fff", sw=1.5)
s.text(X0 + 412, 81, "dc_lift (structural)", 13, "#222", anchor="start")
s.save("fig_addback.svg")
