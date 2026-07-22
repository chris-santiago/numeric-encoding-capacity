# /// script
# requires-python = ">=3.10"
# ///
"""Generate the five inline-SVG figures for the encoding-mechanism explainer.

Colours are CSS custom properties (var(--token)), so the inline SVGs inherit the
page palette and adapt to light/dark automatically. Curves (Gaussian needle,
loss plateau, convex bowl, sigmoid) are computed, not hand-drawn, so the shapes
are faithful. Output: figs_partial.svg — each figure wrapped in <!-- FIGn -->.

Run:  uv run --python 3.12 gen_figs.py
"""
import math
import pathlib

HERE = pathlib.Path(__file__).parent


class S:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
                  f'font-family="var(--sans)" role="img">']

    def esc(self, s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def t(self, x, y, s, size=13, fill="var(--ink)", anc="middle", w="normal", mono=False, it=False):
        fam = ' font-family="var(--mono)"' if mono else ""
        st = ' font-style="italic"' if it else ""
        self.p.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                      f'text-anchor="{anc}" font-weight="{w}"{fam}{st}>{self.esc(s)}</text>')

    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, stroke="var(--hair)", w=1.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.p.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                      f'stroke="{stroke}" stroke-width="{w}"{d}/>')

    def poly(self, pts, stroke="var(--accent)", w=2.4, fill="none", dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pp = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.p.append(f'<polyline points="{pp}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
                      f'stroke-linejoin="round" stroke-linecap="round"{d}/>')

    def area(self, pts, y0, fill="var(--amber-bg)"):
        pp = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        x_first, x_last = pts[0][0], pts[-1][0]
        self.p.append(f'<polygon points="{x_first:.1f},{y0:.1f} {pp} {x_last:.1f},{y0:.1f}" '
                      f'fill="{fill}" stroke="none"/>')

    def circ(self, x, y, r, fill="var(--accent)", stroke="none", sw=1.0):
        self.p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
                      f'stroke="{stroke}" stroke-width="{sw}"/>')

    def arrow(self, x1, y1, x2, y2, stroke="var(--faint)", w=2.0, head=7):
        self.line(x1, y1, x2, y2, stroke=stroke, w=w)
        dx, dy = x2 - x1, y2 - y1
        n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        self.p.append(f'<polygon points="{x2:.1f},{y2:.1f} '
                      f'{x2-head*ux+head*0.5*px:.1f},{y2-head*uy+head*0.5*py:.1f} '
                      f'{x2-head*ux-head*0.5*px:.1f},{y2-head*uy-head*0.5*py:.1f}" fill="{stroke}"/>')

    def done(self):
        self.p.append("</svg>")
        return "\n".join(self.p)


def curve(f, x0, x1, y0, y1, lo, hi, n=64, xr=(-2.6, 2.6)):
    """map f over xr into a pixel box (x0..x1 left→right, y0 top .. y1 bottom), value lo..hi."""
    pts = []
    for i in range(n + 1):
        xr_ = xr[0] + (xr[1] - xr[0]) * i / n
        v = f(xr_)
        px = x0 + (x1 - x0) * i / n
        py = y1 - (y1 - y0) * (v - lo) / (hi - lo)
        pts.append((px, py))
    return pts


def gauss(mu, sig):
    return lambda x: math.exp(-((x - mu) ** 2) / (2 * sig * sig))


def plotbox(s, x, y, w, h, title, sub, xlab="feature value", ylab="risk"):
    s.rect(x, y, w, h, fill="var(--panel)", stroke="var(--hair)", sw=1.2, rx=10)
    s.t(x + 12, y - 20, title, 13.5, "var(--ink)", "start", "bold", mono=True)
    if sub:
        s.t(x + 12, y - 4, sub, 12, "var(--muted)", "start")
    s.t(x + w - 6, y + h + 18, xlab + " →", 11, "var(--faint)", "end")
    s.t(x + 6, y + 12, ylab, 11, "var(--faint)", "start")


# ============================================================ FIG 1 — two kinds of risk
def fig1():
    s = S(660, 250)
    bw, bh, by = 258, 150, 56
    plotbox(s, 24, by, bw, bh, "smooth / monotone", "risk tracks magnitude")
    sig = lambda x: 1 / (1 + math.exp(-2.1 * (x - 0.1)))
    s.poly(curve(sig, 24 + 6, 24 + bw - 6, by + 12, by + bh - 10, 0.02, 0.98), stroke="var(--accent)", w=2.6)

    x2 = 378
    plotbox(s, x2, by, bw, bh, "localized (sharp)", "risk in a narrow band")
    g = gauss(0.15, 0.16)
    pts = curve(g, x2 + 6, x2 + bw - 6, by + 12, by + bh - 10, 0.0, 1.0, n=120)
    s.area(pts, by + bh - 10, fill="var(--amber-bg)")
    s.poly(pts, stroke="var(--amber)", w=2.6)
    # tick marking "the band"
    bx = x2 + 6 + (x2 + bw - 6 - (x2 + 6)) * (0.15 + 2.6) / 5.2
    s.line(bx, by + bh - 10, bx, by + bh + 2, stroke="var(--amber)", w=1.4)
    s.t(bx, by + bh + 16, "the band", 10.5, "var(--amber)", "middle", mono=True)

    s.t(330, 238, "A scalar answers “how big?”  —  only the needle needs “exactly where?”",
        13, "var(--muted)", "middle", it=True)
    return s.done()


# ============================================================ FIG 2 — a scalar collapses the axis
def fig2():
    s = S(660, 300)
    ax0, ax1, ay = 60, 600, 66
    s.t(330, 30, "One point on the value axis — two ways to read it", 14, "var(--ink)", "middle", "bold")
    # axis with quantile regions
    s.line(ax0, ay, ax1, ay, stroke="var(--ink)", w=1.6)
    n = 8
    for i in range(n + 1):
        x = ax0 + (ax1 - ax0) * i / n
        s.line(x, ay - 5, x, ay + 5, stroke="var(--faint)", w=1.2)
    s.t(ax0, ay - 14, "low", 10.5, "var(--faint)", "start", mono=True)
    s.t(ax1, ay - 14, "high", 10.5, "var(--faint)", "end", mono=True)
    # the point (in region index 5, 0-based)
    reg = 5
    px = ax0 + (ax1 - ax0) * (reg + 0.5) / n
    s.circ(px, ay, 6.5, fill="var(--ink)")
    s.t(px, ay - 16, "x", 13, "var(--ink)", "middle", "bold", mono=True)

    # left branch: scalar
    lx = 150
    s.arrow(px, ay + 10, lx, 150, stroke="var(--accent-br)", w=1.6)
    s.rect(lx - 118, 150, 236, 78, fill="var(--accent-bg)", stroke="var(--accent-br)", sw=1.3, rx=12)
    s.t(lx, 176, "scalar", 13, "var(--accent)", "middle", "bold", mono=True)
    s.t(lx, 200, "→  1.74", 20, "var(--ink)", "middle", "bold", mono=True)
    s.t(lx, 220, "one number: the magnitude", 11.5, "var(--muted)", "middle")

    # right branch: basis strip
    rx = 500
    s.arrow(px, ay + 10, rx, 150, stroke="var(--red-br)", w=1.6)
    s.rect(rx - 138, 150, 276, 78, fill="var(--red-bg)", stroke="var(--red-br)", sw=1.3, rx=12)
    s.t(rx, 172, "local basis", 13, "var(--red)", "middle", "bold", mono=True)
    cw, cx0, cy = 28, rx - 132, 184
    activ = {5: 1.0, 4: 0.45, 6: 0.35}
    for i in range(8):
        a = activ.get(i, 0.0)
        fill = "var(--red)" if a >= 0.9 else ("var(--red-br)" if a > 0 else "var(--panel)")
        s.rect(cx0 + i * (cw + 3), cy, cw, 22, fill=fill, stroke="var(--red-br)", sw=1.0, rx=4)
    s.t(rx, 222, "which region the value falls in", 11.5, "var(--muted)", "middle")

    s.t(330, 268, "Same point. The scalar keeps the magnitude and throws away the location;",
        13, "var(--muted)", "middle", it=True)
    s.t(330, 286, "the basis keeps the location — which is what a localized target needs.",
        13, "var(--muted)", "middle", it=True)
    return s.done()


# ============================================================ FIG 3 — Obstacle 1: can't FORM
def fig3():
    s = S(660, 288)
    bw, bh, by = 258, 158, 44
    # LEFT: affine read of a scalar -> reachable set is a line; needle out of span
    plotbox(s, 24, by, bw, bh, "affine read of a scalar", "reachable = the span", ylab="")
    x0, x1 = 24 + 8, 24 + bw - 8
    y0, y1 = by + 14, by + bh - 12
    for k, b in [(0.9, 0.15), (0.55, 0.35), (0.2, 0.55)]:
        s.poly([(x0, y1 - (y1 - y0) * b), (x1, y1 - (y1 - y0) * min(0.98, b + k))],
               stroke="var(--faint)", w=1.6)
    g = gauss(0.2, 0.16)
    s.poly(curve(g, x0, x1, y0, y1, 0.0, 1.0, n=100), stroke="var(--amber)", w=2.4, dash="5 4")
    s.t(24 + bw / 2, by + bh + 42, "monotone lines — the needle isn’t one", 11.5, "var(--muted)", "middle")

    # RIGHT: local basis -> weighted sum reproduces the needle
    x2 = 378
    plotbox(s, x2, by, bw, bh, "give it a local basis", "span now includes bumps", ylab="")
    bx0, bx1 = x2 + 8, x2 + bw - 8
    for mu, col in [(-0.9, "var(--green-br)"), (0.2, "var(--red-br)"), (1.3, "var(--green-br)")]:
        s.poly(curve(gauss(mu, 0.34), bx0, bx1, y0, y1, 0.0, 1.05, n=80), stroke=col, w=1.5)
    s.poly(curve(gauss(0.2, 0.16), bx0, bx1, y0, y1, 0.0, 1.0, n=100), stroke="var(--amber)", w=2.6)
    s.t(x2 + bw / 2, by + bh + 42, "weighted sum of local bumps = the needle", 11.5, "var(--muted)", "middle")

    s.t(330, 274, "One number spans a line; a basis spans bumps.", 12.5, "var(--muted)", "middle", it=True)
    return s.done()


# ============================================================ FIG 4 — Obstacle 2: can't FIND
def fig4():
    s = S(660, 288)
    bw, bh, by = 258, 158, 44
    y0, y1 = by + 14, by + bh - 12
    # LEFT: loss over "where to place the bump" — flat plateau + narrow deep notch
    plotbox(s, 24, by, bw, bh, "loss vs. where to place it", "flat, except at the answer",
            xlab="candidate location", ylab="loss")
    x0, x1 = 24 + 8, 24 + bw - 8
    notch = lambda x: 1.0 - 0.92 * math.exp(-((x - 0.2) ** 2) / (2 * 0.14 ** 2))
    lpts = curve(notch, x0, x1, y0, y1, 0.0, 1.05, n=140)
    s.poly(lpts, stroke="var(--muted)", w=2.4)
    # ball on the plateau, flat gradient
    ballx = x0 + (x1 - x0) * (-1.4 + 2.6) / 5.2
    bally = y1 - (y1 - y0) * (notch(-1.4)) / 1.05
    s.circ(ballx, bally - 7, 7, fill="var(--red)")
    s.arrow(ballx + 12, bally - 7, ballx + 46, bally - 7, stroke="var(--red)", w=2.0)
    s.t(ballx + 30, bally - 15, "no slope", 10.5, "var(--red)", "middle", mono=True)
    s.t(24 + bw / 2, by + bh + 42, "SGD has nothing to follow — it wanders", 11.5, "var(--muted)", "middle")

    # RIGHT: with a pre-placed basis, loss is a convex bowl
    x2 = 378
    plotbox(s, x2, by, bw, bh, "with a pre-placed basis", "the target is a bowl",
            xlab="basis weight", ylab="loss")
    bx0, bx1 = x2 + 8, x2 + bw - 8
    bowl = lambda x: 0.08 + 0.9 * ((x - 0.1) / 2.6) ** 2
    bpts = curve(bowl, bx0, bx1, y0, y1, 0.0, 1.05, n=100)
    s.poly(bpts, stroke="var(--green)", w=2.4)
    s.circ(bx0 + (bx1 - bx0) * (1.7 + 2.6) / 5.2, y0 + 16, 7, fill="var(--green)")
    s.arrow(bx0 + (bx1 - bx0) * (1.5 + 2.6) / 5.2, y0 + 30,
            bx0 + (bx1 - bx0) * (0.5 + 2.6) / 5.2, y1 - 18, stroke="var(--green)", w=2.0)
    s.t(x2 + bw / 2, by + bh + 42, "gradient points home", 11.5, "var(--muted)", "middle")

    s.t(330, 274, "A quantile basis hands over the location instead of making SGD search for it.",
        12.5, "var(--muted)", "middle", it=True)
    return s.done()


# ============================================================ FIG 5 — when a scalar is enough
def fig5():
    s = S(660, 214)
    s.t(330, 30, "When is a scalar enough?", 15, "var(--ink)", "middle", "bold")
    bx0, bx1, by, bh = 40, 620, 70, 30
    # gradient bar via segments
    segs = 60
    for i in range(segs):
        f = i / (segs - 1)
        # accent -> amber blend approximated by two overlaid rects with opacity
        x = bx0 + (bx1 - bx0) * i / segs
        w = (bx1 - bx0) / segs + 0.6
        s.rect(x, by, w, bh, fill="var(--accent)")
    for i in range(segs):
        f = i / (segs - 1)
        x = bx0 + (bx1 - bx0) * i / segs
        w = (bx1 - bx0) / segs + 0.6
        s.p.append(f'<rect x="{x:.1f}" y="{by}" width="{w:.1f}" height="{bh}" fill="var(--amber)" '
                   f'opacity="{f:.3f}"/>')
    s.rect(bx0, by, bx1 - bx0, bh, fill="none", stroke="var(--hair)", sw=1.0, rx=6)
    s.t(bx0, by - 10, "smooth · monotone", 12, "var(--accent)", "start", "bold", mono=True)
    s.t(bx1, by - 10, "sharp · localized", 12, "var(--amber)", "end", "bold", mono=True)
    # crossover
    cx = bx0 + (bx1 - bx0) * 0.6
    s.line(cx, by - 4, cx, by + bh + 8, stroke="var(--ink)", w=1.6, dash="4 3")
    # zone labels
    s.t((bx0 + cx) / 2, by + bh + 30, "scalar suffices", 13.5, "var(--accent)", "middle", "bold")
    s.t((bx0 + cx) / 2, by + bh + 48, "extra dimensions are just cost", 11.5, "var(--muted)", "middle")
    s.t((cx + bx1) / 2, by + bh + 30, "encode it", 13.5, "var(--amber)", "middle", "bold")
    s.t((cx + bx1) / 2, by + bh + 48, "relieve form and / or find", 11.5, "var(--muted)", "middle")
    s.t(330, 200, "Affine-read models (a GRU gate, a linear head) shift the crossover left — they need the basis sooner.",
        12, "var(--muted)", "middle", it=True)
    return s.done()


figs = [fig1(), fig2(), fig3(), fig4(), fig5()]
out = []
for i, f in enumerate(figs, 1):
    out.append(f"<!-- FIG{i} -->")
    out.append(f)
(HERE / "figs_partial.svg").write_text("\n".join(out))
print(f"wrote figs_partial.svg ({sum(len(f) for f in figs)} bytes of svg across {len(figs)} figures)")
