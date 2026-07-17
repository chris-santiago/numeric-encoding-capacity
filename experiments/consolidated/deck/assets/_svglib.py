"""Shared SVG helpers + palette for the consolidated-encoding deck figures.

Kept in assets/ next to the generators so every figure stays regenerable. Data
figures read deck_data.json (produced by extract_numbers.py from the real run);
conceptual figures carry only the DGP math or a schematic.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent

# Standard teaching-deck palette (matches gen_figure_template.py).
BLUE = "#1a4a7a"     # primary / headers / log reference
ORANGE = "#d97706"   # secondary / highlights
GREEN = "#15803d"    # good / target / learned projection
RED = "#b91c1c"      # decisive / fixed PLE
PURPLE = "#7c3aed"   # dense
GRAY = "#6b7280"     # annotations
LGRAY = "#d1d5db"    # rules / inactive / oracle ceiling
FILL_BLUE = "#eaf1f8"
FILL_GREEN = "#e4f0e6"
FILL_ORANGE = "#fde8cc"
FILL_RED = "#f7e2e2"

# encoder -> color, shared across data figures
ENC_COLOR = {"raw": GRAY, "log": BLUE, "ple": RED, "projection": GREEN,
             "dense": PURPLE, "oracle": LGRAY}


def load_data():
    return json.loads((HERE / "deck_data.json").read_text())


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'font-family="Helvetica,Arial,sans-serif">'
        ]

    def text(self, x, y, s, size=17, fill="#222", anchor="middle",
             weight="normal", style="", mono=False, rotate=None):
        fam = ' font-family="SFMono-Regular,Menlo,monospace"' if mono else ""
        st = f' font-style="{style}"' if style else ""
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{st}{fam}{tr}>{esc(s)}</text>'
        )

    def rect(self, x, y, w, h, fill="#fff", stroke=GRAY, sw=1.5, rx=5):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=GRAY, w=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{w}"{d}/>'
        )

    def polyline(self, pts, stroke=BLUE, w=2.5, fill="none"):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    def circle(self, x, y, r, fill=BLUE, stroke="none", sw=1.0):
        self.parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def arrow(self, x1, y1, x2, y2, stroke=GRAY, w=2, head=9):
        self.line(x1, y1, x2, y2, stroke=stroke, w=w)
        dx, dy = x2 - x1, y2 - y1
        n = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        self.parts.append(
            f'<polygon points="{x2:.1f},{y2:.1f} '
            f'{x2 - head * ux + head * 0.55 * px:.1f},{y2 - head * uy + head * 0.55 * py:.1f} '
            f'{x2 - head * ux - head * 0.55 * px:.1f},{y2 - head * uy - head * 0.55 * py:.1f}" '
            f'fill="{stroke}"/>'
        )

    def save(self, name):
        self.parts.append("</svg>")
        path = HERE / name
        path.write_text("\n".join(self.parts))
        print("wrote", path.name)
