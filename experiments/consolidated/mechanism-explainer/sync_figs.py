# /// script
# requires-python = ">=3.10"
# ///
"""Sync the inline <svg> figure blocks in encoding-mechanism.html from figs_partial.svg.

Each figure in the HTML is preceded by an <!-- FIGn --> marker; this replaces the
<svg>...</svg> that follows each marker with the freshly generated block, so the
page stays regenerable from gen_figs.py without hand-editing the SVG markup.

Run:  uv run --python 3.12 sync_figs.py
"""
import re
import pathlib

HERE = pathlib.Path(__file__).parent
partial = (HERE / "figs_partial.svg").read_text()
html_path = HERE / "encoding-mechanism.html"
html = html_path.read_text()

# parse figs_partial into {n: svg_text}
figs = {}
for m in re.finditer(r"<!-- FIG(\d+) -->\n(<svg.*?</svg>)", partial, re.S):
    figs[m.group(1)] = m.group(2)

n_swapped = 0
for n, svg in figs.items():
    pat = re.compile(rf"(<!-- FIG{n} -->\s*)<svg.*?</svg>", re.S)
    html, k = pat.subn(lambda mm, s=svg: mm.group(1) + s, html)
    n_swapped += k

html_path.write_text(html)
print(f"synced {n_swapped} figure block(s) into {html_path.name}")
