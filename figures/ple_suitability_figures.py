# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""Illustrative figures for PLE_SUITABILITY_DIAGNOSTICS.md.

Regenerates, from the consolidated flow's own risk-shape DGP (standardized-log feature `s`), the
figures that show WHAT a PLE-suitable feature looks like and how the quantile-binned diagnostic
distinguishes it. No experiment is run — pure synthetic illustration.

Shapes (risk of fraud log-odds vs value): sharp band `exp(-s^2/2σ^2)` (σ=0.15), smooth U `s^2`,
monotone-curved `s^3`, log-linear `s`. Metrics and thresholds mirror notebooks/encoding_diagnostics.py.

Outputs (this dir):
  ple_fig1_sharp_band_histogram.png   -- feature histogram + fraud-rate overlay (the localized band)
  ple_fig2_sharp_band_quantile.png    -- fraud rate by 8 vs 20 quantile bins (PLE's resolution)
  ple_fig3_risk_shapes_diagnostic.png -- 4 shapes x [fraud rate %, logit(rate)] over quantile bins + verdict

Run:  uv run ple_suitability_figures.py
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

HERE = pathlib.Path(__file__).parent
RNG = np.random.default_rng(0)
N, SIGMA, BASE, W = 400_000, 0.15, 0.08, 2.6
# diagnostic thresholds (from notebooks/encoding_diagnostics.py)
NONMONO_T, CURV_T, SHARP_T = 0.30, 0.05, 0.37

RED, GREY, BLUE = "#d62728", "#cfcfcf", "#3b6ea5"


def risk(shape, s):
    if shape == "sharp band":       return np.exp(-(s ** 2) / (2 * SIGMA ** 2))
    if shape == "smooth U":         return s ** 2
    if shape == "monotone-curved":  return s ** 3
    if shape == "log-linear":       return s
    raise ValueError(shape)


def solve_b(z, target):
    lo, hi = -25.0, 25.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if (1.0 / (1.0 + np.exp(-(z + m)))).mean() < target: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def sample(shape, s):
    g = risk(shape, s)
    gz = (g - g.mean()) / (g.std() + 1e-9)
    logit = W * gz
    return (RNG.random(len(s)) < 1.0 / (1.0 + np.exp(-(logit + solve_b(logit, BASE))))).astype(int)


def quantile_diag(s, y, nbins):
    """Fraud rate + Haldane-logit per quantile bin, plus non_mono / curv_gap / contig (notebook metrics)."""
    q = np.unique(np.quantile(s, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(s, q[1:-1]), 0, len(q) - 2)
    keep = [b for b in range(len(q) - 1) if (idx == b).any()]
    ctr = np.array([s[idx == b].mean() for b in keep])
    k = np.array([y[idx == b].sum() for b in keep], float)
    cnt = np.array([(idx == b).sum() for b in keep], float)
    rate = k / cnt
    lg = np.log((k + 0.5) / (cnt - k + 0.5))              # logit — removes the sigmoid link
    w = cnt / cnt.sum(); lbar = np.average(lg, weights=w)
    sst = np.average((lg - lbar) ** 2, weights=w) + 1e-12
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(ctr, lg, sample_weight=cnt)
    r2_iso = 1 - np.average((lg - iso.predict(ctr)) ** 2, weights=w) / sst
    lin = LinearRegression().fit(ctr[:, None], lg, sample_weight=cnt)
    r2_lin = 1 - np.average((lg - lin.predict(ctr[:, None])) ** 2, weights=w) / sst
    exc = np.abs(lg - lbar) * w; thr = np.median(exc); runs, cur = [], 0.0
    for e in exc:
        if e > thr: cur += e
        elif cur > 0: runs.append(cur); cur = 0.0
    if cur > 0: runs.append(cur)
    contig = (max(runs) / exc.sum()) if runs and exc.sum() > 0 else 0.0
    return dict(ctr=ctr, rate=rate, lg=lg, iso=iso.predict(ctr),
                non_mono=max(0.0, 1 - r2_iso), curv_gap=max(0.0, r2_iso - r2_lin), contig=contig)


def classify(m):
    if m["non_mono"] > NONMONO_T:
        return ("non-monotone-sharp", "PLE (fixed quantile knots)") if m["contig"] > SHARP_T \
            else ("non-monotone-smooth", "learned projection")
    return ("monotone-curved", "learned projection") if m["curv_gap"] > CURV_T else ("log-linear", "log scalar")


# ---------------------------------------------------------------- shared data
s = RNG.normal(0.0, 1.0, N)
y_sharp = sample("sharp band", s)


# ---------------------------------------------------------------- fig 1: histogram + rate overlay
def fig1():
    bins = np.linspace(-3, 3, 49); ctr = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(s, bins)
    idx = np.clip(np.digitize(s, bins) - 1, 0, len(bins) - 2)
    rate = np.array([y_sharp[idx == k].mean() if (idx == k).any() else np.nan for k in range(len(bins) - 1)])
    band = 3 * SIGMA
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axvspan(-band, band, color="#ffe0e0", alpha=0.7, zorder=0, label=f"sharp band (|s| < 3σ = {band:.2f})")
    ax.bar(ctr, counts, width=(bins[1] - bins[0]) * 0.95, color=GREY, zorder=1, label="feature distribution")
    for j, k in enumerate(np.quantile(s, np.linspace(0, 1, 9))):
        ax.axvline(k, color=BLUE, ls="--", lw=0.8, alpha=0.7, zorder=2,
                   label="PLE quantile knots (8 bins)" if j == 0 else None)
    ax.set_xlabel("feature value  s  (standardized log)"); ax.set_ylabel("count", color="#777")
    ax.set_ylim(0, counts.max() * 1.15)
    ax2 = ax.twinx()
    ax2.plot(ctr, rate * 100, "o-", color=RED, ms=4.5, lw=2.0, zorder=3, label="fraud rate per bin (%)")
    ax2.axhline(BASE * 100, color=RED, ls=":", lw=1.2, alpha=0.7, label=f"base rate ({BASE:.0%})")
    ax2.set_ylabel("fraud rate (%)", color=RED); ax2.set_ylim(0, np.nanmax(rate) * 100 * 1.15)
    ax.set_title("A “sharp band”: fraud risk localized to a narrow slice of the value range\n"
                 "PLE-suitable — an affine (monotone) read of s cannot form this spike; quantile knots resolve it",
                 fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout(); fig.savefig(HERE / "ple_fig1_sharp_band_histogram.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig 2: rate by quantile bins (8 vs 20)
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, nbins in zip(axes, (8, 20)):
        m = quantile_diag(s, y_sharp, nbins); x = np.arange(len(m["rate"]))
        ax.bar(x, m["rate"] * 100, color=[RED if r > 2 * BASE else GREY for r in m["rate"]],
               edgecolor="#999", linewidth=0.4, width=0.9)
        ax.axhline(BASE * 100, color=RED, ls=":", lw=1.3, alpha=0.8, label=f"base rate ({BASE:.0%})")
        ax.set_xticks(x); ax.set_xticklabels([f"{v:+.2f}" for v in m["ctr"]], rotation=90, fontsize=7)
        ax.set_xlabel("quantile bin  (mean feature value  s)")
        ax.set_title(f"{nbins} quantile bins" + ("   ← PLE's resolution" if nbins == 8 else "   (finer screen)"))
        ax.legend(fontsize=8)
        for xi, r in zip(x, m["rate"]):
            if r > 2 * BASE:
                ax.annotate(f"{r*100:.0f}%", (xi, r * 100), ha="center", va="bottom", fontsize=7, color="#b00000")
    axes[0].set_ylabel("fraud rate in bin (%)")
    fig.suptitle("Fraud rate by quantile-binned feature — a sharp band as PLE sees it\n"
                 "quantile bins are narrow where data is dense (the mode), so the band lands in a few "
                 "CONTIGUOUS central bins; fewer bins dilute the spike", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(HERE / "ple_fig2_sharp_band_quantile.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- fig 3: 4 shapes x [rate, logit] diagnostic
def fig3(nbins=20):
    shapes = ["sharp band", "smooth U", "monotone-curved", "log-linear"]
    fig, axes = plt.subplots(len(shapes), 2, figsize=(12.5, 13.5))
    for row, shape in enumerate(shapes):
        y = y_sharp if shape == "sharp band" else sample(shape, s)
        m = quantile_diag(s, y, nbins); x = np.arange(len(m["rate"]))
        cls, enc = classify(m)
        # col 0: fraud rate %
        a0 = axes[row, 0]
        a0.bar(x, m["rate"] * 100, color=[RED if r > 2 * BASE else GREY for r in m["rate"]],
               edgecolor="#999", linewidth=0.3, width=0.92)
        a0.axhline(BASE * 100, color=RED, ls=":", lw=1.2, alpha=0.8)
        a0.set_ylabel("fraud rate (%)"); a0.set_title(f"{shape}  —  fraud rate by quantile bin", fontsize=10)
        # col 1: logit(rate) + isotonic fit (the diagnostic axis)
        a1 = axes[row, 1]
        a1.plot(x, m["lg"], "o-", color=BLUE, ms=4, lw=1.6, label="logit(fraud rate)")
        a1.plot(x, m["iso"], "-", color="#e08214", lw=2.0, label="isotonic (monotone) fit")
        a1.set_ylabel("logit fraud rate")
        a1.set_title(f"non_mono={m['non_mono']:.2f}  contig={m['contig']:.2f}  curv_gap={m['curv_gap']:.2f}"
                     f"   →  {cls}", fontsize=9.5)
        a1.legend(fontsize=7, loc="best")
        a1.text(0.02, 0.03, f"encoder: {enc}", transform=a1.transAxes, fontsize=8.5, color="#006400",
                fontweight="bold", va="bottom")
        for a in (a0, a1):
            a.set_xticks([]); a.set_xlabel("quantile bin (low s → high s)" if row == len(shapes) - 1 else "")
    fig.suptitle("PLE-suitability by risk shape — fraud rate and logit(rate) over quantile bins\n"
                 f"thresholds: non_mono>{NONMONO_T} ⇒ non-monotone; then contig>{SHARP_T} ⇒ SHARP→PLE, "
                 f"else smooth→projection; curv_gap>{CURV_T} ⇒ curved→projection",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(HERE / "ple_fig3_risk_shapes_diagnostic.png", dpi=150)
    plt.close(fig)
    # echo the verdict table
    print(f"{'shape':>16} {'non_mono':>9} {'contig':>7} {'curv_gap':>9}  -> class / encoder")
    for shape in shapes:
        y = y_sharp if shape == "sharp band" else sample(shape, s)
        m = quantile_diag(s, y, nbins); cls, enc = classify(m)
        print(f"{shape:>16} {m['non_mono']:9.2f} {m['contig']:7.2f} {m['curv_gap']:9.2f}  -> {cls} / {enc}")


if __name__ == "__main__":
    fig1(); print("wrote fig1")
    fig2(); print("wrote fig2")
    fig3()
