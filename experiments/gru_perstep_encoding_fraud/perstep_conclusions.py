# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figures for the per-step-encoding investigation. Reads flow/stats_results.json.
    uv run perstep_conclusions.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "stats_results.json").read_text())
ARMS = ["tab_logreg", "raw", "scalar", "ple", "dense", "oracle"]
LENGTHS = sorted({r["cell"]["length"] for r in S["aggregate_results"]})


def _mean(regime, length, method):
    for r in S["aggregate_results"]:
        c = r["cell"]
        if c["regime"] == regime and c["length"] == length and r["method"] == method:
            return r["pr_auc_mean"]
    return float("nan")


def fig_summary():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    x = np.arange(len(ARMS))
    w = 0.38
    for ax, reg in zip(axes, ["band", "monotone"]):
        for i, L in enumerate(LENGTHS):
            ax.bar(x + (i - 0.5) * w, [_mean(reg, L, m) for m in ARMS], w, label=f"L={L}", alpha=0.85)
        ax.set_title(reg); ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=18, ha="right")
        ax.set_xlabel("arm")
    axes[0].set_ylabel("PR-AUC (5-seed mean)"); axes[-1].legend(fontsize=8)
    fig.suptitle("Per-step encoding in an affine-input GRU — PR-AUC by arm × length × regime\n"
                 "band: ple/dense beat scalar (unbottlenecking helps); monotone: scalar suffices")
    fig.tight_layout(); fig.savefig(HERE / "fig_summary_prauc.png", dpi=120); plt.close(fig)


def fig_forest():
    rows = sorted(S["lift_results"],
                  key=lambda e: (e["cell"]["regime"], e["cell"]["length"], e["lift_mean"]))
    y = np.arange(len(rows))
    means = [e["lift_mean"] for e in rows]
    xerr = np.array([[e["lift_mean"] - e["lift_lo"] for e in rows],
                     [e["lift_hi"] - e["lift_mean"] for e in rows]])
    colors = ["#d62728" if e["ci_excludes_zero"] and e["lift_mean"] < 0
              else ("#2ca02c" if e["ci_excludes_zero"] else "#888") for e in rows]
    labels = [f"{e['cell']['regime']} L{e['cell']['length']}: {e['cell']['pair'].replace('_minus_','−')}"
              + ("  [Holm]" if e.get("holm_significant") else "")
              + ("  ≡" if e.get("equivalent_to_scalar") else "") for e in rows]
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(rows))))
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, color=colors, s=60, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-S["equiv_margin"], S["equiv_margin"], color="#ccc", alpha=0.3, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("PR-AUC lift — seed-level paired-t 95% CI; green=+sig, red=−sig, grey=overlaps 0")
    ax.set_title("Per-step encoding lifts (seed-level paired): band unbottlenecking vs monotone control")
    fig.tight_layout(); fig.savefig(HERE / "fig_lift_forest.png", dpi=120); plt.close(fig)


def main():
    fig_summary(); fig_forest()
    print("wrote fig_summary_prauc.png, fig_lift_forest.png")
    print("controls:", json.dumps(S["controls"], indent=2))
    print("\nband lifts:")
    for e in S["lift_results"]:
        if e["cell"]["regime"] == "band":
            flags = ("*" if e["ci_excludes_zero"] else "") + ("H" if e.get("holm_significant") else "")
            print(f"  L{e['cell']['length']} {e['cell']['pair']:18s} {e['lift_mean']:+.3f} "
                  f"[{e['lift_lo']:+.3f},{e['lift_hi']:+.3f}] p={e['p']:.3f}{flags}")


if __name__ == "__main__":
    main()
