# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figures for the periodic-embedding investigation. Reads stats_results.json.
    uv run periodic_conclusions.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "stats_results.json").read_text())

# display order: trivial baseline, then GRU arms by encoding richness
ARMS = ["tab_logreg", "base_raw", "cyc_sincos", "cyc_periodic", "dt_periodic", "all_periodic"]
COL = {"tab_logreg": "#555", "base_raw": "#999", "cyc_sincos": "#1f77b4",
       "cyc_periodic": "#2ca02c", "dt_periodic": "#d62728", "all_periodic": "#9467bd"}
LIFT_LABELS = {
    "cyc_periodic_minus_cyc_sincos": "H1: periodic − sin/cos (cyclic)",
    "dt_periodic_minus_cyc_sincos": "H2: dt periodic − raw",
    "cyc_sincos_minus_base_raw": "ENC: sin/cos − raw",
    "all_periodic_minus_cyc_sincos": "ALL: all-periodic − sin/cos",
    "cyc_sincos_minus_tab_logreg": "BASE: GRU(sin/cos) − tab",
    "cyc_periodic_minus_tab_logreg": "BASE: GRU(periodic) − tab",
}


def fig_summary():
    agg = {r["method"]: r for r in S["aggregate_results"]}
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ARMS))
    means = [agg[m].get("pr_auc_mean", agg[m]["pr_auc_seedmean"]) for m in ARMS]
    lo = [agg[m].get("pr_auc_lo", means[i]) for i, m in enumerate(ARMS)]
    hi = [agg[m].get("pr_auc_hi", means[i]) for i, m in enumerate(ARMS)]
    yerr = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])
    ax.bar(x, means, 0.6, yerr=yerr, capsize=4, color=[COL[m] for m in ARMS], alpha=0.88)
    ax.axhline(0.072, color="k", ls=":", lw=0.9, label="base rate ~0.072")
    ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=20, ha="right")
    ax.set_ylabel("PR-AUC (test, 95% bootstrap CI)")
    ax.set_title("Periodic embeddings on fraud-GRU time features — PR-AUC by arm (L=32)\n"
                 "learned periodic vs fixed sin/cos vs raw; amount + context held fixed")
    ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "fig_summary_prauc.png", dpi=120); plt.close(fig)


def fig_forest():
    rows = [e for e in S["lift_results"] if e["pair"] in LIFT_LABELS]
    rows = sorted(rows, key=lambda e: e["lift_mean"])
    y = np.arange(len(rows))
    means = [e["lift_mean"] for e in rows]
    xerr = np.array([[e["lift_mean"] - e["lift_lo"] for e in rows],
                     [e["lift_hi"] - e["lift_mean"] for e in rows]])
    colors = ["#d62728" if e["ci_excludes_zero"] and e["lift_mean"] < 0
              else ("#2ca02c" if e["ci_excludes_zero"] else "#888") for e in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, color=colors, s=70, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([LIFT_LABELS[e["pair"]] for e in rows])
    ax.set_xlabel("PR-AUC lift (paired test bootstrap, 95% CI) — L=32")
    ax.set_title("Periodic-embedding lifts: grey = CI overlaps 0 (no effect); "
                 "green = +sig; red = −sig")
    fig.tight_layout(); fig.savefig(HERE / "fig_lift_forest.png", dpi=120); plt.close(fig)


def main():
    fig_summary(); fig_forest()
    print("wrote fig_summary_prauc.png, fig_lift_forest.png")
    # console summary
    agg = {r["method"]: r for r in S["aggregate_results"]}
    print("\nPR-AUC by arm (mean [CI]):")
    for m in ARMS:
        r = agg[m]
        pm = r.get("pr_auc_mean", r["pr_auc_seedmean"])
        lo, hi = r.get("pr_auc_lo", float("nan")), r.get("pr_auc_hi", float("nan"))
        print(f"  {m:14s} {pm:.3f} [{lo:.3f}, {hi:.3f}]  seedmean={r['pr_auc_seedmean']:.3f}")
    print("\nKey lifts:")
    for e in S["lift_results"]:
        if e["pair"] in LIFT_LABELS:
            star = "*" if e["ci_excludes_zero"] else " "
            print(f"  {LIFT_LABELS[e['pair']]:32s} {e['lift_mean']:+.3f} "
                  f"[{e['lift_lo']:+.3f}, {e['lift_hi']:+.3f}]{star}")


if __name__ == "__main__":
    main()
