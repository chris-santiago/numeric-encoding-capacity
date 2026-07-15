# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figure generation for the PLE-vs-raw investigation.

Reads the promoted flow's archived stats (flow/stats_results.json) and produces the
canonical investigation figures: one summary + one per empirical test (T1-T5).
Run:  uv run ple_numeric_conclusions.py
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
# The promoted flow writes its archived stats to the investigation root (its parents[1]).
STATS = json.loads((HERE / "stats_results.json").read_text())

# --- index the artifacts ----------------------------------------------------
AGG = {(r["cell"]["target"], r["method"]): (r["auc_roc_mean"], r["auc_roc_lo"], r["auc_roc_hi"])
       for r in STATS["aggregate_results"]}
LIFT = {(r["pair"], r["cell"]["target"]): r for r in STATS["lift_results"]}
LIN = {r["target"]: r for r in STATS["analyses"]["ple_vs_raw::linearization"]["rows"]}
CONV = {r["method"]: r for r in STATS["analyses"]["ple_vs_raw::convergence"]["rows"]}

ARMS = ["logreg_raw", "logreg_ple", "mlp_raw", "mlp_ple", "mlp_raw_wide", "mlp_rff"]
COLORS = {"logreg_raw": "#bbbbbb", "logreg_ple": "#1f77b4", "mlp_raw": "#ff7f0e",
          "mlp_ple": "#2ca02c", "mlp_raw_wide": "#d62728", "mlp_rff": "#9467bd"}


# === Summary figure: AUC by arm and target ==================================
def fig_summary():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ARMS))
    w = 0.38
    for i, target in enumerate(["nonmono", "linear"]):
        means = [AGG[(target, m)][0] for m in ARMS]
        los = [AGG[(target, m)][1] for m in ARMS]
        his = [AGG[(target, m)][2] for m in ARMS]
        yerr = np.array([[m - lo for m, lo in zip(means, los)],
                         [hi - m for m, hi in zip(means, his)]])
        ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=3,
               label=f"target={target}", alpha=0.85)
    ax.axhline(0.5, color="k", ls=":", lw=0.8, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(ARMS, rotation=20, ha="right")
    ax.set_ylabel("AUC-ROC (mean, 95% bootstrap CI)")
    ax.set_ylim(0.45, 0.92)
    ax.set_title("Summary — AUC-ROC by arm and target (10 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "summary_all_conditions.png", dpi=120)


# === T1 — mechanism: PLE gain is model-agnostic =============================
def fig_t1():
    fig, ax = plt.subplots(figsize=(7, 5))
    pairs = [("logreg_ple_minus_logreg_raw", "logreg\n(PLE - raw)"),
             ("mlp_ple_minus_mlp_raw", "MLP\n(PLE - raw)")]
    means = [LIFT[(p, "nonmono")]["lift_mean"] for p, _ in pairs]
    yerr = np.array([[LIFT[(p, "nonmono")]["lift_mean"] - LIFT[(p, "nonmono")]["lift_lo"] for p, _ in pairs],
                     [LIFT[(p, "nonmono")]["lift_hi"] - LIFT[(p, "nonmono")]["lift_mean"] for p, _ in pairs]])
    ax.bar([l for _, l in pairs], means, 0.5, yerr=yerr, capsize=4,
           color=["#1f77b4", "#2ca02c"], alpha=0.85)
    for i, m in enumerate(means):
        ax.text(i, m + 0.012, f"+{m:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("AUC-ROC lift from PLE (nonmono target)")
    ax.set_title("T1 — PLE benefit is model-agnostic\nlinear model gains ~7x the MLP: general linearization, not MLP-specific")
    fig.tight_layout()
    fig.savefig(HERE / "finding_T1_mechanism.png", dpi=120)


# === T2 — capacity / basis control =========================================
def fig_t2():
    fig, ax = plt.subplots(figsize=(8, 5))
    arms = ["mlp_raw", "mlp_raw_wide", "mlp_rff", "mlp_ple"]
    means = [AGG[("nonmono", m)][0] for m in arms]
    yerr = np.array([[AGG[("nonmono", m)][0] - AGG[("nonmono", m)][1] for m in arms],
                     [AGG[("nonmono", m)][2] - AGG[("nonmono", m)][0] for m in arms]])
    bars = ax.bar(arms, means, 0.6, yerr=yerr, capsize=4,
                  color=[COLORS[m] for m in arms], alpha=0.85)
    for b, m in zip(bars, arms):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f"{CONV[m]['n_params']:,}p" if m in CONV else "", ha="center", fontsize=8)
    ax.set_ylabel("AUC-ROC (nonmono, 95% CI)")
    ax.set_ylim(0.6, 0.88)
    ax.set_title("T2 — capacity ruled out\nmlp_ple beats mlp_raw_wide (MORE params) and mlp_rff (matched dim, non-PLE basis)")
    fig.tight_layout()
    fig.savefig(HERE / "finding_T2_capacity.png", dpi=120)


# === T3 — convergence ======================================================
def fig_t3():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    arms = ["mlp_raw", "mlp_raw_wide", "mlp_rff", "mlp_ple"]
    means = [CONV[m]["n_iter_mean"] for m in arms]
    maxes = [CONV[m]["n_iter_max"] for m in arms]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, means, 0.4, label="n_iter mean", color="#4c72b0")
    ax.bar(x + 0.2, maxes, 0.4, label="n_iter max", color="#dd8452")
    ax.axhline(1000, color="r", ls="--", lw=1.2, label="max_iter cap = 1000")
    ax.set_xticks(x)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("training iterations")
    ax.set_yscale("log")
    ax.set_title("T3 — no MLP is iteration-limited\nall arms converge < 140 iters (cap 1000); raw-MLP not optimization-starved")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "finding_T3_convergence.png", dpi=120)


# === T4 — linearization R^2 ================================================
def fig_t4():
    fig, ax = plt.subplots(figsize=(7, 5))
    targets = ["nonmono", "linear"]
    x = np.arange(len(targets))
    raw = [LIN[t]["ridge_r2_raw"] for t in targets]
    ple = [LIN[t]["ridge_r2_ple"] for t in targets]
    ax.bar(x - 0.2, raw, 0.4, label="raw features", color="#ff7f0e", alpha=0.85)
    ax.bar(x + 0.2, ple, 0.4, label="PLE features", color="#2ca02c", alpha=0.85)
    for xi, (r, p) in enumerate(zip(raw, ple)):
        ax.text(xi - 0.2, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, p + 0.02, f"{p:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Ridge R² predicting the latent logit (held-out)")
    ax.set_ylim(0, 1.12)
    ax.set_title("T4 — PLE linearizes the non-monotonic target\nR² 0.10 -> 0.98 (nonmono); both ~1.0 on linear (as expected)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "finding_T4_linearization.png", dpi=120)


# === T5 — falsification lever ==============================================
def fig_t5():
    fig, ax = plt.subplots(figsize=(7, 5))
    targets = ["nonmono", "linear"]
    rows = [LIFT[("mlp_ple_minus_mlp_raw", t)] for t in targets]
    means = [r["lift_mean"] for r in rows]
    yerr = np.array([[r["lift_mean"] - r["lift_lo"] for r in rows],
                     [r["lift_hi"] - r["lift_mean"] for r in rows]])
    colors = ["#2ca02c" if m > 0 else "#d62728" for m in means]
    ax.bar(targets, means, 0.5, yerr=yerr, capsize=5, color=colors, alpha=0.85)
    ax.axhline(0, color="k", lw=1)
    for i, m in enumerate(means):
        ax.text(i, m + (0.004 if m > 0 else -0.006), f"{m:+.3f}",
                ha="center", va="bottom" if m > 0 else "top", fontweight="bold")
    ax.set_ylabel("mlp_ple - mlp_raw AUC lift (95% CI)")
    ax.set_title("T5 falsification lever — PLE helps only on non-monotonic structure\npositive & CI-separated on nonmono; reverses on the linear control")
    fig.tight_layout()
    fig.savefig(HERE / "finding_T5_falsification.png", dpi=120)


def main():
    fig_summary()
    fig_t1()
    fig_t2()
    fig_t3()
    fig_t4()
    fig_t5()
    print("wrote: summary_all_conditions.png, finding_T1_mechanism.png, "
          "finding_T2_capacity.png, finding_T3_convergence.png, "
          "finding_T4_linearization.png, finding_T5_falsification.png")

    # Console scorecard
    print("\n=== AUC-ROC (mean [lo, hi]) ===")
    for target in ["nonmono", "linear"]:
        print(f"  [{target}]")
        for m in ARMS:
            mean, lo, hi = AGG[(target, m)]
            print(f"    {m:14s} {mean:.4f} [{lo:.4f}, {hi:.4f}]")
    print("\n=== Key lifts (95% CI; * = CI excludes 0) ===")
    for (pair, target), r in sorted(LIFT.items()):
        star = "*" if r["ci_excludes_zero"] else " "
        print(f"  {star} {pair:30s} [{target:8s}] {r['lift_mean']:+.4f} "
              f"[{r['lift_lo']:+.4f}, {r['lift_hi']:+.4f}]")


if __name__ == "__main__":
    main()
