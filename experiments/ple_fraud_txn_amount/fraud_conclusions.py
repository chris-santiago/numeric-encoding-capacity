# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figures for the IEEE-CIS PLE-on-amount investigation. Reads
flow/stats_results.json -> summary PR-AUC, lift forest plot, real fraud-vs-amount curve.
    uv run fraud_conclusions.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "stats_results.json").read_text())

FAM = {"logreg": "#1f77b4", "mlp": "#ff7f0e", "hgb": "#2ca02c"}


def _fam(method):
    return "hgb" if method.startswith("hgb") else ("mlp" if method.startswith("mlp") else "logreg")


def fig_summary():
    rows = sorted(S["aggregate_results"], key=lambda r: r.get("pr_auc_mean", 0))
    names = [r["method"] for r in rows]
    means = [r.get("pr_auc_mean", np.nan) for r in rows]
    lo = [r.get("pr_auc_lo", np.nan) for r in rows]
    hi = [r.get("pr_auc_hi", np.nan) for r in rows]
    xerr = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=xerr, capsize=3,
            color=[FAM[_fam(n)] for n in names], alpha=0.85)
    for yi, m in zip(y, means):
        ax.text(m + 0.004, yi, f"{m:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("PR-AUC (test, 95% bootstrap CI)")
    ax.set_title("Real IEEE-CIS — PR-AUC by arm (temporal split, 150k, 5 seeds)\n"
                 "GBDT dominates; PLE-on-amount gives no lift; PLE-on-C1 (placebo) jumps")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM.values()]
    ax.legend(handles, FAM.keys(), title="model", loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "summary_prauc_by_arm.png", dpi=120)


def fig_lift_forest():
    order = ["logreg_ple_raw_minus_logreg_raw", "mlp_ple_raw_minus_mlp_raw",
             "mlp_ple_log_minus_mlp_raw", "logreg_ple_raw_minus_logreg_quadratic",
             "logreg_ple_log_minus_logreg_ple_raw", "mlp_ple_log_minus_mlp_ple_raw",
             "hgb_ple_minus_hgb_raw", "logreg_ple_placebo_minus_logreg_raw"]
    labels = {"logreg_ple_raw_minus_logreg_raw": "H-main: logreg PLE−raw",
              "mlp_ple_raw_minus_mlp_raw": "T-MLP: mlp PLE−raw",
              "mlp_ple_log_minus_mlp_raw": "T-MLP: mlp PLE-log−raw",
              "logreg_ple_raw_minus_logreg_quadratic": "T-F1: PLE−quadratic",
              "logreg_ple_log_minus_logreg_ple_raw": "T-F2: ple-log−ple-raw",
              "mlp_ple_log_minus_mlp_ple_raw": "T-F2 neural: ple-log−ple-raw",
              "hgb_ple_minus_hgb_raw": "T-F5: hgb PLE−raw (interaction gap)",
              "logreg_ple_placebo_minus_logreg_raw": "T-lever: PLE on C1 (placebo)"}
    d = {e["pair"]: e for e in S["lift_results"]}
    rows = [d[p] for p in order if p in d]
    y = np.arange(len(rows))[::-1]
    means = [r["lift_mean"] for r in rows]
    xerr = np.array([[r["lift_mean"] - r["lift_lo"] for r in rows],
                     [r["lift_hi"] - r["lift_mean"] for r in rows]])
    colors = ["#2ca02c" if r["lift_mean"] > 0 and r["ci_excludes_zero"]
              else ("#d62728" if r["lift_mean"] < 0 and r["ci_excludes_zero"] else "#888888")
              for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, color=colors, s=60, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[r["pair"]] for r in rows])
    ax.set_xlabel("PR-AUC lift (paired test bootstrap, 95% CI)")
    ax.set_title("PLE lifts on real data — all amount lifts ≤0 (red); only C1 placebo is large (green)")
    fig.tight_layout()
    fig.savefig(HERE / "lift_forest.png", dpi=120)


def fig_fraud_curve():
    fc = S["analyses"]["ple_fraud::fraud_curve"]["result"]
    rates = fc["decile_fraud_rates"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(len(rates)), rates, "o-", color="tab:red")
    ax.set_xlabel("transaction-amount decile (low → high)")
    ax.set_ylabel("empirical fraud rate (train)")
    ax.set_title(f"Real IEEE-CIS fraud vs amount — U-shaped but weak\n"
                 f"monotonic|ρ|={fc['monotonic_abs_corr']:.2f}, U-shape ρ={fc['ushape_corr']:.2f} "
                 f"(range ~{max(rates)/min(rates):.1f}x)")
    fig.tight_layout()
    fig.savefig(HERE / "fraud_curve_real.png", dpi=120)


def main():
    fig_summary()
    fig_lift_forest()
    fig_fraud_curve()
    print("wrote summary_prauc_by_arm.png, lift_forest.png, fraud_curve_real.png")


if __name__ == "__main__":
    main()
