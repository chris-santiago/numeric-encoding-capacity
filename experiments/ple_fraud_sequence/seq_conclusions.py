# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figures for the sequence amount-in-context investigation. Reads stats_results.json.
    uv run seq_conclusions.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "stats_results.json").read_text())
ARMS = ["tab_last", "tab_aggregate", "seq_raw", "seq_ple", "seq_dev", "seq_raw_shuffle"]
COL = {"tab_last": "#555", "tab_aggregate": "#999", "seq_raw": "#1f77b4",
       "seq_ple": "#2ca02c", "seq_dev": "#d62728", "seq_raw_shuffle": "#9467bd"}


def fig_summary():
    agg = {(r["cell"]["length"], r["method"]): r for r in S["aggregate_results"]}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(ARMS))
    w = 0.38
    for i, L in enumerate([8, 32]):
        means = [agg[(L, m)]["pr_auc_mean"] for m in ARMS]
        lo = [agg[(L, m)]["pr_auc_lo"] for m in ARMS]
        hi = [agg[(L, m)]["pr_auc_hi"] for m in ARMS]
        yerr = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])
        ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=3, label=f"L={L}", alpha=0.85)
    base = 0.072
    ax.axhline(base, color="k", ls=":", lw=0.9, label=f"base rate ~{base:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(ARMS, rotation=20, ha="right")
    ax.set_ylabel("PR-AUC (test, 95% bootstrap CI)")
    ax.set_title("Sequence amount-in-context — PR-AUC by arm (real account data)\n"
                 "tabular logreg > GRU; PLE & deviation add nothing; shuffle doesn't hurt")
    ax.legend()
    fig.tight_layout(); fig.savefig(HERE / "summary_prauc_seq.png", dpi=120)


def fig_forest():
    labels = {"seq_dev_minus_seq_raw": "sub(a): seq_dev − seq_raw",
              "seq_ple_minus_seq_raw": "sub(b): seq_ple − seq_raw",
              "seq_raw_minus_tab_aggregate": "F2: seq_raw − tab_aggregate",
              "seq_raw_minus_seq_raw_shuffle": "lever: seq_raw − shuffle",
              "seq_raw_minus_tab_last": "seq_raw − tab_last"}
    rows = [e for e in S["lift_results"] if e["cell"]["length"] == 32]
    rows = [e for e in rows if e["pair"] in labels]
    rows = sorted(rows, key=lambda e: e["lift_mean"])
    y = np.arange(len(rows))
    means = [e["lift_mean"] for e in rows]
    xerr = np.array([[e["lift_mean"] - e["lift_lo"] for e in rows],
                     [e["lift_hi"] - e["lift_mean"] for e in rows]])
    colors = ["#d62728" if e["ci_excludes_zero"] and e["lift_mean"] < 0
              else ("#2ca02c" if e["ci_excludes_zero"] else "#888") for e in rows]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, color=colors, s=60, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([labels[e["pair"]] for e in rows])
    ax.set_xlabel("PR-AUC lift (paired test bootstrap, 95% CI) — L=32")
    ax.set_title("Sequence lifts (L=32): GRU loses to tabular; deviation hurts; PLE & order null")
    fig.tight_layout(); fig.savefig(HERE / "lift_forest_seq.png", dpi=120)


def main():
    fig_summary(); fig_forest()
    print("wrote summary_prauc_seq.png, lift_forest_seq.png")


if __name__ == "__main__":
    main()
