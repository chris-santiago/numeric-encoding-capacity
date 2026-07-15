# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib>=3.8"]
# ///
"""Step 7 figures for the Δt-encoding investigation. Reads stats_results.json.
    uv run dt_conclusions.py
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "stats_results.json").read_text())
ENC = ["raw", "log", "ple_raw", "ple_log", "learned", "learned_reg", "log_expand"]
MODELS = ["linear", "mlp"]


def fig_summary():
    agg = {(r["regime"], r["encoding"], r["model"]): r for r in S["aggregate"]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    x = np.arange(len(ENC)); w = 0.38
    for ax, reg in zip(axes, ["nonmono", "mono"]):
        for i, mt in enumerate(MODELS):
            means = [agg[(reg, e, mt)]["ap_seedmean"] for e in ENC]
            ax.bar(x + (i - 0.5) * w, means, w, label=mt, alpha=0.86)
        ax.axhline(S["base_rate_target"], color="k", ls=":", lw=0.9, label=f"base {S['base_rate_target']:.2f}")
        ax.set_title(reg); ax.set_xticks(x); ax.set_xticklabels(ENC, rotation=18, ha="right")
        ax.set_xlabel("Δt encoding")
    axes[0].set_ylabel("PR-AUC (mean over 5 seeds)"); axes[-1].legend(fontsize=8)
    fig.suptitle("Δt encoding × model capacity × regime — under a capable model (MLP), no encoding "
                 "beats log; learned periodic overfits")
    fig.tight_layout(); fig.savefig(HERE / "fig_summary_prauc.png", dpi=120); plt.close(fig)


def fig_forest():
    rows = sorted(S["lifts"], key=lambda e: e["mean"])  # SEED-LEVEL paired CIs (decision-relevant)
    y = np.arange(len(rows))
    means = [e["mean"] for e in rows]
    xerr = np.array([[e["mean"] - e["lo"] for e in rows], [e["hi"] - e["mean"] for e in rows]])
    colors = ["#d62728" if e["ci_excludes_zero"] and e["mean"] < 0
              else ("#2ca02c" if e["ci_excludes_zero"] else "#888") for e in rows]
    labels = [e["tag"] + ("  ~equiv" if e.get("equivalent_to_log") else "")
              + ("  [Holm]" if e.get("holm_significant") else "") for e in rows]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="k", capsize=3, zorder=1)
    ax.scatter(means, y, color=colors, s=70, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(-S["equiv_margin"], S["equiv_margin"], color="#ccc", alpha=0.3, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("PR-AUC lift — SEED-LEVEL paired-t 95% CI (n=5 seeds); grey band = ±equivalence margin")
    ax.set_title("Δt encoding lifts (between-seed paired): green=+sig, red=−sig, grey=overlaps 0\n"
                 "positive control fires; PLE ties/≡ log; learned worse (overfit) but regularized "
                 "learned ≈ log")
    fig.tight_layout(); fig.savefig(HERE / "fig_lift_forest.png", dpi=120); plt.close(fig)


def main():
    fig_summary(); fig_forest()
    print("wrote fig_summary_prauc.png, fig_lift_forest.png")
    print(f"convergence (nonmono,mlp) 5-seed-mean train-loss: {S['convergence_train_loss_nonmono_mlp_5seedmean']}")
    print(f"z-only floor: {S['z_only_floor']}")


if __name__ == "__main__":
    main()
