# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
# ]
# ///
"""
ple_numeric_poc.py — Minimal proof-of-concept for the hypothesis:

    An MLP on piecewise-linear-encoded (PLE) numeric features beats the same MLP on
    raw standardized features, when the target depends NON-MONOTONICALLY on the inputs.

End to end in one command:  uv run ple_numeric_poc.py
Produces: three AUC-ROC numbers (logistic-regression baseline, raw-MLP, PLE-MLP) and a
two-panel mechanism figure (ple_poc_mechanism.png).

Reference implementation note
-----------------------------
PLE is from Gorishniy et al., 2022, "On Embeddings for Numerical Features in Tabular Deep
Learning" (the `rtdl_num_embeddings` package). This PoC implements the *quantile* PLE variant
by hand for transparency and zero extra deps. The encoding here matches the paper's
definition; it does NOT use the tree-based bin variant.

Deliberately left OUT of this PoC (all deferred to the Step 6 experiment):
  - Bootstrap confidence intervals on AUC (single point estimate only here).
  - The LINEAR-TARGET CONTROL condition (the hypothesis's falsification lever): if PLE wins
    equally on a linear target, the benefit is not the claimed non-monotonic mechanism.
  - Parameter-count matching: PLE-MLP has a larger first layer (D*T inputs vs D), so it has
    more parameters than raw-MLP. This capacity confound is left in on purpose for the debate.
  - Tree-based PLE bins (quantile bins only).
  - Hyperparameter tuning, multiple seeds, multiple architectures.
  - A GBDT reference (design is MLP-only).
  - Any real dataset.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
N = 8000
D = 8                # total features
D_INFORMATIVE = 4    # first D_INFORMATIVE features drive a non-monotonic target
N_BINS = 24          # PLE bins per feature (T)
HIDDEN = (64, 64)    # identical MLP backbone for raw-MLP and PLE-MLP


# --------------------------------------------------------------------------- data
def make_data(n, d, d_informative, seed):
    """Non-monotonic target: logit = sum_j amp_j * sin(freq_j * x_j) over informative
    features. The sine makes P(y=1) oscillate with x — exactly the structure a raw-input
    MLP must spend capacity to approximate and PLE can represent near-linearly."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    freqs = rng.uniform(1.5, 3.0, size=d_informative)
    amps = rng.uniform(1.0, 2.0, size=d_informative)
    logit = np.zeros(n)
    for j in range(d_informative):
        logit += amps[j] * np.sin(freqs[j] * X[:, j])
    logit_center = logit.mean()
    logit -= logit_center
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return X, y, {"freqs": freqs, "amps": amps, "logit_center": logit_center}


# ----------------------------------------------------------------------------- PLE
def fit_ple_edges(X, n_bins):
    """Quantile bin edges per feature, fit on TRAINING data only. Shape (d, n_bins+1)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(X, qs, axis=0).T  # (d, n_bins+1)
    eps = 1e-6
    for j in range(edges.shape[0]):  # guard against collided quantiles -> zero-width bins
        for t in range(1, edges.shape[1]):
            if edges[j, t] <= edges[j, t - 1]:
                edges[j, t] = edges[j, t - 1] + eps
    return edges


def ple_transform(X, edges):
    """Piecewise-linear encoding. For each bin t: clip((x - lo)/(hi - lo), 0, 1).
    Bins below x saturate to 1, the active bin holds the fractional position, bins above
    clip to 0. Returns (n, d * n_bins)."""
    n, d = X.shape
    n_bins = edges.shape[1] - 1
    out = np.empty((n, d, n_bins), dtype=np.float64)
    for t in range(n_bins):
        lo = edges[:, t]
        hi = edges[:, t + 1]
        out[:, :, t] = np.clip((X - lo) / (hi - lo), 0.0, 1.0)
    return out.reshape(n, d * n_bins)


# --------------------------------------------------------------------------- models
def new_mlp():
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN,
        activation="relu",
        alpha=1e-4,
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=SEED,
    )


def main():
    X, y, truth = make_data(N, D, D_INFORMATIVE, SEED)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    print(f"n_train={len(ytr)}  n_test={len(yte)}  base_rate={y.mean():.3f}  "
          f"d={D} (informative={D_INFORMATIVE})  n_bins={N_BINS}")

    # --- raw encoding: standardize on train
    scaler = StandardScaler().fit(Xtr)
    Xtr_raw, Xte_raw = scaler.transform(Xtr), scaler.transform(Xte)

    # --- PLE encoding: edges fit on train
    edges = fit_ple_edges(Xtr, N_BINS)
    Xtr_ple, Xte_ple = ple_transform(Xtr, edges), ple_transform(Xte, edges)
    print(f"raw input dim={Xtr_raw.shape[1]}   ple input dim={Xtr_ple.shape[1]}")

    # --- trivial baseline (non-negotiable): logistic regression on raw features
    logreg = LogisticRegression(max_iter=1000).fit(Xtr_raw, ytr)
    auc_logreg = roc_auc_score(yte, logreg.predict_proba(Xte_raw)[:, 1])

    # --- raw-MLP vs PLE-MLP (identical backbone)
    mlp_raw = new_mlp().fit(Xtr_raw, ytr)
    auc_raw = roc_auc_score(yte, mlp_raw.predict_proba(Xte_raw)[:, 1])

    mlp_ple = new_mlp().fit(Xtr_ple, ytr)
    auc_ple = roc_auc_score(yte, mlp_ple.predict_proba(Xte_ple)[:, 1])

    # ------------------------------------------------------------------- summary
    print("\n=== AUC-ROC on held-out test set ===")
    print(f"  logistic regression (trivial baseline) : {auc_logreg:.4f}")
    print(f"  raw-MLP                                 : {auc_raw:.4f}")
    print(f"  PLE-MLP                                 : {auc_ple:.4f}")
    print(f"\n  PLE - raw  = {auc_ple - auc_raw:+.4f}")
    print(f"  PLE - base = {auc_ple - auc_logreg:+.4f}")
    verdict = "PLE > raw" if auc_ple > auc_raw else "PLE <= raw"
    print(f"  point-estimate read: {verdict}  (no CI yet — Step 6 adds bootstrap)")

    # ------------------------------------------------- mechanism visualization
    # Panel A: along feature 0, hold others at 0 (their mean). Compare true P(y=1)
    # to raw-MLP vs PLE-MLP predictions. Panel B: PLE bin activations for feature 0.
    grid = np.linspace(-3.0, 3.0, 300)
    Xg = np.zeros((grid.size, D))
    Xg[:, 0] = grid

    true_logit = truth["amps"][0] * np.sin(truth["freqs"][0] * grid) - truth["logit_center"]
    true_p = 1.0 / (1.0 + np.exp(-true_logit))

    p_raw = mlp_raw.predict_proba(scaler.transform(Xg))[:, 1]
    p_ple = mlp_ple.predict_proba(ple_transform(Xg, edges))[:, 1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    axA.plot(grid, true_p, "k--", lw=2, label="true P(y=1)")
    axA.plot(grid, p_raw, color="tab:red", lw=2, label=f"raw-MLP (AUC {auc_raw:.3f})")
    axA.plot(grid, p_ple, color="tab:blue", lw=2, label=f"PLE-MLP (AUC {auc_ple:.3f})")
    axA.set_title("Mechanism: tracking a non-monotonic target along feature 0")
    axA.set_xlabel("feature 0 value (others held at mean)")
    axA.set_ylabel("P(y=1)")
    axA.legend(loc="best", fontsize=9)

    ple_grid = ple_transform(Xg, edges).reshape(grid.size, D, N_BINS)[:, 0, :]
    for t in range(N_BINS):
        axB.plot(grid, ple_grid[:, t], lw=1, alpha=0.7)
    axB.set_title(f"PLE encoding of feature 0 ({N_BINS} bins)")
    axB.set_xlabel("feature 0 value")
    axB.set_ylabel("bin activation")

    fig.tight_layout()
    fig.savefig("ple_poc_mechanism.png", dpi=120)
    print("\nwrote ple_poc_mechanism.png")


if __name__ == "__main__":
    main()
