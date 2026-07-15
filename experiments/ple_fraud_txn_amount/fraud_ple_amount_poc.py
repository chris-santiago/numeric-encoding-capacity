# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
# ]
# ///
"""fraud_ple_amount_poc.py — Step 1 PoC for the fraud extension hypothesis:

    PLE-encoding the transaction-amount feature beats raw (log) amount in PR-AUC on a
    fraud task with a NON-MONOTONIC fraud-vs-amount curve, under a TEMPORAL split.

Synthetic fraud-amount surrogate (no external data). One command:
    uv run fraud_ple_amount_poc.py
Produces PR-AUC / ROC-AUC / precision@k for three arms and a 2-panel figure.

Reference: PLE = Gorishniy et al. 2022 (quantile variant), implemented by hand.

Surrogate structure (the design targets the cycle-2 open questions):
  - amount ~ lognormal (heavy-tailed), with a mild upward drift in scale over time.
  - fraud logit = base
                  + c_main * U(amount)              # main non-monotonic (U-shape) effect
                  + cat_extra[category] * U(amount)  # amount x category INTERACTION
                  + small card-testing spike at very low amounts
                  + linear generic-feature signal
                  + temporal drift in base rate
    where U(amount) = standardized_log_amount**2 (high at both tails, low mid-range).
  - heavy class imbalance (~1-3% fraud).
  - TEMPORAL split: sort by time, first 70% train, last 30% test (past -> future).

Arms (minimal PoC set; full matrix deferred to Step 6):
  - logreg_raw  : trivial baseline (raw log-amount + one-hot category + generics)
  - logreg_ple  : hypothesis arm  (PLE amount + one-hot category + generics)
  - hgb_raw     : HistGradientBoosting on raw features (GBDT reference; bins + interactions natively)

Deliberately left OUT (deferred to the Step 6 experiment):
  - Bootstrap confidence intervals (single point estimate here).
  - The MONOTONIC-amount control (falsification lever).
  - The full arm matrix (MLP, hgb_ple, PLE+interaction crosses, parameter matching).
  - n_bins sensitivity; multiple seeds.
  - Real IEEE-CIS data (Step 6), identity-table joins, velocity/aggregation features.
  - precision@k threshold tuning, recall@fixed-FPR (only PR-AUC + a single precision@k here).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
N = 40000
N_CAT = 5            # merchant-category levels
N_BINS = 24          # PLE bins for amount
TRAIN_FRAC = 0.70    # temporal split: first 70% by time -> train


# --------------------------------------------------------------------------- data
def make_data(n, seed):
    """Synthetic fraud surrogate with a non-monotonic fraud-vs-amount curve, an
    amount x category interaction, temporal drift, and heavy imbalance."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 1.0, n))                  # time, ascending
    log_amount = rng.normal(3.0 + 0.3 * t, 1.0)            # scale drifts up over time
    amount = np.exp(log_amount)
    category = rng.integers(0, N_CAT, n)
    G = rng.standard_normal((n, 4))                        # 2 weakly-informative + 2 noise

    la_std = (log_amount - log_amount.mean()) / log_amount.std()
    u = la_std ** 2                                        # U-shape: high at both tails
    cat_extra = np.array([-0.4, 0.0, 0.4, 0.9, 1.5])[category]  # interaction strength per cat

    logit = (
        -5.0                                              # base -> heavy imbalance
        + 0.8 * u                                         # MAIN non-monotonic amount effect
        + cat_extra * u                                   # amount x category INTERACTION
        + 1.0 * (la_std < -1.5)                           # card-testing spike at tiny amounts
        + 0.4 * G[:, 0] - 0.3 * G[:, 1]                   # linear generic signal
        + 0.6 * t                                         # temporal drift in base rate
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return {"t": t, "amount": amount, "log_amount": log_amount,
            "category": category, "G": G, "y": y}


# ----------------------------------------------------------------------------- PLE
def fit_ple_edges(x, n_bins):
    """Quantile PLE edges for a 1-D feature, fit on training values. Shape (n_bins+1,)."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    eps = 1e-9
    for t in range(1, edges.size):                        # guard collided quantiles
        if edges[t] <= edges[t - 1]:
            edges[t] = edges[t - 1] + eps
    return edges


def ple_transform_1d(x, edges):
    """PLE-encode a 1-D feature: per bin t, clip((x-lo)/(hi-lo), 0, 1). (n, n_bins)."""
    n_bins = edges.size - 1
    out = np.empty((x.size, n_bins))
    for t in range(n_bins):
        out[:, t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def onehot(category, n_cat):
    """Fit-free one-hot for a known integer category range (no leakage)."""
    return np.eye(n_cat)[category]


def build_features(d, idx, scaler_amt, scaler_g, edges, encoding):
    """Assemble the model matrix for rows `idx` under the chosen amount encoding."""
    log_amt = d["log_amount"][idx].reshape(-1, 1)
    g = scaler_g.transform(d["G"][idx])
    cat = onehot(d["category"][idx], N_CAT)
    if encoding == "raw":
        amt = scaler_amt.transform(log_amt)               # standardized log-amount scalar
        return np.hstack([amt, cat, g])
    if encoding == "ple":
        amt = ple_transform_1d(d["amount"][idx], edges)   # PLE bins replace the scalar
        return np.hstack([amt, cat, g])
    if encoding == "raw_gbdt":
        # GBDT gets raw log-amount + category as an ordinal column it can split on + generics.
        return np.hstack([log_amt, d["category"][idx].reshape(-1, 1).astype(float), d["G"][idx]])
    raise ValueError(f"Unknown encoding {encoding!r}")


def precision_at_k(scores, labels, k):
    """Fraction of true positives among the top-k highest-scored items."""
    k = max(1, min(k, len(scores)))
    top = np.argsort(-scores)[:k]
    return float(labels[top].sum() / k)


def main():
    d = make_data(N, SEED)
    n = N
    n_train = int(TRAIN_FRAC * n)
    tr = np.arange(0, n_train)            # temporal: earliest 70%
    te = np.arange(n_train, n)            # latest 30% (future)
    ytr, yte = d["y"][tr], d["y"][te]
    print(f"n_train={len(tr)} n_test={len(te)}  "
          f"fraud_rate_train={ytr.mean():.4f} fraud_rate_test={yte.mean():.4f}")

    # Fit preprocessing on TRAIN only.
    scaler_amt = StandardScaler().fit(d["log_amount"][tr].reshape(-1, 1))
    scaler_g = StandardScaler().fit(d["G"][tr])
    edges = fit_ple_edges(d["amount"][tr], N_BINS)

    arms = {
        "logreg_raw": ("raw", LogisticRegression(max_iter=1000, class_weight="balanced")),
        "logreg_ple": ("ple", LogisticRegression(max_iter=1000, class_weight="balanced")),
        "hgb_raw": ("raw_gbdt", HistGradientBoostingClassifier(
            random_state=SEED, categorical_features=[1])),
    }

    k = int(yte.sum())  # precision@k at the natural review budget = number of true frauds
    results = {}
    for name, (enc, model) in arms.items():
        Xtr = build_features(d, tr, scaler_amt, scaler_g, edges, enc)
        Xte = build_features(d, te, scaler_amt, scaler_g, edges, enc)
        model.fit(Xtr, ytr)
        s = model.predict_proba(Xte)[:, 1]
        results[name] = {
            "pr_auc": average_precision_score(yte, s),
            "roc_auc": roc_auc_score(yte, s),
            "p_at_k": precision_at_k(s, yte, k),
        }

    print("\n=== Test metrics (temporal split, latest 30%) ===")
    print(f"  {'arm':12s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'P@k':>7s}   (k={k}, base rate={yte.mean():.4f})")
    for name, r in results.items():
        print(f"  {name:12s} {r['pr_auc']:8.4f} {r['roc_auc']:8.4f} {r['p_at_k']:7.4f}")
    print(f"\n  PR-AUC  logreg_ple - logreg_raw = {results['logreg_ple']['pr_auc'] - results['logreg_raw']['pr_auc']:+.4f}")
    print(f"  PR-AUC  hgb_raw    - logreg_ple = {results['hgb_raw']['pr_auc'] - results['logreg_ple']['pr_auc']:+.4f}  (interaction gap proxy)")

    # ----------------------------------------------------- mechanism visualization
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: empirical fraud rate by amount decile -> the non-monotonic U-shape.
    deciles = np.quantile(d["amount"], np.linspace(0, 1, 11))
    centers, rates = [], []
    for i in range(10):
        lo, hi = deciles[i], deciles[i + 1]
        m = (d["amount"] >= lo) & (d["amount"] <= hi if i == 9 else d["amount"] < hi)
        centers.append(0.5 * (lo + hi))
        rates.append(d["y"][m].mean())
    axA.plot(range(10), rates, "o-", color="tab:red")
    axA.set_title("Mechanism: fraud rate is non-monotonic (U-shaped) in amount")
    axA.set_xlabel("transaction-amount decile (low -> high)")
    axA.set_ylabel("empirical fraud rate")

    # Panel B: PR-AUC per arm.
    names = list(results)
    pr = [results[nm]["pr_auc"] for nm in names]
    bars = axB.bar(names, pr, color=["#bbbbbb", "#1f77b4", "#2ca02c"])
    for b, v in zip(bars, pr):
        axB.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    axB.axhline(yte.mean(), color="k", ls=":", lw=0.8, label=f"base rate {yte.mean():.3f}")
    axB.set_title("PR-AUC by arm (temporal split)")
    axB.set_ylabel("PR-AUC (average precision)")
    axB.legend()

    fig.tight_layout()
    fig.savefig("fraud_poc_mechanism.png", dpi=120)
    print("\nwrote fraud_poc_mechanism.png")


if __name__ == "__main__":
    main()
