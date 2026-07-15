# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 7 v2 PoC — Non-monotonicity vs value-curvature (multivariate reformulation).

v1 finding: for a SINGLE feature under a rank metric (PR-AUC), any monotone encoding gives
identical rankings, so curvature is invisible and only non-monotonicity opens a gap. Value-curvature
is a MULTIVARIATE phenomenon: a monotone-but-curved feature's shape matters only through how it
combines additively with other features (this is why C1 helped in a multivariate fraud model).

Decisive question (v2): does PLE beat `log` for an AFFINE-READ model on a SUM of monotone-but-log-
mismatched features (S2)? If yes, log-mismatched value-curvature — with no non-monotonicity — is a
lever via combination.

Grid: 3 signal shapes x 2 heads x 3 encodings {raw, log, ple}. K=6 additive features.
  Heads: linear (affine-read = LogisticRegression) and mlp (free-nonlinearity = MLPClassifier).
  Shapes: S1 log-fit, S2 mono-log-mismatch (DISCRIMINATOR), S3 non-monotone.
Metric: PR-AUC.

DELIBERATELY LEFT OUT (Step 6): `dense` encoding, bootstrap/paired-t CIs + Holm, formal control gates,
large seed count.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATENT = dict(loc=3.0, scale=1.6)
N = 6000
K = 6            # additive features
N_BINS = 16
COEF = 1.0       # per-feature signal weight (sum of 6 std-1 terms -> logit std ~2.4)
SEEDS = [0, 1, 2]
SHAPES = ["S1_logfit", "S2_logmismatch", "S3_nonmono"]
ENCS = ["raw", "log", "ple"]
HEADS = ["linear", "mlp"]
BASE_RATE = 0.085


def _std(v, mu=None, sd=None):
    if mu is None:
        mu, sd = v.mean(), v.std() + 1e-12
    return (v - mu) / sd, mu, sd


def _calibrate(logit, target):
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _feature_signal(xk, shape):
    logxk = np.log(xk)
    if shape == "S1_logfit":
        g, _, _ = _std(np.log1p(xk))
    elif shape == "S2_logmismatch":
        g, _, _ = _std(rankdata(xk) / len(xk))
    elif shape == "S3_nonmono":
        s, _, _ = _std(logxk)
        g, _, _ = _std(s ** 2)
    else:
        raise ValueError(shape)
    return g


def make_data(shape, n, rng):
    latent = rng.normal(LATENT["loc"], LATENT["scale"], size=(n, K))
    X = np.exp(latent)
    G = np.column_stack([_feature_signal(X[:, k], shape) for k in range(K)])
    logit = COEF * G.sum(axis=1)
    b = _calibrate(logit, BASE_RATE)
    p = 1 / (1 + np.exp(-(logit + b)))
    y = (rng.random(n) < p).astype(int)
    return X, y, logit + b          # return oracle logit for precondition check


def ple_edges(x, nbins):
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def ple_transform(x, edges):
    T = len(edges) - 1
    out = np.zeros((len(x), T))
    for t in range(T):
        out[:, t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def encode(name, Xtr, Xte, edges_list):
    if name in ("raw", "log"):
        cols_tr, cols_te = [], []
        for k in range(K):
            vtr = Xtr[:, k] if name == "raw" else np.log1p(Xtr[:, k])
            vte = Xte[:, k] if name == "raw" else np.log1p(Xte[:, k])
            a, mu, sd = _std(vtr)
            b, _, _ = _std(vte, mu, sd)
            cols_tr.append(a[:, None]); cols_te.append(b[:, None])
        return np.hstack(cols_tr), np.hstack(cols_te)
    if name == "ple":
        tr = [ple_transform(Xtr[:, k], edges_list[k]) for k in range(K)]
        te = [ple_transform(Xte[:, k], edges_list[k]) for k in range(K)]
        return np.hstack(tr), np.hstack(te)
    raise ValueError(name)


def fit_score(head, Xtr, ytr, Xte, yte, seed):
    model = (LogisticRegression(max_iter=3000, C=1.0) if head == "linear"
             else MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=seed))
    model.fit(Xtr, ytr)
    return average_precision_score(yte, model.predict_proba(Xte)[:, 1])


results = {k: [] for k in ((s, h, e) for s in SHAPES for h in HEADS for e in ENCS)}
oracle_ap = {s: [] for s in SHAPES}
base_rates = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for shape in SHAPES:
        Xtr, ytr, _ = make_data(shape, N, rng)
        Xte, yte, oracle_logit = make_data(shape, N, rng)
        base_rates.append(yte.mean())
        oracle_ap[shape].append(average_precision_score(yte, oracle_logit))  # precondition
        edges = [ple_edges(Xtr[:, k], N_BINS) for k in range(K)]
        for enc in ENCS:
            Etr, Ete = encode(enc, Xtr, Xte, edges)
            for head in HEADS:
                results[(shape, head, enc)].append(fit_score(head, Etr, ytr, Ete, yte, seed))


def m(k):
    return float(np.mean(results[k]))


print(f"=== base rate: {np.mean(base_rates):.3f} (target {BASE_RATE}); K={K} additive features ===")
print("=== precondition (oracle PR-AUC, must be >> base) ===")
for s in SHAPES:
    print(f"  {s:16s} oracle AP = {np.mean(oracle_ap[s]):.3f}")

print("\n=== PR-AUC (mean over 3 seeds) ===")
print(f"{'shape':16s} {'head':7s} {'raw':>6s} {'log':>6s} {'ple':>6s}  {'ple-log':>8s}")
for shape in SHAPES:
    for head in HEADS:
        r, l, p = m((shape, head, "raw")), m((shape, head, "log")), m((shape, head, "ple"))
        print(f"{shape:16s} {head:7s} {r:6.3f} {l:6.3f} {p:6.3f}  {p - l:+8.3f}")

print("\n=== DECISIVE TEST: ple - log on the LINEAR (affine-read) head ===")
for shape in SHAPES:
    d = m((shape, "linear", "ple")) - m((shape, "linear", "log"))
    v = "PLE WINS" if d > 0.005 else ("tie" if abs(d) <= 0.005 else "PLE loses")
    print(f"  {shape:16s} ple-log = {d:+.3f}   [{v}]")
print("\nH_curv predicts: PLE WINS in S2 and S3, tie in S1 (linear head); MLP head ties everywhere.")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
w = 0.38
for i, head in enumerate(HEADS):
    vals = [m((s, head, "ple")) - m((s, head, "log")) for s in SHAPES]
    ax.bar(np.arange(len(SHAPES)) + i * w, vals, w, label=head)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(0.005, color="gray", ls=":", lw=0.8, label="±0.005 margin")
ax.axhline(-0.005, color="gray", ls=":", lw=0.8)
ax.set_xticks(np.arange(len(SHAPES)) + w / 2)
ax.set_xticklabels(SHAPES, rotation=12)
ax.set_ylabel("PR-AUC(ple) − PR-AUC(log)")
ax.set_title(f"Cycle 7 v2 PoC — PLE-over-log ({K}-feature additive) by shape and head")
ax.legend()
plt.tight_layout()
plt.savefig("curvature_poc_mechanism.png", dpi=110)
print("\nsaved curvature_poc_mechanism.png")
