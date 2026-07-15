# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 7 Step-6 experiment — Non-monotonicity vs value-curvature (multivariate, controlled).

Addresses the debate findings:
  F2 (FATAL): the old discriminator (S2a = rank/CDF) is isomorphic to PLE's quantile basis
              -> add S2b (arcsinh, primary; x^0.3, robustness): monotone, log-mismatched, NON-quantile.
              S2a is kept only as a basis-aligned reference.
  F1: per-encoding regularization (LogisticRegressionCV for the linear head; MLP early_stopping)
      so the tie controls (S1, MLP) are not sabotaged by PLE's 96-dim overfitting.
  F3: 8 seeds, seed-level paired-t 95% CIs + Holm over the decision family, +/-0.005 equivalence.
  F4: one heterogeneous-shape + correlated arm on S2b.

Note: in a STATIC model the `dense` encoding collapses into the MLP head (linear+Linear-ReLU = a
1-hidden-layer MLP), so the free-nonlinearity path is the MLP head; the encoding axis {raw,log,ple}
lives on the linear (affine-read) head.

Decisive test: `ple - log` on the LINEAR head in S2b (arcsinh). CI clear of 0 and >+0.005,
with S1 + all-MLP tie controls clean -> H_curv general (curvature-via-combination, basis-agnostic).
"""
import numpy as np
from scipy.stats import rankdata, t as tdist
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOC, SCALE = 3.0, 1.6
N = 4000
K = 6
N_BINS = 16
COEF = 1.0
SEEDS = list(range(8))
BASE_RATE = 0.085
EQUIV = 0.005
SHAPES = ["S1_logfit", "S2a_rank", "S2b_arcsinh", "S2b_pow", "S3_nonmono", "S2b_hetero"]
ENCS = ["raw", "log", "ple"]
HEADS = ["linear", "mlp"]


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


def _shape_fn(xk, shape, hetero_idx=None):
    logxk = np.log(xk)
    if shape == "S1_logfit":
        g, _, _ = _std(np.log1p(xk))
    elif shape == "S2a_rank":                       # quantile-aligned (basis-favorable reference)
        g, _, _ = _std(rankdata(xk) / len(xk))
    elif shape == "S2b_arcsinh":                    # monotone, log-mismatched, NON-quantile
        g, _, _ = _std(np.arcsinh(xk))
    elif shape == "S2b_pow":                         # robustness: power-law, non-quantile
        g, _, _ = _std(xk ** 0.3)
    elif shape == "S3_nonmono":
        s, _, _ = _std(logxk)
        g, _, _ = _std(s ** 2)
    elif shape == "S2b_hetero":                      # mixed non-quantile monotone shapes (F4)
        forms = [np.arcsinh(xk), xk ** 0.3, np.log1p(xk) ** 1.5, np.sqrt(xk),
                 np.arcsinh(xk * 2), xk ** 0.25]
        g, _, _ = _std(forms[hetero_idx % len(forms)])
    else:
        raise ValueError(shape)
    return g


def make_data(shape, n, rng):
    latent = rng.normal(LOC, SCALE, size=(n, K))
    if shape == "S2b_hetero":                        # add mild positive correlation across features
        common = rng.normal(0, 1, size=(n, 1))
        latent = LOC + SCALE * (0.7 * (latent - LOC) / SCALE + 0.3 * common)
    X = np.exp(latent)
    G = np.column_stack([_shape_fn(X[:, k], shape, hetero_idx=k) for k in range(K)])
    logit = COEF * G.sum(axis=1)
    b = _calibrate(logit, BASE_RATE)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(int)
    return X, y, logit + b


def ple_edges(x, nbins):
    e = np.quantile(x, np.linspace(0, 1, nbins + 1))
    e[0] -= 1e-9; e[-1] += 1e-9
    for i in range(1, len(e)):
        if e[i] <= e[i - 1]:
            e[i] = e[i - 1] + 1e-9
    return e


def ple_transform(x, edges):
    out = np.zeros((len(x), len(edges) - 1))
    for t in range(len(edges) - 1):
        out[:, t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def encode(name, Xtr, Xte, edges_list):
    if name in ("raw", "log"):
        tr, te = [], []
        for k in range(K):
            vtr = Xtr[:, k] if name == "raw" else np.log1p(Xtr[:, k])
            vte = Xte[:, k] if name == "raw" else np.log1p(Xte[:, k])
            a, mu, sd = _std(vtr); b, _, _ = _std(vte, mu, sd)
            tr.append(a[:, None]); te.append(b[:, None])
        return np.hstack(tr), np.hstack(te)
    return (np.hstack([ple_transform(Xtr[:, k], edges_list[k]) for k in range(K)]),
            np.hstack([ple_transform(Xte[:, k], edges_list[k]) for k in range(K)]))


def fit_score(head, Xtr, ytr, Xte, yte, seed):
    if head == "linear":                             # per-encoding C via CV (F1)
        m = LogisticRegressionCV(Cs=np.logspace(-2, 2, 5), cv=3, max_iter=4000, scoring="average_precision")
    else:                                            # free-nonlinearity; early stopping (F1)
        m = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=12, max_iter=400, random_state=seed)
    m.fit(Xtr, ytr)
    return average_precision_score(yte, m.predict_proba(Xte)[:, 1])


# ---- run grid ----
ap = {(s, h, e): [] for s in SHAPES for h in HEADS for e in ENCS}
oracle = {s: [] for s in SHAPES}
brates = []
for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    for shape in SHAPES:
        Xtr, ytr, _ = make_data(shape, N, rng)
        Xte, yte, olog = make_data(shape, N, rng)
        brates.append(yte.mean()); oracle[shape].append(average_precision_score(yte, olog))
        edges = [ple_edges(Xtr[:, k], N_BINS) for k in range(K)]
        for enc in ENCS:
            Etr, Ete = encode(enc, Xtr, Xte, edges)
            for head in HEADS:
                ap[(shape, head, enc)].append(fit_score(head, Etr, ytr, Ete, yte, seed))


def paired_ci(a, b):                                  # seed-level paired-t 95% CI of (a-b)
    d = np.asarray(a) - np.asarray(b)
    n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    h = tdist.ppf(0.975, n - 1) * se
    tstat = mean / (se + 1e-12)
    p = 2 * tdist.sf(abs(tstat), n - 1)
    return mean, mean - h, mean + h, p


# ---- decision family: ple - log on both heads x all shapes; Holm-correct ----
fam = [(s, h) for s in SHAPES for h in HEADS]
raw_stats = {}
for s, h in fam:
    raw_stats[(s, h)] = paired_ci(ap[(s, h, "ple")], ap[(s, h, "log")])
order = sorted(fam, key=lambda k: raw_stats[k][3])   # ascending p for Holm
mtests = len(order)
holm = {}
for i, k in enumerate(order):
    holm[k] = min(raw_stats[k][3] * (mtests - i), 1.0)

print(f"=== base rate {np.mean(brates):.3f}; N={N}, K={K}, {len(SEEDS)} seeds ===")
print("=== precondition: oracle PR-AUC (>> base) ===")
for s in SHAPES:
    print(f"  {s:14s} {np.mean(oracle[s]):.3f}")

print("\n=== PR-AUC mean (8 seeds) ===")
print(f"{'shape':14s} {'head':7s} {'raw':>6s} {'log':>6s} {'ple':>6s}")
for s in SHAPES:
    for h in HEADS:
        print(f"{s:14s} {h:7s} " + " ".join(f"{np.mean(ap[(s,h,e)]):6.3f}" for e in ENCS))

print("\n=== ple - log : seed-level paired-t 95% CI [Holm p] ===")
print(f"{'shape':14s} {'head':7s} {'mean':>7s} {'95% CI':>18s} {'Holm p':>8s}  verdict")
for s in SHAPES:
    for h in HEADS:
        mean, lo, hi, _ = raw_stats[(s, h)]
        hp = holm[(s, h)]
        if lo > EQUIV:
            v = "PLE WINS"
        elif hi < -EQUIV:
            v = "PLE LOSES"
        elif lo > -EQUIV and hi < EQUIV:
            v = "TIE (equiv)"
        else:
            v = "inconclusive"
        star = "*" if (lo > 0 or hi < 0) and hp < 0.05 else " "
        print(f"{s:14s} {h:7s} {mean:+7.3f} [{lo:+.3f},{hi:+.3f}] {hp:8.3f}{star} {v}")

print("\n=== DECISIVE (linear head): does PLE beat log on a NON-quantile monotone signal? ===")
for s in ["S2b_arcsinh", "S2b_pow", "S2b_hetero"]:
    mean, lo, hi, _ = raw_stats[(s, "linear")]
    print(f"  {s:14s} ple-log = {mean:+.3f} [{lo:+.3f},{hi:+.3f}]  Holm p={holm[(s,'linear')]:.3f}")
print("  (S2a_rank linear is the basis-aligned reference; S1 + MLP cells are the tie controls)")

# figure: ple-log with CIs, linear head, per shape
fig, ax = plt.subplots(figsize=(8, 4.4))
xs = [s for s in SHAPES]
means = [raw_stats[(s, "linear")][0] for s in xs]
los = [raw_stats[(s, "linear")][0] - raw_stats[(s, "linear")][1] for s in xs]
ax.bar(range(len(xs)), means, yerr=[los, los], capsize=4, color="#4477aa")
ax.axhline(0, color="k", lw=0.8); ax.axhline(EQUIV, color="gray", ls=":", lw=0.8)
ax.axhline(-EQUIV, color="gray", ls=":", lw=0.8)
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=20, ha="right")
ax.set_ylabel("ple − log (PR-AUC), linear head"); ax.set_title("Cycle 7 — PLE vs log by shape (affine-read head, 95% CI)")
plt.tight_layout(); plt.savefig("finding_ple_vs_log_linear.png", dpi=110)
print("\nsaved finding_ple_vs_log_linear.png")
