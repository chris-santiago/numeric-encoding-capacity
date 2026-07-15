# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 7 Step-6 experiment (iteration 3, clean) — value-curvature above PLE's structural deficit.

Two fixes over experiment3:
  1. S1 is the DEFICIT BASELINE, not a tie control. On a log-optimal signal a flexible basis can
     only be >= slightly worse (a structural PLE cost, ~-0.04, not overfitting). So curvature benefit
     is measured as the seed-level paired difference-of-differences:
         Delta(shape) = (ple-log)_shape - (ple-log)_S1     [per seed, paired-t 95% CI + Holm]
     Delta > 0 (CI clear) = PLE beats log ABOVE its structural cost -> a real curvature benefit.
  2. arcsinh dropped (it is ~= log asymptotically, so not log-mismatched). Genuine log-mismatched
     monotone curves span the shape space, all defined on s = std(log x) (log-head sees ~ s, linear):
        S2b_convex = exp(0.7 s)      (convex)
        S2b_sat    = sigmoid(2 s)    (saturating S; an S-curve that is NOT the quantile coordinate)
        S2b_cubic  = s**3            (monotone cubic)
     S2a_quantile = rank(x) ~= Phi(s) is the basis-aligned reference (PLE's own coordinate).
     S3_nonmono = s**2 (U). S1_logfit = s (log-optimal).

Decisive: is Delta > 0 for the genuine log-mismatched S2b curves (curvature is a general lever), or
only for S2a_quantile (basis-alignment only)?
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
N, K, N_BINS, COEF = 4000, 6, 8, 1.0
SEEDS = list(range(8))
BASE_RATE, EQUIV = 0.085, 0.005
SHAPES = ["S1_logfit", "S2a_quantile", "S2b_convex", "S2b_sat", "S2b_cubic", "S3_nonmono"]
GENUINE = ["S2b_convex", "S2b_sat", "S2b_cubic"]     # genuine log-mismatched monotone curves
FAMILY = ["S2a_quantile", "S2b_convex", "S2b_sat", "S2b_cubic", "S3_nonmono"]  # vs-S1 test family
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


def _shape_fn(xk, shape):
    s, _, _ = _std(np.log(xk))                       # log-head sees ~ s (linear in s)
    if shape == "S1_logfit":
        return s                                     # log-optimal (already standardized)
    if shape == "S2a_quantile":
        g, _, _ = _std(rankdata(xk) / len(xk))       # ~ Phi(s): PLE's own quantile coordinate
    elif shape == "S2b_convex":
        g, _, _ = _std(np.exp(0.7 * s))
    elif shape == "S2b_sat":
        g, _, _ = _std(1.0 / (1.0 + np.exp(-2.0 * s)))
    elif shape == "S2b_cubic":
        g, _, _ = _std(s ** 3)
    elif shape == "S3_nonmono":
        g, _, _ = _std(s ** 2)
    else:
        raise ValueError(shape)
    return g


def make_data(shape, n, rng):
    latent = rng.normal(LOC, SCALE, size=(n, K))
    X = np.exp(latent)
    G = np.column_stack([_shape_fn(X[:, k], shape) for k in range(K)])
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
    if head == "linear":
        m = LogisticRegressionCV(Cs=np.logspace(-4, 2, 7), cv=3, max_iter=4000, scoring="average_precision")
    else:
        m = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=12, max_iter=400, random_state=seed)
    m.fit(Xtr, ytr)
    return average_precision_score(yte, m.predict_proba(Xte)[:, 1])


ap = {(s, h, e): [] for s in SHAPES for h in HEADS for e in ENCS}
oracle = {s: [] for s in SHAPES}
for seed in SEEDS:
    rng = np.random.default_rng(2000 + seed)
    for shape in SHAPES:
        Xtr, ytr, _ = make_data(shape, N, rng)
        Xte, yte, olog = make_data(shape, N, rng)
        oracle[shape].append(average_precision_score(yte, olog))
        edges = [ple_edges(Xtr[:, k], N_BINS) for k in range(K)]
        for enc in ENCS:
            Etr, Ete = encode(enc, Xtr, Xte, edges)
            for head in HEADS:
                ap[(shape, head, enc)].append(fit_score(head, Etr, ytr, Ete, yte, seed))


def paired_ci(d):
    d = np.asarray(d); n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    h = tdist.ppf(0.975, n - 1) * se
    p = 2 * tdist.sf(abs(mean / (se + 1e-12)), n - 1)
    return mean, mean - h, mean + h, p


def diff(shape, head, e1, e2):
    return np.array(ap[(shape, head, e1)]) - np.array(ap[(shape, head, e2)])


print(f"=== base rate ~{BASE_RATE}; N={N}, K={K}, {len(SEEDS)} seeds, {N_BINS} bins ===")
print("=== precondition oracle PR-AUC ===")
for s in SHAPES:
    print(f"  {s:14s} {np.mean(oracle[s]):.3f}")

print("\n=== raw ple-log (linear head) ===")
for s in SHAPES:
    mean, lo, hi, _ = paired_ci(diff(s, "linear", "ple", "log"))
    print(f"  {s:14s} {mean:+.3f} [{lo:+.3f},{hi:+.3f}]")

# DEFICIT-CORRECTED: Delta = (ple-log)_shape - (ple-log)_S1, paired per seed, Holm over FAMILY
d_s1 = diff("S1_logfit", "linear", "ple", "log")
deltas = {s: diff(s, "linear", "ple", "log") - d_s1 for s in FAMILY}
stats = {s: paired_ci(deltas[s]) for s in FAMILY}
order = sorted(FAMILY, key=lambda s: stats[s][3])
holm = {s: min(stats[s][3] * (len(FAMILY) - i), 1.0) for i, s in enumerate(order)}

print("\n=== DEFICIT-CORRECTED curvature benefit: Delta = (ple-log)_shape - (ple-log)_S1 ===")
print(f"S1 structural deficit (ple-log on log-optimal signal): {d_s1.mean():+.3f}")
print(f"{'shape':14s} {'Delta':>7s} {'95% CI':>18s} {'Holm p':>8s}  verdict")
for s in FAMILY:
    mean, lo, hi, _ = stats[s]
    v = "REAL benefit" if lo > EQUIV and holm[s] < 0.05 else ("no benefit" if hi < EQUIV else "marginal")
    tag = " <- quantile (basis-aligned)" if s == "S2a_quantile" else (" <- non-monotone" if s == "S3_nonmono" else "")
    print(f"{s:14s} {mean:+7.3f} [{lo:+.3f},{hi:+.3f}] {holm[s]:8.3f}  {v}{tag}")

print("\n=== KEY: do the GENUINE log-mismatched monotone curves show a real benefit? ===")
gen = [s for s in GENUINE if stats[s][1] > EQUIV and holm[s] < 0.05]
print(f"  genuine curves with CI-clear benefit above deficit: {gen if gen else 'NONE'}")
print(f"  -> {'H_curv SUPPORTED (curvature is a general lever)' if len(gen) >= 2 else 'H_curv WEAK/basis-dependent'}")

fig, ax = plt.subplots(figsize=(8, 4.4))
means = [stats[s][0] for s in FAMILY]
errs = [stats[s][0] - stats[s][1] for s in FAMILY]
colors = ["#ee6677" if s == "S2a_quantile" else ("#228833" if s == "S3_nonmono" else "#4477aa") for s in FAMILY]
ax.bar(range(len(FAMILY)), means, yerr=[errs, errs], capsize=4, color=colors)
ax.axhline(0, color="k", lw=0.8); ax.axhline(EQUIV, color="gray", ls=":", lw=0.8)
ax.set_xticks(range(len(FAMILY))); ax.set_xticklabels(FAMILY, rotation=20, ha="right")
ax.set_ylabel("Δ = (ple−log)_shape − (ple−log)_S1")
ax.set_title("Cycle 7 — curvature benefit above PLE's structural deficit (affine head, 95% CI)")
plt.tight_layout(); plt.savefig("finding_curvature_above_deficit.png", dpi=110)
print("\nsaved finding_curvature_above_deficit.png")
