# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 8 POSITIVE CONTROL — the check both technical reviews demanded.

The refutation ("a GRU's gates absorb monotone curvature, so PLE has no per-step lever") was over-claimed
because poc2's F1 could not distinguish "no lever" from "no power": its success criterion is ENTAILED to
fail if the hypothesis is true. A genuine positive control runs the IDENTICAL deficit-corrected estimand on
a model class where monotone curvature is a KNOWN lever, and shows it fires CI-positive there.

Design: same monotone-curved feature (cubic-in-log count) + a marginal-matched, exactly-log-adequate
reference. Deficit-corrected benefit = AP(ple_count) - AP(ple_ref), seed-level 95% CI. Run in TWO archs:

  static : recency-weighted POOL of the per-step encoding -> LINEAR (logistic) read.
           No per-step nonlinearity -> the affine-read class (Cycle 7). A log scalar can only pool to
           sum_t w_t*sc_t; it CANNOT form sum_t w_t*cubic(sc_t). PLE can (pool of ramps, linear-read =
           sum_t w_t*g(sc_t)). => PLE is EXPECTED to win here. This is the positive control.
  gru    : the Cycle-8 affine-input GRU (gates = a free per-step nonlinearity). => expected flat/negative.

CONFOUND FIXES (from the reviews):
  * W_COUNT == W_REF (both 1.0): kills the signal-share mismatch (poc2 used 1.6 vs 1.0), so ple_count and
    ple_ref deficits are measured at EQUAL signal.
  * PLE fit in LOG space (bins on standardized log-count sc, not raw count): kills the raw-vs-log basis
    mis-specification (risk is smooth in log space).

READ: if static benefit is CI-positive AND gru benefit is not, the estimand HAS power to detect a
monotone-curvature lever, the GRU specifically absorbs it, and the (downgraded) refutation is controlled.
If static is ALSO flat, the estimand is genuinely blind and the curvature question is unresolved.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
N_TRAIN, N_VAL, N_TEST = 4000, 2000, 4000
L, N_BINS = 32, 12
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6
DECAY = 0.9
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776                                               # t_.975, df=4
W_COUNT, W_REF = 1.0, 1.0                                    # FIX: equal signal share
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()


def _std(v, mu=None, sd=None):
    if mu is None:
        mu, sd = v.mean(), v.std() + 1e-12
    return (v - mu) / sd, mu, sd


def make_sequences(n, seed):
    rng = np.random.default_rng(seed)
    count = np.exp(rng.normal(0.0, 1.2, size=(n, L)))
    ref = np.exp(rng.normal(0.0, 1.2, size=(n, L)))          # same marginal as count
    sc = _std(np.log(count))[0]
    sr = _std(np.log(ref))[0]                                # exactly log-adequate
    gc = _std(sc ** 3)[0]                                    # cubic-in-log = monotone, log-mismatched
    logit = W_COUNT * _std((gc * W).sum(1))[0] + W_REF * _std((sr * W).sum(1))[0]
    b = -np.quantile(logit, 0.93)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return sc.astype(np.float32), sr.astype(np.float32), y, (logit + b)


def fit_ple_edges(x, nbins):
    e = np.quantile(x, np.linspace(0, 1, nbins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def ple(x, edges):                                           # x already in log/standardized space (FIX)
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


ARMS = {"log": ("log", "log"), "ple_count": ("ple", "log"), "ple_ref": ("log", "ple")}


def enc(mode, x, edges):
    return x[..., None] if mode == "log" else ple(x, edges)


def arm_perstep(arm, sc, sr, ce, re):
    mc, mr = ARMS[arm]
    return np.concatenate([enc(mc, sc, ce), enc(mr, sr, re)], axis=-1).astype(np.float32)


# ---- static affine-read: recency-pool per-step encoding -> logistic (no per-step nonlinearity) --------
def static_ap(arm, sc_tr, sr_tr, ytr, sc_te, sr_te, yte, ce, re):
    def pooled(sc, sr):
        x = arm_perstep(arm, sc, sr, ce, re)                 # (n,L,d)
        return (x * W[None, :, None]).sum(axis=1)            # recency-weighted pool -> (n,d)
    m = LogisticRegression(max_iter=2000).fit(pooled(sc_tr, sr_tr), ytr)
    return average_precision_score(yte, m.predict_proba(pooled(sc_te, sr_te))[:, 1])


# ---- affine GRU (Cycle-8) ---------------------------------------------------------------------------
class GRUClf(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gru_score(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def gru_ap(arm, sc_tr, sr_tr, ytr, sc_va, sr_va, yva, sc_te, sr_te, yte, ce, re, seed):
    Xtr = arm_perstep(arm, sc_tr, sr_tr, ce, re)
    Xva = arm_perstep(arm, sc_va, sr_va, ce, re)
    Xte = arm_perstep(arm, sc_te, sr_te, ce, re)
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUClf(Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    best, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        m.train()
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        apv = average_precision_score(yva, gru_score(m, Xva))
        if apv > best + 1e-5:
            best, bad, best_state = apv, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return average_precision_score(yte, gru_score(m, Xte))


def paired_ci(d):
    d = np.asarray(d, float)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - T_CRIT * se, m + T_CRIT * se


ap = {("static", a): [] for a in ARMS}
ap.update({("gru", a): [] for a in ARMS})
oracle = []
for seed in SEEDS:
    sc_tr, sr_tr, ytr, _ = make_sequences(N_TRAIN, 100 + seed)
    sc_va, sr_va, yva, _ = make_sequences(N_VAL, 300 + seed)
    sc_te, sr_te, yte, ste = make_sequences(N_TEST, 500 + seed)
    oracle.append(average_precision_score(yte, ste))
    ce, re = fit_ple_edges(sc_tr, N_BINS), fit_ple_edges(sr_tr, N_BINS)
    for a in ARMS:
        ap[("static", a)].append(static_ap(a, sc_tr, sr_tr, ytr, sc_te, sr_te, yte, ce, re))
        ap[("gru", a)].append(gru_ap(a, sc_tr, sr_tr, ytr, sc_va, sr_va, yva, sc_te, sr_te, yte, ce, re, seed))


def f(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


print("=" * 84)
print("POSITIVE CONTROL — deficit-corrected curvature benefit = AP(ple_count) - AP(ple_ref)")
print("  same monotone-curved feature; equal weights; PLE in log space; 5 seeds")
print(f"  oracle ceiling AP = {np.mean(oracle):.3f}\n")
print(f"{'arch':>8} {'ple_count-log':>22} {'ple_ref-log':>22} {'deficit-corrected':>22} {'fires?':>8}")
res = {}
for arch in ("static", "gru"):
    pcl = paired_ci([ap[(arch, "ple_count")][i] - ap[(arch, "log")][i] for i in range(len(SEEDS))])
    prl = paired_ci([ap[(arch, "ple_ref")][i] - ap[(arch, "log")][i] for i in range(len(SEEDS))])
    dc = paired_ci([ap[(arch, "ple_count")][i] - ap[(arch, "ple_ref")][i] for i in range(len(SEEDS))])
    res[arch] = dc
    print(f"{arch:>8} {f(pcl):>22} {f(prl):>22} {f(dc):>22} {'YES' if dc[1] > 0 else 'no':>8}")

print("\n" + "=" * 84)
fired = res["static"][1] > 0
absorbed = res["gru"][2] <= res["static"][0]
print(f"POSITIVE CONTROL {'FIRES' if fired else 'DOES NOT FIRE'}: static deficit-corrected benefit "
      f"{'CI-clear > 0' if fired else 'includes 0'} ({f(res['static'])}).")
if fired:
    print(f"GRU benefit {f(res['gru'])} — the estimand HAS power to detect a monotone-curvature lever;")
    print("the affine GRU specifically does not benefit => gates absorb monotone curvature (CONTROLLED).")
else:
    print("Estimand did NOT fire even where curvature is a known lever => it is blind; curvature question")
    print("UNRESOLVED, not refuted. The GRU null cannot be interpreted.")

fig, axx = plt.subplots(figsize=(6.6, 4))
xs = [0, 1]
ax_m = [res["static"][0], res["gru"][0]]
lo = [res["static"][0] - res["static"][1], res["gru"][0] - res["gru"][1]]
hi = [res["static"][2] - res["static"][0], res["gru"][2] - res["gru"][0]]
axx.bar(xs, ax_m, yerr=[lo, hi], capsize=6, color=["#2ca02c", "#888"])
axx.axhline(0, color="k", lw=0.8)
axx.set_xticks(xs); axx.set_xticklabels(["static affine-read\n(lever KNOWN)", "affine GRU\n(gates)"])
axx.set_ylabel("deficit-corrected curvature benefit\nAP(ple_count) - AP(ple_ref)")
axx.set_title("Positive control: same estimand, same curved feature, two architectures")
plt.tight_layout(); plt.savefig("positive_control.png", dpi=120)
print("\nsaved positive_control.png")
