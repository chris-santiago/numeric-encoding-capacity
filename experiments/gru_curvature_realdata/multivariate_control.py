# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 8 — MULTIVARIATE reproduction (the correct experiment).

The positive control (positive_control.py) did NOT fire, and the root cause is a design flaw inherited
from poc2: Cycle 7 established value-curvature is a MULTIVARIATE phenomenon — for a SINGLE curved feature
under a rank metric (PR-AUC) any monotone encoding gives identical rankings, so curvature is INVISIBLE; it
becomes visible only through K additive curved features combining. poc2 and the single-feature positive
control both used ONE curved feature -> curvature was invisible by construction, regardless of architecture.

This reproduces Cycle 7's K=6 additive condition inside the sequence setting, and runs the IDENTICAL
deficit-corrected estimand in TWO architectures to make the null interpretable:

  Conditions (each: K=6 per-step features, iid, recency-aggregated, additive in the logit; equal weights):
    curved : risk_j = cubic(sc_j)   (monotone, log-mismatched)  -> the curvature lever under test
    nonmono: risk_j = sc_j**2       (non-monotone)              -> strong lever; pipeline sanity (Cycle 6/7)
    logadq : risk_j = sc_j          (log-linear)                -> deficit baseline (log is optimal)

  Arms: log (all 6 features log-scalar) | ple (all 6 PLE-encoded, in LOG space).
  deficit-corrected benefit(cond) = (ple-log)_cond - (ple-log)_logadq   [nets PLE's structural deficit].

  Architectures:
    static : recency-POOL the per-step encoding -> logistic (affine-read; NO per-step nonlinearity = the
             Cycle-7 class where curvature is a KNOWN lever). Positive control MUST fire here.
    gru    : Cycle-8 affine-input GRU (gates = a free per-step nonlinearity).

READ:
  * static curved deficit-corrected CI>0  => estimand HAS power (Cycle 7 reproduced) -> null is interpretable
  * then gru curved ~0 while gru nonmono >0 => the GRU absorbs monotone curvature but not non-monotonicity
    -> the (now controlled) refutation holds. gru curved >0 => curvature IS a GRU lever after all.
  * static curved ~0 even here => curvature truly needs richer structure; escalate K / revisit metric.

Confound fixes retained: equal feature weights; PLE fit in log space. PoC scale (L=32, 5 seeds).
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
K = 6
N_TRAIN, N_VAL, N_TEST = 4000, 2000, 4000
L, N_BINS = 32, 12
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6
DECAY = 0.9
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776
CONDS = ["curved", "nonmono", "logadq"]
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()


def _std(v, axis=None):
    m = v.mean(axis=axis, keepdims=axis is not None)
    s = v.std(axis=axis, keepdims=axis is not None) + 1e-12
    return (v - m) / s


def risk(cond, sc):                                          # sc: standardized log feature
    if cond == "curved":
        return sc ** 3
    if cond == "nonmono":
        return sc ** 2
    return sc                                                # logadq


def make_sequences(cond, n, seed):
    rng = np.random.default_rng(seed)
    feats = np.exp(rng.normal(0.0, 1.2, size=(n, L, K)))     # K iid per-step features
    sc = _std(np.log(feats), axis=(0, 1))                    # standardize each feature in log space
    logit = np.zeros(n)
    for j in range(K):
        g = _std(risk(cond, sc[..., j]))                     # per-feature risk contribution
        agg = (g * W[None, :]).sum(axis=1)                   # recency-weighted aggregation
        logit = logit + _std(agg)                            # equal weights, additive
    b = -np.quantile(logit, 0.93)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return sc.astype(np.float32), y, (logit + b)             # sc: (n,L,K) log-standardized features


def fit_edges(sc_tr):                                        # per-feature PLE edges (log space)
    E = []
    for j in range(K):
        e = np.quantile(sc_tr[..., j], np.linspace(0, 1, N_BINS + 1))
        for t in range(1, e.size):
            if e[t] <= e[t - 1]:
                e[t] = e[t - 1] + 1e-9
        E.append(e)
    return E


def encode(arm, sc, E):
    if arm == "log":
        return sc.astype(np.float32)                         # (n,L,K)
    cols = []
    for j in range(K):
        e = E[j]
        r = np.empty(sc.shape[:2] + (N_BINS,), dtype=np.float32)
        x = sc[..., j]
        for t in range(N_BINS):
            r[..., t] = np.clip((x - e[t]) / (e[t + 1] - e[t]), 0.0, 1.0)
        cols.append(r)
    return np.concatenate(cols, axis=-1).astype(np.float32)  # (n,L,K*N_BINS)


def static_ap(arm, sc_tr, ytr, sc_te, yte, E):
    def pool(sc):
        return (encode(arm, sc, E) * W[None, :, None]).sum(axis=1)
    m = LogisticRegression(max_iter=3000).fit(pool(sc_tr), ytr)
    return average_precision_score(yte, m.predict_proba(pool(sc_te))[:, 1])


class GRUClf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gru_score(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed):
    Xtr, Xva, Xte = encode(arm, sc_tr, E), encode(arm, sc_va, E), encode(arm, sc_te, E)
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


# per (arch, cond, arm) -> list of seed APs
AP = {}
for cond in CONDS:
    for seed in SEEDS:
        sc_tr, ytr, _ = make_sequences(cond, N_TRAIN, 100 + seed)
        sc_va, yva, _ = make_sequences(cond, N_VAL, 300 + seed)
        sc_te, yte, _ = make_sequences(cond, N_TEST, 500 + seed)
        E = fit_edges(sc_tr)
        for arm in ("log", "ple"):
            AP.setdefault(("static", cond, arm), []).append(static_ap(arm, sc_tr, ytr, sc_te, yte, E))
            AP.setdefault(("gru", cond, arm), []).append(
                gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed))


def pl(arch, cond):                                          # (ple - log) per seed
    return [AP[(arch, cond, "ple")][i] - AP[(arch, cond, "log")][i] for i in range(len(SEEDS))]


def f(c):
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


print("=" * 92)
print("MULTIVARIATE (K=6) — deficit-corrected benefit = (ple-log)_cond - (ple-log)_logadq ; 5 seeds")
print(f"{'arch':>8} {'cond':>9} {'ple-log':>22} {'deficit-corrected':>22} {'CI>0?':>7}")
DC = {}
for arch in ("static", "gru"):
    for cond in ("curved", "nonmono"):
        raw = paired_ci(pl(arch, cond))
        dc = paired_ci([a - b for a, b in zip(pl(arch, cond), pl(arch, "logadq"))])
        DC[(arch, cond)] = dc
        print(f"{arch:>8} {cond:>9} {f(raw):>22} {f(dc):>22} {'YES' if dc[1] > 0 else 'no':>7}")

print("\n" + "=" * 92)
sc_fires = DC[("static", "curved")][1] > 0
sn_fires = DC[("static", "nonmono")][1] > 0
gc = DC[("gru", "curved")]
gn = DC[("gru", "nonmono")]
print(f"POSITIVE CONTROLS (static, lever KNOWN): curved {f(DC[('static','curved')])} -> "
      f"{'FIRES' if sc_fires else 'does NOT fire'}; nonmono {f(DC[('static','nonmono')])} -> "
      f"{'FIRES' if sn_fires else 'does NOT fire'}")
if sc_fires:
    print("\nEstimand HAS power for monotone curvature (Cycle 7 reproduced in the affine-read class).")
    print(f"GRU curved  : {f(gc)} -> {'GRU absorbs curvature (controlled refutation holds)' if gc[2] <= 0.005 else 'curvature IS a GRU lever' if gc[1] > 0 else 'GRU curved inconclusive'}")
    print(f"GRU nonmono : {f(gn)} -> {'non-monotonicity IS a GRU lever (expected, Cycle 6)' if gn[1] > 0 else 'unexpected: nonmono null in GRU'}")
elif sn_fires:
    print("\nEstimand detects NON-MONOTONE levers but NOT monotone curvature even in the affine-read class.")
    print("=> curvature may need larger K / a magnitude-sensitive metric; monotone-curvature question OPEN.")
else:
    print("\nEstimand fires for NOTHING even where levers are known => pipeline/metric problem; halt & fix.")

fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels = ["static\ncurved", "static\nnonmono", "gru\ncurved", "gru\nnonmono"]
keys = [("static", "curved"), ("static", "nonmono"), ("gru", "curved"), ("gru", "nonmono")]
ms = [DC[k][0] for k in keys]
lo = [DC[k][0] - DC[k][1] for k in keys]
hi = [DC[k][2] - DC[k][0] for k in keys]
ax.bar(range(4), ms, yerr=[lo, hi], capsize=5, color=["#2ca02c", "#2ca02c", "#888", "#888"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(labels)
ax.set_ylabel("deficit-corrected benefit\n(ple-log)_cond - (ple-log)_logadq")
ax.set_title("K=6 multivariate: does the estimand detect curvature / non-monotonicity, by architecture?")
plt.tight_layout(); plt.savefig("multivariate_control.png", dpi=120)
print("\nsaved multivariate_control.png")
