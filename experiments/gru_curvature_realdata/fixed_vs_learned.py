# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "scipy", "matplotlib"]
# ///
"""
OPEN QUESTION C — fixed PLE vs a learned per-feature embed, done right.

The single-feature proj_vs_ple ordering claims (ple beats mlp on sharp by +0.028) did NOT survive review:
n=5 CIs barely clearing 0, no multiplicity correction, and the two encoders compared in DIFFERENT coordinates
(PLE on raw count, mlp on log). This rerun fixes all three:
  * MULTIVARIATE (K=6) — the regime where the lever is actually visible.
  * SHARED COORDINATE — both PLE and the learned embed operate on the SAME standardized-log feature sc.
  * 8 seeds + Holm correction across the reported contrasts.
Arms (same GRU backbone, affine read), same expand dim d=BINS per feature:
  log : K log-scalars
  ple : per-feature fixed quantile ramps on sc            (fixed knots)
  mlp : per-feature Linear(1->d)->ReLU on sc              (learned knots, SAME coordinate)
Conditions: curved (cubic) + logadq (deficit baseline). deficit-corrected estimand. Static positive control.
Resolves: does fixed PLE beat the learned embed (or vice-versa) on multivariate monotone curvature?
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
K, N_TRAIN, N_VAL, N_TEST = 6, 4000, 2000, 4000
L, D = 32, 12
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6
DECAY = 0.9
SEEDS = list(range(8))
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()
CONDS = ["curved", "logadq"]


def _std(v, axis=None):
    m = v.mean(axis=axis, keepdims=axis is not None)
    s = v.std(axis=axis, keepdims=axis is not None) + 1e-12
    return (v - m) / s


def make(cond, n, seed):
    rng = np.random.default_rng(seed)
    feats = np.exp(rng.normal(0.0, 1.2, size=(n, L, K)))
    sc = _std(np.log(feats), axis=(0, 1))
    logit = np.zeros(n)
    for j in range(K):
        g = _std(sc[..., j] ** 3 if cond == "curved" else sc[..., j])
        logit = logit + _std((g * W[None, :]).sum(1))
    b = -np.quantile(logit, 0.93)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return sc.astype(np.float32), y


def edges(sc):
    E = []
    for j in range(K):
        e = np.quantile(sc[..., j], np.linspace(0, 1, D + 1))
        for t in range(1, e.size):
            if e[t] <= e[t - 1]:
                e[t] = e[t - 1] + 1e-9
        E.append(e)
    return E


def ple_enc(sc, E):
    cols = []
    for j in range(K):
        e = E[j]; x = sc[..., j]
        r = np.empty(sc.shape[:2] + (D,), dtype=np.float32)
        for t in range(D):
            r[..., t] = np.clip((x - e[t]) / (e[t + 1] - e[t]), 0.0, 1.0)
        cols.append(r)
    return np.concatenate(cols, axis=-1).astype(np.float32)


class GRUClf(nn.Module):
    """arm 'mlp' embeds each of K scalar channels via its own Linear(1->D)->ReLU (shared coordinate=sc)."""
    def __init__(self, arm, in_dim):
        super().__init__()
        self.arm = arm
        if arm == "mlp":
            self.emb = nn.ModuleList([nn.Sequential(nn.Linear(1, D), nn.ReLU()) for _ in range(K)])
            gin = K * D
        else:
            gin = in_dim
        self.gru = nn.GRU(gin, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        if self.arm == "mlp":
            x = torch.cat([self.emb[j](x[..., j:j + 1]) for j in range(K)], dim=-1)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gscore(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed):
    Xtr, Xva, Xte = _feat(arm, sc_tr, E), _feat(arm, sc_va, E), _feat(arm, sc_te, E)
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUClf(arm, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    best, bs, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        m.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        a = average_precision_score(yva, gscore(m, Xva))
        if a > best + 1e-5:
            best, bad, bs = a, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if bs is not None:
        m.load_state_dict(bs)
    return average_precision_score(yte, gscore(m, Xte))


def _feat(arm, sc, E):
    return sc.astype(np.float32) if arm in ("log", "mlp") else ple_enc(sc, E)


def static_ap(arm, sc_tr, ytr, sc_te, yte, E):              # static uses log or ple (mlp needs a net)
    def pool(sc):
        return (_feat(arm, sc, E) * W[None, :, None]).sum(1)
    m = LogisticRegression(max_iter=3000).fit(pool(sc_tr), ytr)
    return average_precision_score(yte, m.predict_proba(pool(sc_te))[:, 1])


AP = {}
for cond in CONDS:
    for si, seed in enumerate(SEEDS):
        sc_tr, ytr = make(cond, N_TRAIN, 100 + seed)
        sc_va, yva = make(cond, N_VAL, 300 + seed)
        sc_te, yte = make(cond, N_TEST, 500 + seed)
        E = edges(sc_tr)
        for arm in ("log", "ple", "mlp"):
            AP[("gru", cond, arm, si)] = gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed)
        for arm in ("log", "ple"):
            AP[("static", cond, arm, si)] = static_ap(arm, sc_tr, ytr, sc_te, yte, E)


def paired(vals):
    v = np.asarray(vals, float)
    m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    t = m / (se + 1e-12)
    p = 2 * stats.t.sf(abs(t), df=len(v) - 1)
    tc = stats.t.ppf(0.975, df=len(v) - 1)
    return m, m - tc * se, m + tc * se, p


def dcorr(arch, arm):                                       # deficit-corrected (arm-log) vs logadq
    return [(AP[(arch, "curved", arm, si)] - AP[(arch, "curved", "log", si)]) -
            (AP[(arch, "logadq", arm, si)] - AP[(arch, "logadq", "log", si)]) for si in range(len(SEEDS))]


def holm(pairs):                                            # [(name,p)] -> adjusted significance at .05
    order = sorted(pairs, key=lambda x: x[1])
    out, m = {}, len(pairs)
    for i, (name, p) in enumerate(order):
        out[name] = p * (m - i)
    return out


print("=" * 88)
print("FIXED PLE vs LEARNED EMBED — multivariate K=6, shared coordinate (sc), 8 seeds, deficit-corrected")
contrasts = {
    "static ple (control)": paired(dcorr("static", "ple")),
    "gru ple - log": paired([AP[("gru", "curved", "ple", si)] - AP[("gru", "curved", "log", si)] -
                             (AP[("gru", "logadq", "ple", si)] - AP[("gru", "logadq", "log", si)])
                             for si in range(len(SEEDS))]),
    "gru mlp - log": paired([AP[("gru", "curved", "mlp", si)] - AP[("gru", "curved", "log", si)] -
                             (AP[("gru", "logadq", "mlp", si)] - AP[("gru", "logadq", "log", si)])
                             for si in range(len(SEEDS))]),
    "gru ple - mlp": paired([AP[("gru", "curved", "ple", si)] - AP[("gru", "curved", "mlp", si)]
                             for si in range(len(SEEDS))]),
}
hp = holm([(k, v[3]) for k, v in contrasts.items()])
print(f"{'contrast':>22} {'estimate':>22} {'raw p':>9} {'Holm p':>9} {'sig .05?':>9}")
for k, v in contrasts.items():
    print(f"{k:>22} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] {v[3]:9.3f} {hp[k]:9.3f} {'YES' if hp[k] < 0.05 else 'no':>9}")

pm = contrasts["gru ple - mlp"]
print("\n" + "=" * 88)
if hp["static ple (control)"] < 0.05:
    print("Positive control fires (static ple deficit-corrected sig).")
    if hp["gru ple - mlp"] < 0.05 and pm[0] > 0:
        print(f"RESOLVED: fixed PLE BEATS the learned embed on multivariate curvature (ple-mlp {pm[0]:+.3f}, Holm-sig).")
    elif hp["gru ple - mlp"] < 0.05 and pm[0] < 0:
        print(f"RESOLVED: learned embed BEATS fixed PLE (ple-mlp {pm[0]:+.3f}, Holm-sig).")
    else:
        print(f"RESOLVED: fixed PLE and learned embed TIE on multivariate curvature (ple-mlp {pm[0]:+.3f}, "
              f"Holm ns) -> pick on cost/robustness, not accuracy.")
else:
    print("Positive control did NOT fire -> instrument blind at 8 seeds; inconclusive.")

fig, ax = plt.subplots(figsize=(7.4, 4))
ks = list(contrasts.keys())
ax.bar(range(len(ks)), [contrasts[k][0] for k in ks],
       yerr=[[contrasts[k][0] - contrasts[k][1] for k in ks], [contrasts[k][2] - contrasts[k][0] for k in ks]],
       capsize=5, color=["#2ca02c", "#4477aa", "#d62728", "#888"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("deficit-corrected benefit"); ax.set_title("Fixed PLE vs learned embed (multivariate, shared coordinate, 8 seeds)")
plt.tight_layout(); plt.savefig("fixed_vs_learned.png", dpi=120)
print("\nsaved fixed_vs_learned.png")
