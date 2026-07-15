# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
OPEN QUESTION B — reconcile the GRU non-monotone anomaly.

multivariate_control.py found GRU non-monotone (SMOOTH quadratic sc^2) deficit-corrected +0.035 ns, which
looks inconsistent with Cycle 6's sharp-band +0.19. Hypothesis (from proj_vs_ple): the GRU gates ALREADY
approximate a smooth symmetric non-monotonicity, so PLE adds little there; a SHARP/localized non-monotone
band is what the affine read cannot form -> PLE fires. Test both, multivariate (K=6), static + GRU.

Conditions (per-step, standardized log feature sc):
  smooth : risk_j = sc_j^2                              (broad symmetric non-monotone)
  sharp  : risk_j = exp(-(sc_j)^2 / (2*0.15^2))          (narrow bump = localized non-monotone)
  logadq : risk_j = sc_j                                (deficit baseline)
deficit-corrected benefit = (ple-log)_cond - (ple-log)_logadq. 5 seeds, paired 95% CI. bins=12.
Prediction: static smooth AND sharp fire (affine head can't do either); GRU sharp FIRES, GRU smooth ns.
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
K, N_TRAIN, N_VAL, N_TEST = 6, 4000, 2000, 4000
L, B = 32, 12
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6
DECAY = 0.9
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776
SHARP_SIGMA = 0.15
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()
CONDS = ["smooth", "sharp", "logadq"]


def _std(v, axis=None):
    m = v.mean(axis=axis, keepdims=axis is not None)
    s = v.std(axis=axis, keepdims=axis is not None) + 1e-12
    return (v - m) / s


def risk(cond, sc):
    if cond == "smooth":
        return sc ** 2
    if cond == "sharp":
        return np.exp(-(sc ** 2) / (2 * SHARP_SIGMA ** 2))
    return sc


def make(cond, n, seed):
    rng = np.random.default_rng(seed)
    feats = np.exp(rng.normal(0.0, 1.2, size=(n, L, K)))
    sc = _std(np.log(feats), axis=(0, 1))
    logit = np.zeros(n)
    for j in range(K):
        logit = logit + _std((_std(risk(cond, sc[..., j])) * W[None, :]).sum(1))
    b = -np.quantile(logit, 0.93)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return sc.astype(np.float32), y


def edges(sc):
    E = []
    for j in range(K):
        e = np.quantile(sc[..., j], np.linspace(0, 1, B + 1))
        for t in range(1, e.size):
            if e[t] <= e[t - 1]:
                e[t] = e[t - 1] + 1e-9
        E.append(e)
    return E


def enc(arm, sc, E):
    if arm == "log":
        return sc.astype(np.float32)
    cols = []
    for j in range(K):
        e = E[j]; x = sc[..., j]
        r = np.empty(sc.shape[:2] + (B,), dtype=np.float32)
        for t in range(B):
            r[..., t] = np.clip((x - e[t]) / (e[t + 1] - e[t]), 0.0, 1.0)
        cols.append(r)
    return np.concatenate(cols, axis=-1).astype(np.float32)


class GRUClf(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gscore(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed):
    Xtr, Xva, Xte = enc(arm, sc_tr, E), enc(arm, sc_va, E), enc(arm, sc_te, E)
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUClf(Xtr.shape[2])
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


def static_ap(arm, sc_tr, ytr, sc_te, yte, E):
    def pool(sc):
        return (enc(arm, sc, E) * W[None, :, None]).sum(1)
    m = LogisticRegression(max_iter=3000).fit(pool(sc_tr), ytr)
    return average_precision_score(yte, m.predict_proba(pool(sc_te))[:, 1])


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - T_CRIT * se, m + T_CRIT * se


AP = {}
for cond in CONDS:
    for si, seed in enumerate(SEEDS):
        sc_tr, ytr = make(cond, N_TRAIN, 100 + seed)
        sc_va, yva = make(cond, N_VAL, 300 + seed)
        sc_te, yte = make(cond, N_TEST, 500 + seed)
        E = edges(sc_tr)
        for arm in ("log", "ple"):
            AP[("gru", cond, arm, si)] = gru_ap(arm, sc_tr, ytr, sc_va, yva, sc_te, yte, E, seed)
            AP[("static", cond, arm, si)] = static_ap(arm, sc_tr, ytr, sc_te, yte, E)


def dc(arch, cond):                                          # deficit-corrected vs logadq, seed-paired
    return ci([(AP[(arch, cond, "ple", si)] - AP[(arch, cond, "log", si)]) -
               (AP[(arch, "logadq", "ple", si)] - AP[(arch, "logadq", "log", si)])
               for si in range(len(SEEDS))])


print("=" * 84)
print("SHARP vs SMOOTH non-monotone in a GRU — deficit-corrected benefit, 5 seeds")
print(f"{'arch':>8} {'cond':>8} {'deficit-corrected':>24} {'CI>0?':>7}")
for arch in ("static", "gru"):
    for cond in ("smooth", "sharp"):
        c = dc(arch, cond)
        print(f"{arch:>8} {cond:>8} {c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}] {'YES' if c[1] > 0 else 'no':>7}")

gs, gsh = dc("gru", "smooth"), dc("gru", "sharp")
ss, ssh = dc("static", "smooth"), dc("static", "sharp")
print("\n" + "=" * 84)
print(f"static controls: smooth {'FIRES' if ss[1] > 0 else 'ns'}, sharp {'FIRES' if ssh[1] > 0 else 'ns'}")
if gsh[1] > 0 and gs[1] <= 0:
    print("RESOLVED: GRU absorbs SMOOTH non-monotonicity (gates approximate it) but NOT SHARP -> the")
    print("multivariate_control smooth-quadratic null was a smooth-vs-sharp effect; Cycle 6's sharp band stands.")
elif gsh[1] > 0 and gs[1] > 0:
    print("Both fire in GRU -> the earlier smooth null was noise/deficit, not a smooth-vs-sharp distinction.")
else:
    print(f"GRU sharp {gsh} did not fire -> anomaly NOT explained by sharpness; needs deeper look.")

fig, ax = plt.subplots(figsize=(6.8, 4))
keys = [("static", "smooth"), ("static", "sharp"), ("gru", "smooth"), ("gru", "sharp")]
vals = [dc(a, c) for a, c in keys]
ax.bar(range(4), [v[0] for v in vals],
       yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=5,
       color=["#2ca02c", "#2ca02c", "#888", "#888"])
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(["static\nsmooth", "static\nsharp", "gru\nsmooth", "gru\nsharp"])
ax.set_ylabel("deficit-corrected benefit"); ax.set_title("Sharp vs smooth non-monotone: does the GRU absorb only smooth?")
plt.tight_layout(); plt.savefig("sharp_vs_smooth.png", dpi=120)
print("\nsaved sharp_vs_smooth.png")
