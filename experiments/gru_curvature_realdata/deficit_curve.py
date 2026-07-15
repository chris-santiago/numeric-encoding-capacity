# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
OPEN QUESTION A — where does the curvature lever clear PLE's GRU dimensionality deficit?

Cycle 8 (corrected) showed monotone curvature is a real GRU lever (deficit-corrected +0.143) but that RAW
ple-log ~= 0 because PLE's per-step deficit in the recurrence is large (~-0.13 for K=6 x 12 bins). The
deployment-decisive question: at what feature count K and bin count B does the summed lever exceed the
summed deficit, i.e. RAW ple-log > 0? PLE's cost grows with K*B; the lever grows with K (more multivariate
curvature). So few bins should be the winning regime.

Design: K curved-in-log features (cubic), additive risk, recency-aggregated; log-adequate twin condition to
net the deficit. Affine GRU (Cycle-6 regime). Two 1-D sweeps + a static positive control at the reference:
  bins in {2,4,8,16} at K=6 ; K in {2,4,8} at bins=8.
Report RAW ple-log (deployment number) and deficit-corrected (lever) per config; find the raw>0 crossing.
5 seeds, paired 95% CI. The log arm is bins-independent -> trained once per (K,cond,seed) and reused.
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
L = 32
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6
DECAY = 0.9
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()
BINS_SWEEP = [(6, 2), (6, 4), (6, 8), (6, 16)]              # (K, bins) vary bins at K=6
K_SWEEP = [(2, 8), (4, 8), (8, 8)]                          # vary K at bins=8
CONFIGS = BINS_SWEEP + K_SWEEP


def _std(v, axis=None):
    m = v.mean(axis=axis, keepdims=axis is not None)
    s = v.std(axis=axis, keepdims=axis is not None) + 1e-12
    return (v - m) / s


def make(cond, K, n, seed):
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


def edges(sc, K, B):
    E = []
    for j in range(K):
        e = np.quantile(sc[..., j], np.linspace(0, 1, B + 1))
        for t in range(1, e.size):
            if e[t] <= e[t - 1]:
                e[t] = e[t - 1] + 1e-9
        E.append(e)
    return E


def enc(arm, sc, E, K, B):
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


def gru_ap(Xtr, ytr, Xva, yva, Xte, yte, seed):
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


def static_ap(arm, sc_tr, ytr, sc_te, yte, E, K, B):
    def pool(sc):
        return (enc(arm, sc, E, K, B) * W[None, :, None]).sum(1)
    m = LogisticRegression(max_iter=3000).fit(pool(sc_tr), ytr)
    return average_precision_score(yte, m.predict_proba(pool(sc_te))[:, 1])


def ci(d):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - T_CRIT * se, m + T_CRIT * se


# cache log APs by (K, cond, seed); ple APs by (K, B, cond, seed)
logap, pleap, statap = {}, {}, {}
Ks = sorted({K for K, _ in CONFIGS})
for K in Ks:
    for cond in ("curved", "logadq"):
        for si, seed in enumerate(SEEDS):
            sc_tr, ytr = make(cond, K, N_TRAIN, 100 + seed)
            sc_va, yva = make(cond, K, N_VAL, 300 + seed)
            sc_te, yte = make(cond, K, N_TEST, 500 + seed)
            logap[(K, cond, si)] = gru_ap(enc("log", sc_tr, None, K, 0), ytr,
                                          enc("log", sc_va, None, K, 0), yva,
                                          enc("log", sc_te, None, K, 0), yte, seed)
            for (KK, B) in CONFIGS:
                if KK != K:
                    continue
                E = edges(sc_tr, K, B)
                pleap[(K, B, cond, si)] = gru_ap(enc("ple", sc_tr, E, K, B), ytr,
                                                 enc("ple", sc_va, E, K, B), yva,
                                                 enc("ple", sc_te, E, K, B), yte, seed)
                if (K, B) == (6, 8):                        # static positive control at reference config
                    statap[(cond, si)] = static_ap("ple", sc_tr, ytr, sc_te, yte, E, K, B) - \
                        static_ap("log", sc_tr, ytr, sc_te, yte, E, K, B)

print("=" * 88)
print("DEFICIT CURVE — RAW ple-log (deployment) and deficit-corrected (lever), affine GRU, 5 seeds")
print(f"{'K':>3} {'bins':>5} {'RAW ple-log':>22} {'deficit-corrected':>22} {'raw>0?':>7}")
rows = []
for (K, B) in CONFIGS:
    raw = ci([pleap[(K, B, "curved", si)] - logap[(K, "curved", si)] for si in range(len(SEEDS))])
    dc = ci([(pleap[(K, B, "curved", si)] - logap[(K, "curved", si)]) -
             (pleap[(K, B, "logadq", si)] - logap[(K, "logadq", si)]) for si in range(len(SEEDS))])
    rows.append((K, B, raw, dc))
    print(f"{K:>3} {B:>5} {raw[0]:+.3f} [{raw[1]:+.3f},{raw[2]:+.3f}] "
          f"{dc[0]:+.3f} [{dc[1]:+.3f},{dc[2]:+.3f}] {'YES' if raw[1] > 0 else 'no':>7}")

sc_ctrl = ci([statap[("curved", si)] for si in range(len(SEEDS))])
print(f"\nstatic positive control (K=6,bins=8) curved ple-log: {sc_ctrl[0]:+.3f} "
      f"[{sc_ctrl[1]:+.3f},{sc_ctrl[2]:+.3f}] -> {'FIRES' if sc_ctrl[1] > 0 else 'blind'}")

print("\nReadout: deficit-corrected (lever) should stay ~positive across configs; RAW crosses 0 as bins")
print("shrink / K grows. The winning deployment regime is where RAW ple-log CI clears 0.")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
bp = [r for r in rows if (r[0], r[1]) in BINS_SWEEP]
a1.errorbar([r[1] for r in bp], [r[2][0] for r in bp],
            yerr=[[r[2][0] - r[2][1] for r in bp], [r[2][2] - r[2][0] for r in bp]],
            marker="o", capsize=3, color="#1f77b4", label="RAW ple-log")
a1.errorbar([r[1] for r in bp], [r[3][0] for r in bp],
            yerr=[[r[3][0] - r[3][1] for r in bp], [r[3][2] - r[3][0] for r in bp]],
            marker="s", capsize=3, color="#2ca02c", label="deficit-corrected")
a1.axhline(0, color="k", lw=0.8, ls=":"); a1.set_xlabel("bins (K=6)"); a1.set_ylabel("PR-AUC benefit"); a1.legend(fontsize=8); a1.set_title("vs bins")
kp = [r for r in rows if (r[0], r[1]) in K_SWEEP]
a2.errorbar([r[0] for r in kp], [r[2][0] for r in kp],
            yerr=[[r[2][0] - r[2][1] for r in kp], [r[2][2] - r[2][0] for r in kp]],
            marker="o", capsize=3, color="#1f77b4", label="RAW ple-log")
a2.errorbar([r[0] for r in kp], [r[3][0] for r in kp],
            yerr=[[r[3][0] - r[3][1] for r in kp], [r[3][2] - r[3][0] for r in kp]],
            marker="s", capsize=3, color="#2ca02c", label="deficit-corrected")
a2.axhline(0, color="k", lw=0.8, ls=":"); a2.set_xlabel("K (bins=8)"); a2.legend(fontsize=8); a2.set_title("vs K")
fig.suptitle("Deficit curve: does the curvature lever clear PLE's GRU dimensionality deficit?")
fig.tight_layout(); fig.savefig("deficit_curve.png", dpi=120)
print("\nsaved deficit_curve.png")
