# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "scipy", "matplotlib"]
# ///
"""
Which embedding is MOST ROBUST for NON-MONOTONIC (sharp-band) features in an affine GRU?

We settled fixed-PLE vs learned-embed on monotone curvature (learned wins). Non-monotone was not directly
compared. The robustness discriminator is BAND LOCATION: PLE's knots sit at data quantiles (dense at the
mode), a learned Linear(1->d)->ReLU places knots adaptively. Test a sharp band AT the mode (s0=0) vs OFF
the mode (s0=1.5, in the sparse tail) -> if PLE degrades off-mode while the learned embed holds, the learned
embed is the more robust choice; if PLE holds via dense-enough tails, they tie.

K=6 multivariate, shared coordinate (both encoders on standardized-log sc), 8 seeds, Holm. Arms log/ple/mlp;
logadq twin for deficit-correction. Conditions: sharp@mode, sharp@off. Static positive control per condition.
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
SIGMA = 0.15
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()
# conditions: (label, s0) ; logadq shares s0=None
CONDS = [("mode", 0.0), ("off", 1.5)]


def _std(v, axis=None):
    m = v.mean(axis=axis, keepdims=axis is not None)
    s = v.std(axis=axis, keepdims=axis is not None) + 1e-12
    return (v - m) / s


def make(kind, s0, n, seed):
    rng = np.random.default_rng(seed)
    feats = np.exp(rng.normal(0.0, 1.2, size=(n, L, K)))
    sc = _std(np.log(feats), axis=(0, 1))
    logit = np.zeros(n)
    for j in range(K):
        r = sc[..., j] if kind == "logadq" else np.exp(-((sc[..., j] - s0) ** 2) / (2 * SIGMA ** 2))
        logit = logit + _std((_std(r) * W[None, :]).sum(1))
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


def feat(arm, sc, E):
    return sc.astype(np.float32) if arm in ("log", "mlp") else ple_enc(sc, E)


class GRUClf(nn.Module):
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


def gru_ap(arm, tr, va, te, E, seed):
    Xtr, Xva, Xte = feat(arm, tr[0], E), feat(arm, va[0], E), feat(arm, te[0], E)
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUClf(arm, Xtr.shape[2])
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(tr[1])
    best, bs, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        m.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(tr[1]), generator=g)
        for i in range(0, len(tr[1]), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(m(Xt[idx]), yt[idx]).backward(); opt.step()
        a = average_precision_score(va[1], gscore(m, Xva))
        if a > best + 1e-5:
            best, bad, bs = a, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if bs is not None:
        m.load_state_dict(bs)
    return average_precision_score(te[1], gscore(m, Xte))


def static_ap(arm, tr, te, E):
    def pool(sc):
        return (feat(arm, sc, E) * W[None, :, None]).sum(1)
    m = LogisticRegression(max_iter=3000).fit(pool(tr[0]), tr[1])
    return average_precision_score(te[1], m.predict_proba(pool(te[0]))[:, 1])


def paired(vals):
    v = np.asarray(vals, float); m = v.mean(); se = v.std(ddof=1) / np.sqrt(len(v))
    p = 2 * stats.t.sf(abs(m / (se + 1e-12)), df=len(v) - 1)
    tc = stats.t.ppf(0.975, df=len(v) - 1)
    return m, m - tc * se, m + tc * se, p


def holm(pairs):
    order = sorted(pairs, key=lambda x: x[1]); out = {}; m = len(pairs)
    for i, (name, p) in enumerate(order):
        out[name] = min(1.0, p * (m - i))
    return out


AP = {}
for (lab, s0) in CONDS:
    for kind in (lab, "logadq"):
        s = s0 if kind == lab else None
        for si, seed in enumerate(SEEDS):
            tr = make(kind, s, N_TRAIN, 100 + seed)
            va = make(kind, s, N_VAL, 300 + seed)
            te = make(kind, s, N_TEST, 500 + seed)
            E = edges(tr[0])
            for arm in ("log", "ple", "mlp"):
                AP[(lab, kind, arm, si)] = gru_ap(arm, tr, va, te, E, seed)
            AP[(lab, "static_ple", kind, si)] = static_ap("ple", tr, te, E) - static_ap("log", tr, te, E)


def dc(lab, arm):                                            # deficit-corrected arm-log for this band loc
    return [(AP[(lab, lab, arm, si)] - AP[(lab, lab, "log", si)]) -
            (AP[(lab, "logadq", arm, si)] - AP[(lab, "logadq", "log", si)]) for si in range(len(SEEDS))]


def raw_ple_mlp(lab):
    return [AP[(lab, lab, "ple", si)] - AP[(lab, lab, "mlp", si)] for si in range(len(SEEDS))]


print("=" * 92)
print("ROBUST ENCODER for NON-MONOTONE (sharp band) features — band AT mode vs OFF mode, 8 seeds, Holm")
rows = {}
for (lab, s0) in CONDS:
    rows[(lab, "ple-log")] = paired(dc(lab, "ple"))
    rows[(lab, "mlp-log")] = paired(dc(lab, "mlp"))
    rows[(lab, "ple-mlp(raw)")] = paired(raw_ple_mlp(lab))
    rows[(lab, "static_ple(ctrl)")] = paired([AP[(lab, "static_ple", lab, si)] for si in range(len(SEEDS))])
hp = holm([(k, v[3]) for k, v in rows.items()])
print(f"{'band':>6} {'contrast':>18} {'estimate':>22} {'Holm p':>9} {'sig?':>5}")
for (lab, s0) in CONDS:
    for c in ("ple-log", "mlp-log", "ple-mlp(raw)", "static_ple(ctrl)"):
        v = rows[(lab, c)]
        print(f"{lab:>6} {c:>18} {v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}] {hp[(lab, c)]:9.3f} {'YES' if hp[(lab, c)] < 0.05 else 'no':>5}")

print("\n" + "=" * 92)
for (lab, s0) in CONDS:
    pm = rows[(lab, "ple-mlp(raw)")]
    win = "PLE" if pm[0] > 0 and hp[(lab, "ple-mlp(raw)")] < 0.05 else \
          "learned embed" if pm[0] < 0 and hp[(lab, "ple-mlp(raw)")] < 0.05 else "tie"
    print(f"band @ {lab:>4} (s0={s0}): ple-log {rows[(lab,'ple-log')][0]:+.3f}, mlp-log {rows[(lab,'mlp-log')][0]:+.3f}"
          f" -> winner: {win} (raw ple-mlp {pm[0]:+.3f})")

fig, ax = plt.subplots(figsize=(7.6, 4.2))
labs = [f"{lab}\nple-log" for lab, _ in CONDS] + [f"{lab}\nmlp-log" for lab, _ in CONDS]
vals = [rows[(lab, "ple-log")] for lab, _ in CONDS] + [rows[(lab, "mlp-log")] for lab, _ in CONDS]
cols = ["#4477aa", "#4477aa", "#d62728", "#d62728"]
ax.bar(range(4), [v[0] for v in vals],
       yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=5, color=cols)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
ax.set_ylabel("deficit-corrected benefit"); ax.set_title("Non-monotone encoders: PLE (blue) vs learned embed (red), band at/off mode")
plt.tight_layout(); plt.savefig("nonmono_encoders.png", dpi=120)
print("\nsaved nonmono_encoders.png")
