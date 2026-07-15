# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
Follow-up probe (Cycle 8 harness) — two questions:

Q1 (head-to-head): for ONE per-step scalar feature, is a small LEARNED embedding (Linear(1->d)->ReLU,
    scalar->d, learned knots) better or worse than a FIXED PLE (scalar->d, quantile knots)? Both expand to
    the SAME dim d, are concatenated with the untouched reference feature, and read AFFINELY by the GRU.
    -> This is fixed-vs-learned knot placement, same piecewise-linear function class, same dim.

Q2 (resolution): for a feature that is technically NON-MONOTONE but whose non-monotonicity is finer than a
    bin (so it looks ~monotone at bin resolution), does PLE still help? Sweep the spatial SCALE sigma of a
    std-normalized Gaussian risk band (constant signal variance across sigma -> we vary RESOLVABILITY, not
    signal strength). Report both encoders' benefit vs a bin-monotonicity diagnostic.
    Prediction: sub-bin non-monotonicity -> PLE ~ log (blind); the LEARNED embed (continuous knots) may
    resolve it better. Multi-bin -> both help, fixed PLE robust.

Shapes on the target `count` feature (risk contribution std-normalized to unit variance -> constant signal):
    mono_cubic : gc = std(sc^3)                  monotone, curved (Cycle-8 control: expect no lever)
    band(sigma): gc = std(exp(-sc^2/2 sigma^2))  inverted-U, non-monotone; sigma = spatial scale
`ref` is a marginal-matched, exactly-log-adequate second feature (deficit reference, always log-scalar).

Arms (same GRU backbone, affine read; only the target's encoding differs):
    log : [std log count (1), std log ref (1)]
    ple : [PLE(count) (d), std log ref (1)]                       fixed quantile knots
    mlp : [std log count (1), std log ref (1)] -> model embeds ch0 via Linear(1->d)->ReLU  learned knots
d = N_BINS so ple and mlp expand the target to the SAME width. Cycle-6 training regime; 5 seeds; PR-AUC.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
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
W_COUNT, W_REF = 1.6, 1.0
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()
SHAPES = [("mono_cubic", None), ("band", 0.10), ("band", 0.25), ("band", 0.50), ("band", 1.00)]


def _std(v, mu=None, sd=None):
    if mu is None:
        mu, sd = v.mean(), v.std() + 1e-12
    return (v - mu) / sd, mu, sd


def target_signal(shape, sc):
    """Per-step risk contribution of the target, PRE-normalization (std applied by caller)."""
    kind, sigma = shape
    if kind == "mono_cubic":
        return sc ** 3
    if kind == "band":
        return np.exp(-(sc ** 2) / (2 * sigma ** 2))         # inverted-U: high at center, low at tails
    raise ValueError(shape)


def make_sequences(shape, n, seed):
    rng = np.random.default_rng(seed)
    count = np.exp(rng.normal(0.0, 1.2, size=(n, L)))
    ref = np.exp(rng.normal(0.0, 1.2, size=(n, L)))
    sc = _std(np.log(count))[0]
    sr = _std(np.log(ref))[0]
    gc = _std(target_signal(shape, sc))[0]                   # unit-variance -> constant signal across shapes
    agg_c = (gc * W).sum(axis=1)
    agg_r = (sr * W).sum(axis=1)
    logit = W_COUNT * _std(agg_c)[0] + W_REF * _std(agg_r)[0]
    b = -np.quantile(logit, 0.93)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return count, ref, y, (logit + b)


def bin_nonmono(shape):
    """Fraction of binned-risk variance NOT explained by the best monotone fit (0=bins monotone)."""
    sc = np.linspace(-3, 3, 20000)
    risk = _std(target_signal(shape, sc))[0]
    edges = np.quantile(sc, np.linspace(0, 1, N_BINS + 1))
    idx = np.clip(np.digitize(sc, edges[1:-1]), 0, N_BINS - 1)
    centers = np.array([sc[idx == b].mean() for b in range(N_BINS)])
    means = np.array([risk[idx == b].mean() for b in range(N_BINS)])
    tot = ((means - means.mean()) ** 2).sum() + 1e-12
    best_r2 = max(1 - ((means - IsotonicRegression(increasing=d).fit_transform(centers, means)) ** 2).sum() / tot
                  for d in (True, False))
    return max(0.0, 1 - best_r2)


def fit_ple_edges(x, nbins):
    e = np.quantile(x, np.linspace(0, 1, nbins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def ple(x, edges):
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def fit_ref(ctr, rtr):
    lc, lr = np.log1p(ctr), np.log1p(rtr)
    return {"lc_mu": lc.mean(), "lc_sd": lc.std() + 1e-12, "lr_mu": lr.mean(), "lr_sd": lr.std() + 1e-12,
            "c_edges": fit_ple_edges(ctr, N_BINS)}


def slog(x, mu, sd):
    return ((np.log1p(x) - mu) / sd).astype(np.float32)


def arm_input(arm, c, r, R):
    scount = slog(c, R["lc_mu"], R["lc_sd"])[..., None]
    sref = slog(r, R["lr_mu"], R["lr_sd"])[..., None]
    if arm == "log":
        return np.concatenate([scount, sref], axis=-1)
    if arm == "mlp":                                         # model embeds ch0; feed raw scalars
        return np.concatenate([scount, sref], axis=-1)
    if arm == "ple":
        return np.concatenate([ple(c, R["c_edges"]), sref], axis=-1).astype(np.float32)
    raise ValueError(arm)


class GRUEnc(nn.Module):
    def __init__(self, in_dim, embed_ch0=False, d=N_BINS):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, d), nn.ReLU()) if embed_ch0 else None
        gin = (d + in_dim - 1) if embed_ch0 else in_dim
        self.gru = nn.GRU(gin, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        if self.embed is not None:
            x = torch.cat([self.embed(x[..., 0:1]), x[..., 1:]], dim=-1)   # learned per-step embed of ch0
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gru_score(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def train_gru(Xtr, ytr, Xva, yva, seed, embed_ch0=False):
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUEnc(Xtr.shape[2], embed_ch0=embed_ch0)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(EPOCHS):
        m.train()
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(m(Xtr_t[idx]), ytr_t[idx]).backward(); opt.step()
        apv = average_precision_score(yva, gru_score(m, Xva))
        if apv > best_ap + 1e-5:
            best_ap, bad = apv, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def paired_ci(diffs):
    d = np.asarray(diffs, float)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - T_CRIT * se, m + T_CRIT * se


scores, ytest, oracle_ap = {}, {}, {}
for shape in SHAPES:
    for seed in SEEDS:
        ctr, rtr, ytr, _ = make_sequences(shape, N_TRAIN, 100 + seed)
        cva, rva, yva, _ = make_sequences(shape, N_VAL, 300 + seed)
        cte, rte, yte, ste = make_sequences(shape, N_TEST, 500 + seed)
        ytest[(shape, seed)] = yte
        oracle_ap[(shape, seed)] = average_precision_score(yte, ste)
        R = fit_ref(ctr, rtr)
        for arm in ("log", "ple", "mlp"):
            m = train_gru(arm_input(arm, ctr, rtr, R), ytr, arm_input(arm, cva, rva, R), yva, seed,
                          embed_ch0=(arm == "mlp"))
            scores[(shape, seed, arm)] = gru_score(m, arm_input(arm, cte, rte, R))


def ap(shape, seed, arm):
    return average_precision_score(ytest[(shape, seed)], scores[(shape, seed, arm)])


def label(shape):
    return shape[0] if shape[1] is None else f"band s={shape[1]:.2f}"


print("=" * 92)
print("Q1/Q2 — learned per-feature embed (mlp) vs fixed PLE, same dim; benefit vs log; bin-monotonicity")
print(f"{'shape':>14} {'binNonMono':>11} {'oracle':>7} {'ple-log':>18} {'mlp-log':>18} {'ple-mlp':>18}")
rows = []
for shape in SHAPES:
    nm = bin_nonmono(shape)
    orc = np.mean([oracle_ap[(shape, s)] for s in SEEDS])
    pl = paired_ci([ap(shape, s, "ple") - ap(shape, s, "log") for s in SEEDS])
    ml = paired_ci([ap(shape, s, "mlp") - ap(shape, s, "log") for s in SEEDS])
    pm = paired_ci([ap(shape, s, "ple") - ap(shape, s, "mlp") for s in SEEDS])
    rows.append((shape, nm, orc, pl, ml, pm))
    def f(c):
        return f"{c[0]:+.3f}[{c[1]:+.3f},{c[2]:+.3f}]"
    print(f"{label(shape):>14} {nm:11.3f} {orc:7.3f} {f(pl):>18} {f(ml):>18} {f(pm):>18}")

print("\nReadout:")
print("  * ple-log CI>0  -> fixed PLE helps ;  mlp-log CI>0 -> learned embed helps")
print("  * ple-mlp CI>0  -> PLE beats learned embed at same dim ;  <0 -> learned embed wins")
print("  * binNonMono ~0 (bins look monotone) should coincide with ple-log ~ 0")

fig, ax = plt.subplots(figsize=(7.2, 4.4))
xs = list(range(len(SHAPES)))
for key, col, lab in [(3, "#1f77b4", "ple - log"), (4, "#d62728", "mlp - log")]:
    ax.errorbar(xs, [r[key][0] for r in rows],
                yerr=[[r[key][0] - r[key][1] for r in rows], [r[key][2] - r[key][0] for r in rows]],
                marker="o", capsize=3, color=col, label=lab)
ax.plot(xs, [r[1] for r in rows], "k--", alpha=0.6, label="bin non-monotonicity (0=monotone bins)")
ax.axhline(0, color="k", lw=0.8, ls=":")
ax.set_xticks(xs); ax.set_xticklabels([label(s) for s in SHAPES], rotation=15)
ax.set_ylabel("PR-AUC benefit vs log  /  bin non-monotonicity")
ax.set_title("Fixed PLE vs learned per-feature embed as non-monotonicity scale shrinks toward a bin")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("proj_vs_ple.png", dpi=120)
print("\nsaved proj_vs_ple.png")
