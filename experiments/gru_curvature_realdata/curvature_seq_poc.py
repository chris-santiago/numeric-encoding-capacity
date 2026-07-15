# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 8 PoC (synthetic) — affine-input GRU, curved-feature PLE, precondition-gated.

Proves the machinery end-to-end on a CONSTRUCTED-sequential signal:
  (1) Precondition gate: an affine-input GRU beats a strong tabular baseline AND uses temporal order
      (a shuffle-prior-steps test drops PR-AUC). On real data (Step 6) this gate decides whether the
      encoding question is even askable — Cycle 3 failed it.
  (2) Deficit-aware curvature test: does per-step PLE on a CURVED count feature beat log, above the
      PLE structural deficit measured on a LOG-ADEQUATE amount feature (the negative control)?

Two per-step features:
  count c_t: risk = recency-weighted sum of a CUBIC (log-mismatched) function of c -> PLE should help.
  amount a_t: risk = recency-weighted sum of log(a) (log-adequate) -> PLE should NOT help (deficit only).
Order matters: aggregation is recency-weighted, so shuffling prior steps changes it.

TRAINING REGIME — mirrors Cycle 6 (gru_perstep_encoding_fraud/flow, conf/training/default.yaml):
  minibatched SGD (batch 256, reshuffled every epoch), up to 30 epochs, a held-out VALIDATION split,
  early-stopping on validation PR-AUC (patience 6) with best-state restore. This is non-negotiable: the
  earlier full-batch / 25-step / no-validation loop UNDERTRAINED the wide per-step PLE path (12 bins/
  feature = ~10x the parameters of the log scalar) and produced a spurious PLE deficit. PLE only gets a
  fair read when the GRU is optimized like an actual model, per Cycle 6's caveat.

DELIBERATELY LEFT OUT (-> Step 6): real account-sequence data, L=300, hidden sweep {32,64}, 8-seed
paired-t + Holm, departure-from-log feature ranking on real features. PoC uses L=32, hidden 32, 3 seeds.
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
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 32, 30, 0.01, 256, 6   # Cycle-6 regime
DECAY = 0.9
SEEDS = [0, 1, 2]


def _std(v, mu=None, sd=None):
    if mu is None:
        mu, sd = v.mean(), v.std() + 1e-12
    return (v - mu) / sd, mu, sd


def make_sequences(n, L, seed):
    rng = np.random.default_rng(seed)
    count = np.exp(rng.normal(0.0, 1.2, size=(n, L)))          # curved-risk feature
    amount = np.exp(rng.normal(3.0, 0.9, size=(n, L)))         # log-risk feature
    w = DECAY ** np.arange(L)[::-1]                            # recency weights (recent = higher)
    w = w / w.sum()
    sc, _, _ = _std(np.log(count))
    gc = _std(sc ** 3)[0]                                      # CUBIC in log-count: log-mismatched
    sa, _, _ = _std(np.log(amount))                           # log-adequate
    agg_c = (gc * w).sum(axis=1)
    agg_a = (sa * w).sum(axis=1)
    logit = 1.6 * _std(agg_c)[0] + 1.0 * _std(agg_a)[0]
    b = -np.quantile(logit, 0.93)                             # ~7% base
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return count, amount, y


def fit_ple_edges(x, nbins):
    e = np.quantile(x, np.linspace(0, 1, nbins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def ple(x, edges):                                            # (n,L) -> (n,L,nbins)
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def fit_ref(ctr, atr):
    """Fit all encoders on TRAIN only (edges + std params), applied to any split."""
    lc, la = np.log1p(ctr), np.log1p(atr)
    return {"lc_mu": lc.mean(), "lc_sd": lc.std() + 1e-12,
            "la_mu": la.mean(), "la_sd": la.std() + 1e-12,
            "c_mu": ctr.mean(), "c_sd": ctr.std() + 1e-12,
            "a_mu": atr.mean(), "a_sd": atr.std() + 1e-12,
            "c_edges": fit_ple_edges(ctr, N_BINS), "a_edges": fit_ple_edges(atr, N_BINS)}


def enc_col(mode, x, ref, which):
    """mode: 'log' | 'raw' | 'ple' for column `which` in {'c','a'} -> (n,L,d)."""
    if mode == "log":
        return (((np.log1p(x) - ref[f"l{which}_mu"]) / ref[f"l{which}_sd"])[..., None]).astype(np.float32)
    if mode == "raw":
        return (((x - ref[f"{which}_mu"]) / ref[f"{which}_sd"])[..., None]).astype(np.float32)
    if mode == "ple":
        return ple(x, ref[f"{which}_edges"])
    raise ValueError(mode)


def featurize(mode_c, mode_a, c, a, ref):
    return np.concatenate([enc_col(mode_c, c, ref, "c"), enc_col(mode_a, a, ref, "a")], axis=-1).astype(np.float32)


class GRUClf(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, HIDDEN, batch_first=True)   # affine per-step read
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gru_score(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def train_gru(Xtr, ytr, Xva, yva, seed):
    """Cycle-6 regime: minibatch + per-epoch reshuffle + val early-stopping w/ best-state restore."""
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = GRUClf(Xtr.shape[2])
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
        ap = average_precision_score(yva, gru_score(m, Xva))
        if ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def tabular_ap(ctr, atr, ytr, cte, ate, yte):
    def feats(c, a):
        lc, la = np.log1p(c), np.log1p(a)
        return np.column_stack([lc[:, -1], la[:, -1], lc.mean(1), la.mean(1), lc.std(1), la.std(1)])
    m = LogisticRegression(max_iter=2000).fit(feats(ctr, atr), ytr)
    return average_precision_score(yte, m.predict_proba(feats(cte, ate))[:, 1])


arms = {"log": ("log", "log"), "ple_count": ("ple", "log"),
        "ple_amount": ("log", "ple"), "raw": ("raw", "raw")}
res = {k: [] for k in arms}
tab, shuf, base = [], [], []
for seed in SEEDS:
    ctr, atr, ytr = make_sequences(N_TRAIN, L, 100 + seed)
    cva, ava, yva = make_sequences(N_VAL, L, 300 + seed)
    cte, ate, yte = make_sequences(N_TEST, L, 500 + seed)
    base.append(yte.mean())
    ref = fit_ref(ctr, atr)
    tab.append(tabular_ap(ctr, atr, ytr, cte, ate, yte))
    log_model = None
    for name, (mc, ma) in arms.items():
        Xtr = featurize(mc, ma, ctr, atr, ref)
        Xva = featurize(mc, ma, cva, ava, ref)
        Xte = featurize(mc, ma, cte, ate, ref)
        m = train_gru(Xtr, ytr, Xva, yva, seed)
        res[name].append(average_precision_score(yte, gru_score(m, Xte)))
        if name == "log":
            log_model = m
    # shuffle test on the trained log arm: SAME model, scrambled test-time order
    rng = np.random.default_rng(9000 + seed)
    cte_s, ate_s = cte.copy(), ate.copy()
    for r in range(len(cte_s)):
        p = rng.permutation(L - 1)
        cte_s[r, :L - 1] = cte_s[r, p]; ate_s[r, :L - 1] = ate_s[r, p]
    Xte_s = featurize("log", "log", cte_s, ate_s, ref)
    shuf.append(average_precision_score(yte, gru_score(log_model, Xte_s)))


def mn(x):
    return float(np.mean(x))


print(f"=== base {mn(base):.3f}; L={L}, hidden={HIDDEN}, up to {EPOCHS} epochs (val early-stop), {len(SEEDS)} seeds ===")
print("\n=== PRECONDITION GATE ===")
print(f"  GRU(log) PR-AUC        : {mn(res['log']):.3f}")
print(f"  tabular baseline       : {mn(tab):.3f}   -> GRU beats tabular: {mn(res['log'])>mn(tab)}")
print(f"  GRU(log) order-shuffled: {mn(shuf):.3f}   -> uses order (drop): {mn(res['log'])-mn(shuf):+.3f}")
gate = mn(res['log']) > mn(tab) and (mn(res['log']) - mn(shuf)) > 0.01
print(f"  PRECONDITION {'PASS' if gate else 'FAIL'}")

print("\n=== PR-AUC by arm ===")
for k in arms:
    print(f"  {k:12s} {mn(res[k]):.3f}")

print("\n=== DEFICIT-AWARE curvature test ===")
d_count = mn(res["ple_count"]) - mn(res["log"])    # PLE on curved count vs log
d_amt = mn(res["ple_amount"]) - mn(res["log"])     # PLE on log-adequate amount vs log (deficit proxy)
print(f"  ple_count - log (curved)      : {d_count:+.3f}")
print(f"  ple_amount - log (deficit ref): {d_amt:+.3f}   (amount = negative control / log-adequate)")
print(f"  curvature benefit above deficit: {d_count - d_amt:+.3f}")
print("  H_curv (real-data class) directionally supported if this is > 0.")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(range(len(arms)), [mn(res[k]) for k in arms], color=["#888", "#4477aa", "#ccbb44", "#bbb"])
ax.axhline(mn(tab), color="r", ls="--", lw=1, label="tabular baseline")
ax.set_xticks(range(len(arms))); ax.set_xticklabels(list(arms), rotation=15)
ax.set_ylabel("PR-AUC"); ax.set_title("Cycle 8 PoC — affine GRU, curved-feature PLE (synthetic)"); ax.legend()
plt.tight_layout(); plt.savefig("curvature_seq_poc.png", dpi=110)
print("\nsaved curvature_seq_poc.png")
