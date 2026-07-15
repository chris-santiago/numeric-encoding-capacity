# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib>=3.8"]
# ///
"""
Step 1 PoC — per-step numeric encoding in an AFFINE-INPUT sequence GRU.

QUESTION: in a GRU whose per-step numerics enter the recurrence affinely (W·e(x_t), no per-step
nonlinear projection — the reference architecture), does UNBOTTLENECKING the per-step numeric path
help when the signal is band-selective + cross-step?

  Per step: amount, Δt (log-normal). Two regimes:
    band     : fraud = burst (cross-step count) of steps where Δt in a SHORT band AND amount in a
               SMALL band — band-selective (non-monotone) in both, conjunction across features,
               aggregated across steps. The case a single affine-read log scalar struggles with.
    monotone : fraud monotone in sequence-mean log-amount (negative control); log scalar suffices.

  Arms (shared GRU backbone, affine per-step input; only the per-step numeric encoding differs):
    scalar : GRU on [std log amt, std log Δt]            (reference baseline)
    ple    : GRU on [PLE(amt), PLE(Δt)] (quantile, on RAW)(fixed basis)
    dense  : GRU on ReLU(Dense([std log amt, std log Δt]))(free per-step nonlinearity = MECHANISM /
                                                           POSITIVE control)
    tab    : logreg on generic sequence aggregates        (trivial baseline, non-negotiable)

  CONTROLS (pre-registered):
    positive control = (band): dense MUST beat scalar -> the affine bottleneck is the cause + power.
    negative control = (monotone): no arm beats scalar.
    precondition gate = an ORACLE feature (true band-match count) logreg PR-AUC >> base.

PRIMARY METRIC: PR-AUC. PoC: small L, 3 seeds, means. (Seed-level CIs + length axis at Step 6.)
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

REGIMES = ["band", "monotone"]
ARMS = ["scalar", "ple", "dense"]
L_POC = 48
N_TRAIN, N_TEST = 5000, 5000
SEEDS = [0, 1, 2]
EPOCHS, LR, HIDDEN, BATCH = 25, 1e-2, 32, 256
N_BINS, DENSE_H = 16, 16
TARGET_POS = 0.09
# bands (raw units): amount small band, Δt short band — both band-selective
A_LO, A_HI = 3.0, 18.0
D_LO, D_HI = 0.3, 6.0


def gen(regime, n, L, rng):
    amt = np.exp(rng.normal(3.0, 0.9, (n, L))).astype(np.float32)        # ~median 20
    dt = np.exp(rng.normal(2.6, 1.5, (n, L))).astype(np.float32)         # minutes
    if regime == "band":
        match = ((amt >= A_LO) & (amt <= A_HI) & (dt >= D_LO) & (dt <= D_HI)).astype(np.float32)
        w = (0.97 ** np.arange(L - 1, -1, -1)).astype(np.float32)        # recency-weighted (GRU-tractable)
        score = (match * w[None, :]).sum(axis=1)                         # leaky-integrator cross-step aggregation
    elif regime == "monotone":
        score = np.log1p(amt).mean(axis=1)                               # monotone aggregate
    else:
        raise ValueError(regime)
    s = (score - score.mean()) / (score.std() + 1e-9)
    a = 2.8
    bs = np.linspace(-8, 4, 240)
    b = bs[np.argmin(np.abs(np.array([1 / (1 + np.exp(-(a * s + bb))) for bb in bs]).mean(1) - TARGET_POS))]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(a * s + b)))).astype(np.float32)
    return amt, dt, y, score


def fit_ple_edges(x, n_bins):
    e = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def ple(x, edges):  # (n,L) -> (n,L,nbins)
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def featurize(arm, amt, dt, ref):
    la, ld = np.log1p(amt), np.log1p(dt)
    sa = (la - ref["la_mu"]) / ref["la_sd"]
    sd = (ld - ref["ld_mu"]) / ref["ld_sd"]
    if arm in ("scalar", "dense"):
        return np.stack([sa, sd], axis=-1).astype(np.float32)
    if arm == "ple":
        return np.concatenate([ple(amt, ref["amt_edges"]), ple(dt, ref["dt_edges"])], axis=-1).astype(np.float32)
    raise ValueError(arm)


def tab_features(amt, dt):  # generic aggregates (no band-count leak)
    la, ld = np.log1p(amt), np.log1p(dt)
    return np.stack([la.mean(1), la.std(1), la.min(1), la.max(1),
                     ld.mean(1), ld.std(1), ld.min(1), ld.max(1)], axis=1).astype(np.float32)


class SeqEncGRU(nn.Module):
    def __init__(self, arm, in_dim, hidden=HIDDEN):
        super().__init__()
        self.arm = arm
        self.proj = nn.Sequential(nn.Linear(in_dim, DENSE_H), nn.ReLU()) if arm == "dense" else None
        gru_in = DENSE_H if arm == "dense" else in_dim
        self.gru = nn.GRU(gru_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X):
        if self.proj is not None:
            X = self.proj(X)            # per-step free nonlinearity, THEN affine GRU read
        out, _ = self.gru(X)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_gru(arm, Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = SeqEncGRU(arm, Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    n = len(ytr)
    for epoch in range(EPOCHS):
        model.train()
        g = torch.Generator().manual_seed(seed * 100 + epoch)
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad(); lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xte))).numpy()


def run_cell(regime, arm, seed):
    rng = np.random.default_rng(seed)
    amt_tr, dt_tr, ytr, _ = gen(regime, N_TRAIN, L_POC, rng)
    amt_te, dt_te, yte, _ = gen(regime, N_TEST, L_POC, rng)
    la, ld = np.log1p(amt_tr), np.log1p(dt_tr)
    ref = {"la_mu": la.mean(), "la_sd": la.std() + 1e-6, "ld_mu": ld.mean(), "ld_sd": ld.std() + 1e-6,
           "amt_edges": fit_ple_edges(amt_tr, N_BINS), "dt_edges": fit_ple_edges(dt_tr, N_BINS)}
    Xtr, Xte = featurize(arm, amt_tr, dt_tr, ref), featurize(arm, amt_te, dt_te, ref)
    s = train_gru(arm, Xtr, ytr, Xte, seed)
    return float(average_precision_score(yte, s))


def run_tab_and_oracle(regime, seed):
    rng = np.random.default_rng(seed)
    amt_tr, dt_tr, ytr, sc_tr = gen(regime, N_TRAIN, L_POC, rng)
    amt_te, dt_te, yte, sc_te = gen(regime, N_TEST, L_POC, rng)
    tab = LogisticRegression(max_iter=2000).fit(tab_features(amt_tr, dt_tr), ytr)
    ap_tab = average_precision_score(yte, tab.predict_proba(tab_features(amt_te, dt_te))[:, 1])
    orc = LogisticRegression(max_iter=2000).fit(sc_tr[:, None], ytr)
    ap_orc = average_precision_score(yte, orc.predict_proba(sc_te[:, None])[:, 1])
    return ap_tab, ap_orc


def main():
    tab = {}
    for reg in REGIMES:
        for arm in ARMS:
            tab[(reg, arm)] = float(np.mean([run_cell(reg, arm, s) for s in SEEDS]))
        tabs, orcs = zip(*[run_tab_and_oracle(reg, s) for s in SEEDS])
        tab[(reg, "tab")] = float(np.mean(tabs))
        tab[(reg, "oracle")] = float(np.mean(orcs))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    order = ["tab", "scalar", "ple", "dense"]
    for ax, reg in zip(axes, REGIMES):
        ax.bar(order, [tab[(reg, a)] for a in order],
               color=["#555", "#1f77b4", "#2ca02c", "#d62728"], alpha=0.85)
        ax.axhline(TARGET_POS, color="k", ls=":", lw=0.9, label=f"base {TARGET_POS:.2f}")
        ax.axhline(tab[(reg, "oracle")], color="#888", ls="--", lw=0.9, label="oracle (precond)")
        ax.set_title(reg); ax.set_ylabel("PR-AUC (mean over seeds)"); ax.legend(fontsize=8)
    fig.suptitle("Per-step encoding in an affine-input GRU (PoC, L=48): does unbottlenecking help?")
    fig.tight_layout(); fig.savefig("fig_poc_prauc.png", dpi=120); plt.close(fig)

    print(f"\nGRU per-step encoding PoC | L={L_POC} | base {TARGET_POS}")
    print("=" * 64)
    for reg in REGIMES:
        print(f"\n[{reg}]  tab={tab[(reg,'tab')]:.3f}  scalar={tab[(reg,'scalar')]:.3f}  "
              f"ple={tab[(reg,'ple')]:.3f}  dense={tab[(reg,'dense')]:.3f}  "
              f"| oracle={tab[(reg,'oracle')]:.3f}")
        print(f"        dense−scalar={tab[(reg,'dense')]-tab[(reg,'scalar')]:+.3f}  "
              f"ple−scalar={tab[(reg,'ple')]-tab[(reg,'scalar')]:+.3f}  "
              f"ple−dense={tab[(reg,'ple')]-tab[(reg,'dense')]:+.3f}")
    print("\n" + "=" * 64)
    bp = tab[("band", "oracle")]
    print(f"PRECONDITION (band oracle {bp:.3f} vs base {TARGET_POS}): "
          f"{'PASS' if bp > 2 * TARGET_POS else 'FAIL-VOID'}")
    pc = tab[("band", "dense")] - tab[("band", "scalar")]
    print(f"POSITIVE CONTROL (band dense−scalar {pc:+.3f}): {'FIRES' if pc > 0.02 else 'DID NOT FIRE'}")
    nc = max(tab[("monotone", "dense")] - tab[("monotone", "scalar")],
             tab[("monotone", "ple")] - tab[("monotone", "scalar")])
    print(f"NEG CONTROL (monotone max(dense,ple)−scalar {nc:+.3f}): {'CLEAN' if nc < 0.02 else 'LEAKS'}")
    print("wrote fig_poc_prauc.png")


if __name__ == "__main__":
    main()
