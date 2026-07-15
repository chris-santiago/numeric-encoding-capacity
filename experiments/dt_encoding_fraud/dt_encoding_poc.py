# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib>=3.8"]
# ///
"""
Step 1 PoC — Δt (inter-transaction time) encoding, on a CONSTRUCT-VALID informative feature.

WHAT THIS TESTS (the cycle-4 fix):
  Cycle 4 was void: it compared encodings of an UNINFORMATIVE feature (no power). Here Δt is
  informative BY CONSTRUCTION, with a built-in positive control so a null is meaningful.

  Δt = inter-transaction minutes. latent m_log ~ Normal; minutes = exp(m_log) (always > 0, heavy-
  tailed; sub-minute when m_log<0). log = log1p(minutes). The fraud signal is defined on the
  standardized log scale. Two regimes:
    - nonmono : U-shaped risk (short Δt = card-testing, long Δt = dormant-reactivation, both high-risk).
    - mono    : monotone risk (short Δt riskier). Negative control: log should already suffice.

  FIVE Δt encodings:
    raw     : standardized minutes (heavy-tailed)
    log     : standardized log1p(minutes)  [the reference model's transform]
    ple_raw : quantile PLE fit on RAW minutes        [standard Gorishniy PLE]
    ple_log : quantile PLE fit on log1p(minutes)     [pre-log variant — bin edges align by quantile,
                                                      but within-bin interpolation differs from ple_raw]
    learned : periodic PLR on standardized log (trainable; periodic basis needs a normalized input,
              so applying it to the heavy-tailed raw scale is ill-posed — log is the principled scale)
  Two model capacities: linear (weak) | mlp (strong).

  POWER BY DESIGN:
    - positive control = (linear, nonmono): a linear head CANNOT fit a U-shape from raw/log, so the
      PLE/learned bases MUST win there if encodings do anything.
    - real question    = (mlp, nonmono): does any richer encoding beat log once the model learns the
      transform itself? (capacity argument prior: no.)
    - negative control = (mlp, mono): richer encodings should NOT beat log; log >= raw.
  Read verdicts off the MLP row (all arms share MLP capacity; differ only in the Δt basis).
  PRECONDITION (hard gate): Δt-only best-MLP PR-AUC must be >> base, else the run is void.

PRIMARY METRIC: PR-AUC (average precision), ~8% base rate. 3 seeds, mean. (Bootstrap CIs at Step 6.)

DELIBERATELY LEFT OUT: real data (demo can't test this; real-world data not accessible); sequence/GRU arm
(MLP proxies 'strong model'); σ/k/n_bins tuning.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

REGIMES = ["nonmono", "mono"]
ENCODINGS = ["raw", "log", "ple_raw", "ple_log", "learned"]
MODELS = ["linear", "mlp"]
N_TRAIN, N_TEST = 9000, 9000
SEEDS = [0, 1, 2]
EPOCHS, LR = 350, 1e-2
N_BINS = 16
K_FREQS, SIGMA, PLR_OUT = 16, 2.0, 16
M_MU, M_SD = 3.0, 1.6             # latent m_log ~ Normal; minutes = exp(m_log)
A_SIG, C_CO = 2.6, 0.6
TARGET_POS = 0.08


# ----------------------------------------------------------------------------- data
def gen(regime, n, rng):
    m_log = rng.normal(M_MU, M_SD, n).astype(np.float32)
    minutes = np.exp(m_log).astype(np.float32)                  # always > 0, no clip needed
    loginc = np.log1p(minutes).astype(np.float32)               # the 'log' encoding base
    z = rng.normal(0, 1, n).astype(np.float32)
    s = (loginc - loginc.mean()) / (loginc.std() + 1e-12)
    g = s * s if regime == "nonmono" else -s
    g = (g - g.mean()) / (g.std() + 1e-12)
    lin = A_SIG * g + C_CO * z
    bs = np.linspace(-8, 4, 240)
    rates = np.array([1 / (1 + np.exp(-(lin + b))) for b in bs]).mean(axis=1)
    b = bs[np.argmin(np.abs(rates - TARGET_POS))]
    p = 1 / (1 + np.exp(-(lin + b)))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return {"minutes": minutes, "loginc": loginc, "z": z, "y": y}


# ----------------------------------------------------------------------------- encoders
def fit_ple_edges(x, n_bins):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, edges.size):
        if edges[t] <= edges[t - 1]:
            edges[t] = edges[t - 1] + 1e-9
    return edges


def ple_transform(x, edges):
    n_bins = edges.size - 1
    out = np.empty(x.shape + (n_bins,), dtype=np.float32)
    for t in range(n_bins):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def _std(col, ref):
    mu, sd = ref.mean(), ref.std() + 1e-6
    return ((col - mu) / sd).astype(np.float32)


def precompute_features(enc, tr, te):
    """(Xtr, Xte) for fixed encodings; 'learned' is in-model. Δt cols + co-feature z. Train-fit only."""
    ztr, zte = tr["z"][:, None], te["z"][:, None]
    if enc == "raw":
        a, b = _std(tr["minutes"], tr["minutes"])[:, None], _std(te["minutes"], tr["minutes"])[:, None]
    elif enc == "log":
        a, b = _std(tr["loginc"], tr["loginc"])[:, None], _std(te["loginc"], tr["loginc"])[:, None]
    elif enc == "ple_raw":
        edges = fit_ple_edges(tr["minutes"], N_BINS)            # quantile bins on RAW minutes
        a, b = ple_transform(tr["minutes"], edges), ple_transform(te["minutes"], edges)
    elif enc == "ple_log":
        edges = fit_ple_edges(tr["loginc"], N_BINS)             # quantile bins on log1p(minutes)
        a, b = ple_transform(tr["loginc"], edges), ple_transform(te["loginc"], edges)
    else:
        raise ValueError(enc)
    return (np.concatenate([a, ztr], axis=1).astype(np.float32),
            np.concatenate([b, zte], axis=1).astype(np.float32))


class PeriodicEmbed(nn.Module):
    def __init__(self, k=K_FREQS, sigma=SIGMA, out=PLR_OUT):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(k) * sigma)
        self.lin = nn.Linear(2 * k, out)
        self.out_dim = out

    def forward(self, x):
        zz = 2 * math.pi * x * self.freqs
        return torch.relu(self.lin(torch.cat([torch.sin(zz), torch.cos(zz)], dim=-1)))


def make_head(model_type, in_dim):
    if model_type == "linear":
        return nn.Linear(in_dim, 1)
    if model_type == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 48), nn.ReLU(), nn.Linear(48, 1))
    raise ValueError(model_type)


class FixedEncModel(nn.Module):
    def __init__(self, model_type, in_dim):
        super().__init__()
        self.head = make_head(model_type, in_dim)

    def forward(self, X):
        return self.head(X).squeeze(-1)


class LearnedEncModel(nn.Module):
    """PLR on standardized log Δt (col 0) + passthrough co-feature (col 1) -> head."""

    def __init__(self, model_type):
        super().__init__()
        self.emb = PeriodicEmbed()
        self.head = make_head(model_type, self.emb.out_dim + 1)

    def forward(self, X):
        return self.head(torch.cat([self.emb(X[:, 0:1]), X[:, 1:2]], dim=-1)).squeeze(-1)


# ----------------------------------------------------------------------------- train/eval
def train_eval(regime, enc, model_type, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, te = gen(regime, N_TRAIN, rng), gen(regime, N_TEST, rng)
    if enc == "learned":
        Xtr = np.stack([_std(tr["loginc"], tr["loginc"]), tr["z"]], axis=1).astype(np.float32)
        Xte = np.stack([_std(te["loginc"], tr["loginc"]), te["z"]], axis=1).astype(np.float32)
        model = LearnedEncModel(model_type)
    else:
        Xtr, Xte = precompute_features(enc, tr, te)
        model = FixedEncModel(model_type, Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(tr["y"])
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad(); lossf(model(Xtr_t), ytr_t).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(Xte))).numpy()
    return float(average_precision_score(te["y"], s))


# ----------------------------------------------------------------------------- figures
def fig_shapes():
    x = np.linspace(M_MU - 3 * M_SD, M_MU + 3 * M_SD, 300)
    loginc = np.log1p(np.exp(x))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, reg in zip(axes, REGIMES):
        s = (loginc - loginc.mean()) / (loginc.std() + 1e-12)
        g = s * s if reg == "nonmono" else -s
        g = (g - g.mean()) / (g.std() + 1e-12)
        p = 1 / (1 + np.exp(-(A_SIG * g - 2.6)))
        ax.plot(np.exp(x), p, color="#1f77b4")
        ax.set_xscale("log")
        ax.set_title(f"{reg}: P(fraud | Δt)"); ax.set_xlabel("minutes (log axis)"); ax.set_ylabel("P(fraud)")
    fig.suptitle("Construct-valid Δt signal (informative by design): U-shape vs monotone")
    fig.tight_layout(); fig.savefig("fig_dt_shapes.png", dpi=120); plt.close(fig)


def fig_prauc(tab, base):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    x = np.arange(len(ENCODINGS)); w = 0.38
    for ax, reg in zip(axes, REGIMES):
        for i, mt in enumerate(MODELS):
            ax.bar(x + (i - 0.5) * w, [tab[(reg, e, mt)] for e in ENCODINGS], w, label=mt, alpha=0.85)
        ax.axhline(base, color="k", ls=":", lw=0.9, label=f"base {base:.2f}")
        ax.set_title(reg); ax.set_xticks(x); ax.set_xticklabels(ENCODINGS, rotation=15, ha="right")
        ax.set_xlabel("Δt encoding")
    axes[0].set_ylabel("PR-AUC (mean over seeds)"); axes[-1].legend(fontsize=8)
    fig.suptitle("Δt encoding × model capacity × regime — when does encoding matter?")
    fig.tight_layout(); fig.savefig("fig_dt_prauc.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    tab = {(reg, enc, mt): float(np.mean([train_eval(reg, enc, mt, s) for s in SEEDS]))
           for reg in REGIMES for enc in ENCODINGS for mt in MODELS}
    fig_shapes(); fig_prauc(tab, TARGET_POS)

    print(f"\nΔt encoding PoC | base rate {TARGET_POS:.3f}")
    print("=" * 72)
    for reg in REGIMES:
        print(f"\n[{reg}]   {'enc':10s}{'linear':>10s}{'mlp':>10s}")
        for enc in ENCODINGS:
            print(f"          {enc:10s}{tab[(reg,enc,'linear')]:10.3f}{tab[(reg,enc,'mlp')]:10.3f}")

    best_mlp = max(tab[("nonmono", e, "mlp")] for e in ENCODINGS)
    print("\n" + "=" * 72)
    print(f"PRECONDITION: best-MLP nonmono PR-AUC = {best_mlp:.3f} vs base {TARGET_POS:.3f} -> "
          f"{'INFORMATIVE (PASS)' if best_mlp > 2 * TARGET_POS else 'WEAK (VOID)'}")
    ln = {e: tab[("nonmono", e, "linear")] for e in ENCODINGS}
    pos_ok = (min(ln['ple_raw'], ln['ple_log'], ln['learned']) > max(ln['raw'], ln['log']) + 0.03)
    print(f"POSITIVE CONTROL (linear,nonmono): ple_raw={ln['ple_raw']:.3f} ple_log={ln['ple_log']:.3f} "
          f"learned={ln['learned']:.3f} vs raw={ln['raw']:.3f} log={ln['log']:.3f} -> "
          f"{'FIRES' if pos_ok else 'DID NOT FIRE'}")
    nm = {e: tab[("nonmono", e, "mlp")] for e in ENCODINGS}
    print(f"REAL Q (mlp,nonmono) vs log: ple_raw={nm['ple_raw']-nm['log']:+.3f} "
          f"ple_log={nm['ple_log']-nm['log']:+.3f} learned={nm['learned']-nm['log']:+.3f}")
    mo = {e: tab[("mono", e, "mlp")] for e in ENCODINGS}
    print(f"NEG CONTROL (mlp,mono) vs log: ple_raw={mo['ple_raw']-mo['log']:+.3f} "
          f"ple_log={mo['ple_log']-mo['log']:+.3f} learned={mo['learned']-mo['log']:+.3f}  "
          f"log−raw={mo['log']-mo['raw']:+.3f}")
    print("wrote fig_dt_shapes.png, fig_dt_prauc.png")


if __name__ == "__main__":
    main()
