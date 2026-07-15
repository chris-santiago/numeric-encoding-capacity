# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib>=3.8"]
# ///
"""
Step 1 PoC — Learned periodic embeddings vs fixed sin/cos vs raw, on controlled 1-D targets.

WHAT THIS TESTS (mechanism, in isolation):
  Gorishniy et al. 2022 periodic embedding (the "PLR" form): expand a scalar x into
  [sin(2*pi*c_i*x), cos(2*pi*c_i*x)]_{i=1..k} -> Linear -> ReLU, with the k frequencies c_i
  TRAINABLE (init N(0, sigma^2)). Fixed sin/cos is the special case: one hand-chosen frequency
  for a known period, not learned. Raw is the bare scalar.

  Three controlled response shapes for P(y=1 | x), x in [0,1]:
    - known_period   : sin(2*pi*x)            -> period known; fixed sin/cos is already sufficient.
    - multi_harmonic : sin(2*pi*3*x)          -> a 3rd harmonic (e.g. fraud peaks 3x/day); the fixed
                                                 fundamental sin/cos CANNOT see it; learned freqs can.
    - two_bumps      : two Gaussian bumps      -> NON-periodic, non-monotone (e.g. inter-txn time);
                                                 periodic acts as a learned Fourier basis vs raw.

  Two head capacities to expose the capacity argument from cycles 1-3:
    - linear : a weak head (logistic regression on the encoding)
    - mlp    : a strong head (1 hidden layer) that can bend a 1-D transform itself

PRIMARY METRIC: PR-AUC (average precision) on a held-out test set, ~10% positive base rate
  (matches the imbalanced fraud regime). Reported as mean over seeds.

MECHANISM VISUALS:
  fig_shapes.png     - the three ground-truth response curves the encoders must fit.
  fig_frequencies.png- the LEARNED |c_i| spectrum per regime (does periodic discover the true freq?).
  fig_prauc_poc.png  - PR-AUC across (regime x encoder x head): when does each encoder win?

DELIBERATELY LEFT OUT (this is a PoC, not the experiment):
  - No real data (synthetic 1-D only); real account-sequence data enters at Step 6.
  - No sequence model (the GRU is Step 6). Here the encoding is tested on a STATIC classifier so the
    encoder effect is not entangled with recurrence.
  - No tuning of sigma (init scale) or k; fixed sigma=2.0, k=16. sigma-sensitivity is a known leftover.
  - One feature per regime; real data has many, fit jointly.
  - IID synthetic split (no temporal/causal split); causality matters only at Step 6.
  - Trivial baseline = constant predictor (PR-AUC == base rate), printed for reference.
"""
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

REGIMES = ["known_period", "multi_harmonic", "two_bumps"]
ENCODERS = ["raw", "sincos_fixed", "periodic_learned"]
HEADS = ["linear", "mlp"]
N_TRAIN, N_TEST = 8000, 8000
SEEDS = [0, 1, 2]
EPOCHS, LR = 400, 1e-2
K_FREQS, SIGMA, PLR_OUT = 16, 2.0, 16  # periodic embedding: k freqs, init scale, PLR linear width
TARGET_POS_RATE = 0.10
DEVICE = torch.device("cpu")


# ----------------------------------------------------------------------------- data
def latent_shape(regime: str, x: np.ndarray) -> np.ndarray:
    """Ground-truth latent driver g(x); standardized to unit std before use."""
    if regime == "known_period":
        g = np.sin(2 * np.pi * x)
    elif regime == "multi_harmonic":
        g = np.sin(2 * np.pi * 3.0 * x)  # period 1/3 — invisible to a fundamental sin/cos
    elif regime == "two_bumps":
        g = np.exp(-(((x - 0.30) / 0.08) ** 2)) - np.exp(-(((x - 0.70) / 0.06) ** 2))
    else:
        raise ValueError(f"unknown regime {regime}")
    return (g - g.mean()) / (g.std() + 1e-12)


def make_data(regime: str, n: int, rng: np.random.Generator):
    """x ~ U(0,1); y ~ Bernoulli(sigmoid(a*g(x) + b)), b set for ~TARGET_POS_RATE positives."""
    x = rng.uniform(0.0, 1.0, size=(n, 1)).astype(np.float32)
    g = latent_shape(regime, x[:, 0])
    a = 3.0  # signal amplitude (shared across regimes since g is standardized)
    # solve bias b so mean(sigmoid(a*g + b)) ~= TARGET_POS_RATE via a quick scan
    bs = np.linspace(-8, 4, 240)
    rates = np.array([1 / (1 + np.exp(-(a * g + b))) for b in bs]).mean(axis=1)
    b = bs[np.argmin(np.abs(rates - TARGET_POS_RATE))]
    p = 1.0 / (1.0 + np.exp(-(a * g + b)))
    y = (rng.uniform(size=n) < p).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)


# ----------------------------------------------------------------------------- encoders
class RawEncoder(nn.Module):
    out_dim = 1

    def forward(self, x):
        return x


class FixedSinCos(nn.Module):
    """Fixed fundamental sin/cos (period 1) — the reference-style cyclic encoding, NOT learned."""
    out_dim = 2

    def forward(self, x):
        z = 2 * math.pi * x
        return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)


class PeriodicLearned(nn.Module):
    """Gorishniy PLR: k learned frequencies -> sin/cos -> Linear -> ReLU."""

    def __init__(self, k=K_FREQS, sigma=SIGMA, out=PLR_OUT):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(k) * sigma)  # init N(0, sigma^2)
        self.lin = nn.Linear(2 * k, out)
        self.out_dim = out

    def forward(self, x):
        z = 2 * math.pi * x * self.freqs  # (N, k) via broadcast
        per = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)  # (N, 2k)
        return torch.relu(self.lin(per))


def build_encoder(name: str) -> nn.Module:
    if name == "raw":
        return RawEncoder()
    if name == "sincos_fixed":
        return FixedSinCos()
    if name == "periodic_learned":
        return PeriodicLearned()
    raise ValueError(f"unknown encoder {name}")


def build_head(name: str, in_dim: int) -> nn.Module:
    if name == "linear":
        return nn.Linear(in_dim, 1)
    if name == "mlp":
        return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    raise ValueError(f"unknown head {name}")


# ----------------------------------------------------------------------------- train/eval
def train_eval(regime, encoder_name, head_name, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr, ytr = make_data(regime, N_TRAIN, rng)
    xte, yte = make_data(regime, N_TEST, rng)

    enc = build_encoder(encoder_name).to(DEVICE)
    head = build_head(head_name, enc.out_dim).to(DEVICE)
    model = nn.Sequential(enc, head)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        logit = model(xtr).squeeze(-1)
        lossf(logit, ytr).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(xte).squeeze(-1)).cpu().numpy()
    ap = average_precision_score(yte.numpy(), scores)
    learned_freqs = None
    if encoder_name == "periodic_learned":
        learned_freqs = enc.freqs.detach().cpu().numpy().copy()
    return ap, learned_freqs


# ----------------------------------------------------------------------------- figures
def fig_shapes():
    x = np.linspace(0, 1, 500)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, reg in zip(axes, REGIMES):
        g = latent_shape(reg, x)
        p = 1 / (1 + np.exp(-(3.0 * g - 2.2)))
        ax.plot(x, p, color="#1f77b4")
        ax.set_title(reg); ax.set_xlabel("x"); ax.set_ylabel("P(y=1|x)")
    fig.suptitle("Ground-truth response shapes the encoders must fit")
    fig.tight_layout(); fig.savefig("fig_shapes.png", dpi=120); plt.close(fig)


def fig_frequencies(freq_by_regime):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    truth = {"known_period": 1.0, "multi_harmonic": 3.0, "two_bumps": None}
    for ax, reg in zip(axes, REGIMES):
        f = np.abs(freq_by_regime[reg])
        ax.hist(f, bins=20, color="#2ca02c", alpha=0.85)
        if truth[reg] is not None:
            ax.axvline(truth[reg], color="#d62728", ls="--", lw=1.5,
                       label=f"true freq = {truth[reg]:.0f}")
            ax.legend(fontsize=8)
        ax.set_title(f"learned |c_i| — {reg}"); ax.set_xlabel("|frequency|")
    fig.suptitle("Mechanism: frequencies the periodic embedding DISCOVERS (linear head, seed 0)")
    fig.tight_layout(); fig.savefig("fig_frequencies.png", dpi=120); plt.close(fig)


def fig_prauc(ap_table, base_rate):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    width, xpos = 0.38, np.arange(len(ENCODERS))
    for ax, reg in zip(axes, REGIMES):
        for i, head in enumerate(HEADS):
            means = [ap_table[(reg, enc, head)] for enc in ENCODERS]
            ax.bar(xpos + (i - 0.5) * width, means, width, label=head, alpha=0.85)
        ax.axhline(base_rate, color="k", ls=":", lw=0.9, label=f"base rate {base_rate:.2f}")
        ax.set_title(reg); ax.set_xticks(xpos); ax.set_xticklabels(ENCODERS, rotation=15, ha="right")
    axes[0].set_ylabel("PR-AUC (mean over seeds)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("PR-AUC by regime x encoder x head capacity")
    fig.tight_layout(); fig.savefig("fig_prauc_poc.png", dpi=120); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap_table, freq_by_regime = {}, {}
    for reg in REGIMES:
        for enc in ENCODERS:
            for head in HEADS:
                aps, freqs = [], None
                for s in SEEDS:
                    ap, lf = train_eval(reg, enc, head, s)
                    aps.append(ap)
                    if lf is not None and s == 0 and head == "linear":
                        freqs = lf
                ap_table[(reg, enc, head)] = float(np.mean(aps))
                if freqs is not None:
                    freq_by_regime[reg] = freqs

    base_rate = TARGET_POS_RATE
    fig_shapes(); fig_frequencies(freq_by_regime); fig_prauc(ap_table, base_rate)

    # ---- structured summary (the "number") ----
    print(f"\nPoC: learned periodic vs fixed sin/cos vs raw  | base rate (trivial baseline) = {base_rate:.3f}")
    print("=" * 86)
    print(f"{'regime':16s} {'head':7s} {'raw':>8s} {'sincos':>8s} {'periodic':>9s} "
          f"{'per-sin':>9s} {'per-raw':>9s}")
    print("-" * 86)
    for reg in REGIMES:
        for head in HEADS:
            r = ap_table[(reg, "raw", head)]
            s = ap_table[(reg, "sincos_fixed", head)]
            p = ap_table[(reg, "periodic_learned", head)]
            print(f"{reg:16s} {head:7s} {r:8.3f} {s:8.3f} {p:9.3f} "
                  f"{p - s:+9.3f} {p - r:+9.3f}")
    print("=" * 86)
    print("Reading: per-sin = periodic - sincos (H1 lever); per-raw = periodic - raw (H2/Fourier lever).")
    print("Wrote fig_shapes.png, fig_frequencies.png, fig_prauc_poc.png")


if __name__ == "__main__":
    main()
