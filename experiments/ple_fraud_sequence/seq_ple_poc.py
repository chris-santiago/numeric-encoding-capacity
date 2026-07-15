# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "torch>=2.2",
#   "scikit-learn>=1.4",
#   "matplotlib>=3.8",
# ]
# ///
"""seq_ple_poc.py — Step 1 PoC for the sequence-model reformulation:

    In a sequence model over per-card history, the amount fraud signal is amount-IN-CONTEXT
    (relative to the card's history). An explicit amount-deviation feature helps; per-step PLE
    of raw amount adds little beyond what the GRU already infers from the raw amount sequence.

Synthetic per-card sequences (no external data). One command:
    uv run seq_ple_poc.py

Surrogate: each sequence is one card's recent history of L transactions; predict whether the
LAST transaction is fraud. Fraud = the last amount is anomalous FOR THAT CARD (a big spike or a
tiny test charge relative to the card's typical level). The signal is cross-time context, so a
model that only sees the last amount (no history) cannot detect it.

Arms:
  - tab_last_raw : logistic regression on the last-step raw amount only (no sequence) — baseline
  - seq_raw      : GRU over per-step raw (std log) amount
  - seq_ple      : GRU over per-step PLE(amount) bins
  - seq_dev      : GRU over per-step [raw, amount-deviation-vs-card-history]

Deliberately OUT (deferred to Step 6): bootstrap CIs, shuffled-history falsification lever,
real IEEE-CIS sequences, multiple seeds, n_bins tuning, hyperparameter tuning.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
N_SEQ = 6000
L = 16            # sequence length (card history window)
N_BINS = 24
HIDDEN = 32
EPOCHS = 8
BATCH = 256


def make_sequences(n_seq, length, seed):
    """Per-card sequences of log-amounts; last step is fraud iff its amount is anomalous
    for that card (big spike or tiny test charge). Returns log_amt (n,L), y_last (n,)."""
    rng = np.random.default_rng(seed)
    card_typ = rng.normal(3.0, 1.0, size=n_seq)                       # card's typical log-amount
    log_amt = rng.normal(card_typ[:, None], 0.4, size=(n_seq, length))  # normal history
    y = np.zeros(n_seq, dtype=int)
    is_fraud = rng.uniform(size=n_seq) < 0.12
    spike = rng.uniform(size=n_seq) < 0.5
    for i in range(n_seq):
        if is_fraud[i]:
            y[i] = 1
            if spike[i]:
                log_amt[i, -1] = card_typ[i] + rng.uniform(2.0, 4.0)   # anomalous large
            else:
                log_amt[i, -1] = card_typ[i] - rng.uniform(2.0, 4.0)   # anomalous tiny (test charge)
    # add difficulty: some NON-fraud last steps also moderately deviate (no label)
    moderate = (~is_fraud) & (rng.uniform(size=n_seq) < 0.15)
    log_amt[moderate, -1] = card_typ[moderate] + rng.normal(0, 1.2, size=moderate.sum())
    return log_amt, y


def causal_deviation(log_amt):
    """Per-step deviation of amount from the card's running history (causal). (n,L)."""
    length = log_amt.shape[1]
    dev = np.zeros_like(log_amt)
    for t in range(1, length):
        m = log_amt[:, :t].mean(axis=1)
        s = log_amt[:, :t].std(axis=1) + 1e-6
        dev[:, t] = (log_amt[:, t] - m) / s
    return dev


def fit_ple_edges(x, n_bins):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, edges.size):
        if edges[t] <= edges[t - 1]:
            edges[t] = edges[t - 1] + 1e-9
    return edges


def ple_transform(x, edges):  # x: (...,) -> (..., n_bins)
    n_bins = edges.size - 1
    out = np.empty(x.shape + (n_bins,))
    for t in range(n_bins):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


class GRUClf(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def train_gru(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    model = GRUClf(Xtr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    n = len(ytr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy()
    return s


def main():
    log_amt, y = make_sequences(N_SEQ, L, SEED)
    dev = causal_deviation(log_amt)
    n_train = int(0.7 * N_SEQ)
    tr, te = slice(0, n_train), slice(n_train, N_SEQ)
    ytr, yte = y[tr], y[te]
    print(f"n_train={n_train} n_test={N_SEQ-n_train} fraud_rate={y.mean():.3f}")

    # standardize log amount on train (flattened)
    mu, sd = log_amt[tr].mean(), log_amt[tr].std()
    raw = (log_amt - mu) / sd
    edges = fit_ple_edges(log_amt[tr].ravel(), N_BINS)
    ple = ple_transform(log_amt, edges)                       # (n,L,n_bins)

    feats = {
        "seq_raw": raw[..., None],                             # (n,L,1)
        "seq_ple": ple,                                        # (n,L,n_bins)
        "seq_dev": np.stack([raw, dev], axis=-1),             # (n,L,2)
    }
    results = {}
    # non-sequence baseline: logreg on last-step raw amount only
    lr = LogisticRegression(max_iter=1000).fit(raw[tr, -1].reshape(-1, 1), ytr)
    results["tab_last_raw"] = average_precision_score(
        yte, lr.predict_proba(raw[te, -1].reshape(-1, 1))[:, 1])
    for name, X in feats.items():
        s = train_gru(X[tr], ytr, X[te], SEED)
        results[name] = average_precision_score(yte, s)

    print("\n=== PR-AUC (test) ===")
    for k, v in results.items():
        print(f"  {k:14s} {v:.4f}")
    print(f"\n  seq_raw - tab_last_raw = {results['seq_raw']-results['tab_last_raw']:+.4f} (sequence context value)")
    print(f"  seq_dev - seq_raw      = {results['seq_dev']-results['seq_raw']:+.4f} (explicit deviation value)")
    print(f"  seq_ple - seq_raw      = {results['seq_ple']-results['seq_raw']:+.4f} (PLE-of-raw value)")

    # mechanism figure: PR-AUC bars + fraud detectability vs |last-step deviation|
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(results)
    axA.bar(names, [results[n] for n in names],
            color=["#bbbbbb", "#1f77b4", "#2ca02c", "#d62728"])
    for i, n in enumerate(names):
        axA.text(i, results[n] + 0.01, f"{results[n]:.3f}", ha="center", fontweight="bold")
    axA.set_ylabel("PR-AUC")
    axA.set_title("Sequence arms vs non-sequence baseline")
    axA.tick_params(axis="x", rotation=20)

    last_dev = dev[:, -1]
    bins = np.quantile(np.abs(last_dev), np.linspace(0, 1, 11))
    centers, rates = [], []
    for i in range(10):
        m = (np.abs(last_dev) >= bins[i]) & (np.abs(last_dev) <= bins[i + 1] if i == 9 else np.abs(last_dev) < bins[i + 1])
        centers.append(i)
        rates.append(y[m].mean() if m.sum() else 0)
    axB.plot(centers, rates, "o-", color="tab:purple")
    axB.set_title("Fraud rate vs |last-step amount-deviation| decile")
    axB.set_xlabel("|deviation from card history| decile")
    axB.set_ylabel("fraud rate")
    fig.tight_layout()
    fig.savefig("seq_poc_mechanism.png", dpi=120)
    print("\nwrote seq_poc_mechanism.png")


if __name__ == "__main__":
    main()
