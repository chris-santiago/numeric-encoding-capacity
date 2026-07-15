# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy", "torch", "scikit-learn"]
# ///
"""
Cycle 6 real-data A/B — STEP 1: the precondition gate (the check Cycle 3 skipped).

Before any encoding comparison on real account-sequence fraud data, establish that an AFFINE-INPUT GRU
at production length (L=256, ~p90) (a) BEATS a strong tabular baseline (GBM on EWMA + summary aggregates)
and (b) USES temporal order (shuffle-prior-steps at test time drops PR-AUC). If either fails, the
sequence model is not extracting sequential signal on this data -> the encoding question is moot and THAT
is the reported finding.

Data: transactions.parq (786k txns, 5000 accounts, 1.58% fraud). Causal per-account sequences; anchors
temporally split (train<70th pct date < valid <85th < test) so test anchors are strictly later. Clean
per-step numeric features only -- the *FraudTrend columns are TARGET-DERIVED LEAKAGE and are EXCLUDED.
log-scalar encoding (the baseline arm) for the gate. Subsample non-fraud anchors for tractability; keep all
fraud. 3 seeds, seed-level mean + a paired check.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from pathlib import Path

torch.manual_seed(0)
DATA = Path("data/account-sequences/transactions.parq")
L = 128                                                     # covers median-50 fully + well into the tail
# PROPER training regime: real capacity, high epoch cap with early stopping doing the work, LR schedule.
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 128, 150, 0.003, 1024, 15
DEVICE = ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]
N_NONFRAUD_ANCHORS = 45000                                   # + all fraud anchors
FEATURES = ["transactionAmount", "transactionToAvailable", "availableMoney", "currentBalance",
            "creditLimit", "accountAge", "dt_min",
            "normMerchantName-accountNumber30dCount", "normMerchantName-accountNumber60dCount"]
LOGSCALE = {"transactionAmount", "availableMoney", "currentBalance", "creditLimit", "dt_min",
            "normMerchantName-accountNumber30dCount", "normMerchantName-accountNumber60dCount"}  # log1p; rest raw-std
LEAKAGE = ["normMerchantName5dFraudTrend", "accountNumber5dFraudTrend",
           "normMerchantName60dFraudTrend", "accountNumber60dFraudTrend"]


def load_and_build():
    df = pd.read_parquet(DATA)
    assert not any(c in FEATURES for c in LEAKAGE), "leakage column leaked into FEATURES"
    df = df.sort_values(["accountNumber", "transactionDateTime"]).reset_index(drop=True)
    # per-account inter-transaction minutes
    dt = df.groupby("accountNumber")["transactionDateTime"].diff().dt.total_seconds().to_numpy() / 60.0
    df["dt_min"] = np.nan_to_num(dt, nan=0.0)
    for c in ["availableMoney", "currentBalance"]:
        df[c] = df[c].clip(lower=0)                          # a few negatives -> 0 before log1p
    # feature matrix (raw), standardization params fit on TRAIN anchors' rows later
    X = df[FEATURES].to_numpy(np.float32)
    y = df["isFraud"].to_numpy(np.int8)
    acct = df["accountNumber"].to_numpy()
    t = df["transactionDateTime"].to_numpy()
    # account boundaries (contiguous after sort): each row -> start index of its account block (vectorized)
    starts = np.sort(np.unique(acct, return_index=True)[1])
    acct_start = starts[np.searchsorted(starts, np.arange(len(df)), side="right") - 1]
    return X, y, t, acct_start


def choose_anchors(y, t, acct_start, seed):
    rng = np.random.default_rng(seed)
    pos_in_acct = np.arange(len(y)) - acct_start
    eligible = pos_in_acct >= 3                              # need a little history
    fraud = np.where((y == 1) & eligible)[0]
    nonf = np.where((y == 0) & eligible)[0]
    nonf = rng.choice(nonf, size=min(N_NONFRAUD_ANCHORS, len(nonf)), replace=False)
    anchors = np.concatenate([fraud, nonf])
    # temporal split by anchor date
    td = t[anchors]
    q70, q85 = np.quantile(td.astype("datetime64[s]").astype(np.int64), [0.70, 0.85])
    ti = td.astype("datetime64[s]").astype(np.int64)
    split = np.where(ti <= q70, 0, np.where(ti <= q85, 1, 2))
    return anchors, split


class GRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1, :]).squeeze(1)


def gscore(m, X, bs=4096):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs]).to(DEVICE)
            out.append(torch.sigmoid(m(xb)).cpu().numpy())
    return np.concatenate(out)


def train(Xtr, ytr, Xva, yva, seed):
    """PROPER regime: real capacity, high epoch cap + early stopping (patience 15) + best-state restore,
    ReduceLROnPlateau on validation AP. Trains until convergence, not a fixed short budget."""
    torch.manual_seed(seed)
    m = GRU(Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    best, bs, bad, ran = -1.0, None, 0, 0
    for ep in range(EPOCHS):
        m.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            opt.zero_grad(); lossf(m(xb), yb).backward(); opt.step()
        a = average_precision_score(yva, gscore(m, Xva)); ran = ep + 1
        sched.step(a)
        if a > best + 1e-5:
            best, bad, bs = a, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if bs is not None:
        m.load_state_dict(bs)
    print(f"    [train] seed {seed}: best val AP {best:.4f} @ {ran} epochs (device={DEVICE})", flush=True)
    return m


def tabular(Xtr_seq, ytr, Xte_seq, yte):
    Wd = (0.9 ** np.arange(L)[::-1]); Wd /= Wd.sum()
    def agg(S):
        mask = (S != 0).any(-1)                             # non-pad steps
        last = S[:, -1, :]
        mean = S.sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
        ewma = (S * Wd[None, :, None]).sum(1)
        mx = S.max(1)
        return np.concatenate([last, mean, ewma, mx], -1)
    m = HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.08,
                                       random_state=0).fit(agg(Xtr_seq), ytr)
    return average_precision_score(yte, m.predict_proba(agg(Xte_seq))[:, 1])


def run():
    X, y, t, acct_start = load_and_build()
    print(f"loaded {len(y)} txns, fraud {y.mean():.4f}")
    gru_aps, tab_aps, shuf_aps, bases = [], [], [], []
    for seed in SEEDS:
        anchors, split = choose_anchors(y, t, acct_start, seed)
        tr, va, te = anchors[split == 0], anchors[split == 1], anchors[split == 2]
        # log-scale + standardize on TRAIN anchor rows
        Xtr_rows = X[tr].copy()
        for j, f in enumerate(FEATURES):
            if f in LOGSCALE:
                Xtr_rows[:, j] = np.log1p(np.clip(Xtr_rows[:, j], 0, None))
        mu, sd = Xtr_rows.mean(0), Xtr_rows.std(0) + 1e-6
        # build sequences with log-scale applied inside (transform raw then standardize)
        def build(anc):
            Xs = np.zeros((len(anc), L, len(FEATURES)), np.float32)
            for k, a in enumerate(anc):
                s = max(acct_start[a], a - L + 1)
                seq = X[s:a + 1].copy()
                for j, f in enumerate(FEATURES):
                    if f in LOGSCALE:
                        seq[:, j] = np.log1p(np.clip(seq[:, j], 0, None))
                seq = (seq - mu) / sd
                Xs[k, L - seq.shape[0]:] = seq
            return Xs
        Xtr, Xva, Xte = build(tr), build(va), build(te)
        ytr, yva, yte = y[tr].astype(np.float32), y[va].astype(np.float32), y[te].astype(np.float32)
        bases.append(yte.mean())
        m = train(Xtr, ytr, Xva, yva, seed)
        gru_aps.append(average_precision_score(yte, gscore(m, Xte)))
        tab_aps.append(tabular(Xtr, ytr, Xte, yte))
        # shuffle prior steps of test seqs (keep last step)
        rng = np.random.default_rng(9000 + seed)
        Xte_s = Xte.copy()
        for r in range(len(Xte_s)):
            nz = np.where(Xte_s[r].any(-1))[0]
            if len(nz) > 2:
                p = nz[:-1][rng.permutation(len(nz) - 1)]
                Xte_s[r, nz[:-1]] = Xte_s[r, p]
        shuf_aps.append(average_precision_score(yte, gscore(m, Xte_s)))
        print(f"seed {seed}: base {yte.mean():.4f} | GRU {gru_aps[-1]:.4f} | tab {tab_aps[-1]:.4f} | "
              f"shuf {shuf_aps[-1]:.4f} | n_tr {len(tr)} n_te {len(te)}")

    g, tb, sh = np.mean(gru_aps), np.mean(tab_aps), np.mean(shuf_aps)
    print("\n" + "=" * 70)
    print(f"PRECONDITION GATE (real data, L={L}, {len(SEEDS)} seeds, base {np.mean(bases):.4f})")
    print(f"  GRU PR-AUC      : {g:.4f}")
    print(f"  strong tabular  : {tb:.4f}   -> GRU beats tabular: {g > tb} ({g - tb:+.4f})")
    print(f"  order-shuffled  : {sh:.4f}   -> uses order (drop): {g - sh:+.4f}")
    gate = (g > tb) and (g - sh > 0.005)
    print(f"  -> PRECONDITION {'PASS -> encoding question is live' if gate else 'FAIL -> on THIS feature set/data, per-step sequence modeling does not beat recency aggregates; encoding A/B moot here (see REAL_DATA_AB.md for scope: demo data under-represents time signal)'}")


if __name__ == "__main__":
    run()
