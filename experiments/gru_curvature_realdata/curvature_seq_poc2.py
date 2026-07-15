# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
Cycle 8 HARDENED PoC — addresses the debate's critique_wins remediation (F1-F6).

The plain PoC proved the machinery runs but the debate found a Step-6 null would be UNINTERPRETABLE:
no positive control (F1), a rank-invariant metric that can hide representation (F2), a tabular baseline
too weak to make the gate meaningful (F3), and a contaminated deficit reference biasing the estimand
conservative (F4), plus a missing attribution arm (F5). This iteration closes all of them on synthetic
data BEFORE any real-data compute.

F1 — POWER / POSITIVE CONTROL (the load-bearing test):
  Sweep departure-from-log with a single monotone knob k on the count feature's risk:
    g_k(s) = (exp(k*s) - 1) / k     # k->0 -> s (log-adequate); larger k -> sharper curvature
  Deficit-corrected curvature benefit = AP(ple_count) - AP(ple_ref) per seed (the shared `log` term
  cancels). If this estimate + 95% CI move monotonically off zero and cross the CI-excludes-0 adoption
  bar as k grows, the estimand HAS power to detect real curvature -> a Step-6 null is trustworthy.
  If even exaggerated curvature stays at zero, the estimand is blind and Step 6 must not be run.

F4 — CLEAN DEFICIT REFERENCE: the reference feature is drawn from the SAME marginal as count
  (exp(N(0,1.2))) and is EXACTLY log-adequate (risk linear in its log, zero higher-order terms), so
  AP(ple_ref) - AP(log) is an uncontaminated, marginal-matched structural deficit — unlike Cycle-8's
  `amount`, which was weakly-curved AND marginal-mismatched.

F3 — STRONG TABULAR BASELINE: HistGradientBoosting on features that INCLUDE recency/EWMA-weighted
  aggregates matching the DGP's 0.9-decay, with a seed-level CI on the GRU-minus-tabular margin.

F5 — DENSE ARM: per-step Linear->ReLU (free per-step nonlinearity), then the affine GRU. If dense
  recovers the same benefit as ple_count, the gain is generic capacity, not curvature-matched encoding.

F2 — MAGNITUDE METRICS: log-loss and Brier reported alongside PR-AUC (not rank-invariant), to reveal a
  representational gap PR-AUC could mask.

F6 — SCOPE: single-feature-at-a-time targeting (count is the only PLE target); multi-feature synergy is
  a follow-up cycle (HYPOTHESIS.md tightened).

Training mirrors Cycle 6: minibatch bs256 + per-epoch reshuffle, 30 epochs, held-out validation split,
early-stop patience 6, best-state restore. Still PoC scale (L=32, hidden 32, 5 seeds); L=300 = Step 6.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0)
N_TRAIN, N_VAL, N_TEST = 4000, 2000, 4000
L, N_BINS = 32, 12
HIDDEN, EPOCHS, LR, BATCH, PATIENCE, DENSE_H = 32, 30, 0.01, 256, 6, 16
DECAY = 0.9
SEEDS = [0, 1, 2, 3, 4]
T_CRIT = 2.776                                      # t_.975, df=4 (5 seeds)
K_LEVELS = [(0.01, "~linear"), (0.7, "mild"), (1.4, "moderate"), (2.1, "exaggerated")]
K_CANON = 1.4                                       # canonical curvature for gate/attribution/metrics
W_COUNT, W_REF = 1.6, 1.0
W = (DECAY ** np.arange(L)[::-1]); W = W / W.sum()  # recency weights (shared by DGP and EWMA baseline)


def _std(v, mu=None, sd=None):
    if mu is None:
        mu, sd = v.mean(), v.std() + 1e-12
    return (v - mu) / sd, mu, sd


def curve(k, s):
    """Monotone departure-from-log family: k->0 gives s (log-adequate), larger k more convex."""
    return (np.exp(k * s) - 1.0) / k


def make_sequences(k, n, seed):
    """count = curved-risk target; ref = marginal-matched, exactly log-adequate deficit reference."""
    rng = np.random.default_rng(seed)
    count = np.exp(rng.normal(0.0, 1.2, size=(n, L)))
    ref = np.exp(rng.normal(0.0, 1.2, size=(n, L)))            # SAME marginal as count (F4)
    sc = _std(np.log(count))[0]
    sr = _std(np.log(ref))[0]                                  # linear in its log = log-adequate (F4)
    gc = _std(curve(k, sc))[0]                                 # curved by knob k
    agg_c = (gc * W).sum(axis=1)
    agg_r = (sr * W).sum(axis=1)
    logit = W_COUNT * _std(agg_c)[0] + W_REF * _std(agg_r)[0]
    b = -np.quantile(logit, 0.93)                             # ~7% base
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit + b)))).astype(np.float32)
    return count, ref, y, (logit + b)                         # true per-seq log-odds = oracle ceiling


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
    return {"lc_mu": lc.mean(), "lc_sd": lc.std() + 1e-12,
            "lr_mu": lr.mean(), "lr_sd": lr.std() + 1e-12,
            "c_mu": ctr.mean(), "c_sd": ctr.std() + 1e-12,
            "r_mu": rtr.mean(), "r_sd": rtr.std() + 1e-12,
            "c_edges": fit_ple_edges(ctr, N_BINS), "r_edges": fit_ple_edges(rtr, N_BINS)}


def enc_col(mode, x, R, which):
    if mode == "log":
        return (((np.log1p(x) - R[f"l{which}_mu"]) / R[f"l{which}_sd"])[..., None]).astype(np.float32)
    if mode == "raw":
        return (((x - R[f"{which}_mu"]) / R[f"{which}_sd"])[..., None]).astype(np.float32)
    if mode == "ple":
        return ple(x, R[f"{which}_edges"])
    raise ValueError(mode)


ARM_MODES = {"log": ("log", "log"), "ple_count": ("ple", "log"),
             "ple_ref": ("log", "ple"), "raw": ("raw", "raw")}


def arm_input(arm, c, r, R):
    if arm == "dense":                                        # per-step log scalars; ReLU proj in model
        return np.stack([(np.log1p(c) - R["lc_mu"]) / R["lc_sd"],
                         (np.log1p(r) - R["lr_mu"]) / R["lr_sd"]], axis=-1).astype(np.float32)
    mc, mr = ARM_MODES[arm]
    return np.concatenate([enc_col(mc, c, R, "c"), enc_col(mr, r, R, "r")], axis=-1).astype(np.float32)


class SeqEncGRU(nn.Module):
    def __init__(self, in_dim, dense=False):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, DENSE_H), nn.ReLU()) if dense else None
        self.gru = nn.GRU(DENSE_H if dense else in_dim, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)                                  # free per-step nonlinearity, THEN affine GRU
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(1)


def gru_score(m, X):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(X))).numpy()


def train_gru(Xtr, ytr, Xva, yva, seed, dense=False):
    torch.manual_seed(seed); torch.set_num_threads(1)
    m = SeqEncGRU(Xtr.shape[2], dense=dense)
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
            best_state = {kk: v.clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m


def strong_tabular_ap(ctr, rtr, ytr, cte, rte, yte):
    """F3: GBM on features that INCLUDE recency/EWMA aggregates matching the DGP decay."""
    def feats(c, r):
        lc, lr = np.log1p(c), np.log1p(r)
        return np.column_stack([lc[:, -1], lr[:, -1], lc.mean(1), lr.mean(1), lc.std(1), lr.std(1),
                                lc.max(1), lr.max(1), (lc * W).sum(1), (lr * W).sum(1)])  # EWMA last two
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                       random_state=0).fit(feats(ctr, rtr), ytr)
    return average_precision_score(yte, m.predict_proba(feats(cte, rte))[:, 1])


def paired_ci(diffs):
    d = np.asarray(diffs, float)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - T_CRIT * se, m + T_CRIT * se


# ---- run: train all arms per (k, seed); store test scores -------------------------------------------
scores, ytest, tab_ap, oracle_ap = {}, {}, {}, {}
for k, klab in K_LEVELS:
    canon = abs(k - K_CANON) < 1e-9
    arms = ["log", "ple_count", "ple_ref"] + (["dense", "raw"] if canon else [])
    for seed in SEEDS:
        ctr, rtr, ytr, _ = make_sequences(k, N_TRAIN, 100 + seed)
        cva, rva, yva, _ = make_sequences(k, N_VAL, 300 + seed)
        cte, rte, yte, ste = make_sequences(k, N_TEST, 500 + seed)
        ytest[(k, seed)] = yte
        if canon:                                             # oracle ceiling: rank by TRUE log-odds
            oracle_ap[seed] = average_precision_score(yte, ste)
        R = fit_ref(ctr, rtr)
        for arm in arms:
            dense = (arm == "dense")
            m = train_gru(arm_input(arm, ctr, rtr, R), ytr,
                          arm_input(arm, cva, rva, R), yva, seed, dense=dense)
            scores[(k, seed, arm)] = gru_score(m, arm_input(arm, cte, rte, R))
            if canon and arm == "log":                        # F3 gate: order-shuffle + strong tabular
                rng = np.random.default_rng(9000 + seed)
                cte_s, rte_s = cte.copy(), rte.copy()
                for row in range(len(cte_s)):
                    p = rng.permutation(L - 1)
                    cte_s[row, :L - 1] = cte_s[row, p]; rte_s[row, :L - 1] = rte_s[row, p]
                scores[(k, seed, "log_shuf")] = gru_score(m, arm_input("log", cte_s, rte_s, R))
                tab_ap[seed] = strong_tabular_ap(ctr, rtr, ytr, cte, rte, yte)


def ap(k, seed, arm):
    return average_precision_score(ytest[(k, seed)], scores[(k, seed, arm)])


# ---- F1: power curve — deficit-corrected benefit vs curvature knob ----------------------------------
print("=" * 74)
print("F1 — POWER / POSITIVE CONTROL: deficit-corrected curvature benefit vs departure-from-log")
print("  benefit(seed) = AP(ple_count) - AP(ple_ref)   [shared log term cancels]")
print(f"{'k':>6} {'label':>12} {'benefit':>9} {'95% CI':>20} {'crosses0':>9}")
power = []
for k, klab in K_LEVELS:
    diffs = [ap(k, s, "ple_count") - ap(k, s, "ple_ref") for s in SEEDS]
    m, lo, hi = paired_ci(diffs)
    power.append((k, klab, m, lo, hi))
    print(f"{k:6.2f} {klab:>12} {m:+9.3f} [{lo:+.3f}, {hi:+.3f}] {'YES' if lo > 0 else 'no':>9}")
monotonic = all(power[i][2] <= power[i + 1][2] + 1e-6 for i in range(len(power) - 1))
detects = any(p[3] > 0 for p in power)
print(f"  -> monotonic in curvature: {monotonic} ; detects (some CI>0): {detects}")
print(f"  -> ESTIMAND {'HAS POWER (Step-6 null trustworthy)' if monotonic and detects else 'IS BLIND (do NOT run Step 6)'}")

# ---- F3: precondition gate with strong baseline + CIs (canonical k) ----------------------------------
kc = K_CANON
gru_log = [ap(kc, s, "log") for s in SEEDS]
margin = [ap(kc, s, "log") - tab_ap[s] for s in SEEDS]
drop = [ap(kc, s, "log") - average_precision_score(ytest[(kc, s)], scores[(kc, s, "log_shuf")]) for s in SEEDS]
mm, mlo, mhi = paired_ci(margin); dm, dlo, dhi = paired_ci(drop)
print("\n" + "=" * 74)
print(f"F3 — PRECONDITION GATE (strong GBM+EWMA baseline, k={kc})")
print(f"  GRU(log) AP {np.mean(gru_log):.3f}  vs  strong tabular {np.mean(list(tab_ap.values())):.3f}")
print(f"  GRU - tabular margin : {mm:+.3f}  [{mlo:+.3f}, {mhi:+.3f}]  {'CLEAR>0' if mlo > 0 else 'NOT CLEAR'}")
print(f"  order-shuffle drop   : {dm:+.3f}  [{dlo:+.3f}, {dhi:+.3f}]  {'CLEAR>0' if dlo > 0 else 'NOT CLEAR'}")
gate = mlo > 0 and dlo > 0
print(f"  -> PRECONDITION {'PASS (CI-clear both legs)' if gate else 'FAIL'}")

# ---- ceiling check: is the log-GRU already near the oracle? (disambiguates 'blind' vs 'no lever') ----
orc = np.mean(list(oracle_ap.values()))
ceil_gap = [oracle_ap[s] - ap(kc, s, "log") for s in SEEDS]
cm, clo, chi = paired_ci(ceil_gap)
print("\n" + "=" * 74)
print(f"CEILING CHECK (k={kc}): oracle (rank by true log-odds) AP = {orc:.3f}")
print(f"  oracle - GRU(log) gap : {cm:+.3f}  [{clo:+.3f}, {chi:+.3f}]")
print(f"  -> log-GRU {'AT ceiling: monotone curvature already captured; no per-step lever exists' if cm < 0.03 else 'BELOW ceiling: headroom remains — flat power curve may be underpowering, not no-lever'}")

# ---- F5 + F2: attribution and magnitude metrics (canonical k) ---------------------------------------
print("\n" + "=" * 74)
print(f"F5 — ATTRIBUTION (k={kc}): is the benefit curvature-specific, or generic per-step capacity?")
pc = [ap(kc, s, "ple_count") - ap(kc, s, "log") for s in SEEDS]
dn = [ap(kc, s, "dense") - ap(kc, s, "log") for s in SEEDS]
pcm, pclo, pchi = paired_ci(pc); dnm, dnlo, dnhi = paired_ci(dn)
print(f"  ple_count - log : {pcm:+.3f} [{pclo:+.3f}, {pchi:+.3f}]")
print(f"  dense    - log  : {dnm:+.3f} [{dnlo:+.3f}, {dnhi:+.3f}]")
print(f"  -> {'PLE-specific (dense does NOT recover it)' if dnm < pcm - 0.01 else 'generic capacity (dense recovers it) — attribution weak'}")

print(f"\nF2 — MAGNITUDE-SENSITIVE METRICS (k={kc}; lower is better):")
print(f"{'arm':>10} {'PR-AUC':>8} {'log-loss':>9} {'Brier':>8}")
for arm in ["log", "ple_count", "ple_ref", "dense", "raw"]:
    aps = np.mean([ap(kc, s, arm) for s in SEEDS])
    lls = np.mean([log_loss(ytest[(kc, s)], np.clip(scores[(kc, s, arm)], 1e-6, 1 - 1e-6)) for s in SEEDS])
    brs = np.mean([brier_score_loss(ytest[(kc, s)], scores[(kc, s, arm)]) for s in SEEDS])
    print(f"{arm:>10} {aps:8.3f} {lls:9.4f} {brs:8.4f}")

# ---- figure: power curve (the F1 deliverable) -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = [p[0] for p in power]; ms = [p[2] for p in power]
los = [p[2] - p[3] for p in power]; his = [p[4] - p[2] for p in power]
ax.errorbar(ks, ms, yerr=[los, his], marker="o", capsize=4, color="#1f77b4", label="deficit-corrected benefit")
ax.axhline(0, color="k", lw=0.8, ls=":")
for p in power:
    ax.annotate(p[1], (p[0], p[2]), textcoords="offset points", xytext=(6, 6), fontsize=8)
ax.set_xlabel("curvature knob k (departure-from-log)"); ax.set_ylabel("AP(ple_count) - AP(ple_ref)")
ax.set_title("F1 power curve: does the estimand detect curvature as it grows?"); ax.legend()
plt.tight_layout(); plt.savefig("curvature_power_curve.png", dpi=120)
print("\nsaved curvature_power_curve.png")
