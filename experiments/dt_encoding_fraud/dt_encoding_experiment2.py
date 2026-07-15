# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "scipy"]
# ///
"""
Step 6 experiment — Δt encoding, debate- AND peer-review-hardened. Writes stats_results.json.
    uv run dt_encoding_experiment2.py

Hardening:
  Debate F5  bootstrap CIs; F1 capacity control `log_expand` + convergence verification (train loss);
        F4  Δt = exp(latent), no clip; F8 z-only floor; PLE on BOTH raw and log.
  Peer-review R1:
    M1  DECISION CIs are now SEED-LEVEL paired (5 per-seed paired diffs -> paired-t CI + p-value),
        the between-run uncertainty relevant to deployment. The old seed-0 row-bootstrap is kept only
        as a labeled SECONDARY diagnostic (within-seed evaluation noise).
    M3  add a REGULARIZED learned arm (`learned_reg`, weight_decay) and report 5-SEED-MEAN train loss
        per arm, to test whether the learned-vs-log test gap is overfitting (closes with reg) or not.
    m5  Holm multiple-comparison adjustment over the non-monotone-MLP decision family.

Encodings: raw|log|ple_raw|ple_log|learned|learned_reg|log_expand. Models: linear|mlp.
Regimes: nonmono|mono. Metric: PR-AUC. Base rate target 0.08 (realized ~0.072).
"""
import json
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import average_precision_score

REGIMES = ["nonmono", "mono"]
ENCODINGS = ["raw", "log", "ple_raw", "ple_log", "learned", "learned_reg", "log_expand"]
MODELS = ["linear", "mlp"]
N_TRAIN, N_TEST = 9000, 9000
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, LR = 500, 1e-2
WEIGHT_DECAY_REG = 1e-3          # M3: regularization for the learned_reg arm
N_BINS, K_FREQS, SIGMA, PLR_OUT = 16, 16, 2.0, 16
M_MU, M_SD, A_SIG, C_CO, TARGET_POS = 3.0, 1.6, 2.6, 0.6, 0.08
N_BOOT = 1000
HERE = pathlib.Path(__file__).parent


def gen(regime, n, rng):
    m_log = rng.normal(M_MU, M_SD, n).astype(np.float32)
    minutes = np.exp(m_log).astype(np.float32)
    loginc = np.log1p(minutes).astype(np.float32)
    z = rng.normal(0, 1, n).astype(np.float32)
    s = (loginc - loginc.mean()) / (loginc.std() + 1e-12)
    g = s * s if regime == "nonmono" else -s
    g = (g - g.mean()) / (g.std() + 1e-12)
    lin = A_SIG * g + C_CO * z
    bs = np.linspace(-8, 4, 240)
    b = bs[np.argmin(np.abs(np.array([1 / (1 + np.exp(-(lin + bb))) for bb in bs]).mean(1) - TARGET_POS))]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(lin + b)))).astype(np.float32)
    return {"minutes": minutes, "loginc": loginc, "z": z, "y": y}


def fit_ple_edges(x, n_bins):
    e = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def ple_transform(x, edges):
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def _std(col, ref):
    return ((col - ref.mean()) / (ref.std() + 1e-6)).astype(np.float32)


def precompute(enc, tr, te):
    ztr, zte = tr["z"][:, None], te["z"][:, None]
    if enc == "raw":
        a, b = _std(tr["minutes"], tr["minutes"])[:, None], _std(te["minutes"], tr["minutes"])[:, None]
    elif enc == "log":
        a, b = _std(tr["loginc"], tr["loginc"])[:, None], _std(te["loginc"], tr["loginc"])[:, None]
    elif enc == "ple_raw":
        ed = fit_ple_edges(tr["minutes"], N_BINS)
        a, b = ple_transform(tr["minutes"], ed), ple_transform(te["minutes"], ed)
    elif enc == "ple_log":
        ed = fit_ple_edges(tr["loginc"], N_BINS)
        a, b = ple_transform(tr["loginc"], ed), ple_transform(te["loginc"], ed)
    else:
        raise ValueError(enc)
    return (np.concatenate([a, ztr], 1).astype(np.float32),
            np.concatenate([b, zte], 1).astype(np.float32))


class PeriodicEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(K_FREQS) * SIGMA)
        self.lin = nn.Linear(2 * K_FREQS, PLR_OUT)

    def forward(self, x):
        zz = 2 * math.pi * x * self.freqs
        return torch.relu(self.lin(torch.cat([torch.sin(zz), torch.cos(zz)], -1)))


def make_head(model_type, in_dim):
    if model_type == "linear":
        return nn.Linear(in_dim, 1)
    return nn.Sequential(nn.Linear(in_dim, 48), nn.ReLU(), nn.Linear(48, 1))


class FixedEncModel(nn.Module):
    def __init__(self, model_type, in_dim):
        super().__init__()
        self.head = make_head(model_type, in_dim)

    def forward(self, X):
        return self.head(X).squeeze(-1)


class LearnedEncModel(nn.Module):  # periodic PLR on std-log col0 + z col1
    def __init__(self, model_type):
        super().__init__()
        self.emb = PeriodicEmbed()
        self.head = make_head(model_type, PLR_OUT + 1)

    def forward(self, X):
        return self.head(torch.cat([self.emb(X[:, 0:1]), X[:, 1:2]], -1)).squeeze(-1)


class ExpandEncModel(nn.Module):  # non-periodic learnable expansion (capacity match)
    def __init__(self, model_type):
        super().__init__()
        self.exp = nn.Sequential(nn.Linear(1, 2 * K_FREQS), nn.ReLU(), nn.Linear(2 * K_FREQS, PLR_OUT), nn.ReLU())
        self.head = make_head(model_type, PLR_OUT + 1)

    def forward(self, X):
        return self.head(torch.cat([self.exp(X[:, 0:1]), X[:, 1:2]], -1)).squeeze(-1)


def train_eval(regime, enc, model_type, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, te = gen(regime, N_TRAIN, rng), gen(regime, N_TEST, rng)
    wd = 0.0
    if enc in ("learned", "learned_reg"):
        Xtr = np.stack([_std(tr["loginc"], tr["loginc"]), tr["z"]], 1).astype(np.float32)
        Xte = np.stack([_std(te["loginc"], tr["loginc"]), te["z"]], 1).astype(np.float32)
        model = LearnedEncModel(model_type)
        wd = WEIGHT_DECAY_REG if enc == "learned_reg" else 0.0
    elif enc == "log_expand":
        Xtr = np.stack([_std(tr["loginc"], tr["loginc"]), tr["z"]], 1).astype(np.float32)
        Xte = np.stack([_std(te["loginc"], tr["loginc"]), te["z"]], 1).astype(np.float32)
        model = ExpandEncModel(model_type)
    else:
        Xtr, Xte = precompute(enc, tr, te)
        model = FixedEncModel(model_type, Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(tr["y"])
    model.train()
    last = 0.0
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(model(Xtr_t), ytr_t)
        loss.backward(); opt.step(); last = float(loss.detach())
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(Xte))).numpy()
    freqs = None
    if enc == "learned" and seed == 0:
        freqs = sorted(np.abs(model.emb.freqs.detach().numpy()).round(3).tolist())
    return {"ap": float(average_precision_score(te["y"], s)), "scores": s, "labels": te["y"],
            "train_loss": last, "freqs": freqs}


def z_only_floor(seed=0):
    out = {}
    for mt in MODELS:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        tr, te = gen("nonmono", N_TRAIN, rng), gen("nonmono", N_TEST, rng)
        Xtr, Xte = tr["z"][:, None].astype(np.float32), te["z"][:, None].astype(np.float32)
        model = FixedEncModel(mt, 1); opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss(); Xt, yt = torch.tensor(Xtr), torch.tensor(tr["y"])
        model.train()
        for _ in range(EPOCHS):
            opt.zero_grad(); lossf(model(Xt), yt).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(torch.tensor(Xte))).numpy()
        out[mt] = float(average_precision_score(te["y"], s))
    return out


def boot_ci(scores, labels, seed=0):  # SECONDARY diagnostic: within-seed evaluation noise (seed 0)
    rng = np.random.default_rng(seed); n = len(scores); v = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        v.append(average_precision_score(labels[idx], scores[idx]))
    return float(average_precision_score(labels, scores)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def seed0_paired_lift(sa, sb, labels, seed=0):  # SECONDARY diagnostic only
    rng = np.random.default_rng(seed); n = len(labels); v = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        v.append(average_precision_score(labels[idx], sa[idx]) - average_precision_score(labels[idx], sb[idx]))
    pt = float(average_precision_score(labels, sa) - average_precision_score(labels, sb))
    return {"lift": pt, "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}


def seed_level_lift(reg, mt, a, b, runs):  # M1: between-seed paired t CI (decision-relevant)
    da = np.array([runs[(reg, a, mt)][i]["ap"] for i in range(len(SEEDS))])
    db = np.array([runs[(reg, b, mt)][i]["ap"] for i in range(len(SEEDS))])
    d = da - db
    mean_d, sd_d, n = float(d.mean()), float(d.std(ddof=1)), len(d)
    if sd_d < 1e-12:
        return {"mean": mean_d, "sd": sd_d, "lo": mean_d, "hi": mean_d, "p": 1.0,
                "ci_excludes_zero": False, "n": n}
    half = float(stats.t.ppf(0.975, n - 1)) * sd_d / math.sqrt(n)
    p = float(stats.ttest_rel(da, db).pvalue)
    return {"mean": mean_d, "sd": sd_d, "lo": mean_d - half, "hi": mean_d + half, "p": p,
            "ci_excludes_zero": bool((mean_d - half) > 0 or (mean_d + half) < 0), "n": n}


def holm(pairs_with_p, alpha=0.05):  # m5: Holm-Bonferroni over a family of (key, p)
    order = sorted(pairs_with_p, key=lambda kv: kv[1])
    m = len(order); rej = {}
    for i, (k, p) in enumerate(order):
        rej[k] = p <= alpha / (m - i)
        if not rej[k]:
            for k2, _ in order[i + 1:]:
                rej[k2] = False
            break
    return rej


LIFTS = [
    ("nonmono", "linear", "ple_raw", "raw", "POSctrl ple_raw−raw"),
    ("nonmono", "linear", "ple_log", "raw", "POSctrl ple_log−raw"),
    ("nonmono", "linear", "learned", "raw", "POSctrl learned−raw"),
    ("nonmono", "linear", "learned", "log", "POSctrl learned−log"),
    ("nonmono", "mlp", "ple_raw", "log", "REALq ple_raw−log"),
    ("nonmono", "mlp", "ple_log", "log", "REALq ple_log−log"),
    ("nonmono", "mlp", "learned", "log", "REALq learned−log"),
    ("nonmono", "mlp", "learned_reg", "log", "REALq learned_reg−log"),
    ("nonmono", "mlp", "learned", "log_expand", "REALq learned−log_expand(matched)"),
    ("nonmono", "mlp", "learned", "learned_reg", "REALq learned−learned_reg(reg effect)"),
    ("mono", "mlp", "log", "raw", "NEGctrl log−raw"),
    ("mono", "mlp", "ple_raw", "log", "NEGctrl ple_raw−log"),
    ("mono", "mlp", "learned", "log", "NEGctrl learned−log"),
]
DECISION_FAMILY = ["ple_raw_minus_log", "ple_log_minus_log", "learned_minus_log",
                   "learned_reg_minus_log", "learned_minus_log_expand"]
EQUIV_MARGIN = 0.005  # M2: |effect| within this (CI inside ±margin) => practically equivalent to log


def main():
    runs = {(reg, enc, mt): [train_eval(reg, enc, mt, s) for s in SEEDS]
            for reg in REGIMES for enc in ENCODINGS for mt in MODELS}

    agg = []
    for (reg, enc, mt), rs in runs.items():
        aps = [r["ap"] for r in rs]
        pt, lo, hi = boot_ci(rs[0]["scores"], rs[0]["labels"])
        agg.append({"regime": reg, "encoding": enc, "model": mt,
                    "ap_seedmean": float(np.mean(aps)), "ap_seedstd": float(np.std(aps, ddof=1)),
                    "ap_seeds": [float(a) for a in aps],
                    "ap_seed0_boot": pt, "ap_seed0_lo": lo, "ap_seed0_hi": hi,
                    "train_loss_seedmean": float(np.mean([r["train_loss"] for r in rs]))})

    lifts = []
    for reg, mt, a, b, tag in LIFTS:
        sl = seed_level_lift(reg, mt, a, b, runs)
        r0a, r0b = runs[(reg, a, mt)][0], runs[(reg, b, mt)][0]
        s0 = seed0_paired_lift(r0a["scores"], r0b["scores"], r0a["labels"])
        sl.update({"regime": reg, "model": mt, "pair": f"{a}_minus_{b}", "tag": tag,
                   "seed0_lift": s0["lift"], "seed0_lo": s0["lo"], "seed0_hi": s0["hi"],
                   "equivalent_to_log": bool(abs(sl["lo"]) <= EQUIV_MARGIN and abs(sl["hi"]) <= EQUIV_MARGIN)})
        lifts.append(sl)

    fam = [(e["pair"], e["p"]) for e in lifts
           if e["regime"] == "nonmono" and e["model"] == "mlp" and e["pair"] in DECISION_FAMILY]
    holm_rej = holm(fam)
    for e in lifts:
        if e["regime"] == "nonmono" and e["model"] == "mlp" and e["pair"] in holm_rej:
            e["holm_significant"] = bool(holm_rej[e["pair"]])

    conv = {e: float(np.mean([runs[("nonmono", e, "mlp")][i]["train_loss"] for i in range(len(SEEDS))]))
            for e in ENCODINGS}
    payload = {"aggregate": agg, "lifts": lifts, "z_only_floor": z_only_floor(),
               "convergence_train_loss_nonmono_mlp_5seedmean": conv,
               "learned_freqs_seed0": runs[("nonmono", "learned", "mlp")][0]["freqs"],
               "base_rate_target": TARGET_POS, "seeds": len(SEEDS), "n_boot": N_BOOT,
               "equiv_margin": EQUIV_MARGIN, "weight_decay_reg": WEIGHT_DECAY_REG,
               "ci_note": "Decision lifts use SEED-LEVEL paired-t CIs (n=5); seed0_* fields are a "
                          "secondary within-seed bootstrap diagnostic only."}
    (HERE / "stats_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    print(f"\nΔt encoding experiment | base target {TARGET_POS} | {len(SEEDS)} seeds")
    for reg in REGIMES:
        print(f"\n[{reg}] {'enc':14s}{'linear':>9s}{'mlp':>9s}")
        for enc in ENCODINGS:
            a = next(r for r in agg if r["regime"] == reg and r["encoding"] == enc and r["model"] == "linear")
            m = next(r for r in agg if r["regime"] == reg and r["encoding"] == enc and r["model"] == "mlp")
            print(f"      {enc:14s}{a['ap_seedmean']:9.3f}{m['ap_seedmean']:9.3f}")
    print(f"\nz-only floor: {payload['z_only_floor']}")
    print(f"5-seed-mean train loss (nonmono,mlp): "
          + ", ".join(f"{e}={conv[e]:.4f}" for e in ENCODINGS))
    print("\nSEED-LEVEL paired lifts (decision-relevant; *=CI excl 0; H=Holm-sig; ~=equiv to log):")
    for e in lifts:
        flags = ("*" if e["ci_excludes_zero"] else "") + ("H" if e.get("holm_significant") else "") \
                + ("~" if e.get("equivalent_to_log") else "")
        print(f"  {e['tag']:34s} {e['mean']:+.3f} [{e['lo']:+.3f},{e['hi']:+.3f}] p={e['p']:.3f} "
              f"(seed0 {e['seed0_lift']:+.3f}[{e['seed0_lo']:+.3f},{e['seed0_hi']:+.3f}]){flags}")
    print("\nwrote stats_results.json")


if __name__ == "__main__":
    main()
