# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8", "numpy", "scipy", "scikit-learn", "metaflow>=2.19", "hydra-core", "omegaconf", "pyyaml",
# ]
# ///
"""Consolidated numeric-encoding-capacity flow (cycles 1-8).

SSOT of the final debated methodology: architecture (static/gru/mlp) x risk-shape condition x
multiplicity (K) x seed, all encoding arms, all controls, every arm calibrated. See HYPOTHESIS.md and
CLAIMS.md. Domain seams are plain module functions (unit-testable via bare `import flow`).
"""
from __future__ import annotations

import os
# Force single-threaded BLAS BEFORE numpy/torch import so CPU reductions are bit-reproducible
# (thread-count-dependent reduction order is what broke the order_independent determinism check).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"          # (no first-party pkg; shim kept harmless)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_CONF_DIR = pathlib.Path(__file__).parent / "conf"              # __file__-anchored (lint: not cwd-relative)

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss, brier_score_loss

DEVICE = "cpu"          # canonical reproducible run: CPU + single-thread => single_worker determinism holds
                        # (MPS/CUDA GRU kernels are not bit-reproducible; would break the determinism gate)
_SPLIT_ID = {"train": 1, "val": 2, "test": 3}
BAND_OFFSET = {"sharp_mode": 0.0, "sharp_off": 1.5}


def _solve_intercept(logit, target, iters=60):
    """Bisection for b s.t. mean(sigmoid(logit + b)) == target (realized prevalence = base_rate)."""
    lo, hi = -25.0, 25.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (1.0 / (1.0 + np.exp(-(logit + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ============================================================ seam 1: make_data
def _risk(cond, sc, offset, bsig):
    if cond == "log_linear":
        return sc
    if cond == "monotone_curved":
        return sc ** 3
    if cond == "smooth_nonmono":
        return sc ** 2
    if cond in ("sharp_mode", "sharp_off"):
        return np.exp(-((sc - offset) ** 2) / (2 * bsig ** 2))
    raise ValueError(f"unknown condition {cond!r}")


def make_data(data_spec: dict, data_axes: dict, seed: int) -> dict:
    """K iid lognormal per-step features; additive risk (equal weights); condition sets the shape.
    Standardization + risk-normalization fit on TRAIN and reused for all splits (identical label DGP)."""
    cond = data_axes.get("condition", data_spec["condition"])
    K = int(data_axes.get("K", data_spec["K"]))
    L = int(data_spec["seq_len"]); sig = float(data_spec["feat_sigma"]); w = float(data_spec["weight"])
    base = float(data_spec["base_rate"]); decay = float(data_spec["decay"]); bsig = float(data_spec["band_sigma"])
    offset = BAND_OFFSET.get(cond, 0.0)
    n = {"train": int(data_spec["n_train"]), "val": int(data_spec["n_val"]), "test": int(data_spec["n_test"])}
    W = (decay ** np.arange(L)[::-1]); W = (W / W.sum()).astype(np.float32)
    rng = np.random.default_rng(seed)
    raw = {sp: np.exp(rng.normal(0.0, sig, size=(n[sp], L, K))).astype(np.float32) for sp in n}
    # log-standardization fit on TRAIN
    ltr = np.log(raw["train"]); mu = ltr.mean(axis=(0, 1)); sd = ltr.std(axis=(0, 1)) + 1e-9
    # risk-contribution normalization fit on TRAIN (so the label DGP is identical across splits)
    sctr = (ltr - mu) / sd
    gmu = np.array([_risk(cond, sctr[..., j], offset, bsig).mean() for j in range(K)])
    gsd = np.array([_risk(cond, sctr[..., j], offset, bsig).std() + 1e-9 for j in range(K)])
    scd = {sp: (np.log(raw[sp]) - mu) / sd for sp in n}
    aggd = {}                                            # per-split per-feature recency-aggregated risk
    for sp in n:
        A = np.zeros((n[sp], K))
        for j in range(K):
            g = (_risk(cond, scd[sp][..., j], offset, bsig) - gmu[j]) / gsd[j]
            A[:, j] = (g * W[None, :]).sum(1)
        aggd[sp] = A
    am = aggd["train"].mean(0); asd = aggd["train"].std(0) + 1e-9   # aggregate-norm fit on TRAIN
    def _logit(sp):
        return (w * (aggd[sp] - am) / asd).sum(1)
    b = _solve_intercept(_logit("train"), base)                     # intercept fit on TRAIN (mean prev = base)
    rmu = raw["train"].reshape(-1, K).mean(0); rsd = raw["train"].reshape(-1, K).std(0) + 1e-9  # raw stats: TRAIN
    out = {}
    for sp in n:
        logit = _logit(sp) + b                          # single fixed DGP across splits
        prob = 1.0 / (1.0 + np.exp(-logit))
        yrng = np.random.default_rng(seed * 101 + _SPLIT_ID[sp])
        y = (yrng.random(n[sp]) < prob).astype(np.float32)
        out[sp] = {"raw": raw[sp], "sc": scd[sp].astype(np.float32), "y": y, "logit": logit.astype(np.float32)}
    out["meta"] = {"mu": mu, "sd": sd, "W": W, "K": K, "L": L, "raw_mu": rmu, "raw_sd": rsd}
    return out


# ============================================================ encoders (shared coordinate = log-space sc)
def _fit_ple_edges(sc_tr, nbins):
    E = []
    for j in range(sc_tr.shape[-1]):
        e = np.quantile(sc_tr[..., j], np.linspace(0, 1, nbins + 1))
        for t in range(1, e.size):
            if e[t] <= e[t - 1]:
                e[t] = e[t - 1] + 1e-9
        E.append(e)
    return E


def encode(enc, data_split, meta, edges, nbins):
    """Return per-step model input (n,L,d). log/raw/ple are data transforms; projection/dense feed the
    2-arg log scalar and the MODEL applies the per-step nonlinearity (shared log coordinate)."""
    sc = data_split["sc"]                                    # standardized log features (n,L,K)
    if enc in ("log", "projection", "dense"):
        return sc.astype(np.float32)                         # projection/dense embed inside the model
    if enc == "raw":
        return ((data_split["raw"] - meta["raw_mu"]) / meta["raw_sd"]).astype(np.float32)  # TRAIN stats
    if enc == "ple":
        K = sc.shape[-1]; cols = []
        for j in range(K):
            e = edges[j]; x = sc[..., j]
            r = np.empty(sc.shape[:2] + (nbins,), dtype=np.float32)
            for t in range(nbins):
                r[..., t] = np.clip((x - e[t]) / (e[t + 1] - e[t]), 0.0, 1.0)
            cols.append(r)
        return np.concatenate(cols, axis=-1).astype(np.float32)
    raise ValueError(f"unknown enc {enc!r}")


# ============================================================ seam 2: build_model
# NOTE: `static` (affine-read) is intentionally a deterministic sklearn LogisticRegression on the
# recency-pooled encoding (see _train_static) — NOT a torch module — so it has no build_model branch.
class _PerStepMLP(nn.Module):
    """free per-step nonlinearity, no recurrence: per-step MLP (`depth` hidden ReLU layers) ->
    recency-pool -> linear head. Depth/width are the C1 knob: this arm must have GENUINE capacity to
    rebuild any 1-D transform from the scalar, else 'no arm beats log' is an undercapacity artifact, not
    redundancy (review M2). It even receives the true recency weights W, so the test is maximally fair."""
    def __init__(self, in_dim, hidden, W, depth=2):
        super().__init__()
        self.register_buffer("W", torch.tensor(W))
        layers, d = [], in_dim
        for _ in range(max(1, depth)):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.proj = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head((self.proj(x) * self.W[None, :, None]).sum(1)).squeeze(-1)


class _GRU(nn.Module):
    """affine-input GRU. enc 'projection' embeds each feature via Linear(1->d)->ReLU (shared coord);
    'dense' applies a joint per-step Linear->ReLU; else reads the encoding affinely."""
    def __init__(self, in_dim, hidden, mode, K, embed_dim, dense_h):
        super().__init__()
        self.mode, self.K = mode, K
        if mode == "projection":
            self.emb = nn.ModuleList([nn.Sequential(nn.Linear(1, embed_dim), nn.ReLU()) for _ in range(K)])
            gin = K * embed_dim
        elif mode == "dense":
            self.proj = nn.Sequential(nn.Linear(in_dim, dense_h), nn.ReLU()); gin = dense_h
        else:
            gin = in_dim
        self.gru = nn.GRU(gin, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        if self.mode == "projection":
            x = torch.cat([self.emb[j](x[..., j:j + 1]) for j in range(self.K)], dim=-1)
        elif self.mode == "dense":
            x = self.proj(x)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(model_spec: dict) -> nn.Module:
    kind, enc, in_dim = model_spec["kind"], model_spec["enc"], int(model_spec["in_dim"])
    W = model_spec["W"]; tc = model_spec["train_cfg"]
    if kind == "mlp":
        return _PerStepMLP(in_dim, int(tc["mlp_hidden"]), W, depth=int(tc["mlp_depth"]))
    if kind == "gru":
        return _GRU(in_dim, int(tc["gru_hidden"]), enc, int(model_spec["K"]),
                    int(tc["embed_dim"]), int(tc["dense_h"]))
    raise ValueError(f"build_model: unknown kind {kind!r}")


# ============================================================ calibration (every arm, temperature scaling)
def _fit_temperature(val_logits, val_y, iters):
    t = torch.zeros(1, requires_grad=True)                  # log-temperature
    z = torch.tensor(val_logits, dtype=torch.float32); yv = torch.tensor(val_y, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=iters)
    lossf = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); loss = lossf(z / torch.exp(t), yv); loss.backward(); return loss
    opt.step(closure)
    T = float(torch.exp(t).item())
    return T if np.isfinite(T) and T > 1e-3 else 1.0        # guard: separable val can push LBFGS to T->0/NaN


def _ece(probs, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for i in range(bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < bins - 1 else probs <= 1.0)
        if m.any():
            e += m.mean() * abs(probs[m].mean() - y[m].mean())
    return float(e)


# ============================================================ seam 3: train_arm (registry)
class TrainResult(dict):
    def __init__(self, model, scores, val_score, cal):
        super().__init__(model=model, scores=scores, val_score=val_score, cal=cal)
        self.model, self.scores, self.val_score, self.cal = model, scores, val_score, cal


def _torch_train(model, Xtr, ytr, Xva, yva, seed, tc):
    torch.manual_seed(seed); torch.set_num_threads(1)      # single-thread => run-to-run identical on CPU
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["lr"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=float(tc["lr_factor"]),
                                                       patience=int(tc["lr_patience"]))
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    batch, epochs, patience = int(tc["batch"]), int(tc["epochs"]), int(tc["patience"])
    # finite fallback: initial (untrained) weights, so a diverged run can NEVER leak NaN into scores
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best, bad = -1.0, 0

    def _logits(X):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(model(torch.tensor(X[i:i + 4096]).to(DEVICE)).cpu().numpy())
        return np.nan_to_num(np.concatenate(out), nan=0.0, posinf=30.0, neginf=-30.0)
    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(ytr), generator=g)         # reshuffle every epoch — SYMMETRIC across arms
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            xb, yb = Xt[idx].to(DEVICE), yt[idx].to(DEVICE)
            opt.zero_grad(); lossf(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # prevent RNN gradient explosion -> NaN
            opt.step()
        ap = average_precision_score(yva, 1 / (1 + np.exp(-_logits(Xva))))
        if not np.isfinite(ap):
            ap = -1.0                                            # diverged epoch -> worst; keep best_state
        sched.step(ap)
        if ap > best + 1e-5:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, _logits, best


def _calibrate(val_logits, val_y, test_logits, iters):
    T = _fit_temperature(np.nan_to_num(val_logits, nan=0.0, posinf=30.0, neginf=-30.0), val_y, iters)
    probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits / T, -30, 30)))
    return np.nan_to_num(probs, nan=0.5), T


def _prep_torch(method_spec, data, tc):
    enc, kind = method_spec["enc"], method_spec["kind"]
    meta = data["meta"]; nbins = int(tc["n_bins"])
    edges = _fit_ple_edges(data["train"]["sc"], nbins) if enc == "ple" else None
    X = {sp: encode(enc, data[sp], meta, edges, nbins) for sp in ("train", "val", "test")}
    in_dim = X["train"].shape[2]
    spec = {"kind": kind, "enc": enc, "in_dim": in_dim, "K": meta["K"], "W": meta["W"], "train_cfg": tc}
    return X, spec


def _train_torch(method_spec, data, seed, tc):
    X, spec = _prep_torch(method_spec, data, tc)
    model = build_model(spec)
    model, logits_of, val_ap = _torch_train(model, X["train"], data["train"]["y"], X["val"], data["val"]["y"], seed, tc)
    test_logits, val_logits = logits_of(X["test"]), logits_of(X["val"])
    raw_scores = 1 / (1 + np.exp(-test_logits))
    cal_probs, T = _calibrate(val_logits, data["val"]["y"], test_logits, int(tc["calib_max_iter"]))
    return TrainResult(None, raw_scores, val_ap,
                       {"probs": cal_probs, "T": T, "ece_raw": _ece(raw_scores, data["test"]["y"]),
                        "ece_cal": _ece(cal_probs, data["test"]["y"])})


def _train_static(method_spec, data, seed, tc):
    """affine-read pooled-linear via deterministic logistic regression (positive control needs determinism)."""
    enc = method_spec["enc"]; meta = data["meta"]; nbins = int(tc["n_bins"])
    edges = _fit_ple_edges(data["train"]["sc"], nbins) if enc == "ple" else None
    W = meta["W"]
    def pool(sp):
        Xe = encode(enc, data[sp], meta, edges, nbins)
        return (Xe * W[None, :, None]).sum(1)
    clf = LogisticRegression(max_iter=int(tc["logreg_max_iter"])).fit(pool("train"), data["train"]["y"])
    raw = clf.predict_proba(pool("test"))[:, 1]
    val_logits = clf.decision_function(pool("val")); test_logits = clf.decision_function(pool("test"))
    cal, T = _calibrate(val_logits, data["val"]["y"], test_logits, int(tc["calib_max_iter"]))
    return TrainResult(None, raw, average_precision_score(data["val"]["y"], clf.predict_proba(pool("val"))[:, 1]),
                       {"probs": cal, "T": T, "ece_raw": _ece(raw, data["test"]["y"]),
                        "ece_cal": _ece(cal, data["test"]["y"])})


def _train_oracle(method_spec, data, seed, tc):
    """ceiling: rank by the TRUE per-sequence log-odds. Calibrated on val logodds."""
    raw = 1 / (1 + np.exp(-data["test"]["logit"]))
    cal, T = _calibrate(data["val"]["logit"], data["val"]["y"], data["test"]["logit"], int(tc["calib_max_iter"]))
    return TrainResult(None, raw, average_precision_score(data["val"]["y"], 1 / (1 + np.exp(-data["val"]["logit"]))),
                       {"probs": cal, "T": T, "ece_raw": _ece(raw, data["test"]["y"]),
                        "ece_cal": _ece(cal, data["test"]["y"])})


def _train_tabular(method_spec, data, seed, tc):
    """strong precondition comparator: GBM on last/mean/std/EWMA aggregates of the log features."""
    meta = data["meta"]; W = meta["W"]
    def feats(sp):
        sc = data[sp]["sc"]                                  # (n,L,K)
        return np.concatenate([sc[:, -1, :], sc.mean(1), sc.std(1), (sc * W[None, :, None]).sum(1)], axis=1)
    clf = HistGradientBoostingClassifier(max_iter=int(tc["gbm_max_iter"]), max_depth=4,
                                         learning_rate=0.08, random_state=0).fit(feats("train"), data["train"]["y"])
    raw = clf.predict_proba(feats("test"))[:, 1]
    val_p = clf.predict_proba(feats("val"))[:, 1]
    vl = np.log(np.clip(val_p, 1e-6, 1 - 1e-6) / (1 - np.clip(val_p, 1e-6, 1 - 1e-6)))
    tl = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6)))
    cal, T = _calibrate(vl, data["val"]["y"], tl, int(tc["calib_max_iter"]))
    return TrainResult(None, raw, average_precision_score(data["val"]["y"], val_p),
                       {"probs": cal, "T": T, "ece_raw": _ece(raw, data["test"]["y"]),
                        "ece_cal": _ece(cal, data["test"]["y"])})


TRAIN_REGISTRY = {"static": _train_static, "gru": _train_torch, "mlp": _train_torch,
                  "oracle": _train_oracle, "tabular": _train_tabular}


def train_arm(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    kind = method_spec["kind"]
    try:
        fn = TRAIN_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown method.kind {kind!r}")
    return fn(method_spec, data, seed, train_cfg)


def is_axis_agnostic_method(kind: str) -> bool:
    if kind not in TRAIN_REGISTRY:
        raise ValueError(f"is_axis_agnostic_method: unknown kind {kind!r}")
    return True                                              # no training axis in this experiment


# ============================================================ seam 4: metric (+ aux, calibrated)
def metric(scores: np.ndarray, labels: np.ndarray, **cfg) -> float:
    return float(average_precision_score(labels, scores))   # PR-AUC — primary


def _aux_metrics(scores, labels, cal):
    p = np.clip(cal["probs"], 1e-7, 1 - 1e-7)
    return {"prauc": float(average_precision_score(labels, scores)),
            "logloss_cal": float(log_loss(labels, p)), "brier_cal": float(brier_score_loss(labels, p)),
            "ece_raw": cal["ece_raw"], "ece_cal": cal["ece_cal"], "temperature": cal["T"]}


def bootstrap_ci(values, n_resamples: int = 1000, seed: int = 0):
    v = np.asarray(values, float); rng = np.random.default_rng(seed)
    if len(v) < 2:
        return float(v.mean()), float(v.mean()), float(v.mean())
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_resamples)]
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_t(diffs):
    """Seed-level paired-t: (mean, 95% lo, 95% hi, two-sided p). Feeds the control gate + the lift table."""
    from scipy import stats
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return float(d.mean()), float(d.mean()), float(d.mean()), 1.0
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(len(d)) + 1e-12
    tc = float(stats.t.ppf(0.975, df=len(d) - 1))
    p = float(2 * stats.t.sf(abs(m / se), df=len(d) - 1))
    return float(m), float(m - tc * se), float(m + tc * se), p


# ============================================================ conf + cell helpers
def load_method_cfg(method_name: str) -> dict:
    import yaml
    with open(_CONF_DIR / "method" / f"{method_name}.yaml") as f:
        return yaml.safe_load(f)


def _cell_product(axes: dict) -> list:
    import itertools
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*[axes[k] for k in keys])]


def _data_cell_of(cell: dict, data_axes: list) -> dict:
    return {k: cell[k] for k in data_axes}


def _cell_key(cell: dict) -> tuple:
    return tuple(sorted((k, (tuple(v) if isinstance(v, list) else v)) for k, v in cell.items()))


def build_dataset_keys(experiment: str, exp_cfg: dict, seeds: int) -> list:
    """foreach grain = (data_cell + seed): one dataset per branch; all methods train on it in-process."""
    data_axes = exp_cfg["data_axes"]
    data_cells = {_cell_key(_data_cell_of(c, data_axes)): _data_cell_of(c, data_axes)
                  for c in _cell_product(exp_cfg["axes"])}
    keys = []
    for dc in data_cells.values():
        for seed in range(seeds):
            keys.append({"experiment": experiment, "data_cell": dc, "seed": seed})
    return keys


def _merge_training(method_cfg: dict, train_cfg: dict) -> dict:
    spec = dict(train_cfg)          # training authoritative for shared knobs
    spec.update(method_cfg)         # method wins for keys it sets (kind, enc)
    return spec


# ============================================================ analysis helpers (pure readers)
def filter_records(records, experiment):
    return [r for r in records if r.get("experiment") == experiment]


def _arch_of(method):
    return method.split("_")[0]


def _prauc_by(records, condition, K, method, seed):
    for r in records:
        if (r["cell"].get("condition") == condition and r["cell"].get("K") == K
                and r["method"] == method and r["seed"] == seed):
            return r["test"]["prauc"]
    return None


def deficit_corrected_lifts(records, seeds):
    """(arm - log)_condition - (arm - log)_log_linear, seed-paired, per (arch, K, condition, arm-enc).
    This is THE estimand: it nets PLE's structural deficit measured at the log-adequate condition.
    Holm-Bonferroni is applied across the reported family (the single-hypothesis gate is uncorrected)."""
    archs = {"static": ["ple", "raw"], "gru": ["ple", "projection", "dense", "raw"], "mlp": ["ple"]}
    conds = ["monotone_curved", "smooth_nonmono", "sharp_mode", "sharp_off"]
    out = []
    for arch, encs in archs.items():
        logm = f"{arch}_log"
        for K in (1, 6):
            for cond in conds:
                for enc in encs:
                    arm = f"{arch}_{enc}"
                    diffs, raw_d, def_d = [], [], []
                    for s in range(seeds):
                        a_c, l_c = _prauc_by(records, cond, K, arm, s), _prauc_by(records, cond, K, logm, s)
                        a_0, l_0 = _prauc_by(records, "log_linear", K, arm, s), _prauc_by(records, "log_linear", K, logm, s)
                        if None in (a_c, l_c, a_0, l_0):
                            continue
                        raw_d.append(a_c - l_c)              # DEPLOYMENT gap: arm vs log ON the condition
                        def_d.append(a_0 - l_0)              # structural deficit: arm vs log on log_linear
                        diffs.append((a_c - l_c) - (a_0 - l_0))
                    if len(diffs) >= 2:
                        m, lo, hi, p = paired_t(diffs)      # seed-level paired-t (HYPOTHESIS §metric)
                        rm, rlo, rhi, _ = paired_t(raw_d)   # raw-gap CI (review #1: report beside dc_lift)
                        out.append({"cell": {"arch": arch, "K": K, "condition": cond, "arm": arm},  # SSOT key
                                    "lift_mean": round(m, 4), "lift_lo": round(lo, 4), "lift_hi": round(hi, 4),
                                    "n_seeds": len(diffs),   # standard run-output contract fields
                                    "arch": arch, "K": K, "condition": cond, "arm": arm,  # report convenience
                                    "dc_lift": round(m, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                    # decomposition dc_lift = raw_gap - deficit (deficit usually <=0) — makes the
                                    # add-back explicit so weak/broken arms cannot masquerade as levers (review M1/M2/M8)
                                    "raw_gap": round(rm, 4), "raw_gap_lo": round(rlo, 4), "raw_gap_hi": round(rhi, 4),
                                    "raw_gap_excludes_zero": bool(rlo > 0 or rhi < 0),
                                    "deficit": round(float(np.mean(def_d)), 4),
                                    "ci_excludes_zero": bool(lo > 0 or hi < 0), "p": p, "n": len(diffs)})
    # Holm-Bonferroni across the reported family (gate remains a single uncorrected pre-registered test)
    order = sorted(range(len(out)), key=lambda i: out[i]["p"])
    m_tot = len(out)
    for rank, i in enumerate(order):
        out[i]["holm_p"] = round(min(1.0, out[i]["p"] * (m_tot - rank)), 4)
        out[i]["holm_significant"] = bool(out[i]["holm_p"] < 0.05 and out[i]["dc_lift"] != 0)
    return out


def control_gate(dc_lifts):
    """TWO POSITIVE controls, both halt if they fail — the estimand must detect the ROBUST lever (PLE on the
    sharp non-monotone condition, deficit-corrected, K=6, CI excludes zero) in BOTH vehicles:
      (a) static affine-read (deterministic logreg) — validates the estimand/label-DGP path;
      (b) the trained GRU (`gru_ple`)              — validates the torch training/convergence path, so a
          silently under-trained GRU cannot depress every GRU reading while (a) still fires (review M4).
    Only if BOTH see the strongest known lever are the GRU curvature/smooth readings interpretable.
    Curvature (a weak lever) and the multivariate control remain REPORTED findings, not halt conditions."""
    def _find(arch, cond, arm, K=6):
        for r in dc_lifts:
            if r["arch"] == arch and r["condition"] == cond and r["arm"] == arm and r["K"] == K:
                return r
        return None
    sharp = _find("static", "sharp_mode", "static_ple")
    pos_ok = bool(sharp and sharp["ci_lo"] > 0)             # static-path instrument-validity lever
    gru_sharp = _find("gru", "sharp_mode", "gru_ple")
    gru_ok = bool(gru_sharp and gru_sharp["ci_lo"] > 0)     # GRU-path (training/convergence) validity lever
    both_ok = pos_ok and gru_ok
    curv = _find("static", "monotone_curved", "static_ple")
    curv1 = _find("static", "monotone_curved", "static_ple", K=1)
    return {"positive_control_fires": pos_ok, "gru_positive_control_fires": gru_ok,
            "instrument_ok": both_ok,
            "verdict": "TRUSTWORTHY" if both_ok else "BLIND-INSTRUMENT — halt; readings not interpretable",
            # reported findings (not gated):
            "curvature_lever_detected": bool(curv and curv["ci_lo"] > 0),
            "multivariate_control_ok": bool(curv1 and not curv1["ci_excludes_zero"] and curv and curv["ci_lo"] > 0),
            "static_sharp": sharp, "gru_sharp": gru_sharp, "static_curvature": curv, "static_curvature_K1": curv1}


# ============================================================ Hydra parser + FlowSpec
def _hydra_parser(text: str) -> dict:
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    raw = yaml.safe_load(text) or {}
    overrides = raw.pop("hydra_overrides", [])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR.resolve()), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


try:
    from metaflow import Config, FlowSpec, card, step
    from metaflow.cards import Markdown

    class ConsolidatedFlow(FlowSpec):
        """Encoding-capacity consolidated flow. foreach grain = (condition,K,seed); all arms per cell."""
        cfg = Config("cfg", default=str(_CONF_DIR / "_flowcfg.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            exp = self.cfg["experiment"]
            self.experiment_name = exp["name"]
            self.dataset_keys = build_dataset_keys(exp["name"], exp, int(self.cfg["seeds"]))
            self.determinism = exp["determinism"]
            print(f"[start] {self.experiment_name}: {len(self.dataset_keys)} data-branches x "
                  f"{len(exp['methods'])} methods", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            key = self.input
            exp = self.cfg["experiment"]; tc = self.cfg["training"]
            data = make_data(self.cfg["data"], key["data_cell"], key["seed"])
            recs = []
            for mname in exp["methods"]:
                mcfg = load_method_cfg(mname)
                spec = _merge_training(mcfg, tc)
                res = train_arm(spec, data, key["seed"], tc)
                aux = _aux_metrics(res.scores, data["test"]["y"], res.cal)
                recs.append({"experiment": key["experiment"], "cell": key["data_cell"],
                             "seed": key["seed"], "method": mname, "config": {"kind": mcfg["kind"], "enc": mcfg["enc"]},
                             "test": aux, "diagnostics": {"ece_cal": res.cal["ece_cal"], "temperature": res.cal["T"]}})
            self.records = recs
            self.next(self.join)

        @step
        def join(self, inputs):
            self.all_records = [r for inp in inputs for r in inp.records]
            self.merge_artifacts(inputs, include=["cfg", "experiment_name", "determinism"])
            print(f"[join] records: {len(self.all_records)}", flush=True)
            self.next(self.aggregate)

        @step
        def aggregate(self):
            seeds = int(self.cfg["seeds"])
            recs = filter_records(self.all_records, self.experiment_name)
            agg = {}
            for r in recs:
                k = (_cell_key(r["cell"]), r["method"])
                agg.setdefault(k, []).append(r["test"]["prauc"])
            self.aggregate_results = [{"cell": dict(ck), "method": mth, "prauc_mean": round(float(np.mean(v)), 4)}
                                      for (ck, mth), v in agg.items()]
            self.lift_results = deficit_corrected_lifts(recs, seeds)
            self.next(self.an_lifts, self.an_controls, self.an_calibration)

        @card
        @step
        def an_lifts(self):
            self.an_result = {"experiment": self.experiment_name, "rows": self.lift_results}
            self.next(self.join_analyses)

        @card
        @step
        def an_controls(self):
            self.an_result = {"experiment": self.experiment_name, "result": control_gate(self.lift_results)}
            self.next(self.join_analyses)

        @card
        @step
        def an_calibration(self):
            recs = filter_records(self.all_records, self.experiment_name)
            bym = {}
            for r in recs:
                bym.setdefault(r["method"], []).append((r["test"]["ece_raw"], r["test"]["ece_cal"]))
            self.an_result = {"experiment": self.experiment_name,
                              "rows": [{"method": m, "ece_raw": round(float(np.mean([x[0] for x in v])), 4),
                                        "ece_cal": round(float(np.mean([x[1] for x in v])), 4)}
                                       for m, v in sorted(bym.items())]}
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            self.analyses = {inp._current_step: inp.an_result for inp in inputs}
            self.merge_artifacts(inputs, include=["cfg", "experiment_name", "determinism",
                                                  "lift_results", "aggregate_results"])
            self.next(self.report)

        @step
        def report(self):
            gate = self.analyses["an_controls"]["result"]
            lines = [f"# Consolidated encoding-capacity report — {self.experiment_name}",
                     f"\n**Instrument verdict:** {gate['verdict']}",
                     f"- static positive control (static PLE fires on sharp non-monotone): {gate['positive_control_fires']}",
                     f"- GRU-path positive control (gru_ple fires on sharp non-monotone): {gate['gru_positive_control_fires']}",
                     f"- curvature lever detected (static, reported): {gate['curvature_lever_detected']}",
                     f"- multivariate control (curvature K=1~0, K=6>0): {gate['multivariate_control_ok']}",
                     "\n## Deficit-corrected lifts — `gap` (uncorrected arm-vs-log gap) beside dc_lift (dc = gap - deficit)"]
            # `gap` = the arm-minus-log gap ON the condition, BEFORE deficit-correction. NB: this is a quantity,
            # NOT the `raw` ENCODER arm (static_raw/gru_raw) — those appear in the `arm` column. (fidelity CONCERN)
            lines.append("(H = Holm-significant; * = dc CI excludes 0; g = gap CI excludes 0 = real uncorrected gap)")
            for r in sorted(self.lift_results, key=lambda x: (x["arch"], x["condition"], x["arm"], x["K"])):
                flag = "H" if r.get("holm_significant") else ("*" if r["ci_excludes_zero"] else " ")
                gflag = "g" if r.get("raw_gap_excludes_zero") else " "
                lines.append(f"{flag}{gflag} {r['arch']:>6} K={r['K']} {r['condition']:>15} {r['arm']:>16}: "
                             f"dc {r['dc_lift']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] | "
                             f"gap {r['raw_gap']:+.3f} [{r['raw_gap_lo']:+.3f},{r['raw_gap_hi']:+.3f}] | "
                             f"deficit {r['deficit']:+.3f} | holm_p={r.get('holm_p')}")
            self.report_md = "\n".join(lines)
            print(self.report_md, flush=True)
            self.next(self.end)

        @step
        def end(self):
            gate = self.analyses["an_controls"]["result"]
            print(f"[end] determinism={self.determinism} | instrument={gate['verdict']} "
                  f"(static={gate['positive_control_fires']}, gru={gate['gru_positive_control_fires']})", flush=True)
            if not gate["instrument_ok"]:
                raise RuntimeError(
                    f"POSITIVE CONTROL FAILED — {gate['verdict']} "
                    f"(static={gate['positive_control_fires']}, gru={gate['gru_positive_control_fires']}). "
                    "Report + lift artifacts persisted for post-mortem, but readings are NOT interpretable; "
                    "run marked FAILED.")

    if __name__ == "__main__":
        ConsolidatedFlow()

except ImportError:
    pass
