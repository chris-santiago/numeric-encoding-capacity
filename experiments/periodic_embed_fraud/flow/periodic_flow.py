# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas>=2.0",
#   "pyarrow",
#   "torch>=2.2",
#   "scikit-learn>=1.4",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""Promoted Metaflow + Hydra flow — learned periodic embeddings on fraud-GRU time features (cycle 4).

Single source of truth for the debated Step-6 methodology (HYPOTHESIS.md / EXPERIMENT_PLAN.md).

  data         : ~/Dropbox/GitHub/demo/tmp/data account sequences (clean accountNumber key)
  time features: hour-of-day, day-of-week (cyclic), log inter-transaction-minutes (non-periodic)
  arms         : base_raw, cyc_sincos, cyc_periodic, dt_periodic, all_periodic, tab_logreg
                 (arms differ ONLY in the per-step time-encoding block; amount raw-log + context fixed)
  encoders     : raw scalar | fixed sin/cos (periods 24,7) | learned periodic PLR (in-model, trainable)
  foreach grain: (length L, init seed)   [length is a data axis; seed is model init]
  metric       : PR-AUC; aux ROC-AUC, precision@{0.5%,1%}, recall@1%FPR
  CIs          : test-set bootstrap (N=1000) from seed-0 scores; paired lifts for key pairs
  preprocessing: CAUSAL-only — amount + dt scalers fit on TRAIN-period history (cyclic bounded /24,/7)
  convergence  : GRU early-stopped on validation PR-AUC

Debate resolutions: F1 scope; F2 H2 (dt_periodic-cyc_sincos); F3 GRU-transfer of H1; F4 capacity-iso.
"""

from __future__ import annotations

import math
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"   # shell shim (no first-party imports here)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

_CONF_DIR = pathlib.Path(__file__).parent / "conf"
_STATS_OUT = pathlib.Path(__file__).resolve().parents[1] / "stats_results.json"

# raw-feature column layout produced by featurize_model (per step): [hour_norm, dow_norm, dt_std, amt_std, ctx...]
_HOUR, _DOW, _DT, _AMT, _CTX0 = 0, 1, 2, 3, 4


# ===========================================================================
# Seam 3 — components: causal data, in-model encoders, train/eval
# ===========================================================================

def _build_ctx(df, cnum, cbool):
    ctx = np.zeros((len(df), len(cnum) + len(cbool)), dtype=np.float32)
    for j, c in enumerate(cnum):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        ctx[:, j] = np.nan_to_num(v, nan=float(np.nanmedian(v)))
    for j, c in enumerate(cbool):
        ctx[:, len(cnum) + j] = df[c].astype(float).to_numpy()
    return ctx


def make_data(data_spec, data_axes, seed):
    """Causal per-account sequences for length L with time features.

    Targets come from the curated split files (~7% fraud). Each target's history is reconstructed
    from the FULL transactions.parq log (its account's last L txns at or before the target time,
    incl. the target; left edge-padded). Time features per step: hour-of-day, day-of-week,
    log inter-transaction-minutes. Amount (raw log) and context are held fixed across arms. The
    amount/dt scalers are fit on TRAIN-period history only (causal). Seed-independent.
    """
    spec = dict(data_spec)
    spec.update(data_axes)
    L = int(spec.get("length", 32))
    d = pathlib.Path(spec["dir"])
    key, tcol = spec["key"], spec["time_col"]
    amt, hourc, label = spec["amount_col"], spec["hour_col"], spec["label_col"]
    cnum = list(spec.get("context_numeric", []))
    cbool = list(spec.get("context_bool", []))

    tx = pd.read_parquet(d / spec["transactions_file"],
                         columns=[key, tcol, amt, hourc] + cnum + cbool)
    tx = tx.sort_values([key, tcol]).reset_index(drop=True)
    tx_amt_log = np.log1p(np.clip(tx[amt].to_numpy(float), 0, None)).astype(np.float32)
    tx_hour = tx[hourc].to_numpy(float).astype(np.float32)
    tx_dow = tx[tcol].dt.dayofweek.to_numpy(float).astype(np.float32)
    tx_time = tx[tcol].to_numpy()
    tx_ctx = _build_ctx(tx, cnum, cbool)

    codes = pd.factorize(tx[key])[0]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    acct_map = {a: (int(s), int(e)) for a, s, e in
                zip(tx[key].to_numpy()[starts], starts, ends)}

    # inter-transaction time (minutes), per-account reset at each account's first row
    dt_min = np.zeros(len(tx), dtype=np.float64)
    dt_min[1:] = np.diff(tx_time) / np.timedelta64(1, "m")
    dt_min[starts] = 0.0
    tx_dt_log = np.log1p(np.clip(dt_min, 0, None)).astype(np.float32)

    train_mask = tx_time < np.datetime64(spec["valid_start"])
    out = {"_amt_pool": tx_amt_log[train_mask], "_dt_pool": tx_dt_log[train_mask]}

    rng = np.random.default_rng(20240501)
    files = {"train": (spec["train_file"], int(spec["n_targets_train"])),
             "valid": (spec["valid_file"], int(spec["n_targets_valid"])),
             "test": (spec["test_file"], int(spec["n_targets_test"]))}
    c = tx_ctx.shape[1]
    for sp, (fname, cap) in files.items():
        tgt = pd.read_parquet(d / fname, columns=[key, tcol, label])
        if len(tgt) > cap:
            tgt = tgt.iloc[np.sort(rng.choice(len(tgt), size=cap, replace=False))].reset_index(drop=True)
        n = len(tgt)
        hour_seq = np.zeros((n, L), dtype=np.float32)
        dow_seq = np.zeros((n, L), dtype=np.float32)
        dt_seq = np.zeros((n, L), dtype=np.float32)
        amt_seq = np.zeros((n, L), dtype=np.float32)
        ctx_seq = np.zeros((n, L, c), dtype=np.float32)
        y = tgt[label].to_numpy(int)
        tacct, ttime = tgt[key].to_numpy(), tgt[tcol].to_numpy()
        keep = np.ones(n, dtype=bool)
        for r in range(n):
            se = acct_map.get(tacct[r])
            if se is None:
                keep[r] = False
                continue
            s, e = se
            pos = int(np.searchsorted(tx_time[s:e], ttime[r], side="right")) - 1
            if pos < 0:
                keep[r] = False
                continue
            gi = s + pos
            window = np.arange(max(s, gi - L + 1), gi + 1)
            k = len(window)
            for dst, src in ((hour_seq, tx_hour), (dow_seq, tx_dow),
                             (dt_seq, tx_dt_log), (amt_seq, tx_amt_log)):
                dst[r, L - k:] = src[window]
                dst[r, :L - k] = src[window[0]]
            ctx_seq[r, L - k:] = tx_ctx[window]
            ctx_seq[r, :L - k] = tx_ctx[window[0]]
        out[sp] = {"hour": hour_seq[keep], "dow": dow_seq[keep], "dt": dt_seq[keep],
                   "amt": amt_seq[keep], "ctx": ctx_seq[keep], "y": y[keep]}
    return out


def featurize_model(split, amt_scaler, dt_scaler):
    """Raw per-step model input (n, L, 4+c): [hour/24, dow/7, std log-dt, std log-amt, ctx].

    The per-arm time encoding is applied INSIDE the model (so periodic frequencies are trainable);
    this tensor is identical across arms, which is what makes the comparison isolate the encoding.
    """
    amu, asd = amt_scaler
    dmu, dsd = dt_scaler
    hour_norm = (split["hour"] / 24.0)[..., None]
    dow_norm = (split["dow"] / 7.0)[..., None]
    dt_std = ((split["dt"] - dmu) / dsd)[..., None]
    amt_std = ((split["amt"] - amu) / asd)[..., None]
    return np.concatenate([hour_norm, dow_norm, dt_std, amt_std, split["ctx"]],
                          axis=-1).astype(np.float32)


def featurize_tab(split, amt_scaler, dt_scaler):
    """Last-step raw features for the trivial logreg baseline -> (n, 4+c)."""
    return featurize_model(split, amt_scaler, dt_scaler)[:, -1, :]


class PeriodicEmbed(nn.Module):
    """Gorishniy PLR: k learned frequencies -> sin/cos -> Linear -> ReLU. Applied per scalar feature."""

    def __init__(self, k, sigma, out):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(k) * sigma)   # init N(0, sigma^2)
        self.lin = nn.Linear(2 * k, out)

    def forward(self, x):                                    # x: (n, L, 1)
        z = 2 * math.pi * x * self.freqs                    # (n, L, k) broadcast
        per = torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return torch.relu(self.lin(per))


def _cyc_out_dim(enc, plr_out):
    return {"raw": 1, "sincos": 2, "periodic": plr_out}[enc]


def _dt_out_dim(enc, plr_out):
    return {"raw": 1, "periodic": plr_out}[enc]


class TimeGRUClf(nn.Module):
    """GRU whose per-step time block is encoded per-arm (raw / fixed sin/cos / learned periodic).

    Amount (col _AMT) passes through raw; context (cols _CTX0:) passes through fixed.
    """

    def __init__(self, n_ctx, cyc_enc, dt_enc, hidden, k, sigma, plr_out):
        super().__init__()
        self.cyc_enc, self.dt_enc = cyc_enc, dt_enc
        if cyc_enc not in ("raw", "sincos", "periodic"):
            raise ValueError(f"Unknown cyc_enc {cyc_enc!r}")
        if dt_enc not in ("raw", "periodic"):
            raise ValueError(f"Unknown dt_enc {dt_enc!r}")
        if cyc_enc == "periodic":
            self.hour_emb = PeriodicEmbed(k, sigma, plr_out)
            self.dow_emb = PeriodicEmbed(k, sigma, plr_out)
        if dt_enc == "periodic":
            self.dt_emb = PeriodicEmbed(k, sigma, plr_out)
        in_dim = 2 * _cyc_out_dim(cyc_enc, plr_out) + _dt_out_dim(dt_enc, plr_out) + 1 + n_ctx
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def _enc_cyc(self, x, emb):
        if self.cyc_enc == "raw":
            return x
        if self.cyc_enc == "sincos":
            z = 2 * math.pi * x
            return torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
        return emb(x)

    def _enc_dt(self, x):
        if self.dt_enc == "raw":
            return x
        return self.dt_emb(x)

    def forward(self, X):
        hour = self._enc_cyc(X[..., _HOUR:_HOUR + 1], getattr(self, "hour_emb", None))
        dow = self._enc_cyc(X[..., _DOW:_DOW + 1], getattr(self, "dow_emb", None))
        dt = self._enc_dt(X[..., _DT:_DT + 1])
        rest = X[..., _AMT:]                                  # amount (raw) + context (fixed)
        z = torch.cat([hour, dow, dt, rest], dim=-1)
        out, _ = self.gru(z)
        return self.head(out[:, -1, :]).squeeze(1)


def _gru_scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def is_axis_agnostic_method(kind):
    if kind in ("tab", "gru"):
        return True
    raise ValueError(f"Unknown method.kind {kind!r}")


def train_arm(method_spec, data, seed, train_cfg):
    """Dispatch on kind. tab -> logreg on last step; gru -> early-stopped torch GRU with per-arm encoder."""
    kind = method_spec["kind"]
    if not is_axis_agnostic_method(kind):
        raise ValueError(kind)
    amt_pool, dt_pool = data["_amt_pool"], data["_dt_pool"]
    amt_scaler = (float(amt_pool.mean()), float(amt_pool.std() + 1e-6))   # causal: train-period only
    dt_scaler = (float(dt_pool.mean()), float(dt_pool.std() + 1e-6))

    if kind == "tab":
        Xtr = featurize_tab(data["train"], amt_scaler, dt_scaler)
        Xte = featurize_tab(data["test"], amt_scaler, dt_scaler)
        model = LogisticRegression(max_iter=int(train_cfg.get("logreg_max_iter", 2000)))
        model.fit(Xtr, data["train"]["y"])
        s = model.predict_proba(Xte)[:, 1]
        return {"scores": s, "meta": {"kind": "tab", "in_dim": int(Xtr.shape[1])}}

    cyc_enc, dt_enc = method_spec["cyc_enc"], method_spec["dt_enc"]
    Xtr = featurize_model(data["train"], amt_scaler, dt_scaler)
    Xva = featurize_model(data["valid"], amt_scaler, dt_scaler)
    Xte = featurize_model(data["test"], amt_scaler, dt_scaler)
    ytr, yva = data["train"]["y"], data["valid"]["y"]
    n_ctx = Xtr.shape[2] - _CTX0

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = TimeGRUClf(n_ctx, cyc_enc, dt_enc, int(train_cfg["gru_hidden"]),
                       int(train_cfg["plr_k"]), float(train_cfg["plr_sigma"]),
                       int(train_cfg["plr_out"]))
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["gru_lr"]))
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    batch = int(train_cfg["gru_batch"])
    patience = int(train_cfg["gru_patience"])
    best_ap, best_state, bad, ran = -1.0, None, 0, 0
    for epoch in range(int(train_cfg["gru_epochs"])):
        model.train()
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        ran = epoch + 1
        ap = average_precision_score(yva, _gru_scores(model, Xva))
        if ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"scores": _gru_scores(model, Xte),
            "meta": {"kind": "gru", "cyc_enc": cyc_enc, "dt_enc": dt_enc,
                     "in_dim": int(model.gru.input_size), "epochs_ran": ran,
                     "best_val_ap": float(best_ap)}}


def metric(scores, labels, **cfg):
    return float(average_precision_score(labels, scores))


def _recall_at_fpr(scores, labels, fpr_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.searchsorted(fpr, fpr_target, side="right")) - 1
    return float(tpr[max(0, idx)])


def _aux_metrics(scores, labels, budget_fracs, fpr_target):
    n = len(scores)
    order = np.argsort(-scores)
    out = {"roc_auc": float(roc_auc_score(labels, scores))}
    for f in budget_fracs:
        k = max(1, int(f * n))
        out[f"p_at_{f}"] = float(labels[order[:k]].sum() / k)
    out[f"recall_at_fpr_{fpr_target}"] = _recall_at_fpr(scores, labels, fpr_target)
    return out


# ===========================================================================
# Library generics — test-set bootstrap CIs
# ===========================================================================

def bootstrap_prauc_ci(scores, labels, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        vals.append(average_precision_score(labels[idx], scores[idx]))
    return (float(average_precision_score(labels, scores)),
            float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def paired_prauc_lift(sa, sb, labels, n_resamples, seed):
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        vals.append(average_precision_score(labels[idx], sa[idx])
                    - average_precision_score(labels[idx], sb[idx]))
    point = float(average_precision_score(labels, sa) - average_precision_score(labels, sb))
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return {"lift_mean": point, "lift_lo": lo, "lift_hi": hi,
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


_LIFT_PAIRS = [
    ("cyc_periodic", "cyc_sincos"),    # H1: learned periodic vs fixed sin/cos (cyclic, known period)
    ("dt_periodic", "cyc_sincos"),     # H2: learned periodic on dt vs raw dt (cyc held = sincos)
    ("cyc_sincos", "base_raw"),        # ENC: does any cyclic encoding beat raw under the GRU?
    ("all_periodic", "cyc_sincos"),    # ALL: combined periodic vs reference encoding
    ("cyc_sincos", "tab_logreg"),      # BASE: reference-encoding GRU vs trivial baseline
    ("cyc_periodic", "tab_logreg"),    # BASE: periodic GRU vs trivial baseline
]


def load_method_cfg(name):
    import yaml

    return yaml.safe_load((_CONF_DIR / "method" / f"{name}.yaml").read_text())


def build_dataset_keys(experiment, exp_cfg, seeds, method_loader=load_method_cfg):
    """foreach grain = (length, seed). One dataset per (length, seed); all arms train on it."""
    lengths = exp_cfg.get("axes", {}).get("length", [32])
    methods = exp_cfg.get("methods", [])
    combos = []
    for name in methods:
        spec = dict(method_loader(name))
        spec["_method_name"] = name
        is_axis_agnostic_method(spec["kind"])
        combos.append({"method": name, "method_spec": spec})
    keys = []
    for L in lengths:
        for s in range(seeds):
            keys.append({"experiment": experiment, "data_cell": {"length": L},
                         "seed": s, "combos": combos})
    return keys


# ===========================================================================
# Analyses (pure readers of train records)
# ===========================================================================

def filter_records(records, experiment):
    return [r for r in records if r.get("experiment") == experiment]


def analyze_pr_auc_by_arm(records, experiment):
    recs = filter_records(records, experiment)
    acc = {}
    for r in recs:
        acc.setdefault((r["length"], r["method"]), []).append(r["pr_auc"])
    return [{"length": k[0], "method": k[1], "pr_auc_mean": float(np.mean(v)),
             "pr_auc_std": float(np.std(v)), "n_seeds": len(v)}
            for k, v in sorted(acc.items(), key=lambda kv: str(kv[0]))]


def analyze_encoding_check(records, experiment):
    """Encoding readout: per-length mean PR-AUC for each time-encoding arm."""
    recs = filter_records(records, experiment)
    arms = ("base_raw", "cyc_sincos", "cyc_periodic", "dt_periodic", "all_periodic")
    acc = {}
    for r in recs:
        if r["method"] in arms:
            acc.setdefault((r["length"], r["method"]), []).append(r["pr_auc"])
    return [{"length": k[0], "method": k[1], "pr_auc_mean": float(np.mean(v))}
            for k, v in sorted(acc.items(), key=lambda kv: str(kv[0]))]


# ===========================================================================
# Seam 2 — DAG
# ===========================================================================

try:
    from metaflow import Config, FlowSpec, card, step
    from metaflow.cards import Markdown  # noqa: F401

    _METAFLOW_AVAILABLE = True
except ImportError:
    _METAFLOW_AVAILABLE = False


def _hydra_parser(text):
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


if _METAFLOW_AVAILABLE:

    class PeriodicFlow(FlowSpec):
        """Periodic time-encoding flow. foreach grain = (length, seed); CIs from test bootstrap."""

        cfg = Config("cfg", default=str(_CONF_DIR / "config.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            self.data_cfg = dict(cfg["data"])
            self.training_cfg = dict(cfg["training"])
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = exp_cfg["name"]
            self.determinism = exp_cfg.get("determinism", "single_worker")
            self.bootstrap_cfg = dict(cfg["bootstrap"])
            self.requests_scores = bool(exp_cfg.get("requests_scores", False))
            self.dataset_keys = build_dataset_keys(self.experiment_name, exp_cfg, cfg["seeds"])
            print(f"[start] {self.experiment_name} foreach(length,seed)={len(self.dataset_keys)} "
                  f"arms={len(self.dataset_keys[0]['combos'])}", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            dk = self.input
            seed = dk["seed"]
            L = dk["data_cell"]["length"]
            tcfg = self.training_cfg
            bf = list(tcfg.get("budget_fracs", [0.01]))
            fpr = float(tcfg.get("fpr_target", 0.01))
            data = make_data(self.data_cfg, dk["data_cell"], seed)
            yte = data["test"]["y"]
            records = []
            for combo in dk["combos"]:
                res = train_arm(combo["method_spec"], data, seed, tcfg)
                s = res["scores"]
                rec = {"experiment": self.experiment_name, "length": L, "seed": seed,
                       "method": combo["method"], "pr_auc": metric(s, yte),
                       "aux": _aux_metrics(s, yte, bf, fpr), "meta": res["meta"]}
                if seed == 0 and self.requests_scores:
                    rec["scores"] = s
                    rec["labels"] = yte
                records.append(rec)
            self.records = records
            self.next(self.join)

        @step
        def join(self, inputs):
            self.all_records = [r for inp in inputs for r in inp.records]
            self.merge_artifacts(inputs, include=[
                "data_cfg", "training_cfg", "experiment_cfg", "experiment_name",
                "bootstrap_cfg", "determinism", "requests_scores"])
            print(f"[join] records: {len(self.all_records)}", flush=True)
            self.next(self.aggregate)

        @step
        def aggregate(self):
            n_res = int(self.bootstrap_cfg.get("n_resamples", 1000))
            bseed = int(self.bootstrap_cfg.get("seed", 0))
            seed0 = {(r["length"], r["method"]): r for r in self.all_records
                     if r["seed"] == 0 and "scores" in r}
            by = {}
            for r in self.all_records:
                by.setdefault((r["length"], r["method"]), []).append(r["pr_auc"])
            self.aggregate_results = []
            for (L, m), vals in sorted(by.items(), key=lambda kv: str(kv[0])):
                row = {"cell": {"length": L}, "method": m,
                       "pr_auc_seedmean": float(np.mean(vals)), "pr_auc_seedstd": float(np.std(vals))}
                if (L, m) in seed0:
                    pt, lo, hi = bootstrap_prauc_ci(np.asarray(seed0[(L, m)]["scores"]),
                                                    np.asarray(seed0[(L, m)]["labels"]), n_res, bseed)
                    row.update({"pr_auc_mean": pt, "pr_auc_lo": lo, "pr_auc_hi": hi})
                self.aggregate_results.append(row)
            self.lift_results = []
            lengths = sorted({L for (L, _m) in seed0})
            for L in lengths:
                for a, b in _LIFT_PAIRS:
                    if (L, a) in seed0 and (L, b) in seed0:
                        e = paired_prauc_lift(np.asarray(seed0[(L, a)]["scores"]),
                                              np.asarray(seed0[(L, b)]["scores"]),
                                              np.asarray(seed0[(L, a)]["labels"]), n_res, bseed)
                        e.update({"cell": {"length": L}, "pair": f"{a}_minus_{b}"})
                        self.lift_results.append(e)
            print(f"[aggregate] {len(self.aggregate_results)} arm-cells, {len(self.lift_results)} lifts",
                  flush=True)
            self.next(self.an_pr_auc_by_arm, self.an_encoding_check)

        @card
        @step
        def an_pr_auc_by_arm(self):
            rows = analyze_pr_auc_by_arm(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "branch": "pr_auc_by_arm", "rows": rows}
            self.next(self.join_analyses)

        @card
        @step
        def an_encoding_check(self):
            rows = analyze_encoding_check(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "branch": "encoding_check", "rows": rows}
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            self.analyses = {(inp.an_result.get("experiment", ""), inp.an_result.get("branch", "")):
                             inp.an_result for inp in inputs}
            first = inputs[0]
            self.aggregate_results = first.aggregate_results
            self.lift_results = first.lift_results
            self.experiment_name = first.experiment_name
            self.determinism = first.determinism
            self.next(self.report)

        @step
        def report(self):
            print("\n=== PeriodicFlow analyses ===", flush=True)
            for a in self.analyses.values():
                print(f"  {a.get('branch')}: {a.get('rows')}", flush=True)
            self.next(self.end)

        @step
        def end(self):
            import json

            payload = {"experiment": self.experiment_name, "determinism": self.determinism,
                       "aggregate_results": self.aggregate_results, "lift_results": self.lift_results,
                       "analyses": {f"{k[0]}::{k[1]}": v for k, v in self.analyses.items()}}
            _STATS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\n=== PeriodicFlow complete: {self.experiment_name} === wrote {_STATS_OUT.name}",
                  flush=True)

    if __name__ == "__main__":
        PeriodicFlow()
