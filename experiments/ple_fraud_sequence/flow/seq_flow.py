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
"""Promoted Metaflow + Hydra flow — sequence model, amount-in-context (real account data).

Single source of truth for the debated Step-6 methodology (HYPOTHESIS.md / EXPERIMENT_PLAN.md).

  data        : ~/Dropbox/GitHub/demo/tmp/data account sequences (clean accountNumber key)
  foreach grain : (length L, init seed)   [length is a data axis; seed is model init]
  arms        : tab_last, tab_aggregate, seq_raw, seq_ple, seq_dev, seq_raw_shuffle
  metric      : PR-AUC; aux ROC-AUC, precision@{0.5%,1%}, recall@1%FPR
  CIs         : test-set bootstrap (N=1000) from seed-0 scores; paired lifts for key pairs
  preprocessing : CAUSAL-only — PLE edges + amount scaler fit on TRAIN-period amounts (F3)
  convergence : GRU early-stopped on validation PR-AUC (F4)
  lever       : seq_raw_shuffle permutes within-account prior order (destroys cross-time context)

Debate resolutions: F1 real headroom; F2 tab_aggregate; F3 causal-only; F4 early stop; F5 length axis.
"""

from __future__ import annotations

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
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


# ===========================================================================
# Encoders (causal: PLE edges + scaler fit on TRAIN-period amounts only)
# ===========================================================================

def fit_ple_edges(x, n_bins):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, edges.size):
        if edges[t] <= edges[t - 1]:
            edges[t] = edges[t - 1] + 1e-9
    return edges


def ple_transform(x, edges):  # x:(...) -> (..., n_bins)
    n_bins = edges.size - 1
    out = np.empty(x.shape + (n_bins,), dtype=np.float32)
    for t in range(n_bins):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def causal_deviation(logamt):  # (n,L) -> (n,L) deviation of each step vs prior steps in the row
    n, L = logamt.shape
    dev = np.zeros_like(logamt, dtype=np.float32)
    for t in range(1, L):
        m = logamt[:, :t].mean(axis=1)
        s = logamt[:, :t].std(axis=1) + 1e-6
        dev[:, t] = (logamt[:, t] - m) / s
    return dev


# ===========================================================================
# Seam 3 — components
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
    """Causal per-account sequences for length L.

    Targets come from the curated split files (train/valid/test.parq, ~7% fraud). Each target's
    sequence history is reconstructed from the FULL transactions.parq log (the account's last L
    transactions at or before the target's time, incl. the target; left edge-padded). The encoder
    pool (PLE edges + amount scaler) is the TRAIN-period history only (causal, F3). Seed-independent.
    """
    spec = dict(data_spec)
    spec.update(data_axes)
    L = int(spec.get("length", 32))
    d = pathlib.Path(spec["dir"])
    key, tcol = spec["key"], spec["time_col"]
    amt, label = spec["amount_col"], spec["label_col"]
    cnum = list(spec.get("context_numeric", []))
    cbool = list(spec.get("context_bool", []))

    # --- full history (for sequence context) ---
    tx = pd.read_parquet(d / spec["transactions_file"], columns=[key, tcol, amt] + cnum + cbool)
    tx = tx.sort_values([key, tcol]).reset_index(drop=True)
    tx_log = np.log1p(np.clip(tx[amt].to_numpy(float), 0, None)).astype(np.float32)
    tx_time = tx[tcol].to_numpy()
    tx_ctx = _build_ctx(tx, cnum, cbool)
    codes = pd.factorize(tx[key])[0]                      # grouped (tx sorted by key)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    acct_map = {a: (int(s), int(e)) for a, s, e in
                zip(tx[key].to_numpy()[starts], starts, ends)}

    out = {"_train_logamt_pool": tx_log[tx_time < np.datetime64(spec["valid_start"])]}
    rng = np.random.default_rng(20240501)
    files = {"train": (spec["train_file"], int(spec["n_targets_train"])),
             "valid": (spec["valid_file"], int(spec["n_targets_valid"])),
             "test": (spec["test_file"], int(spec["n_targets_test"]))}
    for sp, (fname, cap) in files.items():
        tgt = pd.read_parquet(d / fname, columns=[key, tcol, label])
        if len(tgt) > cap:
            tgt = tgt.iloc[np.sort(rng.choice(len(tgt), size=cap, replace=False))].reset_index(drop=True)
        n = len(tgt)
        c = tx_ctx.shape[1]
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
            amt_seq[r, L - k:] = tx_log[window]
            amt_seq[r, :L - k] = tx_log[window[0]]
            ctx_seq[r, L - k:] = tx_ctx[window]
            ctx_seq[r, :L - k] = tx_ctx[window[0]]
        out[sp] = {"amt": amt_seq[keep], "ctx": ctx_seq[keep], "y": y[keep]}
    return out


def featurize_seq(amt_seq, ctx_seq, enc, scaler, edges, shuffle, seed):
    """Per-step sequence features for a GRU arm -> (n, L, d). Causal encoders passed in."""
    n, L = amt_seq.shape
    if shuffle:                       # destroy cross-time order of PRIOR steps; keep last (target)
        g = np.random.default_rng(7000 + seed)
        amt_seq = amt_seq.copy(); ctx_seq = ctx_seq.copy()
        for r in range(n):
            perm = g.permutation(L - 1)
            amt_seq[r, :L - 1] = amt_seq[r, perm]
            ctx_seq[r, :L - 1] = ctx_seq[r, perm]
    mu, sd = scaler
    std_amt = ((amt_seq - mu) / sd).astype(np.float32)
    if enc == "raw":
        a = std_amt[..., None]
    elif enc == "ple":
        a = ple_transform(amt_seq, edges)
    elif enc == "dev":
        a = np.stack([std_amt, causal_deviation(amt_seq)], axis=-1)
    else:
        raise ValueError(f"Unknown enc {enc!r}")
    return np.concatenate([a, ctx_seq], axis=-1).astype(np.float32)


def featurize_tab(amt_seq, ctx_seq, aggregate, scaler):
    """Last-step (+ optional prior aggregates) features for a tabular arm -> (n, d)."""
    mu, sd = scaler
    last = (amt_seq[:, -1] - mu) / sd
    cols = [last[:, None]]
    if aggregate:
        prior = amt_seq[:, :-1]
        cols.append(((prior.mean(axis=1) - mu) / sd)[:, None])
        cols.append((prior.std(axis=1) / sd)[:, None])
    cols.append(ctx_seq[:, -1, :])
    return np.concatenate(cols, axis=1).astype(np.float32)


class GRUClf(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
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
    """Dispatch on kind. tab -> logreg; gru -> early-stopped torch GRU. Returns test scores + meta."""
    kind = method_spec["kind"]
    if not is_axis_agnostic_method(kind):
        raise ValueError(kind)
    enc = method_spec.get("enc", "raw")
    n_bins = int(train_cfg["n_bins"])
    pool = data["_train_logamt_pool"]
    scaler = (float(pool.mean()), float(pool.std() + 1e-6))      # causal: train-period only
    edges = fit_ple_edges(pool, n_bins)

    if kind == "tab":
        Xtr = featurize_tab(data["train"]["amt"], data["train"]["ctx"],
                            bool(method_spec.get("aggregate", False)), scaler)
        Xte = featurize_tab(data["test"]["amt"], data["test"]["ctx"],
                            bool(method_spec.get("aggregate", False)), scaler)
        model = LogisticRegression(max_iter=int(train_cfg.get("logreg_max_iter", 2000)))
        model.fit(Xtr, data["train"]["y"])
        s = model.predict_proba(Xte)[:, 1]
        return {"scores": s, "meta": {"kind": "tab", "enc": enc, "in_dim": int(Xtr.shape[1])}}

    # GRU
    shuffle = bool(method_spec.get("shuffle", False))
    Xtr = featurize_seq(data["train"]["amt"], data["train"]["ctx"], enc, scaler, edges, shuffle, seed)
    Xva = featurize_seq(data["valid"]["amt"], data["valid"]["ctx"], enc, scaler, edges, shuffle, seed)
    Xte = featurize_seq(data["test"]["amt"], data["test"]["ctx"], enc, scaler, edges, shuffle, seed)
    ytr, yva = data["train"]["y"], data["valid"]["y"]

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = GRUClf(Xtr.shape[2], int(train_cfg["gru_hidden"]))
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
            "meta": {"kind": "gru", "enc": enc, "shuffle": shuffle,
                     "in_dim": int(Xtr.shape[2]), "epochs_ran": ran, "best_val_ap": float(best_ap)}}


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
    ("seq_dev", "seq_raw"),            # H-main (a): does explicit deviation help?
    ("seq_ple", "seq_raw"),            # H-main (b): does PLE help?
    ("seq_raw", "tab_aggregate"),      # F2 precondition: sequence model beats aggregates?
    ("seq_raw", "seq_raw_shuffle"),    # lever: cross-time context value
    ("seq_raw", "tab_last"),           # sequence vs trivial baseline
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
# Analyses
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


def analyze_lever_check(records, experiment):
    """Lever readout: seq_raw vs seq_raw_shuffle vs tab_aggregate mean PR-AUC per length."""
    recs = filter_records(records, experiment)
    acc = {}
    for r in recs:
        if r["method"] in ("seq_raw", "seq_raw_shuffle", "tab_aggregate"):
            acc.setdefault((r["length"], r["method"]), []).append(r["pr_auc"])
    return [{"length": k[0], "method": k[1], "pr_auc_mean": float(np.mean(v))}
            for k, v in sorted(acc.items(), key=lambda kv: str(kv[0]))]


# ===========================================================================
# Seam 2 — DAG
# ===========================================================================

try:
    from metaflow import Config, FlowSpec, card, step
    from metaflow.cards import Markdown

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

    class SeqPLEFlow(FlowSpec):
        """Sequence amount-in-context flow. foreach grain = (length, seed); CIs from test bootstrap."""

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
            self.next(self.an_pr_auc_by_arm, self.an_lever_check)

        @card
        @step
        def an_pr_auc_by_arm(self):
            rows = analyze_pr_auc_by_arm(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "branch": "pr_auc_by_arm", "rows": rows}
            self.next(self.join_analyses)

        @card
        @step
        def an_lever_check(self):
            rows = analyze_lever_check(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "branch": "lever_check", "rows": rows}
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
            print("\n=== SeqPLEFlow analyses ===", flush=True)
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
            print(f"\n=== SeqPLEFlow complete: {self.experiment_name} === wrote {_STATS_OUT.name}",
                  flush=True)

    if __name__ == "__main__":
        SeqPLEFlow()
