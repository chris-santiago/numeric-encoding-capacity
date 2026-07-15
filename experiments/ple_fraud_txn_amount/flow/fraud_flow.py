# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas>=2.0",
#   "scikit-learn>=1.4",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""Promoted Metaflow + Hydra flow — PLE on transaction amount, real IEEE-CIS fraud.

Single source of truth for the debated Step-6 methodology (HYPOTHESIS.md / EXPERIMENT_PLAN.md).

Shape:
  data        : real IEEE-CIS train_transaction.csv, temporal split by TransactionDT (fixed)
  foreach grain : model-init seed (data/split are seed-independent)
  arms        : logreg{raw,quadratic,ple_raw,ple_log,ple_placebo}, mlp{raw,ple_raw,ple_log}, hgb{raw,ple}
  metric      : PR-AUC primary; aux ROC-AUC, precision@{0.5%,1%}, recall@1%FPR
  CIs         : test-set bootstrap (N=1000) from seed-0 scores; paired lifts for key pairs
  diagnostics : fraud-vs-amount curve (precondition), PLE top-bin saturation (drift)

Debate resolutions realized: F1 logreg_quadratic, F2 ple_log arms, F5 hgb_ple (interaction gap),
F6 harmonized (no class weighting), F7 fixed-budget precision/recall, lever = ple_placebo on a
~monotone feature (C1).
"""

from __future__ import annotations

import pathlib
import sys

# --- repo-package import shim (Seam 1, verbatim core) ------------------------
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

_CONF_DIR = pathlib.Path(__file__).parent / "conf"
_STATS_OUT = pathlib.Path(__file__).resolve().parents[1] / "stats_results.json"


# ===========================================================================
# PLE
# ===========================================================================

def fit_ple_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    eps = 1e-9
    for t in range(1, edges.size):
        if edges[t] <= edges[t - 1]:
            edges[t] = edges[t - 1] + eps
    return edges


def ple_transform_1d(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = edges.size - 1
    out = np.empty((x.size, n_bins))
    for t in range(n_bins):
        out[:, t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def _encode_ple(s_tr, s_te, n_bins, fill):
    a_tr = s_tr.fillna(fill).to_numpy(float)
    a_te = s_te.fillna(fill).to_numpy(float)
    edges = fit_ple_edges(a_tr, n_bins)
    return ple_transform_1d(a_tr, edges), ple_transform_1d(a_te, edges)


def _encode_amount(s_tr, s_te, encoding, n_bins, kind, fill):
    a_tr = s_tr.fillna(fill).to_numpy(float)
    a_te = s_te.fillna(fill).to_numpy(float)
    log_tr = np.log1p(np.clip(a_tr, 0, None))
    log_te = np.log1p(np.clip(a_te, 0, None))
    if encoding == "ple_raw":
        edges = fit_ple_edges(a_tr, n_bins)
        return ple_transform_1d(a_tr, edges), ple_transform_1d(a_te, edges)
    if encoding == "ple_log":
        edges = fit_ple_edges(log_tr, n_bins)
        return ple_transform_1d(log_tr, edges), ple_transform_1d(log_te, edges)
    if encoding == "raw":
        x_tr, x_te = log_tr.reshape(-1, 1), log_te.reshape(-1, 1)
    elif encoding == "quadratic":
        x_tr = np.column_stack([log_tr, log_tr ** 2])
        x_te = np.column_stack([log_te, log_te ** 2])
    else:
        raise ValueError(f"Unknown amt_encoding {encoding!r}")
    if kind in ("logreg", "mlp"):
        sc = StandardScaler().fit(x_tr)
        x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    return x_tr, x_te


# ===========================================================================
# Seam 3 — components
# ===========================================================================

def make_data(data_spec: dict, data_axes: dict, seed: int) -> dict:
    """Load real IEEE-CIS (subset cols), temporal-sort, uniform-stride subsample, split.

    Data is seed-independent (the foreach grain is the model-init seed; the temporal
    split is fixed). Returns raw train/test DataFrames + label arrays.
    """
    spec = dict(data_spec)
    spec.update(data_axes)
    label = spec["label_col"]
    time_col = spec["time_col"]
    usecols = sorted(set(spec["num_features"]) | set(spec["cat_features"]) | {label, time_col})
    df = pd.read_csv(spec["path"], usecols=usecols)
    df = df.sort_values(time_col).reset_index(drop=True)
    n_rows = int(spec.get("n_rows", len(df)))
    if n_rows < len(df):
        idx = np.linspace(0, len(df) - 1, n_rows).astype(int)  # stride: keep full time range
        df = df.iloc[idx].reset_index(drop=True)
    n_train = int(spec["train_frac"] * len(df))
    tr = df.iloc[:n_train].reset_index(drop=True)
    te = df.iloc[n_train:].reset_index(drop=True)
    return {"train_df": tr, "test_df": te,
            "y_train": tr[label].to_numpy(int), "y_test": te[label].to_numpy(int),
            "amount_col": spec["amount_col"], "placebo_col": spec["placebo_col"],
            "num_features": list(spec["num_features"]), "cat_features": list(spec["cat_features"])}


def featurize(frames: dict, kind: str, spec: dict, cfg: dict):
    """Build the model matrix for an arm. Numerics standardized (linear/mlp) or raw (hgb);
    cats one-hot (linear/mlp) or ordinal (hgb); amount per amt_encoding; placebo PLE optional.
    Returns (Xtr, Xte, cat_indices)."""
    tr, te = frames["train_df"], frames["test_df"]
    amt, placebo = frames["amount_col"], frames["placebo_col"]
    n_bins = int(cfg["n_bins"])
    num, cats = list(frames["num_features"]), list(frames["cat_features"])
    encoding = spec["amt_encoding"]
    placebo_ple = bool(spec.get("placebo_ple", False))

    plain_num = [c for c in num if c != amt and not (placebo_ple and c == placebo)]
    med = tr[plain_num].median()
    Ntr = tr[plain_num].fillna(med).to_numpy(float)
    Nte = te[plain_num].fillna(med).to_numpy(float)
    if kind in ("logreg", "mlp"):
        sc = StandardScaler().fit(Ntr)
        Ntr, Nte = sc.transform(Ntr), sc.transform(Nte)

    blocks_tr, blocks_te = [Ntr], [Nte]
    a_tr, a_te = _encode_amount(tr[amt], te[amt], encoding, n_bins, kind, float(tr[amt].median()))
    blocks_tr.append(a_tr)
    blocks_te.append(a_te)
    if placebo_ple:
        p_tr, p_te = _encode_ple(tr[placebo], te[placebo], n_bins, float(tr[placebo].median()))
        blocks_tr.append(p_tr)
        blocks_te.append(p_te)

    n_numeric_cols = sum(b.shape[1] for b in blocks_tr)
    cat_indices: list = []
    if kind in ("logreg", "mlp"):
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        oh.fit(tr[cats].fillna("NA").astype(str))
        blocks_tr.append(oh.transform(tr[cats].fillna("NA").astype(str)))
        blocks_te.append(oh.transform(te[cats].fillna("NA").astype(str)))
    else:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        oe.fit(tr[cats].fillna("NA").astype(str))
        blocks_tr.append(oe.transform(tr[cats].fillna("NA").astype(str)))
        blocks_te.append(oe.transform(te[cats].fillna("NA").astype(str)))
        cat_indices = list(range(n_numeric_cols, n_numeric_cols + len(cats)))
    return np.hstack(blocks_tr), np.hstack(blocks_te), cat_indices


def build_model(model_spec: dict):
    """Construct an unfit estimator. Imbalance handling harmonized (no class weighting)."""
    kind = model_spec["kind"]
    if kind == "logreg":
        return LogisticRegression(max_iter=int(model_spec.get("logreg_max_iter", 2000)))
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(model_spec.get("mlp_hidden", [32])),
            alpha=float(model_spec.get("mlp_alpha", 1e-4)),
            max_iter=int(model_spec.get("mlp_max_iter", 80)),
            early_stopping=True,
            n_iter_no_change=int(model_spec.get("mlp_n_iter_no_change", 10)),
            random_state=int(model_spec.get("random_state", 0)),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            random_state=int(model_spec.get("random_state", 0)),
            categorical_features=model_spec.get("cat_indices") or None,
        )
    raise ValueError(f"Unknown model kind {kind!r}")


class TrainResult(dict):
    def __init__(self, model, scores, val_score, meta):
        super().__init__(model=model, scores=scores, val_score=val_score, meta=meta)


def train_arm(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    kind = method_spec["kind"]
    if not is_axis_agnostic_method(kind):  # raise-guard active (all agnostic here)
        raise ValueError(f"axis-dependent method {kind!r} unsupported (no training axis)")
    cfg = {**train_cfg, **method_spec}
    Xtr, Xte, cat_idx = featurize(data, kind, method_spec, train_cfg)
    model = build_model({**cfg, "random_state": seed, "cat_indices": cat_idx})
    model.fit(Xtr, data["y_train"])
    scores = model.predict_proba(Xte)[:, 1]
    meta = {"kind": kind, "encoding": method_spec["amt_encoding"], "in_dim": int(Xtr.shape[1])}
    if hasattr(model, "n_iter_"):
        meta["n_iter"] = int(np.ravel(model.n_iter_)[0])
    return TrainResult(model, scores, None, meta)


def metric(scores: np.ndarray, labels: np.ndarray, **cfg) -> float:
    """Primary metric: PR-AUC (average precision)."""
    return float(average_precision_score(labels, scores))


def _recall_at_fpr(scores, labels, fpr_target) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = int(np.searchsorted(fpr, fpr_target, side="right")) - 1
    return float(tpr[max(0, idx)])


def _aux_metrics(scores, labels, budget_fracs, fpr_target) -> dict:
    n = len(scores)
    order = np.argsort(-scores)
    out = {"roc_auc": float(roc_auc_score(labels, scores))}
    for f in budget_fracs:
        k = max(1, int(f * n))
        out[f"p_at_{f}"] = float(labels[order[:k]].sum() / k)
    out[f"recall_at_fpr_{fpr_target}"] = _recall_at_fpr(scores, labels, fpr_target)
    return out


# ===========================================================================
# Library generics — bootstrap CIs over the TEST SET
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
    point = float(average_precision_score(labels, scores))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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


# ===========================================================================
# Seam 4 — axis classification + foreach over seeds
# ===========================================================================

_AXIS_AGNOSTIC_KINDS = frozenset({"logreg", "mlp", "hgb"})
_AXIS_DEPENDENT_KINDS = frozenset()


def is_axis_agnostic_method(kind: str) -> bool:
    if kind in _AXIS_AGNOSTIC_KINDS:
        return True
    if kind in _AXIS_DEPENDENT_KINDS:
        return False
    raise ValueError(f"Unknown method.kind {kind!r}")


_LIFT_PAIRS = [
    ("logreg_ple_raw", "logreg_raw"),         # H-main (linear)
    ("mlp_ple_raw", "mlp_raw"),               # T-MLP
    ("mlp_ple_log", "mlp_raw"),               # T-MLP (log)
    ("logreg_ple_raw", "logreg_quadratic"),   # T-F1 attribution
    ("logreg_ple_log", "logreg_ple_raw"),     # T-F2 encoding space
    ("mlp_ple_log", "mlp_ple_raw"),           # T-F2 (neural)
    ("hgb_ple", "hgb_raw"),                   # T-F5 interaction gap
    ("logreg_ple_placebo", "logreg_raw"),     # T-lever placebo
]


def load_method_cfg(method_name: str) -> dict:
    import yaml

    return yaml.safe_load((_CONF_DIR / "method" / f"{method_name}.yaml").read_text())


def build_dataset_keys(experiment, exp_cfg, seeds, method_loader=load_method_cfg):
    """One dataset key per seed (no data axis; the dataset is seed-independent)."""
    methods = exp_cfg.get("methods", [])
    overrides = exp_cfg.get("method_overrides", {})
    combos = []
    for name in methods:
        spec = dict(method_loader(name))
        if name in overrides:
            spec.update(overrides[name])
        spec["_method_name"] = name
        if not is_axis_agnostic_method(spec["kind"]):
            raise ValueError(f"{name}: axis-dependent unsupported")
        combos.append({"method": name, "method_spec": spec, "axis_dep": False})
    return [{"experiment": experiment, "data_cell": {}, "seed": s, "combos": combos}
            for s in range(seeds)]


# ===========================================================================
# Analyses (pure consumers / deterministic data re-derivation)
# ===========================================================================

def filter_records(records, experiment):
    return [r for r in records if r.get("experiment") == experiment]


def analyze_auc_by_arm(records, experiment):
    recs = filter_records(records, experiment)
    if not recs:
        return []
    acc: dict = {}
    for r in recs:
        acc.setdefault(r["method"], []).append(r["pr_auc"])
    return [{"method": m, "pr_auc_mean": float(np.mean(v)), "pr_auc_std": float(np.std(v)),
             "n_seeds": len(v)} for m, v in sorted(acc.items())]


def analyze_fraud_curve(data_cfg, n_bins=10):
    """Precondition diagnostic: fraud-rate-vs-amount decile curve on the (regenerated) train set."""
    data = make_data(data_cfg, {}, 0)
    tr = data["train_df"]
    amt = data_cfg["amount_col"]
    x = tr[amt].to_numpy(float)
    y = data["y_train"]
    order = np.argsort(x)
    bins = np.array_split(order, n_bins)
    rates = [float(y[b].mean()) for b in bins]
    centers = np.arange(n_bins)
    mono = float(abs(np.corrcoef(centers, rates)[0, 1]))
    ush = float(np.corrcoef(np.abs(centers - (n_bins - 1) / 2), rates)[0, 1])
    return {"decile_fraud_rates": rates, "monotonic_abs_corr": mono, "ushape_corr": ush}


def analyze_saturation(data_cfg, n_bins):
    """F3 diagnostic: fraction of test amounts at/above the train PLE top edge (saturation)."""
    data = make_data(data_cfg, {}, 0)
    amt = data_cfg["amount_col"]
    a_tr = data["train_df"][amt].fillna(float(data["train_df"][amt].median())).to_numpy(float)
    a_te = data["test_df"][amt].fillna(float(data["train_df"][amt].median())).to_numpy(float)
    edges = fit_ple_edges(a_tr, int(n_bins))
    top_edge = edges[-2]  # entering the last bin
    return {"test_top_bin_saturation_rate": float((a_te >= top_edge).mean()),
            "train_top_bin_saturation_rate": float((a_tr >= top_edge).mean())}


# ===========================================================================
# Seam 2 — DAG
# ===========================================================================

try:
    from metaflow import Config, FlowSpec, card, step
    from metaflow.cards import Markdown

    _METAFLOW_AVAILABLE = True
except ImportError:
    _METAFLOW_AVAILABLE = False


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


if _METAFLOW_AVAILABLE:

    class FraudPLEFlow(FlowSpec):
        """PLE-on-amount fraud flow. foreach grain = init seed; CIs from test bootstrap."""

        cfg = Config("cfg", default=str(_CONF_DIR / "config.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            self.data_cfg = dict(cfg["data"])
            self.training_cfg = dict(cfg["training"])
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = exp_cfg["name"]
            self.determinism = exp_cfg.get("determinism", "order_independent")
            self.bootstrap_cfg = dict(cfg["bootstrap"])
            self.requests_scores = bool(exp_cfg.get("requests_scores", False))
            self.dataset_keys = build_dataset_keys(self.experiment_name, exp_cfg, cfg["seeds"])
            print(f"[start] experiment={self.experiment_name} seeds(foreach)={len(self.dataset_keys)} "
                  f"arms={len(self.dataset_keys[0]['combos'])}", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            dk = self.input
            seed = dk["seed"]
            tcfg = self.training_cfg
            budget_fracs = list(tcfg.get("budget_fracs", [0.01]))
            fpr_target = float(tcfg.get("fpr_target", 0.01))
            data = make_data(self.data_cfg, dk["data_cell"], seed)
            y_te = data["y_test"]
            records = []
            for combo in dk["combos"]:
                res = train_arm(combo["method_spec"], data, seed, tcfg)
                s = res["scores"]
                rec = {
                    "experiment": self.experiment_name, "seed": seed,
                    "method": combo["method"],
                    "pr_auc": metric(s, y_te),
                    "aux": _aux_metrics(s, y_te, budget_fracs, fpr_target),
                    "meta": res["meta"],
                }
                if seed == 0 and self.requests_scores:
                    rec["scores"] = s
                    rec["labels"] = y_te
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
            # seed-0 scores per arm for test-set bootstrap
            seed0 = {r["method"]: r for r in self.all_records
                     if r["seed"] == 0 and "scores" in r}

            by_arm: dict = {}
            for r in self.all_records:
                by_arm.setdefault(r["method"], []).append(r["pr_auc"])

            self.aggregate_results = []
            for method in sorted(by_arm):
                vals = np.array(by_arm[method])
                row = {"cell": {}, "method": method,
                       "pr_auc_seedmean": float(vals.mean()),
                       "pr_auc_seedstd": float(vals.std())}
                if method in seed0:
                    pt, lo, hi = bootstrap_prauc_ci(
                        np.asarray(seed0[method]["scores"]),
                        np.asarray(seed0[method]["labels"]), n_res, bseed)
                    row.update({"pr_auc_mean": pt, "pr_auc_lo": lo, "pr_auc_hi": hi})
                self.aggregate_results.append(row)

            self.lift_results = []
            for a, b in _LIFT_PAIRS:
                if a in seed0 and b in seed0:
                    entry = paired_prauc_lift(
                        np.asarray(seed0[a]["scores"]), np.asarray(seed0[b]["scores"]),
                        np.asarray(seed0[a]["labels"]), n_res, bseed)
                    entry.update({"cell": {}, "pair": f"{a}_minus_{b}", "n_seeds": 1})
                    self.lift_results.append(entry)
            print(f"[aggregate] {len(self.aggregate_results)} arms, {len(self.lift_results)} lifts",
                  flush=True)
            self.next(self.an_auc_by_arm, self.an_fraud_curve, self.an_saturation)

        @card
        @step
        def an_auc_by_arm(self):
            rows = analyze_auc_by_arm(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "branch": "auc_by_arm", "rows": rows}
            self.next(self.join_analyses)

        @card
        @step
        def an_fraud_curve(self):
            res = analyze_fraud_curve(self.data_cfg)
            self.an_result = {"experiment": self.experiment_name, "branch": "fraud_curve", "result": res}
            self.next(self.join_analyses)

        @card
        @step
        def an_saturation(self):
            res = analyze_saturation(self.data_cfg, self.training_cfg.get("n_bins", 24))
            self.an_result = {"experiment": self.experiment_name, "branch": "saturation", "result": res}
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            self.analyses = {
                (inp.an_result.get("experiment", ""), inp.an_result.get("branch", "")): inp.an_result
                for inp in inputs
            }
            first = inputs[0]
            self.aggregate_results = first.aggregate_results
            self.lift_results = first.lift_results
            self.experiment_name = first.experiment_name
            self.determinism = first.determinism
            self.next(self.report)

        @step
        def report(self):
            print("\n=== FraudPLEFlow analyses ===", flush=True)
            for a in self.analyses.values():
                print(f"  {a.get('branch')}: {a.get('rows') or a.get('result')}", flush=True)
            self.next(self.end)

        @step
        def end(self):
            import json

            payload = {
                "experiment": self.experiment_name, "determinism": self.determinism,
                "aggregate_results": self.aggregate_results, "lift_results": self.lift_results,
                "analyses": {f"{k[0]}::{k[1]}": v for k, v in self.analyses.items()},
            }
            _STATS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\n=== FraudPLEFlow complete: {self.experiment_name} ===", flush=True)
            print(f"arms={len(self.aggregate_results)} lifts={len(self.lift_results)} "
                  f"wrote {_STATS_OUT.name}", flush=True)

    if __name__ == "__main__":
        FraudPLEFlow()
