# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "scikit-learn>=1.4",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""Promoted Metaflow + Hydra flow for the PLE-vs-raw-numeric investigation.

Single source of truth for the FINAL, debated methodology (HYPOTHESIS.md). It does
NOT carry forward the throwaway PoC's numbers — the debate redirected those; the
journal records superseded assumptions.

Experiment shape:
  data axis   : target in {nonmono, linear}   (linear is the falsification control)
  methods     : logreg_raw, logreg_ple, mlp_raw, mlp_ple, mlp_raw_wide, mlp_rff
  metric      : AUC-ROC (primary), average precision (aux)   [HYPOTHESIS.md]
  stats       : bootstrap 95% CIs (N=1000), seed-paired lifts, 10 seeds

Tests realized (EXPERIMENT_PLAN.md):
  T-main : mlp_ple vs mlp_raw on nonmono           (the hypothesis)
  T1     : logreg_ple vs logreg_raw                (mechanism: MLP-specific vs general)
  T2     : mlp_ple vs mlp_raw_wide / mlp_rff       (capacity / basis control)
  T3     : n_iter_ + converged per MLP arm         (convergence; an_convergence)
  T4     : Ridge R^2 on latent logit, PLE vs raw   (linearization; an_linearization)
  T5     : the mlp_ple-mlp_raw lift on nonmono vs linear cells (falsification lever)

T3 operationalization note: rather than a 2-point max_iter sweep {300,1000}, every
MLP arm trains at a generous cap (max_iter=1000) with early stopping and records
n_iter_ + a `converged` flag. Training to convergence is strictly stronger than the
debated sweep: if raw-MLP converges below the cap, it was never iteration-limited,
so the gap at convergence is the real gap. This satisfies the F3 settling test's
intent (is raw-MLP optimization-limited?).
"""

from __future__ import annotations

import itertools
import pathlib
import sys

# --- repo-package import shim (Seam 1, verbatim core) ------------------------
# Anchored to __file__ (not CWD) so `uv run` script mode resolves any first-party
# package. This investigation is self-contained (no src package), so the inserted
# path simply does not exist — harmless, kept for standard conformance.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# --- Hydra -> Metaflow Config wiring (Seam 1, verbatim core) -----------------
# __file__-anchored conf dir; a CWD-relative default breaks under `uv run`.
_CONF_DIR = pathlib.Path(__file__).parent / "conf"
# Where the flow drops a convenience stats dump for the Step 7 conclusions/figures
# (the Metaflow datastore remains the SSOT; this file is an archive copy).
_STATS_OUT = pathlib.Path(__file__).resolve().parents[1] / "stats_results.json"


# ===========================================================================
# Encoders (fit on TRAIN only — leakage prevention is per-arm)
# ===========================================================================

def fit_ple_edges(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile PLE bin edges per feature, fit on training data. Shape (d, n_bins+1)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(X, qs, axis=0).T
    eps = 1e-6
    for j in range(edges.shape[0]):  # guard collided quantiles -> zero-width bins
        for t in range(1, edges.shape[1]):
            if edges[j, t] <= edges[j, t - 1]:
                edges[j, t] = edges[j, t - 1] + eps
    return edges


def ple_transform(X: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Piecewise-linear encoding: per bin t, clip((x-lo)/(hi-lo), 0, 1). (n, d*n_bins)."""
    n, d = X.shape
    n_bins = edges.shape[1] - 1
    out = np.empty((n, d, n_bins), dtype=np.float64)
    for t in range(n_bins):
        lo, hi = edges[:, t], edges[:, t + 1]
        out[:, :, t] = np.clip((X - lo) / (hi - lo), 0.0, 1.0)
    return out.reshape(n, d * n_bins)


def _encode_rff(Xtr, Xte, rff_dim, seed):
    """Random Fourier features (RBF, length-scale 1) on standardized inputs.

    A capacity/dimension control: matches PLE's input dimensionality with a global
    periodic basis that lacks PLE's bin-local inductive bias. Deterministic in seed.
    """
    sc = StandardScaler().fit(Xtr)
    Atr, Ate = sc.transform(Xtr), sc.transform(Xte)
    rng = np.random.default_rng(20000 + seed)
    d = Atr.shape[1]
    W = rng.standard_normal((d, rff_dim))
    b = rng.uniform(0.0, 2.0 * np.pi, size=rff_dim)
    scale = np.sqrt(2.0 / rff_dim)

    def _map(A):
        return scale * np.cos(A @ W + b)

    return _map(Atr), _map(Ate)


def encode(encoding: str, Xtr, Xte, spec: dict, seed: int):
    """Dispatch the per-arm encoder. Raises on unknown encoding (no silent default)."""
    if encoding == "raw":
        sc = StandardScaler().fit(Xtr)
        return sc.transform(Xtr), sc.transform(Xte)
    if encoding == "ple":
        edges = fit_ple_edges(Xtr, int(spec.get("n_bins", 24)))
        return ple_transform(Xtr, edges), ple_transform(Xte, edges)
    if encoding == "rff":
        return _encode_rff(Xtr, Xte, int(spec.get("rff_dim", 192)), seed)
    raise ValueError(
        f"Unknown encoding {encoding!r}: add it to encode()."
    )


# ===========================================================================
# Seam 3 — component roles (module-level, unit-testable via bare import)
# ===========================================================================

def make_data(data_spec: dict, data_axes: dict, seed: int) -> dict:
    """make_data(data_spec, data_axes, seed) -> Dataset.

    Sequential split convention: a single RNG draws train then test from one stream.
    `target` (a data axis) selects a non-monotonic (sine) or linear logit. Returns
    raw features, labels, AND the latent logit (the T4 linearization analysis target).
    """
    spec = dict(data_spec)
    spec.update(data_axes)  # data-axis overrides win over the data-group defaults

    n_feat = int(spec.get("n_feat", 8))
    d_inf = int(spec.get("n_informative", 4))
    n_train = int(spec.get("n_train", 5600))
    n_test = int(spec.get("n_test", 2400))
    seed_base = int(spec.get("seed_base", 7000))
    target = spec.get("target", "nonmono")

    rng = np.random.default_rng(seed_base + seed)
    n = n_train + n_test
    X = rng.standard_normal((n, n_feat))

    if target == "nonmono":
        freqs = rng.uniform(1.5, 3.0, size=d_inf)
        amps = rng.uniform(1.0, 2.0, size=d_inf)
        logit = np.zeros(n)
        for j in range(d_inf):
            logit += amps[j] * np.sin(freqs[j] * X[:, j])
    elif target == "linear":
        signs = rng.choice(np.array([-1.0, 1.0]), size=d_inf)
        weights = rng.uniform(0.5, 1.5, size=d_inf) * signs
        logit = X[:, :d_inf] @ weights
    else:
        raise ValueError(f"Unknown target {target!r}: expected 'nonmono' or 'linear'.")

    logit = logit - logit.mean()
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)

    sl_tr, sl_te = slice(0, n_train), slice(n_train, n)
    return {
        "X_train": X[sl_tr], "X_test": X[sl_te],
        "y_train": y[sl_tr], "y_test": y[sl_te],
        "logit_train": logit[sl_tr], "logit_test": logit[sl_te],
        "_y_test_np": y[sl_te],
    }


def build_model(model_spec: dict):
    """build_model(model_spec) -> estimator. Standalone; train_arm composes it.

    Domain note: the debated methodology uses sklearn estimators (the F3 finding is
    specifically about MLPClassifier.n_iter_/early_stopping), so the model role
    returns an sklearn estimator rather than a torch nn.Module.
    """
    kind = model_spec["kind"]
    if kind == "logreg":
        return LogisticRegression(max_iter=int(model_spec.get("logreg_max_iter", 1000)))
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=tuple(model_spec.get("hidden_layers", [64, 64])),
            activation="relu",
            alpha=float(model_spec.get("alpha", 1e-4)),
            max_iter=int(model_spec.get("max_iter", 1000)),
            early_stopping=True,
            n_iter_no_change=int(model_spec.get("n_iter_no_change", 20)),
            random_state=int(model_spec.get("random_state", 0)),
        )
    raise ValueError(f"Unknown model kind {kind!r}: add it to build_model().")


class TrainResult(dict):
    """TrainResult(model, scores, val_score, meta) — typed-ish seam record."""

    def __init__(self, model, scores, val_score, meta):
        super().__init__(model=model, scores=scores, val_score=val_score, meta=meta)


def _train_logreg(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """Axis-agnostic: logistic regression on the arm's encoding."""
    spec = _merge_training_into_method(method_spec, train_cfg)
    Ztr, Zte = encode(spec["encoding"], data["X_train"], data["X_test"], spec, seed)
    model = build_model({"kind": "logreg", "logreg_max_iter": spec.get("logreg_max_iter", 1000)})
    model.fit(Ztr, data["y_train"])
    scores = model.predict_proba(Zte)[:, 1]
    meta = {
        "encoding": spec["encoding"], "kind": "logreg", "in_dim": int(Ztr.shape[1]),
        "n_iter": int(np.ravel(model.n_iter_)[0]),
        "max_iter": int(spec.get("logreg_max_iter", 1000)),
        "converged": None,
        "n_params": int(model.coef_.size + model.intercept_.size),
    }
    return TrainResult(model, scores, None, meta)


def _train_mlp(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """Axis-agnostic: MLP on the arm's encoding. Records n_iter_ + converged (T3)."""
    spec = _merge_training_into_method(method_spec, train_cfg)
    Ztr, Zte = encode(spec["encoding"], data["X_train"], data["X_test"], spec, seed)
    max_iter = int(spec.get("max_iter", 1000))
    model = build_model({
        "kind": "mlp", "hidden_layers": spec.get("hidden_layers", [64, 64]),
        "alpha": spec.get("alpha", 1e-4), "max_iter": max_iter,
        "n_iter_no_change": spec.get("n_iter_no_change", 20), "random_state": seed,
    })
    model.fit(Ztr, data["y_train"])
    scores = model.predict_proba(Zte)[:, 1]
    n_iter = int(model.n_iter_)
    n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    meta = {
        "encoding": spec["encoding"], "kind": "mlp", "in_dim": int(Ztr.shape[1]),
        "n_iter": n_iter, "max_iter": max_iter, "converged": bool(n_iter < max_iter),
        "n_params": n_params,
    }
    return TrainResult(model, scores, None, meta)


TRAIN_REGISTRY = {
    "logreg": _train_logreg,
    "mlp": _train_mlp,
}


def train_arm(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """Dispatched on method_spec['kind'] via TRAIN_REGISTRY. Raises on unknown kind."""
    kind = method_spec["kind"]
    try:
        fn = TRAIN_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown method.kind {kind!r}: add it to TRAIN_REGISTRY.") from None
    return fn(method_spec, data, seed, train_cfg)


def metric(scores: np.ndarray, labels: np.ndarray, **cfg) -> float:
    """metric(scores, labels, **cfg) -> float — PRIMARY metric: AUC-ROC [HYPOTHESIS.md]."""
    return float(roc_auc_score(labels, scores))


def _aux_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Auxiliary metric stored alongside the primary one: average precision (PR-AUC)."""
    return {"auprc": float(average_precision_score(labels, scores))}


# ===========================================================================
# Library-provided generics (NOT seams)
# ===========================================================================

def bootstrap_ci(values, n_resamples: int = 1000, seed: int = 0) -> tuple:
    """Percentile bootstrap CI -> (mean, lo, hi)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean()
                     for _ in range(n_resamples)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _merge_training_into_method(method_spec: dict, train_cfg: dict) -> dict:
    """Merge training-group fallbacks into method_spec without overriding the method."""
    spec = dict(method_spec)
    for key in ("hidden_layers", "alpha", "max_iter", "n_iter_no_change",
                "logreg_max_iter", "n_bins", "rff_dim"):
        if key not in spec and key in train_cfg:
            spec[key] = train_cfg[key]
    return spec


def _ridge_r2(Ztr, Zte, ytr, yte) -> float:
    """Held-out R^2 of a linear model predicting the latent logit from encoded features."""
    rg = Ridge(alpha=1.0).fit(Ztr, ytr)
    return float(r2_score(yte, rg.predict(Zte)))


# ===========================================================================
# Seam 4 — axis classification + dataset-keyed foreach expansion
# ===========================================================================
# No training axis in this experiment (AUC-ROC has no parameterization), so every
# method is axis-agnostic. The raise-on-unknown guard stays active regardless.

_AXIS_AGNOSTIC_KINDS = frozenset({"logreg", "mlp"})
_AXIS_DEPENDENT_KINDS = frozenset()


def is_axis_agnostic_method(kind: str) -> bool:
    """True if a method trains independently of any eval-axis. Raises on unknown kind."""
    if kind in _AXIS_AGNOSTIC_KINDS:
        return True
    if kind in _AXIS_DEPENDENT_KINDS:
        return False
    raise ValueError(
        f"Unknown method.kind {kind!r}: add it to _AXIS_AGNOSTIC_KINDS or "
        "_AXIS_DEPENDENT_KINDS."
    )


_LIFT_PAIRS = [
    ("mlp_ple", "mlp_raw"),        # T-main + T5 (per target cell)
    ("logreg_ple", "logreg_raw"),  # T1 mechanism
    ("mlp_ple", "mlp_raw_wide"),   # T2 capacity
    ("mlp_ple", "mlp_rff"),        # T2 basis
]


def load_method_cfg(method_name: str) -> dict:
    """Load one method config YAML as a plain dict."""
    import yaml

    return yaml.safe_load((_CONF_DIR / "method" / f"{method_name}.yaml").read_text())


def _cell_product(axes: dict) -> list:
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def _data_cell_of(cell: dict, data_axes: list) -> dict:
    return {k: cell[k] for k in data_axes if k in cell}


def _cell_key(cell: dict) -> tuple:
    """Hashable, order-independent cell key (lists -> tuples)."""
    def _h(v):
        return tuple(v) if isinstance(v, list) else v

    return tuple(sorted((k, _h(v)) for k, v in cell.items()))


def _expand_method(method_name: str, method_cfg: dict, overrides: dict) -> list:
    """Apply per-method overrides and return the concrete spec(s). No sweeps here."""
    eff = dict(method_cfg)
    if method_name in overrides:
        eff.update(overrides[method_name])
    eff["_method_name"] = method_name
    return [eff]


def build_dataset_keys(experiment: str, exp_cfg: dict, seeds: int,
                       method_loader=load_method_cfg) -> list:
    """Expand into dataset keys: one per (data-cell, seed), each carrying its method combos.

    The foreach grain is the DATASET KEY (data_axes + seed), NOT the per-method combo:
    each dataset is generated once and all methods train on the shared tensors.
    """
    axes = exp_cfg.get("axes", {})
    data_axes = list(exp_cfg.get("data_axes", []))
    methods = exp_cfg.get("methods", [])
    overrides = exp_cfg.get("method_overrides", {})
    cells = _cell_product(axes)

    by_data: dict = {}
    for cell in cells:
        by_data.setdefault(_cell_key(_data_cell_of(cell, data_axes)), []).append(cell)

    dataset_keys: list = []
    for _, group_cells in by_data.items():
        data_cell = _data_cell_of(group_cells[0], data_axes)
        for seed in range(seeds):
            combos = _build_combos(methods, overrides, method_loader)
            dataset_keys.append({"experiment": experiment, "data_cell": data_cell,
                                 "seed": seed, "combos": combos})
    return dataset_keys


def _build_combos(methods: list, overrides: dict, method_loader) -> list:
    """One combo per method (all axis-agnostic). Raise-guard via is_axis_agnostic_method."""
    combos: list = []
    for method_name in methods:
        raw = method_loader(method_name)
        for spec in _expand_method(method_name, raw, overrides):
            if not is_axis_agnostic_method(spec["kind"]):
                raise ValueError(
                    f"Method {method_name!r} (kind {spec['kind']!r}) is axis-dependent, "
                    "but this experiment declares no training axis."
                )
            combos.append({"method": method_name, "method_spec": spec, "axis_dep": False})
    return combos


# ===========================================================================
# Static-branch analysis helpers (pure consumers of train records)
# ===========================================================================

def filter_records(records: list, experiment: str) -> list:
    return [r for r in records if r.get("experiment") == experiment]


def analyze_auc_by_arm(records: list, experiment: str) -> list:
    """ANALYSIS BRANCH 1 — mean AUC per (target, method)."""
    recs = filter_records(records, experiment)
    if not recs:
        return []
    acc: dict = {}
    for r in recs:
        acc.setdefault((r["cell"].get("target"), r["method"]), []).append(r["test"]["primary"])
    rows = []
    for (target, method), vals in sorted(acc.items(), key=lambda kv: str(kv[0])):
        rows.append({"target": target, "method": method,
                     "auc_mean": float(np.mean(vals)), "n_seeds": len(vals)})
    return rows


def analyze_convergence(records: list, experiment: str) -> list:
    """ANALYSIS BRANCH 2 — T3: n_iter_ / converged / n_params per MLP arm."""
    recs = [r for r in filter_records(records, experiment) if r["meta"]["kind"] == "mlp"]
    if not recs:
        return []
    acc: dict = {}
    for r in recs:
        m = r["method"]
        d = acc.setdefault(m, {"n_iter": [], "converged": [],
                               "max_iter": r["meta"]["max_iter"],
                               "n_params": r["meta"]["n_params"]})
        d["n_iter"].append(r["meta"]["n_iter"])
        d["converged"].append(bool(r["meta"]["converged"]))
    rows = []
    for method, d in sorted(acc.items()):
        rows.append({"method": method, "max_iter": d["max_iter"], "n_params": d["n_params"],
                     "n_iter_mean": float(np.mean(d["n_iter"])),
                     "n_iter_max": int(np.max(d["n_iter"])),
                     "frac_converged": float(np.mean(d["converged"]))})
    return rows


def analyze_linearization(records: list, experiment: str, data_cfg: dict, n_bins: int) -> list:
    """ANALYSIS BRANCH 3 — T4: held-out Ridge R^2 on the latent logit, PLE vs raw.

    Regenerates each (target, seed) dataset deterministically (data is reproducible
    from the data-cell + seed) so the linearization diagnostic stays OUT of train.
    """
    recs = filter_records(records, experiment)
    if not recs:
        return []
    jobs, seen = [], set()
    for r in recs:
        key = (r["cell"].get("target"), r["seed"])
        if key not in seen:
            seen.add(key)
            jobs.append(key)
    acc: dict = {}
    for target, seed in jobs:
        data = make_data(data_cfg, {"target": target}, seed)
        ztr_raw, zte_raw = encode("raw", data["X_train"], data["X_test"], {}, seed)
        ztr_ple, zte_ple = encode("ple", data["X_train"], data["X_test"], {"n_bins": n_bins}, seed)
        d = acc.setdefault(target, {"raw": [], "ple": []})
        d["raw"].append(_ridge_r2(ztr_raw, zte_raw, data["logit_train"], data["logit_test"]))
        d["ple"].append(_ridge_r2(ztr_ple, zte_ple, data["logit_train"], data["logit_test"]))
    return [{"target": t, "ridge_r2_raw": float(np.mean(v["raw"])),
             "ridge_r2_ple": float(np.mean(v["ple"]))} for t, v in sorted(acc.items())]


def _seed_paired_lift(records: list, method_a: str, method_b: str,
                      cell_keys: list, bseed: int = 0) -> list:
    """Per-cell seed-paired AUC lift (a - b) with bootstrap CI. Pairs by seed."""
    by_key: dict = {}
    for r in records:
        ck = _cell_key({k: r["cell"].get(k) for k in cell_keys})
        by_key.setdefault(ck, {}).setdefault(r["method"], {})[r["seed"]] = r["test"]["primary"]
    out = []
    for ck, methods in by_key.items():
        a, b = methods.get(method_a, {}), methods.get(method_b, {})
        shared = sorted(set(a) & set(b))
        if not shared:
            continue
        lifts = np.array([a[s] - b[s] for s in shared])
        m, lo, hi = bootstrap_ci(lifts, n_resamples=1000, seed=bseed)
        out.append({"cell": dict(ck), "lift_mean": m, "lift_lo": lo, "lift_hi": hi,
                    "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_seeds": len(shared)})
    return out


# ===========================================================================
# Seam 2 — the DAG (guarded so the module imports without metaflow installed)
# ===========================================================================

try:
    from metaflow import Config, FlowSpec, card, step  # verified against metaflow 2.19.x
    from metaflow.cards import Markdown

    _METAFLOW_AVAILABLE = True
except ImportError:
    _METAFLOW_AVAILABLE = False


def _hydra_parser(text: str) -> dict:
    """Config parser: Hydra-compose the conf tree into a plain dict (Seam 1)."""
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

    class PLEFlow(FlowSpec):
        """PLE-vs-raw promoted pipeline. foreach grain = (target, seed) dataset key."""

        cfg = Config("cfg", default=str(_CONF_DIR / "config.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            """Compose config; expand the experiment into DATASET KEYS (data-cell x seed)."""
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            self.data_cfg = dict(cfg["data"])
            self.training_cfg = dict(cfg["training"])
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = exp_cfg["name"]
            self.determinism = exp_cfg.get("determinism", "order_independent")
            self.bootstrap_cfg = dict(cfg["bootstrap"])
            self.dataset_keys = build_dataset_keys(self.experiment_name, exp_cfg, cfg["seeds"])
            print(f"[start] experiment={self.experiment_name} "
                  f"dataset_keys(foreach grain)={len(self.dataset_keys)} "
                  f"determinism={self.determinism}", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            """Generate ONE dataset; train every arm on the shared tensors."""
            dk = self.input
            data_cell = dk["data_cell"]
            seed = dk["seed"]
            data = make_data(self.data_cfg, data_cell, seed)
            records = []
            for combo in dk["combos"]:
                res = train_arm(combo["method_spec"], data, seed, self.training_cfg)
                records.append(self._record(self.experiment_name, data_cell, seed,
                                            combo["method"], combo["method_spec"], res, data))
            self.records = records
            self.next(self.join)

        @staticmethod
        def _record(experiment_name, data_cell, seed, method_name, spec, res, data) -> dict:
            """Build one (method, data-cell, seed) record from a TrainResult."""
            scores = res["scores"]
            y_te = data["_y_test_np"]
            return {
                "experiment": experiment_name,
                "data_cell": dict(data_cell),
                "cell": dict(data_cell),  # no training axis -> cell == data_cell
                "seed": seed,
                "method": method_name,
                "config": {k: v for k, v in spec.items() if not k.startswith("_")},
                "val_score": res["val_score"],
                "test": {"primary": metric(scores, y_te), "aux": _aux_metrics(scores, y_te)},
                "meta": res["meta"],
            }

        @step
        def join(self, inputs):
            """Flatten dataset-branch records; merge only flat, comparable artifacts."""
            self.all_records = [r for inp in inputs for r in inp.records]
            self.merge_artifacts(inputs, include=[
                "data_cfg", "training_cfg", "experiment_cfg", "experiment_name",
                "bootstrap_cfg", "determinism"])
            print(f"[join] total records: {len(self.all_records)}", flush=True)
            self.next(self.aggregate)

        @step
        def aggregate(self):
            """Bootstrap-CI mean AUC per (cell, method) + seed-paired lifts (pinned shapes)."""
            from collections import defaultdict

            n_res = self.bootstrap_cfg.get("n_resamples", 1000)
            bseed = self.bootstrap_cfg.get("seed", 0)

            grouped: dict = defaultdict(list)
            for rec in self.all_records:
                grouped[(_cell_key(rec["cell"]), rec["method"])].append(rec)

            self.aggregate_results = []
            for (ck, method), recs in grouped.items():
                # Sort by seed so the bootstrap input order is canonical regardless of
                # foreach branch arrival order — required for the order_independent
                # contract (percentile bootstrap is order-sensitive under a fixed RNG).
                vals = np.array([r["test"]["primary"]
                                 for r in sorted(recs, key=lambda r: r["seed"])])
                m, lo, hi = bootstrap_ci(vals, n_resamples=n_res, seed=bseed)
                self.aggregate_results.append({
                    "cell": dict(ck), "method": method,
                    "auc_roc_mean": m, "auc_roc_lo": lo, "auc_roc_hi": hi})

            self.lift_results = []
            for a, b in _LIFT_PAIRS:
                for entry in _seed_paired_lift(self.all_records, a, b, ["target"], bseed):
                    entry["pair"] = f"{a}_minus_{b}"
                    self.lift_results.append(entry)

            print(f"[aggregate] {len(self.aggregate_results)} cell/method summaries, "
                  f"{len(self.lift_results)} paired lifts", flush=True)
            self.next(self.an_auc_by_arm, self.an_convergence, self.an_linearization)

        @card
        @step
        def an_auc_by_arm(self):
            """ANALYSIS BRANCH 1: mean AUC per (target, method)."""
            from metaflow import current

            rows = analyze_auc_by_arm(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name,
                              "branch": "auc_by_arm", "rows": rows}
            if rows:
                current.card.append(Markdown(f"## AUC by arm ({len(rows)} cells)"))
            self.next(self.join_analyses)

        @card
        @step
        def an_convergence(self):
            """ANALYSIS BRANCH 2 (T3): convergence diagnostics per MLP arm."""
            from metaflow import current

            rows = analyze_convergence(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name,
                              "branch": "convergence", "rows": rows}
            if rows:
                current.card.append(Markdown(f"## Convergence ({len(rows)} MLP arms)"))
            self.next(self.join_analyses)

        @card
        @step
        def an_linearization(self):
            """ANALYSIS BRANCH 3 (T4): Ridge R^2 on the latent logit, PLE vs raw."""
            from metaflow import current

            n_bins = int(self.training_cfg.get("n_bins", 24))
            rows = analyze_linearization(self.all_records, self.experiment_name,
                                         self.data_cfg, n_bins)
            self.an_result = {"experiment": self.experiment_name,
                              "branch": "linearization", "rows": rows}
            if rows:
                current.card.append(Markdown(f"## Linearization R^2 ({len(rows)} targets)"))
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            """Collect analysis artifacts; propagate shared artifacts from inputs[0]."""
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
            """Print a concise summary of every analysis branch."""
            print("\n=== PLEFlow analyses ===", flush=True)
            for a in self.analyses.values():
                print(f"  {a.get('experiment')}: {a.get('rows')}", flush=True)
            self.next(self.end)

        @step
        def end(self):
            """Final summary; archive a stats dump for Step 7 conclusions/figures."""
            import json

            payload = {
                "experiment": self.experiment_name,
                "determinism": self.determinism,
                "aggregate_results": self.aggregate_results,
                "lift_results": self.lift_results,
                "analyses": {f"{k[0]}::{k[1]}": v for k, v in self.analyses.items()},
            }
            _STATS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\n=== PLEFlow complete: {self.experiment_name} ===", flush=True)
            print(f"cell/method summaries: {len(self.aggregate_results)} | "
                  f"lifts: {len(self.lift_results)} | wrote {_STATS_OUT.name}", flush=True)

    if __name__ == "__main__":
        PLEFlow()
