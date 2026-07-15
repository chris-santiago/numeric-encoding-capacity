# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.8",
#   "numpy",
#   "scikit-learn",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""Annotated REFERENCE flow for the Metaflow + Hydra pipeline standard.

WHAT THIS IS
------------
A heavily-commented, syntactically-real reference that demonstrates the WIRING
and the DAG SHAPE of the standard on a trivial toy domain ("feature X separates
two classes"). It exists to be READ when authoring a real flow, and to be
import-checked (`import reference_flow`) so the lazy-metaflow guard and the
component signatures stay honest.

WHAT THIS IS NOT
----------------
NOT runnable end-to-end and NOT part of CI. The `from my_project import ...` line
below intentionally raises ModuleNotFoundError on import — that is the correct,
expected behaviour and it demonstrates the script-mode import trap that the
`sys.path` shim solves in a real repo. `py_compile` is the gate for this file.
Do not add a runnable toy investigation, a smoke test, or rot-prevention
machinery. Replace the toy bodies with your domain code when you author for real.

The TOY is deliberately trivial (linear-ish separation, tiny MLP). The point is
DAG shape and seam wiring, not ML substance.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

# --- repo-package import shim (Seam 1) ---------------------------------------
# In a real repo this makes your package importable under `uv run` script mode.
# Here it is illustrative; parents[2]/"src" will not exist next to this asset.
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

# --- the bare project import that demonstrates the script-mode trap ----------
# THIS LINE INTENTIONALLY RAISES ModuleNotFoundError on import. In a real repo it
# resolves via the _SRC shim above. Keeping it here is the whole point of the
# reference: it shows authors exactly which import the shim rescues. Do NOT remove
# it to make the module import cleanly — that would hide the trap.
from my_project import primary_metric_lib  # noqa: E402,F401  (expected to raise)


# ===========================================================================
# Seam 3 — component roles (module-level functions; unit-testable via import)
# ===========================================================================

def make_data(data_spec: dict, data_axes: dict, seed: int) -> dict:
    """make_data(data_spec, data_axes, seed) -> Dataset.

    Generate the toy dataset deterministically. `data_axes` is the data-affecting
    subset of the cell (here: `separation`); it overrides `data_spec` so the
    dataset-keyed foreach is correct. Honors split_convention.
    """
    spec = dict(data_spec)
    spec.update(data_axes)  # data-axis overrides win over the data group defaults

    n_feat = spec.get("n_feat", 20)
    signal_feats = list(spec.get("signal_feats", [0, 1, 2]))
    separation = spec.get("separation", 2.0)
    noise_scale = spec.get("noise_scale", 1.0)
    pos_rate = spec.get("pos_rate", 0.1)
    seed_base = spec.get("seed_base", 7000)
    convention = spec.get("split_convention", "sequential")
    n_train = spec.get("n_train", 20000)
    n_test = spec.get("n_test", 40000)
    n_val = spec.get("n_val", 10000)

    def _one(n: int, rng: np.random.Generator) -> tuple:
        X = (rng.standard_normal((n, n_feat)) * noise_scale).astype(np.float32)
        y = (rng.random(n) < pos_rate).astype(np.float32)
        pos = np.where(y > 0.5)[0]
        X[np.ix_(pos, signal_feats)] += separation  # the class signal
        return torch.tensor(X), torch.tensor(y).unsqueeze(1)

    if convention == "sequential":
        # Single RNG advanced train-then-test; no val set (test continues stream).
        rng = np.random.default_rng(seed_base + seed)
        X_tr, y_tr = _one(n_train, rng)
        X_te, y_te = _one(n_test, rng)
        return {"X_train": X_tr, "y_train": y_tr, "X_val": None, "y_val": None,
                "X_test": X_te, "y_test": y_te,
                "_y_test_np": y_te.squeeze(1).numpy()}
    elif convention == "independent":
        # Independent seed offsets per split -> supports val-based selection.
        X_tr, y_tr = _one(n_train, np.random.default_rng(seed_base + seed))
        X_val, y_val = _one(n_val, np.random.default_rng(seed_base + seed + 1000))
        X_te, y_te = _one(n_test, np.random.default_rng(seed_base + seed + 2000))
        return {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val,
                "X_test": X_te, "y_test": y_te,
                "_y_val_np": y_val.squeeze(1).numpy(),
                "_y_test_np": y_te.squeeze(1).numpy()}
    else:
        raise ValueError(f"Unknown split_convention: {convention!r}")


def build_model(model_spec: dict) -> nn.Module:
    """build_model(model_spec) -> nn.Module. Standalone backbone; train composes it."""
    d_in = model_spec.get("d_in", 20)
    hidden = model_spec.get("hidden", 64)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))


@torch.no_grad()
def _scores_of(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Eval-mode forward returning squeezed numpy scores; restores train mode."""
    was_training = model.training
    model.eval()
    s = model(X).squeeze(1).numpy()
    model.train(was_training)
    return s


# --- Seam 3: registry dispatch for train_arm --------------------------------
# Each trainer fills the TrainResult role (model, scores, val_score). The two
# kinds demonstrate the axis-agnostic / axis-dependent split:
#   ce   -> axis-agnostic (trains once, evaluated at every eval_k)
#   topk -> axis-dependent (loss bakes in eval_k -> retrains per eval_k)

class TrainResult(dict):
    """TrainResult(model, scores, val_score) — a typed-ish record for the seam.

    A thin dict subclass so the three fields are named at the boundary while
    staying trivially picklable for the Metaflow datastore.
    """

    def __init__(self, model: nn.Module, scores: np.ndarray, val_score):
        super().__init__(model=model, scores=scores, val_score=val_score)


def _train_ce(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """Axis-agnostic baseline: plain weighted CE. No eval-axis dependence."""
    spec = _merge_training_into_method(method_spec, train_cfg)
    torch.manual_seed(seed)
    model = build_model({"d_in": data["X_train"].shape[1], "hidden": spec["hidden"]})
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    X, y = data["X_train"], data["y_train"]
    # Anchor to the data device so GPU data does not trigger a device mismatch.
    pw = torch.tensor([(y.numel() - y.sum()) / max(1.0, float(y.sum()))],
                      device=y.device)
    for epoch in range(spec["epochs"]):
        for idx in _epoch_batches(X.size(0), seed + epoch, spec["batch"]):
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(
                model(X[idx]), y[idx], pos_weight=pw
            ).backward()
            opt.step()
    return TrainResult(model, _scores_of(model, data["X_test"]),
                       _val_score(model, data, spec))


def _train_topk(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """Axis-dependent method: a margin ranking loss parameterized by eval_k.

    The eval_k value (the top-k cutoff) enters the objective directly: only the
    eval_k highest-scoring negatives participate in the pairwise margin, so the
    model is penalised specifically for negatives that intrude on the top-k region.
    This makes the retrain-per-eval_k invariant real — a different eval_k produces
    a different gradient, not just a re-evaluation of the same model.
    The flow handles that by giving axis-dependent methods one combo per
    training-axis value (see build_dataset_keys).
    """
    spec = _merge_training_into_method(method_spec, train_cfg)
    torch.manual_seed(seed)
    eval_k = int(spec.get("eval_k", 50))
    model = build_model({"d_in": data["X_train"].shape[1], "hidden": spec["hidden"]})
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
    X, y = data["X_train"], data["y_train"]
    margin = spec.get("margin", 1.0)
    for epoch in range(spec["epochs"]):
        # Same per-epoch reshuffle as the CE arm — reshuffle SYMMETRY across
        # methods is a fidelity invariant; an asymmetry silently shifts numbers.
        for idx in _epoch_batches(X.size(0), seed + epoch, spec["batch"]):
            logits = model(X[idx]).squeeze(1)
            yb = y[idx].squeeze(1)
            pos, neg = logits[yb > 0.5], logits[yb <= 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            # Restrict negatives to the top-eval_k region: take the eval_k
            # highest-scoring negatives so the loss specifically penalises
            # negatives ranked within the cutoff.  eval_k thus shapes the
            # gradient — a smaller cutoff focuses pressure on the very top of
            # the ranking, causing a different optimum than a larger cutoff.
            k_neg = min(eval_k, neg.numel())
            hard_neg = neg.topk(k_neg).values          # (k_neg,)
            loss = F.relu(margin - (pos.unsqueeze(1) - hard_neg.unsqueeze(0))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return TrainResult(model, _scores_of(model, data["X_test"]),
                       _val_score(model, data, spec))


TRAIN_REGISTRY = {
    "ce": _train_ce,        # axis-agnostic
    "topk": _train_topk,    # axis-dependent
}


def train_arm(method_spec: dict, data: dict, seed: int, train_cfg: dict) -> TrainResult:
    """train_arm(method_spec, data, seed, train_cfg) -> TrainResult.

    Dispatched on method_spec['kind'] via TRAIN_REGISTRY. Raises on unknown kind
    — never silently default a new method.
    """
    kind = method_spec["kind"]
    try:
        fn = TRAIN_REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"Unknown method.kind {kind!r}: add it to TRAIN_REGISTRY."
        ) from None
    return fn(method_spec, data, seed, train_cfg)


def metric(scores: np.ndarray, labels: np.ndarray, **cfg) -> float:
    """metric(scores, labels, **cfg) -> float — the workflow's PRIMARY metric.

    Here the primary metric is recall_at_k: fraction of positives among the top-k
    scored items. `k` arrives via cfg (threaded from the eval_k training axis).
    Bind this to your workflow's existing headline metric; do not invent a new one.
    """
    k = int(cfg.get("k", 50))
    k = max(1, min(k, len(scores)))
    top = np.argsort(-scores)[:k]
    n_pos = float(labels.sum())
    return 0.0 if n_pos == 0 else float(labels[top].sum() / n_pos)


def _aux_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Auxiliary metrics stored alongside the primary one (not the gate metric)."""
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


# ===========================================================================
# Library-provided generics (NOT seams): bootstrap CI + small merge helpers
# ===========================================================================

def bootstrap_ci(values, n_resamples: int = 1000, seed: int = 0) -> tuple:
    """Percentile bootstrap CI -> (mean, lo, hi). Standard-owned; aggregate uses it."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(v, size=len(v), replace=True).mean()
                     for _ in range(n_resamples)])
    return float(v.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _epoch_batches(n: int, seed: int, batch: int) -> list:
    """Deterministic per-epoch shuffle as a list of index tensors."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    return [perm[i:i + batch] for i in range(0, n, batch)]


def _merge_training_into_method(method_spec: dict, train_cfg: dict) -> dict:
    """Merge training-group defaults into method_spec WITHOUT overriding the method.

    The method YAML is authoritative for keys it sets; the training group supplies
    the shared fallbacks (epochs/lr/batch/hidden). This is the merge-priority rule
    from conf/SCHEMA.md — keeping epochs/lr/batch in the training group only.
    """
    spec = dict(method_spec)
    for key in ("epochs", "lr", "batch", "hidden", "warmup_frac"):
        if key not in spec and key in train_cfg:
            spec[key] = train_cfg[key]
    spec.setdefault("epochs", 20)
    spec.setdefault("lr", 1e-3)
    spec.setdefault("batch", 512)
    spec.setdefault("hidden", 64)
    return spec


def _val_score(model: nn.Module, data: dict, spec: dict):
    """Val-selection scalar, or None under the sequential (no-val) split."""
    if data.get("X_val") is None:
        return None
    return metric(_scores_of(model, data["X_val"]), data["_y_val_np"],
                  k=spec.get("eval_k", 50))


# ===========================================================================
# Seam 4 — axis classification + dataset-keyed foreach expansion
# ===========================================================================
#
# Most experiment axes do NOT change the data; only the declared `data_axes` do.
# The flow generates each dataset ONCE per (data-cell, seed) and trains every
# method that uses it on the shared in-memory tensors. These pure helpers build
# the dataset-key list (the foreach grain).

# Axis-agnostic method kinds: the trained model does NOT depend on the eval-axis,
# so the method trains ONCE and is evaluated at every eval_k.
_AXIS_AGNOSTIC_KINDS = frozenset({"ce"})
# Axis-dependent method kinds: the loss bakes in the eval-axis -> retrain per value.
_AXIS_DEPENDENT_KINDS = frozenset({"topk"})


def is_axis_agnostic_method(kind: str) -> bool:
    """True if a method of this kind trains independently of the eval-axis.

    Raises on unknown kind so a new method cannot be silently misclassified as
    agnostic (which would train an axis-dependent method only once).
    """
    if kind in _AXIS_AGNOSTIC_KINDS:
        return True
    if kind in _AXIS_DEPENDENT_KINDS:
        return False
    raise ValueError(
        f"Unknown method.kind {kind!r}: add it to _AXIS_AGNOSTIC_KINDS or "
        "_AXIS_DEPENDENT_KINDS."
    )


_CONF_DIR = pathlib.Path(__file__).parent / "conf"


def load_method_cfg(method_name: str) -> dict:
    """Load one method config YAML as a plain dict (no Hydra composition needed)."""
    import yaml

    return yaml.safe_load((_CONF_DIR / "method" / f"{method_name}.yaml").read_text())


def _cell_product(axes: dict) -> list:
    """Cartesian product of axis values as a list of cell dicts."""
    if not axes:
        return [{}]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def _data_cell_of(cell: dict, data_axes: list) -> dict:
    """Project a full cell onto its data-axis subset (the part that varies data)."""
    return {k: cell[k] for k in data_axes if k in cell}


def _cell_key(cell: dict) -> tuple:
    """Hashable, order-independent key for a cell. Lists -> tuples (unhashable fix)."""
    def _h(v):
        return tuple(v) if isinstance(v, list) else v

    return tuple(sorted((k, _h(v)) for k, v in cell.items()))


def _expand_method(method_name: str, method_cfg: dict, overrides: dict) -> list:
    """Expand a method's in-process sweep (e.g. temp_sweep) into concrete specs.

    `method_overrides` are merged BEFORE expansion, so an experiment can collapse a
    sweep to a singleton (disabling val-based selection to match a fixed source).
    """
    eff = dict(method_cfg)
    if method_name in overrides:
        eff.update(overrides[method_name])
    base = {k: v for k, v in eff.items() if not k.endswith("_sweep")}
    base["_method_name"] = method_name
    sweep = {k[:-6]: list(v) for k, v in eff.items() if k.endswith("_sweep")}
    if not sweep:
        return [dict(base)]
    keys = list(sweep)
    specs = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        spec = dict(base)
        spec.update(dict(zip(keys, combo)))
        specs.append(spec)
    return specs


def build_dataset_keys(experiment: str, exp_cfg: dict, seeds: int,
                       method_loader=load_method_cfg) -> list:
    """Expand an experiment into dataset keys, each carrying its training combos.

    A dataset key = (experiment, data-cell, seed): the data is generated ONCE per
    key. Each combo is a (method, concrete method_spec, eval_axis values) unit:
      * axis-agnostic methods appear ONCE per (method, non-axis cell, sweep) with
        an `eval_ks` list of every eval_k that cell uses -> train once, eval each.
      * axis-dependent methods appear once per (method, cell, sweep, eval_k) with a
        single-element `eval_ks` list -> retrain per eval_k.

    The returned list length is the DATASET count (data-cells x seeds), NOT the
    combo count — that is the whole point of the dataset-keyed foreach.
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
            combos = _build_combos(group_cells, methods, overrides, method_loader)
            dataset_keys.append({"experiment": experiment, "data_cell": data_cell,
                                 "seed": seed, "combos": combos})
    return dataset_keys


def _build_combos(group_cells: list, methods: list, overrides: dict,
                  method_loader) -> list:
    """Build the training-combo list for one dataset (all cells sharing data).

    Axis-agnostic methods are de-duplicated across eval_k: one combo per
    (method, non-axis cell, sweep) collecting every eval_k. Axis-dependent methods
    get one combo per eval_k.
    """
    agnostic: dict = {}
    dependent: list = []
    for method_name in methods:
        raw = method_loader(method_name)
        for cell in group_cells:
            eval_k = cell.get("eval_k", 50)
            for spec in _expand_method(method_name, raw, overrides):
                kind = spec["kind"]
                if is_axis_agnostic_method(kind):
                    # Strip the eval-axis: the same model serves every eval_k.
                    agn = {k: v for k, v in spec.items() if k != "eval_k"}
                    sig = (method_name,) + _cell_key(agn)
                    combo = agnostic.setdefault(sig, {
                        "method": method_name, "method_spec": agn,
                        "eval_ks": [], "axis_dep": False})
                    if eval_k not in combo["eval_ks"]:
                        combo["eval_ks"].append(eval_k)
                else:
                    dep_spec = dict(spec)
                    dep_spec["eval_k"] = eval_k
                    dependent.append({"method": method_name, "method_spec": dep_spec,
                                      "eval_ks": [eval_k], "axis_dep": True})
    combos = list(agnostic.values()) + dependent
    for c in combos:
        c["eval_ks"] = sorted(set(c["eval_ks"]))
    return combos


# ===========================================================================
# Static-branch analysis helpers (pure; consumers of train records)
# ===========================================================================
#
# Each analysis filters the aggregated records for ITS experiment and computes a
# result. Pure, so unit-testable without metaflow. Diagnostics live HERE (in the
# analysis branches), NEVER inside train — train->analyze separation is a hard
# invariant.

def filter_records(records: list, experiment: str) -> list:
    return [r for r in records if r.get("experiment") == experiment]


def _seed_paired_lift(records: list, method_a: str, method_b: str,
                      cell_keys: list, bseed: int = 0) -> list:
    """Per-cell seed-paired primary-metric lift (a - b) with bootstrap CI.

    Pairs by seed (robust to foreach delivery order). Emits the pinned
    lift_results shape (see conf/SCHEMA.md).
    `bseed` is threaded from bootstrap_cfg so the CI is reproducibly tied to the
    configured seed rather than silently fixed to 0.
    """
    by_key: dict = {}
    for r in records:
        # Use _cell_key for canonicalization: list-valued axes hash the same way
        # everywhere (including aggregate and select_by_val).
        ck = _cell_key({k: r["cell"].get(k) for k in cell_keys})
        by_key.setdefault(ck, {}).setdefault(r["method"], {})[r["seed"]] = (
            r["test"]["primary"])
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


def analyze_metric_trend(records: list, experiment: str) -> list:
    """ANALYSIS BRANCH 1 — primary-metric trend across the eval-axis per method.

    Reproduces the headline "method beats baseline at small k" reading: per
    (separation, eval_k) cell, mean primary metric per method.
    `experiment` is the threaded experiment name so renaming the experiment in
    experiment/<name>.yaml does not silently return [].
    """
    recs = filter_records(records, experiment)
    if not recs:
        return []
    acc: dict = {}
    for r in recs:
        ck = (r["cell"].get("separation"), r["cell"].get("eval_k"))
        acc.setdefault((ck, r["method"]), []).append(r["test"]["primary"])
    rows = []
    for (ck, method), vals in sorted(acc.items(), key=lambda kv: str(kv[0])):
        rows.append({"separation": ck[0], "eval_k": ck[1], "method": method,
                     "primary_mean": float(np.mean(vals))})
    return rows


def analyze_auxiliary_gap(records: list, experiment: str) -> dict:
    """ANALYSIS BRANCH 2 — auxiliary-metric gap (a scalar diagnostic).

    Demonstrates the SCALAR-shaped analysis (returns {result: {...}}, not rows).
    Composes off the SAME train artifacts as branch 1 — two analyses fanning out
    from one train, the canonical >=2 analysis branches of the standard.
    `experiment` is the threaded experiment name; same rationale as branch 1.
    """
    recs = filter_records(records, experiment)
    if not recs:
        return {}

    def _mean_aux(method: str, key: str):
        vals = [r["test"]["aux"][key] for r in recs if r["method"] == method]
        return float(np.mean(vals)) if vals else None

    return {"auroc_topk": _mean_aux("topk_ranking", "auroc"),
            "auroc_baseline": _mean_aux("baseline_ce", "auroc")}


# ===========================================================================
# Seam 2 — the DAG. Guarded so the module imports without metaflow installed.
# ===========================================================================
#
# Default spine (a DEFAULT shape, not a mandate):
#   start -> foreach(dataset) -> train -> join -> [select_by_val?]
#         -> aggregate -> branch(an_metric_trend, an_auxiliary_gap)
#         -> join_analyses -> report -> end
#
# select_by_val is OPTIONAL: present only when methods carry an internal sweep that
# needs validation selection (independent split). It is shown here for reference.

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

    class ReferenceFlow(FlowSpec):
        """Reference pipeline (dataset-keyed). Demonstrates the standard's DAG.

        The foreach grain in `train` is the DATASET KEY (experiment x data-cell x
        seed), NOT the per-config combo. Each dataset is generated once and all
        methods that use it train on the shared in-memory tensors. Axis-agnostic
        methods train once and are evaluated at every eval_k; axis-dependent
        methods retrain per eval_k.
        """

        cfg = Config("cfg", default=str(_CONF_DIR / "config.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            """Compose config; expand the enabled experiment into DATASET KEYS."""
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            experiment = exp_cfg["name"]
            self.data_cfg = dict(cfg["data"])
            self.training_cfg = dict(cfg["training"])
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = experiment
            # Declared reproducibility contract, echoed as a run artifact so any
            # reader of a finished run (and scripts/determinism-check.py) knows the
            # claim. order_independent (default) | single_worker | nondeterministic.
            self.determinism = exp_cfg.get("determinism", "order_independent")
            self.bootstrap_cfg = dict(cfg["bootstrap"])
            self.dataset_keys = build_dataset_keys(experiment, exp_cfg, cfg["seeds"])
            print(f"[start] experiment={experiment} "
                  f"dataset_keys(foreach grain)={len(self.dataset_keys)}", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            """Generate ONE dataset, train every combo on the shared tensors.

            Emits one record per (method, config, eval_k). Axis-agnostic methods
            train once and the primary metric is read at each eval_k; axis-dependent
            methods retrain per eval_k. Diagnostics are NOT computed here — they are
            the analysis branches' job (train->analyze separation).
            """
            torch.set_num_threads(1)
            dk = self.input
            data_cell = dk["data_cell"]
            seed = dk["seed"]
            exp_cfg = self.experiment_cfg
            requests_model = bool(exp_cfg.get("requests_model", False))
            requests_scores = bool(exp_cfg.get("requests_scores", False))

            # Build the dataset ONCE (data_cell carries the data-axis overrides).
            data = make_data(self.data_cfg, data_cell, seed)

            records = []
            for combo in dk["combos"]:
                method_name = combo["method"]
                if combo["axis_dep"]:
                    # One training per eval_k (loss bakes in the cutoff).
                    for k in combo["eval_ks"]:
                        spec = dict(combo["method_spec"], eval_k=k)
                        res = train_arm(spec, data, seed, self.training_cfg)
                        records.append(self._record(
                            self.experiment_name,
                            data_cell, seed, method_name, spec, k, res, data,
                            requests_model, requests_scores))
                else:
                    # Train ONCE; evaluate the primary metric at every eval_k.
                    res = train_arm(combo["method_spec"], data, seed, self.training_cfg)
                    for k in combo["eval_ks"]:
                        records.append(self._record(
                            self.experiment_name,
                            data_cell, seed, method_name, combo["method_spec"], k,
                            res, data, requests_model, requests_scores))
            self.records = records
            self.next(self.join)

        @staticmethod
        def _record(experiment_name, data_cell, seed, method_name, spec, eval_k, res,
                    data, requests_model, requests_scores) -> dict:
            """Build one (method, config, eval_k) record from a TrainResult.

            `experiment_name` is threaded from self.experiment_name so renaming
            the experiment in experiment/<name>.yaml propagates to every record and
            analysis filter without code changes.
            """
            scores = res["scores"]
            y_te = data["_y_test_np"]
            rec = {
                "experiment": experiment_name,
                "data_cell": dict(data_cell),
                "cell": {**data_cell, "eval_k": eval_k},  # canonical (data + axis) key
                "seed": seed,
                "method": method_name,  # NOT "arm"
                "config": {k: v for k, v in spec.items() if not k.startswith("_")},
                "eval_k": eval_k,
                "val_score": res["val_score"],
                "test": {"primary": metric(scores, y_te, k=eval_k),
                         "aux": _aux_metrics(scores, y_te)},
            }
            if requests_model:
                rec["model"] = res["model"]
            if requests_scores:
                rec["scores"] = scores
                rec["labels"] = y_te
            return rec

        @step
        def join(self, inputs):
            """Flatten all dataset-branch records into a single list.

            Reads inputs[0] for the propagated config artifacts and NEVER
            merge_artifacts over an nn.Module-carrying artifact: equality
            comparison on tensors raises. Only flat, comparable artifacts are
            merged; all_records is concatenated by hand.
            """
            self.all_records = [r for inp in inputs for r in inp.records]
            self.merge_artifacts(inputs, include=[
                "data_cfg", "training_cfg", "experiment_cfg", "experiment_name",
                "bootstrap_cfg", "determinism"])
            print(f"[join] total records: {len(self.all_records)}", flush=True)
            self.next(self.select_by_val)

        @step
        def select_by_val(self):
            """OPTIONAL: per (cell, method, eval_k, seed) keep the best-val config.

            Only meaningful when a method carries an internal sweep under the
            independent split. Under the sequential split val_score is None and
            every record passes through. Remove this step entirely if no method in
            the standard ever sweeps with validation selection.
            """
            from collections import defaultdict

            grouped: dict = defaultdict(list)
            for rec in self.all_records:
                key = (_cell_key(rec["cell"]), rec["method"], rec["seed"])
                grouped[key].append(rec)
            selected = []
            for recs in grouped.values():
                # Filter to recs with a real val_score before calling max; a mixed
                # group (some None, some float) would raise TypeError in max().
                scored = [r for r in recs if r["val_score"] is not None]
                if not scored or len(recs) == 1:
                    selected.append(recs[0])  # no val set, nothing to choose, or all None
                else:
                    selected.append(max(scored, key=lambda r: r["val_score"]))
            self.selected_records = selected
            self.next(self.aggregate)

        @step
        def aggregate(self):
            """Bootstrap-CI aggregation + paired lifts (the standard owns this).

            Emits the PINNED aggregate-output schema (conf/SCHEMA.md):
              self.aggregate_results -> [{cell, method, primary_metric_mean}]
              self.lift_results      -> [{cell, lift_mean, lift_lo, lift_hi,
                                          ci_excludes_zero, n_seeds}]
            """
            from collections import defaultdict

            n_res = self.bootstrap_cfg.get("n_resamples", 1000)
            bseed = self.bootstrap_cfg.get("seed", 0)

            grouped: dict = defaultdict(list)
            for rec in self.selected_records:
                grouped[(_cell_key(rec["cell"]), rec["method"])].append(rec)

            self.aggregate_results = []
            for (ck, method), recs in grouped.items():
                vals = np.array([r["test"]["primary"] for r in recs])
                m, _, _ = bootstrap_ci(vals, n_resamples=n_res, seed=bseed)
                # Generic field name: <primary_metric>_mean (here recall_at_k_mean).
                self.aggregate_results.append({
                    "cell": dict(recs[0]["cell"]), "method": method,
                    "recall_at_k_mean": m})

            # Seed-paired lift: topk_ranking - baseline_ce, per cell.
            self.lift_results = _seed_paired_lift(
                self.selected_records, "topk_ranking", "baseline_ce",
                cell_keys=["separation", "eval_k"], bseed=bseed)
            print(f"[aggregate] {len(self.aggregate_results)} cell/method summaries, "
                  f"{len(self.lift_results)} paired lifts", flush=True)
            # Fan out to the static analysis branches (>=2).
            self.next(self.an_metric_trend, self.an_auxiliary_gap)

        @card
        @step
        def an_metric_trend(self):
            """ANALYSIS BRANCH 1 (list-shaped): primary-metric trend per method."""
            from metaflow import current

            rows = analyze_metric_trend(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "rows": rows}
            if rows:
                current.card.append(Markdown(f"## Metric trend ({len(rows)} cells)"))
            self.next(self.join_analyses)

        @card
        @step
        def an_auxiliary_gap(self):
            """ANALYSIS BRANCH 2 (scalar-shaped): auxiliary-metric gap diagnostic."""
            from metaflow import current

            result = analyze_auxiliary_gap(self.all_records, self.experiment_name)
            self.an_result = {"experiment": self.experiment_name, "result": result}
            if result:
                current.card.append(Markdown(f"## Auxiliary gap: {result}"))
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            """Collect analysis artifacts; propagate shared artifacts from inputs[0].

            all_records can carry nn.Module models (requests_model), which do not
            compare equal across pickled branch copies -> merge_artifacts would
            raise. Every an_* branch is a PURE READER (sets only an_result), so
            assigning the shared artifacts from inputs[0] is safe.
            """
            for i, inp in enumerate(inputs):
                assert hasattr(inp, "all_records"), (
                    f"join_analyses: inputs[{i}] missing 'all_records'")
            # Key by (experiment, branch step name): stable and meaningful.
            # id(inp) is non-deterministic and meaningless after the run ends.
            self.analyses = {
                (inp.an_result.get("experiment", ""), inp.current_step): inp.an_result
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
            """Print a concise summary of every analysis that produced results."""
            print("\n=== ReferenceFlow analyses ===", flush=True)
            for a in self.analyses.values():
                print(f"  {a.get('experiment')}: {a.get('rows') or a.get('result')}",
                      flush=True)
            self.next(self.end)

        @step
        def end(self):
            """Final summary."""
            print(f"\n=== ReferenceFlow complete: {self.experiment_name} ===",
                  flush=True)
            print(f"cell/method summaries: {len(self.aggregate_results)}", flush=True)

    if __name__ == "__main__":
        ReferenceFlow()
