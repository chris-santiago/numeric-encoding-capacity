# Component-role contract (pipeline standard)

These are the **component seams** of the standard: the four roles your domain code
must fill. The DAG (Seam 2) and the library helpers below are generic and provided
by the standard; the four roles here are domain-specific. Implement them as plain
**module-level functions** (an `nn.Module` backbone built by `build_model`, the
rest free functions), so they are unit-testable via a bare `import flow` without
metaflow. Reference bodies live in `reference_flow.py`.

## The four roles

```python
make_data(data_spec, data_axes, seed)          -> Dataset
build_model(model_spec)                         -> nn.Module
train_arm(method_spec, data, seed, train_cfg)   -> TrainResult(model, scores, val_score)
metric(scores, labels, **cfg)                   -> float
```

### `make_data(data_spec, data_axes, seed) -> Dataset`

Deterministically generate the dataset from `data_spec` (the resolved `data/`
group), the per-cell `data_axes` overrides, and the branch `seed`. Returns the
train/test (and optional val) tensors. Honors `data_spec["split_convention"]`:

- `sequential`: one RNG advanced train-then-test (no val set).
- `independent`: independent seed offsets per split (supports val selection).

The data-axis overrides (the `data_axes` subset of the cell) are applied on top of
`data_spec` here — this is what makes the dataset-keyed `foreach` correct.

### `build_model(model_spec) -> nn.Module`

Build the backbone from a model spec (e.g. width, input dim). Standalone
`nn.Module`; `train_arm` composes it. Keep model construction out of the
`LightningModule`/flow step — define the architecture here.

### `train_arm(method_spec, data, seed, train_cfg) -> TrainResult`

**Dispatched on `method_spec["kind"]` via a registry** (see below). Returns
`TrainResult(model, scores, val_score)`:

- `model`: the trained `nn.Module` (stored in a record only when
  `requests_model`).
- `scores`: test scores (stored only when `requests_scores`; always used to
  compute `metric`).
- `val_score`: the val-selection scalar, or `None` under `sequential` split.

`train_cfg` is the resolved `training/` group, merged into `method_spec` as
**fallback** defaults (the method YAML wins for keys it sets; training wins for the
shared `epochs`/`lr`/`batch`/`hidden`). Reshuffle symmetry across methods is a
fidelity invariant: all methods must reshuffle (or not) consistently per epoch.

#### Registry dispatch

```python
TRAIN_REGISTRY = {
    "ce":   _train_ce,      # axis-agnostic
    "topk": _train_topk,    # axis-dependent
}

def train_arm(method_spec, data, seed, train_cfg):
    kind = method_spec["kind"]
    try:
        fn = TRAIN_REGISTRY[kind]
    except KeyError:
        raise ValueError(f"Unknown method.kind {kind!r}")   # never silently default
    return fn(method_spec, data, seed, train_cfg)
```

A parallel `is_axis_agnostic_method(kind)` classifies the method for the
dataset-keyed `foreach` and **raises on unknown kind** (do not default a new
method to agnostic — it would silently mis-train per-axis methods once).

### `metric(scores, labels, **cfg) -> float`

The workflow's **primary evaluation metric** — not a new concept. Binds to the
existing headline metric (in the exemplar, `recall_at_k(scores, labels, k=...)`).
The `cfg` kwargs carry the training-axis value (e.g. `k` from `eval_k`). Auxiliary
metrics (AUROC, AUPRC) are computed alongside and stored in the record's `test`
block, but `metric` is the one that drives the paired lift in the run output contract.

## Library-provided (generic — NOT seams)

- **`bootstrap_ci(values, n_resamples, seed) -> (mean, lo, hi)`** — percentile
  bootstrap; the standard owns this, used by `aggregate` for CIs and paired lifts.
- **Per-config record schema** — the `{experiment, cell, seed, method, config,
  test, diagnostics}` record dict and the `(cell, method, seed)` keying are owned
  by the standard (see `conf/SCHEMA.md`).

## Diagnostics are NOT components

Gradient-mass, representation-probe, geometry readouts, etc. are **DAG analysis
branches** (Seam 2), each its own `@card @step` consuming `train` artifacts. They
are not reusable component functions and must never run inside `train`
(train->analyze separation is a hard invariant).
