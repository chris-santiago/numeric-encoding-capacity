# `conf/` schema (pipeline standard)

Config groups: **`data/`**, **`method/`**, **`experiment/`**, **`training/`**.
(The reference build named these `geometry/` and `arm/`; this standard uses the
generic `data/` and `method/`. The aggregate field is `method`, never `arm`, and
the headline metric is `<primary_metric>`, never `coverage`.)

`config.yaml` carries the Hydra `defaults` list (`data`, `experiment`, `training`,
`_self_`) plus global knobs (`seeds`, `bootstrap`, `max_workers`). The `method/`
group is **not** in the defaults list: methods are named in each experiment's
`methods:` list and loaded by name from `conf/method/<name>.yaml`.

## `experiment/` — pinned keys

Every experiment YAML pins exactly these keys:

| key | type | meaning |
|---|---|---|
| `name` | str | tags every record; the static `an_*` branches filter on it |
| `axes` | dict[str, list] | cartesian-product axes; one cell key per axis |
| `data_axes` | list[str] | subset of `axes` keys that change the **generated dataset** |
| `methods` | list[str] | method config-group names compared in each cell |
| `method_overrides` | dict | per-method override map applied **before** sweep expansion |
| `split_convention` | str | `sequential` or `independent` (authoritative copy lives in the `data/` group) |
| `diagnostics` | list[str] | named diagnostics computed from train artifacts by the `an_*` branches |
| `requests_model` | bool | store the trained `nn.Module` in each record |
| `requests_scores` | bool | store raw scores + labels in each record |
| `determinism` | str | reproducibility contract: `order_independent` (default), `single_worker`, or `nondeterministic` — see below |

## Data-axis vs training-axis split

Each axis is implicitly tagged **data-affecting** (it is in `data_axes`) or
**training-affecting** (it is not). The `foreach` grain is `(data_axes + seed)`:

- The dataset is **generated once** per `(data-cell, seed)` branch.
- Every method/cell that shares that data trains **in-process** on the shared
  in-memory tensors — no redundant data generation, fewer branches, lower
  per-task subprocess overhead.

`separation` is a data axis (changes `data_spec`); `eval_k` is a training axis
(changes only how the metric / loss is parameterized, not the data).

## Axis-agnostic vs axis-dependent methods

Classified by `method.kind` (see `is_axis_agnostic_method` in the reference flow):

- **Axis-agnostic** (e.g. `kind: ce`): the trained model does not depend on the
  eval-axis, so the method trains **once** and is **evaluated at every** value of
  the training axis (e.g. `recall_at_k` is read off the same scores at each
  `eval_k`).
- **Axis-dependent** (e.g. `kind: topk`): the loss bakes in the axis value, so
  the model **retrains per value**.

**Raise on unknown `kind`** — never silently default a new method to agnostic.

## Merge priority (training vs method)

`training/` is **authoritative** for shared knobs (`epochs`, `lr`, `batch`,
`hidden`). A `method/` YAML carries only method-specific params (`margin`, `temp`,
`warmup_frac`, sweep axes) and must **omit** `epochs`/`lr`/`batch`. A method that
hard-set them would block a per-experiment training override — a known fidelity
trap. `epochs` in the training group may be an `int` or a dict keyed by a method
axis when one experiment needs per-cell epoch budgets.

## Determinism contract (`experiment.determinism`)

A promoted flow **declares** how reproducible it is; the determinism gate then
verifies the flow holds the contract it declared (it does not impose a universal
rule). The flow stores the value as a run artifact so a reader of any finished
run knows the claim without guessing.

- **`order_independent`** (default) — the aggregated outputs are identical across
  worker counts. Verify by running at two different `--max-workers` and diffing
  the run output contract (`uv run scripts/determinism-check.py <run_a> <run_b>`).
- **`single_worker`** — the flow is pinned to `--max-workers 1` because a
  dependency is nondeterministic under parallelism (e.g. gensim in the ATO
  investigation). Determinism is claimed only at one worker; verify run-twice
  identical at `--max-workers 1`. Cross-worker reproducibility is **not** claimed.
- **`nondeterministic`** — the experiment cannot guarantee reproducible aggregates
  by design. The determinism gate is N/A and skipped. This is the **escape hatch**:
  an explicit, recorded, reviewable declaration — not a silent gap. The bootstrap
  CIs already absorb the seed/run variance such an experiment reports.

## Run output contract — the SSOT read-surface (pinned)

A finished promoted run exposes its results as Metaflow artifacts in these pinned
shapes, so the `report` step, the `an_*` analysis branches, and any human
inspecting the run via the Metaflow client can read the conclusions **without
reading flow code**. The config schema above pins what goes *in*; this pins what
a trustworthy finished run exposes *out*. `<primary_metric>` is the workflow's
primary evaluation metric (here `recall_at_k`).

```text
aggregate.lift_results -> [
    {
      cell: dict,            # canonicalized (data_axes + training-axis) key
      lift_mean: float,
      lift_lo: float,
      lift_hi: float,
      ci_excludes_zero: bool,
      n_seeds: int,
    }, ...
]

aggregate.aggregate_results -> [
    {
      cell: dict,
      method: str,                       # NOT "arm"
      <primary_metric>_mean: float,      # e.g. recall_at_k_mean
    }, ...
]

an_<name>.an_result -> {experiment: str, rows: [...]}     # list-shaped analysis
                    or {experiment: str, result: {...}}   # scalar diagnostic

determinism -> str   # the declared contract, echoed as a run artifact
```

`cell` is the canonicalized `(data_axes + seed)` / training-axis key. Lists inside
a cell (e.g. a band axis `[lo, hi]`) must be converted to tuples before use as a
dict key (they are otherwise unhashable).
