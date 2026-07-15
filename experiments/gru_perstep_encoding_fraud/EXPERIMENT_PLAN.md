# Experiment Plan — Step 6 (Gate 1), per-step encoding in an affine-input GRU (cycle 6)

**Review mode:** none (no debate, per user) — rigor carried by pre-registered controls + gates.
**Report mode:** conclusions_only. Source of truth: `HYPOTHESIS.md`. Step 6 = a **gated Metaflow flow**.

## Pre-registered gates and controls (replace the debate as the rigor mechanism)

| # | Control | Rule | Type |
|---|---------|------|------|
| 1 | **Precondition** | band-regime oracle (true band-match count) logreg PR-AUC ≫ base (≥ 2× base) | HARD GATE — fail ⇒ run void |
| 2 | **Positive control** | band regime `dense − scalar` seed-level CI **excludes zero (positive)** | HARD GATE — fail ⇒ no power, void |
| 3 | **Negative control** | monotone regime: `dense − scalar`, `ple − scalar` CIs overlap zero (no arm beats scalar) | expected clean |
| 4 | **Trivial baseline** | `tab_logreg` present | non-negotiable |

## Arms (shared GRU backbone, affine per-step input; only per-step numeric encoding varies)

`raw` (std raw amount/Δt — conditioning baseline) · `scalar` (log amount, log Δt — reference baseline) · `ple` (PLE per numeric, on raw) · `dense` (per-step Dense+ReLU on the log scalars — mechanism/positive control) · `tab_logreg` (trivial baseline on generic aggregates) · `oracle` (precondition probe). The four decision encodings: **raw / scalar(=log) / ple / dense**.

## Pre-specified comparisons → verdicts (seed-level paired-t 95% CI; CI-excludes-zero bar; Holm over the band decision family)

| ID | Comparison | "Bottleneck matters" (claim) | "No benefit" |
|----|-----------|------------------------------|--------------|
| H1 | band `dense − scalar` | CI-separated positive (also the positive-control gate) | overlaps 0 ⇒ void (no power) |
| H2 | band `ple − scalar` | CI-separated positive (fixed basis helps) | overlaps 0 / within ±0.005 |
| H3 | band `ple − dense` | quantifies cheap-basis vs learned-nonlinearity (either sign informative) | ≈ 0 ⇒ PLE matches Dense |
| LEN | band lift at L=300 vs L=32 | larger at 300 (benefit grows with aggregation length) | equal ⇒ length-independent |
| N1 | monotone `dense − scalar`, `ple − scalar` | — | overlaps 0 (negative control) |
| BASE | each GRU arm − `tab_logreg` | band: GRU arms beat tab | — |

## Conditions

Regimes {band, monotone} × **length axis {32, 300}** × encodings {scalar, ple, dense, tab_logreg} × 5 seeds. Synthetic sequences (amount, Δt log-normal); band = cross-feature-band burst (count ≥ K), monotone = mean-log-amount. PR-AUC; paired-t seed-level CIs (N seeds) + Holm + ±0.005 equivalence margin; train-only scalers/PLE edges. Determinism contract: **single_worker** (torch GRU).

## Promotion (four-gate, non-optional)

Promote via `pipeline-init`: scaffold flow/ + conf/; migrate seams `make_data(regime,length,seed)`, `build_model`, `train_arm` (registry, raises on unknown kind), `metric` (PR-AUC). foreach grain = (regime, length, seed). Gates: **lint** (`flow-lint.py` exit 0) → **review** (`pipeline-reviewer`, no FAIL) → **prove** (`determinism-check.py`, single_worker). Seed-level CI + Holm + precondition/positive/negative controls computed in the aggregate step.

## Out of scope

Real-world data (the reference-model A/B is the definitive external test); per-step-projection-vs-PLE cost/latency benchmarking; multiple band shapes; σ tuning. PLE only (learned-periodic excluded — it overfit in cycle 5).
