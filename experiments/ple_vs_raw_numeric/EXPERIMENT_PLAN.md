# Experiment Plan (Gate 1) — PLE vs raw numeric

**Review mode:** debate. **Case verdict:** `empirical_test_agreed` (converged, 2 rounds).
**Source:** debate-extracted DEFER findings F1–F4 + the HYPOTHESIS.md falsification lever.

## Pre-flight checklist

| # | Source | Finding | Point verdict | Item | Verification method | Status |
|---|--------|---------|---------------|------|---------------------|--------|
| 1 | debate | F1 | empirical_test_agreed | Mechanism: is PLE's benefit MLP-specific or general linearization? | Add LR-PLE arm (Test T1) | PENDING |
| 2 | debate | F2 | empirical_test_agreed | Capacity confound: is +gap just "bigger first layer wins"? | Param-matched raw-MLP-wide + RFF basis control (Test T2) | PENDING |
| 3 | debate | F3 | empirical_test_agreed | Convergence: was raw-MLP iteration-limited? | Report n_iter_ for all MLPs; max_iter sweep {300,1000} (Test T3) | PENDING |
| 4 | debate | F4 | empirical_test_agreed | Linearization: does PLE actually linearize the target? | Ridge R² on latent logit, PLE vs raw features (Test T4) | PENDING |
| 5 | debate | F5 | defense_wins | Noise-feature asymmetry | Immaterial — closed by debate (no action) | CLOSED |
| 6 | HYPOTHESIS.md | falsification lever | — | PLE must NOT win on a linear target | Linear-target control dataset (Test T5) | PENDING |
| 7 | protocol | trivial baseline | — | Beat logistic-regression-on-raw | Already included; carried into all conditions | PENDING |

## Datasets

- **A — non-monotonic** (primary): the PoC's `Σ ampⱼ·sin(freqⱼ·xⱼ)` logit target.
- **B — linear control** (falsification lever): logit = `Σ wⱼ·xⱼ` (purely linear), same feature
  distribution, base rate centered to ~0.5. PLE should confer **no** advantage here.

## Conditions (run on BOTH datasets unless noted)

| Arm | Encoding | Model | Purpose |
|-----|----------|-------|---------|
| `logreg-raw` | raw (standardized) | LogisticRegression | Trivial baseline (non-negotiable) |
| `logreg-ple` | PLE | LogisticRegression | **T1** — mechanism: does a linear model also gain from PLE? |
| `mlp-raw` | raw | MLP (64,64) | Reference |
| `mlp-ple` | PLE | MLP (64,64) | The hypothesis arm |
| `mlp-raw-wide` | raw | MLP widened so total params ≥ `mlp-ple` | **T2** — pure capacity control |
| `mlp-rff` | 192 random Fourier features | MLP (64,64) | **T2** — matched input dim, non-bin-local basis |

## Tests and pre-specified verdicts

**T1 — Mechanism (F1).** Compare `logreg-ple − logreg-raw` margin to `mlp-ple − mlp-raw` margin on Dataset A.
- *Defense (MLP-specific mechanism):* logreg gains little from PLE while the MLP gains clearly.
- *Critique (general linearization):* logreg-ple gains a comparable margin → PLE is feature engineering any model exploits.
- *Ambiguous:* both gain modestly with overlapping CIs.

**T2 — Capacity (F2).** Compare `mlp-ple` to `mlp-raw-wide` and `mlp-rff` on Dataset A (param counts reported).
- *Defense (representation):* `mlp-ple` still beats both capacity/dimension-matched arms with non-overlapping CIs.
- *Critique (capacity-driven):* `mlp-raw-wide` (or `mlp-rff`) matches `mlp-ple` within overlapping CIs → the gap was capacity/dimensionality, not PLE structure.

**T3 — Convergence (F3).** Report `n_iter_` for every MLP; rerun at `max_iter ∈ {300, 1000}`.
- *Defense:* `mlp-raw` converges below the cap, or the gap is stable across `max_iter`.
- *Critique:* `mlp-raw` hits the cap at 300 and the gap narrows materially at 1000 → optimization confound.

**T4 — Linearization (F4).** Ridge regression predicting the latent logit from PLE features vs raw features; report R² (Dataset A).
- *Defense (mechanism):* PLE R² ≫ raw R² → PLE genuinely linearizes the non-monotonic target.
- *Critique:* PLE R² ≈ raw R² → linearization is not the mechanism; gains come from capacity/optimization.

**T5 — Falsification lever (HYPOTHESIS.md).** Compare `mlp-ple − mlp-raw` on Dataset B (linear) vs Dataset A.
- *Defense (mechanism confirmed):* the gap is large/CI-separated on A but collapses (CIs overlap) on B.
- *Critique (benefit not mechanism-specific):* PLE wins comparably on B → advantage is generic, not from non-monotonic structure.

## Statistics

- AUC-ROC primary metric, **bootstrap 95% CIs (N=1,000, percentile)** on every arm.
- **10 data/train seeds**; report per-seed AUC distribution and the mean `PLE − raw` gap with a bootstrap CI on the gap (guards against single-seed flukes — the PoC's +0.029 was one seed).
- All arms evaluated on identical splits per seed.

## Promotion judgment

**Recommendation: stay ad-hoc (single experiment script), do NOT promote to Metaflow.** This is a
self-contained toy experiment in `conclusions_only` mode; the Metaflow lint→reviewer→determinism
gate is overhead disproportionate to a quick mechanism test. One PEP 723 script (`ple_numeric_experiment2.py`)
implementing all five tests is the right granularity.

## Artifact

`ple_numeric_experiment2.py` — implements T1–T5 + bootstrap CIs + seed sweep, prints a structured
summary and writes `stats_results.json`.
