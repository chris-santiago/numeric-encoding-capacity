# Hypothesis — Consolidated numeric-encoding-capacity experiment (cycles 1–8)

**Status:** promoted Metaflow flow (SSOT of the final, debated methodology). Synthetic — isolates the
mechanism; real-data applicability is a separate, documented outcome (`../gru_curvature_realdata/REAL_DATA_AB.md`).

## Claim (the governing law the whole line converges on)

**Whether a numeric encoding beats a `log` scalar for a per-step feature is decided by three things, and by
nothing else:**

1. **Architecture — does the model have a free per-step nonlinearity?** A fixed/learned basis is a lever
   only for a model whose per-step read is affine. A model with a free per-step nonlinearity (per-step
   MLP) rebuilds any 1-D transform itself, so encoding is redundant. A GRU is the subtle middle: its gates
   are *smooth* approximators, so they absorb smooth shapes but not localized ones.
2. **Risk-shape localization.** In an affine-read model the lever hierarchy is
   **sharp non-monotone ≫ monotone curvature ≫ smooth non-monotone (absorbed) ≈ log-linear (none)**.
3. **The dimensionality deficit.** Every encoding costs ~`d` extra dims; in a recurrence that cost is large
   and is usually the binding constraint, so a real lever can still net-lose.

Sub-claims (each carries a cycle): curvature is a *multivariate* phenomenon (invisible at K=1 under a rank
metric); the best encoder depends on sharpness (sharp → fixed **PLE**, smooth-curved → learned
**projection**); `raw < log` for an affine read; and the encoding question is *moot* unless a precondition
holds (the model beats a strong aggregate baseline and uses temporal order).

## Mechanism · Signal · Expected observable

- **Mechanism:** an affine per-step read can only form `W·e(x_t)`; a basis `e` supplies structure the read
  cannot. Gates/MLPs supply that structure themselves to the degree they are nonlinear per step.
- **Signal:** deficit-corrected `(arm − log)_condition − (arm − log)_log_linear`, seed-level.
- **Expected observable:** in `static_linear`, curvature and non-monotone estimands fire CI-positive (the
  positive control that licenses every other reading); in `gru`, sharp non-monotone fires, smooth is
  absorbed, monotone curvature is a real lever masked by a large deficit; in `perstep_mlp`, no arm beats
  `log`. Best encoder: PLE on sharp, projection on smooth-curved.

## Evaluation metric

**Primary:** PR-AUC (average precision) — the headline lift metric, rank-based.
**Auxiliary (calibration-sensitive, computed on temperature-calibrated probabilities so the comparison is
representational, not a calibration artifact):** log-loss, Brier. **Ceiling:** oracle (rank by true
log-odds). **Precondition comparator:** GBM + EWMA tabular baseline.

**Domain:** encoding-capacity.

## Controls (hard, wired into the verdict — never print-only)

- **Positive control (halts the run if it fails):** the estimand must detect the *robust* lever —
  `static_linear` PLE on the **sharp non-monotone** condition (deficit-corrected, K=6, CI excludes zero).
  If it cannot see the strongest known lever, it is blind and no GRU reading is trusted. Curvature is a
  *weak* lever (barely CI-clear even in Cycle 7), so it is a **reported finding**, not a halt condition —
  gating a hard halt on a noise-dominated lever would be fragile.
- **Curvature detection (reported):** `static_linear` curvature deficit-corrected lift — does the estimand
  also see the weaker monotone-curvature lever?
- **Negative control:** `log_linear` condition — no arm beats `log`.
- **Multivariate control (reported):** K=1 vs K=6 — curvature appears only multivariate.
- **Oracle ceiling:** reported per cell (context only; does not license attribution by itself).

## Fairness invariants (every arm calibrated + trained identically)

Equal feature weights; PLE and projection on the same log-space coordinate; all arms trained to convergence
under one regime (real capacity, val early-stopping, best-state, LR schedule) with identical per-epoch
reshuffle; **every arm temperature-calibrated on validation before magnitude metrics**; single fixed config
per arm (no sweep-override selection); one shared dataset per cell. Seed-level paired-t 95% CI + Holm.
