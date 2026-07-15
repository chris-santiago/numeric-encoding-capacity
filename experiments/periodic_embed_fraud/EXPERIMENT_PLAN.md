# Experiment Plan — Step 6 (Gate 1), periodic embeddings on the fraud GRU

**Review mode:** debate. **Case verdict:** `empirical_test_agreed` (converged round 3, all-ACCEPT
early-exit). **Report mode:** conclusions_only. Source of truth: `HYPOTHESIS.md`.

The debate cleared the PoC's H1 mechanism finding as `defense_wins` but DEFERRED the two questions
that decide the reference-model case to this experiment, with variance-derived pre-registered thresholds.

## Pre-flight checklist (debate-derived)

| # | Source | Finding | Point verdict | Item | Verification method | Status |
|---|--------|---------|---------------|------|---------------------|--------|
| 1 | DEFER | F2 | ETA | Capacity-controlled H2: does learned-periodic on log-dt beat **raw** under the GRU? | Empirical test (arm `dt_periodic` − `cyc_sincos`), CI-excludes-zero bar | CLOSED (test specified) |
| 2 | DEFER | F3 | ETA | Static→GRU proxy: does H1 (`cyc_periodic` − `cyc_sincos` ≈ 0) hold under the **real GRU**? | Empirical test under GRU, tolerance = bootstrap CI of the lift | CLOSED (test specified) |
| 3 | defense commit | F1 | defense_wins | Scope CONCLUSIONS.md to "directional signal on synthetic best-case targets," not "calibrated prior" | Documentation | CLOSED (commitment recorded) |
| 4 | defense commit | F4 | defense_wins | Attribute two_bumps linear per-raw (+0.659) as capacity-confounded; map H2 to MLP/GRU-head results only | Documentation | CLOSED (commitment recorded) |
| 5 | non-negotiable | — | — | Trivial baseline arm present | `tab_logreg` arm | CLOSED |

## Pre-registered "meaningful effect" rule (resolves F2/F3 threshold residuals)

No arbitrary constants. A lift is "meaningful" iff its **paired bootstrap 95% CI excludes zero**
(N=1000). For the F3 proxy check, the PoC predicts `cyc_periodic − cyc_sincos ≈ 0`; the proxy holds
if the GRU lift's CI **overlaps zero**. Seed standard deviation is reported as a primary diagnostic.
These bars are variance-derived by construction.

## Precondition / diagnostic (run first — addresses F2 "real response shape unverified")

On the TRAIN window only, plot empirical fraud rate vs:
- hour-of-day, day-of-week (are the cyclic responses simple sinusoids, or multi-peaked with
  harmonics fixed sin/cos would miss? → a priori expectation for H1)
- log inter-transaction-time decile (monotone/smooth → periodic unlikely to help H2; bumpy/
  non-monotone → periodic *could* help)

This diagnostic sets the prior and contextualizes every verdict. Halt interpretation if the GRU
fails its precondition (must encode the time signal at all — checked vs the trivial baseline).

## Arms (shared GRU backbone; ONLY the per-step time-encoding block varies)

All arms hold fixed: amount (raw log-scaled), context numerics (`transactionToAvailable`,
`normMerchantName-accountNumber60dCount`), `cardPresent`. `*FraudTrend` columns excluded (leakage).

| Arm | hour / day-of-week | inter-txn time (log-min) | Role |
|-----|--------------------|--------------------------|------|
| `base_raw` | raw scalar | raw scalar | reference (no time encoding) |
| `cyc_sincos` | fixed sin/cos (P=24, 7) | raw scalar | **reference-style encoding** |
| `cyc_periodic` | learned periodic (PLR) | raw scalar | H1 lever |
| `dt_periodic` | fixed sin/cos | learned periodic (PLR) | H2 lever |
| `all_periodic` | learned periodic | learned periodic | combined |
| `tab_logreg` | last-step tabular logreg | — | **trivial baseline (non-negotiable)** |

## Pre-specified comparisons → verdicts

| ID | Comparison | Critique-right (refutes the null) | Defense-right (confirms the null) | Ambiguous |
|----|-----------|-----------------------------------|-----------------------------------|-----------|
| H1 | `cyc_periodic − cyc_sincos` | CI-separated **positive** (learned beats fixed) | CI **overlaps 0** (no harmonic gain) | — |
| H2 | `dt_periodic − cyc_sincos` | CI-separated **positive** > baseline (GRU can't learn dt transform) | CI overlaps 0 / marginal | small positive within noise |
| ENC | `cyc_sincos − base_raw` | any encoding helps the GRU (CI>0) | raw is sufficient (CI overlaps 0) | — |
| ALL | `all_periodic − cyc_sincos` | combined periodic helps | no combined gain | — |
| BASE | every arm − `tab_logreg` | GRU earns its place (CI>0) | GRU underperforms tabular (CI<0) | — |

## Implementation

- Real account-sequence data `~/Dropbox/GitHub/demo/tmp/data`; targets from `train/valid/test.parq`,
  causal per-account history from `transactions.parq` (reuse cycle-3 `seq_flow` make_data pattern).
- Periodic/scaler params fit on **train-period history only** (causal). Left edge-padding.
- PLR per-step embedding: k learned frequencies (init `N(0,σ²)`), → sin/cos → Linear → ReLU.
  σ spot-checked (the one tuning knob the PoC flagged as dominant).
- Sequence length: single **L=32** (length axis dropped at Gate 1 — the time-encoding effect is
  largely orthogonal to sequence length; account p90≈47 → L=300 not validatable here regardless,
  carry that caveat). 3 seeds. PR-AUC with paired test-set bootstrap 95% CIs (N=1000).
- Early-stopped GRU on validation PR-AUC; epochs reported.

## Promotion

Promote to a config-driven Metaflow flow via `pipeline-init` (5 GRU arms × L=32 × 3 seeds +
baseline = clearly multi-cell). Reuse cycle-3 `seq_flow` as the structural starting point. Four-gate
enforcement: prevent → lint (`flow-lint.py` exit 0) → review (`pipeline-reviewer`, no FAIL) → prove
(`determinism-check.py`). Determinism contract: **single_worker** (torch GRU, as in cycle 3).
