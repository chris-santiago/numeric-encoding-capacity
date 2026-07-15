# Conclusions — Learned periodic embeddings on fraud-GRU time features (real account data)

> ## ⛔ STATUS: VOID / INVALIDATED — precondition failure (no construct validity)
> This experiment does **not** test its hypothesis and provides **no evidence** about learned
> periodic embeddings for fraud. The precondition for the test — that the time features carry fraud
> signal — **failed** on the demo dataset (time-only PR-AUC 0.078 vs 0.064 base; the features are
> near-noise). Comparing *encodings* of an uninformative feature is vacuous: with no underlying
> signal, no encoding can produce lift, so the H1/H2 nulls were guaranteed a priori and carry zero
> evidential weight. Per ml-lab's precondition-verification rule, the diagnostic that found the weak
> signal should have **halted** the experiment; reporting the nulls as "H1/H2 confirmed" was an
> error. The debate's F2 finding (verify the real response shape first) foresaw exactly this.
>
> **What survives:** nothing about fraud. The Step-1 PoC mechanism demonstration (on synthetic
> features with controlled, real signal) is unaffected, but it is only a mechanism demo, not a fraud
> result. **A valid test requires a dataset where the time features are informative** (the reference model —
> where SHAP confirms Δt and sin/cos are highly important — or synthetic data with construct-valid,
> structured time signal). See `REPORT_ADDENDUM.md`. The results below are retained only as a record
> of what was run; they are not to be cited as findings.

**Cycle 4 of the fraud-encoding line. Review: debate (`empirical_test_agreed`, converged round 3).
Mode: conclusions_only.**
Real account-sequence data (`~/Dropbox/GitHub/demo/tmp/data`, ~7% fraud, temporal split, targets from
curated split files + causal history from `transactions.parq`). Promoted Metaflow GRU flow, 6 arms ×
L=32 × 3 seeds, **4/4 gates passed** (lint, fidelity review, determinism `single_worker` verified).
PR-AUC with test-set bootstrap 95% CIs (N=1000). Pre-registered bar: a lift counts only if its CI
excludes zero.

## Headline

**Learned periodic embeddings provide no lift for the fraud GRU's time features.** Both pre-registered
sub-claims hold (the null was confirmed), and — consistent with cycle 3 — the GRU does not beat a
trivial tabular baseline:
- **H1 (cyclic, known period):** `cyc_periodic − cyc_sincos` = **−0.006 [−0.020, +0.010]**, CI
  overlaps zero. Learned frequencies do **not** beat fixed sin/cos on hour-of-day / day-of-week.
- **H2 (non-periodic dt):** `dt_periodic − cyc_sincos` = **−0.008 [−0.021, +0.006]**, CI overlaps
  zero. A learned Fourier basis on inter-transaction time does **not** beat the raw scalar.
- **No time encoding beats raw at all:** `cyc_sincos − base_raw` = −0.010 [−0.029, +0.004] (overlaps 0).
- **The trivial tabular baseline wins:** `tab_logreg` (0.143) is the best arm; `GRU(periodic) − tab`
  = **−0.022 [−0.046, −0.002]*** (CI-separated, the GRU *underperforms* the two-line baseline).

## PR-AUC by arm (test bootstrap mean [CI]; base rate ~0.072, L=32)

| Arm | time encoding | PR-AUC |
|---|---|---|
| **tab_logreg** | last-step tabular (trivial baseline) | **0.143 [0.117, 0.180]** |
| base_raw | all time raw | 0.137 [0.116, 0.173] |
| all_periodic | cyclic + dt learned-periodic | 0.138 [0.114, 0.173] |
| cyc_sincos | fixed sin/cos (reference-style) | 0.127 [0.108, 0.154] |
| cyc_periodic | learned periodic (cyclic) | 0.121 [0.103, 0.149] |
| dt_periodic | learned periodic (dt) | 0.119 [0.102, 0.145] |

See `fig_summary_prauc.png`, `fig_lift_forest.png`.

## Precondition (why this was the expected outcome)

Diagnostic on train targets (`fig_response_shapes.png`) and a signal ablation: the time features carry
almost no fraud signal. Logreg PR-AUC by feature group — **time-only 0.078** (vs base 0.064),
amount-only 0.139, context-only 0.096. Fraud-vs-time curves wobble only ±0.015 around base. There is
*mild* bimodality in hour-of-day (the one place learned-periodic could in principle catch a harmonic
fixed sin/cos misses), but the amplitude is far too small for any encoding to extract. **When a
feature barely predicts the target, how you encode it cannot matter** — which is exactly what the arms
show.

## Debate scorecard (real-data verdicts)

| Pt | Topic | Debate disposition | Empirical verdict | Evidence |
|----|-------|--------------------|-------------------|----------|
| H1 | learned periodic vs fixed sin/cos (cyclic) | defense_wins (PoC) | **Confirmed (no gain)** | −0.006 [−0.020, +0.010], CI overlaps 0 |
| H2 | learned periodic on dt vs raw | DEFER → ETA (F2) | **Confirmed (no gain)** | −0.008 [−0.021, +0.006], CI overlaps 0 |
| F3 | static→GRU transfer of the H1 null | DEFER → ETA | **Proxy held** | GRU per-sin ≈ 0 (−0.006), matching the PoC's strong-head prediction (per-sin ≈ 0) |
| ENC | any cyclic encoding beats raw | — | **No** | cyc_sincos − base_raw −0.010, overlaps 0 |
| ALL | combined periodic helps | — | **No** | all_periodic − cyc_sincos +0.011, overlaps 0 |
| BASE | GRU earns its place vs trivial tab | non-negotiable | **No — GRU loses** | GRU(periodic) − tab −0.022* (sig); GRU(sin/cos) − tab −0.016 [−0.036, 0.000] |

## Interpretation

Gorishniy's periodic embeddings help deep models exploit numeric features that carry *learnable
nonlinear structure*. The fraud time features do not: hour/day-of-week and inter-transaction time are
near-flat predictors here, so a richer basis has nothing to recover. The F3 proxy check is the
satisfying part — the synthetic PoC predicted (under a strong head) that learned periodic would tie
fixed sin/cos; the real GRU confirms it (per-sin ≈ 0). The static→recurrent transfer held, and the
deferred empirical questions resolved in favor of the null.

## Honest caveats (debate commitments honored)

- **The PoC was a synthetic best-case mechanism demonstration, not a calibrated prior.** Its regimes
  (a clean 3rd harmonic; sign-changing bumps) were constructed to *favor* periodic; this real-data
  experiment is what adjudicates H1/H2 (F1 commitment).
- **The PoC's large `two_bumps` periodic-beats-raw number (+0.659) was capacity-confounded** — it was
  a weak-linear-head artifact. The capacity-controlled comparison (strong head / GRU) shows no gain,
  which the real data confirms (F4 commitment).
- **The GRU is under-powered** (hidden 24, modest subsample, no tuning, single L=32) and it
  underperforms the tabular baseline. The **within-GRU** encoding comparisons (H1/H2/ENC) are
  internally valid regardless; the GRU-vs-tabular gap is the capacity-sensitive claim (retest at
  scale before treating "tabular wins" as final).
- **L=300 not validatable here** (account p90 ≈ 47 transactions). The encoding null is expected to
  hold at the reference model's sequence length (the time signal is weak regardless of sequence length), but that is an
  extrapolation.
- **σ (PLR init scale) used at 2.0, k=16, not swept.** σ is the dominant knob, but with a near-zero
  time signal no σ is likely to manufacture lift.
- One dataset, one architecture.

## External validity — the reference model's time-feature importance (post-hoc correction)

**This is the load-bearing caveat.** The entire null rests on the precondition that the time features
carry little fraud signal *in this demo dataset* (time-only PR-AUC 0.078 vs 0.064 base). The user
reports that **in the reference model, SHAP analysis shows inter-transaction Δt and the cyclic sin/cos
date-time transforms are highly important.** The `demo/tmp/data` set is a *simulated* fraud process
that did not inject strong hour-of-day / velocity patterns, so it is unrepresentative on exactly the
dimension under test. Consequences:
- The "weak signal → encoding can't matter" argument **does not transfer to the reference model**, where the
  precondition fails. Cycle 4 measured the uninformative corner (encoding when there is little to
  encode).
- What cycle 4 *does* still support: (a) the PoC→GRU mechanism transfer held (per-sin ≈ 0 under a
  strong model); (b) keeping sin/cos and Δt is correct — SHAP independently confirms they matter.
- What is **reopened**: whether *learned* periodic beats *fixed* sin/cos (H1) or raw Δt (H2) is NOT
  settled for the reference model. The PoC prior: cyclic likely ties fixed sin/cos for a capable model unless
  the reference model has non-fundamental harmonics; Δt (if non-monotone) is the candidate worth a direct test.
- **Definitive test:** the one-line per-step encoding swap in the reference GRU (where time signal is
  SHAP-confirmed), CI-excludes-zero bar — analogous to the amount-PLE recommendation.

## Hypothesis closure

- **H1 — REFUTED-the-effect / CONFIRMED-the-null:** learned periodic does not beat fixed sin/cos on
  known-period cyclic features.
- **H2 — CONFIRMED-the-null:** learned periodic on inter-transaction time does not beat raw (the
  "marginal at best" prediction; here it is flat-to-slightly-negative).
- **Through-line across all four cycles (scoped to the demo dataset):** no learned numeric encoding
  — neither PLE (amount, cycles 1–3) nor periodic embeddings (time, cycle 4) — improved fraud
  detection *on the demo data*. **Caveat (see External validity):** the demo data under-represents
  time signal vs the reference model (where SHAP flags Δt and sin/cos as highly important), so the cycle-4
  time-encoding null is bounded to the weak-signal regime and does not settle the reference-model case. Amount is best left raw-log; time features are best left as
  the reference model's sin/cos + log encoding.

## New hypothesis (next cycle, not run)

The Gorishniy representation lever is now empirically closed for this feature set. If there is lift to
be found, it is in **signal**, not encoding: engineered velocity/deviation features on a tuned
GBDT/sequence model at full scale, evaluated against the same trivial baseline.
