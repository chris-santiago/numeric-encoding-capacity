# Report Addendum — Cycle 8 reference-model re-evaluation (corrected)

> **Correction notice.** An earlier version of this addendum issued the deployment rule "do NOT PLE-encode a
> monotone feature (curved or not); the GRU handles it." That rule followed from the retracted refutation
> and is **wrong**. The corrected finding (`CONCLUSIONS.md`, `multivariate_control.py`) is that monotone
> curvature IS a per-step lever in the affine GRU — but one masked by a large PLE dimensionality deficit.
> The revised rule is below.

Step 9: the corrected mechanism finding (multivariate monotone curvature is a real per-step PLE lever in an
affine GRU, deficit-corrected +0.143, but the recurrence's PLE deficit is large ~−0.13 for K=6×12 bins)
re-examined against deploying per-step encoding on a reference sequence model.

## The deployment rule (corrected)

The decision is a **race between the curvature lever and PLE's dimensionality cost**, and in a GRU that cost
is much larger than in a static model:

- **The lever is real:** on multivariate monotone-curved features, PLE unlocks ~+0.14 of signal the affine
  GRU cannot reach on its own. Do not dismiss monotone (curved) features as "the recurrence handles them" —
  it does not.
- **But the cost is large and scales with bins × features:** ~−0.13 for 6 features × 12 bins fed per step
  into the recurrence, versus ~−0.03 in a static head. Blanket per-step PLE therefore **loses on net** at
  meaningful feature counts — the raw benefit reads ≈ 0 because the lever and the deficit nearly cancel.
- **So: target narrowly and keep bins low.** Encode only the few features with the strongest curved-or-sharp
  per-step risk, at few bins (8–12), so the summed dimensionality deficit stays below the summed lever. The
  deficit-vs-(K, bins) curve is the deployment-decisive measurement and is not yet characterized at scale.

This *supersedes* the earlier "curvature is not a GRU lever" rule and also refines Cycle 6's "try PLE on
{amount, Δt}": the lever exists for the right features, but PLE's per-step width must be paid for.

## The precondition gate is a validated, reusable instrument (unchanged)

Cycle 3 failed silently (GRU lost to tabular, ignored order, never checked). Cycle 8 built and validated the
gate: against a **GBM with EWMA aggregates** the GRU margin is +0.056 [+0.042, +0.070] and the order-shuffle
drop is +0.236 [+0.187, +0.286], both CI-clear. Any real-data encoding study must gate against the stronger
baseline or its "the GRU works here" premise is unearned.

## The estimand must carry positive controls (the hard lesson of this cycle)

The single most important methodological output: **an encoding null is uninterpretable without a positive
control that fires.** Two independent failures made the original refutation wrong — a single-curved-feature
design (curvature invisible under a rank metric, per Cycle 7's multivariate requirement) and an un-netted
dimensionality deficit. Neither was caught by the debate or by reading the raw numbers; both were caught
only by running the *identical estimand* on a static model where the lever is known to exist and confirming
it fires. Any real-data A/B must include: (a) a multivariate design, (b) the deficit-aware difference-of-
differences, and (c) a static-head positive control on the same features.

## Operational cost of PLE (reaffirmed and sharpened)

- **Dimensionality is the dominant cost in a recurrence, and it is larger than previously stated.** The
  ~−0.13 GRU deficit for K=6×12 bins is the number that decides deployment, not the ~−0.04 static figure.
  Favor few bins and selective targeting; the correlated-bins-into-the-recurrence cost is real (Cycle 6
  training-sensitivity caveat, now quantified).
- **Monitoring:** quantile bin edges are fit on training data and must be refit on drift; `log` has no
  moving part.

## Direction-only; magnitudes are synthetic

Magnitudes (both the +0.143 lever and the −0.13 deficit) are synthetic and PoC-scale. Cycle 8 establishes
**mechanism and direction** — curvature is a real GRU lever, masked by a large encoding deficit — not a
real-fraud number. The reference-model A/B, targeting a small set of curved/sharp features at few bins with
a static-head positive control, remains the only test that yields a deployable magnitude.
