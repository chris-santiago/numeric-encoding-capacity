# Report Addendum — Cycle 7 production re-evaluation

Step 9: the mechanism finding (value-curvature is a real, general PLE lever in an affine-read model)
re-examined against the operational reality of deploying PLE on amount/Δt/count features in the
reference model.

## The structural deficit is a deployment rule, not just a caveat

PLE pays a **~−0.04 PR-AUC structural cost** on any feature `log` already fits adequately (it is a
coarser representation of a shape `log` nails). This inverts the naive "PLE ≥ log, so switch
everything" reading:

- **Apply PLE selectively.** Net win only where a feature's risk shape genuinely departs from `log`.
  Blanket PLE on a feature set dominated by log-adequate features would *lose* ~0.04 per such feature.
- **Rank features by departure-from-log, target the top.** Benefit scales with curvature (cubic +0.265
  ≫ mild convex +0.092), so the highest-value targets are the most sharply log-mismatched features.

## Operational cost of PLE

- **Dimensionality:** K features × bins. 8 bins was sufficient here (16 added variance with no benefit)
  — favor few bins to limit serving width and retraining variance.
- **Training stability:** PLE's many correlated bins need adequate capacity / a stable early-stopping
  signal (consistent with the cycle-6 caveat); under-resourced runs show spurious negative lifts.
- **Monitoring:** quantile bin edges are fit on training data — they must be refit on drift, and a
  distribution shift silently re-bins the feature. `log` has no such moving part.

## Direction-only; the real test is the reference-model A/B

Magnitudes are synthetic. This cycle establishes **direction and mechanism** (curvature is a real lever;
deficit must be corrected; non-quantile curves benefit), not a real-fraud lift.

## Revised recommendation → Cycle 8

Cycle 8 (real-data v2) is **justified and re-scoped by this result:**
1. Target PLE at the most sharply **curved-in-value** count/recency features (e.g. the account dataset's
   `*60dCount`, `transactionToAvailable`), not amount (whose real curvature is weak, per cycle 2).
2. Use the **deficit-aware estimand**: compare against `log` with a log-optimal baseline feature to net
   out PLE's structural cost, so a small real benefit isn't masked (the error that cost this cycle two
   iterations).
3. Run at reference-model scale (affine-input GRU, L≈300) with seed-level CI-excludes-zero adoption bar.
4. Expected from Cycle 7: PLE beats `log` on genuinely log-mismatched features, modestly, net of deficit.

## Open questions
- Does the deficit persist at reference-model scale, or does more data shrink it (making blanket PLE
  safer)?
- Feature heterogeneity + strong correlation (beyond the mild arm here) — does per-feature curvature
  benefit survive when features are collinear?
