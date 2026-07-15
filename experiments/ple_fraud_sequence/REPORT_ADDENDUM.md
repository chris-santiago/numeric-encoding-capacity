# Reference-Model Re-Evaluation Addendum — Sequence model, amount encoding (fraud)

Step 9. The reference model is a sequence model over per-account history. This addendum maps the
result onto that target and bounds it honestly.

## Recommendation

**Do not add PLE-encoding or an explicit amount-deviation feature to the reference model's amount
input.** On this real account data: PLE adds nothing (CIs overlap zero at both sequence lengths),
the deviation feature adds nothing and hurts at L=32, and the GRU ignores history order entirely
(shuffling is harmless). Keep transaction amount as a raw (log-scaled) per-step input. This is the
direct, internally-valid result and is consistent across all three investigation cycles.

## Assumptions / bounds for the reference model

1. **Within-architecture claims are robust.** "PLE vs raw vs deviation, and order vs shuffled" were
   all tested *inside the same GRU* on real data — those comparisons are clean. The amount-encoding
   choice does not matter for the sequence model.
2. **The "tabular beats GRU" claim is capacity-sensitive — do not over-read it.** The GRU here was
   small (hidden 24), early-stopped, on a 6k-target subsample with no tuning, and it underperformed
   a 3-feature logistic regression (`seq_raw − tab_aggregate` ≈ −0.04 to −0.05, CI-separated). A
   full-scale, tuned sequence model might beat the tabular baseline — but that would not
   change the amount-encoding conclusion (encoding amount cleverly still wouldn't be the lever).
3. **Amount-in-context, as a *raw-amount sequence* signal, is not being exploited here** (shuffle is
   harmless). If amount-in-context matters operationally, it more likely surfaces via *engineered*
   velocity/deviation features in a strong tabular model than via sequence encoding of raw amount.

## The four deployment constraint areas

1. **Retraining dynamics.** PLE adds a per-feature edge artifact to refit on drift; a deviation
   feature adds a stateful running-statistic to maintain. Both are pure cost here (no measured
   benefit). Skip.
2. **Update latency.** Per-step PLE / deviation add preprocessing to every inference step for no
   gain. Keeping amount raw is the cheapest and equally accurate option.
3. **Operational complexity.** Avoiding PLE/deviation on amount removes versioned-artifact and
   monitoring burden with no accuracy cost. The strong signal lives in context features
   (balance/availability, velocity counts, card-present) — invest monitoring there.
4. **Failure modes.** Since neither treatment helps, adding them is downside-only (more drift
   surface, more code). The simplest amount handling is also the most robust here.

## Limits (what this does NOT establish)
- Not that amount is unimportant — only that *encoding* it (PLE / deviation) and *sequencing* raw
  amount add nothing on this data.
- Not a verdict on a tuned, full-scale sequence model vs tabular (capacity-sensitive; retest).
- One dataset, one architecture (GRU), modest subsample, no tuning.

## Deployment
Keep amount as a raw log-scaled per-step input. Spend model-improvement budget on context/velocity
features and model tuning, not on amount encoding. If revisiting amount-in-context, test engineered
velocity features on a tuned GBDT at full scale before any sequence/PLE encoding of amount.

## Cross-cycle synthesis (the portable takeaway)
Across synthetic numeric, real tabular (IEEE-CIS), and real account sequences, **PLE-encoding
transaction amount never improved real fraud detection.** PLE's genuine value (cycle 1) is for
additive, non-monotone numeric features that a *weak* linear model cannot bend; real transaction
amount is not such a feature, and any model strong enough to matter (GBDT, MLP, sequence) extracts
its modest signal without PLE.
