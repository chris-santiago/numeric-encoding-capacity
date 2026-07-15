## Hypothesis — Cycle 1

**Claim:** An MLP trained on piecewise-linear-encoded (PLE) numeric features achieves
higher AUC-ROC than the *same* MLP trained on raw standardized numeric features, on a
synthetic binary classification task whose target depends **non-monotonically** on the
numeric inputs.

**Mechanism:** PLE replaces each scalar feature `x` with a bin-aware piecewise-linear
vector. For `x` falling in bin `t` with boundaries `[b_{t-1}, b_t]`, the encoding is
`[1, …, 1, (x − b_{t-1})/(b_t − b_{t-1}), 0, …, 0]` — all bins below `t` saturated to 1,
the active bin holding the fractional position, all bins above set to 0. Each bin boundary
becomes a low-cost inflection point. Non-monotonic / piecewise thresholds that a raw-input
MLP must approximate with many hidden units become near-linear in PLE space, so the MLP can
fit the same decision surface with less capacity, less data, and less optimization effort.

**Signal:** Non-monotonic / piecewise feature→target structure — event probability that
peaks in a middle band of a feature, or steps at internal thresholds. This is the structure
PLE is specifically designed to exploit; a raw scalar gives a monotone-friendly MLP nothing
to latch onto cheaply.

**Expected observable:**
- PLE-MLP AUC-ROC exceeds raw-MLP AUC-ROC with **non-overlapping bootstrap 95% CIs** on the
  non-monotonic target.
- The gap **collapses** (CIs overlap) on a **linear-target control** where PLE confers no
  representational advantage. This control is the falsification lever: if PLE wins equally on
  a purely linear target, the benefit is not the hypothesized mechanism (it is just "more
  parameters / more features").
- Both MLPs beat the trivial baseline (logistic regression on raw features) on the
  non-monotonic target.

## Evaluation Metrics

**Primary:** AUC-ROC — threshold-free, standard for binary classification, directly
comparable across encodings evaluated on identical data splits. Reported with bootstrap 95%
confidence intervals (N=1,000, percentile method).

**Secondary:** Average precision (PR-AUC) as a robustness check if any class imbalance is
introduced into a condition.

**Domain:** ple_numeric
