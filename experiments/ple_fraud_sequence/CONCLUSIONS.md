# Conclusions — Sequence model, amount-in-context (real account fraud data)

**Cycle 1 of the sequence extension (macro-iteration Outcome C from the tabular fraud cycle).
Review: debate (`empirical_test_agreed`, converged). Mode: conclusions_only.**
Real account-sequence data (`~/Dropbox/GitHub/demo/tmp/data`, ~7% fraud, temporal split, targets
from curated split files + causal history from `transactions.parq`). Promoted Metaflow GRU flow,
6 arms × L∈{8,32} × 3 seeds, 4/4 gates passed (determinism `single_worker` verified). PR-AUC with
test-set bootstrap 95% CIs.

## Headline

For fraud detection on this real account data, **a tabular logistic regression beats the GRU
sequence model**, and **none of the amount treatments help**:
- **The sequence model does not earn its place** (the precondition fails): `seq_raw − tab_aggregate`
  = **−0.035 [−0.064, −0.005]** (L=8) and **−0.049 [−0.088, −0.018]** (L=32), CI-separated. A
  3-feature logreg (last amount + prior mean/std + context) outperforms the GRU.
- **An explicit deviation feature does not help** (sub-claim a refuted): `seq_dev − seq_raw` =
  −0.007 (L=8, CI overlaps 0) and **−0.023 [−0.046, −0.000]** (L=32, it *hurts*).
- **Per-step PLE of amount does not help** (sub-claim b): `seq_ple − seq_raw` = +0.008 / +0.009,
  CIs overlap zero at both lengths.
- **The GRU does not use cross-time order at all** (lever): `seq_raw − seq_raw_shuffle` ≈ 0 at both
  lengths (CIs overlap zero) — permuting the account history doesn't change performance.

## PR-AUC by arm (bootstrap mean [CI]; base rate ~0.072)

| Arm | L=8 | L=32 |
|---|---|---|
| **tab_last** | **0.187 [0.148, 0.236]** | **0.187 [0.148, 0.236]** |
| tab_aggregate | 0.186 [0.147, 0.236] | 0.186 [0.147, 0.235] |
| seq_ple | 0.158 [0.123, 0.204] | 0.146 [0.117, 0.190] |
| seq_raw_shuffle | 0.158 [0.128, 0.202] | 0.143 [0.118, 0.179] |
| seq_raw | 0.151 [0.123, 0.194] | 0.137 [0.110, 0.171] |
| seq_dev | 0.144 [0.107, 0.193] | 0.114 [0.092, 0.147] |

See `summary_prauc_seq.png`, `lift_forest_seq.png`.

## Debate scorecard (real-data verdicts)

| Pt | Topic | Debate disposition | Empirical verdict | Evidence |
|----|-------|--------------------|-------------------|----------|
| H-main (a) | deviation feature helps | DEFER (sev 9, F1) | **Refuted** | `seq_dev − seq_raw` −0.023 [−0.046, 0.000] @L32 (hurts); overlaps 0 @L8 |
| H-main (b) | PLE-of-raw adds little | — | **Confirmed (adds nothing)** | `seq_ple − seq_raw` CIs overlap 0 at both L |
| F2 | sequence model beats aggregates | DEFER (sev 8) | **Refuted — fails** | `seq_raw − tab_aggregate` −0.035*/−0.049* (GRU loses) |
| F3 | causal-only preprocessing | DEFER (sev 6) | Implemented | PLE edges + scaler fit on train-period history only |
| F4 | convergence | DEFER (sev 6) | Implemented | GRU early-stopped on val PR-AUC; epochs reported |
| F5 | length dependence | DEFER (sev 5) | **No help at either L** | lifts null/negative at L=8 and L=32; GRU worsens at L=32 |
| lever | shuffled history collapses advantage | — | **No advantage to collapse** | `seq_raw − seq_raw_shuffle` ≈ 0 — order unused |

## Interpretation

The amount signal these models can use is fully captured by the **last transaction plus simple
context** (a logreg on last-amount + prior mean/std + balance/velocity context is the best arm).
Adding a recurrent model, an explicit amount-deviation feature, or PLE binning does not improve —
and the GRU's insensitivity to history shuffling shows it is **not exploiting cross-time amount
context** on this data. The premise that motivated this cycle (a sequence model needs amount-in-
context, and PLE/deviation would help it) is not supported here.

## Honest caveats

- **GRU may be under-powered.** Small hidden size (24), ≤15 epochs (early-stopped), 6k training
  targets. A larger, tuned sequence model on the full data might close or reverse the gap to the
  tabular baseline. The **within-GRU** comparisons (PLE/deviation/shuffle add nothing) are
  internally valid regardless; the **sequence-vs-tabular** gap is the capacity-sensitive claim and
  is the one to retest at scale before treating "tabular wins" as final.
- **One dataset, modest subsample**, no hyperparameter tuning.

## Hypothesis closure

- **Claim (amount-in-context drives the sequence model's amount signal; deviation/PLE help):**
  REFUTED on this data. Deviation and PLE add nothing; the GRU ignores history order; tabular wins.
- **Through-line across all three cycles:** **PLE-encoding transaction amount provides no lift for
  real fraud detection** — tabular-linear, GBDT, MLP, and sequence-GRU alike. PLE's value (cycle 1)
  is for additive non-monotone numeric features a *weak* model can't bend; real transaction amount
  is not that feature, and stronger models (GBDT, sequence) capture what little amount signal exists
  without it.

## New hypothesis (next cycle, not run)

If amount-in-context matters for fraud, the lever is likely **engineered velocity/deviation features
fed to a strong tabular model (GBDT)**, not a sequence architecture or PLE — and should be tested at
full scale with a tuned model before concluding sequences add nothing.
