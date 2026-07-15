# Conclusions — PLE on transaction amount (IEEE-CIS fraud)

**Cycle 1 of the fraud extension. Review: debate (`empirical_test_agreed`). Mode: conclusions_only.**
Real IEEE-CIS data (150k temporal subsample, 3.5% fraud), temporal past→future split, 10 arms ×
5 seeds, via the promoted Metaflow flow (4/4 gates passed; determinism `order_independent`).
Primary metric PR-AUC with test-set bootstrap 95% CIs (N=1,000).

## Headline

**The hypothesis is REFUTED on real data.** PLE-encoding transaction amount does **not** improve
fraud PR-AUC — it is slightly *worse* than raw amount for both the linear model (−0.009) and the
MLP (−0.012), CI-separated, and even **worse than a simple quadratic** (−0.008). The strong
synthetic result (+0.559) did not transfer because the real fraud-vs-amount U-shape, while present
(U-shape ρ=0.76), is **weak** (~2.6× low/high-vs-middle, not ~45×).

**The decisive surprise:** the placebo arm — PLE applied to a count feature `C1`, meant as a null
control — produced a **+0.144** PR-AUC jump. PLE's real value on this data is for **heavy-tailed,
nonlinearly-related features a linear model cannot bend**, *not* for transaction amount. The
mechanism is "let a model fit a nonlinear 1-D feature→risk curve," which is general — exactly the
F1 "any non-linearity" critique, now confirmed empirically and in an unexpected place.

## PR-AUC by arm (mean [95% CI])

| Arm | PR-AUC | | Arm | PR-AUC |
|---|---|---|---|---|
| hgb_ple | **0.409** [0.383, 0.438] | | mlp_ple_log | 0.294 [0.271, 0.320] |
| hgb_raw | 0.401 [0.375, 0.431] | | logreg_raw | 0.153 [0.137, 0.172] |
| mlp_raw | 0.310 [0.286, 0.335] | | logreg_quadratic | 0.153 [0.137, 0.171] |
| mlp_ple_raw | 0.299 [0.275, 0.324] | | logreg_ple_log | 0.144 [0.130, 0.162] |
| logreg_ple_placebo | 0.297 [0.274, 0.322] | | logreg_ple_raw | 0.144 [0.130, 0.161] |

GBDT dominates (0.41) ≫ MLP (0.31) ≫ linear (0.15). See `summary_prauc_by_arm.png`,
`lift_forest.png`, `fraud_curve_real.png`.

## Debate scorecard (real-data verdicts)

| Pt | Topic | Debate disposition | Empirical verdict | Evidence |
|----|-------|--------------------|-------------------|----------|
| H-main | PLE-amount beats raw | — | **REFUTED** | logreg PLE−raw −0.009 [−0.015, −0.002]; mlp PLE−raw −0.012 [−0.019, −0.004] (both CI-excl-0, negative) |
| F1 | PLE-specific vs any non-linearity | DEFER → T-F1 | **Critique validated** | PLE−quadratic = −0.008 [−0.014, −0.002]: PLE is *worse* than a 2-term polynomial |
| F2 | PLE on raw vs log amount | DEFER → T-F2 | **No difference** | ple_log−ple_raw = +0.000 [−0.001, +0.001] (linear); −0.005 (MLP) |
| F5 | Interaction gap (hgb_ple−hgb_raw) | DEFER → T-F5 | **Tiny lift only** | +0.008 [+0.001, +0.014] — CI-significant but marginal, on the already-best model |
| F3 | Drift bin-saturation | defense | **Negligible** | test top-bin 4.3% ≈ train 4.2% |
| F6 | class_weight asymmetry | defense | Resolved | harmonized (no weighting) across all arms |
| lever | PLE no help on a "monotone" feature | T-lever | **Misfired → discovery** | PLE on C1 = **+0.144** [+0.126, +0.161]: lever premise wrong (monotone ≠ linear-in-value) |

## Per-test detail

**H-main / T-MLP — refuted.** On real data PLE-on-amount slightly *hurts* every neural/linear arm
(CI-separated negatives). The amount signal that exists is already captured by the raw scalar plus
the rest of the model; PLE's extra parameters add variance, not signal. `lift_forest.png`.

**T-F1 — the cycle-1 lesson, confirmed on real data.** PLE is not merely non-special; it is
*worse* than `log_amt + log_amt²`. Whatever curvature amount carries, a 2-parameter polynomial
captures it more efficiently than 24 PLE bins. The "is it PLE or any non-linearity" question
resolves firmly against PLE here.

**T-F5 — interaction gap.** GBDT (which models amount×other interactions natively) gains a tiny
+0.008 from PLE-amount — the only positive amount lift, but marginal and on the model that already
wins by a wide margin. For an interaction-aware reference model, PLE-on-amount is not worth the
complexity.

**T-lever — the discovery.** The placebo was meant to show "PLE doesn't help a monotone feature."
Instead PLE on `C1` (a heavy-tailed count, monotone *by rank* but highly *curved in value*) lifted
the linear model by +0.144 — larger than any model-architecture choice tested. PLE's benefit is
about **nonlinearity in feature value**, not non-monotonicity. The hypothesis aimed PLE at the
wrong feature.

**Precondition.** Real fraud-vs-amount is non-monotone (U-shape ρ=0.76, monotonic |ρ|=0.02) but
shallow — confirming the EDA and explaining the null amount result. `fraud_curve_real.png`.

## Hypothesis closure

- **Claim (PLE-on-amount improves fraud PR-AUC under a temporal split):** REFUTED. No lift
  (slightly negative) for linear and MLP; +0.008 for GBDT; worse than a quadratic.
- **Mechanism (PLE exploits the non-monotonic amount curve):** the curve exists but is too weak to
  matter; and PLE's real lever is nonlinearity-in-value, surfaced on `C1`, not amount.
- **For the reference model (extrapolation, hedged):** the *direct* evidence is that
  single-transaction tabular models — linear, a small MLP, and a GBDT — gain no meaningful lift
  from PLE-on-raw-amount (GBDT +0.008; linear/MLP negative). Extending this to the sequence model
  assumes its per-step amount encoder behaves like the MLP (plausible — both got no PLE benefit).
  **It does NOT test amount-in-context** (deviation from a card's history, velocity), which is the
  more likely place amount matters for a sequence model and which PLE-of-raw-amount would not
  capture anyway. Defensible recommendation: do not PLE-encode the *raw* amount scalar; the
  amount-in-context question is open and requires a sequence-model arm (see addendum / new
  hypothesis).

## New hypothesis (next cycle, not run here)

PLE applied to **heavy-tailed count/recency features** (e.g. the `C*`/`D*` families) materially
improves linear and shallow models on fraud — and may help the sequence model where those features
enter per step. The placebo result (+0.144 on `C1`) motivates a feature-targeted PLE study.
