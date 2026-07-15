# Peer Review R1 — Δt Encoding for Fraud Detection

**Artifact under review:** `TECHNICAL_REPORT.md` (results-mode technical report)
**Grounding documents:** `CONCLUSIONS.md`, `REPORT_ADDENDUM.md`, `HYPOTHESIS.md`, `stats_results.json`, `dt_encoding_experiment2.py`
**Review depth:** full
**Context:** Internal methodology study driving an encoding decision for the reference model (how to encode inter-transaction time Δt in a fraud-detection sequence GRU). Reviewed as an engineering-decision methodology study, not for an academic venue.
**Reviewer stance:** skeptical, specific, no rubber-stamping.

---

## 1. Summary

The study asks whether a richer encoding of an informative, non-monotone inter-transaction-time feature (Δt) beats the standard `log1p` transform, and whether the answer depends on model capacity. On synthetic data where Δt is informative by construction (a U-shaped fraud-vs-Δt risk), it compares six encodings (`raw`, `log`, `ple_raw`, `ple_log`, learned periodic `learned`, and a capacity-matched non-periodic `log_expand`) at two capacities (linear, MLP), across non-monotone and monotone regimes, over 5 seeds, with paired test-set bootstrap 95% CIs, a positive control, a negative control, a precondition gate, and a train-loss convergence check. The headline: under the MLP, PLE ties `log` and the learned periodic basis is significantly worse (attributed to overfitting); the positive control fires on the linear head, which the report uses to argue the MLP-row null is powered rather than vacuous. The recommendation is to keep `log1p` in the reference GRU.

---

## 2. Strengths

These are earned, not balancing praise.

1. **The control architecture is genuinely well-designed and correctly motivated.** The positive control (linear head on the U-shape) is the right instrument for the "is the null powered?" question, and it fires decisively (`learned − raw` = +0.204, CI-separated). This directly addresses the failure mode that voided the prior cycle (encoding comparison on an uninformative feature). The precondition gate (Δt-only MLP PR-AUC ≈ 0.80 ≫ 0.08 base; z-only floor ≈ 0.10 at base) is a clean construct-validity check and is verified in `stats_results.json` (z_only_floor: linear 0.1007, mlp 0.0997). This is a notably more rigorous experimental skeleton than most internal studies bring.

2. **The convergence check is the right idea against the most dangerous alternative explanation.** Distinguishing "learned overfits" from "learned is under-trained" via train BCE (learned 0.0925 < log 0.1034, verified in the JSON) is exactly the confound a careful reviewer would demand. Lower train loss with lower test PR-AUC is the correct signature of overfitting, and pre-specifying the direction of the check is good practice.

3. **The capacity confound is named, justified, and used deliberately rather than hidden.** Methods §"Encodings and capacity" explicitly states that PLE/learned carry more parameters on the linear row (that *is* the positive-control mechanism) and that the capacity-controlled comparison is the MLP row where every arm shares MLP capacity. This is honest and architecturally correct.

4. **The `log_expand` matched-capacity arm is a strong addition.** It isolates "periodic basis is harmful" from "extra capacity is harmful": `learned − log_expand` = −0.020 (CI-excluded, seed 0), and `log_expand` only ties `log`. This is the right way to rule out the "any expansion would lose" alternative, and it materially strengthens the σ-robustness argument in the limitations.

5. **The limitations section is unusually candid and is structured (threat / evidence-on-magnitude / mitigation).** The MLP→GRU transfer gap, the σ-dependence of the "worse" magnitude, the synthetic-data caveat, and the single-U-shape caveat are all surfaced, and the report correctly separates the σ-robust claim ("does not *beat* log") from the σ-dependent claim ("is *worse*"). The documents are internally consistent on this distinction (CONCLUSIONS "Honest caveats," ADDENDUM "Bounds").

6. **Cross-document coherence is high.** HYPOTHESIS pre-registers the three-part claim and the hard-gate halt criterion; CONCLUSIONS reports verdicts against that pre-registration; the ADDENDUM's recommendation follows from the verdicts. Numbers in the prose match `stats_results.json` (I verified the headline lifts, the PR-AUC table, the convergence losses, and the z-floor against the JSON — all reproduce). This is the discipline that makes the work reviewable.

---

## 3. Critical Issues

### MAJOR

**M1. All confidence intervals — including the headline null, the "learned is worse" verdict, and the positive control — are single-seed (seed-0) bootstraps, while the point estimates are 5-seed means. The CI machinery measures the wrong uncertainty for a deployment decision.**

This is the central issue and it undercuts the report's primary epistemic claim. In `dt_encoding_experiment2.py`, `paired_lift` and `boot_ci` are called on `runs[(reg, a, mt)][0]` — index `[0]`, i.e. seed 0 only (lines 237, 245–246). Every CI in the report (the "CI excludes zero" verdicts, the lift forest, the abstract's `[−0.030, −0.014]`) resamples *test rows within a single seed's single trained model*. The 5-seed means in the PR-AUC tables come from a different computation. The report never states that the CIs are seed-0-only; the abstract and Methods §"Metric and uncertainty" read as if the CI-excludes-zero bar is applied to the headline 5-seed quantity. It is not.

Why this is a problem, quantitatively. Test-row bootstrap captures sampling noise in the *evaluation set* but holds the *trained model and the training draw* fixed. The deployment-relevant uncertainty is run-to-run variation (different data draw + different init), which is the between-seed variance. These differ by ~3x here:
- Reported seed-0 bootstrap half-width for `learned − log`: ~0.008.
- Between-seed SD of the same difference, implied by the per-cell seed stds (learned σ=0.0184, log σ=0.0140): ~0.023.

A conservative (unpaired) between-seed t-test on the 5 seeds gives `learned − log`: diff −0.024, t ≈ −2.1, df ≈ 8 — i.e. **borderline, roughly p ≈ 0.07, not the clean "significantly worse" the single-seed bootstrap reports.** (A paired-across-seeds test would likely sharpen this back toward significance, since the seed-0 and other-seed structure is shared — but the report does not run that test, so the claim is currently unsupported at the seed level.) The headline "−0.022 [−0.030, −0.014], significantly worse" overstates the certainty available from a 5-seed experiment.

The same critique applies to the positive control, but there it is harmless: the positive-control lifts are so large (+0.12 to +0.24) relative to any plausible between-seed SD that they survive trivially. The damage is concentrated in the headline (the small, decision-driving effects).

Remediation: compute lifts and CIs at the seed level. The cleanest approach is a paired test across the 5 seeds (each seed contributes one paired `a − b` PR-AUC difference; report mean and a paired t / Wilcoxon CI, or a seed-cluster bootstrap). If you want to keep the row-bootstrap, present it as a secondary diagnostic and label it "seed-0 test-set bootstrap," not as the decision bar. With only 5 seeds the seed-level CI will be wide; that is the honest answer, and it strengthens (not weakens) the conservative recommendation to keep `log`.

**M2. A "tie" is asserted from a near-zero point estimate plus a CI that *includes* zero — but failing to reject a difference is not the same as establishing equivalence.** The report's core reference-model claim is "PLE ties `log`" (`ple_raw − log` = +0.000 [−0.002, +0.003]; `ple_log − log` = +0.002 [−0.000, +0.004]). A CI that spans zero is *absence of evidence for a difference*, not *evidence of equivalence*. The logical bar for "adopt neither because there is no gain" should be an equivalence argument: the CI should exclude any *practically meaningful* gain (you must define the margin — e.g. "no gain larger than +0.005 PR-AUC"). The seed-0 CIs happen to be tight enough that this might hold, but the report never frames it as equivalence and never states a margin, so the "ties → no gain → don't adopt" inference is currently a logical shortcut. This compounds with M1: at the *seed* level the equivalence bound will be much wider, and "PLE provides no meaningful gain" needs to be re-checked against the seed-level CI.

**M3. The "overfitting, not under-training" conclusion rests on a single-seed train loss and no validation-based evidence, and the experiment has no regularization/early-stopping baseline to support the overfitting label.** The convergence verification (`conv` in the script, line 251) uses `runs[("nonmono", e, "mlp")][0]` — seed 0 only. The learned arm also has the *largest* seed-to-seed AP variance in the table (σ=0.0184), so a single-seed train-loss comparison is the weakest possible support for a mechanism claim. More fundamentally: the training loop is a fixed 500 epochs of Adam at lr=1e-2 with **no validation split, no early stopping, and no weight decay** (lines 152–160). "Overfitting" is the correct *interpretation* of lower-train/higher-test, but the report elevates it to a verified mechanism and then uses it to justify a deployment warning ("silent degradation risk"). To call it overfitting rather than "this particular untuned training recipe generalizes worse," you need at minimum: (a) the train-loss gap across all 5 seeds, and (b) evidence that a standard regularizer (weight decay, early stop on a val fold, or σ tuning) closes the gap — which the report itself concedes "could shrink the gap toward a tie." As written, the mechanism claim is over-attributed; the honest statement is "the learned basis as configured (σ=2.0, no regularization, 500 epochs) generalizes worse."

**M4. The reference-model recommendation generalizes from an MLP on a single synthetic U-shape to a GRU over ~300-step real Δt sequences — a triple extrapolation the evidence does not cover.** The report is commendably explicit that the GRU was not tested (Limitations §1, CONCLUSIONS F3, ADDENDUM Bounds), and it gates the recommendation behind an A/B on the reference model. That honesty is why this is MAJOR-but-bounded rather than fatal. But the *framing* still overreaches in three compounding ways the limitations do not fully neutralize:
   - **Architecture.** An MLP on a single scalar Δt is not a proxy for a GRU consuming a *sequence* of Δt values where the encoding is applied per-step and the recurrence can compose temporal structure across steps. The capacity argument ("a capable model learns the transform from `log`") is plausible but is asserted, not demonstrated, for the recurrent case; sequence models can exploit a basis differently than a pointwise MLP (e.g., interactions between consecutive Δt). The report treats "capable model" as architecture-agnostic; that is the load-bearing assumption and it is unproven.
   - **Operational claims stated as fact without measurement.** The ADDENDUM asserts `log` is "the cheapest at inference," PLE "adds preprocessing/compute at every one of ~300 steps," and the learned basis is a "silent degradation risk." None of these latency/cost claims are measured in this study; they are plausible engineering priors presented in the register of findings. Label them as expectations, not results.
   - **The real Δt response shape is unknown.** The entire study is conditioned on a symmetric U-shape chosen by the authors. The report argues a harder shape "only makes `log` look better" — this is a reasonable conjecture for *richer-basis* gains, but it does not cover the case where the true response is, e.g., multi-modal or has a sharp threshold that `log` smears and PLE captures. The "conservative" claim holds for the family of smooth non-monotone shapes; it is not obviously conservative for all shapes the reference model might exhibit.

   The recommendation ("keep `log`, don't adopt PLE/learned, confirm via the reference-model A/B") is *defensible* as a default given costs, but the report should foreground that it is a **cost-asymmetry decision under acknowledged uncertainty**, not an evidence-established dominance of `log`. The current Conclusions section ("the evidence establishes that ... `log` is the right encoding under a capable model") is stronger than the evidence licenses for the GRU.

### MINOR

**m5. No multiple-comparison consideration across 11 pre-specified lifts.** Eleven paired CIs are computed and several drive verdicts. Even with pre-registration, the "CI excludes zero" bar applied 11 times inflates the family-wise error of at least one false positive. Not damning (the key effects are either huge or near-zero), but the negative-control `learned − log` mono (−0.037) and the headline `learned − log` (−0.022) are exactly the medium-sized effects most exposed to this. Mention it, and consider a Bonferroni/Holm note or simply report it as a caveat.

**m6. Base-rate inconsistency across documents (0.08 vs ~0.072).** HYPOTHESIS says ~8%; `stats_results.json` `base_rate` = 0.08 (the *target*); the report and CONCLUSIONS PR-AUC tables caption "base ~0.072." The 0.072 is presumably the realized empirical positive rate after the intercept calibration and sampling, but this is never stated, so a reader sees three numbers. Reconcile explicitly: "target 0.08, realized ≈ 0.072."

**m7. The bootstrap reuses RNG seed 0 for every CI (`np.random.default_rng(0)` inside `boot_ci`/`paired_lift`).** This makes resample indices identical across all paired lifts — fine and arguably desirable for *pairing within a comparison*, but it means the CIs across different pairs are not independent draws, and the whole CI suite is conditioned on one resample-index stream. Worth a one-line disclosure. (It does not bias any single CI.)

**m8. The learned-frequency-spectrum argument is suggestive, not evidential.** "Frequencies span 0.2 to 4.3, consistent with high-frequency components fitting noise" (Results §convergence) is a single-seed (seed-0) qualitative observation. It is fine as color but is presented adjacent to the verified train-loss claim in a way that may read as corroborating evidence. Either tie it to something (e.g., show that ablating the high-freq components recovers test PR-AUC) or downgrade it explicitly to "illustrative."

**m9. PR-AUC point estimates in tables (5-seed mean) and the bootstrap point estimates (seed-0) differ by up to 0.017 (e.g., `raw` MLP: seedmean 0.744 vs seed-0 boot 0.761), and the report mixes them.** The lift table CIs are seed-0 but the PR-AUC summary table is seed-mean, so a reader cannot reconstruct the lifts from the table (e.g., seed-mean `learned − log` = −0.024, but the reported lift is −0.022 from seed 0). This is a presentation trap that will confuse anyone who tries to check the arithmetic. State which quantity each table reports, and ideally make them consistent.

**m10. No related-work / prior-art grounding for the encoding choices.** The report names "the standard Gorishniy form" for PLE and "PLR" for the periodic basis but cites nothing. For an internal report this is acceptable, but since the recommendation is "don't adopt PLE/PLR," a one-paragraph pointer to where these methods do help (the tabular-DL literature reports PLE/PLR gains primarily for *non-recurrent* tabular models on heterogeneous features) would sharpen the scope claim and pre-empt the obvious "but the literature says embeddings help" objection.

**m11. Generator co-feature is minimal (one Gaussian `z`, C_CO=0.6) and additive.** Δt is made deliberately dominant and the only interaction is additive (`A_SIG*g + C_CO*z`). Real fraud signal has Δt interacting with amount/velocity. This caps the external validity of "spend budget on signal, not representation" — the study cannot speak to Δt×other-feature interactions, which is precisely where a per-step learned basis in a GRU might earn its keep. Add to limitations.

---

## 4. Prioritized Recommendations

1. **[M1 — Statistical bar] Recompute every decision-driving CI at the seed level, not the seed-0 test-row level.** For each pre-specified pair, collect the 5 per-seed paired PR-AUC differences and report the mean with a paired test (paired t or Wilcoxon) and/or a seed-cluster bootstrap CI. Re-state the headline with that uncertainty. Keep the seed-0 row bootstrap only as a labeled secondary diagnostic. Expect the `learned − log` verdict to weaken to "borderline/worse" and the recommendation to rest more on cost asymmetry than on a clean significant deficit — report that honestly. Code pointer: the fix is to lift the `[0]` indexing in `main()` (lines 237, 245–246, 251) into a per-seed loop.

2. **[M4 — Overclaim / scope] Reframe the Conclusions and Abstract from "the evidence establishes `log` is the right encoding under a capable model" to a cost-asymmetry decision under acknowledged transfer uncertainty.** State plainly that (a) the GRU is untested and is the binding assumption, (b) the recommendation is "keep `log` as the default because the synthetic capable-model evidence shows no gain and the operational costs of PLE/PLR are real, pending the reference-model A/B," and (c) the operational cost claims (inference cost, per-step preprocessing, "silent degradation") are engineering expectations, not measured results. This is mostly a wording change; the underlying decision can stand.

3. **[M3 — Mechanism over-attribution] Support the "overfitting" label or soften it.** Minimum: report the train-loss gap across all 5 seeds (not seed 0). Stronger: add one regularized learned arm (weight decay or early-stopping on a held-out val fold) and show whether the test gap closes. If it closes, the honest claim becomes "the learned basis needs regularization the others don't," which is a *cleaner* argument for preferring parameter-free `log`. If it does not close, you have earned the "overfitting" mechanism claim.

4. **[M2 — Tie vs equivalence] Recast "PLE ties `log`" as an equivalence claim with a stated margin.** Define a minimal practically-relevant PR-AUC gain (e.g., +0.005) and show the seed-level CI for `ple − log` excludes it (TOST-style). "No gain" is a much stronger and more defensible claim than "CI includes zero," and it is what the recommendation actually needs.

5. **[M4 — Architecture proxy] Either run a minimal sequence-model arm or explicitly downgrade the MLP-as-GRU inference.** A cheap, decisive addition: a small GRU (or even a 1-D temporal conv) over short synthetic Δt sequences with the same six encodings, on a regime where consecutive-Δt structure matters. If `log` still ties/wins, the transfer argument becomes evidence rather than conjecture. If you will not run it, state in the Abstract that the capacity argument's extension to recurrence is assumed, not shown.

6. **[m6, m9 — Consistency/presentation] Reconcile the numbers a reader will try to check.** State "target base rate 0.08, realized ≈ 0.072." Label every table as 5-seed-mean vs seed-0-bootstrap, and make the lift table reconstructable from a single consistent quantity. Note the shared bootstrap RNG seed (m7).

7. **[m5 — Multiple comparisons] Add a one-line multiplicity caveat** (11 lifts, no correction) and verify the two medium-sized effects (`learned − log` nonmono −0.022 and mono −0.037) survive a Holm adjustment at the seed level.

8. **[m11, m10 — External validity] Extend limitations to cover the additive-single-co-feature generator and add one paragraph on where PLE/PLR are reported to help** (non-recurrent tabular DL on heterogeneous features), so the negative result is correctly scoped rather than read as a blanket "embeddings don't help."

---

### Bottom line

The experimental *design* is strong — the control suite is the right one and is rare to see done this carefully, and the cross-document discipline is excellent. The work's weakness is the gap between the rigor of the design and the rigor of the *inference*: the decision-driving CIs measure within-seed evaluation noise rather than run-to-run variance, so the headline "significantly worse" and "ties" verdicts are stated with more certainty than 5 seeds support, and the reference-model recommendation is framed as evidence-established dominance when it is really a sound cost-asymmetry default under acknowledged transfer uncertainty. None of these are fatal. Fixing the seed-level statistics (R1) and re-framing the conclusion (R2) would make the recommendation as solid as the apparatus that produced it. The decision to keep `log` is very likely correct; the report should be honest that it is the *prudent* choice, not a *proven* one.

---

## Response (remediation applied — full)

All four MAJOR and all seven MINOR findings were addressed; none deferred.

- **M1 (seed-level CIs) — FIXED.** `dt_encoding_experiment2.py` now computes each decision lift as the
  5 per-seed paired PR-AUC differences → a paired-t 95% CI + p-value (`seed_level_lift`); the seed-0
  row bootstrap is retained only as labeled secondary `seed0_*` fields. Outcome: the proper *paired*
  seed-level test **confirms** the headline (the reviewer's unpaired estimate p≈0.07 was conservative,
  as the reviewer anticipated): `learned−log` = −0.024 [−0.038,−0.010], p=0.009, Holm-significant.
  Updated in `CONCLUSIONS.md`, `TECHNICAL_REPORT.md`, `fig_lift_forest.png`.
- **M2 (equivalence) — FIXED.** "PLE ties log" is now an equivalence claim with a ±0.005 margin:
  `ple_log−log` = +0.000 [−0.002,+0.002] (inside margin → equivalent); `ple_raw−log` = −0.002, p=0.148
  (no detectable gain).
- **M3 (overfitting EARNED) — FIXED.** Added a regularized `learned_reg` arm (weight decay 1e-3) and
  5-seed-mean train loss. The deficit closes under regularization (`learned_reg−log` = −0.004; reg
  effect `learned−learned_reg` = −0.020*, p=0.011) and train loss confirms direction (learned 0.096 <
  log 0.108; learned_reg 0.106 ≈ log). "Overfitting" is now verified, and scoped to the unregularized
  σ=2.0 config.
- **M4 (reframe) — FIXED.** Abstract/Conclusions/Addendum now frame `log` as a prudent cost-asymmetry
  default under acknowledged transfer uncertainty (robust core: "no encoding beats log for a capable
  model"); MLP→GRU transfer is stated as the binding untested assumption (resolved by the reference-model
  A/B); operational claims are labeled expectations, not measurements.
- **MINORS — FIXED.** m5 Holm over the decision family (applied; learned/learned_reg/log_expand vs log
  survive); m6 base rate target 0.08 / realized ~0.072; m7 seed-0 bootstrap disclosed as secondary,
  shared RNG; m8 frequency spectrum labeled illustrative; m9 tables labeled 5-seed-mean (PR-AUC) vs
  seed-level (lifts); m10 prior-art scoping (Gorishniy et al.; PLE/PLR help non-recurrent tabular DL);
  m11 additive-single-co-feature interaction limitation noted.

**Net effect:** the headline survived and sharpened (proper paired stats + Holm), the overfit
mechanism is now verified rather than asserted, and the recommendation voice moved from "proven" to
"prudent default." No MAJOR issue remains open.
