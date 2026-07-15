# Conclusions — Δt encoding for fraud (construct-valid synthetic test)

**Cycle 5 of the fraud-encoding line. Review: debate (`empirical_test_agreed`) + Opus peer review R1
(full remediation applied). Mode: conclusions_only.** Synthetic data with Δt informative *by
construction* (the cycle-4 fix), a positive control, negative control, precondition gate, and
convergence verification. 7 encodings × 2 capacities × 2 regimes × 5 seeds.

**Statistics (post-R1):** decision verdicts use **seed-level paired CIs** — for each pair, the 5
per-seed paired PR-AUC differences → a paired-t 95% CI + p-value (the between-run uncertainty relevant
to deployment). The earlier seed-0 within-seed bootstrap is retained only as a secondary diagnostic.
Multiplicity over the non-monotone-MLP decision family is controlled with Holm. Equivalence to `log` is
declared only when a CI lies entirely within ±0.005 PR-AUC.

## Headline

**For inter-transaction time (Δt) under a capable model, `log` is the prudent encoding: no encoding
beats it. PLE is equivalent to `log` (no gain); an unregularized learned periodic basis is
significantly worse (overfitting — verified, and recovered by regularization).** The result is
powered: the positive control fires decisively, so the null is real evidence.

- **Positive control FIRES** (linear, nonmono; seed-level): `learned − raw` = **+0.196 [+0.174,
  +0.217]**, `learned − log` = **+0.256 [+0.216, +0.295]**, `ple_log − raw` = +0.149*, `ple_raw − raw`
  = +0.104*. A weak model genuinely cannot fit the U-shape from raw/log; a basis rescues it.
- **The real question (MLP, nonmono):** `ple_raw − log` = **−0.002 [−0.006, +0.001], p=0.148** (no
  gain); `ple_log − log` = **+0.000 [−0.002, +0.002]** (within ±0.005 → **equivalent to log**);
  `learned − log` = **−0.024 [−0.038, −0.010], p=0.009 (Holm-significant)** — the learned periodic
  basis is significantly *worse*.
- **Overfitting is verified, not assumed (M3):** the regularized learned arm closes the gap —
  `learned_reg − log` = **−0.004 [−0.006, −0.002]** (≈ log) and the regularization effect
  `learned − learned_reg` = **−0.020 [−0.033, −0.008], p=0.011***. 5-seed-mean train loss: learned
  **0.096** < log 0.108 (fit train better) → overfit; learned_reg 0.106 ≈ log (overfit suppressed).
- **Negative control clean** (MLP, mono): `log − raw` = +0.033* (log beats raw), `ple_raw − log` =
  −0.003 (ties), `learned − log` = −0.041* (worse). **z-only floor = 0.10** (≈ base) — Δt is the
  dominant signal.

## PR-AUC by encoding (5-seed mean; base target 0.08, realized ~0.072; nonmono regime)

| Encoding | linear (weak) | MLP (strong) |
|---|---|---|
| raw | 0.601 | 0.744 |
| **log** | 0.541 | **0.804** |
| ple_raw (standard PLE) | 0.705 | 0.802 |
| ple_log (PLE on log) | 0.750 | 0.805 |
| learned (periodic PLR, σ=2.0) | 0.797 | 0.780 |
| learned_reg (periodic + weight-decay) | 0.802 | 0.801 |
| log_expand (matched, non-periodic) | 0.805 | 0.804 |

*(PR-AUC values are 5-seed means; lifts and their CIs below are seed-level paired — the two are
computed differently, so a lift ≠ the difference of two table cells exactly.)* See
`fig_summary_prauc.png`, `fig_lift_forest.png`, `fig_dt_shapes.png`.

## Seed-level lift table (paired-t 95% CI, n=5; * = CI excludes 0; H = Holm-sig; ≡ = equiv to log)

| Pair | Model | Regime | Lift | 95% CI | p | flags |
|---|---|---|---|---|---|---|
| ple_raw − log | MLP | nonmono | −0.002 | [−0.006, +0.001] | 0.148 | — |
| ple_log − log | MLP | nonmono | +0.000 | [−0.002, +0.002] | 0.579 | ≡ |
| learned − log | MLP | nonmono | −0.024 | [−0.038, −0.010] | 0.009 | *H |
| learned_reg − log | MLP | nonmono | −0.004 | [−0.006, −0.002] | 0.005 | *H |
| learned − log_expand | MLP | nonmono | −0.024 | [−0.037, −0.011] | 0.007 | *H |
| learned − learned_reg | MLP | nonmono | −0.020 | [−0.033, −0.008] | 0.011 | * |
| learned − raw | linear | nonmono | +0.196 | [+0.174, +0.217] | 0.000 | * (pos ctrl) |
| log − raw | MLP | mono | +0.033 | [+0.010, +0.056] | 0.016 | * (neg ctrl) |
| learned − log | MLP | mono | −0.041 | [−0.064, −0.017] | 0.008 | * (neg ctrl) |

## Interpretation

Δt carries strong non-monotone fraud signal (positive control + z-floor confirm). The encoding
question is therefore real — and the answer is the capacity argument, sharpened by the regularization
result. A weak model needs a rich basis to bend the U-shape (PLE/learned win big). A capable model
(MLP) learns the transform from plain `log`, so PLE is **equivalent** (no gain) and an unregularized
learned periodic basis is **worse**: with σ=2.0 it has the freedom to fit training detail (lowest train
loss 0.096) that does not generalize (lowest test PR-AUC). Adding weight decay removes ~85% of that
deficit (`learned_reg − log` = −0.004) — confirming the mechanism is overfitting, not a fundamental
property of the basis. Either way, **no learnable basis (periodic, regularized-periodic, or the matched
non-periodic `log_expand`) beats `log` for a capable model.**

## Honest caveats (peer-review-calibrated)

- **The recommendation is a prudent cost-asymmetry default, not proven dominance for the reference
  model (M4).** It rests on "no encoding *beats* log for a capable model," which is robust across the
  seed-level CIs, the equivalence test, the regularized arm, and the matched expansion. "Learned is
  *worse*" is specific to the unregularized σ=2.0 config (regularization nearly closes it).
- **MLP is the strong-model proxy; the reference GRU over ~300-step Δt sequences was not tested
  (M4).** A recurrent model could exploit a basis via cross-step interactions an MLP cannot. The
  capacity argument predicts the same outcome, but this is an *assumption*; the definitive test is the
  one-line per-step encoding swap in the reference GRU (where SHAP confirms Δt is important).
- **Operational claims (inference cost, "silent degradation") are expectations, not measurements
  (M4)** — nothing about latency/serving was benchmarked here.
- **Decision CIs are seed-level (n=5), so they are wide;** the positive control and the learned-log
  verdict survive Holm, but the small effects rest on 5 seeds — more seeds would tighten them (M1/m5).
- **Synthetic, single symmetric U-shape (m11/limitations);** a harder/asymmetric shape only makes log
  look better (less structure for a basis), so the no-gain conclusion is conservative. The additive
  single co-feature cannot speak to Δt×other-feature interactions — exactly where a learned basis in a
  GRU might earn its keep.
- **Minor disclosures:** base rate is target 0.08 / realized ~0.072 (m6); seed-0 bootstrap shares one
  RNG stream and is a secondary diagnostic only (m7); the learned-frequency spectrum (seed-0) is
  illustrative (m8); PR-AUC table cells are 5-seed means while lifts are seed-level paired (m9). PLE/PLR
  are known to help *non-recurrent tabular* deep nets (Gorishniy et al.); this negative result scopes
  to a single informative numeric feature under a capable model (m10).

## Hypothesis closure

- **Positive-control claim — CONFIRMED:** richer encodings beat raw/log for a weak model on
  non-monotone Δt (the test has power; seed-level p=0.000).
- **Real-question claim — CONFIRMED:** PLE does not beat log for a capable model (equivalent); the
  learned periodic basis is worse (overfit, recovered by regularization). No learnable basis beats log.
- **Negative-control claim — CONFIRMED:** log suffices for monotone Δt.
- **Through-line:** where a numeric feature is informative *and* the model is capable, the standard
  transform (`log`) is the prudent choice; richer learned encodings are redundant (PLE) or harmful
  (unregularized periodic). PLE/periodic pay off for *weak* models — the capacity argument, now shown
  on an informative feature with full power and peer-review-calibrated statistics.

## New hypothesis (next, not run)

The remaining open question is GRU transfer: does the MLP result hold for a recurrent model on Δt
sequences? Test via the one-line encoding swap in the reference GRU (where SHAP confirms Δt matters) —
the construct-valid setting cycle 4 lacked.
