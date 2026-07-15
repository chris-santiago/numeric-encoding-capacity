# Conclusions — Cycle 8: does monotone value-curvature give PLE a lever in a GRU?

**Cycle 8 of the numeric-encoding-capacity line. Review: debate (`critique_wins`; remediation executed).
Mode: conclusions_only.** Vehicle: constructed-sequential synthetic (recency-weighted per-step
aggregation), affine-input GRU trained under the Cycle-6 regime (minibatch + val early-stop + best-state
restore), 5 seeds, seed-level paired-t 95% CIs. The real-data Step 6 was **not run** — this cycle's
positive control shows there is nothing for it to find.

## Headline

**Monotone value-curvature is NOT a PLE lever in a GRU.** Cycle 7 proved that in a *static affine-read*
model (linear head), a log-mismatched monotone curve is a real, general lever for PLE over `log`. That
result **does not transfer to a sequence model**: a GRU's gate nonlinearities already supply an effective
per-step nonlinearity, so it absorbs monotone curvature straight from a `log` scalar. Per-step PLE on the
curved feature only imports its structural deficit — and *worse* as the feature carries more signal. The
lever that matters for a GRU is **non-monotonicity** (Cycle 6), not monotone curvature.

## The decisive evidence (why this is "no lever," not "blind instrument")

The hardened PoC (F1–F6 remediation from the debate) triangulates three probes that together exclude the
"the test just can't see it" explanation the critic (rightly) demanded be ruled out:

1. **Power curve (F1) is flat-negative.** Deficit-corrected benefit `AP(ple_count) − AP(ple_ref)` across a
   monotone departure-from-log knob `k = (exp(k·s)−1)/k`:

   | k | curvature | benefit | 95% CI | crosses 0? |
   |---|---|---|---|---|
   | 0.01 | ~linear | −0.004 | [−0.018, +0.009] | no |
   | 0.70 | mild | −0.041 | [−0.075, −0.008] | no (negative) |
   | 1.40 | moderate | −0.042 | [−0.056, −0.027] | no (negative) |
   | 2.10 | exaggerated | −0.036 | [−0.055, −0.017] | no (negative) |

   PLE never wins; it *loses more* as curvature grows, because binning the now-more-important curved
   feature costs more.

2. **The `dense` arm (F5) is the disambiguator.** A learned per-step Linear→ReLU — a **superset** of PLE's
   per-step expressivity (a ReLU MLP can represent PLE's clipped-ramp basis and more, and adapts to the
   signal) — yields `dense − log = +0.004 [−0.006, +0.015]`, i.e. nothing. If *any* per-step transform of
   the curved feature could recover benefit, `dense` would. It doesn't → **no per-step-value lever exists**;
   the flat power curve is a true null, not a metric artifact.

3. **The oracle ceiling (F4/F5) locates the residual.** Oracle (rank by true log-odds) AP = 0.628;
   `log`-GRU = 0.596, a real but small gap `+0.032 [+0.027, +0.038]`. Crucially **nothing closes it** —
   `log`, `raw`, `ple_ref`, `dense` all cluster 0.596–0.600. So the headroom lives in the recurrent
   aggregation / capacity, **not** in per-step value representation, and no encoding can address it.

**Magnitude-sensitive metrics (F2) agree with PR-AUC** — `ple_count` is worst on log-loss (0.3058 vs
0.2954) and Brier (0.0913 vs 0.0876) too, so PR-AUC's rank-invariance is not hiding a win.

## Debate scorecard (`critique_wins` → all remediations executed)

| Finding | Verdict | Empirical resolution in the hardened PoC |
|---|---|---|
| **F1 (FATAL)** — no positive control; a Step-6 null would be uninterpretable | conceded → **resolved** | Power sweep built. Combined with `dense` + oracle, the null is now interpretable as *no lever*, not *no power*. |
| **F3 (FATAL)** — tabular baseline too weak (recency-agnostic vs recency-weighted DGP) | conceded → **resolved** | GBM + EWMA baseline (0.540). Gate still passes CI-clear: margin +0.056 [+0.042,+0.070], shuffle drop +0.236 [+0.187,+0.286]. |
| **F4 (FATAL)** — `amount` deficit reference contaminated (weakly-curved + marginal-mismatched) | conceded → **resolved** | Replaced by a marginal-matched, exactly-log-adequate reference. `ple_ref ≈ log` (clean ~0 deficit); correction is uncontaminated. |
| **F5 (MATERIAL)** — `dense` arm missing (drift from HYPOTHESIS) | conceded → **resolved** | `dense` arm added; it became the single most decisive probe. |
| **F2 (FATAL→5)** — PR-AUC rank-invariance could mask representation | partial rebut → **resolved** | log-loss + Brier reported; they corroborate PR-AUC (no hidden win). |
| **F6 (MATERIAL)** — additivity untested for correlated/co-encoded features | deferred | Scope tightened to single-feature-at-a-time; multi-feature synergy explicitly out of scope. |

## The unified theory (why the whole line now coheres)

The governing variable across every cycle is a single question: **does the architecture already own a
free per-step nonlinearity?**

| | monotone, log-linear | monotone, curved | non-monotone |
|---|---|---|---|
| **static affine-read** (linear head) | log adequate (C7 S1) | **PLE lever** (Cycle 7) | **PLE lever** (Cycle 7) |
| **recurrent** (GRU) | log adequate (Cycle 6 neg. control) | **no lever** (Cycle 8) | **PLE lever** (Cycle 6, L=300) |

A static linear head has no per-step nonlinearity, so both curvature *and* non-monotonicity are levers for
a fixed basis. A GRU's gates *are* a per-step nonlinearity, so monotone shapes (linear or curved) are
handled for free; only **non-monotone / band-selective** structure — which the gates synthesize
inefficiently while also serving memory — remains a lever. Cycle 8 is the previously-empty
"recurrent × monotone-curved" cell, and filling it completes the table.

## Verdicts against the pre-registered decision rule

- **H_curv-in-GRU: REFUTED.** The decisive test (`ple − log` deficit-corrected, CI-excludes-zero positive)
  fails at every curvature level — the CI is at or below zero throughout. The precondition **passed**
  (CI-clear vs a strong baseline, CI-clear order dependence), so this is a genuine refutation, not a moot
  one — the distinction Cycle 3 lacked.
- **Mechanism identified:** the GRU's gate nonlinearities absorb monotone value-curvature; the `dense`
  null and the un-closable oracle gap jointly show no per-step-value lever exists.
- **Step 6 (real-data monotone-curvature targeting): NOT justified** and not run. A validated positive
  control shows the lever is absent under ideal (exaggerated) curvature; on real data PLE would import its
  deficit with no offsetting gain.

## What is (and isn't) open

- **Closed by Cycle 8:** the Cycle 7 → GRU transfer question, and the C1 puzzle (Cycle 2's monotone-curved
  +0.144). C1's curvature effect is a *static-model* phenomenon; in a GRU it does not survive.
- **Not a Cycle 9:** non-monotonicity in a GRU at scale is already settled by Cycle 6 (L=300, controls
  fire, PLE beats both scalar and Dense). No further cycle is warranted.
- **Standing follow-up (Cycle 6's stated "next, not run"):** a real-data A/B of the **non-monotone** lever
  — per-step PLE on genuinely band-selective features through a precondition-gated GRU. Cycle 8's
  contribution to it is the **validated precondition gate** (GBM+EWMA baseline + CI-clear order test) that
  Cycle 3 lacked. Target non-monotone structure, not monotone curvature.
