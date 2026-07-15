# Conclusions — Cycle 8: does value-curvature give PLE a lever in a GRU?

**Cycle 8 of the numeric-encoding-capacity line. Review: debate (`critique_wins`) + three technical reviews
+ a positive-control audit that RETRACTED and then CORRECTED the original verdict. Mode: conclusions_only.**

> **Correction notice.** An earlier version of this file concluded the *opposite* of what is below —
> "monotone curvature is NOT a lever in a GRU (REFUTED)." That conclusion was **wrong** and has been
> retracted. It rested on a single-curved-feature design (`curvature_seq_poc2.py`) that violated a binding
> precondition from Cycle 7 (curvature is a *multivariate* phenomenon; a single curved feature is invisible
> under a rank metric), so it measured only PLE's structural deficit and never a curvature signal. A K=6
> multivariate reproduction with firing positive controls (`multivariate_control.py`) reverses it. The
> retracted script and its figure are kept for the record; this file is the authoritative verdict.

## Headline

**Monotone value-curvature IS a lever for PLE in an affine-input GRU** — Cycle 7's static-model mechanism
*does* transfer to the sequence model. But the benefit is **masked in raw terms by a large PLE
dimensionality deficit in the recurrence**: feeding many correlated bins per step costs far more in a GRU
(~−0.13 for 6 features × 12 bins) than in a static head (~−0.03), so raw `ple − log` reads ≈ 0 even though
the deficit-corrected lever is `+0.143` (CI-clear). The practical rule is therefore neither "PLE helps"
nor "PLE doesn't" but: **the curvature lever is real, and net deployment value depends on keeping PLE's
dimensionality cost below it — target few features, few bins.**

## The decisive evidence (K=6 multivariate, static vs GRU, 5 seeds)

Deficit-corrected benefit `= (ple−log)_condition − (ple−log)_log-adequate`, nets PLE's structural cost.

| arch | condition | raw `ple−log` | deficit-corrected | CI excludes 0? |
|---|---|---|---|---|
| **static** (affine-read, no per-step nonlinearity) | curved (cubic) | −0.019 | **+0.015** [+0.003, +0.027] | yes — **positive control fires** |
| static | non-monotone (quadratic) | +0.339 | **+0.373** [+0.346, +0.400] | yes — sanity control fires |
| **GRU** (gates = per-step nonlinearity) | curved (cubic) | +0.008 (ns) | **+0.143** [+0.068, +0.218] | **yes — curvature is a GRU lever** |
| GRU | non-monotone (quadratic) | −0.101 | +0.035 [−0.021, +0.090] | no — anomalous, see caveat |

The static positive controls firing is what makes the GRU numbers interpretable: the estimand demonstrably
detects a curvature lever where one is *known* to exist (Cycle 7 reproduced), so the GRU result is a real
measurement, not an artifact of a blind instrument.

## Why the original refutation was wrong (the two compounding flaws)

1. **Single-feature invisibility (the fatal one).** Cycle 7 established value-curvature is *multivariate*:
   for a single curved feature under PR-AUC, every monotone encoding gives identical rankings, so curvature
   is invisible (`ple−log = −0.000` exact tie, Cycle 7 single-feature S2). `poc2` used one curved feature
   → no curvature signal was expressible, regardless of architecture. The single-feature positive control
   (`positive_control.py`) confirmed this: it did **not** fire even in the static head (+0.004, ns), because
   it too was single-feature. K=6 makes the lever visible (static +0.015, CI-clear).
2. **Un-netted dimensionality deficit.** In the GRU, PLE pays ~−0.135 for K=6×12 bins. `poc2` read the raw
   `ple_count − log ≈ 0`/negative and called it "no lever," when it was "real lever − large deficit ≈ 0."

So `poc2`'s "gates absorb monotone curvature" was a mirage produced by an invisible signal plus a masked deficit.

## Verdicts against the pre-registered decision rule

- **H_curv-in-GRU: SUPPORTED (corrected).** Deficit-corrected `ple − log` is CI-clear positive (+0.143) on
  multivariate monotone-curved features in the affine GRU. Curvature is a genuine per-step lever the gates
  do not reach on their own.
- **Deployment reality:** raw/blanket per-step PLE does **not** net a win in the GRU at this feature count —
  its dimensionality deficit (~−0.13) masks the lever. Net value requires selective targeting and few bins.
- **Instrument validated:** static positive controls fire for both curvature (+0.015) and non-monotonicity
  (+0.373); the estimand has power once the design is multivariate.

## The corrected unified picture

The governing question is still "does the architecture own a free per-step nonlinearity?", but the answer
is more graded than the retracted 2×2 claimed:

| | static affine-read (linear head) | recurrent (GRU) |
|---|---|---|
| **monotone, log-linear** | log adequate | log adequate |
| **monotone, curved** | PLE lever, small deficit (Cycle 7) | **PLE lever (+0.143), but LARGE deficit masks raw benefit (Cycle 8, corrected)** |
| **non-monotone (sharp/band)** | PLE lever (Cycle 7) | PLE lever at L=300 (Cycle 6) |
| **non-monotone (smooth)** | PLE lever (this cycle, +0.373) | **unresolved** — ns here (+0.035); see caveat |

The GRU does *not* absorb monotone curvature (the retracted claim). What it does is impose a much larger
encoding **cost**, so the same lever that clears easily in a static head can be swamped in a recurrence.

## Caveats and what remains open

- **GRU smooth-non-monotone is anomalous and unresolved.** The GRU `nonmono` cell is ns (+0.035) and
  negative raw (−0.101), inconsistent with Cycle 6's +0.19. The likely cause: this used *smooth* quadratics
  (gate-approximable, per `proj_vs_ple`) rather than Cycle 6's *sharp conjunctive band*, compounded by the
  large deficit and 5-seed noise. Do **not** read it as "non-monotonicity isn't a GRU lever" — Cycle 6's
  sharp-band result at L=300 stands. Needs a sharp-band multivariate rerun to resolve.
- **PoC scale.** L=32, hidden 32, 5 seeds, synthetic. The large GRU deficit is real but its magnitude at
  L=300 / production capacity is untested. Direction (curvature is a lever; deficit is large and scales
  with bins×features) is the claim; magnitudes are illustrative.
- **Deficit-correction leans hard on the log-adequate baseline** in the GRU (it supplies −0.135 of the
  +0.143). The estimand is the Cycle-7 difference-of-differences, but at n=5 with a large correction term
  the CI is wide; more seeds would tighten it.

## Resolved follow-ups (three experiments, positive-controlled)

All three carried a static-head positive control (the discipline that caught the original error).

- **A — deficit-vs-(K, bins) (`deficit_curve.py`).** The deficit-corrected curvature lever is robust across
  configs (~+0.09–0.12), but **raw** `ple−log` (deployment) clears CI>0 at only **one** config — K=4,
  bins=8 (+0.047 [+0.004, +0.089]). Too many bins (16) or features (K=8) → the dimensionality deficit
  swamps the lever (raw → negative); too few → the lever isn't resolved. **PLE's net win is a narrow,
  fragile ridge (~+0.05), not a broad regime.** (This script's *raw* static-control line mislabels itself
  "blind"; the proper deficit-corrected static control fires in B/C on the same signal.)
- **B — sharp vs smooth non-monotone (`sharp_vs_smooth.py`).** RESOLVED: the GRU **absorbs smooth**
  non-monotonicity (+0.035 ns) but a **sharp/localized band fires +0.446** (both static controls fire).
  The earlier smooth-quadratic null was a smooth-vs-sharp effect; **Cycle 6's sharp-band lever stands.**
- **C — fixed PLE vs learned per-feature embed (`fixed_vs_learned.py`, multivariate, shared coordinate,
  8 seeds, Holm).** RESOLVED: the **learned embed beats fixed PLE by +0.094 (Holm-sig)**. Deficit-corrected
  the two unlock nearly identical curvature signal (+0.137 vs +0.148); the raw gap is PLE paying a **larger
  dimensionality deficit** at equal width. Reverses the retracted single-feature "PLE wins on sharp" claim.

## The organizing principle (what all of Cycles 6–8 reduce to)

An affine GRU's gates are **smooth function approximators**, so the only per-step structure the affine read
cannot build on its own is **localized/sharp** structure. That single fact orders every result:

**sharp non-monotone (+0.446) ≫ monotone curvature (+0.143) ≫ smooth non-monotone (+0.035, absorbed) ≈
log-linear (0).**

Encoding buys *localized resolution*; its binding cost is *dimensionality*. Hence: PLE net-wins only on a
narrow K/bins ridge, and a **learned per-feature embed dominates it** by delivering the same resolution at
lower effective width.

## Deployment recommendation (updated, final)

For an affine GRU: **target few features** with **sharp or sharply-curved** per-step risk, at **low
width**, and **choose the encoder by target sharpness** — a **learned per-feature embed for smooth/curved**
targets (lower deficit; wins +0.094), **fixed PLE for sharp/localized** targets (SGD can't place sharp
knots; PLE wins +0.11–0.18, robust to band location). Sharp non-monotone structure is the highest-value
target (~+0.45) and PLE is its right encoder; monotone curvature is a moderate lever best served by the
learned embed; smooth non-monotone and log-linear features need no encoding. Standing external item: the
real-data A/B (Cycle 6), now with the validated precondition gate and this targeting rule.
