# Conclusions — Cycle 7: Non-monotonicity vs value-curvature

**Cycle 7 of the numeric-encoding-capacity line. Review: debate (fast-tracked, `empirical_test_agreed`).
Mode: conclusions_only.** Vehicle: static two-head grid (linear = affine-read, MLP = free-nonlinearity),
K=6 additive synthetic features, PR-AUC, 8-seed paired-t 95% CIs + Holm.

## Headline

**In an affine-read model, log-mismatched value-curvature is a real and general lever for PLE over
`log` — not merely a basis-alignment artifact, and not only non-monotonicity.** Once PLE's *structural
deficit* is accounted for, every genuinely log-mismatched monotone curve tested (convex, saturating-S,
cubic) shows a CI-clear PLE benefit; a saturating S-curve that is **not** PLE's quantile coordinate
benefits too, ruling out the tautology concern. Non-monotonicity remains the strongest lever, but
monotone curvature is in the same order of magnitude (cubic ≈ half the non-monotone effect).

## The two mechanism findings (the reason this cycle exists)

1. **Value-curvature is a *multivariate* phenomenon** (Step-1/2 PoC). For a single feature under a
   rank metric (PR-AUC), any monotone encoding gives identical rankings, so curvature is invisible and
   only non-monotonicity opens a gap (`ple−log = −0.000` exact tie, single-feature S2). Curvature
   matters only through how a monotone-but-curved feature **combines additively** with others — which is
   why cycle-2's `C1` appeared in a multivariate fraud model, not in isolation. The design was
   reformulated to K=6 additive features (`HYPOTHESIS.md` § Cycle 7 v2).

2. **PLE carries a structural deficit that must be corrected before the benefit is visible.** On a
   signal `log` fits optimally, a flexible quantile basis can only be ≥ slightly worse (`ple−log ≈
   −0.042` on the log-fit control S1, *robust to bin count and regularization* — it is representational,
   not overfitting). Measuring the raw `ple−log` therefore *understates* curvature benefit by ~0.04. The
   correct estimand is the seed-level difference-of-differences `Δ = (ple−log)_shape − (ple−log)_S1`.

## Debate scorecard

| Finding | Verdict | Empirical resolution |
|---------|---------|----------------------|
| **F2 (FATAL)** — S2 rank signal is isomorphic to PLE's quantile basis → win is tautological | **Correct → addressed** | Added genuine non-quantile curves (convex/sat/cubic). The non-quantile `S2b_sat` benefits (+0.092) and quantile `S2a` is the *smallest* (+0.070) → basis-alignment ruled out, but curvature benefit survives. |
| **F1 (MATERIAL)** — regularization not scaled to encoding dimensionality | **Partially right, deeper cause found** | CV regularization + fewer bins did **not** clean the S1 control. The deficit is structural, not overfitting → reframed S1 as the deficit baseline. |
| **F3 (MATERIAL)** — 3-seed +0.037 below noise floor | **Correct → addressed** | 8-seed paired-t + Holm; all reported Δ are CI-clear at Holm p ≈ 0. |
| **F4 (MINOR)** — i.i.d. identical-shape features | **Addressed** | Heterogeneous + correlated arm ran; effect preserved. Full heterogeneity → Cycle 8. |

## Results (deficit-corrected Δ, linear/affine-read head, 8 seeds, Holm)

| Shape | Δ = curvature benefit above deficit | 95% CI | verdict |
|-------|-------------------------------------|--------|---------|
| S2a quantile (basis-aligned reference) | +0.070 | [+0.045, +0.094] | real (smallest — *not* basis-driven) |
| S2b convex `exp(0.7s)` | +0.142 | [+0.110, +0.173] | real |
| S2b saturating-S `sigmoid(2s)` (non-quantile) | +0.092 | [+0.070, +0.114] | real — kills the tautology concern |
| S2b cubic `s³` | +0.265 | [+0.243, +0.287] | real (largest monotone) |
| S3 non-monotone `s²` | +0.500 | [+0.469, +0.530] | real (dominant lever) |
| S1 log-fit | 0 (baseline) | — | PLE structural deficit = −0.042 |

Precondition passed (oracle AP 0.57–0.74 ≫ 0.085 base). Figures: `finding_curvature_above_deficit.png`
(headline), `finding_ple_vs_log_linear.png` (raw per-shape).

## Surprise (marked explicitly)

Neither critic nor defender anticipated the **structural PLE deficit**. It reversed the verdict twice:
raw reads (experiment 2–3) looked like a *refutation* of H_curv because the ~−0.04 deficit swamped the
real benefit, compounded by a mis-specified `arcsinh` signal (arcsinh(x) ≈ log(x)+const, so not actually
log-mismatched). Only the difference-of-differences correction against a log-optimal baseline revealed
the true, robust curvature benefit. **Lesson: comparing a flexible basis to a fixed transform requires a
same-family deficit baseline; the raw delta conflates "the basis is worse where the transform is
adequate" with "the basis is better where the shape departs."**

## Verdicts against the pre-registered decision rule

- **H_curv (general): SUPPORTED.** `ple − log` CI-clear positive above deficit for all genuine
  log-mismatched monotone curves; non-monotonicity not necessary; basis-alignment not the cause.
- Benefit is **curve-dependent** (cubic +0.265 > convex +0.142 > saturating +0.092): the further a
  monotone shape departs from `log`, the larger the PLE gain.
- Non-monotonicity (+0.500) remains the strongest single lever, but monotone curvature is comparable in
  scale, not negligible.

## Implication for `C1` and Cycle 8

The cycle-2 `C1` lift (+0.144, a monotone-by-rank, curved-in-value count feature) **is now plausibly
explained by value-curvature** — a genuinely log-mismatched monotone feature combining with others.
**The Cycle-8 rationale is revived:** targeting PLE at real curved-in-value count/recency features is
justified — with two caveats this cycle adds: (1) PLE pays a ~−0.04 structural cost on features `log`
already fits, so it is a net win only where the log-mismatch is real; (2) the benefit scales with
departure-from-log, so target the most sharply-curved features first.
