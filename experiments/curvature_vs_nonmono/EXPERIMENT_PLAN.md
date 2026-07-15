# Experiment Plan — Cycle 7 (Gate 1)

**Review:** debate (fast-tracked; case verdict `empirical_test_agreed`). All four ml-critic findings
accepted as empirical tests, not defended away.

## Pre-flight checklist

| # | Source | Finding | Verdict | Item | How addressed | Status |
|---|--------|---------|---------|------|---------------|--------|
| 1 | debate | **F2** (FATAL) | DEFER | S2 rank/CDF signal is isomorphic to PLE's own quantile basis → near-tautological | Add a **non-quantile** monotone log-mismatched shape (S2b) as the true discriminator; keep S2a (rank) only as a basis-aligned *reference* | CLOSED (design) |
| 2 | debate | F1 (MATERIAL) | DEFER | Regularization not scaled to encoding dimensionality → negative controls failed | Per-encoding regularization selection (CV on LogReg `C`; MLP `alpha` + `early_stopping=True`) | CLOSED (design) |
| 3 | debate | F3 (MATERIAL) | DEFER | +0.037 at 3 seeds is below the pipeline's own noise floor | ≥8 seeds; seed-level paired-t 95% CIs + Holm; ±0.005 equivalence | CLOSED (design) |
| 4 | debate | F4 (MINOR) | DEFER | i.i.d. identical-shape features ≠ heterogeneous real features | Add one heterogeneous-shape + correlated arm on S2b; full heterogeneity study deferred to Cycle 8 | CLOSED (documented) |

## Signal shapes (K=6 additive features per shape)

| Shape | per-feature `f_k` | monotone? | log-fit? | quantile-aligned? | role |
|-------|-------------------|-----------|----------|-------------------|------|
| S1 | `std(log1p x_k)` | yes | yes | — | negative control |
| S2a | `std(rank x_k)` | yes | no | **yes** | basis-aligned reference (was v2's discriminator) |
| **S2b** | `std(arcsinh(x_k))` and (robustness) `std(x_k**0.3)` | yes | **no** | **no** | **true discriminator** |
| S3 | `std((std log x_k)²)` | no | no | — | non-monotone control |

S2b is monotone and log-mismatched but is a genuine value-curve PLE must *approximate*, not its own
coordinate — so a PLE win there is curvature capture, not tautology (F2 fix).

## Conditions

4 shapes × 2 heads × 4 encodings × ≥8 seeds, per-encoding regularization selected by CV.
- **Heads:** `linear` (affine-read), `mlp` (free-nonlinearity, `early_stopping=True`).
- **Encodings:** `raw`, `log`, `ple` (16 bins/feature), **`dense`** (learned Linear→ReLU expansion — now included).
- **Trivial baseline:** `raw` under the linear head is the floor; the **oracle** (logreg on the true generative logit) is the ceiling / precondition probe.

## Controls (pre-registered gates)

- **Precondition:** oracle AP ≫ base rate in every shape (PoC: 0.58–0.72 vs 0.086 — expected to hold).
- **Positive control:** on the linear head, `ple`/`dense − raw` large in S2b and S3 → apparatus has power.
- **Negative control:** S1 ties across encodings **and** the entire MLP head ties (`ple ≈ log`). These
  FAILED in the PoC due to F1; the matched-regularization fix must make them pass, or the design is not
  isolating the mechanism (→ void/redesign).

## Empirical tests with pre-specified verdicts

**Decisive test — `ple − log` on the linear head in S2b:**

| Result | Meaning |
|--------|---------|
| CI excludes 0 **positive** (Holm), clear of ±0.005, **and** S1/MLP negative controls clean | **H_curv general** — curvature-via-combination is a basis-agnostic lever; non-monotonicity not necessary |
| S2b CI straddles 0 / within ±0.005 **but** S2a positive | effect was **basis-alignment only** — narrow H_curv to "PLE recovers quantile-shaped curvature" (kills the Cycle-8 "any curved feature" rationale) |
| S2b CI excludes 0 **negative** | H_curv **refuted** for non-quantile curvature |
| Negative controls still fail after F1 fix | design not isolating the mechanism → **void, redesign** |

**Secondary:** `dense − log` (linear head, S2b) > 0 confirms an affine-read model benefits from *any*
added per-feature nonlinearity; `ple − dense` sign tests whether the fixed basis beats the learned one
(as in cycle 6). Heterogeneous-arm test: sign/magnitude of the S2b linear `ple − log` gap preserved
under mixed per-feature shapes + correlation.

## Stats

PR-AUC primary. Seed-level paired-t 95% CIs + Holm step-down over the decision family; ±0.005
equivalence margin. Within-seed bootstrap not used for verdicts (measures eval noise, not run-to-run
variance) — consistent with the line's convention.

## Before Step 6 runs
1. All pre-flight items CLOSED (above). ✔
2. **User approval of this plan.** ⟵ Gate 1
3. `/intent-watch` clean pass against `HYPOTHESIS.md` (pre-registration lock).
