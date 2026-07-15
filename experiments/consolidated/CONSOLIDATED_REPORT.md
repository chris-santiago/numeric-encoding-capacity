# Consolidated encoding-capacity experiment — report (cycles 1–8)

**One promoted Metaflow flow, positive-controlled, that puts every arm on the same footing.** Architecture
(static / gru / mlp) × risk-shape condition (log_linear, monotone_curved, smooth_nonmono, sharp_mode,
sharp_off) × multiplicity (K ∈ {1, 6}) × 8 seeds × 12 arms, every arm temperature-calibrated. Synthetic by
design — it *isolates* the mechanism; real-data applicability is the separate documented outcome
(`../gru_curvature_realdata/REAL_DATA_AB.md`, where the precondition failed on the demo data).

## Reproduce

```
cd flow
uv run flow.py --config-value cfg "hydra_overrides: [experiment=consolidated]" run --max-workers 8
```
Gates: `flow-lint` (pass) · `pipeline-reviewer` fidelity (5/5 after fixes) · code-correctness review (all
fixed) · determinism `nondeterministic` (see below). Read results from the Metaflow run artifacts
(`lift_results`, `aggregate_results`, `analyses`) or the `report` step.

## Instrument verdict: TRUSTWORTHY ✓

The positive control fires strongly — `static_ple` on the sharp band, deficit-corrected **+0.46
[+0.42, +0.50]** (Holm p ≈ 0). The estimand demonstrably detects the strongest known lever, so every GRU
reading below is interpretable rather than blind. The gate is wired to **halt** the run otherwise (it is
not print-only). (Curvature and the multivariate check are *reported*, not gated — see anomalies.)

## The estimand

Deficit-corrected, seed-paired: `Δ = (arm − log)_condition − (arm − log)_log_linear`, netting the
encoder's structural cost measured on the log-adequate condition. PR-AUC on raw scores (rank);
log-loss/Brier on **calibrated** probabilities (so magnitude metrics reflect representation, not
calibration); 8-seed paired-t 95% CI + **Holm** across the reported family.

## Headline findings (Holm-significant, K=6, deficit-corrected)

| claim (cycle) | evidence |
|---|---|
| **Sharp non-monotone is the dominant PLE lever** | `static_ple` **+0.46**, `gru_ple` **+0.43** — the largest effects in the grid |
| **Curvature IS a lever in the GRU** (corrected C8) | `gru_ple` +0.10, `gru_projection` +0.12 |
| **Smooth non-monotone → learned per-step nonlinearity, not PLE** | `gru_projection`/`gru_dense` +0.23; `gru_ple` not significant (gates approximate the smooth hump) |
| **Sharp lever robust to band location** | `gru_ple` fires at both `sharp_mode` (+0.43) and `sharp_off` (+0.14) |

## Three anomalies — flagged, not hidden

1. **Static curvature came out weak/negative** (`static_ple` −0.02, CI touches 0), where Cycle 7 found a
   weak *+*0.015. Both are ≈0 → curvature is a *marginal* lever in the static vehicle; the sign flip vs
   Cycle 7 traces to a different static model here (recency-pooled logistic vs Cycle 7's additive model).
   Curvature fires in the *GRU* (+0.10) but not this static vehicle — an inversion worth a follow-up.
2. **The `raw` arm shows large *positive* deficit-corrected lifts** (+0.28 to +0.47). This is an estimand
   artifact for the floor arm, not raw being good: raw's deficit is measured on `log_linear` (where `log`
   crushes badly-conditioned raw), so the difference-of-differences flatters raw on every other condition.
   The floor arm doesn't fit the difference-of-differences framing; read `raw` from raw PR-AUC, not Δ.
3. **`mlp_ple` fires large** (+0.57 on sharp). The free-per-step-nonlinearity model *should* make encoding
   redundant, but its MLP-on-log-scalar did not learn the sharp band at this scale, so PLE still helped it
   — a capacity/optimization nuance, not clean redundancy. Larger `mlp` capacity is the follow-up.

## Determinism: `nondeterministic` (honest declaration)

Torch CPU GRU kernels are not bit-reproducible run-to-run even with single-threaded BLAS
(`OMP/MKL/OPENBLAS/VECLIB = 1`); residual per-cell variance is ~0.005–0.01, an order of magnitude below the
seed-level CIs. The **8-seed CIs are the reproducibility guarantee**: signs, CI-exclusions, and the gate
verdict reproduce across runs; exact aggregates vary at the 3rd decimal. The determinism gate self-skips
for this declared contract (not a silent gap — a recorded, reviewable declaration).

## What this is / isn't

**Is:** a positive-controlled, calibrated, Holm-corrected synthetic experiment isolating the mechanism and
its signs, reproducible-by-conclusion via one `flow.py run`. **Isn't:** a real-fraud magnitude (synthetic;
directional) — the deployment question needs data where the precondition passes. See `CLAIMS.md` for the
full claim→arm map and `HYPOTHESIS.md` for the governing law and fairness invariants.
