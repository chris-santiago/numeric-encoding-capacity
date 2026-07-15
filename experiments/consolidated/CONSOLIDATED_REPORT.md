# Consolidated encoding-capacity experiment — report (cycles 1–8)

**One promoted Metaflow flow, dual positive-controlled, that puts every arm on the same footing.** Architecture (static / gru / mlp) $\times$ risk-shape condition (log_linear, monotone_curved, smooth_nonmono, sharp_mode, sharp_off) $\times$ multiplicity ($K \in \{1, 6\}$) $\times$ 8 seeds $\times$ 12 arms, every arm temperature-calibrated. Synthetic by design — it *isolates* the mechanism; real-data applicability is the separate documented outcome (`../gru_curvature_realdata/REAL_DATA_AB.md`, where the precondition failed on the demo data).

> **This revision (post peer-review).** An adversarial review re-ran the flow's own artifacts and showed the deficit-corrected estimand equals `raw_gap + |log_linear_deficit|`, so a broken or weak arm can post a large "lift" that is pure deficit add-back. Three changes followed and are baked into this run: (1) the estimand now emits **`raw_gap`** (the deployment-terms arm−log gap on the condition) beside every `dc_lift`; (2) the `mlp` arm was given **genuine capacity** (2-layer, width 64) so C1 is tested, not confounded by an undertrained floor; (3) a **second, GRU-path positive control** was added so GRU readings aren't licensed by the static path alone. The headline law (C1) did not survive; what replaced it is stronger.

## Reproduce

```
cd flow
uv run flow.py --config-value cfg "hydra_overrides: [experiment=consolidated]" run --max-workers 8
```
`--max-workers` is a **speed knob only**: this flow is declared `nondeterministic` (torch CPU GRU kernels are not bit-reproducible), so `foreach` branch parallelism cannot change any branch's result — per-branch determinism is enforced inside the branch (single-thread torch + BLAS + per-branch seed). The 8-seed CIs are the reproducibility guarantee. Gates: `flow-lint` (pass) · `pipeline-reviewer` fidelity · determinism `nondeterministic`. Read results from the run artifacts (`lift_results` now carries `raw_gap`, `aggregate_results`, `analyses`) or the `report` step.

## Instrument verdict: TRUSTWORTHY ✓ (both controls fire)

Two positive controls, both wired to **halt** the run (not print-only, `flow.py` `end`):
- **static path** — `static_ple` on the sharp band, deficit-corrected **+0.46 [+0.42, +0.50]**, raw **+0.414**. Validates the estimand / label-DGP path.
- **GRU path** — `gru_ple` on the sharp band, **+0.42 [+0.36, +0.48]**, raw **+0.346**. Validates the torch training/convergence path, so a silently under-trained GRU cannot depress every GRU reading while the static control still fires (peer-review M4).

Both detect the strongest known lever in **raw** terms, so the GRU curvature/smooth readings are interpretable. Two gate diagnostics are **reported as False** and surfaced honestly below (they are *not* halt conditions).

## The estimand — and its decomposition

Deficit-corrected, seed-paired: `dc_lift = (arm − log)_condition − (arm − log)_log_linear`. Algebraically `dc_lift = raw_gap − deficit`, where **`raw_gap`** = mean seed-paired `(arm − log)` on the condition (the uncorrected arm-vs-log gap) and **`deficit`** = `(arm − log)` on `log_linear` (the structural encoder tax, usually $\le 0$). *`raw_gap` is a quantity, not the `raw` **encoder** arm (`static_raw`/`gru_raw`) — those are rows in the `arm` column; in the printed table this quantity is labelled `gap` to avoid the collision.* **Every lift below is shown with its `raw_gap`** so add-back cannot masquerade as a lever. PR-AUC on raw scores (rank); log-loss/Brier on **calibrated** probabilities; 8-seed paired-t 95% CI + **Holm** across the family. Read magnitude from `raw_gap`; read `dc_lift` as the structural (net-of-tax) quantity.

## Headline findings (K=6, deficit-corrected; raw_gap = deployment gap)

| claim | dc_lift | **raw_gap** | verdict |
|---|---|---|---|
| **Sharp non-monotone is the dominant PLE lever** (`static_ple` / `gru_ple`, sharp_mode) | +0.46 / +0.42 | **+0.41 / +0.35** | ✅ real, large, holds in raw — the one clean lever |
| **Sharp lever robust to band location** (`gru_ple`, sharp_off) | +0.14 | **+0.07** | ✅ Holm-sig, raw-CI-clear |
| **Smooth non-monotone → learned projection, not PLE** (`gru_projection`/`gru_dense`) | +0.21 / +0.21 | **+0.187 / +0.193** | ✅ real raw lever; `gru_ple` +0.03 raw, n.s. |
| **Curvature is a GRU lever — via learned projection** (`gru_projection`) | +0.13 | **+0.101** | ✅ raw-CI-clear (K=6 and K=1); `gru_ple` +0.023 raw, n.s. |

## C1 refuted, and replaced by a stronger claim: **localization, not architecture, is the lever**

The pre-registered law was *"encoding helps iff the model has no free per-step nonlinearity"* — the `mlp` arm (a free per-step nonlinearity) was supposed to show "no arm beats log." With the re-powered MLP (deficit now **$\approx 0$**, artifact removed), the arm splits the law in two:

- **Encoding IS redundant for smooth/curved.** `mlp_log` beats `mlp_ple` on smooth (0.382 vs 0.306) and roughly ties on curved — the MLP rebuilds a smooth 1-D transform from the scalar. `mlp_ple` raw_gap is −0.076 (smooth) and +0.090 n.s. (curved). **This half holds.**
- **Encoding is NOT redundant for the sharp band.** `mlp_log` collapses to **0.132** on `sharp_mode` while `mlp_ple` reaches **0.518** (near the 0.61 oracle) — raw_gap **+0.387**, Holm-sig. And `gru_log` collapses identically (0.083). Both architectures, reading a log scalar, fail to *optimize* their way to a $\sigma = 0.15$ bump; a fixed quantile basis relieves it **even for the free-nonlinearity model.**

So the operative axis is **optimization-hardness / localization**, not "does the model have a per-step nonlinearity." A razor-sharp localized target is hard to learn per-step from a scalar for MLP *and* GRU, and a fixed basis (PLE) is the fix; smooth/curved shapes are learned per-step, so there encoding only pays the deficit. This subsumes the old architecture law and is what the arms actually support.

## Curvature reframed — a real GRU lever via projection, not a "multivariate PLE" effect

The earlier "curvature IS a lever" headline rested on `gru_ple` (dc +0.098) — but its **raw_gap is +0.023 [−0.004, +0.051], not CI-clear**: ~83% of the dc number is deficit add-back. The genuine curvature lever is **`gru_projection`: raw_gap +0.101 [+0.073, +0.129] at K=6 and +0.042 at K=1**, both Holm- and raw-CI-clear. So curvature is real in the GRU, delivered by a **learned projection**, and it is **not** a purely multivariate phenomenon — it fires in raw at K=1. The "invisible at K=1 / multivariate" framing is dropped.

## Two gate diagnostics reported False — surfaced, not buried

- **`curvature_lever_detected = False`** — the *static* vehicle shows curvature −0.018 (K=6, n.s.) and **−0.014 (K=1, CI-negative)**. Curvature is not a lever in the static affine-read model; it is a GRU-only, projection-delivered lever (above). Correctly a reported finding, never gated.
- **`multivariate_control_ok = False`** — this flag required static curvature ~0 at K=1 and >0 at K=6; the static control inverts instead. It was the original license for the C8 "multivariate" framing, and it does not reproduce. That framing is retired; the GRU curvature result stands on `gru_projection`'s raw K=1/K=6 significance, not on this control.

## The `raw` floor — the add-back artifact, now visible

`gru_raw` on sharp_mode posts **dc +0.162 (Holm-sig)** but **raw_gap +0.001 [−0.008, +0.010]** — the entire "lift" is the −0.161 deficit added back. With `raw_gap` printed beside it, the floor arm can no longer read as a lever. The same lens explains why `static_raw` and the old `mlp_ple` posted large dc numbers: read raw.

## Determinism: `nondeterministic` (honest declaration)

Torch CPU GRU kernels are not bit-reproducible run-to-run even single-threaded; residual per-cell variance is ~0.005–0.01, below the seed-level CIs. The **8-seed CIs are the reproducibility guarantee**: signs, CI-exclusions, and the gate verdict reproduce; exact aggregates vary at the 3rd decimal. `--max-workers` does not enter this — it only schedules independent, seeded, single-thread branches.

## Novelty / scope

The core that a fixed/learned basis rescues an affine (or affine-read) model on non-monotone/localized numeric structure is prior art (GAM/spline basis expansion; PLE from Gorishniy et al., *On Embeddings for Numerical Features in Tabular Deep Learning*, NeurIPS 2022). The **contributions specific to this line** are: (1) the **optimization-hardness reframing** — a fixed basis helps a *free-nonlinearity* model on a sharp target, so the lever is localization, not the affine-read assumption; (2) the **fixed-vs-learned-by-shape crossover** in the per-step sequence setting — fixed PLE for sharp (`gru_ple` raw +0.35 $\gg$ `gru_projection` +0.10), learned projection for smooth/curved (`gru_projection` raw +0.19/+0.10 $\gg$ `gru_ple` +0.03/+0.02); and (3) a **cautionary methodology** — the deficit-corrected difference-of-differences estimand inflates weak and broken arms and must be read next to `raw_gap`.

## What this is / isn't

**Is:** a dual-positive-controlled, calibrated, Holm-corrected synthetic experiment isolating the mechanism and its signs, reproducible-by-conclusion via one `flow.py run`, with every lift decomposed into raw gap + deficit. **Isn't:** a real-fraud magnitude (synthetic; directional) — the deployment question needs data where the precondition passes (it failed on the demo data). See `CLAIMS.md` for the claim→arm map and `HYPOTHESIS.md` for the pre-registered claim (C1's architecture pillar is refuted here) and the fairness invariants.
