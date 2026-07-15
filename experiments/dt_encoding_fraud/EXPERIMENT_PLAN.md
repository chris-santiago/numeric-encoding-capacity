# Experiment Plan — Step 6 (Gate 1), Δt encoding (cycle 5)

**Review mode:** debate. **Case verdict:** `empirical_test_agreed` (converged round 2, all-ACCEPT).
**Report mode:** conclusions_only. Source of truth: `HYPOTHESIS.md`.

Step 6 is a direct experiment script (`dt_encoding_experiment2.py`) — **no Metaflow promotion** (fast
synthetic grid; rigor comes from the controls + bootstrap CIs + debate, not pipeline orchestration).

## Pre-flight checklist (debate-derived)

| # | Source | Finding | Verdict | Item | Resolution in Step 6 | Status |
|---|--------|---------|---------|------|----------------------|--------|
| 1 | DEFER | F1 | ETA | learned+MLP has more params than log+MLP; a non-convex PLR+MLP could *under-train* → false null | (a) **param-matched control** (`log_expand` = learnable non-periodic expansion on log, same budget as PLR); (b) **convergence verification**: longer training + report **train-loss** for learned vs log (a false null shows learned with *higher* train loss; a real null shows learned fitting train ≥ log but tying on test) + learned-frequency spectrum | CLOSED (test specified) |
| 2 | DEFER | F5 | ETA | 3-seed means, no CIs | **Bootstrap CIs** (N=1000) on all lifts; bar = **CI excludes zero** (data-derived, not arbitrary) | CLOSED |
| 3 | REBUT-SCOPE | F4 | defense | `raw`=expm1(clip(dt_log,0)) clipped sub-minute info → unfair raw baseline | **FIXED at PoC level:** Δt now = exp(latent) (always >0, no clip); raw and both PLE variants share clean support | CLOSED |
| 4 | DEFER | F3 | ETA | MLP ≠ sequence GRU | **Deferred to the reference model** (cycle 5 scope = linear+MLP per Gate-1 choice); documented as the remaining open question | CLOSED (deferred w/ rationale) |
| 5 | non-negotiable | — | — | trivial baseline + power | base rate; **positive control** (linear,nonmono) must fire; **precondition gate** (Δt-only ≫ base) | CLOSED |

Diagnostics added (cheap, from F7/F8): **z-only floor** (co-feature-only PR-AUC) and learned-freq spectrum.

## Pre-registered verdict rules (CI-excludes-zero)

- **Positive control (power gate)** — (linear, nonmono): `ple−raw`, `learned−raw` CIs must **exclude
  zero (positive)**, and exceed `log−raw`. If this fails, the run is **void** (no power). HARD GATE.
- **Precondition** — best-MLP nonmono PR-AUC ≫ base. HARD GATE.
- **Convergence (F1)** — learned+MLP train-loss ≤ log+MLP train-loss (learned fit the train signal at
  least as well). If learned train-loss is materially higher, the test arm is under-trained → fix
  before reading the test verdict.
- **The real question** — (MLP, nonmono): `ple_raw−log`, `ple_log−log`, `learned−log`, and
  `learned−log_expand` (matched capacity) CIs. Defense-right (capacity argument): CIs **overlap zero**.
  Critique-right: CI-separated positive AND point estimate > 0.05.
- **Negative control** — (MLP, mono): `log−raw` ≥ 0; `ple−log`, `learned−log` CIs overlap zero.

## Conditions

Regimes {nonmono, mono} × encodings {raw, log, ple_raw, ple_log, learned, log_expand(matched)} ×
models {linear, mlp}. 5 seeds. PR-AUC, paired test-set bootstrap 95% CIs (N=1000). Synthetic, Δt =
exp(latent) informative by construction (no clip); co-feature mild; train-only scalers/PLE edges.

## Out of scope (documented)

Sequence/GRU arm (F3 → reference-model swap); σ/k/n_bins sweep (fixed, σ spot-checked via convergence);
asymmetric/spike Δt shapes (F6 — the symmetric U is the strongest case for richer encodings, so the
null is conservative).
