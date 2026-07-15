# Claim → verification map (cycles 1–8 → the consolidated flow)

Every mechanistic claim in the line, and the exact arm/control/estimand in `flow/` that verifies it. Read
alongside `HYPOTHESIS.md`. Results live in the Metaflow run artifacts (`lift_results`, `aggregate_results`,
`analyses`), surfaced by `report`.

| # | Claim (cycle) | Verified by |
|---|---|---|
| 1 | ~~Encoding helps iff no free per-step nonlinearity~~ → **REFUTED & REFRAMED: localization (optimization-hardness), not architecture, is the lever** | `mlp` arm (free per-step NL, re-powered 2×64, deficit≈0): redundant for smooth/curved (`mlp_log`≥`mlp_ple`) but **PLE still helps on sharp, raw +0.387** — and `gru_log` collapses on sharp identically (0.083). A sharp localized target is optimization-hard to learn per-step from a scalar for *both* MLP and GRU; a fixed basis relieves it regardless of the per-step nonlinearity |
| 2 | Affine GRU gates are smooth approximators; only localized structure is a lever (C6–8) | `gru` arms across conditions: `sharp_*` fires, `smooth_nonmono` absorbed, `monotone_curved` a masked lever |
| 3 | Lever hierarchy sharp ≫ curvature ≫ smooth ≈ log-linear (C7, sharp_vs_smooth) | `deficit_corrected_lifts` across the 5 conditions in `static_linear` (the powered vehicle) |
| 4 | ~~Curvature is a *multivariate* phenomenon~~ → **NOT reproduced; curvature is a GRU lever via learned projection, real at K=1 too** | `control_gate.multivariate_control_ok = **False**` (static curvature inverts: −0.014 CI-negative at K=1). The genuine lever is `gru_projection` curvature: **raw_gap +0.101 (K=6) and +0.042 (K=1)**, both raw-CI-clear. `gru_ple` curvature is mostly add-back (raw +0.023, n.s.). Multivariate framing retired |
| 5 | Encoder by sharpness: sharp → PLE, smooth-curved → projection (fixed_vs_learned, nonmono_encoders) | `gru_ple` vs `gru_projection` deficit-corrected lifts by condition (sharp vs curved) |
| 6 | `raw` < `log` for an affine read (C6) | `*_raw` arms vs `*_log` — raw is the floor |
| 7 | The dimensionality deficit is the binding constraint (C8, deficit_curve) | PLE structural deficit measured on `log_linear` (static ~small, gru large); the difference-of-differences estimand nets it |
| 8 | Precondition: model must beat a strong aggregate baseline (C3/C8 F3) | `tabular` (GBM+EWMA) as comparator; oracle as ceiling |
| 9 | Instrument must be positive-controlled, or a null is uninterpretable (C8 retraction, reviews) | **hard gate**: `control_gate` halts the verdict unless the **sharp** lever fires CI-positive in BOTH `static_ple` (estimand/label path) AND `gru_ple` (training/convergence path). Sharp-only by pre-registration (HYPOTHESIS.md §Controls); curvature is a reported finding, not a halt condition |

## Deficits / mistakes accounted for (all baked in structurally — see HYPOTHESIS.md fairness invariants)

undertraining (proper regime, val early-stop, equal budget) · single-feature invisibility (K=1 control) ·
un-netted deficit (difference-of-differences) · contaminated reference (`log_linear` = marginal-matched
log-adequate) · signal-share (equal weights) · rank-metric blindness (calibrated log-loss/Brier) ·
print-only gates (control_gate wired into `report`, halts) · weak baseline (GBM+EWMA) · raw-fed-affinely
(`raw` floor, log-space encoding) · coordinate mismatch (PLE + projection both on log-space `sc`) ·
on-mode confound (`sharp_mode` + `sharp_off`) · multiplicity (Holm to be applied at read) · in-sample eval
(train/val/test, val selection) · sweep-override (single fixed config/arm) · reshuffle asymmetry (identical
per-epoch shuffle) · **calibration (every arm temperature-scaled before magnitude metrics)**.

**Peer-review round (this revision):** deficit add-back masquerading as a lever (`raw_gap` now emitted beside
every `dc_lift`) · undercapacity `mlp` floor confounding C1 (arm re-powered to 2×64, deficit≈0) · single-vehicle
control (second **GRU-path** positive control added, both halt) · worker/determinism doc conflict (flow is
`nondeterministic`; `--max-workers` is a speed knob, not a contract).

## Reproduce

```
cd flow
uv run flow.py --config-value cfg "hydra_overrides: [experiment=consolidated]" run --max-workers 8
```
`--max-workers` is a **speed knob only** — the flow is `nondeterministic` (declared), so `foreach` branch
parallelism cannot change any branch's result; the 8-seed CIs are the reproducibility guarantee.
Gates: `flow-lint.py` (pass) → `pipeline-reviewer` (fidelity) → determinism `nondeterministic` (gate self-skips).
