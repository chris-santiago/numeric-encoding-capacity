# Claim → verification map (cycles 1–8 → the consolidated flow)

Every mechanistic claim in the line, and the exact arm/control/estimand in `flow/` that verifies it. Read
alongside `HYPOTHESIS.md`. Results live in the Metaflow run artifacts (`lift_results`, `aggregate_results`,
`analyses`), surfaced by `report`.

| # | Claim (cycle) | Verified by |
|---|---|---|
| 1 | Encoding helps iff no free per-step nonlinearity (C1–6 through-line) | `perstep_mlp` arms (free per-step NL) show no arm beats `log`; `static_linear` (no NL) shows PLE lever — cross-architecture contrast |
| 2 | Affine GRU gates are smooth approximators; only localized structure is a lever (C6–8) | `gru` arms across conditions: `sharp_*` fires, `smooth_nonmono` absorbed, `monotone_curved` a masked lever |
| 3 | Lever hierarchy sharp ≫ curvature ≫ smooth ≈ log-linear (C7, sharp_vs_smooth) | `deficit_corrected_lifts` across the 5 conditions in `static_linear` (the powered vehicle) |
| 4 | Curvature is a *multivariate* phenomenon (C7/C8 root cause) | multivariate control: `monotone_curved` deficit-corrected lift ~0 at **K=1**, positive at **K=6** (`control_gate.multivariate_control_ok`) |
| 5 | Encoder by sharpness: sharp → PLE, smooth-curved → projection (fixed_vs_learned, nonmono_encoders) | `gru_ple` vs `gru_projection` deficit-corrected lifts by condition (sharp vs curved) |
| 6 | `raw` < `log` for an affine read (C6) | `*_raw` arms vs `*_log` — raw is the floor |
| 7 | The dimensionality deficit is the binding constraint (C8, deficit_curve) | PLE structural deficit measured on `log_linear` (static ~small, gru large); the difference-of-differences estimand nets it |
| 8 | Precondition: model must beat a strong aggregate baseline (C3/C8 F3) | `tabular` (GBM+EWMA) as comparator; oracle as ceiling |
| 9 | Instrument must be positive-controlled, or a null is uninterpretable (C8 retraction, reviews) | **hard gate**: `control_gate` halts the verdict unless `static_ple` curvature & sharp lifts fire CI-positive |

## Deficits / mistakes accounted for (all baked in structurally — see HYPOTHESIS.md fairness invariants)

undertraining (proper regime, val early-stop, equal budget) · single-feature invisibility (K=1 control) ·
un-netted deficit (difference-of-differences) · contaminated reference (`log_linear` = marginal-matched
log-adequate) · signal-share (equal weights) · rank-metric blindness (calibrated log-loss/Brier) ·
print-only gates (control_gate wired into `report`, halts) · weak baseline (GBM+EWMA) · raw-fed-affinely
(`raw` floor, log-space encoding) · coordinate mismatch (PLE + projection both on log-space `sc`) ·
on-mode confound (`sharp_mode` + `sharp_off`) · multiplicity (Holm to be applied at read) · in-sample eval
(train/val/test, val selection) · sweep-override (single fixed config/arm) · reshuffle asymmetry (identical
per-epoch shuffle) · **calibration (every arm temperature-scaled before magnitude metrics)**.

## Reproduce

```
cd flow
uv run flow.py --config-value cfg "hydra_overrides: [experiment=consolidated]" run --max-workers 1
```
Gates: `flow-lint.py` (pass) → `pipeline-reviewer` (fidelity) → `determinism-check.py` (run-twice @1 worker).
