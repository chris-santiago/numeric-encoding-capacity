## Hypothesis — Cycle 5 (Δt encoding; cycle 5 of the fraud-encoding line)

**Context.** Cycle 4 was VOID: it compared encodings of *uninformative* time features (demo data,
time-only PR-AUC ≈ base), which is no test at all. The reference model's SHAP shows inter-transaction time (Δt)
is highly important. This cycle tests Δt encoding **where Δt is informative by construction**, with a
built-in positive control so a null is meaningful.

**Claim.** For an informative, **non-monotone** Δt feature (short Δt = card-testing burst, long Δt =
dormant-reactivation, both high-risk; medium low-risk):
- **(positive control)** A **weak (linear)** model cannot fit the non-monotone shape from `raw` or
  `log` Δt, so richer encodings (`PLE` — fit on **raw minutes** (`ple_raw`, standard) and on
  log1p(minutes) (`ple_log`, variant) — and `learned` periodic) will **beat** raw and log.
- **(the real question)** A **strong (MLP)** model learns the non-monotone transform from raw/log
  itself, so `ple_raw`/`ple_log`/`learned` will **not** beat `log` — the gap collapses (capacity argument).
- **(negative control)** In a **monotone** Δt regime, `log` already suffices: richer encodings do not
  beat log, and log ≥ raw.

**Mechanism.** PLE (localized piecewise-linear) and learned-periodic (smooth Fourier) both expand the
scalar into a basis a linear head can recombine into a non-monotone function. An MLP already
approximates non-monotone 1-D functions from raw/log, so the basis is redundant for it.

**Signal.** PR-AUC lifts between Δt encodings, per model capacity (linear, MLP) and regime
(non-monotone, monotone), with bootstrap CIs.

**Expected observable.**
- Positive control — (linear, non-monotone): `ple − raw` and `learned − raw` CI-separated **positive**
  (and larger than `log − raw`). **If this fails, the experiment has no power → halt** (cycle-4 lesson).
- Real question — (MLP, non-monotone): `ple − log` and `learned − log` CIs **overlap zero**.
- Negative control — (MLP, monotone): all encodings tie; `log − raw` ≥ 0; `ple − log`, `learned − log`
  overlap zero.

**Precondition (HARD GATE, cycle-4 fix).** Before reading any encoding verdict, assert Δt is
informative: Δt-only PR-AUC (best encoding under MLP) must be well above the base rate. By
construction it will be; if it ever is not, the run is void and must halt — encoding comparisons on a
no-signal feature carry zero evidential weight.

## Evaluation Metrics

**Primary:** PR-AUC (average precision); ~8% base rate (imbalanced, fraud-like).
**Secondary (diagnostic):** the fitted fraud-vs-Δt response curves; learned-frequency spectra.

**Data.** Synthetic. Δt = inter-transaction **minutes = exp(latent m_log)** (always > 0, heavy-tailed
— no clipping artifact); `raw` = standardized minutes, `log` = standardized log1p(minutes). Two
regimes: **non-monotone** (U-shaped fraud risk in the log scale) and **monotone** control. One mild
co-feature for realism (Δt dominant). Encodings: `raw`, `log`, **`ple_raw` (PLE quantile bins on raw
minutes — the standard form), `ple_log` (PLE on log1p(minutes))**, `learned` (periodic PLR on the
standardized log scale; a periodic basis needs a normalized input, so the heavy-tailed raw scale is
ill-posed for it). Train-only scalers/PLE edges (causal-style). PoC (Step 1) is synthetic; this *is* the construct-valid test (no separate real-data
step — the demo cannot test this, and real-world data is not accessible here).

**Domain:** dt_encoding
