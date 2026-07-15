## Hypothesis — Cycle 1 (per-step numeric encoding in an affine-input long-sequence GRU; cycle 6 of the fraud-encoding line)

**Context.** The reference model is a GRU (seq ~300) in which per-step numeric features (transaction amount, inter-transaction time Δt) enter the recurrence **affinely** — `W·e(x_t)` into the gates, with **no per-step nonlinear projection** (confirmed by the user). Under that architecture the per-step reachable function class for each numeric is `span(e)`: a single `log` scalar gives one monotone shape per feature. Prior cycles found learned encodings useless for amount/time, but every one used a free-nonlinearity model (linear, GBDT, MLP) or a small/short GRU — never an affine-input GRU at the reference model's sequence length. This cycle tests the one untested regime.

**Claim.** When per-step numerics carry **band-selective, non-monotone, in-context signal that must be aggregated across steps**, unbottlenecking the per-step numeric path improves fraud PR-AUC in the affine-input GRU:
- **Mechanism control / positive control:** a per-step **Dense** projection (free per-step nonlinearity) beats the `scalar` (`log`) baseline in the band regime — `dense − scalar` CI-separated positive. This both establishes power and proves the affine bottleneck is the cause.
- **The real question:** a fixed **PLE** basis per numeric also beats `scalar` (`ple − scalar` > 0), recovering some/all of the `dense` gain cheaply (no learned per-step transform). `ple − dense` quantifies whether the cheap basis matches full per-step nonlinearity.
- **Negative control:** in a **monotone** regime (per-step monotone response, no bands), `scalar` already suffices; neither `ple` nor `dense` beats it.
- **Length dependence (measured, weak prior):** the cross-step aggregation is recency-weighted (leaky-integrator, GRU-tractable at any length), so the unbottlenecking benefit may be roughly **length-flat** — the GRU's effective memory window, not L, bounds the aggregation. (A total-count-over-L signal would instead test long-range counting capacity, a confound; it is deliberately avoided.) The length axis {32, 300} measures whether the benefit holds at the reference model's sequence length, not necessarily that it grows.

**Mechanism.** The GRU reads per-step numerics only as `W·e(x_t)` before the gate nonlinearity, so the per-step function class is `span(e)`. A `log` scalar → one monotone shape per feature; the GRU can synthesize *some* per-step non-monotonicity via gate×candidate products, but forming band-selective, cross-feature (amount-band AND Δt-band) per-step detectors and feeding them to cross-step aggregation is inefficient. PLE supplies localized 1-D bands directly; a per-step Dense supplies arbitrary per-step nonlinearity including the cross-feature conjunction. The benefit is expressiveness/decoupling/conditioning, not strict capacity.

**Signal/observable.** Seed-level paired PR-AUC lifts `dense − scalar`, `ple − scalar`, `ple − dense`, per regime and per length, with paired-t 95% CIs + Holm over the band-regime decision family. Equivalence to `scalar` declared only within a ±0.005 margin.

**Expected observable.**
- Band regime: `dense − scalar` CI-separated positive (positive control fires → powered). `ple − scalar` positive (basis helps). Holds at both L=32 and L=300 (length-flat is acceptable).
- Monotone regime: `dense − scalar`, `ple − scalar` CIs overlap zero (negative control).
- If the positive control does NOT fire, the run is **void** (no power) — HARD GATE.

**Precondition (HARD GATE).** The band signal must be learnable: an oracle feature (the true per-step band-match count) under a logreg must give PR-AUC ≫ base rate. Asserted before any encoding verdict; by construction it will pass.

## Evaluation Metrics

**Primary:** PR-AUC (average precision); imbalanced fraud-like base rate ~0.08.
**Secondary (diagnostic):** convergence epochs; per-arm input dim.

**Data.** Synthetic per-account sequences (L steps). Two per-step numerics: amount and Δt (log-normal). **Band regime:** fraud rises with a **recency-weighted (leaky-integrator) aggregation** of steps where Δt is in a short band AND amount is in a small band (band-selective in both → non-monotone; conjunction → cross-feature; recency-weighting keeps the aggregation GRU-tractable at any length, isolating the per-step band-detection encoding lever from long-range counting capacity). **Monotone regime:** fraud monotone in the sequence-mean log-amount (control). Mild label noise; intercept calibrated to ~8% base. PLE bin edges / scalers fit on train only. The PoC (Step 1) is a small-L synthetic proof; the promoted Metaflow flow runs the length axis {32, 300}.

**Arms (shared GRU backbone, affine per-step input; only the per-step numeric encoding varies):** `raw` (std raw amount, std raw Δt — conditioning baseline) · `scalar` (log amount, log Δt — reference baseline) · `ple` (PLE per numeric) · `dense` (per-step Dense+ReLU on the log scalars — free per-step nonlinearity / mechanism control) · `tab_logreg` (trivial baseline: logreg on generic sequence aggregates — non-negotiable) · `oracle` (precondition probe). The four decision encodings are **raw / scalar(=log) / ple / dense**.

**Review mode:** none (no debate, per user); rigor carried by the pre-registered positive control, negative control, and precondition hard gate. **Report mode:** conclusions_only. **Promotion:** gated Metaflow flow (lint → pipeline-reviewer → determinism `single_worker`). **Stats:** seed-level paired-t CIs + Holm (cycle-5 lesson), not single-seed bootstrap.

**Domain:** gru_perstep_encoding
