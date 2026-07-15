# Conclusions — Per-step numeric encoding in an affine-input long-sequence GRU

**Cycle 6 of the fraud-encoding line. Review: none (controls-based, per user). Mode: conclusions_only.**
Promoted gated Metaflow flow (PerStepFlow), **4/4 gates passed** (lint, fidelity review, determinism
`single_worker` verified). Synthetic per-account sequences; regime {band, monotone} × length {32, 300}
× 5 arms × 5 seeds. PR-AUC; **seed-level paired-t 95% CIs + Holm** over the band decision family
(cycle-5 lesson); ±0.005 equivalence margin. Pre-registered gates: precondition, positive control,
negative control.

## Headline

**In an affine-input GRU — the reference architecture, where per-step numerics enter the recurrence
as `W·e(x_t)` with no per-step nonlinearity — unbottlenecking the per-step numeric path improves fraud
PR-AUC substantially, and a fixed PLE basis is the best encoding, beating both the `scalar` baseline
AND a learned per-step Dense, at both L=32 and reference sequence length L=300.** Every band-regime lift is
CI-separated and Holm-significant; all three control gates pass.

- **PLE beats scalar:** `ple − scalar` = **+0.190 [+0.151, +0.230]** (L32), **+0.208 [+0.149, +0.266]**
  (L300), p<0.001, Holm.
- **Per-step Dense beats scalar (positive control fires):** `dense − scalar` = **+0.134 [+0.114,
  +0.153]** (L32), **+0.143 [+0.076, +0.210]** (L300), Holm.
- **The cheap fixed PLE basis beats the learned per-step Dense:** `ple − dense` = **+0.057 [+0.026,
  +0.087]** (L32), **+0.065 [+0.045, +0.085]** (L300), Holm.
- **Length:** the benefit is slightly *larger* at L=300 — it holds at the reference model's sequence length (not just present).
- **Controls:** precondition pass (oracle 0.72 / 0.70 ≫ 0.09 base); positive control fires both lengths;
  negative control clean (monotone: no arm beats scalar; PLE slightly *hurts* there, −0.029* at L32).

## PR-AUC by arm (5-seed mean; band regime; base ~0.09)

| Arm | L=32 | L=300 |
|---|---|---|
| tab_logreg (trivial) | 0.297 | 0.141 |
| raw (std raw amount/Δt) | 0.262 | 0.232 |
| **scalar = log** (reference baseline) | **0.401** | **0.372** |
| dense (per-step Dense+ReLU) | 0.534 | 0.514 |
| **ple** (fixed PLE basis) | **0.591** | **0.579** |
| oracle (precondition probe) | 0.723 | 0.703 |

The full four-way ranking is **raw < log < dense < ple**, every gap CI-separated and Holm-significant:
`log − raw` = +0.138 (L32) / +0.140 (L300) (conditioning — heavier here than the static-model +0.06,
since raw's tail fed affinely into the recurrence is especially bad), then `dense − log` and `ple − log`
add more on top. `ple` recovers ~80% of the oracle's PR-AUC; the `log`-scalar baseline leaves ~0.19–0.21
to `ple` above it and sits +0.14 above `raw` below it. See `fig_summary_prauc.png`, `fig_lift_forest.png`, `fig_poc_prauc.png`.

## Interpretation

The GRU reads each per-step numeric only affinely (`W·e(x_t)`) before the gate nonlinearity, so the
per-step function class is `span(e)`. A single `log` scalar gives one monotone shape per feature, which
cannot cheaply form the band-selective, cross-feature per-step detectors the band signal needs; the
GRU's gate×candidate products can synthesize *some* per-step non-monotonicity but inefficiently, while
also having to serve memory. A fixed PLE basis hands the GRU localized 1-D bands directly, decoupled
from the gates — and that **outperforms even a learned per-step Dense**, because the Dense must *learn*
the band structure (harder to optimize, higher variance) whereas PLE supplies it for free. This is the
capacity argument inverted by architecture: where a free-nonlinearity model finds a basis redundant,
the affine-input GRU finds it decisive.

## Honest caveats

- **PLE requires adequate training; the result is sensitive to it.** A first flow run with leaner
  capacity/epochs (hidden 24, 20 epochs, smaller val set) made the PLE arm *unstable* — huge
  seed-variance and a spurious negative lift. With proper capacity (hidden 32, 30 epochs, larger val
  for early-stopping) PLE wins cleanly. **Deployment note:** feeding raw PLE (many correlated bins) into
  a GRU input needs sufficient capacity and a stable early-stopping signal, or it can underperform.
- **Synthetic data; magnitudes are illustrative.** Construct validity is established by the firing
  positive control + precondition; the band signal is deliberately the regime where per-step encoding
  *can* help (band-selective, cross-feature, recency-aggregated). It is not a claim about the specific
  shape of real fraud signal.
- **Recency-weighted aggregation** (leaky-integrator) was used so the cross-step aggregation is
  GRU-tractable at any length; a total-count-over-L signal would instead test long-range counting
  capacity (a confound, deliberately avoided). Consequently length-dependence is mild by design.
- **The reference-model answer still requires the A/B.** This is the construct-valid stand-in: it proves the
  *mechanism* (affine bottleneck → basis helps) at the reference model's sequence length, not the *magnitude* on real
  fraud, which depends on whether real amount/Δt-in-context carry band-selective per-step signal.

## Hypothesis closure

- **Positive-control claim — CONFIRMED:** `dense − scalar` CI-separated positive (Holm) at both lengths
  → the affine bottleneck is real and the test is powered.
- **The real question — CONFIRMED (and stronger):** PLE beats `scalar` (Holm, both lengths); PLE even
  beats the learned per-step Dense. Unbottlenecking helps, and the cheap fixed basis is best.
- **Negative control — CONFIRMED:** in the monotone regime `scalar` suffices; no arm beats it.
- **Through-line (the reconciliation of the whole line):** learned/fixed numeric bases were useless in
  every *free-nonlinearity* model (linear, GBDT, MLP — cycles 1–5) and **help decisively in the
  affine-input GRU** (cycle 6). The architecture, not the feature, decides. PLE-on-amount/Δt — dismissed
  across five cycles — is the right move *for this model class*, exactly as the affine-input analysis
  predicted.

## New hypothesis (next, not run)

The reference-model A/B: per-step PLE on {amount, Δt} vs the `log`-scalar baseline (and vs a per-step Dense)
in the real 300-seq GRU, seed-level CI-excludes-zero bar, with adequate capacity/epochs for the PLE arm.
Expected from this cycle: PLE > Dense > scalar if real amount/Δt-in-context carry band-selective signal.
