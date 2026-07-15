# Report Addendum — Cycle 8 reference-model re-evaluation

Step 9: the refutation (monotone value-curvature is not a PLE lever in a GRU) re-examined against the
operational reality of deploying per-step encoding on a reference sequence model.

## The deployment rule this sharpens

Cycle 6 established that per-step PLE helps an affine-input GRU and is the best encoding — but its
positive result rode entirely on **band-selective (non-monotone), cross-feature** structure. Cycle 8 now
draws the boundary the earlier cycles left implicit:

- **Do NOT PLE-encode a feature whose in-context risk is monotone in value — even if it is sharply
  *curved* in value.** A GRU's gates already represent monotone curvature from a `log` scalar; PLE adds
  only its structural deficit, and the deficit *grows* with the feature's importance. This holds for
  monotone-log-linear (Cycle 6 negative control) and now for monotone-curved (Cycle 8) alike.
- **Reserve per-step PLE for features with genuinely non-monotone / band-selective per-step risk** — the
  one class the recurrence handles inefficiently (Cycle 6). That is where the +0.19–0.21 lifts came from.

## Correction to the prior encoding recommendation

A recommendation of the form "per-step PLE helps this GRU, so encode amount/Δt/count" is **too broad**.
The correct, mechanism-grounded form is:

> Rank per-step features by **shape of risk-in-context**, not by curvature-vs-linear. Encode the ones
> whose risk is **non-monotone / band-selective**; leave monotone features (log-linear *or* curved) on the
> `log` scalar. For a monotone feature, PLE is expected to cost ~0.03–0.04 with no offsetting gain.

Whether real amount/Δt/count-in-context carry non-monotone per-step signal is the empirical question the
reference-model A/B must answer — and it is the *only* remaining open item.

## The precondition gate is now a reusable, validated instrument

Cycle 3 failed silently because its GRU lost to a tabular baseline and ignored temporal order — never
checked. Cycle 8 built and validated the missing gate:

- **Baseline strength matters (F3).** Against a recency-agnostic logreg the gate passes too cheaply;
  against a **GBM with EWMA aggregates matching the decay structure**, the GRU margin is +0.056
  [+0.042, +0.070] and the order-shuffle drop is +0.236 [+0.187, +0.286] — both CI-clear. Any real-data
  encoding study must gate against the *stronger* baseline or its "GRU works here" premise is unearned.
- **Estimand power must be shown, not assumed (F1).** A flat deficit-corrected curve is only interpretable
  alongside a free-nonlinearity arm (`dense`) and an oracle ceiling; without them, a null conflates "no
  lever" with "no power." The real-data A/B must carry the same three-probe triangulation.

## Operational cost of PLE (unchanged, reaffirmed)

- **Dimensionality & training stability:** many correlated bins per step into the recurrence need adequate
  capacity and a stable early-stopping signal (Cycle 6 caveat, reaffirmed here — the un-hardened PoC's
  spurious −0.19 deficit was pure undertraining). Under-resourced PLE-in-GRU shows spurious negative lifts.
- **Monitoring:** quantile bin edges are fit on training data and must be refit on drift; `log` has no
  moving part. This cost is only worth paying where the non-monotone lever is real.

## Direction-only; magnitudes are synthetic

As in every cycle of this line, magnitudes are illustrative. Cycle 8 establishes **mechanism and
direction** (monotone curvature is not a GRU lever; the deficit is), not a real-fraud number. The
reference-model A/B — targeting non-monotone features, precondition-gated with the validated instrument —
remains the only test that yields a deployable magnitude.
