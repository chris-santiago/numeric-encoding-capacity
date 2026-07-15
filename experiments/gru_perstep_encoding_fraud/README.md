# Per-step numeric encoding in an affine-input long-sequence GRU (cycle 6)

**Hypothesis (one paragraph).** The reference GRU reads per-step numerics (amount, Δt) affinely (`W·e(x_t)`, no per-step nonlinear projection), so each numeric's per-step function class is `span(e)` — a single monotone shape for a `log` scalar. When the fraud signal is band-selective and must be aggregated across steps, that bottleneck should cost accuracy, and unbottlenecking the per-step path — via a fixed PLE basis or a learned per-step Dense — should help. In a monotone regime, `log` already suffices. This is the one regime prior cycles never tested (they used free-nonlinearity models or small/short GRUs).

## Quickstart

```bash
uv run gru_perstep_poc.py     # Step-1 synthetic mechanism PoC (small L); writes fig_poc_prauc.png
```

## Pipeline (PoC)

synthetic per-account sequences (band / monotone regime) → per-step encoding (scalar `log` | PLE | per-step Dense) → GRU (affine per-step input) → PR-AUC; plus a trivial tabular-aggregate baseline and an oracle (true band-count) precondition check.

## What the output looks like

PR-AUC per `(regime × arm)` with the three pre-registered checks. Headline PoC result: band regime `ple − scalar = +0.156`, `dense − scalar = +0.114` (positive control fires); PLE (0.578) > dense (0.537) > scalar (0.423); precondition passes (oracle 0.646 ≫ 0.09); monotone negative control clean (max +(-0.005)). Figure: `fig_poc_prauc.png`.

## Intent review (Step 2)

- **`scalar` and `dense` share the same input tensor `[log amt, log Δt]`; they differ only in whether a per-step `Dense→ReLU` runs before the affine GRU read.** That isolates the affine-bottleneck variable exactly — `dense` is the mechanism/positive control.
- **PLE is fit on RAW values** (quantile bins on raw amount/Δt), per the prior-cycle correction; it supplies localized 1-D bands.
- **The band signal is a cross-feature conjunction (amount-band AND Δt-band) aggregated across steps** — deliberately the case a single affine-read scalar struggles with and a basis/Dense can expose.
- **The trivial baseline uses generic aggregates** (mean/std/min/max of log amount and log Δt), which do not encode the band-count, so it is a fair limited baseline (not a leak).
- **No debate this cycle (user choice);** rigor is carried by the pre-registered precondition gate, positive control, and negative control.

## Known limitations / scope exclusions

- Synthetic data; magnitudes illustrative. Construct validity established by the firing positive control + precondition.
- PoC is small-L, 3-seed means; the promoted Metaflow flow runs the length axis {32, 300} with seed-level paired CIs.
- One band shape and one monotone control; the question is the *bottleneck mechanism*, not a specific real fraud shape.
- The result informs the reference model only insofar as it shares the affine per-step input (confirmed); real-data magnitude is the reference-model A/B's job.
