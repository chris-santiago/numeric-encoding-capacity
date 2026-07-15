# Cycle 7 — Non-monotonicity vs value-curvature (mechanism disambiguation)

**Hypothesis (v2).** In an affine-read model, PLE beats a `log` transform for a **sum of
monotone-but-log-mismatched features** — so log-mismatched *value-curvature*, with no
non-monotonicity present, is a lever via feature *combination*. This isolates the mechanism the
synthesis (cycles 5–6) conflated with non-monotonicity, and is the controlled synthetic analog of
cycle 2's `C1` discovery (a monotone-by-rank, curved-in-value feature that lifted +0.144 in a
multivariate model). The decisive test is `ple − log` on the linear (affine-read) head in the
monotone-log-mismatched cell (S2).

## Quickstart

```bash
uv run curvature_poc.py
```

## Pipeline

Synthetic → encode → fit → score:
- **Data:** K=6 i.i.d. features `x_k = exp(latent)`, additive signal `logit = Σ f_k(x_k)`, three
  shapes: **S1** log-fit (control), **S2** mono-but-log-mismatched (`f_k ∝ rank(x_k)`, the
  **discriminator**), **S3** non-monotone (control). Base rate ~8.5%.
- **Encodings:** `raw`, `log`, `ple` (16 quantile bins/feature). (`dense` deferred to Step 6.)
- **Heads:** `linear` (affine-read = LogisticRegression), `mlp` (free-nonlinearity = MLPClassifier).
- **Metric:** PR-AUC (average precision), 3 seeds.

## What the output shows

A per-shape × per-head PR-AUC table, the decisive `ple − log` on the linear head per shape, and
`curvature_poc_mechanism.png`. Prediction (H_curv): PLE beats `log` on the linear head in **S2 and
S3**, ties S1; MLP head ties everywhere.

## Known limitations / scope

- **v1 (single-feature) is superseded.** PR-AUC is rank-invariant, so a monotone single feature can't
  reveal curvature; only the multivariate additive form (v2) can. See `HYPOTHESIS.md` § Cycle 7 v2.
- **PoC is directional, not certified.** 3 seeds, no CIs, no Holm — that is Step 6.
- **Open confound (for the debate / Step 6):** PLE expands to K×16 = 96 correlated features, which at
  this sample size **overfits** — PLE currently *loses* on the S1 linear control (−0.041) and across
  all MLP cells (~−0.11), muddying the "tie" predictions even though the S2 affine-head effect
  (+0.037) is present. Step 6 must control PLE variance (regularization / fewer bins / more data) so
  the S1 and MLP controls read as clean ties before the S2 lift is trusted.
- Synthetic; `dense` encoding and the formal control gates come in Step 6.
