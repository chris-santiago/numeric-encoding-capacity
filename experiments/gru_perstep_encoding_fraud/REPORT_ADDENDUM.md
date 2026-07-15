# Reference-Model Re-Evaluation Addendum — per-step encoding, fraud sequence GRU

Step 9. The reference model is a GRU (seq ~300) in which per-step numerics (amount, Δt) enter
the recurrence **affinely** — confirmed by the user: `[…, log amt, log Δt]` concatenated straight into
the GRU input weights, with no per-step Dense/MLP projection. This cycle tested, in a construct-valid
synthetic GRU at the reference model's sequence length, whether unbottlenecking that per-step path helps.

## Recommendation

**Try per-step PLE on amount and Δt in the reference GRU — it is the most promising single change this
investigation has surfaced.** Given the confirmed affine per-step input, the per-step numerics are
currently bottlenecked at one monotone scalar each; a fixed PLE basis unbottlenecks them and, in the
construct-valid test, **beat both the `log`-scalar baseline (+0.19–0.21 PR-AUC, Holm-sig) and a learned
per-step Dense (+0.06–0.07, Holm-sig), at both L=32 and L=300.** This is the opposite of the cycles 1–5
recommendation — and correctly so, because those used free-nonlinearity models where a basis is
redundant; your GRU is the architecture where it is decisive.

Two ways to unbottleneck, in priority order:
1. **Per-step PLE on {amount, Δt}** (fixed quantile bins on raw values) — cheapest, no learned per-step
   transform, and the best arm in the test. **First thing to try.**
2. **A per-step Dense+ReLU projection on the numeric block** — also helps (general, learns the
   transform), but underperformed PLE and adds trainable params + overfit surface.

## Why (deployment constraint areas)

*(Accuracy effects are measured on synthetic data; operational points are reasoned, not benchmarked.)*
1. **Retraining dynamics.** PLE adds quantile edges to refit on drift (cheap, deterministic). A per-step
   Dense adds trainable params. Both are modest; the measured accuracy gain justifies the cost here
   (unlike cycles 1–5, where there was no gain).
2. **Update latency.** Per-step PLE is a bin-lookup + clip per step — negligible vs the GRU itself.
3. **Operational complexity.** PLE needs versioned bin edges and a stable early-stopping signal (see
   the stability caveat); a per-step Dense needs no edges but more tuning.
4. **Failure modes.** **PLE is training-stability-sensitive:** with insufficient GRU capacity / epochs /
   a noisy early-stopping validation set, the PLE arm was unstable and *underperformed* in a first run.
   Adopt it with adequate hidden width, enough epochs, and a large enough validation set for
   early-stopping — otherwise it can silently regress.

## The decisive architectural fact

This recommendation flips *because* the per-step input is affine. If a per-step Dense/MLP projection
were ever added to the reference model's numeric path, it would restore free per-step nonlinearity and make
the PLE basis redundant again (the cycle-5 result would then transfer). So the recommendation is
conditional on the current affine input — which the user confirmed.

## Bounds / open question

- **Synthetic construct-valid test, not real-data magnitude.** It proves the mechanism (affine
  bottleneck → per-step basis helps) at the reference model's sequence length; the real gain depends on whether real
  amount/Δt-in-context carry band-selective, recency-aggregated per-step signal. **The definitive test
  is the reference-model A/B.**
- The band signal was engineered to be the regime where encoding helps; a purely monotone per-step
  signal would show no gain (negative control confirmed). The reference model likely has both; the A/B settles it.
- Length-dependence is mild by design (recency-weighted aggregation); the benefit holds at L=300.

## Deployment

Run the reference-model A/B before committing: arms = `log`-scalar (current) vs per-step PLE on {amount, Δt}
vs per-step Dense; seed-level paired CI-excludes-zero bar; ensure the PLE arm has adequate
capacity/epochs/validation. If PLE clears the bar, adopt it (cheapest); else fall back to the per-step
Dense if it clears. Until then, this is the strongest lead for the reference model.

## Cross-cycle synthesis (the resolution)

Cycles 1–5 found no learned/fixed numeric encoding helped — because every model tested had free
nonlinearity on the feature (linear, GBDT, MLP, and statically-evaluated arms), where a basis is
redundant. Cycle 6 shows that in the **affine-input GRU** — the reference architecture — a per-step
PLE basis helps decisively, beating both the scalar baseline and a learned per-step projection. The
through-line is not "encoding never helps" but **"encoding helps exactly when the model has no free
per-step nonlinearity to rebuild it"** — which is precisely the reference GRU.
