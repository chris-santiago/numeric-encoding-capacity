# Reference-Model Re-Evaluation Addendum — periodic embeddings, fraud sequence GRU

> ## ⛔ CYCLE 4 IS VOID — this addendum makes NO reference-model recommendation
> The underlying experiment failed its precondition: the demo dataset's time features carry no fraud
> signal (time-only PR-AUC 0.078 vs 0.064 base), so comparing their encodings is vacuous and provides
> zero evidence about learned periodic embeddings. **Do not use cycle 4 to inform the reference
> model.** The only defensible statements are: (1) keep sin/cos + Δt — the reference model's SHAP independently
> shows they are highly important (cycle 4 is irrelevant to this); (2) whether *learned* periodic
> beats fixed sin/cos / raw-Δt is **untested** and requires a construct-valid experiment (reference-model
> swap, or synthetic data with informative, structured time signal). The text below is retained as a
> record of what was run, not as a recommendation.

Step 9. The reference model is a GRU (seq~300) in which **amount is the only raw numeric**; the
other inputs are already encoded (categoricals → learned embeddings, date/time → fixed sin/cos,
inter-transaction time → log-minutes). This cycle tested the one untested Gorishniy lever — *learned*
periodic embeddings — on those time features. This addendum maps the result onto the reference model.

## Recommendation (scoped — see the reference-model importance correction below)

**Keep sin/cos for cyclic time and Δt for inter-transaction time** — these are the right features and
SHAP independently confirms they are highly important in the reference model. **The "do not adopt learned
periodic" verdict is conditional, not settled.** On the *demo* dataset, learned periodic added nothing
over fixed sin/cos (H1: −0.006, CI overlaps 0) or raw Δt (H2: −0.008, CI overlaps 0) — but that
dataset carries almost no time signal (time-only PR-AUC 0.078 vs 0.064 base), so it is an
uninformative test for the reference model, where time signal is strong. Whether *learned* periodic beats the
*fixed* sin/cos / raw-Δt encoding must be tested where time signal exists (the reference model), not concluded
from the demo null.

## Why (the reference-model constraint areas)

1. **Retraining dynamics.** Learned periodic frequencies are extra trainable parameters that must be
   refit and can drift; they buy no measured accuracy. Pure cost. Fixed sin/cos has zero parameters
   and never drifts. Skip.
2. **Update latency.** A per-step periodic embedding adds sin/cos + linear + ReLU compute at every one
   of ~300 timesteps per inference, for no gain. Fixed sin/cos (or raw log-dt) is cheaper and equally
   accurate.
3. **Operational complexity.** Learned periodic adds a tuned hyperparameter (σ, the dominant knob),
   versioned embedding weights, and another monitoring surface — all downside-only here. Avoiding it
   keeps the model simpler at no accuracy cost.
4. **Failure modes.** Adding trainable parameters on features that barely predict fraud is a mild
   overfit risk with no upside; the simplest encoding is also the most robust.

## Assumptions / bounds for the reference model's 300-seq GRU

1. **Within-architecture encoding claims are robust.** H1/H2/ENC were all tested *inside the same GRU*
   on real data with amount and context held fixed — those comparisons are clean. The time-encoding
   choice does not matter for the sequence model.
2. **The "tabular beats GRU" result is capacity-sensitive — do not over-read it.** The GRU here was
   small (hidden 24), untuned, on a subsample, and at L=32. A reference-scale tuned GRU at L=300 may
   beat the tabular baseline — but that would not change the encoding conclusion (encoding the time
   features cleverly still would not be the lever).
3. **L=300 is an extrapolation.** Account p90 ≈ 47 transactions, so L=300 was not validatable here.
   The encoding null is expected to hold at the reference model's sequence length because the time signal is weak
   regardless of sequence length, but it has not been directly measured.

## The reconciliation with Gorishniy (closes the original question)

Gorishniy's periodic embeddings help deep models on numeric features with *learnable nonlinear
structure*. The reference model's pipeline already captures that benefit where it exists — categoricals are
embedded, cyclic time uses (fixed) periodic encoding, dt is log-transformed. The remaining lever
(learning the frequencies) measured as zero **on the demo data** because its time features carry
almost no fraud signal (time-only PR-AUC 0.078 vs 0.064 base) — there was no representation gain left
to extract *there*.

## The reference model's time-feature importance (correction to scope)

The reference model's SHAP shows Δt and the cyclic sin/cos transforms are **highly important** — the opposite of
the demo data's near-zero time signal. The demo is a simulated fraud process that did not inject
hour-of-day / velocity patterns, so it is unrepresentative on the dimension this cycle tested. This
means the cycle-4 time-encoding null is **bounded to the weak-signal regime and does not transfer to
the reference model.** The learned-periodic question is therefore reopened for the reference model:
- Keep sin/cos + Δt (SHAP-confirmed important; cycle 4 does not challenge this).
- Whether *learned* periodic beats *fixed* sin/cos / raw Δt is untested where it matters. Prior from
  the PoC mechanism: cyclic likely ties fixed sin/cos for a capable model unless the reference model has strong
  non-fundamental harmonics; **Δt — if its fraud relationship is non-monotone (short=card-testing,
  long=dormant-reactivation) — is the candidate most likely to gain from a learned Fourier basis.**
- Definitive test: one-line per-step encoding swap in the reference GRU, CI-excludes-zero bar.

## Limits (what this does NOT establish)

- Not that time is irrelevant — only that *encoding* the time features (sin/cos vs learned-periodic
  vs raw) does not change fraud PR-AUC on this data.
- Not a verdict on a tuned full-scale GRU at L=300 vs tabular (capacity-sensitive; retest).
- One dataset, one architecture, single sequence length, σ not swept.

## Deployment

No change to the reference model. Keep fixed sin/cos for cyclic time features and raw log for
inter-transaction time; keep amount raw-log (established in cycles 1–3). Spend model-improvement
budget on signal — engineered velocity/balance features and model tuning — not on numeric-feature
representation.

## Cross-cycle synthesis (the portable takeaway)

Across four cycles — PLE on amount (synthetic, IEEE-CIS tabular, account sequences) and learned
periodic embeddings on time features — **no learned numeric encoding improved fraud detection on the
datasets tested.** The mechanism is consistent: Gorishniy's representation methods pay off when a
numeric feature has exploitable nonlinear structure a weak model can't bend; the *tested* amount and
time features either lacked that structure or were already handled by a capable model. **Boundary
(important):** the demo dataset under-represents time signal vs the reference model (SHAP flags Δt and sin/cos
as highly important there), so the time-encoding conclusion is bounded to the weak-time regime. The
durable takeaway is the test discipline, not a universal "encoding never helps": where a feature
carries real, structured signal (the reference model's time features), the learned-periodic question must be
tested directly — the one-line swap in the reference model — before concluding either way.
