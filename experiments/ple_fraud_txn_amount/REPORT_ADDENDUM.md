# Reference-Model Re-Evaluation Addendum — PLE on transaction amount (fraud)

Step 9. The reference model is a **sequence model** over per-card transaction history. This
addendum makes explicit how the tabular experiment maps (and fails to map) onto that target.

## Recommendation

**Do not PLE-encode the raw transaction-amount scalar in the reference model.** The direct
evidence (single-transaction tabular) shows no lift for linear/MLP and +0.008 for GBDT — not worth
the per-feature artifact + monitoring cost. This recommendation is about the *raw amount scalar*;
it is **not** a verdict on whether amount matters in context (see Limits).

## Assumptions in extrapolating to the sequence model (the key section)

The experiment trains on **independent single transactions**; the reference model is a **sequence model**.
The "PLE-on-amount buys nothing for your model" reading rests on:

1. **GBDT/MLP proxy for the per-step encoder.** A sequence model projects each step's features
   (amount included) through a linear/MLP layer before the sequence layers. That per-step encoder
   is MLP-like, and PLE gave the MLP no benefit (−0.012). *Plausible, but it is a proxy.*
2. **The marginal amount→fraud curve is the signal of interest.** This is the weak assumption: a
   sequence model's amount edge is most likely **amount-in-context** (deviation from the card's
   recent spend, ratios, velocity). PLE-of-raw-amount encodes none of that and the experiment never
   tested it. So the result argues against PLE-on-*raw*-amount, **not** against amount mattering.
3. **Tabular PR-AUC transfers to the sequential framing.** Different task shape, eval, and feature
   availability; direction plausible, magnitude not guaranteed.
4. **Scale/capacity.** Small MLP, n_bins=24, 150k rows, no identity features. More capacity usually
   reduces the value of hand-engineered encodings (supports the recommendation) but is untested.

## The four reference-model constraint areas

1. **Retraining dynamics.** PLE adds a per-feature quantile-edge artifact to refit on drift. For
   amount this is pure cost (no benefit). For the count features the placebo surfaced (C1), edges
   would also need periodic refit — a real cost to weigh only if that lift replicates.
2. **Update latency.** PLE is cheap at inference (a bin lookup), but adding it per step for amount
   adds preprocessing with no measured payoff. Skip.
3. **Operational complexity.** Each PLE'd feature = a versioned edge artifact + out-of-range
   monitoring. The recommendation (no PLE on amount) avoids this entirely.
4. **Failure modes.** Drift saturation of top PLE bins was negligible here (test 4.3% ≈ train
   4.2%), but since PLE-on-amount gives no benefit, that machinery is all downside.

## Limits (what this does NOT establish)

- It does **not** show amount is unimportant — only that *PLE of the raw amount scalar* doesn't help
  single-transaction tabular models.
- It does **not** test amount-in-context, the likely real signal for a sequence model.
- It does **not** run an actual sequence-model arm.

## Deployment

No change to amount handling (keep raw log-scaled). The genuinely actionable lead is the placebo:
PLE on heavy-tailed count features (C1: +0.144 for the linear model) — but that is a new
investigation, and a sequence model may already capture much of it.

## Recommended next cycle (the high-fidelity test)

Build per-card sequences from IEEE-CIS (group by a card pseudo-key, order by `TransactionDT`); a
small GRU/Transformer over per-step features; compare per-step amount encodings: **raw vs PLE vs an
amount-deviation feature** (amount relative to the card's running mean). That directly tests the
reference-model setup and resolves assumption #2 — the one this cycle could not.
