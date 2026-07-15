# When is PLE worth it over `log`/`raw` for an affine-input GRU?

A per-feature decision protocol distilled from the numeric-encoding-capacity line (cycles 1–8). The
question it answers: **for a given per-step numeric feature entering an affine-read sequence model, will a
piecewise-linear encoding (PLE) beat a `log` scalar (or `raw`)?**

The one-line law the whole line converges on:

> **A fixed numeric basis (PLE) beats `log` iff (1) the model has no free per-step nonlinearity to rebuild
> the shape, (2) the feature set's per-step risk carries a lever the affine read can't form — curvature
> *or* non-monotonicity, visible only *multivariate* — and (3) that lever, summed over encoded features,
> exceeds PLE's dimensionality deficit, which in a GRU is large (~−0.13 for 6 features × 12 bins).** An
> affine GRU satisfies (1) structurally; (2) and (3) must be checked, and PLE's deficit is often the binding
> constraint.

Run the gates in order. Each is cheaper than the next, and a failure at any gate means **keep `log`** — do
not proceed. Most features stop at Gate C; the deficit (Gate D) is what most often kills an otherwise-real
lever.

---

## Gate A — Architecture precondition (static check, once per model)

**A1. Is the per-step numeric path truly affine?** Confirm numerics enter the recurrence as `W·x_t` with
**no per-step Dense/MLP/ReLU** before the RNN cell.
- *Pass:* no per-step nonlinear projection → a basis can be decisive.
- *Fail (a per-step MLP exists):* **STOP for all features.** The model rebuilds any 1-D transform itself;
  PLE is redundant (Cycle 5: PLE ties `log` in every free-nonlinearity model). If a per-step Dense is
  ever added to production, this whole protocol becomes moot.

---

## Gate B — Instrument preconditions (does the encoding question even apply?)

These fix the exact failure that made Cycle 3 uninterpretable. Run once for the model; B3 is per feature.

**B1. The sequence model beats a strong tabular baseline.** Train the GRU (`log` arm) and a **strong**
tabular model — GBM on **recency/EWMA-weighted** aggregates matching the account's decay, not just
mean/std/last. A recency-agnostic baseline passes the gate too cheaply.
- *Pass:* `AP(GRU) − AP(tabular)` seed-level CI-clear > 0.
- *Fail:* the sequence model adds nothing here → per-step encoding of history is moot.

**B2. The model uses temporal order.** Shuffle the prior steps of the **test** sequences, re-score with the
**same** trained model.
- *Pass:* PR-AUC drop CI-clear > 0. (Cycle 8 synthetic: +0.236.)
- *Fail (~0 drop):* order carries no signal → nothing for per-step history encoding to exploit.

**B3. The feature is informative.** An encoding cannot lift a no-signal feature (Cycle 4's vacuous null).
- *Pass:* feature-only fraud signal CI-clear above base — production **SHAP importance** is the best
  proxy; a feature-alone AP or mutual-information screen also works.
- *Fail:* skip the feature. (Cycle 4 lesson: this is a HARD GATE, not a footnote.)

---

## Gate C — Is there a per-step lever, and does it clear PLE's deficit? (Cycle 7 + 8, corrected)

> **Correction.** An earlier version of this gate said "in a GRU, monotone curvature is absorbed by the
> gates → keep `log`." That was **wrong** (it came from a single-feature experiment where curvature is
> invisible under a rank metric). The corrected finding: **both curvature and non-monotonicity ARE per-step
> levers in an affine GRU** — but the lever only appears in a **multivariate** design and only survives
> **net of PLE's dimensionality deficit**, which in a GRU is large.

**Two things must both be true for PLE to beat `log` in an affine GRU:** (1) the feature set carries a
per-step lever the affine read can't form on its own — *multivariate*, and (2) that lever, summed over the
encoded features, **exceeds the encoder's dimensionality deficit**. Measured wrong, either can hide the
other — the trap that made Cycle 8's first pass conclude the opposite of the truth.

**Which shapes are levers (an affine GRU's gates are smooth approximators, so only *localized* structure is
a lever):** sharp/localized non-monotone (**strongest**, deficit-corrected ~+0.45) ≫ monotone curvature
(**moderate**, ~+0.14) ≫ smooth non-monotone (**absorbed by the gates**, ~0) ≈ log-linear (none). Target
sharp or sharply-curved features; do not spend encoding on smooth non-monotone (a broad U/inverted-U) — the
gates already handle it.

**The deficit is usually the binding constraint, and it confines the raw net-win to a narrow ridge.** For
fixed PLE, raw `ple−log > 0` held at only one config in a K×bins sweep (K≈4, bins≈8, +0.047); more bins or
features push it negative. A **learned per-feature embed** pays a smaller deficit and so wins raw where PLE
does not (C3) — prefer it.

**C1. Screen: empirical risk-vs-value shape (necessary, not sufficient).** Bin the feature by percentile,
plot empirical fraud rate per bin, fit an isotonic regression; the non-monotone fraction is `1 − R²_iso`.
A curved-monotone or non-monotone shape is a *candidate*; a purely log-linear shape is not. **But shape
alone does not decide** — curvature is invisible in isolation and only surfaces multivariate (C2), and the
benefit still has to clear the deficit (D). Treat C1 as a cheap filter that only *rejects* log-linear
features.

**C2. Multivariate positive-controlled probe (decisive).** The single reliable test, and the one that must
not be skipped after Cycle 8:
- Build a **multivariate** synthetic mirror of the candidate feature set (K features, additive risk) — a
  single feature cannot express curvature under a rank metric.
- Run the **deficit-aware** estimand `Δ = (ple−log)_target − (ple−log)_log-adequate` (Gate D) in the target
  architecture.
- **Include a static-head positive control on the same features.** If `Δ_static` does not fire CI-positive,
  your instrument is blind and *no* conclusion (positive or null) is trustworthy — halt and fix the design
  before reading the GRU. (This is exactly the check whose absence made Cycle 8's first verdict wrong;
  when added, static curved fired +0.015 and the GRU lever appeared at +0.143.)
- A free-nonlinearity `mlp`/`dense` arm remains a useful corroborator ("does *any* per-step transform help"),
  but it too must be run multivariate — single-feature `dense`/`mlp` numbers are in the invisible regime.

**C3. Fixed PLE vs a learned per-feature embed — RESOLVED: prefer the learned embed.** A multivariate
(K=6), shared-coordinate, 8-seed, Holm-corrected rerun (`fixed_vs_learned.py`) settles it: on monotone
curvature the **learned `Linear(1→d)→ReLU` embed beats fixed PLE by +0.094 raw (Holm-sig)**. Both unlock
nearly identical *lever* (deficit-corrected +0.137 vs +0.148); the gap is PLE paying a **larger
dimensionality deficit** at equal width — its `d` fixed ramps all fire and load the recurrence, while the
learned embed shapes its `d` outputs to carry the same signal at lower effective cost. Since the deficit is
the binding deployment constraint (Gate D), the encoder that pays less deficit wins. **Default to the
learned per-feature embed** (same lever, lower deficit, graceful ~0 floor on shapes with no lever). Scope:
established for monotone curvature; for *sharp non-monotone* structure PLE's dense-quantile-knots-at-the-band
could still compete — untested, so verify per feature-type before assuming the learned embed dominates there.

> This reverses an earlier single-feature claim that "PLE wins on sharp." That comparison was confounded
> (different coordinates: PLE on raw, embed on log; band pinned at the density mode; n=5, no multiplicity).
> Fixed once → the ordering flips.

---

## Gate D — Deficit-aware confirmation A/B (only if C passes)

PLE pays a structural ~−0.03/−0.04 deficit wherever `log` is adequate (Cycle 7). Measure the benefit **net
of that deficit**, or you will understate it and wrongly reject.

**D1. Deficit-aware estimand.** Do **not** read raw `ple − log`. Use a difference-of-differences against a
**log-adequate, marginal-matched reference feature**: `Δ = (ple − log)_target − (ple − log)_reference`. The
reference must be exactly log-adequate *and* share the target's marginal shape (Cycle 8 F4 — `amount` was a
contaminated reference: weakly curved AND marginal-mismatched).

**D2. Magnitude-sensitive metric alongside PR-AUC.** Report **log-loss / Brier** too. PR-AUC is
rank-invariant, so it can hide a representational change; if PR-AUC and Brier disagree, trust the pair, not
PR-AUC alone (Cycle 8 F2).

**D3. Oracle ceiling + static-head positive control.** Compute an oracle (rank by true risk if synthetic),
and — the non-negotiable check from Cycle 8 — run the identical estimand on a **static affine-read head**
where the lever is *known* to exist. If the static positive control does not fire CI-positive, the
instrument is blind and neither a positive nor a null GRU result is trustworthy. (An oracle gap alone does
**not** license "not encoding-addressable" — that inference was the retracted error; only a firing positive
control distinguishes "no lever" from "blind estimand.")

**D4. Adoption bar.** Seed-level paired-t **95% CI** on the deficit-corrected `Δ` (+ Holm across a feature
family). Adopt PLE only if the **CI excludes 0 positive**. Within ±0.005 = tie → keep `log` (simpler, no
moving parts).

---

## Gate E — Training-hygiene guards (against a false negative)

PLE-in-GRU is training-sensitive; an under-resourced run shows a **spurious negative** lift (the un-hardened
Cycle 8 PoC's −0.19 was pure undertraining, not representation).

- **E1. Optimize like production:** minibatch + per-epoch reshuffle, adequate epochs, held-out validation
  with early-stopping + **best-state restore**, sufficient hidden width. (Cycle 6/8 regime.)
- **E2. Few bins:** ~8–12; more adds variance and serving width with no benefit. Fit bin edges on **train
  only**; refit on drift (`log` has no moving part — count this as an ongoing cost).
- **E3. Sanity-check the deficit:** run PLE on a known-log-adequate control; you should recover a stable
  deficit — **but expect it to be large in a GRU** (~−0.13 for 6 features × 12 bins, vs ~−0.03 in a static
  head). This deficit is usually the binding constraint on net deployment value, not the lever; measure it
  explicitly per (K, bins) rather than assuming the small static figure.

---

## Decision flow (summary)

```
A1 affine read? ───no──► keep log (PLE redundant; Cycle 5)
   │ yes
B1 GRU beats strong tabular? ─no─► encoding moot
B2 uses temporal order?      ─no─► encoding moot
B3 feature informative (SHAP)?─no─► skip feature (Cycle 4)
   │ all yes
C1 risk-vs-value log-linear? ──yes──► keep log (no lever to find)
   │ curved or non-monotone
C2 MULTIVARIATE deficit-corrected Δ, static positive control fires?
   │ yes (lever real + instrument valid)      │ no / control blind ─► keep log / fix design
D  does the lever CLEAR PLE's deficit? (Δ CI-excludes-0 net of ~−0.13 GRU cost, Brier agrees)
   │ yes                          │ no / tie  ◄── the deficit usually binds here
   ▼                              ▼
adopt PLE (few features, few bins;   keep log
verify prod A/B)  (guard E1–E3)
```

## What this predicts for the production features

- **amount** — its risk is likely close to log-adequate (null across cycles), so even if a small curvature
  lever exists it is unlikely to clear PLE's GRU deficit → **probably keep `log`** (confirm via C2/D, not by
  assuming "monotone ⇒ no lever," which was the retracted error).
- **Δt (inter-transaction time)** — plausibly non-monotone (short = card-testing, long =
  dormant-reactivation) → **the strongest PLE candidate**; run the full multivariate D confirmation.
- **counts / recency aggregates** — genuine candidates (curvature *is* a lever now), but each must clear the
  deficit; encode only the few strongest at few bins.

Resolved by the Cycle 8 follow-up experiments: the **deficit-vs-(K, bins)** ridge (raw net-win only at
K≈4/bins≈8), the **sharp-vs-smooth** anomaly (GRU absorbs smooth, sharp fires +0.45), and **fixed-vs-learned**
(prefer the learned embed). The only external item left open is **Cycle 6's real-data A/B**, now runnable
with the validated precondition gate and the targeting rule above (few sharp/curved features, low width,
learned per-feature embed).
