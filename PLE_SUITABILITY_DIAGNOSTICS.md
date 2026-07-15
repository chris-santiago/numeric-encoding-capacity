# When is PLE worth it over `log`/`raw` for an affine-input GRU?

A per-feature decision protocol distilled from the numeric-encoding-capacity line (cycles 1–8). The
question it answers: **for a given per-step numeric feature entering an affine-read sequence model, will a
piecewise-linear encoding (PLE) beat a `log` scalar (or `raw`)?**

The one-line law the whole line converges on:

> **A fixed numeric basis (PLE) helps iff the model has no free per-step nonlinearity to rebuild it, AND
> the feature's per-step risk-in-context is non-monotone.** An affine-input GRU satisfies the first
> clause structurally; the second must be checked per feature.

Run the gates in order. Each is cheaper than the next, and a failure at any gate means **keep `log`** — do
not proceed. Most features stop at Gate C.

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

## Gate C — The core discriminator: non-monotone risk-in-context (Cycle 8)

**This is the gate that decides most cases.** In a GRU, monotone shapes — linear *or* curved — are absorbed
by the gate nonlinearities, so PLE only pays its deficit. PLE is a lever **only for non-monotone /
band-selective** per-step risk.

**C1. Screen: empirical risk-vs-value shape.** Bin the feature by value percentile and plot empirical fraud
rate per bin. Quantify departure-from-monotone: fit an **isotonic (monotone) regression** of fraud-rate on
value and compare its loss to an unconstrained fit.
- *Non-monotone (PLE candidate):* interior extremum (U / inverted-U / band); isotonic fit materially worse
  than unconstrained; |Spearman ρ| near 0 despite clear mutual information.
- *Monotone (linear OR curved) → keep `log`:* isotonic fit ≈ unconstrained; |Spearman ρ| near 1. **A
  curved-but-monotone feature fails this gate** — that is the entire Cycle 8 finding.
- *Caveat:* marginal risk-vs-value is a screen. The definitive, in-context check is C2.

**C2. Free-nonlinearity probe — the `dense` arm (decisive).** Add a per-step `Linear→ReLU` (`dense`) arm and
compare `dense − log`. `dense` is a **superset** of PLE's per-step expressivity, learned and adaptive.
- *`dense − log` ≈ 0 (CI includes 0):* no per-step transform helps → the GRU already handles this feature's
  shape → **PLE won't help either. Keep `log`.** (Cycle 8: `dense−log` = +0.004 on a monotone-curved feature.)
- *`dense − log` CI-clear > 0:* a genuine per-step lever exists the GRU can't reach on its own → **PLE is a
  candidate** (and a cheap fixed basis may beat `dense`; Cycle 6). Proceed to Gate D.

> C2 is the single most architecture-honest test: it asks directly "is there a per-step lever this GRU
> cannot already reach?" It cannot be faked by a rank metric or a contaminated reference.

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

**D3. Oracle ceiling.** Compute an oracle (rank by true risk if synthetic; best-achievable proxy on real
data). Confirms headroom exists and whether arms are near-ceiling — this is what separates "no lever" from
"underpowered instrument" (Cycle 8: log-GRU sat +0.032 below oracle, but nothing closed it → not
encoding-addressable).

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
- **E3. Sanity-check the deficit:** run PLE on a known-monotone control; you should recover the ~−0.03/−0.04
  deficit. A wildly different number means training is off (return to E1), not that the finding changed.

---

## Decision flow (summary)

```
A1 affine read? ───no──► keep log (PLE redundant; Cycle 5)
   │ yes
B1 GRU beats strong tabular? ─no─► encoding moot
B2 uses temporal order?      ─no─► encoding moot
B3 feature informative (SHAP)?─no─► skip feature (Cycle 4)
   │ all yes
C1 risk-vs-value non-monotone? ─monotone─► keep log (incl. curved!) ◄── Cycle 8 core
C2 dense − log CI-clear > 0?    ─no──────► keep log (GRU handles it)
   │ yes
D  deficit-corrected Δ CI-excludes-0, Brier agrees, above ceiling gap?
   │ yes                          │ no / tie
   ▼                              ▼
adopt PLE (verify prod A/B)     keep log
   (guard with E1–E3)
```

## What this predicts for the production features

- **amount** — monotone risk (null across every cycle) → fails Gate C → **keep `log`**.
- **Δt (inter-transaction time)** — plausibly non-monotone (short = card-testing, long =
  dormant-reactivation) → the **one feature likely to clear Gate C**; run the full D confirmation.
- **counts / recency aggregates** — case-by-case on Gate C1/C2; do not assume curvature alone qualifies.

The only experiment this line leaves open is Cycle 6's stated real-data A/B of the **non-monotone** lever —
now runnable with the Gate-B instrument validated in Cycle 8.
