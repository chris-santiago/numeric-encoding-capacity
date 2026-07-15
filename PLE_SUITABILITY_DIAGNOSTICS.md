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
by the gate nonlinearities, so PLE only pays its deficit. PLE is a lever **only for SHARP, localized
non-monotone / band-selective** per-step risk. Non-monotonicity is *necessary but not sufficient*: a broad,
smooth inverted-U the gates can already approximate is **not** a lever (probe: a std-normalized band gave
`ple−log` = +0.275 when sharp (σ=0.10) but +0.005 ns when broad (σ=1.00), despite both being strongly
non-monotone at bin resolution). What the affine read cannot form is a *crisp* per-step detector; a gentle
hump it approximates.

**C1. Screen: empirical risk-vs-value shape (at quantile-bin resolution).** Bin the feature **by
percentile** (quantile bins — resolution follows data density, finest where the mass is) and plot empirical
fraud rate per bin. Fit an **isotonic (monotone) regression** of binned fraud-rate on value; the
non-monotone fraction is `1 − R²_isotonic`.
- *Sharp non-monotone (PLE candidate):* an interior extremum concentrated over a *few adjacent bins*
  (band/spike); large non-monotone fraction. This is where the affine bottleneck bites hardest.
- *Broad/smooth non-monotone → likely keep `log`:* high non-monotone fraction but spread across the whole
  range — the gates approximate it; confirm with C2 before spending on PLE.
- *Monotone (linear OR curved) → keep `log`:* non-monotone fraction ≈ 0. **A curved-but-monotone feature
  fails this gate** — the Cycle 8 finding. (When bins look monotone, PLE gave no benefit: `ple−log` = −0.017.)
- *Caveat:* C1 is a screen and can **false-positive on broad non-monotonicity**. The definitive check is C2.

**C2. Free-nonlinearity probe — `dense`/per-feature `mlp` arm (decisive, the reliable gate).** Add a per-step
learned embedding (`Linear→ReLU`) of the feature and compare its `− log`. It is a **superset** of PLE's
per-step expressivity, learned and adaptive, so it detects whether *any* per-step transform helps.
- *`mlp − log` ≈ 0 (CI includes 0):* no per-step transform helps → the GRU already handles this shape →
  **PLE won't help either. Keep `log`.** (Cycle 8 monotone-curved: +0.004; broad band σ=1.00: +0.010 ns.)
- *`mlp − log` CI-clear > 0:* a genuine per-step lever exists → **PLE is a candidate.** Proceed to Gate D.

> C2 is the single most architecture-honest test and, empirically, the **reliable** one: across a full
> non-monotonicity-sharpness sweep, `mlp−log` tracked `ple−log` at every point (both large when sharp, both
> null when broad), whereas the C1 shape screen false-positived on the broad band. When C1 and C2 disagree,
> trust C2. It cannot be faked by a rank metric or a contaminated reference.

**C3. Which encoder, once C2 fires — fixed PLE or a learned per-feature embed (same dim)?** Both are
piecewise-linear embeddings of the scalar; PLE fixes knots at quantiles, the `mlp` learns them. Probe
result (same `d`, concatenated with the untouched other features, affine GRU):
- *Sharp/localized non-monotonicity:* **fixed PLE wins** (`ple−mlp` = +0.028, CI-clear) — quantile knots
  land dense resolution exactly where the band sits; the learned embed must discover it via SGD.
- *Broad non-monotonicity:* **tie** (`ple−mlp` ≈ 0).
- *Monotone / unsure:* **learned embed wins/safer** (`ple−mlp` = −0.038 on monotone-curved; the `mlp`
  degrades gracefully to a ~0 floor while PLE pays its deficit). Because it is per-single-feature, it does
  **not** flip the rest of the numeric path out of the affine-read regime.

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
