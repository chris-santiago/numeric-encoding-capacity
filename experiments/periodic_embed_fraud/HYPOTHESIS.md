## Hypothesis — Cycle 1 (periodic-embeddings investigation; cycle 4 of the fraud-encoding line)

**Context.** Cycles 1–3 established that PLE-encoding *transaction amount* never improves real fraud
detection (linear, GBDT, MLP, GRU). The reference model is a GRU (seq~300) in which **amount
is the only raw numeric**; the other inputs are already encoded — categoricals via learned
embeddings, date/time via fixed sin/cos, inter-transaction time as log-minutes. The one untested
lever from Gorishniy et al. 2022 ("On Embeddings for Numerical Features in Tabular Deep Learning")
is its **periodic-embedding** family applied with *learned* frequencies, on the **time** features —
not amount.

**Claim (two falsifiable sub-claims):**
- **H1 (cyclic / known period):** Replacing fixed sin/cos with **learned-frequency periodic
  embeddings** on the cyclic time features (hour-of-day, day-of-week — periods are known a priori)
  will **not** produce a CI-separated PR-AUC improvement in the fraud GRU. Fixed sin/cos already
  spans the known fundamental; learning frequencies can only help if the true fraud-vs-time response
  needs harmonics the single fixed frequency omits.
- **H2 (non-periodic / Fourier basis):** Applying learned periodic embeddings to the **non-periodic**
  inter-transaction-time feature (log-minutes) — where they act as a learned Fourier-feature basis —
  will beat the **raw** scalar only **marginally at best** (CI overlapping or barely separated from
  zero), because a GRU is a strong enough model to learn the 1-D transform itself. This is the direct
  prediction of the cross-cycle capacity argument.

**Mechanism.** A periodic embedding expands a scalar `x` into `[sin(2π·cᵢ·x), cos(2π·cᵢ·x)]_{i=1..k}`
→ Linear → ReLU, with the `k` frequencies `cᵢ` trainable (init `N(0, σ²)`; `σ` sets the frequency
spread and is the dominant hyperparameter). This gives the network a rich basis so a linear head can
fit a wiggly/high-frequency 1-D response — the same basis-expansion trick as PLE, but with a smooth
Fourier basis instead of a localized piecewise-linear one. Fixed sin/cos is the special case: one
hand-chosen frequency per known period, not learned. Gain over sin/cos therefore requires missing
harmonics; gain over raw requires the model to be too weak to learn the transform unaided.

**Signal.** Paired PR-AUC lift between GRU arms that differ *only* in the time-feature encoding:
learned-periodic vs fixed-sin/cos (H1) and learned-periodic vs raw (H2), with bootstrap CIs.

**Expected observable.**
- H1 confirmed if `cyc_periodic − cyc_sincos` CI overlaps zero (no harmonic gain); refuted if it is
  CI-separated positive.
- H2 confirmed (marginal) if `dt_periodic − cyc_sincos` CI overlaps zero or is only slightly positive;
  refuted-toward-helpful if it is solidly CI-separated positive (would mean the GRU could *not* learn
  the dt transform unaided — a notable result). `cyc_sincos` is the matched **raw-dt baseline**
  (cyclic block in the reference model's sin/cos form, inter-transaction time left raw), so this contrast
  isolates the dt encoding — raw vs learned-periodic — while holding the cyclic block fixed.
- Any arm must also clear the trivial baseline (non-negotiable).

**Comparison structure (finalized in the experiment plan at Gate 1).** A shared GRU backbone over
per-account causal history, arms differing only in the time-feature encoding block:
`base_raw` (all time raw) · `cyc_sincos` (reference-style: hour/dow sin/cos, dt raw) ·
`cyc_periodic` (hour/dow learned-periodic, dt raw) · `dt_periodic` (hour/dow sin/cos, dt
learned-periodic) · `all_periodic`. Plus a trivial tabular logistic-regression baseline. All inputs
other than the time block (amount raw-log, context numerics, cardPresent) are held fixed across arms.

## Evaluation Metrics

**Primary:** PR-AUC (average precision). Rationale: severe class imbalance (~7% fraud); precision
across the operating range is the relevant quantity for the reference model, consistent with cycles 1–3.
**Secondary (diagnostic, not verdict-bearing):** convergence epochs; learned-frequency spectra (to
show the mechanism — which frequencies the periodic arms actually discover).

**Data.** Real account-sequence fraud data at `~/Dropbox/GitHub/demo/tmp/data` (`accountNumber`,
`transactionDateTime`, `transactionAmount`, `isFraud`, ~7% fraud). Targets from the curated temporal
splits `train/valid/test.parq`; causal per-account history from `transactions.parq`. Cyclic features
derived from `transactionDateTime` (hour-of-day, day-of-week); inter-transaction time computed as
per-account consecutive deltas (log-minutes). **Exclude all `*FraudTrend` columns (target-derived
leakage).** Periodic-embedding frequencies and any scalers are fit on the **train period only**
(causal). PoC (Step 1) is synthetic with controlled ground truth; real data enters at Step 6.

**Domain:** periodic_embed
