# Hypothesis — Cycle 8: value-curvature encoding in the affine-input reference GRU (real data)

**Status:** kickoff design, revised by Cycle 7. Real-data v2 of Cycle 3 (`ple_fraud_sequence`).
Runs via the ml-lab protocol. Data: `data/account-sequences` symlink (~7% fraud, temporal split).

## Context — what the prior cycles settled

- **Cycle 3** tested PLE-on-amount in a *small/short* GRU (hidden 24, L≤32, ≤15 epochs) and failed:
  the GRU lost to a tabular baseline (`seq_raw − tab_aggregate = −0.035/−0.049`) and **ignored temporal
  order** (shuffle ≈ 0). Under-powered, and possibly the signal isn't sequential.
- **Cycle 6** showed encoding helps an *affine-input GRU at production length* (synthetic).
- **Cycle 7** established the *mechanism*: in an affine-read model, **log-mismatched value-curvature is a
  real, general PLE lever** (not basis-alignment, not only non-monotonicity), but (a) it is *multivariate*
  (needs feature combination), (b) it is *curve-dependent* (benefit scales with departure-from-log), and
  (c) PLE pays a **structural ~−0.04 deficit** wherever `log` is adequate, so the benefit is only visible
  net of that deficit. Amount's real curvature is weak (cycle 2), so amount is the wrong target.

## Claim (falsifiable)

In an **affine-input GRU at reference scale** (seq ≈ 300, adequate capacity/epochs), per-step **PLE
encoding of the most sharply curved-in-value count/recency features** improves fraud PR-AUC over the
`log`-scalar baseline **net of PLE's structural deficit** — while **amount** (weakly curved) does not.

## Design

### Precondition gate (fixes Cycle 3's fatal flaw — non-negotiable, runs FIRST)
Before any encoding comparison, establish that a properly-scaled GRU (L≈300, hidden ≥32, tuned,
early-stopped) **(a) beats a strong tabular baseline** and **(b) uses temporal order** (shuffle-prior-
steps test shows a CI-clear drop). If either fails, the reference-class sequence model isn't extracting
sequential signal on this data → the encoding question is moot, and *that is the reported finding*
(real account fraud isn't sequential in the hypothesized way).

### Feature targeting (per Cycle 7)
Rank candidate per-step features by **departure-from-log** (empirical curvature of fraud-rate-vs-value
after a log transform). Target PLE at the **most sharply curved count/recency features** (e.g.
`normMerchantName-accountNumber60dCount`, `transactionToAvailable`), **not amount**. Amount is a
pre-registered **negative control** — it should *not* benefit.

### Deficit-aware estimand (per Cycle 7)
Compare per-step `ple` vs `log`-scalar with a **log-adequate feature as the deficit baseline**, and read
the curvature benefit as the difference-of-differences (target − baseline), so PLE's structural cost is
netted out. Raw `ple − log` will *understate* the benefit by ~the deficit.

### Arms
`log`-scalar (production baseline) · per-step `ple` on curved targets · per-step `dense` (learned
Linear→ReLU, the free-nonlinearity reference — genuinely distinct here, unlike the static Cycle 7) ·
`raw` (floor). All at L∈{32, 300}.

### Stats
PR-AUC; seed-level paired-t 95% CIs + Holm; CI-excludes-zero adoption bar; ±0.005 equivalence. Bootstrap
not used for verdicts.

## Decisive test & verdicts

`ple − log` (deficit-corrected) on the curved-feature target, linear per-step read, at L=300:

| Result | Conclusion |
|--------|------------|
| CI excludes 0 positive, amount negative-control flat, precondition passed | **Curvature encoding transfers to real data** → production A/B justified |
| CI straddles 0 / within ±0.005 | curvature benefit does not survive real feature distributions/correlation → keep `log` |
| Precondition fails even at scale | real account fraud isn't sequential as hypothesized → encoding moot (important negative; supersedes Cycle 3's under-powered version) |
| Amount benefits but curved features don't | reverse of the predicted pattern → mechanism misunderstood, reopen |

## Falsification levers
- Precondition gate (GRU must beat tabular + use order).
- Amount negative control (must NOT benefit).
- Deficit baseline (must net out PLE's structural cost or the benefit is masked — the error that cost
  Cycle 7 two iterations).

## Open dependency on Cycle 7
Whether PLE's structural deficit persists at reference-model scale / large data is an open question
(Cycle 7 § Open questions). If more data shrinks it, blanket PLE becomes safer; if not, targeting stays
essential.

---

## Outcome — Cycle 8 (REFUTED for the GRU)

**Review verdict: `critique_wins`** (4 FATAL / 2 MATERIAL findings; all remediated in a hardened PoC
before any real-data compute). **Empirical verdict: the claim is REFUTED for the sequence model.**

- **Scope tightening (F6, applied):** the decisive test is **single-feature-at-a-time**. Multi-feature /
  correlated co-encoding synergy is deferred to a follow-up, not part of this claim.
- **Deficit reference (F4, revised):** `amount` was replaced by a **marginal-matched, exactly-log-adequate
  reference feature** — `amount` is weakly-curved (Cycle 2) and marginal-mismatched, so it was a
  contaminated deficit baseline.
- **What the hardened PoC found:** the deficit-corrected curvature benefit is ≤ 0 at every curvature level
  (never crosses zero); a free per-step nonlinearity (`dense`) also gains nothing (+0.004); and the small
  oracle gap (+0.032) is closed by no arm. → **No per-step-value lever exists in a GRU**; its gates absorb
  monotone curvature. Cycle 7's static-model lever does not transfer.
- **Precondition: PASS** against a strong GBM+EWMA baseline (margin +0.056 CI-clear; order-shuffle drop
  +0.236 CI-clear) — so this is a genuine refutation, not a moot one.
- **Step 6 (real-data monotone-curvature targeting): not justified, not run.**

See `CONCLUSIONS.md` (unified static-vs-recurrent × monotone-vs-non-monotone theory) and
`REPORT_ADDENDUM.md` (deployment rule + the standing non-monotone real-data A/B).
