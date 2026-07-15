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

## Outcome — Cycle 8 (SUPPORTED, corrected — curvature IS a GRU lever, deficit-masked)

**Review verdict: `critique_wins`** (4 FATAL / 2 MATERIAL remediated). **Empirical verdict, after a
positive-control audit that retracted then corrected an initial wrong conclusion: the claim is SUPPORTED —
monotone curvature is a per-step PLE lever in the affine GRU, but masked in raw terms by a large PLE
dimensionality deficit.**

> The hardened single-feature PoC (`curvature_seq_poc2.py`) initially read this as REFUTED. That was
> **wrong**: a single curved feature is invisible under a rank metric (Cycle 7's multivariate requirement),
> so it measured only PLE's deficit. Retracted. See the correction notices in `CONCLUSIONS.md`.

- **Multivariate reproduction (`multivariate_control.py`, K=6):** positive controls FIRE (static curved
  deficit-corrected +0.015 [+0.003,+0.027]; static non-monotone +0.373) → the estimand has power. **GRU
  curved deficit-corrected +0.143 [+0.068,+0.218], CI-clear** → curvature is a genuine GRU lever; Cycle 7's
  mechanism transfers.
- **Deficit reality:** in the GRU, PLE costs ~−0.135 for K=6×12 bins (vs ~−0.03 static), so **raw**
  `ple−log ≈ 0`; the lever is real but masked. Deployment needs selective targeting + few bins.
- **Precondition: PASS** (GBM+EWMA baseline, margin +0.056; order-shuffle drop +0.236; both CI-clear).
- **Open:** the GRU smooth-non-monotone cell is anomalous/ns (+0.035) — likely smooth-vs-sharp (Cycle 6's
  sharp-band lever stands); needs a sharp-band multivariate rerun.

See `CONCLUSIONS.md` (corrected verdict + graded static-vs-recurrent × shape table) and `REPORT_ADDENDUM.md`
(corrected deployment rule + the deficit-vs-(K,bins) and real-data follow-ups).
