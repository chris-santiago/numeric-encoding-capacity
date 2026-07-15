# Hypothesis — Cycle 7: Non-monotonicity vs value-curvature (mechanism disambiguation)

**Status:** pre-registered design (brainstormed 2026-07-14). Runs via the ml-lab protocol.
**Vehicle:** static two-head grid (Approach A). Synthetic, single feature. No sequence model.

## Context

Cycles 5–6 established that a richer encoding (`ple`, `dense`) beats a plain `log` transform in an
affine-read model but not in a free-nonlinearity model — the capacity account. But both cycles built
their synthetic signals to be **non-monotone** (a U-shape in cycle 5, a band conjunction in cycle 6),
so they cannot separate two explanations that were bundled together:

- **H_mono** — PLE helps when the risk-vs-feature relationship is **non-monotone** (`log`'s single
  monotone shape can't bend to a U).
- **H_curv** — PLE helps when the relationship is **nonlinear in the value axis in a way `log` does not
  linearize** (log-mismatch). Non-monotonicity is then just *one* sufficient source of curvature, not a
  necessary condition.

The one real-data result that separates them was never carried into the synthesis: in cycle 2
(`ple_fraud_txn_amount`), a placebo PLE on the count feature `C1` — **monotone by rank but sharply
curved in value** — lifted **+0.144 [+0.126, +0.161]**, while PLE on the weakly-non-monotone amount
did not help. That points at H_curv. This cycle tests it in a controlled synthetic setting.

## Claim (falsifiable)

In an affine-read model, the `ple`-over-`log` benefit is driven by **log-mismatched value-curvature,
not non-monotonicity specifically.** Concretely: `ple` beats `log` for a monotone-but-log-mismatched
signal, so non-monotonicity is **not necessary** for the benefit.

## The decisive comparison

`ple − log` on the **linear (affine-read) head** in the **monotone, log-mismatched** cell (S2):

- CI excludes zero **positive** → curvature is sufficient; non-monotonicity is not necessary (H_curv).
- Ties within ±0.005 in S2 but wins in S3 → non-monotonicity is the real lever (H_mono).

## Design

### Signals (3 shapes) — single feature `x = exp(latent)`, `latent ~ Normal(3.0, 1.6)`, heavy-tailed

| Shape | Risk (logit ∝ …) | `log` linearizes? | Monotone? | Role |
|-------|------------------|-------------------|-----------|------|
| **S1** log-fit | `std(log1p x)` | yes | yes | negative control (PLE should tie `log`) |
| **S2** log-mismatch | `std(rank(x))` — ≈ `Φ(latent)`, an S-curve in log-space | **no** | yes (Spearman ρ≈1) | **discriminator** |
| **S3** non-monotone | `std(log x)²` (U-shape) | no | no | cycle-6-like positive |

S2 is engineered so `log` and `ple` pull apart *by construction*: `rank(x)` is a monotone-but-non-log
function of value (the normal CDF of the latent), so `log` (linear in `latent`) structurally cannot
linearize it, while `ple` (quantile bins ≈ rank) can. One mild Gaussian co-feature `z` as in cycle 5;
intercept calibrated to ~8–9% positive rate.

### Models (2 heads)
- **linear** — affine read of `e(x)` then sigmoid (a strict affine-read: no capacity to synthesize
  nonlinearity, unlike a GRU recurrence).
- **mlp** — one ReLU hidden layer (free per-feature nonlinearity).

### Encodings (4)
`raw` (standardized), `log` (standardized `log1p`), `ple` (16 quantile bins, train-fit,
clip-interpolated), `dense` (learned `Linear→ReLU` expansion). Scalers and PLE edges fit on train only.

### Grid
3 shapes × 2 heads × 4 encodings, fresh data per seed, ≥8 seeds (S2 is the load-bearing cell; extra
seeds tighten its CI). ~9k train / 9k test rows per cell (cycle-5 scale).

## Pre-registered predictions (the double-dissociation that confirms H_curv)

- **linear head:** `ple` > `log` in **S2 and S3** (CI-excludes-0, Holm); `ple` ≈ `log` in S1;
  `raw` < `log` throughout (conditioning).
- **mlp head:** `ple` ≈ `log` in **all three** shapes (it rebuilds any shape itself).

The effect (`ple`>`log`) appears *only* for the affine-read head *and only* on log-mismatched shapes —
a pattern hard to produce by accident.

## Controls / gates (ml-lab standard)

- **Precondition:** oracle logreg on the true generative score reaches PR-AUC ≫ base rate in every
  shape, else that cell is void.
- **Positive control:** on the linear head, `ple`/`dense − raw` is large in S2/S3 (a basis helps a weak
  affine model) → confirms the apparatus has power.
- **Negative control:** S1 (log-fit) shows no encoding beating `log` for either head; and the entire
  mlp head shows `ple` ≈ `log` — if either fails, the design is not isolating what it claims.

## Metric & statistics

PR-AUC primary. **Seed-level paired-t 95% CIs + Holm** step-down over the decision family;
**±0.005 PR-AUC** equivalence margin for declaring a tie (positive evidence of no effect, not mere
failure to reject). Within-seed bootstrap deliberately not used for verdicts (measures eval noise, not
run-to-run variance).

## Verdict mapping

| Observed on linear head | Conclusion |
|-------------------------|------------|
| `ple`>`log` in S2 **and** S3, ties S1; mlp ties all | **H_curv** — value-curvature is the lever; non-monotonicity not necessary |
| `ple`>`log` in S3 only; ties S2 | **H_mono** — non-monotonicity is the lever |
| `ple`>`log` in S1 too, or mlp shows `ple`>`log` | apparatus not isolating the mechanism — void, redesign |

## What it resolves

Updates the synthesis capacity account from "encoding substitutes for **non-monotonicity**" (narrow) to
"…for **any log-mismatched value-shape**" (general), or refutes that and keeps it narrow. Confirms or
rejects the cycle-2 `C1` discovery in a controlled setting.

## Feeds Cycle 8 (real-data v2)

If H_curv holds, Cycle 8 targets `ple` at real **curved-in-value count/recency** features (e.g. the
account dataset's `normMerchantName-accountNumber60dCount`, `transactionToAvailable`), not amount —
S2 is the synthetic analog of those features. If H_mono holds, Cycle 8 keeps the focus on non-monotone
in-context signals.

## Falsification levers

- S1 negative control fails (PLE beats log on a log-fit feature) → PLE arm is rigged / mis-scaled.
- mlp head shows PLE>log anywhere → the "free-nonlinearity rebuilds the shape" premise is wrong, and the
  whole capacity account (not just this cycle) needs revisiting.
- S2 precondition fails (oracle ≈ base) → the log-mismatched signal wasn't actually learnable; void.

---

## Hypothesis — Cycle 7 v2 (multivariate reformulation)

**Why revised (Step-2 PoC finding).** The single-feature v1 design cannot test H_curv. PR-AUC
(average precision) depends only on score *ranking*, and a linear (affine-read) head on any
**monotone** encoding of a single feature yields scores monotone in that feature — so `raw`, `log`,
and `ple` produce *identical rankings* for a monotone risk. The v1 PoC confirmed this: S2 gave
`ple − log = −0.000` (exact tie), while only the non-monotone S3 opened a gap (`+0.343`).
**Value-curvature is invisible at the single-feature margin** — it is a *multivariate* phenomenon:
a monotone-but-curved feature's shape matters only through how it **combines additively** with other
features. This is why the cycle-2 `C1` lift appeared in a multivariate fraud model, not in isolation.

**Revised claim.** In an affine-read model, `ple` beats `log` for a **sum of monotone-but-log-mismatched
features**, so value-curvature is a lever via feature *combination* — with **no non-monotonicity
present**. Non-monotonicity remains a separate, single-feature-sufficient lever.

**Design change.** Each shape becomes a **K-feature additive** signal, `logit = Σ_k f_k(x_k)`,
`x_k = exp(latent_k)` i.i.d.:

| Shape | per-feature `f_k` | monotone? | `log`-fit? |
|-------|-------------------|-----------|-----------|
| S1 | `std(log1p x_k)` | yes | yes (control) |
| S2 | `std(rank x_k)` ≈ `Φ(latent_k)` | yes | **no — discriminator** |
| S3 | `std((std log x_k)²)` | no (U) | no (control) |

K ≈ 6, per-feature coefficient calibrated so the task is learnable (oracle AP ≫ base) at ~8.5% base
rate. All S2 features are monotone in their own value, so any `ple`>`log` gap there is curvature via
combination, not non-monotonicity.

**Decisive test (unchanged in spirit).** `ple − log` on the **linear head** in the multivariate S2
cell; CI excludes zero positive → curvature-via-combination is a lever, non-monotonicity not necessary.

**Predicted double-dissociation.** linear head: `ple`>`log` in S2 **and** S3, ties S1. mlp head: ties
all three. Everything else (2 heads, 4 encodings incl. `dense` at Step 6, seed-level paired-t + Holm,
±0.005 equivalence, precondition/positive/negative controls) carries over from v1.
