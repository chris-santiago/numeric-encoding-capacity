## Hypothesis — Cycle 1 (fraud extension of the PLE-numeric investigation)

**Context.** Extends the PLE-vs-raw-numeric finding (`experiments/ple_vs_raw_numeric/`):
PLE helped on a *synthetic, additive, non-monotonic* target, and the benefit was model-agnostic
target linearization (a linear model on PLE was the best arm). That result's stated open question
was whether the advantage survives **feature interactions** and **real, imbalanced** data. This
cycle tests transaction-amount PLE in a real-time fraud-detection setting.

**Claim:** Encoding the **transaction-amount** feature with PLE yields higher **PR-AUC**
(average precision) than using raw (log-scaled, standardized) transaction amount, on a
fraud-detection task evaluated under a **temporal** train→test split — *when fraud risk is
non-monotonic in amount*.

**Mechanism:** Fraud probability is non-monotonic in transaction amount: elevated at very small
"card-testing" amounts AND at large amounts, low through the typical mid-range (a U-shape or
multi-modal curve). PLE bins let a model assign independent risk to each amount region; a raw
scalar forces a monotone-friendly representation that a linear or shallow model cannot bend into a
U-shape without spending capacity.

**Signal:** A U-shaped / multi-modal fraud-rate-vs-amount curve in the data.

**Expected observable:**
- PLE-amount beats raw-amount in PR-AUC with non-overlapping bootstrap 95% CIs under a temporal
  (past→future) split.
- The advantage shrinks or vanishes on a control where fraud is *monotonic* in amount
  (falsification lever).
- **Carried-forward open question:** the advantage may be *absorbed* by a model that captures
  amount×other-feature interactions natively (GBDT). If `GBDT(raw amount + other features)`
  matches `GBDT(PLE amount + other features)`, PLE's value is redundant once interactions are
  modeled — the cycle-1 "linear-on-PLE wins" story would NOT transfer to real fraud.

## Evaluation Metrics

**Primary:** PR-AUC / average precision — fraud is heavily imbalanced (~0.1–3.5% positive);
ROC-AUC is misleadingly optimistic under imbalance. Reported with bootstrap 95% CIs
(N=1,000, percentile).

**Secondary:** precision@k (the real-time review-budget framing) and recall at a fixed false-
positive rate. ROC-AUC reported as an auxiliary for comparability with cycle 1.

**Domain:** fraud_ple_amount

## Model scope (Step 6 — required arms)

The reference model is a **sequence model** (per-transaction history), so transaction
amount is a per-timestep input and PLE would be its per-step encoding. The Step 6 arm matrix
must therefore include **neural-network arms** — `mlp_raw` and `mlp_ple` — alongside the linear
and GBDT references. The MLP is the in-scope neural proxy for "does PLE-encoding the amount input
help a neural net?"; a **sequence-model arm** (RNN/Transformer over per-card transaction
sequences, with amount PLE-encoded per step) is the highest-fidelity follow-up and the closest
analog to the reference model. Whether a sequence arm runs this cycle depends on how cleanly IEEE-CIS
transactions group into per-card sequences; if not, it is flagged as the next cycle.

Required Step 6 arms: `logreg_raw` (baseline), `logreg_ple`, `mlp_raw`, `mlp_ple`,
`hgb_raw`, `hgb_ple` — each on raw vs PLE amount; sequence arm if feasible.

## Data plan

- **Step 1 PoC (now):** synthetic fraud-amount surrogate with controlled, injected non-monotonic
  fraud-vs-amount structure, heavy class imbalance, a temporal-drift component, and an
  amount×category interaction (to probe the interaction open question). No external data needed.
- **Step 6 experiment:** real IEEE-CIS Fraud data (`TransactionAmt`, joined transaction+identity),
  temporal split inside the labeled `train_transaction.csv` sorted by `TransactionDT`. Acquired
  via Kaggle (requires account + competition-rule acceptance).
