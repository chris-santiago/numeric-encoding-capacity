## Hypothesis — Cycle 1 (sequence-model extension; macro-iteration Outcome C from ple_fraud_txn_amount)

**Context.** The tabular cycle (`experiments/ple_fraud_txn_amount/`) refuted PLE-on-raw-amount for
single-transaction models and surfaced the real open question (its addendum): the reference model's
amount signal is plausibly **amount-in-context** (relative to a card's
history), which PLE-of-raw-amount does not encode and the tabular experiment never tested. This
cycle tests that directly.

**Claim:** In a sequence model over per-card transaction history, the fraud signal in transaction
amount is carried primarily by **amount-in-context** (per-step amount relative to the card's recent
history), not by the raw per-step amount level. Concretely:
  (a) an explicit **amount-deviation** feature (amount vs the card's running mean/scale) improves
      fraud PR-AUC over feeding raw per-step amount; and
  (b) per-step **PLE of raw amount** adds little beyond what the sequence layers already learn from
      the raw amount sequence (consistent with the tabular refutation).

**Mechanism:** Card fraud manifests as amount that is anomalous *for that card* — a sudden large
charge, or a burst of small "test" charges — a cross-time pattern. A sequence model can learn this
from the amount sequence, but an explicit deviation feature makes it cheaper to extract. Raw-amount
PLE is a within-step, context-free 1-D transform, so it does not encode the cross-time signal.

**Signal:** Within-card amount deviation / velocity (amount relative to the card's own history).

**Expected observable:**
- `seq + deviation feature` beats `seq + raw amount` in PR-AUC with non-overlapping bootstrap CIs.
- `seq + PLE(amount)` is within CI of `seq + raw amount` (PLE-of-raw adds little).
- The sequence model itself beats the best single-transaction tabular model (otherwise the
  sequence framing adds nothing and the comparison is moot — a precondition).
- Falsification lever: on a **shuffled-history control** (sequences randomly permuted within card so
  cross-time context is destroyed), the deviation feature's advantage collapses.

## Evaluation Metrics

**Primary:** PR-AUC (average precision), bootstrap 95% CIs (N=1,000). Imbalanced fraud.
**Secondary:** precision@ fixed budgets (top 0.5%, 1%), recall@1%FPR.
**Domain:** ple_fraud_sequence

## Data plan

- **Step 1 PoC (now):** synthetic per-card sequence surrogate where fraud depends on amount-in-context
  (anomalous-vs-card-history), with a tiny GRU. No external data.
- **Step 6:** real account-sequence fraud dataset at `~/Dropbox/GitHub/demo/tmp/data/` (chosen over
  IEEE-CIS, which lacks a clean user ID). Clean `accountNumber` key, `transactionDateTime` ordering,
  `transactionAmount` per-step amount, `isFraud` label; ~7% fraud; pre-split temporally
  (`train.parq` Jan–Aug 2016 / `valid.parq` Sep–Oct / `test.parq` Nov–Dec), accounts span splits so
  test transactions carry real prior history. Build **causal per-account sequences** (each target
  transaction sees its account's prior transactions, including pre-split history — past-only, no
  leakage). Deviation computed from the account's running history.
  **Leakage exclusion:** drop the target-derived `*FraudTrend` columns (`accountNumber5dFraudTrend`,
  `normMerchantName60dFraudTrend`, etc.); keep amount, balance/limit context, merchant/pos, and
  `*Count` features. The hypothesis is about amount encoding, not label-trend leakage.

## Model scope

Sequence model (GRU as the in-scope proxy for the reference model). Per-step amount
encodings compared: raw, PLE(raw amount), amount-deviation. Optional: deviation+PLE. A
single-transaction tabular baseline is included to verify the sequence model earns its place.
