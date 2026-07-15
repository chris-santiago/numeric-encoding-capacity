# Real-data A/B (Cycle 6 follow-up) — outcome: precondition gate FAILS; A/B moot on this data

**Goal:** run the encoding A/B (log vs a targeted encoder) on the real account-sequence fraud data, gated
by the precondition Cycle 3 skipped. **Outcome: the precondition gate fails on a *properly-trained* GRU, so
the encoding A/B is not run — the gate correctly halted an experiment the data can't support.**

## Result (`real_precondition.py`)

Data: `data/account-sequences/transactions.parq` (786k txns, 5000 accounts, 1.58% fraud). Causal
per-account sequences, L=128; anchors = all fraud + 45k non-fraud, temporally split; `*FraudTrend` leakage
columns excluded. Baseline: GBM on last/mean/EWMA/max aggregates. **GRU trained properly** (hidden 128, up
to 150 epochs with patience-15 early stopping + LR scheduler, MPS; converged ~38 epochs, best val AP ~0.40).

| metric | value |
|---|---|
| GRU PR-AUC | 0.4034 |
| strong GBM+EWMA tabular | 0.4307 |
| **GRU beats tabular?** | **No (−0.027)** |
| order-shuffle drop | +0.0037 (negligible) |
| **precondition** | **FAIL** |

The FAIL is **robust to training budget** — a properly-trained, converged, adequately-sized GRU (not the
under-powered earlier runs) reaches the same verdict. Under-power is ruled out.

## What it means (precise scope)

- **On this feature set, the task is ~point-in-time, not sequential.** The features are dominated by
  current account state (`transactionAmount`, `transactionToAvailable`, `availableMoney`, `currentBalance`,
  `creditLimit`), all observable at the last step. A GBM on last-step + aggregates matches a GRU; shuffling
  history barely moves the GRU (+0.004) because it uses neither order nor history. So per-step *encoding*
  (which improves the per-step numeric pathway) is moot — the per-step/sequential pathway isn't where the
  signal is.
- **This is expected, and does NOT answer the production question.** The `demo/tmp/data` set is a simulated
  fraud process known to **under-represent time signal**; in the real production data, SHAP shows Δt and
  cyclic time features are highly important. So the gate failing *here* reflects the demo data's known
  limitation, not a property of the production sequence model. (Cycle 4 lesson: validate time/sequence
  questions where the signal demonstrably exists.)
- **It does not contradict the synthetic mechanism findings.** Those established *"if a feature has sharp
  non-monotone per-step risk that aggregates miss, PLE captures it."* This gate tests the antecedent on
  real data and finds it unmet *here* — a statement about this dataset's applicability, not about the
  mechanism.

## Standing

The production encoding question is **not answerable on this public/demo dataset** — it needs data with
genuine sequential fraud signal (the production data, or a dataset where the precondition passes). The
precondition gate is the reusable instrument for deciding that, and it did its job: it prevented an
uninterpretable encoding A/B on unsuitable data.
