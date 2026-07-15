# Sequence-model PLE / amount-in-context (fraud)

**Hypothesis.** In a sequence model over per-card transaction history, the amount fraud signal is
**amount-in-context** (relative to the card's history). (a) An explicit amount-deviation feature
improves PR-AUC over raw per-step amount; (b) per-step PLE of raw amount adds little. Extends the
tabular cycle (`experiments/ple_fraud_txn_amount/`), whose addendum flagged amount-in-context as
untested. Macro-iteration Outcome C reformulation.

## Quickstart
```bash
uv run seq_ple_poc.py
```

## Pipeline (Step 1 PoC — synthetic sequence surrogate)
1. **Data** — `make_sequences`: per-card sequences of L=16 log-amounts; the last step is fraud iff
   its amount is anomalous *for that card* (a large spike or a tiny test charge vs the card's level).
2. **Encodings** (per step) — raw (std log amount); PLE(amount) bins; amount-deviation (causal,
   amount vs the card's running history).
3. **Models** — `tab_last_raw` (logreg on last-step amount only, no sequence); GRU over each
   per-step encoding (`seq_raw`, `seq_ple`, `seq_dev`).
4. **Score** — PR-AUC on held-out sequences (last-step label).

## Representative output (seed 0)
```
  tab_last_raw   0.4247
  seq_raw        0.9639
  seq_ple        0.9503
  seq_dev        0.9613
  seq_raw - tab_last_raw = +0.539   (sequence context is the signal)
  seq_dev - seq_raw      = -0.003   (explicit deviation adds nothing)
  seq_ple - seq_raw      = -0.014   (PLE-of-raw adds nothing)
```
The GRU learns amount-in-context from the raw amount sequence; neither PLE nor an explicit
deviation feature helps.

## Real data (Step 6)
Account-sequence fraud dataset at `~/Dropbox/GitHub/demo/tmp/data/` (chosen over IEEE-CIS, which
lacks a clean user ID). Clean `accountNumber` key, `transactionDateTime` ordering,
`transactionAmount`, `isFraud` (~7% fraud); pre-split temporally (`train/valid/test.parq`), accounts
span splits → causal per-account history available at test time. Drop the target-derived
`*FraudTrend` columns; keep `*Count` velocity features.

## Known limitations / scope exclusions (deferred to Step 6)
- **No CIs / single seed.**
- **Synthetic signal is cleanly learnable** — fraud ≈ extreme amount-vs-card-mean, which the GRU
  extracts perfectly from raw, leaving no headroom for deviation/PLE. Real data (subtler context) is
  the genuine test of whether an explicit deviation feature helps.
- **No shuffled-history falsification lever** (destroy cross-time context → deviation advantage
  should vanish).
- **Strawman tabular baseline** (last amount only; real tabular would carry aggregates).
- **Tiny GRU, no tuning, real IEEE-CIS sequences not yet used.**
