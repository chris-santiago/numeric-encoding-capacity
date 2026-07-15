# PLE on transaction amount — fraud-detection extension

**Hypothesis.** PLE-encoding the **transaction-amount** feature beats raw (log) amount in
**PR-AUC** on a fraud task with a **non-monotonic** fraud-vs-amount curve, evaluated under a
**temporal** (past→future) split. Extends `experiments/ple_vs_raw_numeric/`; the open question
carried forward is whether the PLE advantage survives **feature interactions** and real imbalance,
or is absorbed by a GBDT that models interactions natively.

## Quickstart

```bash
uv run fraud_ple_amount_poc.py
```

## Pipeline (Step 1 PoC — synthetic surrogate)

1. **Data** — `make_data`: 40k transactions; heavy-tailed log-normal amount with mild upward
   drift over time; 5 merchant categories; fraud logit = main U-shape in amount + amount×category
   interaction + card-testing spike at tiny amounts + linear generic signal + temporal drift.
2. **Split** — temporal: earliest 70% by time → train, latest 30% → test.
3. **Encode** (fit on train only) — raw: standardized log-amount + one-hot category + generics;
   PLE: 24 quantile bins replace the amount scalar; GBDT: raw log-amount + ordinal category.
4. **Arms** — `logreg_raw` (trivial baseline), `logreg_ple` (hypothesis), `hgb_raw`
   (HistGradientBoosting reference — bins + interactions natively).
5. **Score** — PR-AUC (primary), ROC-AUC, precision@k; visualize the fraud-vs-amount curve.

## Representative output (seed 0)

```
                PR-AUC  ROC-AUC     P@k
  logreg_raw    0.2023   0.6718  0.2222
  logreg_ple    0.7607   0.9103  0.6948
  hgb_raw       0.7769   0.9139  0.7055
```

PLE lifts the linear model by +0.559 PR-AUC; the GBDT edges PLE+linear by only +0.016 (the
interaction-gap proxy). The fraud-vs-amount curve is strongly U-shaped (`fraud_poc_mechanism.png`).

## Real data (Step 6)

IEEE-CIS Fraud at `/Users/chrissantiago/Dropbox/GitHub/ml-lab/data/ieee-fraud-detection/`
(`train_transaction.csv` has `isFraud`; join `train_identity.csv` on `TransactionID`). The
competition test set is unlabeled, so the temporal split happens **inside** `train_transaction.csv`
sorted by `TransactionDT`. `TransactionAmt` is the PLE-candidate feature.

## Known limitations / scope exclusions (deferred to Step 6)

- **No confidence intervals** — single point estimate; bootstrap 95% CIs come at Step 6.
- **No monotonic-amount control** — the falsification lever (PLE must NOT help when fraud is
  monotonic in amount).
- **Synthetic fraud rate ~10%** — higher than real fraud (IEEE ~3.5%); PR-AUC scale is base-rate
  sensitive. Real imbalance enters at Step 6.
- **Imbalance handling asymmetry** — `logreg_*` use `class_weight="balanced"`; `hgb_raw` does not.
  To be harmonized in the Step 6 arm matrix.
- **Partial arm matrix** — the PoC omits the neural-net arms. Step 6 **requires** `mlp_raw` and
  `mlp_ple` (the reference model is a sequence model, so neural nets are in scope; the MLP is the
  neural proxy and a sequence arm is the highest-fidelity follow-up), plus `hgb_ple` and explicit
  amount×category crosses for the linear model — these decide whether PLE's value is redundant
  once interactions are modeled.
- **Single seed; n_bins=24 untuned; synthetic only.**
