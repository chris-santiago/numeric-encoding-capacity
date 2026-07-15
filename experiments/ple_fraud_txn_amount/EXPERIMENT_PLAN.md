# Experiment Plan (Gate 1) — PLE on transaction amount, IEEE-CIS fraud

**Review mode:** debate. **Case verdict:** `empirical_test_agreed` (converged, 2 rounds).
**Data:** real IEEE-CIS (`data/ieee-fraud-detection/train_transaction.csv` + `train_identity.csv`).
**Primary metric:** PR-AUC (average precision). **Split:** temporal (past→future) by `TransactionDT`.

## Pre-flight checklist (debate-derived)

| # | Source | Finding | Verdict | Item | Resolution in Step 6 | Status |
|---|--------|---------|---------|------|----------------------|--------|
| 1 | debate | F1 | ETA | Is +lift PLE-specific or any non-linearity? | `logreg_quadratic` arm (log_amt + log_amt²) | PENDING |
| 2 | debate | F2 | ETA | PLE on raw vs log amount | `logreg_ple_log` vs `logreg_ple_raw` arms | PENDING |
| 3 | debate | F3 | defense | Drift saturates top PLE bins | Out-of-range-rate diagnostic (train vs test) | PENDING |
| 4 | debate | F4 | defense | Global `la_std` in synthetic labels | Moot on real data (real labels); N/A unless synthetic kept | CLOSED |
| 5 | debate | F5 | defense | Interaction gap needs hgb_ple | `hgb_ple` arm; gap = hgb_ple − hgb_raw | PENDING |
| 6 | debate | F6 | defense | class_weight asymmetry | Harmonize weighting across ALL arms | PENDING |
| 7 | debate | F7 | defense | Oracle-k precision | precision@ fixed budgets (top 0.5%, 1%) + recall@1%FPR | PENDING |
| 8 | user | — | — | The reference model is a sequence model | MLP arms required; sequence arm = next-cycle follow-up | PENDING |
| 9 | HYPOTHESIS | lever | — | PLE must NOT help when fraud is monotone in a feature | Placebo: PLE on a ~monotone control feature → no lift | PENDING |
| 10 | protocol | baseline | — | Beat trivial baseline | `logreg_raw` carried in matrix | PENDING |

## Data preparation

- Load `train_transaction.csv`; left-join `train_identity.csv` on `TransactionID`. Label `isFraud`.
- Feature set (held identical across arms): `TransactionAmt` (the PLE candidate), a compact set of
  raw numerics (e.g. `card1..6`, `addr1/2`, `dist1/2`, a few `C*`/`D*`), and categoricals
  (`ProductCD`, `card4/6`, `P_emaildomain`, etc.) — modest, documented subset (full 400+ columns
  not needed to test an amount-encoding hypothesis).
- **Temporal split:** sort by `TransactionDT`; earliest 70% train, latest 30% test. No future leakage.
- **Precondition check (must pass before interpreting):** plot real fraud-rate vs `TransactionAmt`
  decile; confirm the curve is non-monotone. If it is monotone, the hypothesis's premise fails and
  PLE is not expected to help (report and stop).

## Arm matrix (amount encoding × model; all other features identical, weighting harmonized)

| Model | raw | quadratic | PLE(raw amt) | PLE(log amt) |
|-------|-----|-----------|--------------|--------------|
| logreg | `logreg_raw` (baseline) | `logreg_quadratic` (F1) | `logreg_ple_raw` | `logreg_ple_log` (F2) |
| MLP (neural, required) | `mlp_raw` | — | `mlp_ple_raw` | `mlp_ple_log` (F2, added at Gate 1) |
| HGB (GBDT ref) | `hgb_raw` | — | `hgb_ple` (F5) | — |

Plus **placebo** (lever): `logreg_ple_placebo` — PLE on a control numeric feature whose fraud-rate
curve is ~monotone (chosen by EDA) vs that feature raw.

## Tests and pre-specified verdicts

**H-main (the hypothesis).** Within each model class, PLE-amount beats raw-amount in PR-AUC.
- *Support:* `*_ple_* − *_raw` > 0 with non-overlapping bootstrap 95% CIs, especially for MLP.
- *Refute:* CIs overlap zero.

**T-F1 (attribution).** `logreg_ple_* − logreg_quadratic`.
- *PLE-specific:* PLE arms beat quadratic with CI separation → binning adds beyond a polynomial.
- *Not PLE-specific (critique-validated):* quadratic within CI of PLE → "any non-linearity" suffices.

**T-F2 (encoding space).** `logreg_ple_log − logreg_ple_raw`. Reports whether log-space PLE matters.

**T-F5 (interaction gap — the cycle's central question).** `hgb_ple − hgb_raw`.
- *PLE redundant under interactions:* gap ≈ 0 (CI includes 0) → once a model captures
  amount×other interactions, PLE adds nothing. (Would mean: for the sequence/GBDT reference model,
  PLE-on-amount may not be worth it.)
- *PLE still helps:* gap > 0, CI-separated → PLE-on-amount earns its place even in an
  interaction-aware model.

**T-MLP (reference-model-relevant).** `mlp_ple − mlp_raw`. Does PLE help the neural net? This is the
closest in-scope proxy for the reference model.

**T-lever (falsification).** PLE on the ~monotone placebo feature shows no PR-AUC lift (CI includes 0).

## Statistics

- PR-AUC primary; **bootstrap 95% CIs (N=1,000)** via test-set resampling.
- Secondary: precision@{top 0.5%, top 1%} and recall@1%FPR (fixed budgets, per F7).
- `class_weight`/`scale_pos_weight` **harmonized** across all arms (per F6).
- ≥5 model-init seeds for stochastic arms (MLP); data split fixed; report mean + CI.
- All arms on identical train/test indices and identical non-amount features.

## Promotion recommendation

**Recommend promotion to a Metaflow flow.** Real 590k-row data, ~10 arms × seeds, and four
diagnostics (fraud-curve precondition, saturation, interaction gap, placebo) exceed the
single-analysis threshold — and you chose the gated path in cycle 1. The flow's foreach grain is
the model-init seed (the temporal split and real data are fixed); bootstrap CIs come from test
resampling. Same four-gate enforcement (prevent → lint → review → prove).

## Artifact

Promoted flow (`flow/`) or `fraud_ple_amount_experiment2.py` implementing the arm matrix, temporal
split, fixed-budget metrics, bootstrap CIs, and the four diagnostics; writes `stats_results.json`.
