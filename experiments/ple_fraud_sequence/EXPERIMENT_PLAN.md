# Experiment Plan (Gate 1) — Sequence model, amount-in-context (real account data)

**Review:** debate (`empirical_test_agreed`, converged). **Data:** `~/Dropbox/GitHub/demo/tmp/data/`
account-sequence fraud dataset (clean `accountNumber`, temporal split, ~7% fraud). **Primary
metric:** PR-AUC, bootstrap 95% CIs.

## Pre-flight checklist (debate-derived)

| # | Source | Finding | Resolution in Step 6 | Status |
|---|--------|---------|----------------------|--------|
| 1 | F1 (sev9) | Synthetic null was circular (no headroom) | **Real data provides headroom**; deviation vs raw tested where signal isn't saturated | PENDING |
| 2 | F2 (sev8) | Strawman tab baseline | **`tab_aggregate`** arm (last amount + prior mean/std); precondition = seq beats it | PENDING |
| 3 | F3 (sev6) | Standardization/PLE edges fit on labeled step | **Causal-only**: fit scalers + PLE edges on prior-history steps; deviation already causal | PENDING |
| 4 | F4 (sev6) | Convergence unverified | Train-to-plateau (loss monitoring / early stop); report epochs | PENDING |
| 5 | F5 (sev5) | Parity may be length-dependent | **Sequence-length axis** L ∈ {8, 32} | PENDING |
| 6 | HYPOTHESIS | Falsification lever | **Shuffled-history control** (`seq_raw_shuffle`): permute within-account order → context destroyed → advantage over tab should collapse | PENDING |
| 7 | protocol | Trivial baseline | `tab_last` carried | PENDING |

## Data preparation

- Load `transactions.parq` (full) for history; use the provided temporal `train/valid/test.parq`
  as the **target** transactions to score. For each target, build its **causal per-account
  sequence**: the account's prior `L` transactions (by `transactionDateTime`), reaching across the
  split boundary (past-only — realistic, no leakage). Label = target `isFraud`.
- **Feature set (identical across arms):** the per-step `transactionAmount` (encoded per arm) + a
  small benign context set (`cardPresent`, `transactionToAvailable`, `posEntryMode`,
  `merchantCategoryCode` top-k one-hot). **Exclude the `*FraudTrend` columns** (target-derived
  leakage); **keep the `*Count` velocity features** (they count prior transactions, not frauds — a
  causal, non-leaking aggregate, per HYPOTHESIS.md). All non-amount features are held identical
  across arms, so they cannot confound the amount-encoding comparison.
- Causal preprocessing: scalers + PLE edges fit on **training prior-history steps only** (F3).

## Arms

| Arm | Type | Amount encoding | Purpose |
|-----|------|-----------------|---------|
| `tab_last` | logreg | last raw amount | trivial baseline |
| `tab_aggregate` | logreg | last amount + prior mean/std | **F2** — does temporal modeling beat aggregates? |
| `seq_raw` | GRU | per-step raw amount | reference |
| `seq_ple` | GRU | per-step PLE amount | sub-claim (b): does PLE help the GRU? |
| `seq_dev` | GRU | per-step raw + causal deviation | sub-claim (a): does explicit deviation help? |
| `seq_raw_shuffle` | GRU | raw, within-account order shuffled | **lever** — destroy cross-time context |

(All GRU arms also receive the benign context features per step.)

## Tests and pre-specified verdicts

- **H-main (a) — `seq_dev − seq_raw`:** >0 CI-excl-0 → explicit deviation helps (sub-claim a
  confirmed); ≈0 → the GRU already learns context from raw (a refuted; "raw is enough" confirmed).
- **H-main (b) — `seq_ple − seq_raw`:** ≈0 / <0 → PLE adds nothing (consistent w/ cycles 1-2);
  >0 → PLE helps the sequence model (would be new).
- **T-F2 precondition — `seq_raw − tab_aggregate`:** >0 CI-excl-0 → temporal modeling earns its
  place beyond static aggregates; ≈0 → aggregates suffice, the sequence model is not needed.
- **T-lever — `seq_raw − seq_raw_shuffle`:** large drop toward `tab_aggregate` → the signal is
  genuinely cross-time (validates the amount-in-context framing).
- **T-F5 length — lifts as a function of L ∈ {8,32}:** does deviation's value (if any) grow at
  shorter L (noisier running mean)?

## Statistics
PR-AUC primary, **bootstrap 95% CIs (N=1,000)** via test-set resampling; precision@{0.5%,1%},
recall@1%FPR; ≥3 model-init seeds for the GRU arms; temporal split (provided).

## Promotion
Metaflow flow (torch GRU). foreach grain = (length L, init seed). Same four gates
(prevent → lint → review → prove); determinism contract `single_worker` if torch CPU nondeterminism
appears, else `order_independent`.

## Artifact
Promoted `flow/` (or `seq_experiment2.py`): sequence builder, 6 arms, length axis, causal-only
preprocessing, bootstrap CIs, shuffled-history lever; writes `stats_results.json`.
