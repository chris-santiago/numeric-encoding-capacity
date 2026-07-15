# Conclusions — PLE vs raw numeric features

**Cycle 1. Review mode: debate (verdict `empirical_test_agreed`). Report mode: conclusions_only.**
Executed via the promoted Metaflow flow (`flow/ple_flow.py`), 2 targets × 6 arms × 10 seeds,
AUC-ROC with bootstrap 95% CIs (N=1,000). All four promotion gates passed
(prevent → lint → review → prove; determinism `order_independent` verified).

## Headline

The hypothesis **claim** is confirmed: PLE-MLP beats raw-MLP on a non-monotonic target,
**+0.035 AUC [0.025, 0.045]** (CI excludes 0), and the advantage is real — not capacity, not
optimization — and it **vanishes on a linear control** (the falsification lever holds).

But the hypothesis **mechanism** is refuted as stated. The benefit is **not** an MLP-specific
representational advantage. PLE is *feature engineering that linearizes the target*: a plain
logistic regression on PLE features gains **+0.235 AUC** on the same task (≈7× the MLP's gain),
and **`logreg_ple` (0.864) is the single best arm**, edging out `mlp_ple` (0.852). Once PLE is
applied, the deep net is superfluous.

## Debate scorecard

| Pt | Topic | Debate disposition | Empirical verdict | Evidence |
|----|-------|--------------------|-------------------|----------|
| F1 | Mechanism: MLP-specific vs general linearization | DEFER → T1 | **Critique validated** — general linearization | `logreg_ple − logreg_raw` = +0.235 [0.213, 0.258] ≫ MLP's +0.035; `logreg_ple` is best arm. `finding_T1_mechanism.png` |
| F2 | Capacity confound (more params ⇒ wins) | DEFER → T2 | **Defense** — not capacity | `mlp_ple − mlp_raw_wide` = +0.030 [0.023, 0.036] despite wide having MORE params (18,817 > 16,577); `mlp_ple − mlp_rff` = +0.177. `finding_T2_capacity.png` |
| F3 | Raw-MLP iteration-limited? | DEFER → T3 | **Defense** — not optimization | All MLPs `frac_converged`=1.0; raw-MLP n_iter max 134 ≪ cap 1000. `finding_T3_convergence.png` |
| F4 | Is the target actually linearized in PLE space? | DEFER → T4 | **Mechanism confirmed** | Ridge R² on latent logit: raw 0.10 → PLE **0.985** (nonmono); both ≈1.0 (linear). `finding_T4_linearization.png` |
| F5 | Noise-feature asymmetry | REBUT-IMMATERIAL | **Defense** — immaterial | Not retested; second-order, ambiguous direction. |
| — | Falsification lever (HYPOTHESIS.md) | T5 | **Holds** — benefit is structure-specific | `mlp_ple − mlp_raw`: +0.035 on nonmono, **−0.010 on linear**. `finding_T5_falsification.png` |
| — | Trivial baseline (must be beaten) | protocol | **Beaten** | `logreg_raw` = 0.629 on nonmono; every other arm exceeds it. `summary_all_conditions.png` |

## Per-test detail

**T-main / T5 — the claim and its falsification lever.** PLE-MLP beats raw-MLP by +0.035
(CI-separated) on the non-monotonic target. On the linear control the lift is **−0.010** (CI
[−0.013, −0.007]) — PLE confers no benefit, in fact a hair worse. The advantage is specific to
non-monotonic structure, exactly as pre-registered. `finding_T5_falsification.png`.

**T1 — mechanism (the decisive finding).** A *linear* model gains +0.235 AUC from PLE on the
non-monotonic target — far more than the MLP's +0.035. The pre-registered T1 rule: "if LR-PLE
exceeds LR-raw by a margin comparable to (or larger than) the MLP's, the mechanism is feature
linearization accessible to any model." It does. The "MLP exploits bin-aware inflection points"
story in HYPOTHESIS.md is **not** what's happening. `finding_T1_mechanism.png`.

**T2 — capacity ruled out.** `mlp_raw_wide` has *more* parameters than `mlp_ple` (18,817 vs
16,577) yet loses by +0.030. `mlp_rff` matches PLE's input dim and param count exactly with a
non-bin-local basis and loses by +0.177. The gap is the PLE structure, not parameter count or
dimensionality. `finding_T2_capacity.png`.

**T3 — optimization ruled out.** Every MLP arm converged (early-stopped) well below the
max_iter=1000 cap; raw-MLP's worst case was 134 iterations. Training to convergence (stronger
than the debated 2-point sweep) shows raw-MLP was never iteration-limited.
`finding_T3_convergence.png`.

**T4 — why it works.** PLE lifts the held-out Ridge R² of a *linear* model predicting the latent
logit from **0.10 (raw) to 0.985 (PLE)** on the non-monotonic target. PLE turns a problem a
linear model cannot touch into a nearly-linear one — which is precisely why `logreg_ple` wins.
On the linear target both encodings already give R² ≈ 1.0, so PLE adds nothing.
`finding_T4_linearization.png`.

## Surprise (anticipated by the debate)

`logreg_ple` (0.864) is the **outright best arm** on the non-monotonic target, beating `mlp_ple`
(0.852). The intuition "non-monotonic numeric features need a deep net" is wrong for this family:
PLE-encode the features and a linear model is competitive or better. This was the exact branch
F1's DEFER anticipated, so it is a confirmed prediction, not an unmodelled surprise — no
macro-iteration re-opening is warranted.

## Hypothesis closure

- **Claim (PLE-MLP > raw-MLP on non-monotonic targets):** CONFIRMED, +0.035 AUC, CI-separated,
  robust to capacity (T2), optimization (T3), and a linear control (T5).
- **Mechanism (MLP-specific representational advantage / "less capacity"):** REFUTED. The benefit
  is model-agnostic target linearization (T1, T4); the deep net is not required.
- **Net:** PLE is an effective *encoding* for non-monotonic numeric features. Credit the encoding,
  not the model — a linear model on PLE features is the most efficient configuration tested.

## Figures

`summary_all_conditions.png`, `finding_T1_mechanism.png`, `finding_T2_capacity.png`,
`finding_T3_convergence.png`, `finding_T4_linearization.png`, `finding_T5_falsification.png`.
