# PLE vs raw numeric features — ml-lab investigation

**Hypothesis.** An MLP trained on **piecewise-linear-encoded (PLE)** numeric features achieves
higher AUC-ROC than the *same* MLP trained on **raw standardized** features, when the target
depends **non-monotonically** on the numeric inputs. The mechanism: PLE turns each scalar into
a bin-aware piecewise-linear vector, so every bin boundary is a cheap inflection point and a
non-monotonic decision surface becomes near-linear in PLE space. Falsification lever (Step 6):
the advantage should **collapse** on a linear-target control.

## Quickstart

```bash
uv run ple_numeric_poc.py
```

No setup needed — dependencies are declared in the script's PEP 723 header and resolved by `uv`.

## Pipeline

1. **Data** — `make_data`: 8000 rows, 8 standard-normal features; the first 4 drive a
   non-monotonic logit `Σ ampⱼ·sin(freqⱼ·xⱼ)`, centered to a ~0.5 base rate, sampled Bernoulli.
2. **Encode** — two views of the same split: (a) raw, standardized on train; (b) PLE, with
   24 quantile bin-edges fit on train, encoded via `clip((x−lo)/(hi−lo), 0, 1)` per bin.
3. **Model** — logistic regression (trivial baseline) on raw; identical `(64, 64)` MLP backbone
   trained once on raw and once on PLE.
4. **Score** — AUC-ROC on a held-out 30% test split, identical across all three models.
5. **Visualize** — `ple_poc_mechanism.png`: (left) true vs raw-MLP vs PLE-MLP `P(y=1)` along
   feature 0; (right) the 24 PLE bin-activation ramps for feature 0.

## What the output looks like

Three AUC-ROC numbers and their deltas, plus the figure. Representative run (seed 0):

```
logistic regression (trivial baseline) : 0.6212
raw-MLP                                 : 0.8243
PLE-MLP                                 : 0.8535
PLE - raw  = +0.0293
```

A weak baseline (~0.62) is expected and desired: it confirms the target is genuinely
non-monotonic, not linearly separable.

## Known limitations / explicit scope exclusions

This is a minimal proof-of-concept. Deferred to the Step 6 experiment:

- **No confidence intervals** — single point estimate; the +0.029 gap is not yet shown to
  exceed run-to-run noise. Step 6 adds bootstrap 95% CIs (N=1,000).
- **No linear-target control** — the hypothesis's falsification lever. PLE must *not* win on
  a purely linear target for the non-monotonic mechanism to be credited.
- **Capacity confound** — PLE-MLP's first layer is 192→64 vs raw's 8→64, so it has more
  parameters. Left in deliberately; Step 6 adds a parameter-matched arm.
- **Quantile bins only** — no tree-based PLE variant.
- **Single seed, single architecture, no tuning, no real dataset, MLP-only (no GBDT).**
