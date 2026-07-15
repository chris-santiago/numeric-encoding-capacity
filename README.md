# numeric-encoding-capacity

When does encoding a numeric feature — piecewise-linear encoding (PLE), periodic embeddings,
or a learned per-feature nonlinear expansion — actually improve a model, and when is it
redundant work?

**Short answer, proposed and hedged, not proven:** whether a richer encoding beats a plain
`log` transform tracks one property of the model — *whether the model applies a free
nonlinearity to the feature on its own before mixing it with everything else.* When it does
(an MLP, a tree ensemble, a learned projection), the richer basis is largely redundant and the
only lever is conditioning (`log` over raw). When it does not (a GRU that reads each per-step
input affinely, i.e. only as a weighted sum before the gate nonlinearity), a fixed PLE basis
helps substantially. This repo is a seven-directory investigation line that reached that answer
the hard way: a clean synthetic win, two real-data refutations, one voided experiment, and then
two controlled experiments built specifically to isolate the property above.

This is a mechanism *consistent with* the evidence, not a proven law — see
[What this does and doesn't establish](#what-this-does-and-doesnt-establish) before you act on
it. It is proposed from **two synthetic experiments on two architectures**, it is
**confounded** (MLP vs. GRU differ in more than the one named property), and the real-data
cycles in this same line *refuted* the naive version of this story. Read the caveats, not just
the table below.

## The finding

`raw` = the untransformed value (standardized). `log` = a standard `log1p` rescale. `ple` =
piecewise-linear encoding, a fixed quantile-bin basis. `dense` = a learned per-feature nonlinear
expansion. Two controlled synthetic experiments put these same four encodings under two model
types and got **opposite verdicts** on whether the richer encodings help:

| Model | ranking over {raw, log, ple, dense} | what moved |
|-------|--------------------------------------|-----------|
| Free-nonlinearity (MLP) | `raw` < `log` ≈ `ple` ≈ `dense` | only conditioning (`log` > `raw`, +0.06); `ple`/`dense` tie `log` |
| Affine-read (GRU) | `raw` < `log` < `dense` < `ple` | every step CI-separated, Holm-significant |

**A quick glossary for the numbers below**, so nothing here is undefined: a **seed-level
paired-t 95% CI** re-trains each arm on 5 different random seeds (each seed sets both the data
draw and the initialization) and reports the confidence interval on the average per-seed
difference — the between-run uncertainty a re-trained model would actually see, not
evaluation-set noise. **Holm-significant** means the result survives a Holm step-down
correction for testing several comparisons at once. The **band regime** is the synthetic fraud
signal the GRU experiment constructs — band-selective, cross-feature, and recency-aggregated —
specifically built to stress a model whose per-step numeric input is read affinely (full
construction in [How we got here](#how-we-got-here)).

In the affine-input GRU, the exact band-regime lifts (PR-AUC, seed-level paired-t 95% CIs,
lengths 32 / 300) were:

- `ple − log` = **+0.190 [+0.151, +0.230]** / **+0.208 [+0.149, +0.266]**
- `dense − log` = +0.134 [+0.114, +0.153] / +0.143 [+0.076, +0.210]
- `ple − dense` = +0.057 [+0.026, +0.087] / +0.065 [+0.045, +0.085]
- `log − raw` = +0.138 / +0.140 (conditioning)

**Which row is you:** if your model reads a per-step numeric scalar affinely into a recurrence
(no per-step nonlinear projection) — the affine-read row applies, and PLE is worth testing. If
your model already applies a free nonlinearity to the feature before mixing it with everything
else (an MLP head, a tree ensemble, any per-step projection) — the free-nonlinearity row
applies, and a richer encoding is not expected to help.

## Practical takeaways

- **For a fraud sequence GRU that reads per-step `log`-scalar amount/Δt affinely (no per-step
  projection): PLE on amount and Δt is worth an A/B.** In this synthetic setting it ranked
  above both `log` and a learned `dense`, and it is the cheaper of the two ways to add a
  per-feature nonlinearity. Validate with an A/B on the reference model over `raw`/`log`/`ple`/`dense`,
  seed-level CI-excludes-zero bar, adequate capacity/epochs for the PLE arm.
- **Do not feed StandardScaler-raw amount/Δt affinely into a recurrence.** It is the worst
  option measured; a `log` transform alone buys ~+0.14 PR-AUC.
- **For a model that already applies a free nonlinearity to the feature (MLP, per-step
  projection, tree ensemble): don't expect `ple` or `dense` to beat `log`.** Spend the effort
  on conditioning and signal instead; an unregularized learned basis carries overfitting risk.

## What this does and doesn't establish

- **The mechanism is proposed, not proven.** It is consistent with both experiments and with
  the within-GRU `dense` arm, but two synthetic experiments on two architectures don't make
  it general.
- **The cross-architecture comparison is confounded.** MLP vs GRU differ in more than the
  named property. The cleanest single piece of evidence is *within* the GRU: adding a free
  per-step `dense` nonlinearity to the same model lifts it over `log`.
- **Everything decisive here is synthetic.** Magnitudes are direction-only, not real-world
  estimates. The real-data cycles (2–3) in fact *refuted* the naive amount-encoding story;
  the synthetic cycles isolate a mechanism, they don't forecast a real-fraud lift.
- **The GRU result depends on a constructed signal regime.** The band-selective, cross-feature,
  recency-aggregated signal is where a per-step basis is *expected* to help; a monotone signal
  showed no gain. Whether real amount/Δt-in-context resemble that regime is unknown here.
- **PLE's ranking is training-sensitive.** Feeding many correlated PLE bins into a recurrent
  input was unstable under-resourced (large seed variance, a spurious negative lift); the
  reported result used larger hidden width, more epochs, and a larger early-stopping split.

## Quickstart

Two hands-on synthetic scripts exist and run standalone (each is a single PEP 723 `uv` script —
no `pyproject.toml`, no other setup, no data needed since both generate their data in-code):

```bash
cd experiments/dt_encoding_fraud && uv run dt_encoding_poc.py            # free-nonlinearity (MLP) case
cd experiments/gru_perstep_encoding_fraud && uv run gru_perstep_poc.py   # affine-read (GRU) case
```

These are the smaller-scale mechanism demos, not a byte-for-byte reproduction of the headline
numbers above (those come from the promoted, config-driven Metaflow flow under each cycle's
`flow/`, run at full seed count and capacity) — but they show the same qualitative pattern. For
the authoritative write-up of the headline numbers, read
[`experiments/encoding_capacity_synthesis/REPORT.md`](experiments/encoding_capacity_synthesis/REPORT.md)
directly.

## How we got here

Cycle 1 showed PLE can beat *raw* on a synthetic non-monotone target, model-agnostically — but
on a target and baseline chosen to favor it. Cycles 2–3 then refuted the natural real-world
version (PLE on real transaction amount, single-transaction and in a sequence model) — and in
the sequence case a tabular baseline beat the GRU outright. Cycle 4 tried learned periodic time
embeddings but was void: it encoded a feature that carried no signal. Cycles 5 and 6 are the
payoff: two *construct-valid, controlled* synthetic experiments that hold the signal fixed and
vary only the model, isolating the one property that flips the verdict. The synthesis rests on
those two.

The proposed reading behind the flip: a richer encoding acts as a *substitute for a per-feature
nonlinearity the model lacks*. A GRU reads each per-step numeric only as `W·e(x_t)` before the
gate nonlinearity, so its per-step function class is the span of the encoding — one monotone
shape for `log`, a localized band-capable basis for `ple`. A model that already applies a free
nonlinearity to the feature can rebuild that shape itself, so the basis buys nothing and the
remaining lever is conditioning. Notably the *fixed* PLE basis beat even a *learned* per-step
`dense` projection (`ple − dense` > 0), consistent with the fixed basis avoiding the
optimization the learned one must carry. In cycle 6, "band regime" specifically means: a
per-step fraud signal defined as amount and Δt both falling in specific bands *simultaneously*
(a cross-feature conjunction), aggregated across steps with recency weighting — constructed to
be exactly the kind of signal a single monotone `log` shape struggles to represent per step.

| # | Directory | Setting | Question | Result |
|--:|-----------|---------|----------|--------|
| 1 | `ple_vs_raw_numeric` | synthetic, static | Does PLE beat **raw** for an MLP on a non-monotone target? | **Confirmed, but vs raw.** PLE−raw = +0.035 AUC [0.025, 0.045] (MLP), +0.235 [0.213, 0.258] (logreg). Mechanism is *linearization* (Ridge R² on the latent logit 0.10 → 0.985), and it is model-agnostic *against raw* — `logreg_ple` was the single best arm. Caveat: an additive target maximally favorable to linear-on-PLE, and the baseline is raw, not `log`. |
| 2 | `ple_fraud_txn_amount` | real IEEE-CIS (150k, 3.5% fraud, temporal split) | Does PLE-on-amount transfer to real single-transaction fraud? | **Refuted.** PLE−raw = −0.009 [−0.015, −0.002] (linear), −0.012 [−0.019, −0.004] (MLP); GBDT only +0.008. The synthetic surrogate had been +0.559; the real amount→fraud U-shape is present (ρ=0.76) but weak. Surprise: a placebo PLE on the count feature `C1` gave +0.144 [+0.126, +0.161]. |
| 3 | `ple_fraud_sequence` | real account sequences (~7% fraud, temporal) | In a sequence model, is amount signal carried *in context*? Does per-step PLE help? | **Refuted.** The GRU failed its precondition — `seq_raw − tab_aggregate` = −0.035 / −0.049 (a tabular logreg beat it); the deviation feature hurt (−0.023 [−0.046, −0.000] at L=32); PLE added nothing (+0.008/+0.009, CIs overlap 0). The GRU ignored cross-time order. |
| 4 | `periodic_embed_fraud` | real account data | Do learned periodic embeddings beat fixed sin/cos on the GRU's time features? | **Void — construct-invalid.** The nulls (H1 −0.006, H2 −0.008, both overlap 0) are vacuous: the precondition failed (time-only PR-AUC 0.078 vs 0.064 base ≈ noise), so there was no signal to encode. Per protocol it should have halted. Carries no evidence about periodic embeddings. |
| 5 | `dt_encoding_fraud` | synthetic, construct-valid | Does a rich encoding of an *informative, non-monotone* Δt beat `log`? | **Powered null (free-nonlinearity case).** Positive control fires (`learned − raw` = +0.196 [+0.174, +0.217] on the weak linear head), so the test has power. Under the MLP: `ple − log` = −0.002 (n.s.), `ple_log ≡ log`, and an unregularized learned periodic basis was *worse* (−0.024, Holm-sig, overfitting). `log` is the prudent choice. |
| 6 | `gru_perstep_encoding_fraud` | synthetic, affine-input GRU (seq 32 & 300) | Does unbottlenecking the per-step numeric path help when the model reads it affinely? | **Confirmed, stronger (affine-read case).** `raw` < `log` < `dense` < `ple`, all CI-separated; `ple − log` = +0.19 to +0.21 at both lengths; `ple` beat the learned `dense`. Negative control clean (monotone regime: no arm beats `log`). |
| — | `encoding_capacity_synthesis` | synthesis (of cycles 5 & 6) | — | The two-experiment write-up and figures stating the capacity account, with its confounds and limits. |

Each experiment directory carries its own `HYPOTHESIS.md`, `EXPERIMENT_PLAN.md`,
`CONCLUSIONS.md`, `REPORT_ADDENDUM.md`, the debate transcript (`CRITIC_*`/`DEFENDER_*`), and,
where promoted, a `flow/` Metaflow+Hydra bundle. The studies themselves were run with the
[`ml-lab`](https://github.com/chris-santiago/ml-lab) adversarial-investigation protocol
(critic/defender debate, pre-registered controls, seed-level paired CIs) — that protocol
produced the artifacts above but isn't required reading to use the finding.

## Reference

<details>
<summary><strong>Running the flows</strong></summary>

- **`uv` required.** Every script has a PEP 723 inline dependency header; run with
  `uv run <script>.py`. There is no `pyproject.toml`.
- **`ml-lab` plugin** (only to re-run the investigation protocol, not the scripts):
  `claude plugin install ml-lab@ml-lab`.
- **Metaflow flows** live under `experiments/<cycle>/flow/`, Hydra-configured (`flow/conf/`).
  Local run stores (`.metaflow/`) are gitignored and regenerate on run.

</details>

<details>
<summary><strong>Data</strong></summary>

Two public datasets, referenced through gitignored symlinks in `data/` (machine-specific
absolute paths and large/licensed raw files, so neither the symlinks nor the data are
committed):

| Symlink | Source dataset |
|---------|----------------|
| `data/ieee-fraud-detection` | IEEE-CIS Fraud Detection (Kaggle / Vesta), `train_transaction.csv` |
| `data/account-sequences` | account-sequence fraud set (`train/valid/test.parq`, `transactions.parq`) |

Recreate on a new machine:

```bash
mkdir -p data
ln -s /path/to/ieee-fraud-detection    data/ieee-fraud-detection
ln -s /path/to/account-sequence-data   data/account-sequences
```

Scripts and flow configs resolve data through these symlinks, so a moved dataset is a one-line
symlink fix, not a code edit. (The two real-data cycles, 2 and 3, are the only ones that touch
these; cycles 1, 5, and 6 are fully synthetic and need no data.)

</details>

<details>
<summary><strong>Layout</strong></summary>

```
experiments/
  ple_vs_raw_numeric/         cycle 1 — synthetic, PLE vs raw (static)
  ple_fraud_txn_amount/       cycle 2 — real IEEE-CIS, single-transaction
  ple_fraud_sequence/         cycle 3 — real account sequences
  periodic_embed_fraud/       cycle 4 — real account data (void)
  dt_encoding_fraud/          cycle 5 — synthetic, free-nonlinearity MLP case
  gru_perstep_encoding_fraud/ cycle 6 — synthetic, affine-input GRU case
  encoding_capacity_synthesis/ synthesis of cycles 5 & 6 (REPORT.md + figures)
```

</details>
