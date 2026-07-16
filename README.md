# numeric-encoding-capacity

When does encoding a numeric feature — piecewise-linear encoding (PLE), a learned per-feature projection, or a periodic embedding — actually improve a model, and when is it redundant work?

**Short answer, proposed and hedged, not proven:** whether a richer encoding beats a plain `log` transform tracks one property of the *target*, not of the model — *whether the feature's risk-vs-value shape is localized (sharp).* A localized band (risk concentrated in a narrow range of the value) is the one shape a basis reliably relieves, and it does so **even for a model that applies a free nonlinearity to the feature** — because a razor-sharp bump is hard for SGD to *find* from a scalar, even though the model can represent it. Away from sharp targets, whether a basis helps is governed by the *model*: it helps an affine-read recurrence (a GRU that reads each per-step input as a weighted sum before the gate) on any non-log-linear shape, and is redundant for a free-nonlinearity model that can rebuild the shape itself.

This repo is a ten-directory investigation line that reached that answer the hard way: a clean synthetic win, two real-data refutations, one voided experiment, two controlled experiments that first proposed an *architecture* law, a curvature disambiguation, a real-data A/B, and finally one consolidated study that put all three architectures on one footing and **refuted the architecture law in favor of localization.** The canonical write-up is [`experiments/consolidated/REPORT.md`](experiments/consolidated/REPORT.md); this README summarizes it.

This is a mechanism *consistent with* the evidence, not a proven law — see [What this does and doesn't establish](#what-this-does-and-doesnt-establish) before you act on it. It is proposed from **one consolidated synthetic flow** across three architectures and five risk shapes, it is **partly confounded** (the cross-architecture contrast is not a single-variable manipulation), and the real-data cycles in this same line *refuted* the naive amount-encoding story. Read the caveats, not just the tables below.

## The finding

`raw` = the untransformed value (standardized). `log` = a standard `log1p` rescale, the reference scalar. `ple` = piecewise-linear encoding, a fixed quantile-bin basis. `projection` / `dense` = a learned per-feature nonlinear expansion (`Linear → ReLU`). One controlled synthetic flow crossed three architectures with five risk shapes and asked, per cell, whether any basis beats `log`. Two results are decisive.

**1. A free-nonlinearity model is *not* uniformly redundant — the refutation.** The earlier account (cycles 5–6) said a model that applies a free nonlinearity to the feature never needs a basis. It does, on the one shape those studies never tested against it — a sharp band. PLE vs `log` on the free-nonlinearity MLP (K=6, 8-seed mean, raw PR-AUC gap):

| risk shape | `mlp_log` | `mlp_ple` | raw gap (ple − log) | reading |
|---|---|---|---|---|
| log-linear | 0.444 | 0.445 | +0.001 | redundant |
| monotone-curved | 0.308 | 0.398 | +0.090 (CI touches 0) | ~redundant |
| smooth non-monotone | 0.382 | 0.306 | **−0.076** | PLE *hurts* (MLP forms it itself) |
| **sharp (mode)** | 0.132 | 0.518 | **+0.387** (Holm-sig) | PLE **decisive** |
| sharp (off) | 0.171 | 0.213 | +0.043 | small |

The MLP learns smooth and curved shapes from the scalar unaided, exactly as the old account predicted — but on the sharp band it collapses to 0.13 (near the `log` GRU's 0.083) while `mlp_ple` reaches 0.52, near the 0.61 oracle. The MLP trains fine everywhere else, so this is not undercapacity: a $\sigma \approx 0.15$ localized target simply is not found by SGD from a scalar. A free per-step nonlinearity does not make encoding redundant; *localization* does.

**2. In the affine-read GRU, the best encoder crosses over by shape.** Fixed PLE wins where the target is sharp; a learned projection wins where it is smooth or curved (raw gap over `log`, K=6, 8-seed mean):

| risk shape | `ple − log` | `projection − log` | best encoder |
|---|---|---|---|
| sharp (mode) | **+0.346** | +0.096 | **fixed PLE** ($\gg$ projection) |
| sharp (off) | **+0.067** | +0.037 | fixed PLE |
| smooth non-monotone | +0.030 (n.s.) | **+0.187** | **learned projection** ($\gg$ PLE) |
| monotone-curved | +0.023 (n.s.) | **+0.101** | learned projection |
| log-linear | −0.075 | −0.024 | `log` (encoders pay a deficit) |

Conditioning is a separate axis and persists underneath all of this: on `log_linear`, `log` (0.531) beats `raw` (0.371) by +0.16 — a heavy-tailed raw value fed straight into an affine recurrence is badly conditioned regardless of shape.

**A quick glossary for the numbers**, so nothing here is undefined: **K** is the feature multiplicity — the number of i.i.d. per-step numeric features summed into the risk signal ($K \in \{1, 6\}$ in the consolidated flow; headline numbers are K=6 unless noted). A **seed-level paired-t 95% CI** re-trains each arm on 8 random seeds (each seed sets both the data draw and the initialization) and reports the confidence interval on the average per-seed difference — the between-run uncertainty a re-trained model would actually see, not evaluation-set noise. **Holm-significant** means the result survives a Holm step-down correction for testing several comparisons at once. **Raw gap** is the uncorrected `arm − log` PR-AUC on a condition — the deployment quantity; read magnitudes from it, not from the deficit-corrected `dc_lift` estimand (see [the estimand caution](#what-this-does-and-doesnt-establish)). **Oracle** ranks by the true log-odds used to generate the labels — the ceiling any arm is measured against, not a trainable model. The **sharp band** is a risk shape $\exp(-(s-\mu)^2/2\sigma^2)$ with $\sigma = 0.15$ over the standardized-log value $s$ — risk concentrated in a narrow band, the localized target that triggers the refutation.

**Which row is you:** if a feature's risk-in-context is **sharp / localized non-monotone**, a fixed PLE basis is worth testing on *any* model, including one with a free nonlinearity. If the feature's risk is **smooth non-monotone or curved** and your model reads it affinely (a per-step GRU input, a linear head), a learned projection is the lever. If your model already applies a free nonlinearity to a **smooth or curved** feature, no basis is expected to help — spend the effort on conditioning (`log` over raw).

## The proposed mechanism

A model consumes a scalar $x$ through a per-feature map $e(x)$ and then does something with the result. Whether a richer $e$ beats `log` depends on **two independent obstacles**, either of which a basis can relieve:

- **Obstacle 1 — affine-read limitation.** A GRU (and a static logistic head) reads each per-step input only as $W \cdot e(x_t)$ before a fixed sigmoid/tanh; it applies no free nonlinear network to $x_t$ alone. Its per-step function class *is* the span of the encoding, so a richer $e$ widens it. This makes a basis help the affine-read models on **any** non-log-linear shape.
- **Obstacle 2 — optimization-hardness of a localized target.** A free-nonlinearity model (an MLP on the scalar) *can represent* any 1-D shape, so the prior account expected it never to need a basis. But representation is not optimization: a razor-sharp bump ($\sigma \approx 0.15$) is very hard to *find* by SGD from a scalar. A fixed quantile basis hands the model that localization for free. This makes a basis help **even a free-nonlinearity model** — but only for a **sharp** target.

The old "free-nonlinearity ⇒ redundant" law is the special case where Obstacle 2 is absent: it holds for smooth/curved targets and fails for sharp ones. The prior studies only ever put a *smooth* $\Delta t$ in front of the MLP, so they never triggered Obstacle 2. Within the affine GRU the *kind* of basis then matters: a fixed quantile basis dominates for sharp targets (no ReLU knots for SGD to place), a learned projection dominates for smooth/curved (SGD-learnable, and it pays a smaller dimensionality deficit than a wide PLE basis).

## Practical takeaways

- **Encode by the shape of the feature's risk-in-context, not by its curvature-vs-linearity.** PLE the features whose risk is **sharp / localized non-monotone** ($\Delta t$ is the leading candidate — short = card-testing, long = dormant-reactivation, a localized band). Use a **learned per-step projection** for features whose risk is **smooth non-monotone or curved**. Leave **monotone / log-adequate** features (amount) on the `log` scalar, where a basis only imports the dimensionality deficit.
- **A free per-step nonlinearity does *not* exempt a sharp feature from needing a basis.** If the sharp lever matters, PLE helps even after you add a per-step projection — this is the whole point of the refutation.
- **Do not feed StandardScaler-raw amount/Δt affinely into a recurrence.** It is the worst-conditioned option measured; a `log` transform alone buys ~+0.16 PR-AUC on a log-linear shape.
- **Validate with a production A/B** on the reference model over `log` / `ple` / `projection`, seed-level CI-excludes-zero bar, adequate capacity/epochs for the PLE arm. Treat the synthetic magnitudes as direction-only.

## What this does and doesn't establish

- **The mechanism is proposed, not proven.** It is consistent with all three architectures and both within-model controls, but one synthetic flow does not make it general.
- **The cross-architecture contrast is partly confounded.** MLP vs GRU differ in more than the named property. The cleanest within-model evidence is (a) the GRU `projection`/`dense` arms, which add a free per-step nonlinearity to the *same* model and beat `log` on smooth/curved, and (b) the MLP sharp result, which holds architecture fixed and varies only the target's sharpness.
- **Everything decisive here is synthetic.** Magnitudes bound the *direction* of effects, not their size on real data. The real-data cycles (2–3) in fact *refuted* the naive amount-encoding story; the real-data A/B in cycle 8 failed its precondition (a point-in-time task), so no real magnitude is claimed.
- **The sharp result depends on a constructed band.** A localized Gaussian band is exactly where a fixed basis is expected to help; whether real amount/$\Delta t$-in-context resemble it is unknown here.
- **PLE is training-sensitive.** Many correlated bins into a recurrent input were unstable in an under-resourced run (a first flow gave a spurious negative). The reported run used real capacity, a 120-epoch cap with validation early-stopping, gradient clipping, and best-state restore.
- **Read every lift next to its raw gap.** The deficit-corrected estimand `dc_lift = (arm − log)_condition − (arm − log)_log_linear` equals `raw_gap − deficit`, so a weak or mis-conditioned arm posts a large "lift" that is pure add-back — e.g. `gru_raw` on sharp shows `dc_lift` +0.16 on a `raw_gap` of $\approx 0$. The estimand is the right *structural* quantity (it nets the fixed dimensionality tax) but it is not an effect size. Read magnitude from `raw_gap`.

## Quickstart

Two hands-on synthetic scripts run standalone (each is a single PEP 723 `uv` script — no `pyproject.toml`, no other setup, no data needed since both generate their data in-code):

```bash
cd experiments/dt_encoding_fraud && uv run dt_encoding_poc.py            # free-nonlinearity (MLP): smooth U-shaped target
cd experiments/gru_perstep_encoding_fraud && uv run gru_perstep_poc.py   # affine-read (GRU): band-selective target
```

These are the smaller-scale mechanism demos, not a byte-for-byte reproduction of the headline numbers (those come from the consolidated Metaflow flow at full seed count and capacity). The GRU demo shows the same qualitative crossover as Result 2. The MLP demo, inherited from cycle 5, tests only a smooth U-shaped $\Delta t$ target — it has no sharp band — so it reproduces the *redundant* half of Result 1 (PLE $\approx$ `log` for the MLP), not the sharp-band refutation (0.13 $\to$ 0.52); that number is specific to the consolidated flow. For the authoritative write-up of the headline numbers, read [`experiments/consolidated/REPORT.md`](experiments/consolidated/REPORT.md) directly.

## How we got here

Cycle 1 showed PLE can beat *raw* on a synthetic non-monotone target, model-agnostically — but on a target and baseline chosen to favor it. Cycles 2–3 then refuted the natural real-world version (PLE on real transaction amount, single-transaction and in a sequence model) — and in the sequence case a tabular baseline beat the GRU outright. Cycle 4 tried learned periodic time embeddings but was void: it encoded a feature that carried no signal. Cycles 5 and 6 were the first payoff: two construct-valid synthetic experiments that held the signal fixed and varied only the model, and proposed an *architecture* law — a basis helps iff the model lacks a free per-feature nonlinearity. Cycle 7 disambiguated value-curvature from non-monotonicity (curvature is a lever via feature *combination*, the controlled analog of cycle 2's `C1` surprise), and cycle 8 confirmed the curvature lever transfers to a GRU while attempting a real-data A/B. The **consolidated study** then put all three architectures on one footing across five risk shapes and, by testing a *sharp* target on the free-nonlinearity MLP — which cycles 5–6 never did — **refuted the architecture law.** The corrected account is localization; the synthesis now rests on that one flow.

| # | Directory | Setting | Question | Result |
|--:|-----------|---------|----------|--------|
| 1 | `ple_vs_raw_numeric` | synthetic, static | Does PLE beat **raw** for an MLP on a non-monotone target? | **Confirmed, but vs raw.** PLE−raw = +0.035 AUC [0.025, 0.045] (MLP), +0.235 [0.213, 0.258] (logreg). Mechanism is *linearization* (Ridge $R^2$ on the latent logit 0.10 → 0.985), model-agnostic *against raw*. Caveat: an additive target maximally favorable to linear-on-PLE, and the baseline is raw, not `log`. |
| 2 | `ple_fraud_txn_amount` | real IEEE-CIS (150k, 3.5% fraud, temporal split) | Does PLE-on-amount transfer to real single-transaction fraud? | **Refuted.** PLE−raw = −0.009 [−0.015, −0.002] (linear), −0.012 (MLP); GBDT only +0.008. The real amount→fraud U-shape is present ($\rho = 0.76$) but weak. Surprise: a placebo PLE on the count feature `C1` gave +0.144 [+0.126, +0.161]. |
| 3 | `ple_fraud_sequence` | real account sequences (~7% fraud, temporal) | In a sequence model, is amount signal carried *in context*? Does per-step PLE help? | **Refuted.** The GRU failed its precondition — a tabular logreg beat it (`seq_raw − tab_aggregate` = −0.035 / −0.049); the deviation feature hurt; PLE added nothing (CIs overlap 0). The GRU ignored cross-time order. |
| 4 | `periodic_embed_fraud` | real account data | Do learned periodic embeddings beat fixed sin/cos on the GRU's time features? | **Void — construct-invalid.** The nulls are vacuous: the precondition failed (time-only PR-AUC 0.078 vs 0.064 base $\approx$ noise), so there was no signal to encode. Per protocol it should have halted. |
| 5 | `dt_encoding_fraud` | synthetic, construct-valid | Does a rich encoding of an *informative, smooth* $\Delta t$ beat `log` for a free-nonlinearity MLP? | **Powered null (for a *smooth* target).** Positive control fires; under the MLP `ple − log` = −0.002 (n.s.) and an unregularized learned periodic basis was *worse*. Correct for smooth targets — but the study never tested a *sharp* one, which is where the consolidated flow overturns the reading. |
| 6 | `gru_perstep_encoding_fraud` | synthetic, affine-input GRU (seq 32 & 300) | Does unbottlenecking the per-step numeric path help when the model reads it affinely? | **Confirmed.** `raw` < `log` < `dense` < `ple`, all CI-separated; `ple − log` = +0.19 to +0.21; `ple` beat the learned `dense`. Negative control clean (monotone regime: no arm beats `log`). |
| 7 | `curvature_vs_nonmono` | synthetic, static | Is log-mismatched *value-curvature* (no non-monotonicity) a lever, separate from non-monotonicity? | **Confirmed via combination.** In an affine-read head, PLE beats `log` for a *sum* of monotone-but-log-mismatched features — the controlled analog of cycle 2's `C1` discovery. Curvature is a multivariate lever; a single curved feature is invisible under a rank metric. |
| 8 | `gru_curvature_realdata` | synthetic multivariate + real-data A/B | Does the curvature lever transfer to a GRU? Does it survive on real data? | **Confirmed but masked; real-data A/B void.** Deficit-corrected `ple−log` = +0.143 [+0.068, +0.218] (K=6 GRU), but PLE's recurrence cost (~−0.13) leaves raw `ple−log ≈ 0` — the lever is real, net value needs few features at few bins. The real-data A/B failed its precondition (point-in-time task); see [`REAL_DATA_AB.md`](experiments/gru_curvature_realdata/REAL_DATA_AB.md). |
| — | `consolidated` | **synthesis — the canonical study** | Across 3 architectures × 5 risk shapes, what actually governs when a basis helps? | **Localization, not architecture.** A free-nonlinearity MLP still gains +0.39 from PLE on a sharp band; the affine GRU crosses over (fixed PLE for sharp, learned projection for smooth/curved). Supersedes `encoding_capacity_synthesis`. |
| — | `encoding_capacity_synthesis` | superseded synthesis (cycles 5 & 6) | — | The original two-experiment write-up proposing the architecture law. **Superseded by `consolidated`.** |

Each experiment directory carries its own `HYPOTHESIS.md`, `EXPERIMENT_PLAN.md`, `CONCLUSIONS.md`, `REPORT_ADDENDUM.md`, the debate transcript (`CRITIC_*`/`DEFENDER_*`), and, where promoted, a `flow/` Metaflow+Hydra bundle. The studies were run with the [`ml-lab`](https://github.com/chris-santiago/ml-lab) adversarial-investigation protocol (critic/defender debate, pre-registered controls, seed-level paired CIs) — that protocol produced the artifacts but isn't required reading to use the finding.

## Reference

<details>
<summary><strong>Running the flows</strong></summary>

- **`uv` required.** Every script has a PEP 723 inline dependency header; run with `uv run <script>.py`. There is no `pyproject.toml`.
- **`ml-lab` plugin** (only to re-run the investigation protocol, not the scripts): `claude plugin install ml-lab@ml-lab`.
- **Metaflow flows** live under `experiments/<cycle>/flow/`, Hydra-configured (`flow/conf/`). The canonical consolidated flow is `experiments/consolidated/flow/flow.py` (run `ConsolidatedFlow/1784135301155957`, 8 seeds). Local run stores (`.metaflow/`) are gitignored and regenerate on run.
- **Figures** in `experiments/consolidated/` regenerate from the run artifacts: `uv run ferrum_figs.py`.

</details>

<details>
<summary><strong>Data</strong></summary>

Two public datasets, referenced through gitignored symlinks in `data/` (machine-specific absolute paths and large/licensed raw files, so neither the symlinks nor the data are committed):

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

Scripts and flow configs resolve data through these symlinks, so a moved dataset is a one-line symlink fix, not a code edit. Only the real-data cycles (2, 3, and the cycle-8 A/B) touch these; the synthetic cycles need no data.

</details>

<details>
<summary><strong>Layout</strong></summary>

```
experiments/
  ple_vs_raw_numeric/          cycle 1 — synthetic, PLE vs raw (static)
  ple_fraud_txn_amount/        cycle 2 — real IEEE-CIS, single-transaction
  ple_fraud_sequence/          cycle 3 — real account sequences
  periodic_embed_fraud/        cycle 4 — real account data (void)
  dt_encoding_fraud/           cycle 5 — synthetic, free-nonlinearity MLP (smooth target)
  gru_perstep_encoding_fraud/  cycle 6 — synthetic, affine-input GRU
  curvature_vs_nonmono/        cycle 7 — synthetic, curvature vs non-monotonicity
  gru_curvature_realdata/      cycle 8 — synthetic multivariate curvature + real-data A/B
  encoding_capacity_synthesis/ superseded synthesis of cycles 5 & 6 (architecture law)
  consolidated/                CANONICAL study — localization account (REPORT.md + figures)
```

</details>
