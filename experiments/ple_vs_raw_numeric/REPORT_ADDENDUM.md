# Reference-Model Re-Evaluation Addendum — PLE vs raw numeric

Step 9. Re-evaluates the experimental recommendation (**PLE encoding + a linear model**) against
the reference model's constraints the experiment deliberately excluded.

## Recommendation status: reinforced, not reversed

Reference-model re-evaluation usually inverts candidate rankings. Here it **reinforces** the
experimental finding. The best-performing configuration (`logreg_ple`, AUC 0.864) is also the
*simplest and cheapest to operate*: a per-feature quantile transform plus a logistic regression —
CPU-only, interpretable, trivially retrained. The deep net (`mlp_ple`, 0.852) is both
marginally worse and operationally heavier. The reference model's constraints push the same direction the
evidence does.

## The four constraint areas

**1. Retraining dynamics.** Two artifacts must be (re)fit: the PLE bin edges (per-feature
training quantiles) and the linear model. Both are cheap — quantiles + a convex fit, seconds on
CPU, no GPU. Cadence is driven by **feature-distribution drift** (which moves the quantile edges),
not concept drift alone. Warm-start is unnecessary (refit is near-instant). Blast radius of a bad
retrain is small and detectable: a degenerate feature collapses its bins (guarded by an
epsilon), and a coefficient-level diff against the prior model is human-auditable because the
model is linear.

**2. Update latency.** Inference is a per-row `O(d·T)` bin lookup followed by a dot product —
real-time friendly, no accelerator required. New information enters the model only at the
edges+coefficients refit (batch); there is no online path here, but the refit is cheap enough to
run frequently.

**3. Operational complexity.** PLE adds **one versioned artifact**: the per-feature bin-edge
matrix, which must ship with the model and be applied identically at serving time (the same
train-fit/serve-apply discipline as a `StandardScaler`). The natural monitoring signal is the
**out-of-range rate** — the fraction of serving values landing below the lowest or above the
highest fitted edge (these saturate to all-0 / all-1 bins). A rising out-of-range rate is an
early drift alarm. Versus a deep net, the operational surface is dramatically smaller: no GPU
fleet, no training-loop monitoring, and per-bin coefficients are directly inspectable.

**4. Failure modes.**
- **Cold start** (new feature, no training data): cannot fit PLE edges. Fallback to raw
  standardized + linear for that feature until enough data accrues.
- **Stale edges** (distribution drift): quantile edges misalign with the live distribution, so
  bins lose resolution where the mass moved. Detect via the out-of-range rate; remediate by
  refitting edges.
- **Degenerate / constant feature**: zero-width bins (epsilon-guarded; contributes no signal).
- **`n_bins` mis-set**: too few bins underfit the non-monotonic structure; too many create sparse,
  high-variance bins. Untuned here (24); the reference model must validate per dataset.

## Deployment roadmap

1. **Shadow** — compute `PLE + logistic regression` alongside the incumbent; compare AUC and the
   out-of-range rate offline. No serving impact.
2. **Canary** — route a small traffic slice; gate promotion on AUC ≥ incumbent and a stable
   out-of-range rate.
3. **Full rollout** — with automated rollback if held-out AUC drops below a pre-set floor or the
   out-of-range rate exceeds threshold (drift alarm).

## Open questions and scope limits (important)

- **Additive-target caveat (the key limitation).** The synthetic target was an **additive** sum
  of per-feature non-monotonic terms. PLE is a **per-feature** encoding, so this data design is
  maximally favorable to "linear-on-PLE" and is *why* the linear model wins. With strong
  **feature interactions**, a linear-on-PLE model cannot represent the cross terms and the MLP
  (or a model with explicit crosses) could re-earn its advantage. This is the most important
  generalization boundary and the natural next hypothesis.
- **Synthetic only.** Real numeric features are correlated, heavy-tailed, and mixed with
  categoricals; quantile-PLE behavior there is untested.
- **Tree-based PLE bins** (the paper's other variant) were not tested; they may align bin
  boundaries with target structure better than quantile bins.
- **Class imbalance.** Targets were balanced; quantile bins + AUC under heavy imbalance is
  unverified (PR-AUC was carried as the aux metric for this reason).
- **`n_bins` sensitivity** was not swept.

## Revised recommendation

Adopt **PLE (quantile) + logistic regression** as the reference-model default for numeric features with
suspected non-monotonic, largely-additive structure. It is the best-performing and
lowest-complexity configuration tested. Reserve an MLP (or explicit interaction features) for
cases where feature **interactions** are expected — the regime this experiment did not probe.
