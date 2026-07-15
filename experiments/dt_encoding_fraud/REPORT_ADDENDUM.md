# Reference-Model Re-Evaluation Addendum — Δt encoding, fraud sequence GRU

Step 9. The reference model runs a GRU (seq~300) where SHAP shows inter-transaction time (Δt) is highly
important. This cycle tested how to *encode* Δt, on a construct-valid synthetic feature (the cycle-4
fix: Δt informative by design, positive control fires, so the null is real evidence).

## Recommendation

**Keep Δt as a `log`-transformed (log1p-minutes) per-step input — the prudent cost-asymmetry default.
Do not adopt PLE or a learned periodic basis for Δt.** (This is a prudent default under acknowledged
transfer uncertainty, not proven dominance for the reference GRU — see Bounds. It rests on the robust
finding that *no encoding beats log for a capable model*.) Seed-level paired CIs (n=5):
- `raw` minutes < `log` (`log−raw` = +0.033* [+0.010, +0.056] on monotone; larger on heavy-tailed
  regimes).
- **PLE is equivalent to `log`** for a capable model: `ple_log−log` = +0.000 [−0.002, +0.002] (within
  the ±0.005 equivalence margin); `ple_raw−log` = −0.002, p=0.148 (no gain) — only added bins/edges.
- **An unregularized learned periodic basis is *worse*** (`learned−log` = −0.024* [−0.038, −0.010],
  p=0.009, Holm-sig) — it overfits a strong model (verified: train loss 0.096 < log 0.108). A
  *regularized* learned basis nearly closes the gap (`learned_reg−log` = −0.004) but still does not
  beat log, and neither does a matched non-periodic expansion (`log_expand` ≈ log).

## Why (reference-model constraint areas)

*(These are reasoned expectations from the encoding's structure, not benchmarked measurements — no
latency/serving profiling was run here. The accuracy claims are the measured part.)*

1. **Retraining dynamics.** PLE adds quantile edges to refit on drift; a learned basis adds trainable
   parameters that overfit. `log` is parameter-free and drift-stable. Skip both.
2. **Update latency.** Per-step PLE/periodic add preprocessing/compute at every one of ~300 steps for
   no gain (PLE) or a loss (learned). `log` is the cheapest and at least as accurate.
3. **Operational complexity.** `log` removes versioned edges / embedding weights / a σ hyperparameter
   and their monitoring — pure simplification at no accuracy cost.
4. **Failure modes.** A learned basis's overfitting is a silent degradation risk on a strong model;
   `log` is the robust choice.

## Capacity argument (the portable takeaway)

Δt is informative and non-monotone, so encoding *could* matter — and for a *weak* model it matters
enormously (positive control: learned−log +0.235*). But the reference model is capable (a GRU), and a
capable model learns the Δt→fraud transform from `log` itself. Richer encodings are then redundant
(PLE) or harmful (learned periodic overfits). This is the same capacity argument as the rest of the
line, now demonstrated on an *informative* feature with full statistical power — the construct-valid
test cycle 4 could not be.

## Bounds / open question

- **MLP is the strong-model proxy; the reference GRU was not directly tested** (cycle-5 scope =
  linear+MLP). The capacity argument predicts the same for the GRU, but the **definitive test is a
  one-line per-step encoding swap (log → PLE / learned) in the reference GRU**, where SHAP confirms Δt
  is important. Expected from this cycle: PLE ties log, learned ≤ log. Run it before any change.
- The "learned is *worse*" magnitude is σ-dependent; "learned/PLE do not *beat* log" is σ-robust (the
  matched non-periodic expansion also ties log).
- Synthetic, single symmetric U-shape (a harder shape only makes log look better).

## Deployment

No change to the reference model's Δt handling: keep log1p-minutes. Spend model-improvement budget on signal
(amount, velocity, balance) and tuning, not on Δt representation. If revisiting, run the reference-model
encoding-swap A/B with the CI-excludes-zero bar before adopting anything richer than log.

## Cross-cycle synthesis

Cycles 1–4 showed no learned encoding helped — but cycle 4's time-feature null was void (uninformative
feature). Cycle 5 closes the gap properly: on an *informative* Δt feature, with a fired positive
control, richer encodings still do not beat `log` for a capable model (and learned periodic overfits).
The fraud lever is signal, not numeric representation; where representation could matter, a capable
model already extracts it from the standard transform.
