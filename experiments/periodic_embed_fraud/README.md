# Periodic embeddings for fraud-GRU time features (cycle 4)

**Hypothesis (one paragraph).** Learned-frequency periodic embeddings (Gorishniy et al. 2022, the
"PLR" form) expand a scalar into many trainable sinusoids so a head can fit a wiggly 1-D response.
Fixed sin/cos is the special case with one hand-chosen frequency per known period. We test whether
learned periodic embeddings beat fixed sin/cos on **cyclic** time features (hour-of-day, day-of-week —
periods known) and beat **raw** on the **non-periodic** inter-transaction-time feature, inside a
fraud sequence GRU. Prior: little gain over sin/cos (known periods leave no harmonics to discover)
and only marginal gain over raw (a GRU can learn a 1-D transform itself) — the capacity argument from
cycles 1–3.

## Quickstart

```bash
uv run periodic_embed_poc.py      # Step-1 synthetic mechanism PoC; writes 3 figures + a summary table
```

## Pipeline (PoC)

data (synthetic 1-D, 3 controlled response shapes) → encoder (raw | fixed sin/cos | learned periodic)
→ head (linear | MLP) → score → PR-AUC (average precision) → visualize (response shapes, learned
frequency spectra, PR-AUC grid).

The three response shapes are chosen to separate the mechanisms:
- `known_period` — `sin(2πx)`: period known; fixed sin/cos already sufficient.
- `multi_harmonic` — `sin(2π·3x)`: a 3rd harmonic a fundamental sin/cos cannot see; learned freqs can.
- `two_bumps` — two Gaussians: non-periodic, non-monotone; periodic acts as a learned Fourier basis.

## What the output looks like

A table of PR-AUC per `(regime × encoder × head)` plus two derived lifts: `per-sin` (periodic −
sin/cos, the H1 lever) and `per-raw` (periodic − raw, the H2 Fourier lever). Headline PoC result:
learned periodic beats fixed sin/cos **only** for `multi_harmonic + linear` (+0.262); it ties sin/cos
when the period is known or when the head is an MLP; it beats raw chiefly for weak heads or
high-frequency targets. Figures: `fig_shapes.png`, `fig_frequencies.png`, `fig_prauc_poc.png`.

## Known limitations / explicit scope exclusions

- **Synthetic 1-D only.** Real account-sequence data (`~/Dropbox/GitHub/demo/tmp/data`) enters at
  Step 6; `*FraudTrend` columns are excluded there (target leakage).
- **No sequence model in the PoC.** The encoding is tested on a static classifier so the encoder
  effect is not entangled with recurrence; the GRU is the Step-6 experiment.
- **Capacity confound (by design in the PoC).** The PLR encoder bundles a `Linear→ReLU`, so
  "periodic + linear head" carries extra capacity vs "raw + linear." The Step-6 GRU experiment must
  control this: every arm feeds the same GRU backbone and differs only in the per-step time-encoding
  block.
- **σ (init scale) and k are fixed, untuned.** σ is likely the dominant knob (frequencies move little
  from init); σ-sensitivity is a Step-6 spot-check, not settled here.
- **Frequencies are not crisply "discovered."** The embedding learns a spread of frequencies and
  relies on the PLR linear layer to recombine them; do not over-read the spectra as exact recovery.
- **No bootstrap CIs in the PoC** (3-seed means only); CIs are added at Step 6.
- Trivial baseline = constant predictor (PR-AUC == base rate ≈ 0.10), printed for reference.
