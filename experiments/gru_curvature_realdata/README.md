# Cycle 8 — Does monotone value-curvature give PLE a lever in a GRU?

**Verdict (corrected): YES — curvature IS a per-step PLE lever in the affine GRU, but masked by a large
dimensionality deficit.** On multivariate monotone-curved features the deficit-corrected benefit is
**+0.143 [+0.068, +0.218]** (CI-clear); Cycle 7's static-model mechanism transfers to the sequence model.
But PLE's cost in the recurrence is large (~−0.13 for 6 features × 12 bins), so **raw** `ple−log ≈ 0` — the
lever is real but net deployment value requires targeting few features at few bins. Full write-up:
`CONCLUSIONS.md`, `REPORT_ADDENDUM.md`.

> **Correction history.** The single-feature hardened PoC (`curvature_seq_poc2.py`) first concluded the
> opposite — "monotone curvature is NOT a GRU lever (REFUTED)." That was **wrong**: a single curved feature
> is invisible under a rank metric (Cycle 7's multivariate requirement), so it measured only PLE's deficit.
> A positive-control audit (`positive_control.py`) exposed the flaw; the K=6 multivariate reproduction
> (`multivariate_control.py`) reverses it. The retracted scripts are kept for the record.

## Quickstart

```bash
uv run multivariate_control.py    # AUTHORITATIVE: K=6 static-vs-GRU, firing positive controls
uv run positive_control.py        # the audit that retracted the wrong single-feature verdict
uv run curvature_seq_poc2.py      # RETRACTED single-feature PoC (kept for the record)
uv run curvature_seq_poc.py       # plain PoC: precondition gate + Cycle-6 training regime
```

## Why the first verdict was wrong (two compounding flaws)

1. **Single-feature invisibility.** Cycle 7 proved curvature is *multivariate*: one curved feature under
   PR-AUC gives identical rankings for any monotone encoding (`ple−log = −0.000`). `poc2` used one curved
   feature → no curvature signal was expressible. The single-feature positive control confirmed it did not
   fire even in the static head (+0.004, ns). K=6 makes it visible (static +0.015, CI-clear).
2. **Un-netted dimensionality deficit.** In the GRU, PLE costs ~−0.135 for K=6×12 bins; `poc2` read raw
   `ple−log ≈ 0` and mislabeled "real lever − large deficit ≈ 0" as "no lever."

## The decisive result (K=6, static vs GRU, 5 seeds; deficit-corrected)

| arch | condition | raw `ple−log` | deficit-corrected | CI>0 |
|---|---|---|---|---|
| static | curved | −0.019 | +0.015 [+0.003,+0.027] | ✅ positive control |
| static | non-monotone | +0.339 | +0.373 [+0.346,+0.400] | ✅ sanity control |
| **GRU** | **curved** | +0.008 | **+0.143 [+0.068,+0.218]** | ✅ **lever** |
| GRU | non-monotone (smooth) | −0.101 | +0.035 [−0.021,+0.090] | ❌ anomalous |

## Pipeline

`make_sequences` (K=6 iid per-step features; risk additive over features; recency-weighted; conditions
curved/non-monotone/log-adequate) → per-step encode (PLE in **log space**, fit on train) → two
architectures: **static** (recency-pool → logistic; affine-read, no per-step nonlinearity) and **GRU**
(Cycle-6 regime). Deficit-corrected benefit = `(ple−log)_cond − (ple−log)_log-adequate`. Equal feature
weights (confound fix). Reference model: `nn.GRU` reads `W·e(x_t)` affinely.

## Known limitations / open

- **GRU smooth-non-monotone is anomalous** (ns; negative raw) — likely smooth-vs-sharp (Cycle 6's sharp band
  stands); needs a sharp-band multivariate rerun. Do not read it as "non-monotonicity isn't a GRU lever."
- **PoC scale** (L=32, hidden 32, 5 seeds, synthetic). Direction, not magnitude.
- **Deficit-vs-(K, bins) not characterized:** the deployment-decisive question is where the curvature lever
  clears its own dimensionality cost (raw `ple−log > 0`).
- **Real-data A/B** (Cycle 6's standing item) — target few curved/sharp features at few bins, with a
  static-head positive control and the validated precondition gate.
