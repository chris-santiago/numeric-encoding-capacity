# Cycle 8 — Does monotone value-curvature give PLE a lever in a GRU?

**Verdict: REFUTED for the sequence model.** Monotone value-curvature is a PLE lever in a *static
affine-read* model (Cycle 7) but **not in a GRU** — the gate nonlinearities already absorb it. Per-step
PLE on a monotone curved feature only imports its structural deficit. The lever that matters for a GRU is
**non-monotonicity** (Cycle 6), not curvature. Full write-up: `CONCLUSIONS.md`, `REPORT_ADDENDUM.md`.

This is the real-data successor to Cycle 3 (which failed an unchecked precondition). Step 6 on real data
was **not run**: the hardened PoC's positive control shows there is nothing to find.

## Quickstart

```bash
uv run curvature_seq_poc.py       # plain PoC: machinery + precondition gate (Cycle-6 training regime)
uv run curvature_seq_poc2.py      # HARDENED PoC: power sweep + gate + attribution; writes power curve PNG
```

## How the refutation was made trustworthy (not just a null)

The debate returned `critique_wins`: a bare null would be uninterpretable (blind metric vs no lever). The
hardened PoC (`curvature_seq_poc2.py`) triangulates three probes so the null is decisive:

1. **Power curve (F1):** sweep departure-from-log `k`; deficit-corrected benefit `AP(ple_count) −
   AP(ple_ref)` stays ≤ 0 and gets *more* negative with curvature — PLE never wins.
2. **`dense` arm (F5):** a learned per-step Linear→ReLU (superset of PLE's per-step expressivity) also
   gains nothing (`dense − log = +0.004`). If any per-step transform could help, `dense` would → **no
   per-step lever exists.**
3. **Oracle ceiling (F4):** `log`-GRU sits +0.032 below the oracle, but `log`/`raw`/`ple_ref`/`dense` all
   cluster — nothing closes the gap, so the residual is recurrent/capacity, not encoding-addressable.

log-loss and Brier (F2) corroborate PR-AUC; the precondition gate (F3) passes CI-clear against a GBM+EWMA
baseline (margin +0.056; order-shuffle drop +0.236).

## Pipeline

`make_sequences` (curved count target + marginal-matched log-adequate reference, recency-weighted
aggregation) → per-step encode (`fit_ref` on train only → `featurize`) → affine GRU (Cycle-6 regime:
minibatch + val early-stop + best-state restore) → PR-AUC / log-loss / Brier on held-out test.

Arms: `log` · `ple_count` (curved target) · `ple_ref` (clean deficit reference) · `dense`
(free-nonlinearity probe) · `raw` (floor). Single-feature-at-a-time targeting (F6).

## Key result (5-seed means, canonical k=1.4)

| arm | PR-AUC | log-loss | Brier |
|---|---|---|---|
| log | 0.596 | 0.2954 | 0.0876 |
| ple_count | **0.555** | 0.3058 | 0.0913 | ← PLE on the curved feature *hurts*
| ple_ref | 0.597 | 0.2971 | 0.0883 |
| dense | 0.600 | 0.2934 | 0.0870 |
| raw | 0.597 | 0.2922 | 0.0864 |
| oracle | 0.628 | — | — |

## Known limitations / scope

- **Synthetic, PoC scale** (L=32, hidden 32, 5 seeds). Non-monotonicity in a GRU is already settled at
  L=300 by Cycle 6, so no scale-up of *this* question is needed.
- **Single-feature-at-a-time** (F6): correlated/co-encoded multi-feature additivity is out of scope.
- **Direction, not magnitude.** The only open item is Cycle 6's stated real-data A/B of the *non-monotone*
  lever, now equipped with Cycle 8's validated precondition gate.
