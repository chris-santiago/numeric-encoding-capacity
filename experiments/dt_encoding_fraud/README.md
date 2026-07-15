# Δt encoding for fraud — raw vs log vs PLE vs learned basis (cycle 5)

**Hypothesis (one paragraph).** For an *informative, non-monotone* inter-transaction-time (Δt)
feature, how you encode it matters in a capacity-dependent way. A weak (linear) model cannot fit a
U-shaped fraud-vs-Δt relationship from `raw` or `log` Δt, so richer encodings (`PLE`, `learned`
periodic) beat them — this is a built-in **positive control** that proves the test has power. A strong
(MLP) model learns the non-monotone transform itself, so the encoding gap collapses and richer
encodings do **not** beat `log` — the reference-model answer. A monotone Δt regime is the
**negative control** (log already suffices). Δt is informative *by construction*, and a precondition
gate asserts it, so this cannot become the vacuous comparison cycle 4 was.

## Quickstart

```bash
uv run dt_encoding_poc.py     # synthetic Δt-encoding grid; writes 2 figures + a controls summary
```

## Pipeline (PoC)

synthetic Δt (minutes = exp(latent), always > 0, no clip) with a controlled fraud-vs-Δt shape
(U-shaped `nonmono` / `mono` control) + a mild co-feature → encode Δt (raw | log | **ple_raw** (PLE on
raw minutes, standard) | **ple_log** (PLE on log1p) | learned-PLR) → linear or MLP head → PR-AUC →
visualize. (PLE fit on raw is the standard Gorishniy form; ple_log is the pre-log variant, included
per review.)

## What the output looks like

A `regime × encoding × model` PR-AUC grid plus three explicit checks: the precondition gate
(Δt-informative?), the positive control (does encoding fire for the weak model?), and the real
question (does PLE/learned beat log for the MLP?). Headline: positive control fires (linear nonmono:
learned 0.83, ple 0.69 ≫ raw 0.46, log 0.11); the MLP closes the gap (ple−log ≈ 0, learned−log ≈
−0.02); monotone control clean (log suffices). Figures: `fig_dt_shapes.png`, `fig_dt_prauc.png`.

## Intent review (Step 2) — design choices and flags

- **The capacity confound is intentional and load-bearing.** `ple`/`learned` carry more parameters
  than `raw`/`log`. For the *linear* row that is the point — it is the positive control showing a
  basis lets a weak head fit a U-shape. The capacity-controlled comparison is the *MLP row*, where all
  arms share MLP capacity and differ only in the Δt basis. Read verdicts off the MLP row.
- **`log`+linear scoring near base (0.11) is correct, not a bug.** A monotone transform of a U-shaped
  signal is non-monotone in the transformed scale too; a linear head cannot bend it, and log compresses
  the heavy tail that gave `raw`+linear its only separation. This is expected.
- **Single Δt feature + one mild co-feature** — deliberate: the question is Δt's encoding in
  isolation, with the co-feature only for realism (Δt dominant). Not the cycle-4 "no signal" trap —
  here Δt is informative by construction (precondition gate asserts it).
- **Synthetic by necessity:** the demo dataset cannot test this (Δt uninformative there) and
  real-world data is not accessible; construct validity requires controlling the signal.

## Known limitations / scope exclusions

- Synthetic 1-D-dominant data; magnitudes are illustrative, not real-world estimates.
- No sequence/GRU arm — the MLP proxies "strong model" for a 1-D-feature encoding question; the
  recurrent regime is left to a reference-model swap.
- σ=2.0, k=16, n_bins=16 fixed (not swept); σ is the dominant PLR knob.
- 3-seed means in the PoC; bootstrap CIs are added at Step 6 to apply the CI-excludes-zero bar.
- Trivial baseline = base rate (0.08), printed for reference.
