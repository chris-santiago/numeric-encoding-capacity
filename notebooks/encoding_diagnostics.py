# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Encoding diagnostics for an affine-input GRU
#
# **Screens, per per-step numeric feature, which encoding is likely best when it is read *affinely* by a
# GRU** (`W·e(x_t)`, no per-step Dense/MLP before the recurrence): **`raw`, `log`, `PLE`, or a nonlinear
# projection** (the latter two ~8-dim, optionally after a log transform).
#
# ### What the evidence says (controlled synthetic experiments that isolate the mechanism)
# An affine GRU's gates are *smooth approximators*, so the only per-step structure they can't build is
# **localized / sharp** structure. That gives the encoder rule this notebook implements:
#
# | risk shape (of the *latent* fraud log-odds vs value) | lever? | encoder |
# |---|---|---|
# | log-linear (straight in log) | no | **`log` scalar** |
# | monotone **curved** (smooth) | weak–moderate | **nonlinear projection** (learned, lower deficit) |
# | **smooth** non-monotone (broad U) | no — gates absorb it | `log` scalar |
# | **sharp / localized** non-monotone (band/spike) | **strongest** | **PLE** (fixed quantile knots) |
#
# ### Scope — read this
# - **Marginal & order-agnostic.** Every test is a per-row `(value, label)` screen; row order is irrelevant,
#   so this runs on **raw, unordered sample data**. It does **not** assume sequential ordering.
# - **It answers "which encoder per feature", not "is per-step encoding worth it at all".** That
#   precondition — does a GRU beat a GBM+EWMA baseline *and* use temporal order? — needs properly
#   time-ordered sequence data and the actual GRU A/B, and is **out of scope here** (see the closing note).
#   On the demo data it *failed* (task was point-in-time), so treat every recommendation below as
#   **conditional on that precondition holding**.
# - **Marginal understates curvature** (it's a multivariate effect) and these are **directional PoC-scale**
#   findings. Output = a prior to decide *which* encoder to spend the real A/B on, not a verdict.

# %%
# Dependencies: polars, numpy, matplotlib, scikit-learn.  No GBMs, no trained predictive models.
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

# Thresholds CALIBRATED against the experiments' own canonical signals (see the self-test cell below).
NBINS = 20
NONMONO_THRESH = 0.30    # 1 - R²_isotonic (logit space) above => non-monotone
CURV_THRESH = 0.05       # R²_iso - R²_linear (logit space) above => monotone-curved (vs log-linear)
SHARP_THRESH = 0.37      # contiguity of |excess| mass above => sharp/localized (vs smooth) non-monotone
SIGNAL_Z = 3.0           # feature is informative if signal > mean+Z*std of a label-permutation null
N_PERM = 15              # permutation-null repetitions
EMBED_DIM = 8            # assumed PLE / projection width (deficit note)
HEAVY_TAIL = 4.0         # q99/median above this (positive feature) => log-transform first (affine read: raw<log)
SIG_HIGH, SIG_MED = 0.30, 0.15   # logit-space signal cutpoints for recommendation PRIORITY (relative)


# %% [markdown]
# ## Core metrics (model-free, logit-space)
#
# All shape analysis is on the **logit of the binned fraud rate** — `logit(rate) ≈ latent log-odds`, which
# removes the sigmoid link (a low base rate makes *every* raw-rate curve look convex; logit undoes that).
# `signal` = weighted std of logit-rate (non-monotone-sensitive). `non_mono` = 1−R²_isotonic.
# `curv_gap` = R²_iso − R²_linear (how much a monotone fit beats a straight line = curvature).
# `contig` = fraction of |excess| mass in the single largest contiguous above-median run (sharp band → high).

# %%
def robust_base(x):
    """raw vs log decision, robust to outliers & signed/ratio features. Returns (base_values, note, warn)."""
    f = x[np.isfinite(x)]
    neg = float((f < 0).mean())
    q50, q99 = np.quantile(f, [0.5, 0.99])
    heavy = (q50 > 0 and q99 / max(q50, 1e-9) > HEAVY_TAIL)
    warn = ""
    if neg > 0.01:                                   # signed / ratio feature: log1p invalid
        signed_heavy = np.nanpercentile(np.abs(f), 99) > 10 * (abs(q50) + 1e-9)
        base = np.sign(x) * np.log1p(np.abs(x)) if signed_heavy else x.copy()
        note = "signed-log" if signed_heavy else "raw(signed)"
        warn = "signed feature — log1p invalid; used signed-log / robust-standardize. Verify conditioning."
    elif heavy:
        base = np.log1p(np.clip(x, 0, None)); note = "log"
        if np.nanmin(f) < -1e-6 * (abs(q50) + 1):    # rare negatives silently clipped to 0 for the log
            warn = f"{neg:.1%} negatives (min {np.nanmin(f):.0f}) clipped to 0 for log1p — inspect if meaningful."
    else:
        base = x.copy(); note = "raw"
    med, iqr = np.nanmedian(base), np.subtract(*np.nanpercentile(base, [75, 25]))
    base = (base - med) / (iqr + 1e-9)               # robust-standardize (median/IQR) — never feed raw scale
    rng = np.nanmax(f) - np.nanmin(f)
    if rng > 1e4 and note.startswith("raw"):
        warn = (warn + " ") + f"raw range ~{rng:.0f} is large for an affine read — consider a transform."
    return base.astype(float), note, warn.strip()


def _binned_logit(base, y, nbins=NBINS):
    qs = np.unique(np.quantile(base[np.isfinite(base)], np.linspace(0, 1, nbins + 1)))
    if len(qs) < 4:
        return None
    idx = np.clip(np.digitize(base, qs[1:-1]), 0, len(qs) - 2)
    keep = [b for b in range(len(qs) - 1) if (idx == b).any()]
    ctr = np.array([base[idx == b].mean() for b in keep])
    k = np.array([y[idx == b].sum() for b in keep], float)
    cnt = np.array([(idx == b).sum() for b in keep], float)
    lg = np.log((k + 0.5) / (cnt - k + 0.5))         # Haldane-smoothed logit
    return ctr, lg, cnt, k


def _signal(base, y, nbins=NBINS):
    b = _binned_logit(base, y, nbins)
    if b is None:
        return 0.0
    _, lg, cnt, _ = b
    w = cnt / cnt.sum()
    return float(np.sqrt(np.average((lg - np.average(lg, weights=w)) ** 2, weights=w)))


def shape_metrics(base, y, nbins=NBINS):
    b = _binned_logit(base, y, nbins)
    if b is None:
        return None
    ctr, lg, cnt, k = b
    w = cnt / cnt.sum()
    lbar = np.average(lg, weights=w)
    sst = np.average((lg - lbar) ** 2, weights=w) + 1e-12
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(ctr, lg, sample_weight=cnt)
    r2_iso = 1 - np.average((lg - iso.predict(ctr)) ** 2, weights=w) / sst
    lin = LinearRegression().fit(ctr[:, None], lg, sample_weight=cnt)
    r2_lin = 1 - np.average((lg - lin.predict(ctr[:, None])) ** 2, weights=w) / sst
    exc = np.abs(lg - lbar) * w
    thr = np.median(exc); runs, cur = [], 0.0
    for e in exc:
        if e > thr:
            cur += e
        elif cur > 0:
            runs.append(cur); cur = 0.0
    if cur > 0:
        runs.append(cur)
    contig = (max(runs) / exc.sum()) if runs and exc.sum() > 0 else 0.0
    return {"signal": float(np.sqrt(sst)), "non_mono": float(max(0, 1 - r2_iso)),
            "curv_gap": float(max(0, r2_iso - r2_lin)), "contig": float(contig),
            "min_pos_bin": int(k.min()), "iso": iso, "ctr": ctr, "lg": lg, "cnt": cnt}


def is_informative(base, y, seed=0):
    """permutation null: is the binned signal above what label-shuffling produces? (robust to base rate/N)."""
    real = _signal(base, y)
    rng = np.random.default_rng(seed)
    null = np.array([_signal(base, rng.permutation(y)) for _ in range(N_PERM)])
    return bool(real > null.mean() + SIGNAL_Z * null.std()), real, float(null.mean())


def classify(m):
    if m["non_mono"] > NONMONO_THRESH:
        return "non-monotone-sharp" if m["contig"] > SHARP_THRESH else "non-monotone-smooth"
    return "monotone-curved" if m["curv_gap"] > CURV_THRESH else "log-linear"


def priority(signal):
    return "high" if signal > SIG_HIGH else "med" if signal > SIG_MED else "low"


def recommend(cls, informative, note, signal):
    b = "log" if "log" in note else ("signed-log" if "signed" in note else "raw")
    if not informative:
        return "keep scalar (no marginal signal — encoding won't help; may still matter multivariately → A/B)"
    pri = priority(signal)
    # only encode where there is signal worth spending the ~8-dim deficit on
    enc = {"log-linear":          f"{b} scalar (no basis; gates+linear suffice)",
           "monotone-curved":     f"nonlinear projection on {b} (~{EMBED_DIM}d)",
           "non-monotone-sharp":  f"PLE on {b} (~{EMBED_DIM} bins)" + (" — top candidate" if pri == "high" else ""),
           "non-monotone-smooth": f"{b} scalar (gates absorb a smooth hump; encoding low-value)"}[cls]
    if cls in ("monotone-curved", "non-monotone-sharp") and pri != "high":
        enc = f"[{pri}-signal — low priority] {enc}"
    return enc


# %% [markdown]
# ## Self-test — the classifier must label the experiments' own signals correctly
# If any assertion fails, the metrics/thresholds have drifted and the recommendations are not trustworthy.

# %%
def _canon(shape, seed, n=20000, base_rate=0.03):
    rng = np.random.default_rng(1000 + seed)
    sc = (lambda v: (v - v.mean()) / (v.std() + 1e-12))(np.log(np.exp(rng.normal(0, 1.2, n))))
    r = {"log_linear": sc, "cubic": sc ** 3, "sqrt": np.sign(sc) * np.sqrt(np.abs(sc)),
         "sigmoid": np.tanh(1.5 * sc), "quad_band": sc ** 2,
         "sharp_band": np.exp(-(sc ** 2) / (2 * 0.15 ** 2))}[shape]
    r = (r - r.mean()) / (r.std() + 1e-12)
    logit = 2.2 * r
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit - np.quantile(logit, 1 - base_rate))))).astype(np.int8)
    return sc, y

_expect = {"log_linear": "log-linear", "cubic": "monotone-curved", "sqrt": "monotone-curved",
           "sigmoid": "monotone-curved", "quad_band": "non-monotone-smooth", "sharp_band": "non-monotone-sharp"}
_fail = []
for shp, exp in _expect.items():
    sc, y = _canon(shp, 0)
    got = classify(shape_metrics(sc, y))
    ok = got == exp
    print(f"  {shp:>11}: {got:>20}  {'ok' if ok else 'MISLABELED (expected ' + exp + ')'}")
    if not ok:
        _fail.append(shp)
assert not _fail, f"classifier mislabeled {_fail} — do not trust recommendations until fixed"
print("self-test passed — classifier matches the experimental ground truth.")

# %% [markdown]
# ## 1. Config — point at your data (raw sample; order does not matter)

# %%
DATA_PATH = "data/account-sequences/transactions.parq"     # <-- your file
TARGET = "isFraud"                                          # <-- binary 0/1
FEATURES = [                                                # <-- numeric per-step features (EXCLUDE *FraudTrend leakage)
    "transactionAmount", "transactionToAvailable", "availableMoney", "currentBalance",
    "creditLimit", "accountAge",
    "normMerchantName-accountNumber30dCount", "normMerchantName-accountNumber60dCount",
]
df = pl.read_parquet(DATA_PATH).select([c for c in FEATURES + [TARGET] if c in FEATURES + [TARGET]]) \
       .drop_nulls(subset=FEATURES + [TARGET])
y = df[TARGET].to_numpy().astype(np.int8)
BASE = float(y.mean())
print(f"n={len(df):,}  base fraud rate={BASE:.4f}  features={len(FEATURES)}")

# %% [markdown]
# ## 2. Per-feature diagnostics + plots
# Left: raw vs chosen-base distribution. Right: **logit(fraud rate) vs value-bin** with the monotone
# (isotonic) fit — read it directly (straight → `log`; smooth bend → projection; localized spike/band → PLE).

# %%
rows = []
for f in FEATURES:
    x = df[f].to_numpy().astype(float)
    base, note, warn = robust_base(x)
    m = shape_metrics(base, y)
    if m is None:
        rows.append({"feature": f, "base": note, "class": "n/a (too few bins)", "recommend": "insufficient distinct values"})
        continue
    inf, sig, nul = is_informative(base, y)
    cls = classify(m)
    rows.append({"feature": f, "base": note, "informative": inf, "signal": round(sig, 2),
                 "priority": priority(sig) if inf else "-",
                 "non_mono": round(m["non_mono"], 2), "curv_gap": round(m["curv_gap"], 2),
                 "contig": round(m["contig"], 2), "min_pos_bin": m["min_pos_bin"],
                 "class": cls, "recommend": recommend(cls, inf, note, sig), "warn": warn})

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 3.3))
    fin = x[np.isfinite(x)]
    a0.hist(fin, bins=60, color="#bbb"); a0.set_yscale("log"); a0.set_title(f"{f}\nbase='{note}'")
    a1.plot(range(len(m["lg"])), m["lg"], "o-", color="#4477aa", ms=3, label="logit(fraud rate)")
    a1.plot(range(len(m["lg"])), m["iso"].predict(m["ctr"]), "r-", lw=2, label="isotonic (monotone) fit")
    warn_txt = ("  ⚠ " + warn) if warn else ""
    a1.set_title(f"{cls}{warn_txt}\nsig {sig:.1f}(null {nul:.1f}) · nonmono {m['non_mono']:.2f} · "
                 f"curv {m['curv_gap']:.2f} · contig {m['contig']:.2f} · min+/bin {m['min_pos_bin']}")
    a1.set_xlabel(f"{note}-value quantile bin"); a1.set_ylabel("logit fraud rate"); a1.legend(fontsize=7)
    if m["min_pos_bin"] < 5:
        a1.text(0.02, 0.02, "⚠ few positives/bin — shape noisy", transform=a1.transAxes, color="crimson", fontsize=8)
    plt.tight_layout(); plt.show()

summary = pl.DataFrame(rows)

# %% [markdown]
# ## 3. Decision map + recommendation table
# Features on the two axes that pick the encoder: **non-monotonicity** (y) × **sharpness/contiguity** (x).
# Color = recommended encoder; size = signal; faded = not marginally informative.

# %%
palette = {"log-linear": "#888888", "monotone-curved": "#d62728",
           "non-monotone-smooth": "#ff9900", "non-monotone-sharp": "#1f77b4"}
S = [r for r in rows if "non_mono" in r]
fig, ax = plt.subplots(figsize=(7.5, 5))
for r in S:
    ax.scatter(r["contig"], r["non_mono"], s=30 + 60 * r["signal"], color=palette.get(r["class"], "#000"),
               edgecolor="k", alpha=0.85 if r["informative"] else 0.2)
    ax.annotate(r["feature"], (r["contig"], r["non_mono"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax.axhline(NONMONO_THRESH, color="k", ls="--", lw=0.8); ax.axvline(SHARP_THRESH, color="k", ls="--", lw=0.8)
ax.set_xlabel("contiguity / sharpness of |excess risk|"); ax.set_ylabel("non-monotone fraction (1 − R²_iso, logit)")
ax.set_title("Encoder decision map (faded = not marginally informative)")
plt.tight_layout(); plt.show()

with pl.Config(fmt_str_lengths=90, tbl_rows=50, tbl_width_chars=220):
    cols = [c for c in ["feature", "base", "signal", "priority", "non_mono", "curv_gap", "contig",
                        "min_pos_bin", "class", "recommend", "warn"] if c in summary.columns]
    print(summary.select(cols))

# %% [markdown]
# ## 4. How to read this — and the decisive next step
#
# - **`raw` vs `log`:** follow `base`. Heavy-tailed positives → `log`; signed/ratio features get a
#   signed-log or robust-standardize (heed any `warn`). Everything is median/IQR-standardized before
#   encoding — never feed a raw thousands-scale feature into an affine read.
# - **Encoder:** follow `recommend`. Encode **few** features — the ~8-dim encoding **deficit in a GRU is
#   large** (≈−0.13 for 6 features × 12 bins in the experiments) and is usually the binding constraint, so a
#   real lever can still net-lose. Prioritize high-`signal`, sharp/curved features; keep log-linear and
#   smooth-non-monotone features on the scalar. `NBINS=20` here is *screen resolution*, **not** an encoder
#   bin count — use ~8 bins for PLE/projection.
# - **This is a prior, not a verdict**, and it is **conditional on the precondition** this notebook does not
#   test: that a properly-trained GRU (real capacity, early stopping) **beats a GBM+EWMA baseline AND loses
#   PR-AUC under a prior-steps shuffle**. If it doesn't (as on the demo data — the task was point-in-time),
#   **no per-step encoding helps regardless of feature shape.** Run that gate on properly time-ordered
#   sequence data first; then A/B `log` vs the recommended encoder *in the GRU*, on the top few features,
#   with a static-head positive control and seed-level CIs. These screens tell you which encoder and which
#   features to spend that (more expensive) test on.
