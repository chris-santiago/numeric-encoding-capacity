# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
Calibrate the encoding-diagnostic shape metrics against the experiments' OWN canonical signals, so the
notebook's classifier provably labels them correctly (fixes review M3/M7). Model-free; no GBM.

Metrics under test (all read off a quantile-binned fraud-rate curve, base-space):
  signal   = weighted std(bin_rate) / base            -- informativeness (non-monotone-sensitive)  [M2]
  non_mono = 1 - R2_isotonic                           -- monotone-vs-not
  curv_gap = R2_isotonic - R2_linear                   -- curvature (departure from linear-in-base)  [M3]
  local    = 1 - participation_ratio(|excess|)         -- sharpness/localization                     [M7]

Canonical shapes (risk as a function of standardized-log value sc), matching the experiments:
  log_linear : sc                (log-adequate -> keep log)
  cubic      : sc**3             (monotone curved -> projection)     [multivariate_control 'curved']
  sqrt       : sign*sqrt(|sc|)   (monotone curved)
  sigmoid    : tanh(1.5*sc)      (monotone curved, saturating)
  quad_band  : sc**2             (SMOOTH non-monotone -> scalar)     [sharp_vs_smooth 'smooth']
  sharp_band : exp(-sc^2/2*.15^2)(SHARP non-monotone -> PLE)         [nonmono_encoders / sharp_vs_smooth]
  noise      : 0                 (no signal -> drop)
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

RNG_BASE, N, NBINS, TARGET_BASE = 20250715, 20000, 20, 0.03
SHARP_SIGMA = 0.15


def risk(shape, sc):
    if shape == "log_linear": return sc
    if shape == "cubic":      return sc ** 3
    if shape == "sqrt":       return np.sign(sc) * np.sqrt(np.abs(sc))
    if shape == "sigmoid":    return np.tanh(1.5 * sc)
    if shape == "quad_band":  return sc ** 2
    if shape == "sharp_band": return np.exp(-(sc ** 2) / (2 * SHARP_SIGMA ** 2))
    if shape == "noise":      return np.zeros_like(sc)
    raise ValueError(shape)


def _std(v):
    return (v - v.mean()) / (v.std() + 1e-12)


def sample(shape, seed):
    rng = np.random.default_rng(RNG_BASE + seed)
    x = np.exp(rng.normal(0, 1.2, N))                 # lognormal feature (like the real ones)
    sc = _std(np.log(x))
    r = _std(risk(shape, sc)) if shape != "noise" else np.zeros(N)
    logit = 2.2 * r
    b = -np.quantile(logit, 1 - TARGET_BASE)
    y = (rng.random(N) < 1 / (1 + np.exp(-(logit + b)))).astype(np.int8)
    return sc, y                                       # screen operates on base-space value sc


def metrics(sc, y, nbins=NBINS):
    qs = np.unique(np.quantile(sc, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(sc, qs[1:-1]), 0, len(qs) - 2)
    nb = len(qs) - 1
    keep = [b for b in range(nb) if (idx == b).any()]
    ctr = np.array([sc[idx == b].mean() for b in keep])
    k = np.array([y[idx == b].sum() for b in keep], float)
    cnt = np.array([(idx == b).sum() for b in keep], float)
    w = cnt / cnt.sum()
    rate = k / cnt
    lg = np.log((k + 0.5) / (cnt - k + 0.5))          # LOGIT space (Haldane) — removes the sigmoid link
    lbar = np.average(lg, weights=w)
    sst = np.average((lg - lbar) ** 2, weights=w) + 1e-12
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip").fit(ctr, lg, sample_weight=cnt)
    r2_iso = 1 - np.average((lg - iso.predict(ctr)) ** 2, weights=w) / sst
    lin = LinearRegression().fit(ctr[:, None], lg, sample_weight=cnt)
    r2_lin = 1 - np.average((lg - lin.predict(ctr[:, None])) ** 2, weights=w) / sst
    # sharpness: fraction of |excess| mass in the single LARGEST CONTIGUOUS above-median run (unimodal band=high)
    exc = np.abs(lg - lbar) * w
    thr = np.median(exc)
    runs, cur = [], 0.0
    for e in exc:
        if e > thr:
            cur += e
        elif cur > 0:
            runs.append(cur); cur = 0.0
    if cur > 0:
        runs.append(cur)
    contig = (max(runs) / exc.sum()) if runs else 0.0
    # signal in logit space: weighted std of logit-rate (informativeness; non-monotone-sensitive)
    return {"signal": float(np.sqrt(sst)),
            "non_mono": float(max(0, 1 - r2_iso)),
            "curv_gap": float(max(0, r2_iso - r2_lin)),
            "contig": float(contig)}


shapes = ["noise", "log_linear", "cubic", "sqrt", "sigmoid", "quad_band", "sharp_band"]
print(f"base~{TARGET_BASE}, N={N}, {NBINS} bins, 5 seeds (mean)\n")
print(f"{'shape':>11} {'signal':>8} {'non_mono':>9} {'curv_gap':>9} {'contig':>7}")
agg = {}
for s in shapes:
    ms = [metrics(*sample(s, seed)) for seed in range(5)]
    agg[s] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    a = agg[s]
    print(f"{s:>11} {a['signal']:8.3f} {a['non_mono']:9.3f} {a['curv_gap']:9.3f} {a['contig']:7.3f}")

print("\n--- threshold suggestions (midpoints between adjacent classes) ---")
mono = max(agg[s]["non_mono"] for s in ["log_linear", "cubic", "sqrt", "sigmoid"])
band = min(agg[s]["non_mono"] for s in ["quad_band", "sharp_band"])
print(f"NONMONO_THRESH  ~ {(mono + band) / 2:.3f}   (monotone<= {mono:.3f} | bands >= {band:.3f})")
ll = agg["log_linear"]["curv_gap"]
curved = min(agg[s]["curv_gap"] for s in ["cubic", "sqrt", "sigmoid"])
print(f"CURV_THRESH     ~ {(ll + curved) / 2:.3f}   (log_linear {ll:.3f} | curved >= {curved:.3f})")
print(f"SHARP_THRESH    ~ {(agg['quad_band']['contig'] + agg['sharp_band']['contig']) / 2:.3f}   "
      f"(smooth {agg['quad_band']['contig']:.3f} | sharp {agg['sharp_band']['contig']:.3f})")
print(f"SIGNAL_FLOOR    ~ {(agg['noise']['signal'] + min(agg[s]['signal'] for s in shapes[1:])) / 2:.3f}   "
      f"(noise {agg['noise']['signal']:.3f} | weakest real {min(agg[s]['signal'] for s in shapes[1:]):.3f})")
