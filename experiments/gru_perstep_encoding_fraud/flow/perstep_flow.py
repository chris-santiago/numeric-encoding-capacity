# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "torch>=2.2",
#   "scikit-learn>=1.4",
#   "scipy",
#   "metaflow>=2.19",
#   "hydra-core",
#   "omegaconf",
#   "pyyaml",
# ]
# ///
"""Promoted Metaflow + Hydra flow — per-step numeric encoding in an affine-input sequence GRU (cycle 6).

Single source of truth for the debated-by-controls Step-6 methodology (HYPOTHESIS.md / EXPERIMENT_PLAN.md).

  data        : synthetic per-account sequences; per-step amount + Δt (log-normal)
  data axes   : regime {band, monotone} × length {32, 300}   (both affect the generated data)
  foreach grain : (regime, length, seed)   — data generated fresh per seed (run-to-run variance)
  arms        : scalar | ple | dense (GRU, affine per-step input; only the encoding varies)
                tab_logreg (trivial baseline) | oracle (precondition: logreg on true band score)
  metric      : PR-AUC
  stats       : SEED-LEVEL paired-t CIs (+ Holm over the band decision family); seed-0 bootstrap is a
                secondary diagnostic only
  controls    : precondition (oracle ≫ base), positive (band dense−scalar CI>0), negative (monotone)
  determinism : single_worker (torch GRU; pinned threads + fixed seeds)
"""

from __future__ import annotations

import math
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"   # shell shim (no first-party imports here)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

_CONF_DIR = pathlib.Path(__file__).parent / "conf"
_STATS_OUT = pathlib.Path(__file__).resolve().parents[1] / "stats_results.json"


# ===========================================================================
# Seam 3 — components: synthetic data, encoders, train/eval
# ===========================================================================

def make_data(data_spec, data_axes, seed):
    """Synthetic per-account sequences for (regime, length). Fresh draw per seed.

    band regime: fraud = burst (cross-step count) of steps where Δt in a short band AND amount in a
    small band (band-selective in both, cross-feature, aggregated). monotone regime: fraud monotone
    in sequence-mean log-amount (control). Scalers + PLE edges fit on TRAIN only; oracle = true score.
    """
    spec = dict(data_spec)
    spec.update(data_axes)
    regime = spec["regime"]
    L = int(spec["length"])
    n = {"train": int(spec["n_train"]), "valid": int(spec["n_valid"]), "test": int(spec["n_test"])}
    a_lo, a_hi = float(spec["amt_band"][0]), float(spec["amt_band"][1])
    d_lo, d_hi = float(spec["dt_band"][0]), float(spec["dt_band"][1])
    amp, target_pos = float(spec["signal_amp"]), float(spec["target_pos"])

    def draw(nn_, rng):
        amt = np.exp(rng.normal(3.0, 0.9, (nn_, L))).astype(np.float32)
        dt = np.exp(rng.normal(2.6, 1.5, (nn_, L))).astype(np.float32)
        if regime == "band":
            match = ((amt >= a_lo) & (amt <= a_hi) & (dt >= d_lo) & (dt <= d_hi)).astype(np.float32)
            # recency-weighted (leaky-integrator) aggregation: GRU-tractable at any length, so the
            # per-step band detection (the encoding lever) is isolated from long-range counting capacity
            w = (float(spec["band_decay"]) ** np.arange(L - 1, -1, -1)).astype(np.float32)
            score = (match * w[None, :]).sum(axis=1)
        elif regime == "monotone":
            score = np.log1p(amt).mean(axis=1)
        else:
            raise ValueError(f"unknown regime {regime!r}")
        s = (score - score.mean()) / (score.std() + 1e-9)
        bs = np.linspace(-8, 4, 240)
        b = bs[np.argmin(np.abs(np.array([1 / (1 + np.exp(-(amp * s + bb))) for bb in bs]).mean(1) - target_pos))]
        y = (rng.uniform(size=nn_) < 1 / (1 + np.exp(-(amp * s + b)))).astype(np.float32)
        return {"amt": amt, "dt": dt, "y": y, "oracle": score.astype(np.float32)}

    rng = np.random.default_rng(seed)
    out = {sp: draw(n[sp], rng) for sp in ("train", "valid", "test")}
    tr = out["train"]
    la, ld = np.log1p(tr["amt"]), np.log1p(tr["dt"])
    out["ref"] = {"la_mu": float(la.mean()), "la_sd": float(la.std() + 1e-6),
                  "ld_mu": float(ld.mean()), "ld_sd": float(ld.std() + 1e-6),
                  "amt_mu": float(tr["amt"].mean()), "amt_sd": float(tr["amt"].std() + 1e-6),
                  "dtraw_mu": float(tr["dt"].mean()), "dtraw_sd": float(tr["dt"].std() + 1e-6),
                  "amt_edges": _ple_edges(tr["amt"], int(spec["n_bins"])),
                  "dt_edges": _ple_edges(tr["dt"], int(spec["n_bins"]))}
    return out


def _ple_edges(x, n_bins):
    e = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    for t in range(1, e.size):
        if e[t] <= e[t - 1]:
            e[t] = e[t - 1] + 1e-9
    return e


def _ple(x, edges):
    out = np.empty(x.shape + (edges.size - 1,), dtype=np.float32)
    for t in range(edges.size - 1):
        out[..., t] = np.clip((x - edges[t]) / (edges[t + 1] - edges[t]), 0.0, 1.0)
    return out


def featurize_seq(enc, split, ref):
    """Per-step GRU features. scalar/dense share [std log amt, std log Δt]; raw = [std raw amt, std raw
    Δt] (no log — conditioning baseline); ple = PLE(raw amt, dt)."""
    if enc == "raw":
        ra = (split["amt"] - ref["amt_mu"]) / ref["amt_sd"]
        rd = (split["dt"] - ref["dtraw_mu"]) / ref["dtraw_sd"]
        return np.stack([ra, rd], axis=-1).astype(np.float32)
    sa = (np.log1p(split["amt"]) - ref["la_mu"]) / ref["la_sd"]
    sd = (np.log1p(split["dt"]) - ref["ld_mu"]) / ref["ld_sd"]
    if enc in ("scalar", "dense"):
        return np.stack([sa, sd], axis=-1).astype(np.float32)
    if enc == "ple":
        return np.concatenate([_ple(split["amt"], ref["amt_edges"]),
                               _ple(split["dt"], ref["dt_edges"])], axis=-1).astype(np.float32)
    raise ValueError(f"unknown enc {enc!r}")


def tab_features(split):
    la, ld = np.log1p(split["amt"]), np.log1p(split["dt"])
    return np.stack([la.mean(1), la.std(1), la.min(1), la.max(1),
                     ld.mean(1), ld.std(1), ld.min(1), ld.max(1)], axis=1).astype(np.float32)


def build_model(model_spec, in_dim, train_cfg):
    """Seam 2/3: GRU with per-step encoding. dense adds a per-step Dense+ReLU before the affine GRU."""
    enc = model_spec["enc"]
    return SeqEncGRU(enc, in_dim, int(train_cfg["gru_hidden"]), int(train_cfg["dense_h"]))


class SeqEncGRU(nn.Module):
    def __init__(self, enc, in_dim, hidden, dense_h):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, dense_h), nn.ReLU()) if enc == "dense" else None
        gru_in = dense_h if enc == "dense" else in_dim
        self.gru = nn.GRU(gru_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, X):
        if self.proj is not None:
            X = self.proj(X)
        out, _ = self.gru(X)
        return self.head(out[:, -1, :]).squeeze(-1)


def _gru_scores(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def is_axis_agnostic_method(kind):
    if kind in ("gru", "tab", "oracle"):
        return True
    raise ValueError(f"unknown method.kind {kind!r}")


def train_arm(method_spec, data, seed, train_cfg):
    """Dispatch on kind. gru -> early-stopped SeqEncGRU; tab/oracle -> logreg. Raises on unknown kind."""
    kind = method_spec["kind"]
    if not is_axis_agnostic_method(kind):
        raise ValueError(kind)

    if kind == "tab":
        Xtr, Xte = tab_features(data["train"]), tab_features(data["test"])
        m = LogisticRegression(max_iter=int(train_cfg.get("logreg_max_iter", 2000))).fit(Xtr, data["train"]["y"])
        return {"scores": m.predict_proba(Xte)[:, 1], "meta": {"kind": "tab"}}
    if kind == "oracle":
        Xtr, Xte = data["train"]["oracle"][:, None], data["test"]["oracle"][:, None]
        m = LogisticRegression(max_iter=2000).fit(Xtr, data["train"]["y"])
        return {"scores": m.predict_proba(Xte)[:, 1], "meta": {"kind": "oracle"}}

    enc = method_spec["enc"]
    ref = data["ref"]
    Xtr = featurize_seq(enc, data["train"], ref)
    Xva = featurize_seq(enc, data["valid"], ref)
    Xte = featurize_seq(enc, data["test"], ref)
    ytr, yva = data["train"]["y"], data["valid"]["y"]
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = build_model(method_spec, Xtr.shape[2], train_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["gru_lr"]))
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    batch = int(train_cfg["gru_batch"])
    patience = int(train_cfg["gru_patience"])
    best_ap, best_state, bad, ran = -1.0, None, 0, 0
    for epoch in range(int(train_cfg["gru_epochs"])):
        model.train()
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        perm = torch.randperm(len(ytr), generator=g)
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
        ran = epoch + 1
        ap = average_precision_score(yva, _gru_scores(model, Xva))
        if ap > best_ap + 1e-5:
            best_ap, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"scores": _gru_scores(model, Xte),
            "meta": {"kind": "gru", "enc": enc, "in_dim": int(Xtr.shape[2]), "epochs_ran": ran}}


def metric(scores, labels, **cfg):
    return float(average_precision_score(labels, scores))


# ===========================================================================
# Library generics — seed-level paired-t (primary) + seed-0 bootstrap (secondary)
# ===========================================================================

def seed_level_lift(ap_a, ap_b):
    """Primary: paired-t 95% CI over per-seed paired PR-AUC diffs (deployment-relevant variance)."""
    d = np.asarray(ap_a, float) - np.asarray(ap_b, float)
    mean_d, n = float(d.mean()), len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    if sd < 1e-12 or n < 2:
        return {"lift_mean": mean_d, "lift_lo": mean_d, "lift_hi": mean_d, "p": 1.0,
                "ci_excludes_zero": False, "n_seeds": n}
    half = float(stats.t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
    p = float(stats.ttest_rel(np.asarray(ap_a, float), np.asarray(ap_b, float)).pvalue)
    return {"lift_mean": mean_d, "lift_lo": mean_d - half, "lift_hi": mean_d + half, "p": p,
            "ci_excludes_zero": bool((mean_d - half) > 0 or (mean_d + half) < 0), "n_seeds": n}


def holm(pairs_with_p, alpha=0.05):
    order = sorted(pairs_with_p, key=lambda kv: kv[1])
    m, rej = len(order), {}
    for i, (k, p) in enumerate(order):
        rej[k] = p <= alpha / (m - i)
        if not rej[k]:
            for k2, _ in order[i + 1:]:
                rej[k2] = False
            break
    return rej


_LIFT_PAIRS = [("dense", "scalar"), ("ple", "scalar"), ("ple", "dense"),
               ("scalar", "raw"), ("ple", "raw"), ("dense", "raw")]


def load_method_cfg(name):
    import yaml

    return yaml.safe_load((_CONF_DIR / "method" / f"{name}.yaml").read_text())


def build_dataset_keys(experiment, exp_cfg, seeds, method_loader=load_method_cfg):
    """foreach grain = (regime, length, seed). One synthetic dataset per cell; all arms train on it."""
    regimes = exp_cfg.get("axes", {}).get("regime", ["band", "monotone"])
    lengths = exp_cfg.get("axes", {}).get("length", [32, 300])
    methods = exp_cfg.get("methods", [])
    combos = []
    for name in methods:
        spec = dict(method_loader(name))
        spec["_method_name"] = name
        is_axis_agnostic_method(spec["kind"])
        combos.append({"method": name, "method_spec": spec})
    keys = []
    for reg in regimes:
        for L in lengths:
            for s in range(seeds):
                keys.append({"experiment": experiment, "data_cell": {"regime": reg, "length": L},
                             "seed": s, "combos": combos})
    return keys


# ===========================================================================
# Analyses (pure readers)
# ===========================================================================

def analyze_aggregate(records, experiment):
    """Pinned run-output contract: aggregate_results rows are {cell, method, <metric>_mean, ...}."""
    acc = {}
    for r in records:
        if r.get("experiment") == experiment:
            acc.setdefault((r["regime"], r["length"], r["method"]), []).append(r["pr_auc"])
    return [{"cell": {"regime": k[0], "length": k[1]}, "method": k[2],
             "pr_auc_mean": float(np.mean(v)), "pr_auc_std": float(np.std(v))}
            for k, v in sorted(acc.items(), key=lambda kv: str(kv[0]))]


def analyze_controls(lift_results, aggregate_results, base_rate):
    """Readout of the three pre-registered gates from the cell-keyed aggregate outputs."""
    def lift(reg, L, pair):
        for e in lift_results:
            c = e["cell"]
            if c["regime"] == reg and c["length"] == L and c["pair"] == pair:
                return e
        return None

    def arm_mean(reg, L, m):
        for r in aggregate_results:
            c = r["cell"]
            if c["regime"] == reg and c["length"] == L and r["method"] == m:
                return r["pr_auc_mean"]
        return float("nan")

    lengths = sorted({r["cell"]["length"] for r in aggregate_results})
    out = {"precondition": {}, "positive_control": {}, "negative_control": {}}
    for L in lengths:
        orc = arm_mean("band", L, "oracle")
        out["precondition"][L] = {"oracle": orc, "pass": bool(orc > 2 * base_rate)}
        pc = lift("band", L, "dense_minus_scalar")
        out["positive_control"][L] = {"lift": pc["lift_mean"],
                                      "fires": bool(pc["ci_excludes_zero"] and pc["lift_mean"] > 0)}
        nlift = [lift("monotone", L, p) for p in ("dense_minus_scalar", "ple_minus_scalar")]
        out["negative_control"][L] = {"clean": bool(all((not e["ci_excludes_zero"]) or e["lift_mean"] <= 0
                                                        for e in nlift))}
    return out


# ===========================================================================
# Seam 2 — DAG
# ===========================================================================

try:
    from metaflow import Config, FlowSpec, card, step
    from metaflow.cards import Markdown  # noqa: F401

    _METAFLOW_AVAILABLE = True
except ImportError:
    _METAFLOW_AVAILABLE = False


def _hydra_parser(text):
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    raw = yaml.safe_load(text) or {}
    overrides = raw.pop("hydra_overrides", [])
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR.resolve()), version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True)


if _METAFLOW_AVAILABLE:

    class PerStepFlow(FlowSpec):
        """Per-step encoding flow. foreach grain = (regime, length, seed); seed-level paired-t CIs."""

        cfg = Config("cfg", default=str(_CONF_DIR / "config.yaml"), parser=_hydra_parser)

        @step
        def start(self):
            cfg = self.cfg
            exp_cfg = cfg["experiment"]
            self.data_cfg = dict(cfg["data"])
            self.training_cfg = dict(cfg["training"])
            self.experiment_cfg = dict(exp_cfg)
            self.experiment_name = exp_cfg["name"]
            self.determinism = exp_cfg.get("determinism", "single_worker")
            self.bootstrap_cfg = dict(cfg["bootstrap"])
            self.base_rate = float(self.data_cfg["target_pos"])
            self.dataset_keys = build_dataset_keys(self.experiment_name, exp_cfg, cfg["seeds"])
            print(f"[start] {self.experiment_name} foreach(regime,length,seed)={len(self.dataset_keys)} "
                  f"arms={len(self.dataset_keys[0]['combos'])}", flush=True)
            self.next(self.train, foreach="dataset_keys")

        @step
        def train(self):
            dk = self.input
            seed = dk["seed"]
            reg, L = dk["data_cell"]["regime"], dk["data_cell"]["length"]
            tcfg = self.training_cfg
            data = make_data(self.data_cfg, dk["data_cell"], seed)
            yte = data["test"]["y"]
            records = []
            for combo in dk["combos"]:
                res = train_arm(combo["method_spec"], data, seed, tcfg)
                s = res["scores"]
                rec = {"experiment": self.experiment_name, "regime": reg, "length": L, "seed": seed,
                       "method": combo["method"], "pr_auc": metric(s, yte), "meta": res["meta"]}
                records.append(rec)
            self.records = records
            self.next(self.join)

        @step
        def join(self, inputs):
            self.all_records = [r for inp in inputs for r in inp.records]
            self.merge_artifacts(inputs, include=[
                "data_cfg", "training_cfg", "experiment_cfg", "experiment_name",
                "bootstrap_cfg", "determinism", "base_rate"])
            print(f"[join] records: {len(self.all_records)}", flush=True)
            self.next(self.aggregate)

        @step
        def aggregate(self):
            # per-seed AP keyed by (regime, length, method)
            ap_by = {}
            for r in self.all_records:
                ap_by.setdefault((r["regime"], r["length"], r["method"]), {})[r["seed"]] = r["pr_auc"]
            self.aggregate_results = analyze_aggregate(self.all_records, self.experiment_name)
            self.equiv_margin = float(self.experiment_cfg.get("equiv_margin", 0.005))

            cells = sorted({(reg, L) for (reg, L, _m) in ap_by})
            self.lift_results = []
            decision_p = []
            for reg, L in cells:
                seeds_sorted = sorted({s for (_r, _l, _m), d in ap_by.items() for s in d
                                       if _r == reg and _l == L})
                for a, b in _LIFT_PAIRS:
                    ka, kb = (reg, L, a), (reg, L, b)
                    if ka not in ap_by or kb not in ap_by:
                        continue
                    ap_a = [ap_by[ka][s] for s in seeds_sorted]
                    ap_b = [ap_by[kb][s] for s in seeds_sorted]
                    e = seed_level_lift(ap_a, ap_b)
                    pair = f"{a}_minus_{b}"
                    e["cell"] = {"regime": reg, "length": L, "pair": pair}
                    e["equivalent_to_scalar"] = bool(b == "scalar" and abs(e["lift_lo"]) <= self.equiv_margin
                                                     and abs(e["lift_hi"]) <= self.equiv_margin)
                    self.lift_results.append(e)
                    if reg == "band":
                        decision_p.append((f"{reg}|{L}|{pair}", e["p"]))
            rej = holm(decision_p)
            for e in self.lift_results:
                c = e["cell"]
                key = f"{c['regime']}|{c['length']}|{c['pair']}"
                if key in rej:
                    e["holm_significant"] = bool(rej[key])
            self.controls = analyze_controls(self.lift_results, self.aggregate_results, self.base_rate)
            print(f"[aggregate] {len(self.aggregate_results)} arm-cells, {len(self.lift_results)} lifts; "
                  f"controls={self.controls}", flush=True)
            self.next(self.an_prauc, self.an_controls)

        @card
        @step
        def an_prauc(self):
            self.an_result = {"experiment": self.experiment_name, "branch": "prauc_by_arm",
                              "rows": self.aggregate_results}
            self.next(self.join_analyses)

        @card
        @step
        def an_controls(self):
            self.an_result = {"experiment": self.experiment_name, "branch": "controls",
                              "rows": self.controls}
            self.next(self.join_analyses)

        @step
        def join_analyses(self, inputs):
            self.analyses = {inp.an_result["branch"]: inp.an_result for inp in inputs}
            first = inputs[0]
            self.aggregate_results = first.aggregate_results
            self.lift_results = first.lift_results
            self.controls = first.controls
            self.equiv_margin = first.equiv_margin
            self.experiment_name = first.experiment_name
            self.determinism = first.determinism
            self.next(self.report)

        @step
        def report(self):
            print("\n=== PerStepFlow analyses ===", flush=True)
            for b, a in self.analyses.items():
                print(f"  {b}: {a['rows']}", flush=True)
            self.next(self.end)

        @step
        def end(self):
            import json

            payload = {"experiment": self.experiment_name, "determinism": self.determinism,
                       "aggregate_results": self.aggregate_results, "lift_results": self.lift_results,
                       "controls": self.controls, "equiv_margin": self.equiv_margin}
            _STATS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"\n=== PerStepFlow complete: {self.experiment_name} === wrote {_STATS_OUT.name}",
                  flush=True)

    if __name__ == "__main__":
        PerStepFlow()
