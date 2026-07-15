# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","pandas>=2.0","pyarrow","torch>=2.2","scikit-learn>=1.4","pyyaml"]
# ///
"""Baseline-verification diagnostic (Step-6 rule): is tab_logreg's low smoke PR-AUC a data-size
artifact or a feature bug? Run make_data at full test caps, fit tab logreg, report base rate,
PR-AUC, and per-feature logreg coefficients so we can see whether context/amount are contributing.
    uv run _check_tab.py
"""
import pathlib
import sys

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

FLOW = pathlib.Path(__file__).parent / "flow"
sys.path.insert(0, str(FLOW))
from periodic_flow import featurize_model, make_data  # noqa: E402

conf = FLOW / "conf"
data_cfg = yaml.safe_load((conf / "data" / "accounts.yaml").read_text())
# decent sample to separate signal from data-size noise
data_cfg["n_targets_train"] = 12000
data_cfg["n_targets_test"] = 4000
data_cfg["n_targets_valid"] = 2000

data = make_data(data_cfg, {"length": 32}, seed=0)
amt_pool, dt_pool = data["_amt_pool"], data["_dt_pool"]
amt_scaler = (float(amt_pool.mean()), float(amt_pool.std() + 1e-6))
dt_scaler = (float(dt_pool.mean()), float(dt_pool.std() + 1e-6))

Xtr = featurize_model(data["train"], amt_scaler, dt_scaler)[:, -1, :]
Xte = featurize_model(data["test"], amt_scaler, dt_scaler)[:, -1, :]
ytr, yte = data["train"]["y"], data["test"]["y"]

cols = ["hour/24", "dow/7", "dt_std", "amt_std",
        "transactionToAvailable", "count60d", "cardPresent"]
print(f"train n={len(ytr)} fraud={ytr.mean():.4f} | test n={len(yte)} fraud={yte.mean():.4f}")
print(f"feature means (test): " + ", ".join(f"{c}={Xte[:,i].mean():.3f}" for i, c in enumerate(cols)))
print(f"feature std  (test): " + ", ".join(f"{c}={Xte[:,i].std():.3f}" for i, c in enumerate(cols)))

clf = LogisticRegression(max_iter=2000)
clf.fit(Xtr, ytr)
ap = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])
print(f"\nFULL tab_logreg PR-AUC = {ap:.4f}  (base rate {yte.mean():.4f})")
print("logreg coefs: " + ", ".join(f"{c}={w:+.3f}" for c, w in zip(cols, clf.coef_[0])))

# ablation: context-only vs amount-only
for name, idx in [("context-only", [4, 5, 6]), ("amount-only", [3]), ("time-only", [0, 1, 2])]:
    c = LogisticRegression(max_iter=2000).fit(Xtr[:, idx], ytr)
    a = average_precision_score(yte, c.predict_proba(Xte[:, idx])[:, 1])
    print(f"  ablation {name:14s} PR-AUC = {a:.4f}")
