# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","pandas>=2.0","pyarrow","torch>=2.2","scikit-learn>=1.4","pyyaml","matplotlib>=3.8"]
# ///
"""Step-6 precondition diagnostic (F2: verify the real fraud-vs-time response shapes).

Plots empirical fraud rate vs hour-of-day, day-of-week, and log inter-transaction-time decile on
the TRAIN targets. Sets the prior for H1/H2: are the cyclic responses multi-peaked (harmonics a
fixed sin/cos would miss -> periodic might help H1)? is dt monotone/smooth (-> periodic unlikely
to help H2)? Reuses the flow's make_data so the features match the experiment exactly.
    uv run periodic_diagnostic.py
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

FLOW = pathlib.Path(__file__).parent / "flow"
sys.path.insert(0, str(FLOW))
from periodic_flow import make_data  # noqa: E402

data_cfg = yaml.safe_load((FLOW / "conf" / "data" / "accounts.yaml").read_text())
data_cfg["n_targets_train"] = 12000
data = make_data(data_cfg, {"length": 32}, seed=0)
tr = data["train"]
hour = tr["hour"][:, -1]            # last-step (the target transaction)
dow = tr["dow"][:, -1]
dt = tr["dt"][:, -1]                # log inter-transaction minutes
y = tr["y"]
base = y.mean()


def rate_by(x, bins):
    idx = np.digitize(x, bins) - 1
    idx = np.clip(idx, 0, len(bins) - 2)
    centers, rates, ns = [], [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 20:
            centers.append(0.5 * (bins[b] + bins[b + 1]))
            rates.append(y[m].mean())
            ns.append(int(m.sum()))
    return np.array(centers), np.array(rates), ns


fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# hour-of-day (0-23)
cx, rt, _ = rate_by(hour, np.arange(0, 25, 2.0))
axes[0].plot(cx, rt, "o-", color="#1f77b4")
axes[0].axhline(base, color="k", ls=":", lw=0.9, label=f"base {base:.3f}")
axes[0].set_title("fraud rate vs hour-of-day"); axes[0].set_xlabel("hour"); axes[0].legend()

# day-of-week (0-6)
cx, rt, _ = rate_by(dow, np.arange(-0.5, 7.5, 1.0))
axes[1].plot(cx, rt, "o-", color="#2ca02c")
axes[1].axhline(base, color="k", ls=":", lw=0.9)
axes[1].set_title("fraud rate vs day-of-week"); axes[1].set_xlabel("day (0=Mon)")

# log inter-txn time decile
qs = np.quantile(dt, np.linspace(0, 1, 11))
qs = np.unique(qs)
cx, rt, _ = rate_by(dt, qs)
axes[2].plot(cx, rt, "o-", color="#d62728")
axes[2].axhline(base, color="k", ls=":", lw=0.9)
axes[2].set_title("fraud rate vs log inter-txn-time (decile)"); axes[2].set_xlabel("log1p(minutes)")

fig.suptitle("Precondition: real fraud-vs-time response shapes (train targets) — weak, near-flat "
             f"signal (base rate {base:.3f})")
fig.tight_layout()
fig.savefig(pathlib.Path(__file__).parent / "fig_response_shapes.png", dpi=120)
print(f"base fraud rate (train) = {base:.4f}")
print("wrote fig_response_shapes.png")
