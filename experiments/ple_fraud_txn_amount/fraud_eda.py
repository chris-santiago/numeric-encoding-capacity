# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas>=2.0", "scipy"]
# ///
"""Precondition EDA on real IEEE-CIS: is fraud non-monotone in TransactionAmt?
Also screens candidate numeric features for a ~MONOTONE one to use as the placebo
(falsification) feature. Read-only; prints a monotonicity table.
    uv run fraud_eda.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA = "/Users/chrissantiago/Dropbox/GitHub/numeric-encoding-capacity/data/ieee-fraud-detection/train_transaction.csv"
CANDIDATES = ["TransactionAmt", "card1", "card2", "addr1", "dist1",
              "C1", "C13", "D1", "D15", "TransactionDT"]
usecols = ["isFraud"] + CANDIDATES

df = pd.read_csv(DATA, usecols=usecols)
print(f"rows={len(df)}  overall fraud rate={df['isFraud'].mean():.4f}")


def decile_curve(x, y, n_bins=10):
    """Fraud rate per quantile-decile of x (non-NaN). Returns (rates, monotonicity)."""
    m = x.notna()
    xv, yv = x[m].to_numpy(), y[m].to_numpy()
    # rank-based deciles (handles ties / heavy tails)
    ranks = pd.qcut(pd.Series(xv).rank(method="first"), n_bins, labels=False)
    rates = np.array([yv[ranks == b].mean() for b in range(n_bins)])
    # monotonicity: |Spearman(decile_index, fraud_rate)|; U-shape -> low; monotone -> ~1
    rho = spearmanr(np.arange(n_bins), rates).correlation
    # U-shape score: correlation of fraud_rate with |decile-4.5| (distance from center)
    u_rho = spearmanr(np.abs(np.arange(n_bins) - 4.5), rates).correlation
    return rates, rho, u_rho


print(f"\n{'feature':14s} {'monotonic|rho|':>14s} {'U-shape rho':>12s}  decile fraud rates (low->high)")
for c in CANDIDATES:
    rates, rho, u_rho = decile_curve(df[c], df["isFraud"])
    flag = ""
    if abs(rho) > 0.85:
        flag = " <- MONOTONE (placebo candidate)"
    if u_rho > 0.7 and abs(rho) < 0.5:
        flag = " <- U-SHAPE / non-monotone"
    rates_s = " ".join(f"{r:.3f}" for r in rates)
    print(f"{c:14s} {abs(rho):14.3f} {u_rho:12.3f}  [{rates_s}]{flag}")
