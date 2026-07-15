# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""One-off: inspect the real account-sequence fraud data to scope the Cycle 6 real-data A/B."""
import pandas as pd
import numpy as np
from pathlib import Path

D = Path("data/account-sequences")
for name in ["train", "valid", "test"]:
    df = pd.read_parquet(D / f"{name}.parq")
    print(f"=== {name}.parq: shape {df.shape} ===")
    if name == "train":
        print("columns:", list(df.columns))
        print("\ndtypes:\n", df.dtypes)
        print("\nhead:\n", df.head(3).to_string())

tx = pd.read_parquet(D / "transactions.parq")
print(f"\n=== transactions.parq: shape {tx.shape} ===")
print("columns:", list(tx.columns))
print("\ndtypes:\n", tx.dtypes)
# key candidates
for key in ["accountNumber", "isFraud", "transactionAmount", "transactionDateTime"]:
    if key in tx.columns:
        s = tx[key]
        print(f"\n{key}: dtype={s.dtype}, nunique={s.nunique()}, nulls={s.isna().sum()}")
        if key == "isFraud":
            print(f"  fraud rate: {s.mean():.4f}")
        if key == "transactionAmount":
            print(f"  amount quantiles: {np.round(s.quantile([.01,.5,.9,.99]).values,2)}")
if "accountNumber" in tx.columns:
    sizes = tx.groupby("accountNumber").size()
    print(f"\nsequence length per account: median={sizes.median():.0f}, p90={sizes.quantile(.9):.0f}, "
          f"p99={sizes.quantile(.99):.0f}, max={sizes.max()}")
    print(f"n accounts: {sizes.size}, n txns: {len(tx)}")
# numeric-looking columns (candidate per-step features)
num = tx.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nnumeric columns ({len(num)}):", num)
