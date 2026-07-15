# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "pandas"]
# ///
"""Throwaway schema inspector for the real account-sequence data.
   uv run _inspect_schema.py
Confirms the datetime/amount/label/key column names the periodic-embedding
flow will depend on, so we don't hit a KeyError mid-flow (cycle-3 lesson).
"""
import pathlib

import pandas as pd
import pyarrow.parquet as pq

DATA = pathlib.Path("/Users/chrissantiago/Dropbox/GitHub/numeric-encoding-capacity/data/account-sequences")

for name in ["train.parq", "transactions.parq"]:
    p = DATA / name
    schema = pq.read_schema(p)
    print(f"\n=== {name} ===  ({len(schema.names)} cols)")
    for fld in schema:
        print(f"  {fld.name:45s} {fld.type}")

# peek a few rows of transactions to see datetime format + per-account ordering
df = pd.read_parquet(DATA / "train.parq", columns=None).head(3)
print("\n=== train.parq head (transposed) ===")
with pd.option_context("display.max_rows", None, "display.max_columns", None):
    print(df.T)
