"""
data_format.py  —  Converts raw Alpaca CSV into the format expected by train.py.

Input  : data/training_data.csv       (from download_data.py)
Output : data/training_data_clean.csv

Usage  : python data_format.py
"""

import pandas as pd
import json, os

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/training_data_extended.csv"
OUTPUT_PATH = "data/training_data_extended_clean.csv"

# ── 1. Load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_PATH)

# ── 2. Strip time, keep date only ─────────────────────────────────────────────
df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
df = df.drop(columns=["timestamp"])

# ── 3. Pivot: one row per date, one column per symbol ─────────────────────────
df = (
    df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    .reset_index()
)

# ── 4. Define ticker → name mapping with order preserved ────────────────────
ticker_mapping = [

    # US Economy sectors:
    ("VOX", "communication_services"),
    ("XLY", "consumer_discretionary"),
    ("XLP", "consumer_staples"),
    ("XLE", "energy"),
    ("XLF", "financials"),
    ("XLV", "health_care"),
    ("XLI", "industrials"),
    ("XLB", "materials"),
    ("XLRE", "real_estate"),
    ("XLK", "technology"),
    ("XLU", "utilities"),

    # Non-US markets:
    ("EEM", "emerging"),
    ("VEA", "developed"),

    # Fixed income:
    ("TLT", "long_bond"),
    
    # Commodities:
    ("GLD", "gold"),
    ("CPER", "copper"),
    ("MOO", "agriculture"),
    ("USO", "oil"),
]

# Build rename dict and column lists programmatically
rename_dict = {ticker: f"{name}_price" for ticker, name in ticker_mapping}
price_cols = [f"{name}_price" for _, name in ticker_mapping]
return_cols = [f"{name}_return" for _, name in ticker_mapping]

# Apply rename
df = df.rename(columns=rename_dict)

# ── 5. Check for missing days ──────────────────────────────────────────────────
missing = df[df.isna().any(axis=1)]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} rows have missing values — dropping them.")
    print(missing)
    df = df.dropna()
else:
    print("No missing values — all symbols aligned on every trading day.")

# ── 6. Normalise prices to 1.0 on first day ───────────────────────────────────
df[price_cols] = df[price_cols] / df[price_cols].iloc[0]

# ── 7. Daily returns ───────────────────────────────────────────────────────────
for price_col, return_col in zip(price_cols, return_cols):
    df[return_col] = df[price_col].pct_change()

# ── 8. Drop first row (NaN returns), sort, reset index ────────────────────────
df = df.dropna().sort_values("date").reset_index(drop=True)

# ── 9. Reorder columns (order is guaranteed by the mapping) ──────────────────
feature_cols = price_cols + return_cols
df = df[["date"] + feature_cols]

# ── 10. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nRows  : {len(df)}")
print(f"Cols  : {list(df.columns)}")
print(f"Dates : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
print(f"\nSaved to {OUTPUT_PATH}")

# Save base prices before normalising — needed by deploy.py
# Extract first row prices for all price columns
base_prices = {col: float(df[col].iloc[0]) for col in price_cols}

os.makedirs("models", exist_ok=True)
with open("models/base_prices.json", "w") as f:
    json.dump(base_prices, f, indent=2)
print(f"Base prices saved: {len(base_prices)} assets")