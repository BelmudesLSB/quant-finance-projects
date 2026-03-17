"""
data_format.py  —  Converts raw Alpaca CSV into the format expected by train.py.

Input  : data/training_data.csv       (from download_data.py)
Output : data/training_data_clean.csv

Usage  : python data_format.py
"""

import pandas as pd
import json, os

# ── Config ─────────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/training_data.csv"
OUTPUT_PATH = "data/training_data_clean.csv"

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

# ── 4. Rename columns to match 02_lstm_trading_strategy.ipynb ─────────────────
df = df.rename(columns={
    "VTI":  "stock_price",
    "SCHZ": "bond_price",
    "PDBC": "commodity_price",
    "VIXM": "volatility_price",
})

# ── 5. Check for missing days ──────────────────────────────────────────────────
missing = df[df.isna().any(axis=1)]
if len(missing) > 0:
    print(f"WARNING: {len(missing)} rows have missing values — dropping them.")
    print(missing)
    df = df.dropna()
else:
    print("No missing values — all 4 symbols aligned on every trading day.")

# ── 6. Normalise prices to 1.0 on first day ───────────────────────────────────
price_cols = ["stock_price", "bond_price", "commodity_price", "volatility_price"]
df[price_cols] = df[price_cols] / df[price_cols].iloc[0]

# ── 7. Daily returns ───────────────────────────────────────────────────────────
for col in price_cols:
    df[col.replace("_price", "_return")] = df[col].pct_change()

# ── 8. Drop first row (NaN returns), sort, reset index ────────────────────────
df = df.dropna().sort_values("date").reset_index(drop=True)

# ── 9. Fix column order to match the notebook exactly ─────────────────────────
feature_cols = [
    "stock_price", "bond_price", "commodity_price", "volatility_price",
    "stock_return", "bond_return", "commodity_return", "volatility_return",
]
df = df[["date"] + feature_cols]

# ── 10. Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nRows  : {len(df)}")
print(f"Cols  : {list(df.columns)}")
print(f"Dates : {df['date'].iloc[0]} → {df['date'].iloc[-1]}")
print(f"\nSaved to {OUTPUT_PATH}")

# Save base prices before normalising — needed by deploy.py
base_prices = {
    "stock_price":     float(df["stock_price"].iloc[0]),
    "bond_price":      float(df["bond_price"].iloc[0]),
    "commodity_price": float(df["commodity_price"].iloc[0]),
    "volatility_price":float(df["volatility_price"].iloc[0]),
}
os.makedirs("models", exist_ok=True)
with open("models/base_prices.json", "w") as f:
    json.dump(base_prices, f, indent=2)
print(f"Base prices saved: {base_prices}")