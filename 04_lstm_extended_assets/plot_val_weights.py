import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ── Config (must match training) ───────────────────────────────────────────────
DATA_PATH   = "data/training_data_clean.csv"
MODEL_PATH  = "models/portfolio_net.pt"
SEQ_LEN     = 50
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
TRAIN_END   = pd.Timestamp("2025-03-01")
VAL_END     = pd.Timestamp("2026-03-01")

feature_cols = [
    "stock_price", "bond_price", "commodity_price", "volatility_price",
    "stock_return", "bond_return", "commodity_return", "volatility_return",
]
return_cols = [
    "stock_return", "bond_return", "commodity_return", "volatility_return",
]
asset_names = ["Stock", "Bond", "Commodity", "Volatility"]

# ── Model definition ──────────────────────────────────────────────────────────
class PortfolioNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_assets, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_assets)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_last  = h_n[-1]
        logits  = self.fc(h_last)
        weights = F.softmax(logits, dim=-1)
        return weights

# ── Load validation set ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df_val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)

X_list = []
for t in range(SEQ_LEN, len(df_val)):
    X_list.append(df_val[feature_cols].iloc[t - SEQ_LEN : t].values)
X_val = np.array(X_list, dtype=np.float32)

val_dates = df_val["date"].iloc[SEQ_LEN:].values

# ── Load model & run inference ─────────────────────────────────────────────────
model = PortfolioNet(len(feature_cols), HIDDEN_SIZE, len(return_cols), NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

with torch.no_grad():
    all_weights = model(torch.from_numpy(X_val)).numpy()

# ── Summary stats ──────────────────────────────────────────────────────────────
print("Average portfolio weights (validation set):")
for name, w in zip(asset_names, all_weights.mean(axis=0)):
    print(f"  {name:>12s}: {w:.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].stackplot(val_dates, all_weights.T, labels=asset_names, alpha=0.85)
axes[0].set_ylabel("Weight")
axes[0].set_ylim(0, 1)
axes[0].set_title("LSTM Portfolio Weights — Validation Set (stacked)")
axes[0].legend(loc="upper left", fontsize=9)

for i, name in enumerate(asset_names):
    axes[1].plot(val_dates, all_weights[:, i], label=name, linewidth=1.2)
axes[1].set_ylabel("Weight")
axes[1].set_ylim(0, 1)
axes[1].set_title("LSTM Portfolio Weights — Validation Set (lines)")
axes[1].legend(loc="upper left", fontsize=9)

fig.autofmt_xdate()
fig.tight_layout()
plt.savefig("val_weights.png", dpi=150)
plt.show()
print("\nSaved → val_weights.png")