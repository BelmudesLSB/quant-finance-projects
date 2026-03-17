import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/training_data_clean.csv"
MODEL_PATH = "models/portfolio_net.pt"

SEQ_LEN     = 50
BATCH_SIZE  = 64
HIDDEN_SIZE = 64
NUM_LAYERS  = 1

TRAIN_END = pd.Timestamp("2025-03-01")
VAL_END   = pd.Timestamp("2026-03-01")

os.makedirs("models", exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

feature_cols = [
    "stock_price", "bond_price", "commodity_price", "volatility_price",
    "stock_return", "bond_return", "commodity_return", "volatility_return",
]
return_cols = [
    "stock_return", "bond_return", "commodity_return", "volatility_return",
]

# ── Split ──────────────────────────────────────────────────────────────────────
df_train = df[df["date"] <= TRAIN_END].reset_index(drop=True)
df_val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)

print(f"Train : {df_train['date'].iloc[0].date()} → {df_train['date'].iloc[-1].date()}  ({len(df_train)} days)")
print(f"Val   : {df_val['date'].iloc[0].date()} → {df_val['date'].iloc[-1].date()}  ({len(df_val)} days)")

# ── Rolling windows ────────────────────────────────────────────────────────────
def make_xy(df, seq_len, feature_cols, return_cols):
    X, Y = [], []
    for t in range(seq_len, len(df)):
        X.append(df[feature_cols].iloc[t - seq_len:t].values)
        Y.append(df[return_cols].iloc[t].values)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_train, Y_train = make_xy(df_train, SEQ_LEN, feature_cols, return_cols)
X_val,   Y_val   = make_xy(df_val,   SEQ_LEN, feature_cols, return_cols)

print(f"\nX_train : {X_train.shape}   Y_train : {Y_train.shape}")
print(f"X_val   : {X_val.shape}     Y_val   : {Y_val.shape}")

# ── DataLoaders ────────────────────────────────────────────────────────────────
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
    batch_size=BATCH_SIZE, shuffle=True, generator=g, drop_last=False,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
)

# ── Model ──────────────────────────────────────────────────────────────────────
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


# ── Loss ───────────────────────────────────────────────────────────────────────
def sharpe_loss_batch(weights, next_day_returns):
    port_rets = (weights * next_day_returns).sum(dim=1)
    mean_R    = port_rets.mean()
    std_R     = port_rets.std(unbiased=False) + 1e-8
    return -(mean_R / std_R)


# ── Training function ──────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader,
                max_epochs=200, patience=20, lr=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )

    best_val_sharpe   = -np.inf
    epochs_no_improve = 0
    best_model_state  = None
    train_history, val_history = [], []

    for epoch in range(max_epochs):

        # Training
        model.train()
        all_train_rets = []
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            weights = model(X_batch)
            loss    = sharpe_loss_batch(weights, Y_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                all_train_rets.append((weights * Y_batch).sum(dim=1))

        all_train_rets = torch.cat(all_train_rets)
        train_sharpe   = (all_train_rets.mean() /
                         (all_train_rets.std(unbiased=False) + 1e-8)).item() * (252 ** 0.5)
        train_history.append(train_sharpe)

        # Validation
        model.eval()
        all_val_rets = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                weights = model(X_batch)
                all_val_rets.append((weights * Y_batch).sum(dim=1))

        all_val_rets = torch.cat(all_val_rets)
        val_sharpe   = (all_val_rets.mean() /
                       (all_val_rets.std(unbiased=False) + 1e-8)).item() * (252 ** 0.5)
        val_history.append(val_sharpe)

        scheduler.step(val_sharpe)

        # Checkpoint
        if val_sharpe > best_val_sharpe:
            best_val_sharpe   = val_sharpe
            best_model_state  = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " ✓"
        else:
            epochs_no_improve += 1
            marker = ""

        print(f"Epoch {epoch+1:>3} | "
              f"Train: {train_sharpe:+.4f} | "
              f"Val: {val_sharpe:+.4f} | "
              f"Best: {best_val_sharpe:+.4f} | "
              f"No improve: {epochs_no_improve:>2}/{patience} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}"
              f"{marker}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_model_state)
    return best_model_state, train_history, val_history


# ── Run ────────────────────────────────────────────────────────────────────────
input_dim  = len(feature_cols)   # 8
num_assets = len(return_cols)    # 4

model = PortfolioNet(input_dim, HIDDEN_SIZE, num_assets, NUM_LAYERS)
print(f"\nParameters: {sum(p.numel() for p in model.parameters()):,}")

best_state, train_history, val_history = train_model(
    model, train_loader, val_loader,
    max_epochs=200, patience=20, lr=1e-3,
)

# ── Save ───────────────────────────────────────────────────────────────────────
torch.save(best_state, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Best val Sharpe (annualised): {max(val_history):.4f}")
