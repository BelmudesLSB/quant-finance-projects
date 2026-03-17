"""
deploy.py  —  Run daily at 15:45 ET via Task Scheduler or cron.

Pipeline:
    1. Pull last 49 daily closes + today's live price from Alpaca
    2. Build X [50 x 8] in the exact same feature order as train.py
    3. Load portfolio_net.pt → forward pass → 4 weights
    4. Get account equity
    5. Submit market orders
    6. Log to logs/daily_log.csv

Feature order (must match train.py exactly):
    col 0: stock_price     (VTI  normalised)
    col 1: bond_price      (SCHZ normalised)
    col 2: commodity_price (PDBC normalised)
    col 3: volatility_price(VIXM normalised)
    col 4: stock_return    (VTI  pct_change)
    col 5: bond_return     (SCHZ pct_change)
    col 6: commodity_return(PDBC pct_change)
    col 7: volatility_return(VIXM pct_change)

Usage:
    python deploy.py
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

from alpaca.trading.client        import TradingClient
from alpaca.trading.requests      import MarketOrderRequest
from alpaca.trading.enums         import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests         import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe        import TimeFrame, TimeFrameUnit
from alpaca.data.enums            import DataFeed, Adjustment

# ── Logging ────────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/deploy.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# Symbol order is FIXED — must match train.py / data_format.py exactly.
# Never sort or reorder this list.
SYMBOLS = ["VTI", "SCHZ", "PDBC", "VIXM"]

SEQ_LEN     = 50
HIDDEN_SIZE = 64
NUM_LAYERS  = 1
MODEL_PATH  = "models/portfolio_net.pt"
BASE_PATH   = "models/base_prices.json"
LOG_PATH    = "logs/daily_log.csv"

# ── Credentials ────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# ── Model (must match train.py exactly) ───────────────────────────────────────
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


# ── Step 1: Fetch market data ──────────────────────────────────────────────────
from datetime import datetime, timezone, timedelta

def fetch_data():
    data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

    # 70 calendar days back guarantees at least 49 trading days
    start = datetime.now(timezone.utc) - timedelta(days=100)
    end   = datetime.now(timezone.utc) - timedelta(days=1)   # yesterday only

    bars_req = StockBarsRequest(
        symbol_or_symbols = SYMBOLS,
        timeframe         = TimeFrame(1, TimeFrameUnit.Day),
        start             = start,
        end               = end,
        feed              = DataFeed.SIP,
        adjustment        = Adjustment.ALL,
    )
    bars_df = data_client.get_stock_bars(bars_req).df.reset_index()
    bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"]).dt.tz_convert(
        "America/New_York"
    )

    closes = (
        bars_df
        .pivot(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .dropna()
    )
    closes = closes[SYMBOLS]   # enforce column order

    # Take only the last 49 rows
    closes = closes.tail(49)

    assert len(closes) == 49, f"Expected 49 daily bars, got {len(closes)}"

    # Today's live price
    latest_req    = StockLatestTradeRequest(symbol_or_symbols=SYMBOLS, feed=DataFeed.IEX)
    latest_trades = data_client.get_stock_latest_trade(latest_req)

    today_row = pd.DataFrame(
        [[latest_trades[sym].price for sym in SYMBOLS]],
        columns=SYMBOLS,
        index=[pd.Timestamp.now(tz="America/New_York")],
    )

    df = pd.concat([closes, today_row])

    assert list(df.columns) == SYMBOLS, "Column order mismatch"
    assert len(df) == SEQ_LEN, f"Expected {SEQ_LEN} rows, got {len(df)}"

    log.info(f"Data fetched: {len(df)} rows  |  "
             f"{df.index[0].date()} → today live")
    log.info("Live prices: " +
             "  ".join(f"{sym}={latest_trades[sym].price:.2f}" for sym in SYMBOLS))

    return df


# ── Step 2: Build feature matrix X [50 x 8] ───────────────────────────────────
def build_features(df):
    """
    Constructs X in the exact feature order the model was trained on:
        cols 0-3 : normalised prices  (close / base_price from 2016-01-04)
        cols 4-7 : daily returns      (pct_change)

    Column mapping:
        0 stock_price      VTI
        1 bond_price       SCHZ
        2 commodity_price  PDBC
        3 volatility_price VIXM
        4 stock_return     VTI
        5 bond_return      SCHZ
        6 commodity_return PDBC
        7 volatility_return VIXM
    """
    with open(BASE_PATH) as f:
        base_prices = json.load(f)

    prices  = df[SYMBOLS].values.astype(np.float32)   # [50, 4]

    # Normalised prices — same base as training (2016-01-04)
    SYMBOL_TO_COL = {
    "VTI":  "stock_price",
    "SCHZ": "bond_price",
    "PDBC": "commodity_price",
    "VIXM": "volatility_price",
    }

    base = np.array([base_prices[SYMBOL_TO_COL[sym]] for sym in SYMBOLS], dtype=np.float32)
    norm_px = prices / base                            # [50, 4]

    # Daily returns — pct_change along rows
    returns      = np.zeros_like(prices)               # [50, 4]
    returns[1:]  = np.diff(prices, axis=0) / prices[:-1]

    # Stack: [50, 4] prices + [50, 4] returns = [50, 8]
    feat = np.concatenate([norm_px, returns], axis=1)  # [50, 8]

    X = torch.tensor(feat).unsqueeze(0)                # [1, 50, 8]

    log.info(f"Feature matrix: {tuple(X.shape)}")
    log.info(f"Normalised prices (row 49): " +
             "  ".join(f"{sym}={norm_px[-1, i]:.4f}" for i, sym in enumerate(SYMBOLS)))

    return X


# ── Step 3: Run model ──────────────────────────────────────────────────────────
def get_weights(X):
    model = PortfolioNet(
        input_dim   = 8,
        hidden_size = HIDDEN_SIZE,
        num_assets  = len(SYMBOLS),
        num_layers  = NUM_LAYERS,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        weights = model(X).squeeze(0).numpy()    # [4]

    for sym, w in zip(SYMBOLS, weights):
        log.info(f"  Target weight  {sym}: {w:.4f}  ({w*100:.1f}%)")

    assert abs(weights.sum() - 1.0) < 1e-5, "Weights do not sum to 1"
    return weights


# ── Step 4 + 5: Compute and execute orders ────────────────────────────────────
def execute_orders(weights, current_prices, dry_run=False):
    trading_client = TradingClient(api_key=API_KEY, secret_key=API_SECRET, paper=True)

    equity = float(trading_client.get_account().equity)
    log.info(f"Account equity: ${equity:,.2f}")

    for sym, w in zip(SYMBOLS, weights):
        target_value = round(equity * float(w), 2)

        # Current position value
        positions     = {p.symbol: float(p.market_value)
                         for p in trading_client.get_all_positions()}
        current_value = positions.get(sym, 0.0)
        delta_value   = target_value - current_value

        # Skip if difference is less than $1 — not worth trading
        if abs(delta_value) < 1.0:
            log.info(f"  No change  {sym}: ${current_value:.2f} ≈ target ${target_value:.2f}")
            continue

        side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL

        if dry_run:
            log.info(f"  DRY RUN  {side.value.upper():4s}  {sym}  "
                     f"${abs(delta_value):.2f}  "
                     f"(current: ${current_value:.2f} → target: ${target_value:.2f})")
            continue

        req = MarketOrderRequest(
            symbol        = sym,
            notional      = round(abs(delta_value), 2),
            side          = side,
            time_in_force = TimeInForce.DAY,
        )
        order = trading_client.submit_order(req)
        log.info(f"  {side.value.upper():4s}  {sym}  ${abs(delta_value):.2f}  "
                 f"(current: ${current_value:.2f} → target: ${target_value:.2f})  "
                 f"order_id={order.id}")

    return equity


# ── Step 6: Log ────────────────────────────────────────────────────────────────
def log_to_csv(weights, equity, current_prices):
    row = {"timestamp": datetime.now(timezone.utc).isoformat(),
           "equity":    round(equity, 2)}
    for sym, w in zip(SYMBOLS, weights):
        row[f"w_{sym}"]  = round(float(w), 6)
        row[f"px_{sym}"] = round(float(current_prices[sym]), 4)

    df_log = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH)
    df_log.to_csv(LOG_PATH, mode="a", header=header, index=False)
    log.info(f"Logged to {LOG_PATH}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("deploy.py  starting")

    df             = fetch_data()
    current_prices = {sym: float(df[sym].iloc[-1]) for sym in SYMBOLS}
    X              = build_features(df)
    weights        = get_weights(X)
    equity         = execute_orders(weights, current_prices, dry_run=False)  # flip to False when ready
    log_to_csv(weights, equity, current_prices)

    log.info("deploy.py  done")
    log.info("=" * 60)