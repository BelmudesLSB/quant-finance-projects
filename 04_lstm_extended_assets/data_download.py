from datetime import datetime, timezone, timedelta
import os
import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment

# ── Credentials ────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY    = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

# ── Config ─────────────────────────────────────────────────────────────────────
SYMBOLS = [
    # US Economy sectors:
    "VOX",  # Replaces XLC - Communication Services ETF.
    "XLY",  # Consumer Discretionary ETF.
    "XLP",  # Consumer Staples ETF.
    "XLE",  # Energy ETF.
    "XLF",  # Financials ETF.
    "XLV",  # Health Care ETF.
    "XLI",  # Industrials ETF.
    "XLB",  # Materials ETF.
    "XLRE", # Real Estate ETF.
    "XLK",  # Technology ETF.
    "XLU",  # Utilities ETF.

    # Non-US markets:
    "EEM",  # Emerging Markets ETF.
    "VEA",  # Developed Markets ETF.

    # Fixed income:
    "TLT",  # 20+ Year Treasury Bond ETF.

    # Commodities:
    "GLD",  # Gold ETF.
    "CPER", # Copper ETF.
    "MOO",  # Agriculture ETF.
    "USO",  # Oil ETF.
]

START    = START = "2016-01-01"  
END      = datetime.now(timezone.utc) - timedelta(days=1)  

os.makedirs("data", exist_ok=True)

# ── Download ───────────────────────────────────────────────────────────────────
data_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

req = StockBarsRequest(
    symbol_or_symbols=SYMBOLS,
    timeframe=TimeFrame(1, TimeFrameUnit.Day),
    start=START,
    end=END,
    feed=DataFeed.SIP,
    adjustment=Adjustment.ALL,
)

# ── Build DataFrame ────────────────────────────────────────────────────────────
df = data_client.get_stock_bars(req).df
df = df.reset_index()
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")

# ── Save ───────────────────────────────────────────────────────────────────────
df.to_csv("data/training_data_extended.csv", index=False)