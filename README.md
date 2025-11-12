# quant-finance-proyects

01_lstm_intro.ipynb

This notebook introduces a complete, end-to-end implementation of an LSTM forecasting model in PyTorch. It starts by generating synthetic time-series data with trends, cycles, and noise, then builds an LSTM network to learn one-step-ahead predictions.

### Overview

Key steps include:

1. Constructing custom features and heteroskedastic noise
2. Creating sliding windows for sequence forecasting
3. Training an nn.LSTM model with early stopping and validation tracking
4. Evaluating performance on a held-out test set and visualizing predictions

The goal is to provide a warm-up introduction to LSTMs for time-series forecasting, serving as a clean and well-documented foundation before applying similar architectures to real financial or economic data.

02_lstm_trading_strategy.ipynb

This notebook implements and extends the methodology from  
_Deep Learning for Portfolio Optimization by Zihao Zhang, Stefan Zohren, and Stephen J. Roberts (The Journal of Financial Data Science, 2020, Oxford-Man Institute of Quantitative Finance, University of Oxford)_.  

### Overview

The notebook builds a complete end-to-end trading strategy prototype using deep learning:

1. Data preparation:
   - Imports daily price data for four major asset classes (`VTI`, `AGG`, `DBC`, `VIX`).
   - Cleans, normalizes, and constructs log returns. 
   - Splits data into training, validation, and test periods.

2. Model architecture:
   - Custom LSTM network (`nn.LSTM`) predicting portfolio weights for the next day.  
   - Fully connected output layer with softmax activation to ensure non-negative weights that sum to 1. 
   - Implements a Sharpe-ratio-based loss function directly optimized during training.

3. Training and validation:
   - Uses PyTorch `DataLoader`s and batch training.
   - Tracks training vs. validation Sharpe ratio. 

4. Evaluation and backtesting
   - Computes cumulative portfolio returns and benchmark comparison vs. S&P 500.  
   - Reports annualized Sharpe ratio, daily volatility, and maximum drawdown. 
   - Models transaction costs and daily portfolio turnover to measure net performance.  
   - Visualizes normalized prices, return distributions, and cumulative growth before and after costs.

### Next steps

Planned extensions include:
- Upload and document the Bloomberg data download pipeline.
- Rolling-window and expanding-window backtests  
- Bayesian hyperparameter optimization (Optuna)  
- Integration of risk-free rates in Sharpe loss  

