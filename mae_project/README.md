
# MAE-Based Portfolio Allocation System

A research and trading pipeline for ETF portfolio allocation using Masked Autoencoder (MAE) representations.

This project includes:
- Data collection and preprocessing
- Representation learning (MAE pretraining)
- Portfolio weight prediction models
- Backtesting framework
- Live trading execution

The system supports end-to-end experimentation from raw financial data to live portfolio execution.

---

# Project Structure

mae_project/

│
├── 1_train/                # Model training pipeline
│   ├── 01_download_data.py
│   ├── 02_revise_data.py
│   ├── 03_process_data.py
│   ├── 04_train_mae.py
│   ├── 05_train_return.py
│   ├── 05_train_sharpe.py
│   ├── 06_infer_return.py
│   └── 06_infer_sharpe.py
│
├── 2_backtest/             # Backtesting trading engine
│   ├── backtest_runner.py
│   ├── gateway.py
│   ├── matching_engine.py
│   ├── order_book.py
│   ├── order_manager.py
│   └── strategy_ml_weights.py
│
├── 3_live_trading/         # Live trading infrastructure
│   ├── alpaca_client.py
│   ├── live_config.py
│   ├── live_data.py
│   ├── live_logging.py
│   ├── live_runner.py
│   └── live_state.py
│
├── data/                   # Market and macroeconomic data
│
├── backtest_outputs/       # Backtest results and logs
│   ├── metrics.json
│   ├── equity_curve.png
│   ├── trade_log.csv
│   ├── order_log.csv
│   └── daily_log.csv
│
├── runs/       # live trading results and logs
│   ├── 2026-03-09
│   ├── eventa_log.jsonl
│   ├── live_state.json
│
├── project_proof.mp4       # live trading proof
│
└── README.md

---

# System Overview

The system has three main modules:

## 1. Model Training

Located in `1_train/`.

Pipeline:

Download Data → Data Cleaning → Feature Processing → MAE Pretraining → Portfolio Model Training → Inference

Key scripts:

- 01_download_data.py — download ETF and macro data
- 02_revise_data.py — clean and align data
- 03_process_data.py — generate model features
- 04_train_mae.py — train masked autoencoder
- 05_train_return.py — train return prediction model
- 05_train_sharpe.py — train Sharpe-ratio optimized model
- 06_infer_return.py — inference using return model
- 06_infer_sharpe.py — inference using Sharpe model

---

## 2. Backtesting Engine

Located in `2_backtest/`.

Components:

- backtest_runner.py — main backtest entry
- strategy_ml_weights.py — ML-based trading strategy
- gateway.py — strategy/execution interface
- order_manager.py — order lifecycle management
- order_book.py — simulated order book
- matching_engine.py — trade execution

Backtest outputs:

- equity curve
- trade logs
- order logs
- performance metrics

Results are stored in:

backtest_outputs/

---

## 3. Live Trading

Located in `3_live_trading/`.

Main components:

- live_runner.py — main trading loop
- live_data.py — market data ingestion
- live_state.py — run state management
- live_logging.py — logging utilities
- live_config.py — runtime configuration
- alpaca_client.py — broker API interface

The system supports daily portfolio rebalancing based on model predictions.

Results are stored in:

runs/

---

# Data

The project uses:

Market Data:
SPY
TLT
GLD
UUP
XLE

Macroeconomic Data from FRED:
- CPI
- Unemployment
- Interest rates
- VIX
- GDP

All processed datasets are stored in:

data/

---

# Model Design

Input tensor shape:

[B, S, W, F]

Where:

B = batch size  
S = trajectory length  
W = rolling window length  
F = feature dimension  

The model treats S × W as tokens similar to patches in a Vision Transformer.

Architecture:

Input → Linear Projection → Positional Embedding → Transformer Encoder → Temporal Pooling → MLP Head → Portfolio Weights (Softmax)

Output shape:

[B, S, N_assets]

The last timestep allocation is used for trading.

---

# Training Objectives

Two objectives are implemented.

Return Optimization

Maximize next-day portfolio return.

Sharpe Ratio Optimization

Loss = -Sharpe Ratio + turnover penalty + smoothness penalty - entropy bonus

This encourages stable allocations and lower turnover.

---

# Backtest Metrics

Typical metrics include:

- Sharpe Ratio
- Annualized Return
- Volatility
- Maximum Drawdown
- Turnover

Metrics are saved in:

backtest_outputs/metrics.json

---

# Running the Pipeline

1. Download data

python 1_train/01_download_data.py

2. Process data

python 1_train/02_revise_data.py
python 1_train/03_process_data.py

3. Train MAE

python 1_train/04_train_mae.py

4. Train portfolio model

python 1_train/05_train_sharpe.py

5. Run inference

python 1_train/06_infer_sharpe.py

6. Run backtest

python 2_backtest/backtest_runner.py

7. Run live trading

python 3_live_trading/live_runner.py

---

# Dependencies

Python 3.10+

Libraries:

- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- yfinance
- pandas-datareader

---

## Extra: Live Trading proof

It's in project_proof.mp4