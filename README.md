# 📈 GBM Stock Price Simulator

An interactive quantitative finance project that investigates whether the assumptions behind the **Geometric Brownian Motion (GBM)** model actually hold on real stock market data.

Rather than only generating Monte Carlo stock price simulations, this project statistically validates the model, compares it against a distribution-free historical bootstrap baseline, and measures forecasting performance through rolling backtesting.

🚀 **Live Demo:** https://gbm-stock-simulator.streamlit.app/

---

## Motivation

Geometric Brownian Motion is one of the most widely used models in quantitative finance and forms the foundation of many pricing and risk management techniques.

A key assumption of GBM is that **log returns are normally distributed**.

This project asks a simple question:

> **Does that assumption actually hold for real stocks?**

To answer it, historical market data is used to:

- Calibrate GBM parameters
- Simulate future price paths
- Statistically test return distributions
- Compare GBM against a historical bootstrap model
- Evaluate prediction accuracy using rolling backtests

---

## Features

✅ Monte Carlo GBM price simulation

✅ Automatic calibration of annualized drift (μ) and volatility (σ)

✅ Historical bootstrap simulation (distribution-free baseline)

✅ Jarque–Bera normality test

✅ QQ Plot diagnostics

✅ Rolling window backtesting

✅ 90% confidence interval coverage analysis

✅ Side-by-side comparison of GBM and Bootstrap models

✅ Interactive Streamlit dashboard

---

## Methodology

```
Historical Prices
        │
        ▼
Compute Log Returns
        │
        ▼
Estimate μ and σ
        │
        ├──────────────► GBM Monte Carlo Simulation
        │
        ▼
Bootstrap Historical Returns
        │
        ▼
Generate Forecast Distribution
        │
        ▼
Rolling Backtesting
        │
        ▼
Coverage & Error Analysis
```

---

## Mathematical Model

GBM assumes that stock prices follow the stochastic differential equation

```
dS = μSdt + σSdW
```

whose discrete-time solution is

```
S(t+Δt) =
S(t) × exp[(μ − σ²/2)Δt + σ√Δt Z]
```

where

- μ = annualized drift
- σ = annualized volatility
- Z ~ N(0,1)

---

## Dashboard

The application contains multiple interactive sections:

### 📊 Simulation

- Select any ticker
- Calibrate μ and σ automatically
- Generate 1000+ Monte Carlo paths
- Visualize forecast distributions

### 📈 Diagnostics

- Histogram of log returns
- QQ Plot
- Jarque–Bera normality test
- Skewness and excess kurtosis

### ⚖ Model Comparison

Compare

- GBM
- Historical Bootstrap

using identical calibration windows.

### 📉 Rolling Backtest

Evaluate forecasting performance over historical windows by measuring

- Actual future price
- Predicted mean
- Prediction interval
- Hit rate
---

## Project Structure

```
gbm-stock-simulator/
│
├── gbm_sim/
│   ├── data.py              # Download historical prices
│   ├── calibration.py       # Estimate μ and σ
│   ├── simulation.py        # GBM Monte Carlo simulation
│   ├── bootstrap.py         # Historical bootstrap model
│   ├── diagnostics.py       # Statistical diagnostics
│   └── backtest.py          # Rolling backtesting engine
│
├── scripts/
│   └── compare_models.py    # GBM vs Bootstrap evaluation
│
├── tests/
│   ├── test_simulation.py
│   ├── test_bootstrap.py
│   ├── test_diagnostics.py
│   └── test_backtest.py
│
├── app.py                   # Streamlit application
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Validation

The project includes automated tests covering

- GBM simulation
- Bootstrap simulation
- Statistical diagnostics
- Rolling backtesting

using **pytest**.

---

## Tech Stack

**Programming**

- Python

**Data Processing**

- NumPy
- pandas

**Statistical Analysis**

- SciPy

**Market Data**

- yfinance

**Visualization**

- Matplotlib
- Streamlit

**Testing**

- pytest

---

## Installation

```bash
git clone https://github.com/AnmolDass961/gbm-stock-simulator.git

cd gbm-stock-simulator

pip install -r requirements.txt

streamlit run app.py
```
