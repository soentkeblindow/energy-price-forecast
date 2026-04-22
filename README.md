# Energy Price Forecast — Day-Ahead Germany (DE-LU)

Day-ahead electricity price forecasting for the German–Luxembourg bidding zone using LightGBM, quantile regression for prediction intervals, and walk-forward backtesting.

**Status:** Under active development (Sprint 1 — data pipeline setup).

## Motivation

Transparent baseline and LightGBM models for short-term electricity price forecasting, with explicit focus on regime-aware evaluation (dark doldrums, negative prices, price spikes) and risk metrics (Expected Shortfall).

## Planned Stack

Python 3.12+, `uv`, pandas + pyarrow, `entsoe-py`, LightGBM, MLflow, Streamlit, pytest + ruff + mypy, GitHub Actions CI.

## Quickstart

```bash
uv sync
cp .env.example .env   # then fill in your ENTSO-E API key
# download script will follow later in Sprint 1
```

## Project Structure
├── data/
│   ├── raw/          
│   ├── interim/      
│   └── processed/   
├── notebooks/
├── src/
│   └── energy_forecast/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── dashboard/
├── scripts/
└── tests/

## Disclaimer

For educational and portfolio purposes only. Not financial or trading advice.