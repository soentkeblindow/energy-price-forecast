# Energy Price Forecast — Day-Ahead Germany (DE-LU)

[![CI](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml)

Day-ahead electricity price forecasting for the German–Luxembourg bidding zone using walk-forward backtesting, regularised linear baselines, and (upcoming) LightGBM with SHAP attribution.

**Status:** Sprint 2 complete — linear baselines evaluated. Sprint 3 (LightGBM) in progress.

## Motivation

Transparent, leakage-free forecasting pipeline for short-term electricity prices, with explicit focus on regime-aware evaluation (energy crisis, negative prices, price spikes) and interpretable feature importance.

## Stack

Python 3.12+, `uv`, pandas + pyarrow, `entsoe-py`, scikit-learn, LightGBM, MLflow, Streamlit, pytest + ruff + mypy, GitHub Actions CI.

## Progress

| Sprint | Scope | Status |
|---|---|---|
| 1 | Data pipeline (ENTSO-E + commodities), EDA | ✅ Complete |
| 2.2 | Walk-forward evaluation harness, leakage tests | ✅ Complete |
| 2.3 | Feature engineering (calendar, fundamentals, lags, cross-border) | ✅ Complete |
| 2.4 | Linear baselines (Lasso, Ridge, OLS), MLflow tracking, `notebooks/02_baselines.ipynb` | ✅ Complete |
| 3 | LightGBM, SHAP attribution, quantile regression | 🔄 Next |

**Key results (Sprint 2):** All three linear models outperform the rule-based benchmark by ~30% MAE on 5 years of walk-forward test data (2021–2025). Lasso is selected as the linear baseline — L1 regularisation eliminates 12 of 34 engineered features, confirming that renewable share forecast and day-ahead load forecast are the dominant price drivers under linear assumptions.

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in your ENTSO-E API key

# Build feature matrix
uv run python scripts/build_interim.py
uv run python scripts/build_features.py

# Run backtest
uv run python scripts/backtest.py --model lasso --target-transform identity --test-start 2021-01-01

# Open results notebook
jupyter lab notebooks/02_baselines.ipynb
```

## Project Structure

```text
├── data/
│   ├── raw/
│   ├── interim/          # hourly price + fundamental data
│   └── processed/        # feature matrix + backtest results
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_baselines.ipynb
├── src/
│   └── energy_price_forecast/
│       ├── data/         # loaders, ENTSO-E + commodities clients
│       ├── features/     # calendar, fundamentals, lags, availability
│       ├── models/       # SimilarDayNaive, LassoForecaster, RidgeForecaster, OLSForecaster
│       ├── evaluation/   # walk-forward harness, metrics, MLflow config
│       └── dashboard/
├── scripts/
│   ├── build_interim.py
│   ├── build_features.py
│   └── backtest.py
└── tests/                # 163 tests, CI green
```

## Disclaimer

For educational and portfolio purposes only. Not financial or trading advice.
