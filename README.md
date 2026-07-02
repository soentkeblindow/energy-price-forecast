# Energy Price Forecast — Day-Ahead Germany (DE-LU)

[![CI](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml)

Day-ahead electricity price forecasting for the German–Luxembourg bidding zone using walk-forward backtesting, regularised linear baselines, gradient-boosted quantile regression (LightGBM), and classical time series (ARIMAX).

**Status:** Sprint 3 complete — LightGBM, ARIMAX, quantile calibration, SHAP attribution, and feature ablation evaluated. Sprint 4 (conformal calibration, regime breakdown) next.

## Motivation

Transparent, leakage-free forecasting pipeline for short-term electricity prices, with explicit focus on regime-aware evaluation (energy crisis, negative prices, price spikes) and interpretable feature importance.

## Stack

Python 3.12+, `uv`, pandas + pyarrow, `entsoe-py`, scikit-learn, LightGBM, statsmodels (SARIMAX), SHAP, MLflow, pytest + ruff + mypy, GitHub Actions CI.

## Progress

| Sprint | Scope | Status |
|---|---|---|
| 1 | Data pipeline (ENTSO-E + commodities), EDA | ✅ Complete |
| 2.2 | Walk-forward evaluation harness, leakage tests | ✅ Complete |
| 2.3 | Feature engineering (calendar, fundamentals, lags, cross-border) | ✅ Complete |
| 2.4 | Linear baselines (Lasso, Ridge, OLS), MLflow tracking | ✅ Complete |
| 3.1 | LightGBM quantile forecaster | ✅ Complete |
| 3.2 | Optuna hyperparameter tuning | ✅ Complete |
| 3.3 | Quantile calibration evaluation | ✅ Complete |
| 3.4 | ARIMAX (AR(2) + daily Fourier + exog) | ✅ Complete |
| 3.5 | SHAP attribution, feature ablation, diagnostics notebook | ✅ Complete |
| 4 | Conformal calibration, regime breakdown, Expected Shortfall | 🔄 Next |

## Key Results

All metrics on a common 2021–2025 walk-forward test set (rolling-90 day window, refit_every=1 for LightGBM, refit_every=7 for ARIMAX):

| Model | MAE (EUR/MWh) |
|---|---|
| Seasonal Naive | ~35 |
| Lasso (expanding window) | ~24 |
| ARIMAX rolling-90 | ~23 |
| **LightGBM rolling-90** | **~15.4** |

**Tuning finding:** Optuna-tuned LightGBM ≈ untuned across two independent runs — `_DEFAULT_PARAMS` are the de-facto baseline. The main gain came from shortening the training window (rolling-90 + daily refit), not from hyperparameter search.

**Quantile calibration:** LightGBM q05–q95 bands are not well-calibrated (coverage 0.17/0.81 vs target 0.05/0.95). ARIMAX bands under-cover even more (~50% for the 90% band) as a contrast. Quantile kalibrating is needed.

**SHAP attribution:** `residual_load_forecast` dominates LightGBM importance (mean |SHAP| 7.1 EUR/MWh), followed by seasonal features (`month_sin`) and price lags. The tree captures a non-linear merit-order effect invisible to Lasso.

**Feature ablation:** Dropping 3 low-importance forecast-error features (wind offshore, solar, cross-border deviation) yields Δ MAE < 0.1 EUR/MWh — within walk-forward noise. These features can safely be removed for parsimony.

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in your ENTSO-E API key

# Build feature matrix
uv run python scripts/build_interim.py
uv run python scripts/build_features.py

# Run backtests
uv run python scripts/backtest.py --model lgbm --alpha 0.5 \
  --window rolling --train-span-days 90 --refit-every 1 \
  --test-start 2021-01-01 --out data/processed/preds_lgbm_q50.parquet

# Open results notebooks
jupyter lab notebooks/02_baselines.ipynb       # linear baseline analysis
jupyter lab notebooks/03_model_diagnostics.ipynb  # Sprint-3 full diagnostics
```

## Project Structure

```text
├── data/
│   ├── raw/
│   ├── interim/          # hourly price + fundamental data
│   └── processed/        # feature matrix + backtest results
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_model_diagnostics.ipynb   # point comparison, SHAP, fan charts, ablation
├── src/
│   └── energy_price_forecast/
│       ├── data/         # loaders, ENTSO-E + commodities clients
│       ├── features/     # calendar, fundamentals, lags, availability, subset
│       ├── models/       # SimilarDayNaive, Lasso/Ridge/OLS, LGBMForecaster,
│       │                 #   ARIMAXForecaster, tuning (Optuna)
│       ├── evaluation/   # walk-forward harness, metrics, quantile calibration
│       └── dashboard/
├── scripts/
│   ├── build_interim.py
│   ├── build_features.py
│   ├── backtest.py          # --model {naive,lasso,ridge,ols,lgbm,arimax}
│   ├── tune.py              # Optuna study + MLflow logging
│   └── evaluate_quantiles.py
└── tests/                   # 244 tests, CI green
```

## Disclaimer

For educational and portfolio purposes only. Not financial or trading advice.
