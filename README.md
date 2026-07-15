# Energy Price Forecast — Day-Ahead Germany (DE-LU)

[![CI](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/soentkeblindow/energy-price-forecast/actions/workflows/ci.yml)

Day-ahead electricity price forecasting for the German–Luxembourg bidding zone using walk-forward backtesting, regularised linear baselines, gradient-boosted quantile regression (LightGBM), and classical time series (ARIMAX).

**Status:** Sprint 4 complete — scaled conformal calibration, quantile rearrangement, regime-based diagnostics, and trading-book VaR/ES validated by a full backtest (Kupiec, Christoffersen, Acerbi-Szekely, stratified bootstrap). Sprint 5 (Streamlit dashboard) next.

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
| 4.1 | Realised-market regime tagging (scarcity, high wind, negative price, spikes) | ✅ Complete |
| 4.2 | Regime-based point-accuracy breakdown | ✅ Complete |
| 4.3 | Reliability diagnostics, scaled conformal calibration, quantile rearrangement | ✅ Complete |
| 4.4 | Trading-book VaR/ES (raw / calibrated / FHS), full backtest validation | ✅ Complete |
| 4.5 | Regime & risk deliverable notebook (`04_regime_risk.ipynb`) | ✅ Complete |
| 5 | Streamlit dashboard | 🔄 Next |

## Key Results

All metrics on a common 2021–2025 walk-forward test set (rolling-90 day window, refit_every=1 for LightGBM, refit_every=7 for ARIMAX):

| Model | MAE (EUR/MWh) |
|---|---|
| Seasonal Naive | ~35 |
| Lasso (expanding window) | ~24 |
| ARIMAX rolling-90 | ~23 |
| **LightGBM rolling-90** | **~15.4** |

**Tuning finding:** Optuna-tuned LightGBM ≈ untuned across two independent runs — `_DEFAULT_PARAMS` are the de-facto baseline. The main gain came from shortening the training window (rolling-90 + daily refit), not from hyperparameter search.

**Quantile calibration:** LightGBM q05–q95 bands are not well-calibrated (coverage 0.17/0.81 vs target 0.05/0.95). ARIMAX bands under-cover even more (~50% for the 90% band) as a contrast — motivating the conformal calibration below.

**SHAP attribution:** `residual_load_forecast` dominates LightGBM importance (mean |SHAP| 7.1 EUR/MWh), followed by seasonal features (`month_sin`) and price lags. The tree captures a non-linear merit-order effect invisible to Lasso.

**Feature ablation:** Dropping 3 low-importance forecast-error features (wind offshore, solar, cross-border deviation) yields Δ MAE < 0.1 EUR/MWh — within walk-forward noise. These features can safely be removed for parsimony.

**Conformal calibration:** scaled/normalized per-quantile conformal recalibration + isotonic rearrangement fixes the raw under-coverage: reporting a raw, uncalibrated q05 as a 5% VaR limit breaches in 18% of hours, not 5% (capital buffer 53–76% too small). After calibration, breach rates are 0.049 (long) / 0.052 (short) and Kupiec does not reject the unconditional null.

**But calibration is unconditional, not conditional:** 35 of 60 validated (variant, side, subset) cells still break coverage in an honest, month-stratified day-block bootstrap test. The mechanism is one thing wearing three faces — conditional heteroskedasticity and regime persistence, concentrated in the evening ramp / Dunkelflaute / renewable-surplus — visible directly as multi-day breach clustering (Christoffersen `LR_ind` 53–104). Production recommendation: the directly-evidenced `calibrated` variant over `fhs` (Filtered Historical Simulation), based on the book's actual exposure profile (the evening ramp is a daily event and dominates it).

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

# Conformal calibration + rearrangement, then trading-book VaR/ES and its backtest
uv run python scripts/calibrate_conformal.py
uv run python scripts/compute_risk.py
uv run python scripts/backtest_risk.py
uv run python scripts/backtest_block_sensitivity.py   # block-bootstrap sensitivity grid

# Open results notebooks
jupyter lab notebooks/02_baselines.ipynb          # linear baseline analysis
jupyter lab notebooks/03_model_diagnostics.ipynb  # Sprint-3 full diagnostics
jupyter lab notebooks/04_regime_risk.ipynb        # Sprint-4 calibration + risk deliverable
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
│   ├── 03_model_diagnostics.ipynb   # point comparison, SHAP, fan charts, ablation
│   └── 04_regime_risk.ipynb         # calibration + trading-book VaR/ES, backtest validation
├── src/
│   └── energy_price_forecast/
│       ├── data/         # loaders, ENTSO-E + commodities clients
│       ├── features/     # calendar, fundamentals, lags, availability, subset
│       ├── models/       # SimilarDayNaive, Lasso/Ridge/OLS, LGBMForecaster,
│       │                 #   ARIMAXForecaster, tuning (Optuna)
│       ├── evaluation/   # walk-forward harness, metrics, quantile calibration,
│       │                 #   regimes, breakdown, reliability, conformal,
│       │                 #   rearrangement, residuals, risk, backtest, bootstrap
│       └── dashboard/
├── scripts/
│   ├── build_interim.py
│   ├── build_features.py
│   ├── backtest.py                    # --model {naive,lasso,ridge,ols,lgbm,arimax}
│   ├── tune.py                        # Optuna study + MLflow logging
│   ├── evaluate_quantiles.py
│   ├── evaluate_reliability.py        # reliability curve diagnostics
│   ├── calibrate_conformal.py         # scaled conformal calibration + rearrangement
│   ├── compute_risk.py                # trading-book VaR/ES (raw / calibrated / fhs)
│   ├── backtest_risk.py               # Kupiec/Christoffersen/Acerbi-Szekely + stratified bootstrap
│   └── backtest_block_sensitivity.py  # block-bootstrap sensitivity grid
└── tests/                             # 417 tests, CI green
```

## Disclaimer

For educational and portfolio purposes only. Not financial or trading advice.
