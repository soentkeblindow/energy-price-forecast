import math

import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.evaluation.metrics import mae, rmse, summarise, wape

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series(*values: float) -> pd.Series:
    return pd.Series(list(values), dtype=float)


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


def test_mae_perfect_forecast() -> None:
    y = _series(10.0, 20.0, 30.0)
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_known_value() -> None:
    # |10-8| + |20-25| + |30-27| = 2 + 5 + 3 = 10; mean = 10/3
    y_true = _series(10.0, 20.0, 30.0)
    y_pred = _series(8.0, 25.0, 27.0)
    assert mae(y_true, y_pred) == pytest.approx(10.0 / 3)


def test_mae_with_negative_prices() -> None:
    y_true = _series(-10.0, -5.0, 5.0)
    y_pred = _series(-8.0, -3.0, 3.0)
    expected = (2.0 + 2.0 + 2.0) / 3
    assert mae(y_true, y_pred) == pytest.approx(expected)


def test_mae_drops_nan_pairs() -> None:
    y_true = _series(10.0, float("nan"), 30.0)
    y_pred = _series(8.0, 99.0, 27.0)
    # Only rows 0 and 2 are valid: (2 + 3) / 2 = 2.5
    assert mae(y_true, y_pred) == pytest.approx(2.5)


def test_mae_nan_on_pred_side_dropped() -> None:
    y_true = _series(10.0, 20.0, 30.0)
    y_pred = _series(8.0, float("nan"), 27.0)
    assert mae(y_true, y_pred) == pytest.approx((2.0 + 3.0) / 2)


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


def test_rmse_known_value() -> None:
    # errors: 2, 5, 3 → squared: 4, 25, 9 → mean: 38/3 → sqrt(38/3)
    y_true = _series(10.0, 20.0, 30.0)
    y_pred = _series(8.0, 25.0, 27.0)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(38.0 / 3))


def test_rmse_perfect_forecast() -> None:
    y = _series(1.0, 2.0, 3.0)
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_drops_nan() -> None:
    y_true = _series(10.0, float("nan"), 10.0)
    y_pred = _series(8.0, 99.0, 6.0)
    # Only rows 0 and 2: errors 2 and 4 → sqrt((4+16)/2) = sqrt(10)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(10.0))


# ---------------------------------------------------------------------------
# WAPE
# ---------------------------------------------------------------------------


def test_wape_known_value() -> None:
    # sum|e| = 2+5+3=10; sum|y_true| = 10+20+30=60; WAPE = 10/60
    y_true = _series(10.0, 20.0, 30.0)
    y_pred = _series(8.0, 25.0, 27.0)
    assert wape(y_true, y_pred) == pytest.approx(10.0 / 60.0)


def test_wape_with_negative_prices() -> None:
    # sum|e| = 2+2; sum|y_true| = |-10|+|10| = 20; WAPE = 4/20 = 0.2
    y_true = _series(-10.0, 10.0)
    y_pred = _series(-8.0, 8.0)
    assert wape(y_true, y_pred) == pytest.approx(0.2)


def test_wape_zero_denominator_returns_nan() -> None:
    y_true = _series(0.0, 0.0)
    y_pred = _series(1.0, 2.0)
    result = wape(y_true, y_pred)
    assert math.isnan(result), "WAPE with zero denominator must return NaN, not inf"


def test_wape_drops_nan() -> None:
    y_true = _series(10.0, float("nan"), 30.0)
    y_pred = _series(8.0, 99.0, 27.0)
    # Only rows 0 and 2: sum|e|=5, sum|y|=40; WAPE=5/40=0.125
    assert wape(y_true, y_pred) == pytest.approx(5.0 / 40.0)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------


def _make_predictions(
    n_days: int = 3,
    hours_per_day: int = 24,
    error: float = 2.0,
) -> pd.DataFrame:
    """Synthetic predictions with a constant absolute error per hour."""
    index = pd.date_range("2024-01-01 23:00", periods=n_days * hours_per_day, freq="h", tz="UTC")
    local = index.tz_convert("Europe/Berlin").normalize()
    days = local.unique().sort_values()
    delivery_day = pd.Series([days[i // hours_per_day] for i in range(len(index))], index=index)
    return pd.DataFrame(
        {
            "y_true": np.full(len(index), 50.0),
            "y_pred": np.full(len(index), 50.0 + error),
            "delivery_day": delivery_day,
        },
        index=index,
    )


def test_summarise_returns_all_keys() -> None:
    preds = _make_predictions()
    result = summarise(preds)
    expected_keys = {
        "mae",
        "rmse",
        "wape",
        "mae_per_day_mean",
        "mae_per_day_std",
        "mae_per_day_p05",
        "mae_per_day_p50",
        "mae_per_day_p95",
    }
    assert set(result.keys()) == expected_keys


def test_summarise_constant_error() -> None:
    error = 3.0
    preds = _make_predictions(n_days=5, error=error)
    result = summarise(preds)

    assert result["mae"] == pytest.approx(error)
    assert result["rmse"] == pytest.approx(error)
    # Per-day MAE is constant across days → std = 0, all quantiles = error
    assert result["mae_per_day_mean"] == pytest.approx(error)
    assert result["mae_per_day_std"] == pytest.approx(0.0, abs=1e-9)
    assert result["mae_per_day_p50"] == pytest.approx(error)


def test_summarise_nan_pairs_excluded() -> None:
    preds = _make_predictions(n_days=3, error=2.0)
    # Corrupt the first row on both sides with NaN
    preds.loc[preds.index[0], "y_pred"] = float("nan")
    result = summarise(preds)
    # Remaining rows still have constant error 2.0; metrics must stay finite
    assert math.isfinite(result["mae"])
    assert result["mae"] == pytest.approx(2.0)
