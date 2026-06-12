import numpy as np
import pandas as pd
import pytest

from energy_price_forecast.features.availability import LeakageError, assert_no_leakage
from energy_price_forecast.features.build import build_feature_matrix
from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig  # noqa: F401
from energy_price_forecast.features.fundamentals import (
    build_commodity_features,
    build_forecast_fundamentals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_utc(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


def test_feature_config_defaults() -> None:
    cfg = FeatureConfig()
    assert cfg.commodity_lag_hours == 48
    assert cfg.crisis_start == pd.Timestamp("2021-09-01", tz="Europe/Berlin")
    assert cfg.post_crisis_start == pd.Timestamp("2023-04-01", tz="Europe/Berlin")


# ---------------------------------------------------------------------------
# Cyclic calendar features
# ---------------------------------------------------------------------------


def test_hour_sin_zero_at_local_midnight() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET (local hour 0); sin(0) = 0
    idx = _hourly_utc("2024-01-14 23:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["hour_sin"].values.iloc[0] == pytest.approx(0.0)


def test_hour_sin_one_at_local_hour_6() -> None:
    # 2024-01-15 05:00 UTC = 2024-01-15 06:00 CET; sin(2π*6/24) = sin(π/2) = 1
    idx = _hourly_utc("2024-01-15 05:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["hour_sin"].values.iloc[0] == pytest.approx(1.0, abs=1e-10)


def test_cyclic_values_in_range() -> None:
    idx = _hourly_utc("2024-01-01 00:00", 24 * 7)
    feats = {f.name: f for f in build_calendar_features(idx)}
    for name in ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos"):
        vals = feats[name].values
        assert vals.min() >= -1.0 - 1e-12 and vals.max() <= 1.0 + 1e-12, f"{name} out of [-1, 1]"


def test_month_cos_january_is_one() -> None:
    # January: position 0; cos(2π*0/12) = 1
    idx = _hourly_utc("2024-01-01 00:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["month_cos"].values.iloc[0] == pytest.approx(1.0)


def test_weekday_cos_monday_is_one() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET = Monday (dayofweek 0); cos(0) = 1
    idx = _hourly_utc("2024-01-14 23:00", 1)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["weekday_cos"].values.iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# is_weekend
# ---------------------------------------------------------------------------


def test_is_weekend_saturday() -> None:
    # 2024-01-06 00:00 UTC = 2024-01-06 01:00 CET = Saturday
    idx = _hourly_utc("2024-01-06 00:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_weekend"].values == 1).all()


def test_is_weekend_wednesday() -> None:
    # 2024-01-09 23:00 UTC = 2024-01-10 00:00 CET = Wednesday
    idx = _hourly_utc("2024-01-09 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_weekend"].values == 0).all()


# ---------------------------------------------------------------------------
# is_holiday (nationwide) and is_regional_holiday
# ---------------------------------------------------------------------------


def test_is_holiday_tag_der_deutschen_einheit() -> None:
    # 2024-10-02 22:00 UTC = 2024-10-03 00:00 CEST = Oct 3 (nationwide)
    idx = _hourly_utc("2024-10-02 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_holiday"].values == 1).all()


def test_is_holiday_zero_on_fronleichnam() -> None:
    # Fronleichnam is regional only; 2024-05-29 22:00 UTC = 2024-05-30 00:00 CEST
    idx = _hourly_utc("2024-05-29 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_holiday"].values == 0).all()


def test_is_regional_holiday_fronleichnam_2024() -> None:
    # Easter 2024 = 2024-03-31; +60d = 2024-05-30 (Fronleichnam)
    idx = _hourly_utc("2024-05-29 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 1).all()


def test_is_regional_holiday_allerheiligen() -> None:
    # 2024-10-31 23:00 UTC = 2024-11-01 00:00 CET = Allerheiligen
    idx = _hourly_utc("2024-10-31 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 1).all()


def test_is_regional_holiday_zero_on_nationwide_holiday() -> None:
    # Oct 3 is nationwide; regional flag must be 0
    idx = _hourly_utc("2024-10-02 22:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 0).all()


def test_is_regional_holiday_zero_on_normal_weekday() -> None:
    # 2024-01-14 23:00 UTC = 2024-01-15 00:00 CET = Monday, no holiday
    idx = _hourly_utc("2024-01-14 23:00", 24)
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert (feats["is_regional_holiday"].values == 0).all()


# ---------------------------------------------------------------------------
# Local-time correctness
# ---------------------------------------------------------------------------


def test_local_time_near_utc_midnight() -> None:
    # 2024-01-07 23:30 UTC = 2024-01-08 00:30 CET → Monday (is_weekend = 0)
    # Without TZ conversion Jan 7 UTC is Sunday → is_weekend would be 1 (wrong)
    idx = pd.DatetimeIndex(["2024-01-07 23:30"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_weekend"].values.iloc[0] == 0


# ---------------------------------------------------------------------------
# Regime dummies
# ---------------------------------------------------------------------------


def test_regime_before_crisis() -> None:
    # 2021-08-31 21:59 UTC = 2021-08-31 23:59 CEST → local day 2021-08-31 → calm
    idx = pd.DatetimeIndex(["2021-08-31 21:59"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 0
    assert feats["is_post_crisis"].values.iloc[0] == 0


def test_regime_at_crisis_start() -> None:
    # 2021-08-31 22:00 UTC = 2021-09-01 00:00 CEST → crisis_start boundary → is_crisis = 1
    idx = pd.DatetimeIndex(["2021-08-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 1
    assert feats["is_post_crisis"].values.iloc[0] == 0


def test_regime_at_post_crisis_start() -> None:
    # 2023-04-01 00:00 CEST = 2023-03-31 22:00 UTC → post_crisis boundary
    idx = pd.DatetimeIndex(["2023-03-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx)}
    assert feats["is_crisis"].values.iloc[0] == 0
    assert feats["is_post_crisis"].values.iloc[0] == 1


def test_regime_custom_config_boundaries() -> None:
    cfg = FeatureConfig(
        crisis_start=pd.Timestamp("2022-01-01", tz="Europe/Berlin"),
        post_crisis_start=pd.Timestamp("2023-01-01", tz="Europe/Berlin"),
    )
    # 2022-06-01 00:00 CEST = 2022-05-31 22:00 UTC → inside crisis window
    idx = pd.DatetimeIndex(["2022-05-31 22:00"], tz="UTC")
    feats = {f.name: f for f in build_calendar_features(idx, cfg)}
    assert feats["is_crisis"].values.iloc[0] == 1
    assert feats["is_post_crisis"].values.iloc[0] == 0


# ---------------------------------------------------------------------------
# Forecast fundamentals
# ---------------------------------------------------------------------------


def _make_fc_df(periods: int = 48) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "solar_forecast": np.full(periods, 5000.0),
        },
        index=idx,
    )
    return df, idx


def test_residual_load_forecast_value() -> None:
    df, idx = _make_fc_df()
    feats = {f.name: f for f in build_forecast_fundamentals(df, idx)}
    expected = 40000.0 - 8000.0 - 2000.0 - 5000.0
    assert feats["residual_load_forecast"].values.to_numpy() == pytest.approx(expected)


def test_renewable_share_forecast_value() -> None:
    df, idx = _make_fc_df()
    feats = {f.name: f for f in build_forecast_fundamentals(df, idx)}
    expected = (8000.0 + 2000.0 + 5000.0) / 40000.0
    assert feats["renewable_share_forecast"].values.to_numpy() == pytest.approx(expected)


def test_forecast_fundamentals_pass_leakage() -> None:
    df, idx = _make_fc_df()
    assert_no_leakage(build_forecast_fundamentals(df, idx))


# ---------------------------------------------------------------------------
# Commodity features
# ---------------------------------------------------------------------------


def _make_commodity_df(periods: int = 5 * 24) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    eua_vals = np.full(periods, 70.0)
    eua_vals[:24] = np.nan  # first 24 hours NaN (pre-EUA placeholder)
    df = pd.DataFrame(
        {
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": eua_vals,
        },
        index=idx,
    )
    return df, idx


def test_commodity_features_pass_leakage() -> None:
    df, idx = _make_commodity_df()
    assert_no_leakage(build_commodity_features(df, idx))


def test_eua_missing_flag_where_nan() -> None:
    df, idx = _make_commodity_df()
    feats = {f.name: f for f in build_commodity_features(df, idx)}
    # target[48] -> source[0] (NaN eua) -> missing = 1
    assert feats["eua_missing"].values.iloc[48] == 1
    # target[72] -> source[24] (eua = 70.0) -> missing = 0
    assert feats["eua_missing"].values.iloc[72] == 0


def test_commodity_feature_name_from_config() -> None:
    cfg = FeatureConfig(commodity_lag_hours=72)
    df, idx = _make_commodity_df(periods=7 * 24)
    feats = {f.name: f for f in build_commodity_features(df, idx, cfg)}
    assert "ttf_gas_lag_72h" in feats
    assert "eua_co2_lag_72h" in feats


def test_commodity_lag_24h_fails_leakage() -> None:
    cfg = FeatureConfig(commodity_lag_hours=24)
    df, idx = _make_commodity_df()
    with pytest.raises(LeakageError):
        assert_no_leakage(build_commodity_features(df, idx, cfg))


# ---------------------------------------------------------------------------
# build_feature_matrix — wiring (all builders together, synthetic data)
# ---------------------------------------------------------------------------


def _make_full_df(periods: int = 5 * 24) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "load_forecast_day_ahead": np.full(periods, 40000.0),
            "wind_onshore_forecast": np.full(periods, 8000.0),
            "wind_offshore_forecast": np.full(periods, 2000.0),
            "solar_forecast": np.full(periods, 5000.0),
            "ttf_gas_eur_per_mwh": np.full(periods, 30.0),
            "eua_co2_eur_per_t": np.full(periods, 70.0),
        },
        index=idx,
    )


def test_build_feature_matrix_passes_leakage_and_shape() -> None:
    df = _make_full_df()
    x = build_feature_matrix(df)
    assert len(x) == len(df)


def test_build_feature_matrix_expected_columns() -> None:
    df = _make_full_df()
    x = build_feature_matrix(df)
    expected = {
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_holiday",
        "is_regional_holiday",
        "is_crisis",
        "is_post_crisis",
        "load_forecast_day_ahead",
        "wind_onshore_forecast",
        "wind_offshore_forecast",
        "solar_forecast",
        "residual_load_forecast",
        "renewable_share_forecast",
        "ttf_gas_lag_48h",
        "eua_co2_lag_48h",
        "eua_missing",
    }
    assert set(x.columns) == expected
