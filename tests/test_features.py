import pandas as pd
import pytest

from energy_price_forecast.features.calendar import build_calendar_features
from energy_price_forecast.features.config import FeatureConfig  # noqa: F401

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
