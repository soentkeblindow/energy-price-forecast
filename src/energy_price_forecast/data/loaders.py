import logging

import pandas as pd

from energy_price_forecast.data.commodities_client import fetch_eua_co2, fetch_ttf_gas
from energy_price_forecast.data.entsoe_client import (
    AREA_DE_LU,
    fetch_cross_border_flows,
    fetch_day_ahead_prices,
    fetch_generation_by_type,
    fetch_load,
    fetch_scheduled_exchanges,
    fetch_wind_solar_forecast_actual,
)

logger = logging.getLogger(__name__)

_COMMODITY_COLUMNS = ["ttf_gas_eur_per_mwh", "eua_co2_eur_per_t"]
# 4 days = 96 hours: bridges weekends (2 days) and the longest common holiday gap
# (Good Friday to Easter Monday = 4 days). Genuine outages ≥ 5 days stay visible as NaN.
_COMMODITY_FFILL_LIMIT = 4 * 24


def _log_fetch(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logger.info("%s: 0 rows (empty)", name)
    else:
        logger.info("%s: %d rows (%s to %s)", name, len(df), df.index.min(), df.index.max())
    return df


def load_all_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    """Load and merge all data sources into a single hourly DataFrame.

    Calls all six ENTSO-E fetchers and both commodities fetchers, then merges
    them on the hourly UTC index using an outer join. The day-ahead price
    column is required: rows where the price is missing are dropped.

    Commodity prices (TTF gas, EUA CO2) come at daily granularity and are
    forward-filled to hourly resolution with a 4-day limit. This bridges
    weekends and typical holiday gaps (Easter, Christmas) but leaves
    multi-day outages visible as NaN for downstream data quality analysis.
    This is the only resampling done by the loader; all other forward-fill
    or interpolation decisions are deferred to feature engineering (Sprint 2).

    Parameters
    ----------
    start : pd.Timestamp
        Start of the requested time range. Converted to UTC at entry.
    end : pd.Timestamp
        End of the requested time range (inclusive). Converted to UTC.
    area : str, default "DE_LU"
        ENTSO-E area code. Currently only DE_LU is fully supported; other
        areas would require neighbor-list adjustments in entsoe_client.py.

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with UTC DatetimeIndex. Columns include the
        day-ahead price (target), all load and renewable forecasts/actuals,
        all generation types, all six neighbor scheduled exchanges and
        physical flows, plus the two commodity prices forward-filled to
        hourly. Columns missing from individual fetchers are not synthesised.
    """
    start_utc = start.tz_convert("UTC") if start.tzinfo is not None else start.tz_localize("UTC")
    end_utc = end.tz_convert("UTC") if end.tzinfo is not None else end.tz_localize("UTC")

    if start_utc >= end_utc:
        raise ValueError(f"start must be before end, got start={start_utc!r}, end={end_utc!r}")

    frames = [
        _log_fetch(
            "fetch_day_ahead_prices",
            fetch_day_ahead_prices(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_load",
            fetch_load(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_wind_solar_forecast_actual",
            fetch_wind_solar_forecast_actual(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_generation_by_type",
            fetch_generation_by_type(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_scheduled_exchanges",
            fetch_scheduled_exchanges(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_cross_border_flows",
            fetch_cross_border_flows(start_utc, end_utc, area),
        ),
        _log_fetch(
            "fetch_ttf_gas",
            fetch_ttf_gas(start_utc, end_utc),
        ),
        _log_fetch(
            "fetch_eua_co2",
            fetch_eua_co2(start_utc, end_utc),
        ),
    ]

    df = pd.concat(frames, axis=1, join="outer")

    # Forward-fill commodity columns only (daily → hourly granularity).
    # All other gaps remain as NaN so EDA can see real data-quality issues.
    for col in _COMMODITY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].ffill(limit=_COMMODITY_FFILL_LIMIT)

    # Rows without a target price are unusable for modelling; drop them early
    # so they don't distort predictor gap-visualisation in EDA.
    df = df[df["day_ahead_price"].notna()]

    dt_index = pd.DatetimeIndex(df.index)
    assert dt_index.tz is not None, "merged index must be timezone-aware"
    assert str(dt_index.tz) == "UTC", f"merged index timezone must be UTC, got {dt_index.tz!r}"

    logger.info("merged result: %d rows × %d columns", len(df), len(df.columns))

    return df
