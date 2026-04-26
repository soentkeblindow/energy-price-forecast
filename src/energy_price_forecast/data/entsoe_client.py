import functools
import logging

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

from energy_price_forecast.config import DATA_RAW, get_entsoe_token
from energy_price_forecast.data._entsoe_cache import cached_fetch
from energy_price_forecast.data._entsoe_retry import call_with_retry

logger = logging.getLogger(__name__)

AREA_DE_LU = "DE_LU"
NEIGHBORS = ["FR", "NL", "AT", "PL", "CH", "DK_1"]


@functools.cache
def _get_client() -> EntsoePandasClient:
    return EntsoePandasClient(api_key=get_entsoe_token())


def fetch_day_ahead_prices(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "day_ahead_prices"

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()
        try:
            series = call_with_retry(lambda: client.query_day_ahead_prices(area, s, e))
        except NoMatchingDataError:
            logger.warning("No day-ahead prices from ENTSO-E for %s–%s", s, e)
            return pd.DataFrame(columns=["day_ahead_price"])
        return series.tz_convert("UTC").rename("day_ahead_price").to_frame()

    return cached_fetch(start, end, cache_dir, area, fetch_fn)


def fetch_load(
    start: pd.Timestamp,
    end: pd.Timestamp,
    area: str = AREA_DE_LU,
) -> pd.DataFrame:
    cache_dir = DATA_RAW / "entsoe" / "load"
    _columns = ["load_actual", "load_forecast_day_ahead"]

    def fetch_fn(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
        client = _get_client()

        try:
            actual = (
                call_with_retry(lambda: client.query_load(area, s, e))["Actual Load"]
                .tz_convert("UTC")
                .rename("load_actual")
            )
        except NoMatchingDataError:
            logger.warning("No load actual from ENTSO-E for %s–%s", s, e)
            actual = pd.Series(dtype=float, name="load_actual")

        try:
            forecast = (
                call_with_retry(lambda: client.query_load_forecast(area, s, e))["Forecasted Load"]
                .tz_convert("UTC")
                .rename("load_forecast_day_ahead")
            )
        except NoMatchingDataError:
            logger.warning("No load forecast from ENTSO-E for %s–%s", s, e)
            forecast = pd.Series(dtype=float, name="load_forecast_day_ahead")

        if actual.empty and forecast.empty:
            return pd.DataFrame(columns=_columns)

        # Partial failure: reindex the empty Series to the other's index so that
        # pd.concat always produces a two-column DataFrame. The all-NaN column is
        # cached as-is — the 90%-row coverage check sees the row count as sufficient.
        # To force a re-fetch of later-backfilled ENTSO-E data, delete the cache
        # file manually.
        if actual.empty:
            actual = actual.reindex(forecast.index)
        elif forecast.empty:
            forecast = forecast.reindex(actual.index)

        return pd.concat([actual, forecast], axis=1)

    return cached_fetch(start, end, cache_dir, area, fetch_fn)
