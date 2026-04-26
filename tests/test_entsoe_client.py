"""Unit tests for the ENTSO-E client fetchers.

All tests mock _get_client() — the boundary between our code and the ENTSO-E
library — so only the API class itself is replaced. Cache and retry logic run
against real implementation code.
"""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from entsoe.exceptions import NoMatchingDataError
from freezegun import freeze_time

from energy_price_forecast.data._entsoe_retry import EntsoeFetchError
from energy_price_forecast.data.entsoe_client import fetch_day_ahead_prices, fetch_load

MODULE = "energy_price_forecast.data.entsoe_client"

# call_with_retry uses time.sleep as a default parameter; cached_fetch provides
# no way to inject a custom sleep into fetch_fn. We therefore patch sleep at
# the _entsoe_retry module level rather than using call_with_retry's injectable
# parameter.
SLEEP = "energy_price_forecast.data._entsoe_retry.time.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(start: str, periods: int) -> pd.Series:  # type: ignore[type-arg]
    idx = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.Series(50.0, index=idx)


def _write_month_cache(root: Path, filename: str, series: pd.Series) -> None:  # type: ignore[type-arg]
    cache_dir = root / "entsoe" / "day_ahead_prices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    series.to_frame("day_ahead_price").to_parquet(cache_dir / filename, compression="snappy")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client_mock() -> Generator[MagicMock, None, None]:
    with patch(f"{MODULE}._get_client") as p:
        yield p.return_value


@pytest.fixture
def cache_root(tmp_path: Path) -> Generator[Path, None, None]:
    with patch(f"{MODULE}.DATA_RAW", tmp_path):
        yield tmp_path


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 1: complete month cached → no API call
# ---------------------------------------------------------------------------


def test_cache_hit(client_mock: MagicMock, cache_root: Path) -> None:
    _write_month_cache(cache_root, "DE_LU_2022-01.parquet", _make_series("2022-01-01", 744))

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_not_called()
    assert len(result) == 744
    assert list(result.columns) == ["day_ahead_price"]


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 2: no cached file → API called, file written
# ---------------------------------------------------------------------------


def test_cache_miss_calls_api_and_writes_file(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.return_value = _make_series("2022-01-01", 744)

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()
    assert (cache_root / "entsoe" / "day_ahead_prices" / "DE_LU_2022-01.parquet").exists()
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 3: current month bypasses cache, re-fetches
# ---------------------------------------------------------------------------


@freeze_time("2026-04-15 12:00:00")
def test_current_month_is_always_refetched(client_mock: MagicMock, cache_root: Path) -> None:
    _write_month_cache(cache_root, "DE_LU_2026-04.parquet", _make_series("2026-04-01", 240))
    client_mock.query_day_ahead_prices.return_value = _make_series("2026-04-01", 576)

    fetch_day_ahead_prices(
        pd.Timestamp("2026-04-01", tz="UTC"),
        pd.Timestamp("2026-04-24 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 4: NoMatchingDataError → empty DataFrame with columns
# ---------------------------------------------------------------------------


def test_empty_response_returns_empty_dataframe(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.side_effect = NoMatchingDataError

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert result.empty
    assert list(result.columns) == ["day_ahead_price"]


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 5: 3 transient failures then success → correct result
# ---------------------------------------------------------------------------


def test_retry_on_transient_error(client_mock: MagicMock, cache_root: Path) -> None:
    with patch(SLEEP):
        client_mock.query_day_ahead_prices.side_effect = [
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
            _make_series("2022-01-01", 744),
        ]

        result = fetch_day_ahead_prices(
            pd.Timestamp("2022-01-01", tz="UTC"),
            pd.Timestamp("2022-01-31 23:00", tz="UTC"),
        )

    assert client_mock.query_day_ahead_prices.call_count == 4
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 6: permanent failure → EntsoeFetchError raised
# ---------------------------------------------------------------------------


def test_retry_exhaustion_raises_fetch_error(client_mock: MagicMock, cache_root: Path) -> None:
    with patch(SLEEP):
        client_mock.query_day_ahead_prices.side_effect = requests.ConnectionError()

        with pytest.raises(EntsoeFetchError):
            fetch_day_ahead_prices(
                pd.Timestamp("2022-01-01", tz="UTC"),
                pd.Timestamp("2022-01-31 23:00", tz="UTC"),
            )


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 7: 3-month range → 3 API calls, concatenated result
# ---------------------------------------------------------------------------


def test_chunk_aggregation(client_mock: MagicMock, cache_root: Path) -> None:
    client_mock.query_day_ahead_prices.side_effect = [
        _make_series("2022-01-01", 744),  # January:  31 days × 24 h
        _make_series("2022-02-01", 672),  # February: 28 days × 24 h
        _make_series("2022-03-01", 744),  # March:    31 days × 24 h
    ]

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-03-31 23:00", tz="UTC"),
    )

    assert client_mock.query_day_ahead_prices.call_count == 3
    assert len(result) == 744 + 672 + 744


# ---------------------------------------------------------------------------
# fetch_day_ahead_prices — Test 8: <90% rows for complete month → re-fetch
# ---------------------------------------------------------------------------


def test_incomplete_cache_triggers_refetch(client_mock: MagicMock, cache_root: Path) -> None:
    # 240 rows (10 days) is well below 90% of the 744 hours expected for January
    _write_month_cache(cache_root, "DE_LU_2022-01.parquet", _make_series("2022-01-01", 240))
    client_mock.query_day_ahead_prices.return_value = _make_series("2022-01-01", 744)

    result = fetch_day_ahead_prices(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    client_mock.query_day_ahead_prices.assert_called_once()
    assert len(result) == 744


# ---------------------------------------------------------------------------
# fetch_load — Test 1: both APIs return data → two-column DataFrame
# ---------------------------------------------------------------------------


def test_fetch_load_returns_both_columns(client_mock: MagicMock, cache_root: Path) -> None:
    idx = pd.date_range("2022-01-01", periods=744, freq="h", tz="UTC")
    client_mock.query_load.return_value = pd.DataFrame({"Actual Load": 30_000.0}, index=idx)
    client_mock.query_load_forecast.return_value = pd.DataFrame(
        {"Forecasted Load": 31_000.0}, index=idx
    )

    result = fetch_load(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert list(result.columns) == ["load_actual", "load_forecast_day_ahead"]
    assert len(result) == 744
    assert result["load_actual"].notna().all()
    assert result["load_forecast_day_ahead"].notna().all()


# ---------------------------------------------------------------------------
# fetch_load — Test 2: one API fails → other column NaN, schema preserved
# ---------------------------------------------------------------------------


def test_fetch_load_preserves_schema_on_partial_failure(
    client_mock: MagicMock, cache_root: Path
) -> None:
    idx = pd.date_range("2022-01-01", periods=744, freq="h", tz="UTC")
    client_mock.query_load.return_value = pd.DataFrame({"Actual Load": 30_000.0}, index=idx)
    client_mock.query_load_forecast.side_effect = NoMatchingDataError

    result = fetch_load(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-31 23:00", tz="UTC"),
    )

    assert list(result.columns) == ["load_actual", "load_forecast_day_ahead"]
    assert len(result) == 744
    assert result["load_actual"].notna().all()
    assert result["load_forecast_day_ahead"].isna().all()


# ---------------------------------------------------------------------------
# Integration test — real API call, excluded from CI by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_day_ahead_prices_live() -> None:
    """Single real ENTSO-E API call for local verification only."""
    result = fetch_day_ahead_prices(
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-01 23:00", tz="UTC"),
    )
    assert not result.empty
    assert "day_ahead_price" in result.columns
    assert result.index.tz is not None
