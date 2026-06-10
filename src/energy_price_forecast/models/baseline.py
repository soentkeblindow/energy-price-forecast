import pandas as pd

from energy_price_forecast.evaluation.walkforward import LOCAL_TZ

# Weekdays (Mon=0 … Sun=6) for which the 24-hour lag applies (Tue–Fri).
# All other weekdays (Mon, Sat, Sun) use the 168-hour (7-day) lag.
_LAG_24H_WEEKDAYS: frozenset[int] = frozenset({1, 2, 3, 4})


class SimilarDayNaive:
    """Same-hour lag forecast: 24 h for Tue–Fri, 168 h (1 week) otherwise.

    A no-fit benchmark after Lago et al. 2021 and the harness smoke test.
    fit is a no-op (no learned state); predict reads the lag from history.
    Hours whose lag is missing from history yield NaN, which the metrics drop.

    DST note: the 168-h lag is a fixed UTC offset. On days immediately after
    a clock change the lag may miss by one local hour, and some timestamps
    (e.g. the extra autumn hour) may be absent 7 days earlier → NaN.
    Accepted benchmark artefact; not over-engineered.
    """

    def fit(self, y_train: pd.Series, x_train: pd.DataFrame | None = None) -> None:
        return None

    def predict(
        self,
        test_index: pd.DatetimeIndex,
        *,
        history: pd.Series,
        x_test: pd.DataFrame | None = None,
    ) -> pd.Series:
        # All hours belong to the same delivery day; weekday is uniform.
        weekday = test_index.tz_convert(LOCAL_TZ)[0].weekday()
        lag = pd.Timedelta(hours=24) if weekday in _LAG_24H_WEEKDAYS else pd.Timedelta(hours=168)
        lagged_utc = test_index - lag
        values = history.reindex(lagged_utc).to_numpy()
        return pd.Series(values, index=test_index, name="y_pred")
