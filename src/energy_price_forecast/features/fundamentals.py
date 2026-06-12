from __future__ import annotations

import pandas as pd

from .availability import Feature, combine, forecast_for_target


def build_forecast_fundamentals(df: pd.DataFrame, target_index: pd.DatetimeIndex) -> list[Feature]:
    """Day-ahead forecast fundamentals for the target day (all DA_FORECAST).

    The raw forecasts are used directly; residual load and renewable share are
    derived via `combine` (knowledge time = max over the forecasts = gate closure).
    Leakage-correct replacement for the actuals-based EDA quantities. Needs no
    config (no tunable knobs here).
    """
    load = forecast_for_target(
        "load_forecast_day_ahead",
        df["load_forecast_day_ahead"],
        "load_forecast_day_ahead",
        target_index=target_index,
    )
    won = forecast_for_target(
        "wind_onshore_forecast",
        df["wind_onshore_forecast"],
        "wind_onshore_forecast",
        target_index=target_index,
    )
    woff = forecast_for_target(
        "wind_offshore_forecast",
        df["wind_offshore_forecast"],
        "wind_offshore_forecast",
        target_index=target_index,
    )
    solar = forecast_for_target(
        "solar_forecast",
        df["solar_forecast"],
        "solar_forecast",
        target_index=target_index,
    )
    residual = combine(
        "residual_load_forecast",
        [load, won, woff, solar],
        lambda load_, won_, woff_, solar_: load_ - won_ - woff_ - solar_,
    )
    share = combine(
        "renewable_share_forecast",
        [load, won, woff, solar],
        lambda load_, won_, woff_, solar_: (won_ + woff_ + solar_) / load_,
    )
    return [load, won, woff, solar, residual, share]
