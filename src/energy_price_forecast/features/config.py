from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..market_time import LOCAL_TZ


@dataclass(frozen=True)
class FeatureConfig:
    """Central, immutable configuration of feature hyperparameters.

    One place for every tunable number in the feature pipeline. Passed into the
    builders (a default instance is used when none is given). Step 2.3.3 extends
    this with its own lag and rolling-window settings, so all feature knobs stay
    collected here rather than scattered across modules.
    """

    # Commodities: 48h is the minimum safe lag (lands on <= D-2 at gate closure).
    commodity_lag_hours: int = 48

    # Crisis-regime boundaries (Europe/Berlin local dates), from the EDA (3.0, D2).
    crisis_start: pd.Timestamp = pd.Timestamp("2021-09-01", tz=LOCAL_TZ)
    post_crisis_start: pd.Timestamp = pd.Timestamp("2023-04-01", tz=LOCAL_TZ)

    # --- Added in step 2.3.3 (placeholders; not used in 2.3.2) ---
    # price_lags_hours: tuple[int, ...] = (24, 48, 168)
    # actual_lags_hours: tuple[int, ...] = (48, 72, 168)
    # rolling_windows_hours: tuple[int, ...] = (24, 168)
