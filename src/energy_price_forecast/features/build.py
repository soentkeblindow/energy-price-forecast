from __future__ import annotations

import pandas as pd

from .availability import build_matrix
from .calendar import build_calendar_features
from .config import FeatureConfig
from .fundamentals import build_commodity_features, build_forecast_fundamentals


def build_feature_matrix(df: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Assemble the Sprint-2.3.2 feature subset and leakage-check it.

    Step 2.3.3 appends lags / rolling means / forecast errors to `features`
    before the same build_matrix call. The feature index is the canonical hourly
    UTC index from step 2.1. All feature hyperparameters come from `config`.
    """
    cfg = config if config is not None else FeatureConfig()
    target_index = pd.DatetimeIndex(df.index)
    features = [
        *build_calendar_features(target_index, cfg),
        *build_forecast_fundamentals(df, target_index),
        *build_commodity_features(df, target_index, cfg),
    ]
    return build_matrix(features)  # asserts no leakage, then concatenates the columns
