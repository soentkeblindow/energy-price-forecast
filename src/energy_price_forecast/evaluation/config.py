from dataclasses import dataclass

import pandas as pd

from ..market_time import LOCAL_TZ

EXPERIMENT_NAME = "baselines"
SPRINT3_EXPERIMENT_NAME = "sprint3_models"


@dataclass(frozen=True)
class RegimeConfig:
    """Thresholds and boundaries for realised-market regime tagging.

    All flags derived from this config are DIAGNOSTIC labels computed from
    realised (actual) market outcomes. They must never enter the model as
    features (that would be look-ahead leakage). See the
    ``evaluation.regimes`` module docstring for the full contract.
    """

    # Renewable scarcity: residual_load / load above this share.
    # residual_load = load - (wind_onshore + wind_offshore + solar).
    # Equivalent to renewable share < (1 - this value). A self-normalising
    # ratio, so it is robust to installed-capacity growth over time.
    renewable_scarcity_residual_share: float = 0.90

    # High wind: combined onshore + offshore generation above this level.
    # Units are MW (interim generation columns are in MW): 20_000 MW = 20 GW.
    high_wind_generation_mw: float = 20_000.0

    # Price spike: day-ahead price above this per-macro-regime quantile,
    # computed over the full interim history. Regime-relative by design.
    price_spike_quantile: float = 0.95

    # Negative price uses a fixed 0.0 boundary; no configurable field needed.

    # Macro-regime date boundaries, canonical from the EDA. Europe/Berlin
    # local midnight -- identical to FeatureConfig.crisis_start /
    # post_crisis_start (features/config.py) -- so this diagnostic axis
    # matches the is_crisis / is_post_crisis model feature exactly.
    crisis_start: pd.Timestamp = pd.Timestamp("2021-09-01", tz=LOCAL_TZ)
    post_crisis_start: pd.Timestamp = pd.Timestamp("2023-04-01", tz=LOCAL_TZ)
