from dataclasses import dataclass

import pandas as pd

from ..market_time import LOCAL_TZ

EXPERIMENT_NAME = "baselines"
SPRINT3_EXPERIMENT_NAME = "sprint3_models"
CALIBRATION_EXPERIMENT_NAME = "calibration_and_risk"


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


# Extended quantile grid from step 4.3a (leftmost = lowest level).
QUANTILE_GRID: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


@dataclass(frozen=True)
class ConformalConfig:
    """Constants for the scaled conformal recalibration (step 4.3b).

    One place for every tunable number in the calibration layer. The rolling
    calibration-window length is additionally exposed on the CLI
    (``--calibration-window``); the CLI value overrides ``window_days`` here.
    All other knobs are config-only, on purpose, to stay reproducible.
    """

    # Rolling calibration-window length in days (CLI-overridable default).
    window_days: int = 90

    # Embargo (gap) in days between the calibration window and the test day.
    # Guarantees the calibration realizations are known at gate closure of the
    # test day's forecast. 1 day already covers the day-ahead convention.
    embargo_days: int = 1

    # k for the kNN-in-forecast-level local-scale estimator (sigma).
    n_neighbors: int = 200

    # Floor for sigma, as a fraction of the global calibration-window std of y.
    # Prevents division by a near-zero local scale in flat neighbourhoods.
    sigma_floor_fraction: float = 0.10

    # Recompute sigma + the seven Q_alpha once per this many days.
    recompute_cadence_days: int = 1

    # Minimum calibration hours required to calibrate a day. Below this the
    # day's raw predictions are passed through UNCHANGED and flagged as
    # "uncalibrated" in the diagnostics (honest: no correction from too little
    # data). Early test days without a full window fall here.
    min_calibration_hours: int = 24 * 14
