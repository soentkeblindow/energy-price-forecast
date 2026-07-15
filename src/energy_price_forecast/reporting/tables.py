"""Summary-table aggregation for the report/dashboard artefact bundle (Sprint 5.1).

Pure functions only -- every function takes already-loaded DataFrames and
returns a small, tidy DataFrame ready to write to CSV. No file I/O, no
MLflow, no new metrics: everything here re-aggregates numbers that Sprint
2-4 already computed and persisted.
"""

from __future__ import annotations

import pandas as pd

from ..evaluation.metrics import summarise

MODEL_COMPARISON_ORDER: tuple[str, ...] = ("Naive", "Lasso", "LightGBM", "ARIMAX")


def model_comparison(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled MAE/RMSE/WAPE per model over the common test period.

    `predictions` maps a display model name (see `MODEL_COMPARISON_ORDER`)
    to its tidy `y_true`/`y_pred`/`delivery_day` prediction frame. Rows are
    returned in `MODEL_COMPARISON_ORDER` order where present; unknown model
    names are appended after, in the order given.
    """
    rows = []
    for name, frame in predictions.items():
        stats = summarise(frame)
        rows.append(
            {
                "model": name,
                "mae": round(stats["mae"], 2),
                "rmse": round(stats["rmse"], 2),
                "wape": round(stats["wape"], 4),
            }
        )
    table = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(MODEL_COMPARISON_ORDER)}
    table["_order"] = table["model"].map(lambda m: order.get(m, len(order)))
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def coverage_summary(
    conformal_raw: pd.DataFrame,
    conformal_sorted: pd.DataFrame,
    reliability_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Nominal vs. empirical one-sided coverage per quantile level, before/after calibration.

    `conformal_raw`/`conformal_sorted` are `conformal_reliability_{raw,sorted}.csv`
    (LightGBM, no `model` column). `reliability_curve` is `reliability_curve.csv`
    (has a `model` column, including the ARIMAX pale-reference curve).
    All three already carry a precomputed `bucket == "overall"` row per level
    whose `n` equals the sum of the per-bucket `n`s (the full-sample marginal
    coverage) -- so no extra n-weighted aggregation is needed here, only a
    filter to `bucket == "overall"` (same data sources as Figure 2 in
    `04_regime_risk.ipynb`, reused rather than re-derived).
    """

    def _overall(frame: pd.DataFrame) -> pd.Series:
        return frame[frame["bucket"] == "overall"].set_index("level")["coverage"]

    lgbm_raw = _overall(conformal_raw)
    lgbm_calibrated = _overall(conformal_sorted)
    arimax = _overall(reliability_curve[reliability_curve["model"] == "arimax"])

    table = pd.DataFrame(
        {
            "nominal_level": lgbm_raw.index,
            "lightgbm_raw_coverage": lgbm_raw.to_numpy(),
        }
    ).set_index("nominal_level")
    table["lightgbm_calibrated_coverage"] = lgbm_calibrated
    table["arimax_raw_coverage"] = arimax
    return table.round(4).reset_index()


_BACKTEST_COVERAGE_REPORTED_COLUMNS: tuple[str, ...] = (
    "variant",
    "side",
    "subset",
    "n",
    "n_breach",
    "breach_rate",
    "breach_rate_ci_low",
    "breach_rate_ci_high",
    "breach_rate_ci_excludes_alpha",
    "kupiec_pvalue",
    "chris_ind_lr_day",
    "chris_ind_pvalue_day",
    "z1",
    "z1_ci_low",
    "z1_ci_high",
    "low_support",
)


def backtest_coverage_export(backtest_coverage: pd.DataFrame) -> pd.DataFrame:
    """`backtest_coverage.csv` reduced to the columns actually reported/cited.

    Keeps the full 60-validated-cell (+ `raw` counterfactual) row structure;
    drops the block-bootstrap-sensitivity/Basel-only columns not used
    outside `04_regime_risk.ipynb`'s own sensitivity sections.
    """
    missing = [c for c in _BACKTEST_COVERAGE_REPORTED_COLUMNS if c not in backtest_coverage.columns]
    if missing:
        raise ValueError(f"backtest_coverage is missing expected columns: {missing}")
    return backtest_coverage[list(_BACKTEST_COVERAGE_REPORTED_COLUMNS)].copy()


def risk_headline(backtest_coverage: pd.DataFrame, risk_summary: pd.DataFrame) -> pd.DataFrame:
    """The `overall`-subset headline row per (variant, side): breach rate with CI,
    Kupiec p-value, `z1` with CI, plus `buffer_factor` (`mean_var(calibrated) /
    mean_var(raw)`, recomputed here from the persisted `risk_summary.csv` --
    not read from MLflow, per Decision 1).
    """
    overall = backtest_coverage[backtest_coverage["subset"] == "overall"][
        [
            "variant",
            "side",
            "n",
            "n_breach",
            "breach_rate",
            "breach_rate_ci_low",
            "breach_rate_ci_high",
            "kupiec_pvalue",
            "z1",
            "z1_ci_low",
            "z1_ci_high",
        ]
    ].copy()

    mean_var = risk_summary.set_index(["variant", "side"])["mean_var"]
    buffer_factor = {}
    for side in ("long", "short"):
        raw_var = mean_var.get(("raw", side))
        calibrated_var = mean_var.get(("calibrated", side))
        if raw_var is not None and calibrated_var is not None and raw_var != 0:
            buffer_factor[side] = calibrated_var / raw_var

    overall["buffer_factor"] = overall.apply(
        lambda row: buffer_factor.get(row["side"]) if row["variant"] == "calibrated" else pd.NA,
        axis=1,
    )
    return overall.reset_index(drop=True)
