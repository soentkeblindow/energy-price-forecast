import numpy as np
import pandas as pd


def _aligned(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align on the index and drop rows where either side is NaN."""
    df = pd.concat([y_true.rename("t"), y_pred.rename("p")], axis=1).dropna()
    return df["t"], df["p"]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean absolute error (EUR/MWh)."""
    t, p = _aligned(y_true, y_pred)
    return float((t - p).abs().mean())


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root mean squared error (EUR/MWh)."""
    t, p = _aligned(y_true, y_pred)
    return float(np.sqrt(((t - p) ** 2).mean()))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted absolute percentage error: sum|e| / sum|y_true|.

    The aggregated denominator makes WAPE robust where individual prices are
    near zero — unlike MAPE, which divides per observation. |y_true| keeps
    negative prices from cancelling in the denominator. Returns NaN if the
    denominator is 0.
    """
    t, p = _aligned(y_true, y_pred)
    denom = t.abs().sum()
    return float((t - p).abs().sum() / denom) if denom > 0 else float("nan")


def summarise(predictions: pd.DataFrame) -> dict[str, float]:
    """Pooled MAE/RMSE/WAPE over all test rows, plus per-delivery-day MAE
    distribution (mean/std/p05/p50/p95) as a stability read. NaN pairs dropped.
    """
    t, p = predictions["y_true"], predictions["y_pred"]
    per_day = (
        predictions.dropna(subset=["y_true", "y_pred"])
        .assign(abs_err=lambda df: (df["y_true"] - df["y_pred"]).abs())
        .groupby("delivery_day")["abs_err"]
        .mean()
    )
    return {
        "mae": mae(t, p),
        "rmse": rmse(t, p),
        "wape": wape(t, p),
        "mae_per_day_mean": float(per_day.mean()),
        "mae_per_day_std": float(per_day.std()),
        "mae_per_day_p05": float(per_day.quantile(0.05)),
        "mae_per_day_p50": float(per_day.quantile(0.50)),
        "mae_per_day_p95": float(per_day.quantile(0.95)),
    }
