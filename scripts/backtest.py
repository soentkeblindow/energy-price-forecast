"""Run SimilarDayNaive walk-forward backtest and log results to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly
from energy_price_forecast.evaluation.metrics import summarise
from energy_price_forecast.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.models.baseline import SimilarDayNaive


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest for SimilarDayNaive.")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument("--window", default="expanding", choices=["expanding", "rolling"])
    p.add_argument("--train-span-days", type=int, default=None)
    p.add_argument("--refit-every", type=int, default=1)
    p.add_argument("--out", default="data/processed/backtest_similarday.parquet", type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    df = load_interim_hourly(args.data_path) if args.data_path else load_interim_hourly()
    y: pd.Series = df["day_ahead_price"].rename("day_ahead_price")

    folds = list(
        walk_forward_splits(
            pd.DatetimeIndex(y.index),
            test_start=args.test_start,
            test_end=args.test_end,
            window=args.window,
            train_span_days=args.train_span_days,
        )
    )

    model = SimilarDayNaive()
    predictions = run_backtest(y, model, folds, refit_every=args.refit_every, x=None)
    summary = summarise(predictions)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sprint2_baselines")

    with mlflow.start_run(run_name="similarday_naive"):
        mlflow.log_params(
            {
                "model": "similarday_naive",
                "window": args.window,
                "refit_every": args.refit_every,
                "test_start": args.test_start,
                "test_end": str(args.test_end),
            }
        )
        mlflow.log_metrics(summary)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(args.out)
        mlflow.log_artifact(str(args.out))

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
