"""Walk-forward backtest for SimilarDayNaive or LassoForecaster; logs to MLflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from energy_price_forecast.data.loaders import load_interim_hourly, load_processed_features
from energy_price_forecast.evaluation.config import EXPERIMENT_NAME
from energy_price_forecast.evaluation.metrics import summarise
from energy_price_forecast.evaluation.walkforward import run_backtest, walk_forward_splits
from energy_price_forecast.models.baseline import (
    LassoForecaster,
    OLSForecaster,
    RidgeForecaster,
    SimilarDayNaive,
)


def _fingerprint(df: pd.DataFrame) -> str:
    return f"{df.index.min()}_{df.index.max()}_{df.shape}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward backtest (naive or lasso).")
    p.add_argument("--model", default="naive", choices=["naive", "lasso", "ridge", "ols"])
    p.add_argument("--target-transform", default="asinh", choices=["asinh", "identity"])
    p.add_argument("--study", default="adhoc", help="MLflow tag: logical study grouping.")
    p.add_argument("--note", default="", help="MLflow tag: free-text run note.")
    p.add_argument("--test-start", default="2021-01-01")
    p.add_argument("--test-end", default=None)
    p.add_argument("--window", default="expanding", choices=["expanding", "rolling"])
    p.add_argument("--train-span-days", type=int, default=None)
    p.add_argument(
        "--refit-every",
        type=int,
        default=None,
        help="Refit cadence in days. Defaults to 1 for naive, 7 for lasso.",
    )
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--features-path", default=None, type=Path)
    p.add_argument("--data-path", default=None, type=Path)
    p.add_argument("--out", default=None, type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    df = load_interim_hourly(args.data_path) if args.data_path else load_interim_hourly()
    price: pd.Series = df["day_ahead_price"].rename("day_ahead_price")

    if args.model == "naive":
        refit_every = args.refit_every if args.refit_every is not None else 1
        model: SimilarDayNaive | LassoForecaster | RidgeForecaster | OLSForecaster = (
            SimilarDayNaive()
        )
        index = pd.DatetimeIndex(price.index)
        y = price
        x = None
        run_name = "similarday_naive"
        out = args.out or Path("data/processed/backtest_similarday.parquet")
        log_params: dict[str, object] = {
            "model": "similarday_naive",
            "target_transform": "none",
            "window": args.window,
            "refit_every": refit_every,
            "test_start": args.test_start,
            "test_end": str(args.test_end),
        }
        log_tags: dict[str, str] = {
            "study": args.study,
            "note": args.note,
        }
    else:
        refit_every = args.refit_every if args.refit_every is not None else 7
        features_path = args.features_path or Path("data/processed/features.parquet")
        features = load_processed_features(features_path)
        y = price.reindex(features.index)
        x = features
        index = pd.DatetimeIndex(features.index)
        if args.model == "lasso":
            model = LassoForecaster(
                target_transform=args.target_transform,
                cv_splits=args.cv_splits,
                n_jobs=args.n_jobs,
            )
            extra_params: dict[str, object] = {"cv_splits": args.cv_splits}
        elif args.model == "ridge":
            model = RidgeForecaster(target_transform=args.target_transform)
            extra_params = {}
        else:  # ols
            model = OLSForecaster(target_transform=args.target_transform)
            extra_params = {}
        run_name = f"{args.model}_{args.target_transform}"
        out = args.out or Path(f"data/processed/backtest_{args.model}.parquet")
        log_params = {
            "model": args.model,
            "target_transform": args.target_transform,
            "window": args.window,
            "refit_every": refit_every,
            "test_start": args.test_start,
            "test_end": str(args.test_end),
            **extra_params,
        }
        log_tags = {
            "study": args.study,
            "note": args.note,
            "features_fingerprint": _fingerprint(features),
        }

    folds = list(
        walk_forward_splits(
            index,
            test_start=args.test_start,
            test_end=args.test_end,
            window=args.window,
            train_span_days=args.train_span_days,
        )
    )

    predictions = run_backtest(y, model, folds, refit_every=refit_every, x=x)
    summary = summarise(predictions)

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        mlflow.set_tags(log_tags)
        mlflow.log_metrics(summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out)
        mlflow.log_artifact(str(out))

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
