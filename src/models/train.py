"""Training entry point with MLflow tracking.

Usage:
    python -m src.models.train            # full dataset, 5 MLflow experiments
    python -m src.models.train --quick    # 5k-row subsample (fast smoke run)

MLflow (file store in <repo>/mlruns):
    * params: model, best/fixed hyperparameters, dataset size, feature count
    * metrics: cv_roc_auc + test roc_auc/precision/recall/f1
    * artifacts: ROC curve, trained model (mlflow.sklearn.log_model)

Also writes data/processed/: best_<model>.joblib, metrics.csv, roc_curve.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from ..data.feature_engineering import build_features
from ..data.load import MODELS_DIR, PROCESSED_DIR, TARGET, clean, load_raw
from .pipeline import (
    FEATURES,
    compute_metrics,
    make_model,
    make_model_pipeline,
    plot_roc_curve,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_EXPERIMENT = "pd-pipeline"

# 5 experiments: 2 with GridSearchCV (logreg, gb) + 3 fixed-config variants
# (different algorithms / hyperparameters) — TZ requires >= 5 runs.
EXPERIMENTS = [
    {"name": "logreg_grid", "search": "logreg"},
    {"name": "gb_grid", "search": "gb"},
    {
        "name": "gb_shallow",
        "model": "gb",
        "params": {"learning_rate": 0.05, "max_depth": 2, "n_estimators": 500},
    },
    {
        "name": "hgb",
        "model": "hgb",
        "params": {"max_iter": 300, "learning_rate": 0.08, "max_depth": 3},
    },
    {"name": "rf", "model": "rf", "params": {"n_estimators": 200, "max_depth": 6}},
]


def prepare_data(quick: bool = False, data_path: str | None = None) -> pd.DataFrame:
    if data_path:
        df = pd.read_csv(data_path)
    else:
        df = clean(load_raw())
        df = build_features(df)
    if quick:
        df = df.sample(5000, random_state=42)
    return df


def run_experiment(exp: dict, X_train, y_train, X_test, y_test, cv: int):
    """Run one experiment under an MLflow run; returns (model, params, metrics)."""
    if "search" in exp:
        search, metrics = train(X_train, y_train, X_test, y_test, model_name=exp["search"], cv=cv)
        model = search.best_estimator_
        params = {"model": exp["search"] + "_grid", **search.best_params_}
        params["cv_roc_auc"] = search.best_score_
    else:
        model_obj = make_model(exp["model"], **exp["params"])
        pipeline = make_model_pipeline(model_obj)
        pipeline.fit(X_train, y_train)
        model = pipeline
        params = {"model": exp["model"], **exp["params"]}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)
    return model, params, metrics, y_prob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="5k-row subsample (smoke run)")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument(
        "--data",
        default=None,
        help="pre-prepared features CSV (DVC prepare stage output) instead of raw data",
    )
    args = parser.parse_args(argv)

    df = prepare_data(quick=args.quick, data_path=args.data)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"train={len(X_train)} test={len(X_test)} features={len(FEATURES)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_mlflow:
        mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows = []
    last_models: dict = {}
    for i, exp in enumerate(EXPERIMENTS):
        if not args.no_mlflow:
            mlflow.start_run(run_name=exp["name"])
        try:
            model, params, metrics, y_prob = run_experiment(
                exp, X_train, y_train, X_test, y_test, cv=3 if args.quick else 5
            )
            print(f"[{exp['name']}] {params}")
            print(f"[{exp['name']}] test={metrics}")

            if not args.no_mlflow:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_params(
                    {
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "n_features": len(FEATURES),
                        "quick": args.quick,
                    }
                )
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
                if "cv_roc_auc" in params:
                    mlflow.log_metrics({"cv_roc_auc": params["cv_roc_auc"]})
                ax = plot_roc_curve(y_test.to_numpy(), y_prob)
                roc_path = PROCESSED_DIR / f"roc_{exp['name']}.png"
                ax.figure.savefig(roc_path, dpi=120)
                mlflow.log_artifact(str(roc_path))
                mlflow.sklearn.log_model(model, name="model")
        finally:
            if not args.no_mlflow:
                mlflow.end_run()

        last_models[exp["name"]] = model
        rows.append({"experiment": exp["name"], **metrics})
        if "search" in exp:
            rows[-1]["cv_roc_auc"] = params["cv_roc_auc"]
            joblib.dump(model, PROCESSED_DIR / f"best_{exp['search']}.joblib")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(PROCESSED_DIR / "metrics.csv", index=False)

    # Save the best model (by test ROC-AUC) as the DVC-tracked model artifact.
    best = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_models[str(best["experiment"])], MODELS_DIR / "best_model.joblib")
    print(
        f"best model: {best['experiment']} (test roc_auc={best['roc_auc']:.4f}) "
        "-> models/best_model.joblib"
    )

    print(f"artifacts -> {PROCESSED_DIR}")
    if not args.no_mlflow:
        print(f"mlflow ui: mlflow ui --backend-store-uri file:{PROJECT_ROOT / 'mlruns'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
