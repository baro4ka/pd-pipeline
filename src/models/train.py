"""Точка входа в обучение с трекингом в MLflow.

Запуск:
    python -m src.models.train             # полный датасет, 5 экспериментов
    python -m src.models.train --subsample # подвыборка 5k строк (быстрый прогон)

MLflow (file store в <repo>/mlruns):
    * параметры: модель, гиперпараметры, размеры выборок, число признаков
    * метрики: cv_roc_auc + test roc_auc/precision/recall/f1
    * артефакты: ROC-кривая, обученная модель (mlflow.sklearn.log_model)

Также пишет в data/processed/: best_<model>.joblib, metrics.csv, roc_*.png.
"""

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
    make_model,
    make_model_pipeline,
    plot_roc_curve,
    score_model,
    train,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLFLOW_EXPERIMENT = "pd-pipeline"

# 5 экспериментов: 2 с GridSearchCV (logreg, gb) + 3 фиксированные конфигурации
# (разные алгоритмы/гиперпараметры) — ТЗ требует не менее 5
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


def prepare_data(subsample=False, data_path=None):
    """Готовит датасет: сырые данные + FE, либо готовый features.csv (выход DVC prepare)."""
    if data_path:
        df = pd.read_csv(data_path)
    else:
        df = clean(load_raw())
        df = build_features(df)
    if subsample:
        df = df.sample(5000, random_state=42)
    return df


def run_exp(exp, X_train, y_train, X_test, y_test, cv):
    """Один эксперимент в MLflow-ране; возвращает (модель, параметры, метрики, вероятности)."""
    if "search" in exp:
        search, metrics = train(X_train, y_train, X_test, y_test, model_name=exp["search"], cv=cv)
        model = search.best_estimator_
        params = {"model": exp["search"] + "_grid", **search.best_params_}
        params["cv_roc_auc"] = float(search.best_score_)
    else:
        model_obj = make_model(exp["model"], **exp["params"])
        pipeline = make_model_pipeline(model_obj)
        pipeline.fit(X_train, y_train)
        model = pipeline
        params = {"model": exp["model"], **exp["params"]}

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = score_model(y_test.to_numpy(), y_pred, y_prob)
    return model, params, metrics, y_prob


def main(argv=None):
    parser = argparse.ArgumentParser(description="Обучение PD-модели с трекингом в MLflow")
    parser.add_argument(
        "--subsample", action="store_true", help="подвыборка 5k строк (быстрый прогон)"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="без логирования в MLflow")
    parser.add_argument(
        "--data",
        default=None,
        help="готовый features.csv (выход стадии DVC prepare) вместо сырых данных",
    )
    args = parser.parse_args(argv)

    df = prepare_data(subsample=args.subsample, data_path=args.data)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"выборки: train={len(X_train)} test={len(X_test)} признаков={len(FEATURES)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not args.no_mlflow:
        mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    rows = []
    last_models = {}
    for exp in EXPERIMENTS:
        if not args.no_mlflow:
            mlflow.start_run(run_name=exp["name"])
        try:
            model, params, metrics, y_prob = run_exp(
                exp, X_train, y_train, X_test, y_test, cv=3 if args.subsample else 5
            )
            print(f"[{exp['name']}] гиперпараметры: {params}")
            print(
                f"[{exp['name']}] тест: ROC-AUC {metrics['roc_auc']:.3f} | "
                f"P {metrics['precision']:.3f} | R {metrics['recall']:.3f} | "
                f"F1 {metrics['f1']:.3f}"
            )

            if not args.no_mlflow:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_params(
                    {
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "n_features": len(FEATURES),
                        "subsample": args.subsample,
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

    # лучшая модель (по test ROC-AUC) — артефакт под DVC
    best = metrics_df.loc[metrics_df["roc_auc"].idxmax()]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(last_models[str(best["experiment"])], MODELS_DIR / "best_model.joblib")
    print(
        f"лучшая модель: {best['experiment']} (test ROC-AUC {best['roc_auc']:.3f}) "
        f"-> models/best_model.joblib"
    )

    print(f"артефакты -> {PROCESSED_DIR}")
    if not args.no_mlflow:
        print(f"mlflow ui: mlflow ui --backend-store-uri file:{PROJECT_ROOT / 'mlruns'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
