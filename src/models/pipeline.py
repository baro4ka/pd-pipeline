"""Sklearn pipeline: preprocessing + model, hyperparameter search, metrics.

Preprocessing (inside the pipeline, so no leakage):
  * numeric columns: SimpleImputer (median) + StandardScaler
  * categorical columns: SimpleImputer (constant) + OneHotEncoder

The feature matrix is built from base numeric columns (LIMIT_BAL, AGE),
engineered features (see src.data.feature_engineering) and categoricals
(SEX, EDUCATION, MARRIAGE, age_group).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_BASE = ["LIMIT_BAL", "AGE"]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]

from ..data.feature_engineering import FEATURED_COLS  # noqa: E402

# Engineered numeric features (all numeric by construction).
ENGINEERED_NUMERIC = [c for c in FEATURED_COLS if c not in ("age_group",)]

NUMERIC_FEATURES = NUMERIC_BASE + ENGINEERED_NUMERIC
FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLS


def _to_str(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to str so a string fill_value works for any input dtype.

    A named (module-level) function: lambdas crash cloudpickle on py3.14
    when the fitted pipeline is dumped with joblib.
    """
    return frame.astype(str)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(_to_str)),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_COLS),
        ]
    )


def make_model_pipeline(model) -> Pipeline:
    return Pipeline(steps=[("preprocessor", make_preprocessor()), ("model", model)])


def make_model(name: str, **params):
    """Model factory: 'logreg', 'gb', 'hgb', 'rf'."""
    if name == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "gb":
        model = GradientBoostingClassifier(random_state=42)
    elif name == "hgb":
        model = HistGradientBoostingClassifier(random_state=42)
    elif name == "rf":
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
    else:
        raise ValueError(f"Unknown model: {name}")
    for key, value in params.items():
        setattr(model, key, value)
    return model


def default_param_grids() -> dict:
    """Param grids for the base models (model.<param> pipeline naming)."""
    return {
        "logreg": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "gb": (
            GradientBoostingClassifier(random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Key binary-classification metrics on a holdout set."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, ax=None):
    """Draw the ROC curve; returns the matplotlib Axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return ax


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "logreg",
    cv: int = 5,
) -> tuple[GridSearchCV, dict]:
    """GridSearchCV over the pipeline; returns (best_search, test_metrics)."""
    model, param_grid = default_param_grids()[model_name]
    pipeline = make_model_pipeline(model)
    # n_jobs is capped (not -1): parallel workers plus the CPU-bound
    # transcription job running on the same machine exhaust 16 GB of RAM.
    search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=2,
        refit=True,
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    y_prob = search.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)
    return search, metrics
