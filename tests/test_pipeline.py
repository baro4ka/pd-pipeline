"""Тесты sklearn-пайплайна, метрик и feature engineering."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import FEATURED_COLS, build_features
from src.data.load import TARGET, load_raw
from src.models.pipeline import (
    CATEGORICAL_COLS,
    FEATURES,
    NUMERIC_FEATURES,
    make_model_pipeline,
    make_preprocessor,
    score_model,
    train,
)


@pytest.fixture(scope="module")
def featured_df():
    df = load_raw()
    df = df.drop(columns=["ID"], errors="ignore")
    return build_features(df)


def test_feature_engineering_columns(featured_df):
    assert set(FEATURED_COLS).issubset(featured_df.columns)
    assert featured_df[FEATURED_COLS].isnull().sum().sum() == 0


def test_feature_matrix_complete(featured_df):
    assert set(FEATURES).issubset(featured_df.columns)
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_COLS))


def test_preprocessor_output(featured_df):
    sample = featured_df.sample(200, random_state=1)
    pre = make_preprocessor()
    out = pre.fit_transform(sample[FEATURES])
    assert out.shape[0] == 200
    assert np.isfinite(out).all()


def test_preprocessor_handles_missing(featured_df):
    sample = featured_df.sample(50, random_state=2).copy()
    sample.loc[sample.index[0], "LIMIT_BAL"] = np.nan
    sample.loc[sample.index[1], "SEX"] = np.nan
    out = make_preprocessor().fit_transform(sample[FEATURES])
    assert np.isfinite(out).all()


def test_score_model_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = score_model(y_true, y_pred, y_prob)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_score_model_zero_division():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])
    m = score_model(y_true, y_pred, y_prob)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_train_end_to_end_quick(featured_df):
    df = featured_df.sample(1500, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    search, metrics = train(X_train, y_train, X_test, y_test, model_name="logreg", cv=3)
    assert search.best_score_ > 0.5  # лучше случайного
    assert metrics["roc_auc"] > 0.5
    assert 0.0 <= metrics["f1"] <= 1.0


def test_model_pipeline_predicts_proba(featured_df):
    df = featured_df.sample(300, random_state=4)
    pipe = make_model_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(df[FEATURES], df[TARGET])
    proba = pipe.predict_proba(df[FEATURES])
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
