"""Sklearn-пайплайн: препроцессинг + модель, подбор гиперпараметров, метрики.

Препроцессинг внутри пайплайна, чтобы не было утечки данных:
  * числовые колонки: SimpleImputer (медиана) + StandardScaler
  * категориальные: SimpleImputer (константа) + OneHotEncoder

Матрица признаков: базовые числовые (LIMIT_BAL, AGE), инженерные
(см. src/data/feature_engineering.py) и категориальные
(SEX, EDUCATION, MARRIAGE, age_group).
"""

import matplotlib
import matplotlib.pyplot as plt
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

from ..data.feature_engineering import FEATURED_COLS

# обучение и CI идут без GUI — графики сохраняем в файлы
matplotlib.use("Agg")

NUMERIC_BASE = ["LIMIT_BAL", "AGE"]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]

# инженерные числовые признаки (все числовые по построению)
ENGINEERED_NUMERIC = [c for c in FEATURED_COLS if c != "age_group"]

NUMERIC_FEATURES = NUMERIC_BASE + ENGINEERED_NUMERIC
FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLS


def _to_str(frame):
    """Приводит колонки к str, чтобы строковый fill_value работал для любого dtype.

    Именованная функция (не lambda): lambda не сериализуется cloudpickle
    на Python 3.14 при joblib.dump обученного пайплайна.
    """
    return frame.astype(str)


def make_preprocessor():
    """ColumnTransformer: числовая и категориальная ветки."""
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


def make_model_pipeline(model):
    """Пайплайн: препроцессинг + модель."""
    return Pipeline(steps=[("preprocessor", make_preprocessor()), ("model", model)])


def make_model(name, **params):
    """Создаёт модель по имени: 'logreg', 'gb', 'hgb', 'rf'."""
    if name == "logreg":
        return LogisticRegression(max_iter=1000, class_weight="balanced", **params)
    if name == "gb":
        return GradientBoostingClassifier(random_state=42, **params)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=42, **params)
    if name == "rf":
        return RandomForestClassifier(random_state=42, class_weight="balanced", **params)
    raise ValueError(f"Неизвестная модель: {name}")


# Сетки подбора для GridSearchCV (нотация pipeline: model__<параметр>)
PARAM_GRIDS = {
    "logreg": (
        make_model("logreg"),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "gb": (
        make_model("gb"),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ),
}


def score_model(y_true, y_pred, y_prob):
    """Ключевые метрики бинарной классификации на тестовой выборке."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_roc_curve(y_true, y_prob, ax=None):
    """Рисует ROC-кривую, возвращает Axes."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("Доля ложных срабатываний (FPR)")
    ax.set_ylabel("Доля истинных срабатываний (TPR)")
    ax.set_title("ROC-кривая")
    ax.legend(loc="lower right")
    return ax


def train(X_train, y_train, X_test, y_test, model_name="logreg", cv=5):
    """GridSearchCV по пайплайну; возвращает (best_search, метрики теста)."""
    model, param_grid = PARAM_GRIDS[model_name]
    pipeline = make_model_pipeline(model)
    # n_jobs ограничен (не -1): параллельные воркеры + фоновые задачи на той же
    # машине съедают 16 ГБ RAM
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
    metrics = score_model(y_test.to_numpy(), y_pred, y_prob)
    return search, metrics
