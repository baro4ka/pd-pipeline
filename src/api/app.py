"""FastAPI-сервинг PD-модели.

Эндпоинты:
    GET  /health   -> статус сервиса и модели
    POST /predict  -> JSON с признаками -> класс + вероятность дефолта

Модель — артефакт под DVC models/best_model.joblib
(полный sklearn-пайплайн: препроцессинг + классификатор).
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..data.load import MODELS_DIR
from ..models.pipeline import FEATURES

MODEL_PATH = MODELS_DIR / "best_model.joblib"

app = FastAPI(title="PD Model API", version="1.0")

_model = None


def get_model():
    """Ленивая загрузка модели (один раз на процесс)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Модель не найдена: {MODEL_PATH}. "
                    "Сначала обучите (python -m src.models.train)."
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_frame(X):
    model = get_model()
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Не хватает признаков: {missing}")
    X = X[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            "default_probability": proba,
            "class": (proba >= 0.5).astype(int),
        }
    )
    out.index = X.index
    return out


@app.get("/health")
def health():
    get_model()  # сразу падаем, если модели нет
    return {"status": "ok", "model": MODEL_PATH.name, "n_features": len(FEATURES)}


@app.post("/predict")
def predict(payload: dict | list[dict]):
    """Принимает dict признаков или список; возвращает класс + вероятность."""
    items = payload if isinstance(payload, list) else [payload]
    if not items:
        raise HTTPException(status_code=422, detail="Пустой payload")
    X = pd.DataFrame(items)
    try:
        result = _predict_frame(X)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Ошибка предсказания: {exc}") from exc

    records = result.to_dict(orient="records")
    if isinstance(payload, list):
        return {"predictions": records}
    return {
        "default_probability": records[0]["default_probability"],
        "class": records[0]["class"],
    }
