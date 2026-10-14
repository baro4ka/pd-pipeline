"""FastAPI serving for the PD model.

Endpoints:
    GET  /health   -> service + model status
    POST /predict  -> JSON with feature(s) -> class + default probability

The model is the DVC-tracked artifact models/best_model.joblib
(a full sklearn Pipeline: preprocessor + classifier).
"""

from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from ..data.load import MODELS_DIR
from ..models.pipeline import FEATURES

MODEL_PATH = MODELS_DIR / "best_model.joblib"

app = FastAPI(title="PD Model API", version="1.0")

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found: {MODEL_PATH}. Run training first "
                "(python -m src.models.train).",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def _predict_frame(X: pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
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
    get_model()  # fail fast if the model is missing
    return {"status": "ok", "model": MODEL_PATH.name, "n_features": len(FEATURES)}


@app.post("/predict")
def predict(payload: dict | list[dict]):
    """Accept one feature dict or a list of them; returns class + probability."""
    items = payload if isinstance(payload, list) else [payload]
    if not items:
        raise HTTPException(status_code=422, detail="Empty payload")
    X = pd.DataFrame(items)
    try:
        result = _predict_frame(X)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    records = result.to_dict(orient="records")
    if isinstance(payload, list):
        return {"predictions": records}
    return {"default_probability": records[0]["default_probability"], "class": records[0]["class"]}
