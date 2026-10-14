"""Tests for the FastAPI serving app."""

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.data.feature_engineering import build_features
from src.data.load import MODELS_DIR, clean, load_raw
from src.models.pipeline import FEATURES

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.joblib").exists(),
    reason="models/best_model.joblib not found — run training first",
)


@pytest.fixture(scope="module")
def sample_rows():
    df = clean(load_raw())
    df = build_features(df)
    return df.sample(10, random_state=7)


@pytest.fixture()
def client(sample_rows):
    # Use a real model if available; the app loads it lazily.
    with TestClient(app_module.app) as c:
        yield c, sample_rows


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_features"] == len(FEATURES)


def test_predict_single(client):
    c, rows = client
    payload = rows[FEATURES].iloc[0].to_dict()
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["class"] in (0, 1)


def test_predict_batch(client):
    c, rows = client
    payload = rows[FEATURES].to_dict(orient="records")
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == len(rows)
    assert all(0.0 <= p["default_probability"] <= 1.0 for p in preds)


def test_predict_missing_features(client):
    c, rows = client
    payload = rows[FEATURES].iloc[0].to_dict()
    del payload["AGE"]
    resp = c.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty(client):
    c, _ = client
    resp = c.post("/predict", json=[])
    assert resp.status_code == 422
