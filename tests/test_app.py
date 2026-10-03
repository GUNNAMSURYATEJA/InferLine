"""Tests for InferLine. Model is loaded once per session (fixture)."""

import pytest
from fastapi.testclient import TestClient

from inferline.app import app, load_model


@pytest.fixture(scope="session")
def client():
    load_model()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["ready"] is True
    assert "distilbert" in body["model"]


def test_predict_shape_and_labels(client):
    r = client.post("/predict", json={"text": "I loved this movie, it was wonderful."})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= body["score"] <= 1.0
    assert r.headers["X-Request-ID"]  # request-id middleware active


def test_predict_negative(client):
    r = client.post("/predict", json={"text": "Awful. A complete waste of time."})
    assert r.status_code == 200
    assert r.json()["label"] == "NEGATIVE"


def test_predict_batch(client):
    texts = ["I loved it.", "I hated it.", "It was fine, I guess."]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    out = r.json()
    assert len(out) == len(texts)
    for p in out:
        assert p["label"] in ("POSITIVE", "NEGATIVE")


def test_predict_too_long(client):
    r = client.post("/predict", json={"text": "x" * 600})
    assert r.status_code == 422


def test_predict_empty(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_blank(client):
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_batch_rejects_blank_entry(client):
    r = client.post("/predict/batch", json={"texts": ["good", "   "]})
    assert r.status_code == 422


def test_batch_rejects_oversize(client):
    r = client.post("/predict/batch", json={"texts": ["x"] * 65})
    assert r.status_code == 422


def test_metrics_populates(client):
    client.post("/predict", json={"text": "Great stuff."})
    client.post("/predict", json={"text": "Not great."})
    r = client.get("/metrics")
    assert r.status_code == 200
    eps = r.json()["endpoints"]
    assert "/predict" in eps
    assert eps["/predict"]["count"] >= 2
    assert eps["/predict"]["p50_ms"] > 0
