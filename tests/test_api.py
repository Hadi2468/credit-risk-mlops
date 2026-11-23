# tests/test_api.py
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_known_user():
    response = client.post("/predict", json={"user_id": 2})
    body = response.json()
    assert response.status_code == 200
    assert 0 <= body["default_probability"] <= 1, "probability must be between 0 and 1"
    assert "model_version" in body


def test_predict_unknown_user():
    response = client.post("/predict", json={"user_id": 99999})
    assert response.status_code == 404


def test_predict_invalid_user_id():
    response = client.post("/predict", json={"user_id": 0})
    assert response.status_code == 422