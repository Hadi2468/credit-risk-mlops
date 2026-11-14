# tests/test_api.py
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body, dict), "Error: it is not a dictionary"
    assert body == {"status": "ok"}

json_data = {"user_id": "sara", "avg_payment_delay": 9}

def test_predict():
    response = client.post("/predict", json=json_data)
    body = response.json()
    assert response.status_code == 200
    assert isinstance(body, dict), "Error: we need dictionary format"
    assert body == {"user_id": "sara", "default_probability": 0.45}

negative_json_data = {"user_id": "sara", "avg_payment_delay": -9}

def test_predict_negative_delay():
    response = client.post("/predict", json=negative_json_data)
    assert response.status_code == 422
    assert "detail" in response.json()

missing_json_data = {"avg_payment_delay": 9}
def test_predict_missing_field():
    response = client.post("/predict", json=missing_json_data)
    assert response.status_code == 422
