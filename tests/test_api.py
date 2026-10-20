from fastapi.testclient import TestClient

from mlops.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "Heart Disease API" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_monitor_and_metrics(sample):
    client.post("/predict", json=sample)
    monitor = client.get("/monitor")
    metrics = client.get("/metrics")
    assert monitor.status_code == 200
    assert "API request log" in monitor.text
    assert metrics.status_code == 200
    assert "heart_api_requests_total" in metrics.text


def test_predict(sample):
    info = client.get("/predict")
    assert info.status_code == 200
    assert "age" in info.json()["features"]

    scored = client.post("/predict", json=sample)
    assert scored.status_code == 200
    body = scored.json()
    assert body["prediction"] in {0, 1}
    assert 0 <= body["probability"] <= 1
