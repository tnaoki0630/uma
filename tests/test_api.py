from fastapi.testclient import TestClient

from uma_api.main import app

client = TestClient(app)


def test_health_and_demo_prediction():
    assert client.get("/health").json() == {"status": "ok"}
    payload = client.get("/api/demo").json()
    payload["samples"] = 2_000
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert abs(sum(item["win_probability"] for item in body["entries"]) - 1) < 1e-9
    assert body["trio"]
    assert body["trifecta"]


def test_static_csv_demo_catalog_is_served():
    response = client.get("/api/demo-catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["modes"]["horse"]["source"] == "UMA_DB.csv"
    assert catalog["modes"]["jockey"]["source"] == "Jockey_DB.csv"


def test_g1_holdout_catalog_is_served():
    response = client.get("/api/g1-demo-catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["split"]["validation_year"] == 2025
    assert catalog["split"]["history_rule"] == "occurred_at < target_race_date"
    assert catalog["modes"]["horse"]["races"]
    assert catalog["modes"]["jockey"]["races"]
