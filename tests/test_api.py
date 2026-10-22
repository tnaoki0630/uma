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
