import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
SAMPLES = json.loads((Path(__file__).resolve().parent.parent / "sample" / "sample_transactions.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info_has_threshold_and_features():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["threshold"] < 1
    assert len(body["features"]) == 29


def test_predict_returns_valid_score():
    r = client.post("/predict", json=SAMPLES["normal_example"])
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert isinstance(body["flagged"], bool)


def test_fraud_example_scores_higher_than_normal():
    normal = client.post("/predict", json=SAMPLES["normal_example"]).json()["fraud_score"]
    fraud = client.post("/predict", json=SAMPLES["fraud_caught_example"]).json()["fraud_score"]
    assert fraud > normal


def test_missing_field_is_rejected():
    bad = dict(SAMPLES["normal_example"])
    del bad["V1"]
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_amount_is_rejected():
    bad = dict(SAMPLES["normal_example"])
    bad["Amount"] = -5
    assert client.post("/predict", json=bad).status_code == 422
