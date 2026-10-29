"""Integration tests for FastAPI student prediction and supervision endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from student_performance.api.main import create_app
from student_performance.core.config import load_config


def test_health_and_prediction_endpoints() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    h_data = health.json()
    assert h_data["status"] == "healthy"
    assert "subgroup_fairness" in h_data
    assert h_data["subgroup_fairness"]["fairness_passed"] is True

    payload = {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 75.0,
        "writing_score": 78.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["predicted_math_score"] <= 100.0
    assert "reading_score" in data["feature_attributions"]
    assert len(data["interventions"]) >= 1


def test_drift_supervision_progression() -> None:
    cfg = load_config()
    app = create_app(cfg)
    client = TestClient(app)

    init_drift = client.get("/monitor/drift")
    assert init_drift.status_code == 200
    assert init_drift.json()["status"] == "insufficient_data"

    for r_score in [50.0, 60.0, 70.0, 80.0, 90.0, 95.0]:
        client.post(
            "/predict",
            json={
                "gender": "male",
                "race_ethnicity": "group C",
                "parental_level_of_education": "some college",
                "lunch": "standard",
                "test_preparation_course": "none",
                "reading_score": r_score,
                "writing_score": r_score - 2.0,
            },
        )

    drift_after = client.get("/monitor/drift")
    assert drift_after.status_code == 200
    d_data = drift_after.json()
    assert d_data["sample_size"] == 6
    assert d_data["status"] in {"healthy", "drift_detected"}
    assert "reading_score" in d_data["numerical_features"]
    assert "lunch" in d_data["categorical_features"]
