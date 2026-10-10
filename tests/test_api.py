
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200

    models = response.json()["available_models"]
    assert "diabetes" in models
    assert "cardiovascular" in models


def test_diabetes_prediction():
    payload = {
        "high_bp": 1,
        "high_chol": 1,
        "chol_check": 1,
        "bmi": 30.0,
        "smoker": 0,
        "stroke": 0,
        "heart_disease_or_attack": 0,
        "physical_activity": 1,
        "fruits": 1,
        "veggies": 1,
        "heavy_alcohol_consumption": 0,
        "any_healthcare": 1,
        "no_doc_bc_cost": 0,
        "general_health": 2,
        "mental_health_days": 2,
        "physical_health_days": 2,
        "difficulty_walking": 0,
        "sex": 1,
        "age_category": 8,
        "education": 5,
        "income": 6,
    }

    response = client.post(
        "/api/v1/predict/diabetes",
        json=payload,
    )

    assert response.status_code == 200
    result = response.json()
    assert result["disease"] == "diabetes"
    assert 0 <= result["risk_probability"] <= 1
    assert result["model_version"] == "diabetes-hgb-1.0"


def test_cardiovascular_prediction():
    payload = {
        "age": 55,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 250,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 1,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": None,
        "thal": None,
    }

    response = client.post(
        "/api/v1/predict/cardiovascular",
        json=payload,
    )

    assert response.status_code == 200
    result = response.json()
    assert result["disease"] == "cardiovascular"
    assert 0 <= result["risk_probability"] <= 1
    assert result["model_version"] == "cardiovascular-lr-1.0"


def test_invalid_cardiovascular_input():
    response = client.post(
        "/api/v1/predict/cardiovascular",
        json={"age": 55},
    )

    assert response.status_code == 422
