
from fastapi import APIRouter, HTTPException

from app.core.registry import MODEL_REGISTRY
from app.schemas.health import (
    CardiovascularProfile,
    DiabetesProfile,
    HealthProfile,
    PredictionResponse,
)

router = APIRouter(prefix="/api/v1")


@router.get("/models")
def list_models():
    return {
        "available_models": list(MODEL_REGISTRY.keys())
    }


@router.post(
    "/predict/diabetes",
    response_model=PredictionResponse,
)
def predict_diabetes(profile: DiabetesProfile):
    predictor = MODEL_REGISTRY.get("diabetes")

    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail="Diabetes model is not available",
        )

    return predictor.predict(profile)


@router.post(
    "/predict/cardiovascular",
    response_model=PredictionResponse,
)
def predict_cardiovascular(profile: CardiovascularProfile):
    predictor = MODEL_REGISTRY.get("cardiovascular")

    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail="Cardiovascular model is not available",
        )

    return predictor.predict(profile)


@router.post(
    "/predict/{disease}",
    response_model=PredictionResponse,
)
def predict(disease: str, profile: HealthProfile):
    predictor = MODEL_REGISTRY.get(disease.lower())

    if predictor is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unsupported disease model: {disease}",
        )

    return predictor.predict(profile)
