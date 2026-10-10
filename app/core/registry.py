from app.models.diabetes.predictor import DiabetesPredictor
from app.models.cardiovascular.predictor import CardiovascularPredictor

MODEL_REGISTRY = {
    "diabetes": DiabetesPredictor(),
    "cardiovascular": CardiovascularPredictor(),
}
