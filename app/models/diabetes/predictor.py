from pathlib import Path

import joblib
import pandas as pd

from app.core.base_predictor import BasePredictor
from app.schemas.health import DiabetesProfile, PredictionResponse


class DiabetesPredictor(BasePredictor):
    name = "diabetes"
    version = "diabetes-hgb-1.0"
    threshold = 0.65

    # Load the trained pipeline once when the application starts.
    model_path = (
        Path(__file__).resolve().parent / "model.joblib"
    )

    model = joblib.load(model_path)

    def predict(
        self,
        profile: DiabetesProfile,
    ) -> PredictionResponse:

        # Convert API input into the exact feature names
        # and order used during model training.
        features = pd.DataFrame(
            [
                {
                    "HighBP": profile.high_bp,
                    "HighChol": profile.high_chol,
                    "CholCheck": profile.chol_check,
                    "BMI": profile.bmi,
                    "Smoker": profile.smoker,
                    "Stroke": profile.stroke,
                    "HeartDiseaseorAttack": (
                        profile.heart_disease_or_attack
                    ),
                    "PhysActivity": profile.physical_activity,
                    "Fruits": profile.fruits,
                    "Veggies": profile.veggies,
                    "HvyAlcoholConsump": (
                        profile.heavy_alcohol_consumption
                    ),
                    "AnyHealthcare": profile.any_healthcare,
                    "NoDocbcCost": profile.no_doc_bc_cost,
                    "GenHlth": profile.general_health,
                    "MentHlth": profile.mental_health_days,
                    "PhysHlth": profile.physical_health_days,
                    "DiffWalk": profile.difficulty_walking,
                    "Sex": profile.sex,
                    "Age": profile.age_category,
                    "Education": profile.education,
                    "Income": profile.income,
                }
            ]
        )

        # Get the model's positive-class score.
        risk_score = float(
            self.model.predict_proba(features)[0][1]
        )

        # Apply the threshold selected using validation data.
        prediction = int(
            risk_score >= self.threshold
        )

        # Risk level is based on the model decision.
        if prediction == 1:
            risk_level = "elevated"
        else:
            risk_level = "lower"

        # Generate simple feature-based explanations.
        reasons = []

        if profile.high_bp == 1:
            reasons.append(
                "High blood pressure was reported."
            )

        if profile.high_chol == 1:
            reasons.append(
                "High cholesterol was reported."
            )

        if profile.bmi >= 30:
            reasons.append(
                "BMI is in the obesity range."
            )

        if profile.general_health >= 4:
            reasons.append(
                "General health was reported as poor or very poor."
            )

        if profile.physical_health_days >= 15:
            reasons.append(
                "A high number of physically unhealthy days was reported."
            )

        if profile.difficulty_walking == 1:
            reasons.append(
                "Difficulty walking was reported."
            )

        if profile.age_category >= 9:
            reasons.append(
                "The reported age category is higher."
            )

        if not reasons:
            reasons.append(
                "No major input-level risk factors were highlighted."
            )

        return PredictionResponse(
            disease=self.name,
            risk_probability=round(risk_score, 3),
            risk_level=risk_level,
            model_version=self.version,
            explanation=reasons,
            disclaimer=(
                "Educational risk-assessment prototype only; "
                "not a medical diagnosis."
            ),
        )