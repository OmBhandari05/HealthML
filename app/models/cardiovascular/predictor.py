
import json
import os

import joblib
import pandas as pd

from app.core.base_predictor import BasePredictor
from app.schemas.health import (
    CardiovascularProfile,
    PredictionResponse,
)


class CardiovascularPredictor(BasePredictor):
    name = "cardiovascular"

    def __init__(self):
        project_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "..", "..", ".."
            )
        )

        model_dir = os.path.join(
            project_root, "app", "models", "cardiovascular"
        )

        self.model = joblib.load(
            os.path.join(model_dir, "model.joblib")
        )

        with open(
            os.path.join(model_dir, "metadata.json"),
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        self.version = self.metadata["model_version"]
        self.threshold = float(self.metadata["threshold"])

    def predict(
        self,
        profile: CardiovascularProfile,
    ) -> PredictionResponse:

        features = pd.DataFrame([{
            "age": profile.age,
            "sex": profile.sex,
            "cp": profile.cp,
            "trestbps": profile.trestbps,
            "chol": profile.chol,
            "fbs": profile.fbs,
            "restecg": profile.restecg,
            "thalach": profile.thalach,
            "exang": profile.exang,
            "oldpeak": profile.oldpeak,
            "slope": profile.slope,
            "ca": profile.ca,
            "thal": profile.thal,
        }])

        score = float(self.model.predict_proba(features)[0, 1])
        prediction = int(score >= self.threshold)

        explanations = []

        # These are observed input values, not causal explanations
        # or feature-attribution results.
        if profile.age >= 55:
            explanations.append("Age is in the older range of this dataset.")
        if profile.exang == 1:
            explanations.append(
                "Exercise-induced angina was reported in the input."
            )
        if profile.oldpeak >= 1.0:
            explanations.append(
                "The entered ST-depression value is elevated relative "
                "to zero in the dataset."
            )
        if profile.trestbps >= 140:
            explanations.append(
                "The entered resting blood pressure is 140 mmHg or higher."
            )
        if profile.chol >= 240:
            explanations.append(
                "The entered cholesterol value is 240 mg/dL or higher."
            )

        if not explanations:
            explanations.append(
                "No rule-based input observations were highlighted."
            )

        return PredictionResponse(
            disease=self.name,
            risk_probability=round(score, 4),
            risk_level=(
                "elevated model score"
                if prediction == 1
                else "lower model score"
            ),
            model_version=self.version,
            explanation=explanations,
            disclaimer=(
                "Educational research prototype only. This model uses "
                "a small historical dataset and is not a medical "
                "diagnosis or a validated clinical risk calculator."
            ),
        )
