
import json
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT, "data", "processed")
MODEL_DIR = os.path.join(ROOT, "app", "models", "cardiovascular")

os.makedirs(MODEL_DIR, exist_ok=True)

X_train = pd.read_csv(
    os.path.join(DATA_DIR, "cardiovascular_X_train.csv")
)
y_train = pd.read_csv(
    os.path.join(DATA_DIR, "cardiovascular_y_train.csv")
)["target"]

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, numeric_features),
    ("categorical", categorical_preprocessor, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# Fit the final pipeline on all training samples.
model.fit(X_train, y_train)

model_path = os.path.join(MODEL_DIR, "model.joblib")
joblib.dump(model, model_path)

metadata = {
    "model_version": "cardiovascular-lr-1.0",
    "model_type": "Logistic Regression",
    "threshold": 0.60,
    "training_samples": int(len(X_train)),
    "features": numeric_features + categorical_features,
    "target_definition": {
        "0": "No disease in the original dataset",
        "1": "Disease present in the original dataset"
    },
    "test_metrics": {
        "accuracy": 0.8689,
        "precision": 0.8571,
        "recall": 0.8571,
        "f1": 0.8571,
        "roc_auc": 0.9524,
        "pr_auc": 0.9459
    },
    "disclaimer": (
        "Educational research prototype only; "
        "not a medical diagnosis system."
    )
}

metadata_path = os.path.join(MODEL_DIR, "metadata.json")

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Cardiovascular model saved successfully.")
print("Model:", model_path)
print("Metadata:", metadata_path)
print("Training samples:", len(X_train))
print("Threshold:", metadata["threshold"])
