import os

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MODEL_VERSION = "diabetes-hgb-1.0"

FEATURES_TO_SCALE = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze("columns")


print("=" * 70)
print("SAVING FINAL DIABETES MODEL")
print("=" * 70)

print(f"Training samples: {len(X_train):,}")
print()


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "scale",
            StandardScaler(),
            FEATURES_TO_SCALE,
        )
    ],
    remainder="passthrough",
)


# ============================================================
# FINAL HIST GRADIENT BOOSTING MODEL
# ============================================================

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=50,
    l2_regularization=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("Training final pipeline...")

pipeline.fit(
    X_train,
    y_train,
)

print("Training complete.")
print()


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

model_directory = "app/models/diabetes"

os.makedirs(
    model_directory,
    exist_ok=True,
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    model_directory,
    "model.joblib",
)

joblib.dump(
    pipeline,
    model_path,
)


print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"Path   : {model_path}")
print(f"Version: {MODEL_VERSION}")
print()


# ============================================================
# SAVE MODEL METADATA
# ============================================================

metadata = {
    "model_version": MODEL_VERSION,
    "model_type": "HistGradientBoostingClassifier",
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "threshold": 0.65,
    "random_state": RANDOM_STATE,
    "training_samples": len(X_train),
    "features": list(X_train.columns),
}


metadata_path = os.path.join(
    model_directory,
    "metadata.json",
)

import json

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


print(f"Metadata: {metadata_path}")
print()
print("Final diabetes model is ready for API integration.")