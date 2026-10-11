
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed"
)


# --------------------------------------------------
# Load training and locked test data
# --------------------------------------------------

X_train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_X_train.csv")
)

y_train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_y_train.csv")
)["target"]

X_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_y_test.csv")
)["target"]


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

numeric_features = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

categorical_features = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]


# --------------------------------------------------
# Preprocessing pipeline
# --------------------------------------------------

numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, numeric_features),
    ("categorical", categorical_preprocessor, categorical_features),
])


# --------------------------------------------------
# Final selected model
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])


# --------------------------------------------------
# Train on all training samples
# --------------------------------------------------

print("Training final Logistic Regression model...")
model.fit(X_train, y_train)

print("Training samples:", len(X_train))
print("Locked test samples:", len(X_test))


# --------------------------------------------------
# Evaluate once on the locked test set
# --------------------------------------------------

THRESHOLD = 0.60

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

metrics = {
    "model": "Logistic Regression",
    "threshold": THRESHOLD,
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "pr_auc": average_precision_score(y_test, probabilities),
}


print("\n" + "=" * 55)
print("FINAL CARDIOVASCULAR TEST RESULTS")
print("=" * 55)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nConfusion matrix:")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["No Disease", "Disease Present"],
    zero_division=0,
))


# --------------------------------------------------
# Save evaluation metrics
# --------------------------------------------------

output_path = os.path.join(
    PROCESSED_DIR,
    "final_cardiovascular_metrics.csv",
)

pd.DataFrame([metrics]).to_csv(output_path, index=False)

print("\nMetrics saved to:")
print(output_path)
print("\nFinal evaluation completed.")
