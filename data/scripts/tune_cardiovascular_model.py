import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

X = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "cardiovascular_X_train.csv",
    )
)

y = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "cardiovascular_y_train.csv",
    )
)["target"]


# --------------------------------------------------
# Training / validation split
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

numeric_features = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

categorical_features = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

numeric_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_preprocessor,
            numeric_features,
        ),
        (
            "categorical",
            categorical_preprocessor,
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(
    X_train,
    y_train,
)

probabilities = model.predict_proba(X_val)[:, 1]


# --------------------------------------------------
# Threshold-independent metrics
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_val,
    probabilities,
)

pr_auc = average_precision_score(
    y_val,
    probabilities,
)

print("\nThreshold-independent metrics:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")


# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


results = []

print("\n")
print("=" * 75)
print("CARDIOVASCULAR THRESHOLD ANALYSIS")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


# --------------------------------------------------
# Find best F1 threshold
# --------------------------------------------------

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n")
print("=" * 75)
print("BEST VALIDATION THRESHOLD BY F1")
print("=" * 75)

print(
    f"Threshold : {best_row['threshold']:.2f}"
)

print(
    f"Precision : {best_row['precision']:.4f}"
)

print(
    f"Recall    : {best_row['recall']:.4f}"
)

print(
    f"F1 Score  : {best_row['f1']:.4f}"
)


# --------------------------------------------------
# Save threshold analysis
# --------------------------------------------------

output_path = os.path.join(
    PROCESSED_DIR,
    "cardiovascular_threshold_analysis.csv",
)

results_df.to_csv(
    output_path,
    index=False,
)

print(
    f"\nThreshold analysis saved to:\n{output_path}"
)