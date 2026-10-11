import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
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
# Load prepared data
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
# Models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline(
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
    ),

    "Random Forest": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),

    "Hist Gradient Boosting": Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=200,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    ),
}


# --------------------------------------------------
# Train and evaluate
# --------------------------------------------------

results = []


for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(X_val)[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

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

    roc_auc = roc_auc_score(
        y_val,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_val,
        probabilities,
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
        }
    )


# --------------------------------------------------
# Comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 80)
print("CARDIOVASCULAR MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# --------------------------------------------------
# Save validation results
# --------------------------------------------------

results_path = os.path.join(
    PROCESSED_DIR,
    "cardiovascular_validation_results.csv",
)

results_df.to_csv(
    results_path,
    index=False,
)

print(
    f"\nValidation results saved to:\n{results_path}"
)