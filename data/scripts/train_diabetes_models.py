import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed")

RANDOM_STATE = 42


# ============================================================
# Load Data
# ============================================================

def load_data():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test


# ============================================================
# Build Preprocessor
# ============================================================

def build_preprocessor():

    # Continuous/count-based features that benefit from scaling
    numeric_features = [
        "BMI",
        "MentHlth",
        "PhysHlth"
    ]

    # Binary and ordinal variables are kept in their existing
    # representation for the first baseline experiment.
    other_features = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features
            )
        ],
        remainder="passthrough"
    )

    return preprocessor


# ============================================================
# Build Models
# ============================================================

def build_models(preprocessor):

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    )
                )
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )
    }

    return models


# ============================================================
# Evaluate Model
# ============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    print("\nTraining model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\nEvaluation Metrics")
    print("-" * 30)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report")
    print("-" * 30)
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Diabetes",
                "Prediabetes/Diabetes"
            ]
        )
    )

    print("Confusion Matrix")
    print("-" * 30)
    print(confusion_matrix(y_test, predictions))

    return model, {
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting diabetes model training...")

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training data: {X_train.shape}")
    print(f"Testing data : {X_test.shape}")

    preprocessor = build_preprocessor()

    models = build_models(preprocessor)

    results = []
    trained_models = {}

    for name, model in models.items():

        trained_model, metrics = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        trained_models[name] = trained_model
        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df
        .sort_values("roc_auc", ascending=False)
        .to_string(index=False)
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()