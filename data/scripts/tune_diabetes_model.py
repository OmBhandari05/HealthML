import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


# ============================================================
# Load Original Train/Test Split
# ============================================================

def load_data():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    ).squeeze("columns")

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze("columns")

    return X_train, X_test, y_train, y_test


# ============================================================
# Preprocessor
# ============================================================

def build_preprocessor():

    numeric_features = [
        "BMI",
        "MentHlth",
        "PhysHlth"
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
# Models
# ============================================================

def build_models():

    models = {}
    

    models["Logistic Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
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
    models["Hist Gradient Boosting"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )
    return models


# ============================================================
# Threshold Evaluation
# ============================================================

def evaluate_thresholds(y_true, probabilities):

    thresholds = [
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(
                y_true,
                predictions
            ),
            "precision": precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "recall": recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            "f1": f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        })

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting validation and threshold tuning...")

    X_train, X_test, y_train, y_test = load_data()

    # --------------------------------------------------------
    # Split ONLY the original training data
    # --------------------------------------------------------

    X_subtrain, X_validation, y_subtrain, y_validation = (
        train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
    )

    print("\nData split:")
    print(f"Sub-training : {len(X_subtrain):,}")
    print(f"Validation   : {len(X_validation):,}")
    print(f"Final test   : {len(X_test):,}")

    models = build_models()

    validation_results = {}

    # --------------------------------------------------------
    # Train and evaluate on validation set
    # --------------------------------------------------------

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(name.upper())
        print("=" * 70)

        print("\nTraining...")

        model.fit(
            X_subtrain,
            y_subtrain
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        roc_auc = roc_auc_score(
            y_validation,
            probabilities
        )

        pr_auc = average_precision_score(
            y_validation,
            probabilities
        )

        print(f"\nValidation ROC-AUC: {roc_auc:.4f}")
        print(f"Validation PR-AUC : {pr_auc:.4f}")

        results = evaluate_thresholds(
            y_validation,
            probabilities
        )

        print("\nThreshold comparison:")
        print(
            results.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

        # Select threshold with highest F1
        best_row = results.loc[
            results["f1"].idxmax()
        ]

        validation_results[name] = {
            "model": model,
            "threshold": best_row["threshold"],
            "f1": best_row["f1"],
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        }

        print("\nBest validation threshold:")
        print(f"Threshold : {best_row['threshold']:.2f}")
        print(f"Precision : {best_row['precision']:.4f}")
        print(f"Recall    : {best_row['recall']:.4f}")
        print(f"F1 Score  : {best_row['f1']:.4f}")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    comparison = pd.DataFrame([
        {
            "model": name,
            "threshold": values["threshold"],
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1"],
            "roc_auc": values["roc_auc"],
            "pr_auc": values["pr_auc"]
        }
        for name, values in validation_results.items()
    ])

    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison
        .sort_values("f1", ascending=False)
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Select model using validation F1
    # --------------------------------------------------------

    best_model_name = comparison.loc[
        comparison["f1"].idxmax(),
        "model"
    ]

    best_threshold = validation_results[
        best_model_name
    ]["threshold"]

    print("\n" + "=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(f"Model     : {best_model_name}")
    print(f"Threshold : {best_threshold:.2f}")

    print(
        "\nThe final test set has NOT been used for "
        "model or threshold selection."
    )

    # --------------------------------------------------------
    # Final evaluation
    #
    # For this experiment we refit the selected model on
    # the complete original training set, then evaluate once
    # on the untouched test set.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    selected_model = build_models()[
        best_model_name
    ]

    selected_model.fit(
        X_train,
        y_train
    )

    test_probabilities = selected_model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    print(
        f"\nAccuracy : "
        f"{accuracy_score(y_test, test_predictions):.4f}"
    )

    print(
        f"Precision: "
        f"{precision_score(y_test, test_predictions):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_test, test_predictions):.4f}"
    )

    print(
        f"F1 Score : "
        f"{f1_score(y_test, test_predictions):.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{roc_auc_score(y_test, test_probabilities):.4f}"
    )

    print(
        f"PR-AUC   : "
        f"{average_precision_score(y_test, test_probabilities):.4f}"
    )

    print("\nFinal Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()