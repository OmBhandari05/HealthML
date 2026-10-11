import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


# ============================================================
# Load Data
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
        "PhysHlth",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        ],
        remainder="passthrough",
    )


# ============================================================
# Build Model
# ============================================================

def build_model(params):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    **params,
                ),
            ),
        ]
    )


# ============================================================
# Threshold Evaluation
# ============================================================

def find_best_threshold(y_true, probabilities):

    thresholds = [
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    best = None

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        result = {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if best is None or f1 > best["f1"]:
            best = result

    return best


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting Hist Gradient Boosting tuning...")

    X_train, X_test, y_train, y_test = load_data()

    # --------------------------------------------------------
    # Create validation set from training data only
    # --------------------------------------------------------

    X_subtrain, X_validation, y_subtrain, y_validation = (
        train_test_split(
            X_train,
            y_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    )

    print("\nData:")
    print(f"Sub-training : {len(X_subtrain):,}")
    print(f"Validation   : {len(X_validation):,}")
    print(f"Final test   : {len(X_test):,}")

    # --------------------------------------------------------
    # Focused hyperparameter configurations
    # --------------------------------------------------------

    configurations = [

        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },

        {
            "learning_rate": 0.08,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },

        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
        },

        {
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        },

        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 50,
            "l2_regularization": 2.0,
        },

        {
            "learning_rate": 0.10,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 2.0,
        },
    ]

    results = []

    # --------------------------------------------------------
    # Train configurations
    # --------------------------------------------------------

    for number, params in enumerate(
        configurations,
        start=1,
    ):

        print("\n" + "=" * 70)
        print(f"CONFIGURATION {number}/{len(configurations)}")
        print("=" * 70)

        print(params)

        model = build_model(params)

        print("\nTraining...")

        model.fit(
            X_subtrain,
            y_subtrain,
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        best_threshold = find_best_threshold(
            y_validation,
            probabilities,
        )

        print(
            f"\nROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"PR-AUC : {pr_auc:.4f}"
        )

        print(
            f"Best threshold: "
            f"{best_threshold['threshold']:.2f}"
        )

        print(
            f"Precision: "
            f"{best_threshold['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_threshold['recall']:.4f}"
        )

        print(
            f"F1: "
            f"{best_threshold['f1']:.4f}"
        )

        results.append(
            {
                **params,
                "threshold": best_threshold["threshold"],
                "precision": best_threshold["precision"],
                "recall": best_threshold["recall"],
                "f1": best_threshold["f1"],
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

    # --------------------------------------------------------
    # Compare configurations
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)

    print(
        results_df
        .sort_values(
            "f1",
            ascending=False,
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Select best configuration
    # --------------------------------------------------------

    best = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    print(
        best.to_string()
    )

    print(
        "\nFinal test set was NOT used "
        "for hyperparameter selection."
    )

    print("\nTuning complete.")


if __name__ == "__main__":
    main()