import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.65

FEATURES_TO_SCALE = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]


# ============================================================
# LOAD TRAINING AND FINAL TEST DATA
# ============================================================

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze("columns")

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze("columns")


print("=" * 70)
print("FINAL DIABETES MODEL EVALUATION")
print("=" * 70)

print(f"Training samples : {len(X_train):,}")
print(f"Final test samples: {len(X_test):,}")
print(f"Final threshold   : {FINAL_THRESHOLD}")
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
# FINAL MODEL
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


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# TRAIN ON COMPLETE TRAINING SET
# ============================================================

print("Training final model on complete training set...")

pipeline.fit(
    X_train,
    y_train,
)

print("Final model training complete.")
print()


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

test_probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    test_predictions,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
)

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)
print()

print("TN:", cm[0, 0])
print("FP:", cm[0, 1])
print("FN:", cm[1, 0])
print("TP:", cm[1, 1])
print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "No Diabetes",
            "Diabetes/Prediabetes",
        ],
        zero_division=0,
    )
)


# ============================================================
# SAVE FINAL METRICS
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "model": "HistGradientBoostingClassifier",
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "threshold": FINAL_THRESHOLD,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "true_negative": cm[0, 0],
            "false_positive": cm[0, 1],
            "false_negative": cm[1, 0],
            "true_positive": cm[1, 1],
        }
    ]
)

metrics.to_csv(
    "data/processed/final_diabetes_metrics.csv",
    index=False,
)

print("=" * 70)
print("Saved:")
print("data/processed/final_diabetes_metrics.csv")
print("=" * 70)

print()
print("FINAL TEST EVALUATION COMPLETE.")