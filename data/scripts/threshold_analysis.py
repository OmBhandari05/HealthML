import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURES_TO_SCALE = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]

THRESHOLDS = [
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

X = pd.read_csv("data/processed/X_train.csv")
y = pd.read_csv("data/processed/y_train.csv").squeeze("columns")


# ============================================================
# CREATE VALIDATION SPLIT
# ============================================================

X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("=" * 70)
print("THRESHOLD ANALYSIS")
print("=" * 70)

print(f"Sub-training : {len(X_subtrain):,}")
print(f"Validation   : {len(X_validation):,}")
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
# BEST HGB CONFIGURATION
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
# TRAIN
# ============================================================

print("Training winning Hist Gradient Boosting configuration...")

pipeline.fit(X_subtrain, y_subtrain)

print("Training complete.")
print()


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

validation_probabilities = pipeline.predict_proba(
    X_validation
)[:, 1]

roc_auc = roc_auc_score(
    y_validation,
    validation_probabilities,
)

pr_auc = average_precision_score(
    y_validation,
    validation_probabilities,
)


print(f"Validation ROC-AUC: {roc_auc:.4f}")
print(f"Validation PR-AUC : {pr_auc:.4f}")
print()


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in THRESHOLDS:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
        },
    )
)

print()


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

print("=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

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

print()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/processed/diabetes_threshold_analysis.csv",
    index=False,
)

print(
    "Saved: data/processed/diabetes_threshold_analysis.csv"
)
print()

print("Threshold analysis complete.")