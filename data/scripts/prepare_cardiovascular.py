import os

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "heart_disease.csv",
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
)

os.makedirs(PROCESSED_DIR, exist_ok=True)


# --------------------------------------------------
# Load raw data
# --------------------------------------------------

df = pd.read_csv(RAW_DATA_PATH)

print("Raw dataset shape:", df.shape)


# --------------------------------------------------
# Create binary target
# --------------------------------------------------
#
# Original UCI target:
# 0 = absence of disease
# 1-4 = presence of disease
#
# Our model:
# 0 = no disease
# 1 = disease present
# --------------------------------------------------

df["target"] = (df["num"] > 0).astype(int)

# Remove original target
X = df.drop(columns=["num", "target"])
y = df["target"]


# --------------------------------------------------
# Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

X_train.to_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_X_train.csv"),
    index=False,
)

X_test.to_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_X_test.csv"),
    index=False,
)

y_train.to_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_y_train.csv"),
    index=False,
)

y_test.to_csv(
    os.path.join(PROCESSED_DIR, "cardiovascular_y_test.csv"),
    index=False,
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\nFeature matrix shape:", X.shape)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTraining target percentage:")
print(
    (y_train.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nTesting target percentage:")
print(
    (y_test.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

print("\nMissing values in training data:")
print(X_train.isna().sum())

print("\nMissing values in testing data:")
print(X_test.isna().sum())

print("\nCardiovascular data preparation completed successfully.")

print("\nSaved files:")

for filename in [
    "cardiovascular_X_train.csv",
    "cardiovascular_X_test.csv",
    "cardiovascular_y_train.csv",
    "cardiovascular_y_test.csv",
]:
    print("-", filename)