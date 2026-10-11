import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/raw/diabetes_health_indicators.csv")
OUTPUT_DIR = Path("data/processed")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting diabetes data preparation...")

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Raw dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    valid_targets = {0, 1}

    actual_targets = set(df["Diabetes_binary"].unique())

    if not actual_targets.issubset(valid_targets):
        raise ValueError(
            f"Unexpected target values: {actual_targets}"
        )

    # --------------------------------------------------------
    # Separate features and target
    # --------------------------------------------------------

    # ID is an identifier, not a predictive feature.
    X = df.drop(
        columns=["ID", "Diabetes_binary"]
    )

    y = df["Diabetes_binary"]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain/Test split:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    # --------------------------------------------------------
    # Check class distribution
    # --------------------------------------------------------

    print("\nTraining target distribution:")
    print(
        y_train.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    print("\nTesting target distribution:")
    print(
        y_test.value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # --------------------------------------------------------
    # Create processed directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save split datasets
    # --------------------------------------------------------

    X_train.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False
    )

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'X_train.csv'}")
    print(f"  {OUTPUT_DIR / 'X_test.csv'}")
    print(f"  {OUTPUT_DIR / 'y_train.csv'}")
    print(f"  {OUTPUT_DIR / 'y_test.csv'}")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()