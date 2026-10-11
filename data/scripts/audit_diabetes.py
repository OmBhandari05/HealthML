import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/raw/diabetes_health_indicators.csv")


# ============================================================
# Load Dataset
# ============================================================

def load_data():
    """Load the raw diabetes dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    return df


# ============================================================
# Dataset Overview
# ============================================================

def dataset_overview(df):
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# ============================================================
# Data Types
# ============================================================

def data_types(df):
    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    print(df.dtypes)


# ============================================================
# Missing Values
# ============================================================

def missing_values(df):
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()

    print(missing)

    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")


# ============================================================
# Duplicate Records
# ============================================================

def duplicate_records(df):
    print("\n" + "=" * 60)
    print("DUPLICATE RECORDS")
    print("=" * 60)

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates:,}")


# ============================================================
# Target Distribution
# ============================================================

def target_distribution(df):
    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    counts = df["Diabetes_binary"].value_counts().sort_index()

    percentages = (
        df["Diabetes_binary"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    result = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    print(result)

    print("\nTarget meaning:")
    print("  0 = No diabetes")
    print("  1 = Prediabetes or diabetes")


# ============================================================
# Unique Values
# ============================================================

def unique_values(df):
    print("\n" + "=" * 60)
    print("UNIQUE VALUES")
    print("=" * 60)

    for column in df.columns:
        unique = df[column].nunique()

        print(f"{column:25s}: {unique}")


# ============================================================
# Numerical Summary
# ============================================================

def numerical_summary(df):
    print("\n" + "=" * 60)
    print("NUMERICAL SUMMARY")
    print("=" * 60)

    print(df.describe().T.to_string())


# ============================================================
# ID Validation
# ============================================================

def id_validation(df):
    print("\n" + "=" * 60)
    print("ID VALIDATION")
    print("=" * 60)

    unique_ids = df["ID"].nunique()
    total_rows = len(df)

    print(f"Total rows : {total_rows:,}")
    print(f"Unique IDs : {unique_ids:,}")

    if unique_ids == total_rows:
        print("Result     : Every ID is unique.")
    else:
        print("Result     : Duplicate IDs detected.")


# ============================================================
# BMI Audit
# ============================================================

def bmi_audit(df):
    print("\n" + "=" * 60)
    print("BMI AUDIT")
    print("=" * 60)

    print(f"BMI >= 50 : {(df['BMI'] >= 50).sum():,}")
    print(f"BMI >= 60 : {(df['BMI'] >= 60).sum():,}")
    print(f"BMI >= 80 : {(df['BMI'] >= 80).sum():,}")
    print(f"BMI == 98 : {(df['BMI'] == 98).sum():,}")


# ============================================================
# Main
# ============================================================

def main():
    print("\nStarting diabetes dataset audit...")

    df = load_data()

    dataset_overview(df)
    data_types(df)
    missing_values(df)
    duplicate_records(df)
    target_distribution(df)
    unique_values(df)
    numerical_summary(df)
    id_validation(df)
    bmi_audit(df)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()