import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/raw/diabetes_health_indicators.csv")
FIGURE_DIR = Path("reports/figures")

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# 1. Target Distribution
# ============================================================

def plot_target_distribution(df):
    counts = df["Diabetes_binary"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))

    plt.bar(
        ["No Diabetes", "Prediabetes/Diabetes"],
        counts.values
    )

    plt.title("Diabetes Target Distribution")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    output = FIGURE_DIR / "01_target_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 2. BMI Distribution
# ============================================================

def plot_bmi_distribution(df):
    plt.figure(figsize=(8, 5))

    plt.hist(
        df["BMI"],
        bins=40
    )

    plt.title("BMI Distribution")
    plt.xlabel("BMI")
    plt.ylabel("Number of Records")
    plt.tight_layout()

    output = FIGURE_DIR / "02_bmi_distribution.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 3. Diabetes Prevalence by BMI Group
# ============================================================

def plot_bmi_diabetes_rate(df):
    df = df.copy()

    df["BMI_Group"] = pd.cut(
        df["BMI"],
        bins=[
            0,
            18.5,
            25,
            30,
            35,
            40,
            50,
            60,
            80,
            100
        ],
        labels=[
            "<18.5",
            "18.5-24.9",
            "25-29.9",
            "30-34.9",
            "35-39.9",
            "40-49.9",
            "50-59.9",
            "60-79.9",
            "80+"
        ]
    )

    prevalence = (
        df.groupby(
            "BMI_Group",
            observed=False
        )["Diabetes_binary"]
        .mean()
        * 100
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        prevalence.index.astype(str),
        prevalence.values
    )

    plt.title("Diabetes/Prediabetes Prevalence by BMI Group")
    plt.xlabel("BMI Group")
    plt.ylabel("Prevalence (%)")
    plt.xticks(rotation=45)

    plt.tight_layout()

    output = FIGURE_DIR / "03_bmi_diabetes_prevalence.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 4. Diabetes Prevalence by High Blood Pressure
# ============================================================

def plot_highbp_relationship(df):
    prevalence = (
        df.groupby("HighBP")["Diabetes_binary"]
        .mean()
        * 100
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        ["No High BP", "High BP"],
        prevalence.values
    )

    plt.title("Diabetes/Prediabetes Prevalence by High Blood Pressure")
    plt.ylabel("Prevalence (%)")

    plt.tight_layout()

    output = FIGURE_DIR / "04_highbp_diabetes_prevalence.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ============================================================
# 5. Diabetes Prevalence by High Cholesterol
# ============================================================

def plot_highchol_relationship(df):
    prevalence = (
        df.groupby("HighChol")["Diabetes_binary"]
        .mean()
        * 100
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        ["No High Cholesterol", "High Cholesterol"],
        prevalence.values
    )

    plt.title("Diabetes/Prediabetes Prevalence by High Cholesterol")
    plt.ylabel("Prevalence (%)")

    plt.tight_layout()

    output = FIGURE_DIR / "05_highchol_diabetes_prevalence.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


# ============================================================
# Main
# ============================================================

def main():

    print("\nStarting diabetes EDA...")

    df = load_data()

    print(f"Dataset loaded: {df.shape}")

    plot_target_distribution(df)
    plot_bmi_distribution(df)
    plot_bmi_diabetes_rate(df)
    plot_highbp_relationship(df)
    plot_highchol_relationship(df)

    print("\nEDA COMPLETE")


if __name__ == "__main__":
    main()