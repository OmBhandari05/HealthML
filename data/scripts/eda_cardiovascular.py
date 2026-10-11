import os

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "heart_disease.csv",
)

FIGURES_DIR = os.path.join(
    PROJECT_ROOT,
    "reports",
    "figures",
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Convert original target:
# 0 = no disease
# 1-4 = disease present
df["target"] = (df["num"] > 0).astype(int)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(df["target"].value_counts().sort_index())


# --------------------------------------------------
# 1. Target distribution
# --------------------------------------------------

target_counts = df["target"].value_counts().sort_index()

plt.figure(figsize=(7, 5))

plt.bar(
    ["No Disease", "Disease Present"],
    target_counts.values,
)

plt.title("Cardiovascular Disease Target Distribution")
plt.xlabel("Target")
plt.ylabel("Number of Patients")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "06_cardiovascular_target_distribution.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 2. Age distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df.loc[df["target"] == 0, "age"],
    bins=15,
    alpha=0.7,
    label="No Disease",
)

plt.hist(
    df.loc[df["target"] == 1, "age"],
    bins=15,
    alpha=0.7,
    label="Disease Present",
)

plt.title("Age Distribution by Cardiovascular Disease Status")
plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "07_cardiovascular_age_distribution.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 3. Cholesterol distribution
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["target"] == 0, "chol"],
        df.loc[df["target"] == 1, "chol"],
    ],
    tick_labels=["No Disease", "Disease Present"],
)

plt.title("Cholesterol Distribution by Disease Status")
plt.xlabel("Target")
plt.ylabel("Cholesterol")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "08_cardiovascular_cholesterol.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 4. Maximum heart rate
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["target"] == 0, "thalach"],
        df.loc[df["target"] == 1, "thalach"],
    ],
    tick_labels=["No Disease", "Disease Present"],
)

plt.title("Maximum Heart Rate by Disease Status")
plt.xlabel("Target")
plt.ylabel("Maximum Heart Rate")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "09_cardiovascular_max_heart_rate.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 5. Resting blood pressure
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["target"] == 0, "trestbps"],
        df.loc[df["target"] == 1, "trestbps"],
    ],
    tick_labels=["No Disease", "Disease Present"],
)

plt.title("Resting Blood Pressure by Disease Status")
plt.xlabel("Target")
plt.ylabel("Resting Blood Pressure")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "10_cardiovascular_blood_pressure.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 6. ST depression
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        df.loc[df["target"] == 0, "oldpeak"],
        df.loc[df["target"] == 1, "oldpeak"],
    ],
    tick_labels=["No Disease", "Disease Present"],
)

plt.title("ST Depression by Disease Status")
plt.xlabel("Target")
plt.ylabel("Oldpeak")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "11_cardiovascular_oldpeak.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 7. Chest pain vs disease prevalence
# --------------------------------------------------

cp_prevalence = (
    df.groupby("cp")["target"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))

plt.bar(
    cp_prevalence.index.astype(str),
    cp_prevalence.values,
)

plt.title("Cardiovascular Disease Prevalence by Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Disease Prevalence (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "12_cardiovascular_chest_pain.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# 8. Exercise-induced angina
# --------------------------------------------------

exang_prevalence = (
    df.groupby("exang")["target"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(7, 5))

plt.bar(
    ["No", "Yes"],
    exang_prevalence.values,
)

plt.title(
    "Cardiovascular Disease Prevalence by Exercise-Induced Angina"
)
plt.xlabel("Exercise-Induced Angina")
plt.ylabel("Disease Prevalence (%)")

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "13_cardiovascular_exercise_angina.png",
    ),
    dpi=150,
)

plt.close()


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\nEDA completed successfully.")

print("\nGenerated figures:")

for filename in sorted(os.listdir(FIGURES_DIR)):
    if filename.startswith(("06_", "07_", "08_", "09_", "10_", "11_", "12_", "13_")):
        print("-", filename)