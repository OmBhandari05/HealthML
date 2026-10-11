
from pathlib import Path
from io import BytesIO

import pandas as pd
import truststore
import urllib.request

# Use the operating system's certificate store
truststore.inject_into_ssl()

DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
OUTPUT_PATH = Path("data/raw/diabetes_health_indicators.csv")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading CDC Diabetes Health Indicators dataset...")

    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        df = pd.read_csv(BytesIO(response.read()))

    if df.shape != (253680, 23):
        raise ValueError(f"Unexpected dataset shape: {df.shape}")

    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset downloaded successfully!")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")
    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(df["Diabetes_binary"].value_counts().sort_index())


if __name__ == "__main__":
    main()
