from pathlib import Path
import pandas as pd


SILVER_PATH = Path("data/silver/cse-cic-ids2018")


def validate_dataset():
    files = list(SILVER_PATH.glob("*.parquet"))

    print(f"Found {len(files)} Silver files")

    for file in files:
        df = pd.read_parquet(file)

        print(f"\nFile: {file.name}")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        print("\nMissing values:")
        print(df.isnull().sum().sum())

        print("\nLabels:")
        print(df["Label"].value_counts())


if __name__ == "__main__":
    validate_dataset()