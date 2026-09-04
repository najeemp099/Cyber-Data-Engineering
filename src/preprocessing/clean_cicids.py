from pathlib import Path
import pandas as pd


BRONZE_PATH = Path("data/bronze/cse-cic-ids2018")
SILVER_PATH = Path("data/silver/cse-cic-ids2018")


def clean_file(file):
    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)

    print(f"Rows before cleaning: {len(df)}")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_per_")
    )

    # Clean label values
    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(str).str.strip().str.upper()

    print(f"Rows after cleaning: {len(df)}")

    output_file = SILVER_PATH / file.name
    df.to_parquet(output_file, index=False)

    print(f"Silver file created: {output_file}")


def clean_dataset():
    SILVER_PATH.mkdir(parents=True, exist_ok=True)

    files = list(BRONZE_PATH.glob("*.parquet"))

    print(f"Found {len(files)} Bronze files")

    for file in files:
        clean_file(file)


if __name__ == "__main__":
    clean_dataset()