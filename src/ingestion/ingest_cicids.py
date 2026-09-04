from pathlib import Path
import pandas as pd


RAW_PATH = Path("data/raw/cse-cic-ids2018")
BRONZE_PATH = Path("data/bronze/cse-cic-ids2018")


def ingest_dataset():
    BRONZE_PATH.mkdir(parents=True, exist_ok=True)

    files = list(RAW_PATH.glob("*.parquet"))

    print(f"Found {len(files)} dataset files")

    for file in files:
        print(f"\nProcessing: {file.name}")

        df = pd.read_parquet(file)

        output_file = BRONZE_PATH / file.name
        df.to_parquet(output_file, index=False)

        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Bronze file: {output_file}")


if __name__ == "__main__":
    ingest_dataset()