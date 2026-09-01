from pathlib import Path
import pandas as pd


BRONZE_FILE = Path("data/bronze/sample_network_traffic.parquet")
SILVER_PATH = Path("data/silver")

def clean_data():
    df = pd.read_parquet(BRONZE_FILE)

    print("Bronze data loaded")
    print(f"Rows before cleaning: {len(df)}")

    df = df.drop_duplicates()
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nData types:")
    print(df.dtypes)

    SILVER_PATH.mkdir(parents=True, exist_ok=True)

    silver_file = SILVER_PATH / "network_traffic_cleaned.parquet"
    df.to_parquet(silver_file, index=False)

    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Silver file created: {silver_file}")

    return df

if __name__ =="__main__":
    clean_data()