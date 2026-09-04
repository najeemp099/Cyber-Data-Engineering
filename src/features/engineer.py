from pathlib import Path
import pandas as pd


SILVER_FILE = Path("data/silver/network_traffic_cleaned.parquet")
GOLD_PATH = Path("data/gold")


def create_features():
    
    df = pd.read_parquet(SILVER_FILE)

    df["duration_bytes"] = df["bytes"] / (df["packet_count"] + 1)
    df["is_attack"] = (df["label"] != "BENIGN").astype(int)

    GOLD_PATH.mkdir(parents=True, exist_ok=True)

    gold_file = GOLD_PATH / "network_traffic_features.parquet"
    df.to_parquet(gold_file, index=False)

    print(f"Gold file created: {gold_file}")

    return df


if __name__ == "__main__":
    df = create_features()

    print("Silver data loaded")
    print(f"Rows: {len(df)}")
    print(f"Features created: {len(df.columns)}")

    print("\nFeature columns:")
    print(df.columns.tolist())