from pathlib import Path
import pandas as pd


SILVER_FILE = Path("data/silver/network_traffic_cleaned.parquet")


def validate_data():
    df = pd.read_parquet(SILVER_FILE)

    print("Silver data loaded")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nMissing values:")
    print(df.isnull().sum())


    print("\nValidation checks:")

    assert df["timestamp"].notna().all(), "Invalid timestamp found"
    assert df["src_ip"].notna().all(), "Missing source IP found"
    assert df["dst_ip"].notna().all(), "Missing destination IP found"
    assert df["protocol"].notna().all(), "Missing protocol found"
    assert (df["src_port"] >= 0).all(), "Invalid source port found"
    assert (df["dst_port"] >= 0).all(), "Invalid destination port found"
    assert (df["packet_count"] >= 0).all(), "Invalid packet count found"
    assert (df["bytes"] >= 0).all(), "Invalid bytes value found"

    print("All validation checks passed")

if __name__ == "__main__":
    validate_data()