from pathlib import Path
import pandas as pd


SILVER_PATH = Path("data/silver/cse-cic-ids2018")
GOLD_PATH = Path("data/gold/cse-cic-ids2018")


def engineer_file(file):
    print(f"\nProcessing: {file.name}")

    df = pd.read_parquet(file)

    # Create binary attack target
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)

    # Select useful network features
    feature_columns = [
        "Protocol",
        "Flow_Duration",
        "Total_Fwd_Packets",
        "Total_Backward_Packets",
        "Fwd_Packets_Length_Total",
        "Bwd_Packets_Length_Total",
        "Flow_Bytes_per_s",
        "Flow_Packets_per_s",
        "Packet_Length_Mean",
        "Packet_Length_Std",
        "Average_Packet_Size",
        "is_attack"
    ]

    available_columns = [
        col for col in feature_columns
        if col in df.columns
    ]

    gold_df = df[available_columns].copy()

    output_file = GOLD_PATH / file.name
    gold_df.to_parquet(output_file, index=False)

    print(f"Rows: {len(gold_df)}")
    print(f"Features created: {len(gold_df.columns)}")
    print(f"Gold file created: {output_file}")


def engineer_dataset():
    GOLD_PATH.mkdir(parents=True, exist_ok=True)

    files = list(SILVER_PATH.glob("*.parquet"))

    print(f"Found {len(files)} Silver files")

    for file in files:
        engineer_file(file)


if __name__ == "__main__":
    engineer_dataset()