from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


GOLD_PATH = Path("data/gold/cse-cic-ids2018")


def train_baseline(file):

    print(f"\nLoading: {file.name}")

    df = pd.read_parquet(file)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Replace infinite values
    df = df.replace([float("inf"), float("-inf")], pd.NA)

    # Remove invalid rows
    df = df.dropna()

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
        "Packet_Length_Std"
    ]

    X = df[feature_columns]
    y = df["is_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nBaseline model trained")

    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test, predictions, zero_division=0):.4f}")


if __name__ == "__main__":

    files = list(GOLD_PATH.glob("*.parquet"))

    print(f"Found {len(files)} Gold files")

    train_baseline(files[0])