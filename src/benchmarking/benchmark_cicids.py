from pathlib import Path
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


GOLD_PATH = Path("data/gold/cse-cic-ids2018")


FEATURE_COLUMNS = [
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


def benchmark_dataset(df, size):

    df = df.iloc[:size].copy()

    X = df[FEATURE_COLUMNS]
    y = df["is_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    return training_time


if __name__ == "__main__":

    files = list(GOLD_PATH.glob("*.parquet"))

    print(f"Found {len(files)} Gold files")

    file = files[0]

    print(f"\nLoading: {file.name}")

    df = pd.read_parquet(file)

    print(f"Total rows: {len(df)}")

    dataset_sizes = [
        100_000,
        250_000,
        500_000,
        750_000
    ]

    results = []

    for size in dataset_sizes:

        if size > len(df):
            continue

        print(f"\nBenchmarking {size:,} rows...")

        training_time = benchmark_dataset(df, size)

        print(
            f"Dataset size: {size:,} | "
            f"Training time: {training_time:.4f} seconds"
        )

        results.append({
            "algorithm": "LogisticRegression",
            "dataset_size": size,
            "training_time": training_time
        })

    results_df = pd.DataFrame(results)

    output_path = Path("data/gold/cicids_benchmark_results.csv")

    results_df.to_csv(output_path, index=False)

    print("\nBenchmark results saved:")
    print(output_path)

    print("\nBenchmark Summary:")
    print(results_df)

    results_df["time_per_row"] = (
        results_df["training_time"] /
        results_df["dataset_size"]
    )

    print("\nTime per row:")
    print(results_df[["dataset_size", "time_per_row"]])