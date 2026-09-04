import time
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


GOLD_FILE = Path("data/gold/network_traffic_features.parquet")


def benchmark():
   
    df = pd.read_parquet(GOLD_FILE)

    X = df[
        [
            "src_port",
            "dst_port",
            "packet_count",
            "bytes",
            "duration_bytes"
        ]
    ]

    y = df["is_attack"]

    results = []

    for size in [5, 10, 25, 50]:

        X_test = X.sample(n=size, replace=True, random_state=42)
        y_test = y.loc[X_test.index]

        model = LogisticRegression(max_iter=1000)

        start_time = time.perf_counter()

        model.fit(X_test, y_test)

        end_time = time.perf_counter()

        print(
            f"Dataset size: {size} | "
            f"Training time: {end_time - start_time:.6f} seconds"
        )
        results.append({
            "algorithm": "LogisticRegression",
            "dataset_size": size,
            "training_time": end_time - start_time
    })
    results_df = pd.DataFrame(results)

    results_path = Path("data/gold/benchmark_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\nBenchmark results saved: {results_path}")

    print("\nBenchmark Summary:")
    print(results_df)

    results_df["time_per_row"] = (
    results_df["training_time"] / results_df["dataset_size"]
)

    print("\nTime per row:")
    print(results_df[["dataset_size", "time_per_row"]])



if __name__ == "__main__":
    benchmark()