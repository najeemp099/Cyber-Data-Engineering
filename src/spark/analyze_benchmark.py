from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_PATH = Path("results/spark_benchmark_results.csv")
PLOTS_DIR = Path("results/plots")


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {RESULTS_PATH}"
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)

    # Preserve the intended benchmark order.
    order = ["10%", "25%", "50%", "75%", "100%"]
    df["dataset_size"] = pd.Categorical(
        df["dataset_size"],
        categories=order,
        ordered=True
    )
    df = df.sort_values("dataset_size")

    # 1. Dataset size vs training time
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["rows"],
        df["training_time_sec"],
        marker="o"
    )
    plt.xlabel("Number of Rows")
    plt.ylabel("Training Time (seconds)")
    plt.title("Dataset Size vs Logistic Regression Training Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_vs_training_time.png", dpi=200)
    plt.close()

    # 2. Dataset size vs prediction time
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["rows"],
        df["prediction_time_sec"],
        marker="o"
    )
    plt.xlabel("Number of Rows")
    plt.ylabel("Prediction Time (seconds)")
    plt.title("Dataset Size vs Prediction Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_vs_prediction_time.png", dpi=200)
    plt.close()

    # 3. Dataset size vs total ML time
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["rows"],
        df["total_ml_time_sec"],
        marker="o"
    )
    plt.xlabel("Number of Rows")
    plt.ylabel("Total ML Time (seconds)")
    plt.title("Dataset Size vs Total ML Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_vs_total_ml_time.png", dpi=200)
    plt.close()

    # 4. Dataset size vs model quality
    plt.figure(figsize=(8, 5))
    plt.plot(
        df["rows"],
        df["f1"],
        marker="o",
        label="F1 Score"
    )
    plt.plot(
        df["rows"],
        df["roc_auc"],
        marker="o",
        label="ROC AUC"
    )
    plt.xlabel("Number of Rows")
    plt.ylabel("Score")
    plt.title("Dataset Size vs Model Performance")
    plt.ylim(0.9, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dataset_vs_model_performance.png", dpi=200)
    plt.close()

    # Print concise analysis
    print("=" * 70)
    print("CyberScaleData - Benchmark Analysis")
    print("=" * 70)

    print("\nBenchmark Results:")
    print(
        df[
            [
                "dataset_size",
                "rows",
                "training_time_sec",
                "prediction_time_sec",
                "total_ml_time_sec",
                "accuracy",
                "f1",
                "roc_auc"
            ]
        ].to_string(index=False)
    )

    print("\nScalability Analysis:")
    print("-" * 70)

    first = df.iloc[0]
    last = df.iloc[-1]

    print(
        f"Rows: {int(first['rows']):,} -> "
        f"{int(last['rows']):,}"
    )

    print(
        f"Training time: {first['training_time_sec']:.2f}s -> "
        f"{last['training_time_sec']:.2f}s"
    )

    print(
        f"Prediction time: {first['prediction_time_sec']:.2f}s -> "
        f"{last['prediction_time_sec']:.2f}s"
    )

    print(
        f"F1 score: {first['f1']:.4f} -> "
        f"{last['f1']:.4f}"
    )

    print(
        f"ROC AUC: {first['roc_auc']:.4f} -> "
        f"{last['roc_auc']:.4f}"
    )

    print("\nPlots created:")
    for path in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  {path}")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
