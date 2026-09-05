from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


LOGISTIC_PATH = Path("results/spark_benchmark_results.csv")
RF_PATH = Path("results/spark_random_forest_results.csv")
OUTPUT_PATH = Path("results/combined_benchmark_results.csv")
PLOTS_DIR = Path("results/plots")


def load_results(path):
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    # The Logistic Regression benchmark may not contain an algorithm column.
    if "algorithm" not in df.columns:
        df["algorithm"] = "Logistic Regression"

    return df


def main():
    print("=" * 70)
    print("CyberScaleData - Algorithm Scalability Comparison")
    print("=" * 70)

    logistic = load_results(LOGISTIC_PATH)
    random_forest = load_results(RF_PATH)

    df = pd.concat(
        [logistic, random_forest],
        ignore_index=True
    )

    order = ["10%", "25%", "50%", "75%", "100%"]

    df["dataset_size"] = pd.Categorical(
        df["dataset_size"],
        categories=order,
        ordered=True
    )

    df = df.sort_values(
        ["dataset_size", "algorithm"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCombined results saved to: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    print("\nTraining Time Comparison:")
    print("-" * 70)

    training = df.pivot(
        index="dataset_size",
        columns="algorithm",
        values="training_time_sec"
    )

    print(training.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nTotal ML Time Comparison:")
    print("-" * 70)

    total = df.pivot(
        index="dataset_size",
        columns="algorithm",
        values="total_ml_time_sec"
    )

    print(total.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nModel Performance:")
    print("-" * 70)

    performance = df[
        [
            "dataset_size",
            "algorithm",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc"
        ]
    ]

    print(
        performance.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # PLOT 1: TRAINING TIME
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for algorithm in df["algorithm"].unique():
        subset = df[df["algorithm"] == algorithm]

        plt.plot(
            subset["rows"],
            subset["training_time_sec"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Number of Rows")
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time vs Dataset Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "algorithm_training_time_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # PLOT 2: TOTAL ML TIME
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for algorithm in df["algorithm"].unique():
        subset = df[df["algorithm"] == algorithm]

        plt.plot(
            subset["rows"],
            subset["total_ml_time_sec"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Number of Rows")
    plt.ylabel("Total ML Time (seconds)")
    plt.title("Total ML Time vs Dataset Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "algorithm_total_ml_time_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # PLOT 3: F1 SCORE
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for algorithm in df["algorithm"].unique():
        subset = df[df["algorithm"] == algorithm]

        plt.plot(
            subset["rows"],
            subset["f1"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Number of Rows")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Dataset Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "algorithm_f1_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # PLOT 4: ROC AUC
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    for algorithm in df["algorithm"].unique():
        subset = df[df["algorithm"] == algorithm]

        plt.plot(
            subset["rows"],
            subset["roc_auc"],
            marker="o",
            label=algorithm
        )

    plt.xlabel("Number of Rows")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC vs Dataset Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "algorithm_roc_auc_comparison.png",
        dpi=200
    )
    plt.close()

    # --------------------------------------------------------
    # FINAL OBSERVATIONS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("KEY OBSERVATIONS")
    print("=" * 70)

    lr_100 = df[
        (df["algorithm"] == "Logistic Regression") &
        (df["dataset_size"] == "100%")
    ].iloc[0]

    rf_100 = df[
        (df["algorithm"] == "Random Forest") &
        (df["dataset_size"] == "100%")
    ].iloc[0]

    training_ratio = (
        rf_100["training_time_sec"] /
        lr_100["training_time_sec"]
    )

    total_ratio = (
        rf_100["total_ml_time_sec"] /
        lr_100["total_ml_time_sec"]
    )

    print(
        f"\nAt 100% ({int(lr_100['rows']):,} rows):"
    )

    print(
        f"Logistic Regression training: "
        f"{lr_100['training_time_sec']:.2f}s"
    )

    print(
        f"Random Forest training: "
        f"{rf_100['training_time_sec']:.2f}s"
    )

    print(
        f"Random Forest training is approximately "
        f"{training_ratio:.2f}x the Logistic Regression training time."
    )

    print(
        f"\nLogistic Regression total ML time: "
        f"{lr_100['total_ml_time_sec']:.2f}s"
    )

    print(
        f"Random Forest total ML time: "
        f"{rf_100['total_ml_time_sec']:.2f}s"
    )

    print(
        f"Random Forest total ML time is approximately "
        f"{total_ratio:.2f}x the Logistic Regression total ML time."
    )

    print(
        f"\nLogistic Regression F1: "
        f"{lr_100['f1']:.4f}"
    )

    print(
        f"Random Forest F1: "
        f"{rf_100['f1']:.4f}"
    )

    print(
        f"\nLogistic Regression ROC AUC: "
        f"{lr_100['roc_auc']:.4f}"
    )

    print(
        f"Random Forest ROC AUC: "
        f"{rf_100['roc_auc']:.4f}"
    )

    print("\nGenerated comparison plots:")

    for path in sorted(
        PLOTS_DIR.glob("algorithm_*_comparison.png")
    ):
        print(f"  {path}")

    print("\nComparison completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
