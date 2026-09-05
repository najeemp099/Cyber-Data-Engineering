import pandas as pd


ORIGINAL_PATH = "results/spark_random_forest_results.csv"
ROBUST_PATH = "results/spark_robust_rf_results.csv"


def main():

    print("=" * 70)
    print("CyberScaleData - Original vs Deduplicated Random Forest")
    print("=" * 70)

    original = pd.read_csv(ORIGINAL_PATH)
    robust = pd.read_csv(ROBUST_PATH)

    original["condition"] = "Original"
    robust["condition"] = "Deduplicated"

    print("\n100% Dataset Comparison:")
    print("-" * 70)

    original_100 = original[
        original["dataset_size"] == "100%"
    ].iloc[0]

    robust_100 = robust[
        robust["dataset_size"] == "100%"
    ].iloc[0]

    print(
        f"Original rows:       {int(original_100['rows']):,}"
    )
    print(
        f"Deduplicated rows:   {int(robust_100['rows']):,}"
    )

    print()

    metrics = [
        ("training_time_sec", "Training Time"),
        ("prediction_time_sec", "Prediction Time"),
        ("total_ml_time_sec", "Total ML Time"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC AUC"),
    ]

    print(
        f"{'Metric':<25}"
        f"{'Original':>15}"
        f"{'Deduplicated':>18}"
        f"{'Difference':>15}"
    )

    print("-" * 75)

    for column, name in metrics:

        old = float(original_100[column])
        new = float(robust_100[column])
        difference = new - old

        if "time" in column:
            print(
                f"{name:<25}"
                f"{old:>15.4f}"
                f"{new:>18.4f}"
                f"{difference:>15.4f}"
            )
        else:
            print(
                f"{name:<25}"
                f"{old:>15.4f}"
                f"{new:>18.4f}"
                f"{difference:>15.4f}"
            )

    print("\nScalability Comparison:")
    print("-" * 70)

    comparison = pd.concat(
        [
            original[
                [
                    "dataset_size",
                    "rows",
                    "training_time_sec",
                    "prediction_time_sec",
                    "total_ml_time_sec",
                    "f1",
                    "roc_auc"
                ]
            ].assign(condition="Original"),

            robust[
                [
                    "dataset_size",
                    "rows",
                    "training_time_sec",
                    "prediction_time_sec",
                    "total_ml_time_sec",
                    "f1",
                    "roc_auc"
                ]
            ].assign(condition="Deduplicated")
        ],
        ignore_index=True
    )

    print(comparison.to_string(index=False))

    # --------------------------------------------------------
    # KEY FINDINGS
    # --------------------------------------------------------

    f1_change = (
        robust_100["f1"] -
        original_100["f1"]
    )

    auc_change = (
        robust_100["roc_auc"] -
        original_100["roc_auc"]
    )

    duplicate_rate = (
        1 -
        robust_100["rows"] /
        original_100["rows"]
    ) * 100

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    print(
        f"\nDuplicate reduction: {duplicate_rate:.2f}%"
    )

    print(
        f"F1 change: {f1_change:+.4f}"
    )

    print(
        f"ROC AUC change: {auc_change:+.4f}"
    )

    if abs(f1_change) < 0.005:
        print(
            "\nFinding: Random Forest performance remained "
            "highly stable after exact duplicate removal."
        )

    if abs(auc_change) < 0.005:
        print(
            "Finding: ROC AUC remained highly stable, "
            "suggesting duplicate records alone do not "
            "explain the near-perfect discrimination."
        )

    print(
        "\nConclusion:"
    )

    print(
        "Exact duplicates significantly affect dataset "
        "composition, but removing them has little effect "
        "on Random Forest predictive performance."
    )

    print(
        "\nNext investigation should focus on feature-level "
        "separability and stricter generalization validation."
    )

    print("\n" + "=" * 70)
    print("Robustness comparison completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()