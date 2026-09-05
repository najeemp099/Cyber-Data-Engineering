import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmark"
    / "scalability"
    / "scalability_results.csv"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmark"
    / "scalability"
    / "scalability_analysis.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmark"
    / "scalability"
    / "figures"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not RESULTS_FILE.exists():

        raise FileNotFoundError(
            f"\nBenchmark results not found:\n{RESULTS_FILE}\n\n"
            "Run benchmark_scalability.py first."
        )

    df = pd.read_csv(RESULTS_FILE)

    return df


# ============================================================
# ANALYZE SCALABILITY
# ============================================================

def analyze_scalability(df):

    df["training_time_growth"] = (
        df["training_time"] / df["training_time"].iloc[0]
    )

    df["prediction_time_growth"] = (
        df["prediction_time"] / df["prediction_time"].iloc[0]
    )

    df["rows_per_second_training"] = (
        df["train_rows"] / df["training_time"]
    )

    df["rows_per_second_prediction"] = (
        df["test_rows"] / df["prediction_time"]
    )

    return df


# ============================================================
# SAVE ANALYSIS
# ============================================================

def save_analysis(df):

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\nAnalysis saved to:")
    print(REPORT_FILE)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 90)
    print("SCALABILITY ANALYSIS")
    print("=" * 90)

    print(
        f"{'Dataset':<12}"
        f"{'Rows':>12}"
        f"{'Train Time':>15}"
        f"{'Prediction':>15}"
        f"{'F1':>10}"
        f"{'ROC AUC':>10}"
    )

    print("-" * 90)

    for _, row in df.iterrows():

        print(
            f"{row['dataset_percent']:>6.0f}%"
            f"{row['rows']:>12,.0f}"
            f"{row['training_time']:>15.2f}"
            f"{row['prediction_time']:>15.2f}"
            f"{row['f1']:>10.4f}"
            f"{row['roc_auc']:>10.4f}"
        )

    print("=" * 90)

    # Largest dataset
    largest = df.loc[df["rows"].idxmax()]

    # Fastest training
    fastest = df.loc[df["training_time"].idxmin()]

    # Best F1
    best_f1 = df.loc[df["f1"].idxmax()]

    print("\nKey findings:")

    print(
        f"Largest dataset: "
        f"{largest['rows']:,.0f} rows "
        f"({largest['dataset_percent']:.0f}%)"
    )

    print(
        f"Fastest training: "
        f"{fastest['training_time']:.2f}s "
        f"({fastest['dataset_percent']:.0f}%)"
    )

    print(
        f"Best F1 score: "
        f"{best_f1['f1']:.4f} "
        f"({best_f1['dataset_percent']:.0f}%)"
    )


# ============================================================
# CREATE FIGURES
# ============================================================

def create_figures(df):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Training Time vs Dataset Size
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["dataset_percent"],
        df["training_time"],
        marker="o"
    )

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Training Time (seconds)")
    plt.title("Random Forest Training Time vs Dataset Size")

    plt.grid(True)

    plt.savefig(
        FIGURES_DIR / "training_time_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Prediction Time
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["dataset_percent"],
        df["prediction_time"],
        marker="o"
    )

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Prediction Time (seconds)")
    plt.title("Random Forest Prediction Time vs Dataset Size")

    plt.grid(True)

    plt.savefig(
        FIGURES_DIR / "prediction_time_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # F1 Score
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["dataset_percent"],
        df["f1"],
        marker="o"
    )

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("F1 Score")
    plt.title("Detection Performance vs Dataset Size")

    plt.grid(True)

    plt.savefig(
        FIGURES_DIR / "f1_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["dataset_percent"],
        df["roc_auc"],
        marker="o"
    )

    plt.xlabel("Dataset Size (%)")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC vs Dataset Size")

    plt.grid(True)

    plt.savefig(
        FIGURES_DIR / "roc_auc_vs_dataset_size.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nFigures saved to:")
    print(FIGURES_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CyberScaleData - Scalability Result Analysis")
    print("=" * 70)

    df = load_results()

    print("\nBenchmark results loaded.")

    print(f"Experiments: {len(df)}")

    df = analyze_scalability(df)

    save_analysis(df)

    print_summary(df)

    create_figures(df)

    print("\n")
    print("=" * 70)
    print("Scalability analysis completed successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
    