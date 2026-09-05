from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession


# ============================================================
# CONFIGURATION
# ============================================================

GOLD_PATH = "data/gold/cse-cic-ids2018"

OUTPUT_DIR = Path("results/feature_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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
]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CyberScaleData - Feature Correlation Analysis")
    print("=" * 70)

    start = time.perf_counter()

    # --------------------------------------------------------
    # SPARK SESSION
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-FeatureCorrelation")
        .master("local[*]")
        .config(
            "spark.hadoop.fs.file.impl",
            "org.apache.hadoop.fs.RawLocalFileSystem"
        )
        .config(
            "spark.hadoop.fs.file.impl.disable.cache",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading Gold dataset...")

    files = list(Path(GOLD_PATH).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {GOLD_PATH}"
        )

    file = files[0]

    print(f"File: {file.name}")

    df = spark.read.parquet(str(file))

    original_rows = df.count()

    print(f"Original rows: {original_rows:,}")

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    print("\nRemoving exact duplicates...")

    df = df.dropDuplicates()

    unique_rows = df.count()

    removed_rows = original_rows - unique_rows

    print(f"Unique rows:   {unique_rows:,}")
    print(f"Removed rows:  {removed_rows:,}")

    # --------------------------------------------------------
    # SELECT FEATURES
    # --------------------------------------------------------

    print("\nSelecting features...")

    selected = df.select(FEATURES)

    # --------------------------------------------------------
    # CONVERT TO PANDAS
    # --------------------------------------------------------

    print("Converting selected data to Pandas...")

    pandas_df = selected.toPandas()

    print(
        f"Rows loaded for correlation analysis: "
        f"{len(pandas_df):,}"
    )

    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    print("\nCleaning numeric values...")

    pandas_df = pandas_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    pandas_df = pandas_df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    pandas_df = pandas_df.dropna()

    print(
        f"Rows after cleaning: {len(pandas_df):,}"
    )

    # --------------------------------------------------------
    # CORRELATION MATRIX
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PEARSON CORRELATION ANALYSIS")
    print("=" * 70)

    correlation_matrix = pandas_df.corr(
        method="pearson"
    )

    print("\nCorrelation Matrix:")
    print("-" * 70)

    print(
        correlation_matrix.round(4).to_string()
    )

    # --------------------------------------------------------
    # SAVE CORRELATION MATRIX
    # --------------------------------------------------------

    matrix_path = (
        OUTPUT_DIR /
        "feature_correlation_matrix.csv"
    )

    correlation_matrix.to_csv(matrix_path)

    print(
        f"\nCorrelation matrix saved to: {matrix_path}"
    )

    # --------------------------------------------------------
    # FIND HIGH CORRELATION PAIRS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("HIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 70)

    pairs = []

    for i in range(len(FEATURES)):

        for j in range(i + 1, len(FEATURES)):

            feature_1 = FEATURES[i]
            feature_2 = FEATURES[j]

            correlation = correlation_matrix.loc[
                feature_1,
                feature_2
            ]

            if abs(correlation) >= 0.70:

                pairs.append(
                    (
                        feature_1,
                        feature_2,
                        correlation
                    )
                )

    pairs.sort(
        key=lambda x: abs(x[2]),
        reverse=True
    )

    if pairs:

        print(
            f"\nFound {len(pairs)} highly correlated pairs:\n"
        )

        for feature_1, feature_2, correlation in pairs:

            print(
                f"{feature_1:<35} "
                f"{feature_2:<35} "
                f"{correlation:.4f}"
            )

    else:

        print(
            "\nNo feature pairs exceeded "
            "the |correlation| >= 0.70 threshold."
        )

    # --------------------------------------------------------
    # SAVE HIGH CORRELATION PAIRS
    # --------------------------------------------------------

    pair_df = pd.DataFrame(
        pairs,
        columns=[
            "feature_1",
            "feature_2",
            "correlation"
        ]
    )

    pair_path = (
        OUTPUT_DIR /
        "high_correlation_pairs.csv"
    )

    pair_df.to_csv(
        pair_path,
        index=False
    )

    print(
        f"\nHigh-correlation pairs saved to: {pair_path}"
    )

    # --------------------------------------------------------
    # GENERATE HEATMAP
    # --------------------------------------------------------

    print("\nGenerating correlation heatmap...")

    plt.figure(
        figsize=(12, 9)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )

    plt.title(
        "Feature Correlation Matrix"
    )

    plt.tight_layout()

    heatmap_path = (
        OUTPUT_DIR /
        "feature_correlation_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=200
    )

    plt.close()

    print(
        f"Heatmap saved to: {heatmap_path}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 70)
    print("FEATURE CORRELATION ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Original rows:       {original_rows:,}"
    )

    print(
        f"Unique rows:         {unique_rows:,}"
    )

    print(
        f"Rows analyzed:       {len(pandas_df):,}"
    )

    print(
        f"Features analyzed:   {len(FEATURES)}"
    )

    print(
        f"Highly correlated pairs: {len(pairs)}"
    )

    if pairs:

        strongest = pairs[0]

        print(
            f"Strongest pair:      "
            f"{strongest[0]} <-> {strongest[1]}"
        )

        print(
            f"Correlation:         "
            f"{strongest[2]:.4f}"
        )

    print(
        f"Total execution:     {elapsed:.2f}s"
    )

    print("=" * 70)
    print(
        "Feature correlation analysis completed successfully"
    )
    print("=" * 70)

    spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()