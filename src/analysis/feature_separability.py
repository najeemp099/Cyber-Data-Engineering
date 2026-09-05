from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.metrics import roc_auc_score


GOLD_PATH = "data/gold/cse-cic-ids2018"

OUTPUT_DIR = Path("results/feature_separability")
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

LABEL = "is_attack"


def calculate_effect_size(normal, attack):
    """Calculate Cohen's d effect size."""

    normal = np.asarray(normal, dtype=float)
    attack = np.asarray(attack, dtype=float)

    normal = normal[np.isfinite(normal)]
    attack = attack[np.isfinite(attack)]

    if len(normal) < 2 or len(attack) < 2:
        return 0.0

    mean_normal = np.mean(normal)
    mean_attack = np.mean(attack)

    var_normal = np.var(normal, ddof=1)
    var_attack = np.var(attack, ddof=1)

    pooled_std = np.sqrt(
        (
            (len(normal) - 1) * var_normal
            + (len(attack) - 1) * var_attack
        )
        / (len(normal) + len(attack) - 2)
    )

    if pooled_std == 0:
        return 0.0

    return abs(mean_attack - mean_normal) / pooled_std


def main():

    print("=" * 70)
    print("CyberScaleData - Feature Separability Analysis")
    print("=" * 70)

    start = time.perf_counter()

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-FeatureSeparability")
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

    # ------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # ------------------------------------------------------------

    print("\nRemoving exact duplicates...")

    df = df.dropDuplicates()

    unique_rows = df.count()

    print(f"Unique rows: {unique_rows:,}")
    print(f"Removed rows: {original_rows - unique_rows:,}")

    # ------------------------------------------------------------
    # SELECT FEATURES
    # ------------------------------------------------------------

    print("\nSelecting features...")

    selected = df.select(
        *[
            col(feature).cast("double").alias(feature)
            for feature in FEATURES
        ],
        col(LABEL).cast("double").alias(LABEL)
    )

    print("Converting selected data to Pandas...")

    pandas_df = selected.toPandas()

    print(f"Rows loaded for analysis: {len(pandas_df):,}")

    # ------------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # ------------------------------------------------------------

    print("\nCleaning numeric values...")

    for feature in FEATURES:
        pandas_df[feature] = pd.to_numeric(
            pandas_df[feature],
            errors="coerce"
        )

        pandas_df[feature] = pandas_df[feature].replace(
            [np.inf, -np.inf],
            np.nan
        )

    pandas_df = pandas_df.dropna(
        subset=[LABEL]
    )

    # ------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------

    print("\nClass distribution:")

    normal_count = int(
        (pandas_df[LABEL] == 0).sum()
    )

    attack_count = int(
        (pandas_df[LABEL] == 1).sum()
    )

    print(f"Normal: {normal_count:,}")
    print(f"Attack: {attack_count:,}")

    # ------------------------------------------------------------
    # FEATURE SEPARABILITY
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("UNIVARIATE FEATURE SEPARABILITY")
    print("=" * 70)

    results = []

    for feature in FEATURES:

        print(f"\nAnalyzing: {feature}")

        normal = pandas_df.loc[
            pandas_df[LABEL] == 0,
            feature
        ].dropna()

        attack = pandas_df.loc[
            pandas_df[LABEL] == 1,
            feature
        ].dropna()

        if len(normal) == 0 or len(attack) == 0:
            print("Insufficient data.")
            continue

        # --------------------------------------------------------
        # AUC
        # --------------------------------------------------------

        values = pd.concat(
            [normal, attack],
            ignore_index=True
        )

        labels = np.concatenate(
            [
                np.zeros(len(normal)),
                np.ones(len(attack))
            ]
        )

        try:
            auc = roc_auc_score(
                labels,
                values
            )

            # AUC below 0.5 means the feature is separating
            # in the opposite direction. For separability,
            # use the larger of AUC and 1-AUC.
            separability_auc = max(
                auc,
                1.0 - auc
            )

        except Exception:
            auc = np.nan
            separability_auc = np.nan

        # --------------------------------------------------------
        # STATISTICS
        # --------------------------------------------------------

        normal_mean = normal.mean()
        attack_mean = attack.mean()

        normal_median = normal.median()
        attack_median = attack.median()

        effect_size = calculate_effect_size(
            normal,
            attack
        )

        results.append(
            {
                "feature": feature,
                "normal_count": len(normal),
                "attack_count": len(attack),
                "normal_mean": normal_mean,
                "attack_mean": attack_mean,
                "normal_median": normal_median,
                "attack_median": attack_median,
                "auc": auc,
                "separability_auc": separability_auc,
                "effect_size": effect_size,
            }
        )

        print(
            f"Normal mean:       {normal_mean:.6g}"
        )

        print(
            f"Attack mean:       {attack_mean:.6g}"
        )

        print(
            f"AUC:               {auc:.6f}"
        )

        print(
            f"Separability AUC:  {separability_auc:.6f}"
        )

        print(
            f"Effect size:       {effect_size:.6f}"
        )

    # ------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "separability_auc",
        ascending=False
    )

    results_path = (
        OUTPUT_DIR / "feature_separability_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FEATURE SEPARABILITY RESULTS")
    print("=" * 70)

    display_columns = [
        "feature",
        "separability_auc",
        "effect_size",
        "normal_mean",
        "attack_mean",
    ]

    print(
        results_df[display_columns].to_string(
            index=False
        )
    )

    print(
        f"\nResults saved to: {results_path}"
    )

    # ------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(
        "\nSeparability AUC close to 0.5:"
        "\n  Weak individual feature separation."
    )

    print(
        "\nSeparability AUC close to 1.0:"
        "\n  Strong individual feature separation."
    )

    print(
        "\nEffect size:"
        "\n  < 0.2   = very small"
        "\n  0.2-0.5 = small"
        "\n  0.5-0.8 = medium"
        "\n  > 0.8   = large"
    )

    # ------------------------------------------------------------
    # PLOT AUC
    # ------------------------------------------------------------

    print("\nGenerating AUC plot...")

    plot_df = results_df.sort_values(
        "separability_auc"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["separability_auc"]
    )

    plt.xlabel("Separability AUC")
    plt.ylabel("Feature")
    plt.title(
        "Univariate Feature Separability"
    )

    plt.xlim(0.5, 1.0)

    plt.tight_layout()

    auc_plot = (
        OUTPUT_DIR / "feature_separability_auc.png"
    )

    plt.savefig(
        auc_plot,
        dpi=150
    )

    plt.close()

    print(
        f"AUC plot saved to: {auc_plot}"
    )

    # ------------------------------------------------------------
    # PLOT EFFECT SIZE
    # ------------------------------------------------------------

    print("Generating effect-size plot...")

    plot_df = results_df.sort_values(
        "effect_size"
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["effect_size"]
    )

    plt.xlabel("Absolute Cohen's d")
    plt.ylabel("Feature")
    plt.title(
        "Feature Effect Sizes"
    )

    plt.tight_layout()

    effect_plot = (
        OUTPUT_DIR / "feature_effect_sizes.png"
    )

    plt.savefig(
        effect_plot,
        dpi=150
    )

    plt.close()

    print(
        f"Effect-size plot saved to: {effect_plot}"
    )

    # ------------------------------------------------------------
    # TOP FEATURES
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP FEATURES BY SEPARABILITY")
    print("=" * 70)

    top_features = results_df.head(5)

    for _, row in top_features.iterrows():

        print(
            f"{row['feature']:<35}"
            f" AUC={row['separability_auc']:.6f}"
            f" Effect={row['effect_size']:.4f}"
        )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    total_time = time.perf_counter() - start

    print("\n" + "=" * 70)
    print("FEATURE SEPARABILITY ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"Original rows:       {original_rows:,}")
    print(f"Unique rows:         {unique_rows:,}")
    print(f"Features analyzed:   {len(results_df)}")
    print(
        f"Top feature:         "
        f"{results_df.iloc[0]['feature']}"
    )
    print(
        f"Top separability AUC: "
        f"{results_df.iloc[0]['separability_auc']:.6f}"
    )
    print(
        f"Total execution:     {total_time:.2f}s"
    )

    print("=" * 70)
    print(
        "Feature separability analysis completed successfully"
    )
    print("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()