from pathlib import Path
import time

import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


GOLD_PATH = "data/gold/cse-cic-ids2018"

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

SEED = 42


def create_spark():

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-RandomForest-Validation")
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

    return spark


def prepare_data(df):

    ml_df = df.select(
        *FEATURE_COLUMNS,
        "is_attack"
    )

    for feature in FEATURE_COLUMNS:
        ml_df = ml_df.withColumn(
            feature,
            when(
                col(feature).isNull() | isnan(col(feature)),
                0.0
            ).otherwise(
                col(feature).cast("double")
            )
        )

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    assembled = assembler.transform(ml_df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    scaler_model = scaler.fit(assembled)

    prepared = (
        scaler_model
        .transform(assembled)
        .select(
            "features",
            col("is_attack").cast("double").alias("label")
        )
    )

    return prepared


def main():

    print("=" * 70)
    print("CyberScaleData - Random Forest Validation")
    print("=" * 70)

    spark = create_spark()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading Gold dataset...")

    files = list(
        Path(GOLD_PATH).glob("*.parquet")
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {GOLD_PATH}"
        )

    file = files[0]

    print(f"File: {file.name}")

    df = spark.read.parquet(str(file)).cache()

    total_rows = df.count()

    print(f"Total rows: {total_rows:,}")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("1. CLASS DISTRIBUTION")
    print("=" * 70)

    class_df = (
        df.groupBy("is_attack")
        .count()
        .orderBy("is_attack")
    )

    class_df.show()

    class_pd = class_df.toPandas()

    total = class_pd["count"].sum()

    for _, row in class_pd.iterrows():

        label = int(row["is_attack"])
        count = int(row["count"])
        percentage = count / total * 100

        name = "Attack" if label == 1 else "Normal"

        print(
            f"{name}: {count:,} "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("2. DUPLICATE CHECK")
    print("=" * 70)

    duplicate_start = time.perf_counter()

    feature_label_columns = FEATURE_COLUMNS + ["is_attack"]

    total_distinct = (
        df.select(*feature_label_columns)
        .distinct()
        .count()
    )

    duplicate_count = total_rows - total_distinct

    duplicate_time = (
        time.perf_counter() - duplicate_start
    )

    duplicate_percentage = (
        duplicate_count / total_rows * 100
    )

    print(f"Total rows:       {total_rows:,}")
    print(f"Distinct rows:    {total_distinct:,}")
    print(f"Duplicate rows:   {duplicate_count:,}")
    print(f"Duplicate rate:   {duplicate_percentage:.4f}%")
    print(f"Check time:       {duplicate_time:.2f}s")

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("3. FEATURE PREPARATION")
    print("=" * 70)

    preparation_start = time.perf_counter()

    prepared = prepare_data(df).cache()

    prepared_count = prepared.count()

    preparation_time = (
        time.perf_counter() - preparation_start
    )

    print(f"Prepared rows: {prepared_count:,}")
    print(f"Preparation time: {preparation_time:.2f}s")

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    train_df, test_df = prepared.randomSplit(
        [0.8, 0.2],
        seed=SEED
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print(f"Training rows: {train_count:,}")
    print(f"Testing rows:  {test_count:,}")

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("4. RANDOM FOREST VALIDATION")
    print("=" * 70)

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=20,
        maxDepth=10,
        seed=SEED
    )

    train_start = time.perf_counter()

    model = rf.fit(train_df)

    train_time = time.perf_counter() - train_start

    print(f"Training time: {train_time:.4f}s")

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    prediction_start = time.perf_counter()

    predictions = model.transform(test_df).cache()

    prediction_count = predictions.count()

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    print(f"Predictions: {prediction_count:,}")
    print(f"Prediction time: {prediction_time:.4f}s")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    roc_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    print("\nModel Performance")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .show()
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5. FEATURE IMPORTANCE")
    print("=" * 70)

    importances = model.featureImportances.toArray()

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    print(
        importance_df.to_string(
            index=False,
            formatters={
                "importance": lambda x: f"{x:.6f}"
            }
        )
    )

    # --------------------------------------------------------
    # SAVE FEATURE IMPORTANCE
    # --------------------------------------------------------

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    importance_path = (
        output_dir / "random_forest_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_path,
        index=False
    )

    print(
        f"\nFeature importance saved to: "
        f"{importance_path}"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Dataset rows:       {total_rows:,}")
    print(f"Distinct rows:      {total_distinct:,}")
    print(f"Duplicate rows:     {duplicate_count:,}")
    print(f"Duplicate rate:     {duplicate_percentage:.4f}%")
    print(f"Training rows:      {train_count:,}")
    print(f"Testing rows:       {test_count:,}")
    print(f"Training time:      {train_time:.4f}s")
    print(f"Prediction time:    {prediction_time:.4f}s")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC AUC:            {roc_auc:.4f}")

    print("\nValidation completed successfully.")
    print("=" * 70)

    predictions.unpersist()
    prepared.unpersist()
    df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()