from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


GOLD_PATH = "data/gold/cse-cic-ids2018"

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


def main():

    print("=" * 70)
    print("CyberScaleData - Random Forest Generalization Validation")
    print("=" * 70)

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-Generalization")
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

    start = time.perf_counter()

    df = spark.read.parquet(str(file))

    original_rows = df.count()

    print(f"Original rows: {original_rows:,}")

    # ------------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # ------------------------------------------------------------

    print("\nRemoving exact duplicates...")

    df = df.dropDuplicates()

    unique_rows = df.count()

    print(f"Unique rows:   {unique_rows:,}")
    print(
        f"Removed rows:  {original_rows - unique_rows:,}"
    )

    # ------------------------------------------------------------
    # FEATURE PREPARATION
    # ------------------------------------------------------------

    print("\nPreparing features...")

    assembler = VectorAssembler(
        inputCols=FEATURES,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    assembled = assembler.transform(df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    scaler_model = scaler.fit(assembled)

    prepared = scaler_model.transform(assembled).select(
        "features",
        col(LABEL).cast("double").alias("label")
    )

    prepared = prepared.cache()

    prepared_rows = prepared.count()

    print(f"Prepared rows: {prepared_rows:,}")

    # ------------------------------------------------------------
    # ORDERED SPLIT
    # ------------------------------------------------------------

    print("\nCreating ordered train/test split...")

    indexed = (
        prepared.rdd
        .zipWithIndex()
        .map(
            lambda x: (
                x[1],
                x[0].features,
                x[0].label
            )
        )
        .toDF(
            ["row_index", "features", "label"]
        )
    )

    split_index = int(prepared_rows * 0.8)

    train = indexed.filter(
        col("row_index") < split_index
    ).drop("row_index")

    test = indexed.filter(
        col("row_index") >= split_index
    ).drop("row_index")

    train = train.cache()
    test = test.cache()

    train_rows = train.count()
    test_rows = test.count()

    print(f"Training rows: {train_rows:,}")
    print(f"Testing rows:  {test_rows:,}")

    # ------------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------------

    print("\nTraining class distribution:")

    train.groupBy("label").count().orderBy("label").show()

    print("Testing class distribution:")

    test.groupBy("label").count().orderBy("label").show()

    # ------------------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------------------

    print("=" * 70)
    print("RANDOM FOREST GENERALIZATION TEST")
    print("=" * 70)

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=50,
        maxDepth=10,
        seed=42
    )

    print("\nTraining Random Forest...")

    train_start = time.perf_counter()

    model = rf.fit(train)

    training_time = time.perf_counter() - train_start

    print(
        f"Training time: {training_time:.4f}s"
    )

    # ------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------

    print("\nRunning predictions...")

    prediction_start = time.perf_counter()

    predictions = model.transform(test).cache()

    prediction_count = predictions.count()

    prediction_time = (
        time.perf_counter() - prediction_start
    )

    print(f"Predictions: {prediction_count:,}")
    print(
        f"Prediction time: {prediction_time:.4f}s"
    )

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------

    print("\nEvaluating model...")

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    roc_auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    roc_auc = roc_auc_evaluator.evaluate(predictions)

    # ------------------------------------------------------------
    # PRECISION AND RECALL
    # ------------------------------------------------------------

    tp = predictions.filter(
        (col("label") == 1.0) &
        (col("prediction") == 1.0)
    ).count()

    fp = predictions.filter(
        (col("label") == 0.0) &
        (col("prediction") == 1.0)
    ).count()

    fn = predictions.filter(
        (col("label") == 1.0) &
        (col("prediction") == 0.0)
    ).count()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # ------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------

    print("\nModel Performance")
    print("-" * 40)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    print("\nConfusion Matrix:")

    predictions.groupBy(
        "label",
        "prediction"
    ).count().orderBy(
        "label",
        "prediction"
    ).show()

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------

    print("\nFeature Importance")
    print("-" * 40)

    importance = model.featureImportances

    feature_importance = sorted(
        zip(FEATURES, importance),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, score in feature_importance:
        print(
            f"{feature:<35} {score:.6f}"
        )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    total_time = time.perf_counter() - start

    print("\n" + "=" * 70)
    print("GENERALIZATION VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Original rows:       {original_rows:,}")
    print(f"Unique rows:         {unique_rows:,}")
    print(f"Training rows:       {train_rows:,}")
    print(f"Testing rows:        {test_rows:,}")
    print(f"Training time:       {training_time:.4f}s")
    print(f"Prediction time:     {prediction_time:.4f}s")
    print(f"F1 Score:            {f1:.4f}")
    print(f"ROC AUC:             {roc_auc:.4f}")
    print(f"Total execution:     {total_time:.4f}s")

    print("=" * 70)
    print("Generalization validation completed successfully")
    print("=" * 70)

    predictions.unpersist()
    train.unpersist()
    test.unpersist()
    prepared.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()