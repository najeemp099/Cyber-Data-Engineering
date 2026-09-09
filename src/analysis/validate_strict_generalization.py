from pathlib import Path
from time import perf_counter
import csv

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


# ============================================================
# CyberScaleData - Strict Random Forest Generalization
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "cse-cic-ids2018"
    / "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "strict_generalization"

FEATURE_COLUMNS = [
    "Fwd_Packets_Length_Total",
    "Packet_Length_Std",
    "Packet_Length_Mean",
    "Total_Backward_Packets",
    "Flow_Bytes_per_s",
    "Flow_Packets_per_s",
    "Bwd_Packets_Length_Total",
    "Total_Fwd_Packets",
    "Flow_Duration",
    "Protocol",
]

LABEL_COLUMN = "is_attack"


def main():

    # ------------------------------------------------------------
    # Start Spark
    # ------------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-Strict-Generalization")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("CyberScaleData - Strict Random Forest Generalization")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load Gold dataset
    # ------------------------------------------------------------

    print("\nLoading Gold dataset...")
    print(f"File: {DATA_PATH.name}")

    if not DATA_PATH.exists():
        spark.stop()

        raise FileNotFoundError(
            f"\nGold dataset not found:\n{DATA_PATH}\n\n"
            "Check that the CICIDS Gold parquet file exists."
        )

    df = spark.read.parquet(str(DATA_PATH))

    original_rows = df.count()

    print(f"Original rows: {original_rows:,}")
    print(f"Target column: {LABEL_COLUMN}")

    # ------------------------------------------------------------
    # 2. Remove exact duplicates
    # ------------------------------------------------------------

    print("\nRemoving exact duplicates...")

    df = df.dropDuplicates()

    unique_rows = df.count()

    duplicates_removed = original_rows - unique_rows

    print(f"Unique rows:   {unique_rows:,}")
    print(f"Removed rows:  {duplicates_removed:,}")

    # ------------------------------------------------------------
    # 3. Validate required columns
    # ------------------------------------------------------------

    print("\nValidating required columns...")

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        spark.stop()

        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {df.columns}"
        )

    print("All required columns are present.")

    # ------------------------------------------------------------
    # 4. Prepare features
    # ------------------------------------------------------------

    print("\nPreparing features...")

    prepared = df.select(
        *[
            col(c).cast("double").alias(c)
            for c in FEATURE_COLUMNS
        ],
        col(LABEL_COLUMN).cast("double").alias("label"),
    )

    prepared = prepared.na.drop()

    prepared_rows = prepared.count()

    print(f"Prepared rows: {prepared_rows:,}")

    # ------------------------------------------------------------
    # 5. Strict ordered train/test split
    # ------------------------------------------------------------

    print("\nCreating STRICT ordered train/test split...")
    print("No random sampling is used.")

    train_fraction = 0.06

    train_count = int(
        prepared_rows * train_fraction
    )

    train_df = prepared.limit(train_count)

    test_df = prepared.subtract(train_df)

    test_count = test_df.count()

    print(f"Training rows: {train_count:,}")
    print(f"Testing rows:  {test_count:,}")

    # ------------------------------------------------------------
    # 6. Class distributions
    # ------------------------------------------------------------

    print("\nTraining class distribution:")

    (
        train_df
        .groupBy("label")
        .count()
        .orderBy("label")
        .show()
    )

    print("Testing class distribution:")

    (
        test_df
        .groupBy("label")
        .count()
        .orderBy("label")
        .show()
    )

    # ------------------------------------------------------------
    # 7. Random Forest pipeline
    # ------------------------------------------------------------

    print("=" * 70)
    print("STRICT GENERALIZATION RANDOM FOREST TEST")
    print("=" * 70)

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip",
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42,
    )

    pipeline = Pipeline(
        stages=[
            assembler,
            rf,
        ]
    )

    # ------------------------------------------------------------
    # 8. Train model
    # ------------------------------------------------------------

    print("\nTraining Random Forest...")

    train_start = perf_counter()

    model = pipeline.fit(train_df)

    training_time = (
        perf_counter() - train_start
    )

    print(
        f"Training time: {training_time:.4f}s"
    )

    # ------------------------------------------------------------
    # 9. Predictions
    # ------------------------------------------------------------

    print("\nRunning predictions...")

    prediction_start = perf_counter()

    predictions = model.transform(test_df)

    prediction_count = predictions.count()

    prediction_time = (
        perf_counter() - prediction_start
    )

    print(f"Predictions: {prediction_count:,}")
    print(
        f"Prediction time: {prediction_time:.4f}s"
    )

    # ------------------------------------------------------------
    # 10. Evaluation
    # ------------------------------------------------------------

    print("\nEvaluating model...")

    accuracy_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy",
        )
    )

    precision_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision",
        )
    )

    recall_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall",
        )
    )

    f1_evaluator = (
        MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1",
        )
    )

    roc_evaluator = (
        BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
    )

    accuracy = accuracy_evaluator.evaluate(
        predictions
    )

    precision = precision_evaluator.evaluate(
        predictions
    )

    recall = recall_evaluator.evaluate(
        predictions
    )

    f1 = f1_evaluator.evaluate(
        predictions
    )

    roc_auc = roc_evaluator.evaluate(
        predictions
    )

    print("\nModel Performance")
    print("-" * 40)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # ------------------------------------------------------------
    # 11. Confusion matrix
    # ------------------------------------------------------------

    print("\nConfusion Matrix:")

    (
        predictions
        .groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
        .show()
    )

    # ------------------------------------------------------------
    # 12. Feature importance
    # ------------------------------------------------------------

    print("\nFeature Importance")
    print("-" * 40)

    rf_model = model.stages[-1]

    feature_importance = list(
        zip(
            FEATURE_COLUMNS,
            rf_model.featureImportances.toArray(),
        )
    )

    feature_importance.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in feature_importance:

        print(
            f"{feature:<35} "
            f"{importance:.6f}"
        )

    # ------------------------------------------------------------
    # 13. Save results
    # ------------------------------------------------------------

    print("\nSaving results...")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_ml_time = (
        training_time + prediction_time
    )

    # ------------------------------------------------------------
    # 13A. Save summary CSV
    # ------------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "strict_generalization_results.csv"
    )

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "original_rows",
                "unique_rows",
                "duplicates_removed",
                "prepared_rows",
                "training_rows",
                "testing_rows",
                "training_time_seconds",
                "prediction_time_seconds",
                "total_ml_time_seconds",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        )

        writer.writerow(
            [
                original_rows,
                unique_rows,
                duplicates_removed,
                prepared_rows,
                train_count,
                prediction_count,
                training_time,
                prediction_time,
                total_ml_time,
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
            ]
        )

    print(
        f"Saved: {results_path}"
    )

    # ------------------------------------------------------------
    # 13B. Save feature importance CSV
    # ------------------------------------------------------------

    importance_path = (
        RESULTS_DIR
        / "strict_feature_importance.csv"
    )

    with open(
        importance_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "feature",
                "importance",
            ]
        )

        for feature, importance in feature_importance:

            writer.writerow(
                [
                    feature,
                    importance,
                ]
            )

    print(
        f"Saved: {importance_path}"
    )

    # ------------------------------------------------------------
    # 14. Final summary
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STRICT GENERALIZATION SUMMARY")
    print("=" * 70)

    print(
        f"Original rows:       {original_rows:,}"
    )

    print(
        f"Unique rows:         {unique_rows:,}"
    )

    print(
        f"Duplicates removed:  {duplicates_removed:,}"
    )

    print(
        f"Prepared rows:       {prepared_rows:,}"
    )

    print(
        f"Training rows:       {train_count:,}"
    )

    print(
        f"Testing rows:        {prediction_count:,}"
    )

    print(
        f"Training time:       {training_time:.4f}s"
    )

    print(
        f"Prediction time:     {prediction_time:.4f}s"
    )

    print(
        f"Accuracy:            {accuracy:.4f}"
    )

    print(
        f"Precision:           {precision:.4f}"
    )

    print(
        f"Recall:              {recall:.4f}"
    )

    print(
        f"F1 Score:            {f1:.4f}"
    )

    print(
        f"ROC AUC:             {roc_auc:.4f}"
    )

    print(
        f"Total ML time:       {total_ml_time:.4f}s"
    )

    print("\nResults saved to:")

    print(results_path)

    print(importance_path)

    print("=" * 70)
    print(
        "Strict generalization validation completed successfully"
    )
    print("=" * 70)

    # ------------------------------------------------------------
    # Stop Spark
    # ------------------------------------------------------------

    spark.stop()


if __name__ == "__main__":
    main()