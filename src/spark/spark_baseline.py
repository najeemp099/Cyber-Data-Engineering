from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator


GOLD_PATH = "data/gold/cse-cic-ids2018"


def main():

    print("=" * 60)
    print("CyberScaleData - PySpark Logistic Regression Baseline")
    print("=" * 60)

    spark = (
        SparkSession.builder
        .appName("CyberScaleData")
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

    print("\nLoading Gold dataset...")

    files = list(Path(GOLD_PATH).glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No parquet files found in {GOLD_PATH}"
        )

    file = files[0]

    print(f"File: {file.name}")

    start_time = time.perf_counter()

    df = spark.read.parquet(str(file))

    row_count = df.count()

    load_time = time.perf_counter() - start_time

    print(f"Rows: {row_count:,}")
    print(f"Load time: {load_time:.4f} seconds")

    print("\nSchema:")
    df.printSchema()

    print("\nSample data:")
    df.show(5, truncate=False)

    print("\nData summary:")

    df.select(
        "Protocol",
        "Flow_Duration",
        "Total_Fwd_Packets",
        "Total_Backward_Packets",
        "is_attack"
    ).describe().show()

    print("\nAttack distribution:")

    df.groupBy("is_attack") \
      .count() \
      .orderBy("is_attack") \
      .show()

    # ------------------------------------------------------------
    # LOGISTIC REGRESSION BASELINE
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Logistic Regression Baseline")
    print("=" * 60)

    feature_columns = [
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

    # Select only the required ML columns
    ml_df = df.select(
        *feature_columns,
        "is_attack"
    )

    # Replace null and NaN values with 0
    for feature in feature_columns:
        ml_df = ml_df.withColumn(
            feature,
            when(
                col(feature).isNull() | isnan(col(feature)),
                0.0
            ).otherwise(col(feature).cast("double"))
        )

    print("\nPreparing features...")

    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="raw_features",
        handleInvalid="skip"
    )

    assembled_df = assembler.transform(ml_df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    print("Scaling features...")

    scaler_model = scaler.fit(assembled_df)

    prepared_df = scaler_model.transform(assembled_df) \
        .select("features", col("is_attack").cast("double").alias("label"))

    # Cache because the dataset will be used multiple times
    prepared_df = prepared_df.cache()

    prepared_count = prepared_df.count()

    print(f"Prepared rows: {prepared_count:,}")

    # ------------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------------------

    print("\nSplitting dataset...")

    train_df, test_df = prepared_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print(f"Training rows: {train_count:,}")
    print(f"Testing rows:  {test_count:,}")

    # ------------------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
    # ------------------------------------------------------------

    print("\nTraining Logistic Regression...")

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=20
    )

    train_start = time.perf_counter()

    lr_model = lr.fit(train_df)

    train_time = time.perf_counter() - train_start

    print(f"Training time: {train_time:.4f} seconds")

    # ------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------

    print("\nRunning predictions...")

    prediction_start = time.perf_counter()

    predictions = lr_model.transform(test_df)

    prediction_count = predictions.count()

    prediction_time = time.perf_counter() - prediction_start

    print(f"Predictions: {prediction_count:,}")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    # ------------------------------------------------------------
    # EVALUATION
    # ------------------------------------------------------------

    print("\nEvaluating model...")

    accuracy_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    precision_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    recall_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    auc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    accuracy = accuracy_evaluator.evaluate(predictions)
    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1 = f1_evaluator.evaluate(predictions)
    auc = auc_evaluator.evaluate(predictions)

    print("\nModel Performance:")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    print("\nConfusion Matrix:")

    predictions.groupBy(
        "label",
        "prediction"
    ).count() \
     .orderBy("label", "prediction") \
     .show()

    # ------------------------------------------------------------
    # BENCHMARK SUMMARY
    # ------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Benchmark Result")
    print("=" * 60)

    print(f"Dataset rows:       {row_count:,}")
    print(f"Prepared rows:      {prepared_count:,}")
    print(f"Training rows:      {train_count:,}")
    print(f"Testing rows:       {test_count:,}")
    print(f"Load time:          {load_time:.4f} seconds")
    print(f"Training time:      {train_time:.4f} seconds")
    print(f"Prediction time:    {prediction_time:.4f} seconds")
    print(f"Total ML time:      {train_time + prediction_time:.4f} seconds")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall:             {recall:.4f}")
    print(f"F1 Score:           {f1:.4f}")
    print(f"ROC AUC:            {auc:.4f}")

    prepared_df.unpersist()
    spark.stop()

    print("=" * 60)
    print("PySpark Logistic Regression benchmark completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()