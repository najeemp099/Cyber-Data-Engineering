from pathlib import Path
import time

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

SIZES = {
    "10%": 0.10,
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.00
}

SEED = 42


def create_spark():

    return (
        SparkSession.builder
        .appName("CyberScaleData-Robust-RandomForest")
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


def prepare_features(df):

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


def evaluate(predictions):

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

    return accuracy, precision, recall, f1, roc_auc


def main():

    print("=" * 70)
    print("CyberScaleData - Robust Random Forest Scalability Benchmark")
    print("=" * 70)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # --------------------------------------------------------
    # LOAD
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

    start = time.perf_counter()

    df = spark.read.parquet(str(file)).cache()

    original_rows = df.count()

    load_time = time.perf_counter() - start

    print(f"Original rows: {original_rows:,}")
    print(f"Load time: {load_time:.4f} seconds")

    # --------------------------------------------------------
    # DEDUPLICATION
    # --------------------------------------------------------

    print("\nRemoving exact duplicate records...")

    start = time.perf_counter()

    dedup_df = (
        df.select(
            *FEATURE_COLUMNS,
            "is_attack"
        )
        .dropDuplicates()
        .cache()
    )

    dedup_rows = dedup_df.count()

    dedup_time = time.perf_counter() - start

    duplicate_rows = original_rows - dedup_rows

    print(f"Original rows:     {original_rows:,}")
    print(f"Unique rows:       {dedup_rows:,}")
    print(f"Removed rows:      {duplicate_rows:,}")
    print(
        f"Duplicate rate:    "
        f"{duplicate_rows / original_rows * 100:.4f}%"
    )
    print(f"Deduplication time: {dedup_time:.4f} seconds")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nClass distribution after deduplication:")

    (
        dedup_df
        .groupBy("is_attack")
        .count()
        .orderBy("is_attack")
        .show()
    )

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    print("\nPreparing features...")

    start = time.perf_counter()

    prepared = prepare_features(dedup_df).cache()

    prepared_rows = prepared.count()

    preparation_time = time.perf_counter() - start

    print(f"Prepared rows: {prepared_rows:,}")
    print(f"Preparation time: {preparation_time:.4f} seconds")

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    results = []

    total_benchmark_start = time.perf_counter()

    for size_name, fraction in SIZES.items():

        print("\n" + "=" * 70)
        print(f"ROBUST RANDOM FOREST BENCHMARK: {size_name}")
        print("=" * 70)

        # Deterministic sample
        selection_start = time.perf_counter()

        if fraction == 1.0:
            selected = prepared
        else:
            selected = prepared.sample(
                withReplacement=False,
                fraction=fraction,
                seed=SEED
            )

        selected = selected.cache()

        rows = selected.count()

        selection_time = (
            time.perf_counter() - selection_start
        )

        print(f"Selected rows: {rows:,}")
        print(
            f"Selection time: "
            f"{selection_time:.4f} seconds"
        )

        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        print("\nSplitting dataset...")

        train_df, test_df = selected.randomSplit(
            [0.8, 0.2],
            seed=SEED
        )

        train_df = train_df.cache()
        test_df = test_df.cache()

        train_rows = train_df.count()
        test_rows = test_df.count()

        print(f"Training rows: {train_rows:,}")
        print(f"Testing rows:  {test_rows:,}")

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print("\nTraining Random Forest...")

        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            numTrees=20,
            maxDepth=10,
            seed=SEED
        )

        train_start = time.perf_counter()

        model = rf.fit(train_df)

        training_time = (
            time.perf_counter() - train_start
        )

        print(
            f"Training time: "
            f"{training_time:.4f} seconds"
        )

        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        print("\nRunning predictions...")

        prediction_start = time.perf_counter()

        predictions = model.transform(test_df).cache()

        prediction_rows = predictions.count()

        prediction_time = (
            time.perf_counter() - prediction_start
        )

        print(f"Predictions: {prediction_rows:,}")
        print(
            f"Prediction time: "
            f"{prediction_time:.4f} seconds"
        )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        print("\nEvaluating...")

        accuracy, precision, recall, f1, roc_auc = evaluate(
            predictions
        )

        total_ml_time = training_time + prediction_time

        print("\nResult:")
        print("-" * 40)
        print(f"Rows:          {rows:,}")
        print(f"Train time:    {training_time:.4f} s")
        print(f"Prediction:    {prediction_time:.4f} s")
        print(f"Total ML:      {total_ml_time:.4f} s")
        print(f"Accuracy:      {accuracy:.4f}")
        print(f"Precision:     {precision:.4f}")
        print(f"Recall:        {recall:.4f}")
        print(f"F1:            {f1:.4f}")
        print(f"ROC AUC:       {roc_auc:.4f}")

        results.append({
            "dataset_size": size_name,
            "rows": rows,
            "training_rows": train_rows,
            "testing_rows": test_rows,
            "selection_time_sec": selection_time,
            "preparation_time_sec": preparation_time,
            "training_time_sec": training_time,
            "prediction_time_sec": prediction_time,
            "total_ml_time_sec": total_ml_time,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })

        predictions.unpersist()
        train_df.unpersist()
        test_df.unpersist()
        selected.unpersist()

    total_benchmark_time = (
        time.perf_counter() - total_benchmark_start
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_dir = Path("results")
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "spark_robust_rf_results.csv"
    )

    import csv

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ROBUST RANDOM FOREST SCALABILITY SUMMARY")
    print("=" * 90)

    print(
        f"{'Size':<10}"
        f"{'Rows':>10}"
        f"{'Train(s)':>12}"
        f"{'Predict(s)':>13}"
        f"{'Total(s)':>12}"
        f"{'Accuracy':>12}"
        f"{'F1':>10}"
        f"{'AUC':>10}"
    )

    print("-" * 90)

    for r in results:

        print(
            f"{r['dataset_size']:<10}"
            f"{r['rows']:>10,}"
            f"{r['training_time_sec']:>12.2f}"
            f"{r['prediction_time_sec']:>13.2f}"
            f"{r['total_ml_time_sec']:>12.2f}"
            f"{r['accuracy']:>12.4f}"
            f"{r['f1']:>10.4f}"
            f"{r['roc_auc']:>10.4f}"
        )

    print("-" * 90)

    print(
        f"Original rows:       {original_rows:,}"
    )

    print(
        f"Unique rows:         {dedup_rows:,}"
    )

    print(
        f"Removed duplicates:  {duplicate_rows:,}"
    )

    print(
        f"Total benchmark time: "
        f"{total_benchmark_time:.2f} seconds"
    )

    print(
        f"\nResults saved to: {output_path}"
    )

    print("\n" + "=" * 70)
    print("Robust Random Forest benchmark completed successfully")
    print("=" * 70)

    prepared.unpersist()
    dedup_df.unpersist()
    df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()