from pathlib import Path
import time
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)


GOLD_PATH = "data/gold/cse-cic-ids2018"
RESULTS_PATH = "results/spark_random_forest_results.csv"

DATASET_SIZES = {
    "10%": 0.10,
    "25%": 0.25,
    "50%": 0.50,
    "75%": 0.75,
    "100%": 1.00
}

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


def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-RandomForest")
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

    assembled_df = assembler.transform(ml_df)

    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    scaler_model = scaler.fit(assembled_df)

    prepared_df = (
        scaler_model
        .transform(assembled_df)
        .select(
            "features",
            col("is_attack").cast("double").alias("label")
        )
    )

    return prepared_df


def evaluate_model(predictions):

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

    return {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1": f1_evaluator.evaluate(predictions),
        "roc_auc": auc_evaluator.evaluate(predictions)
    }


def run_benchmark(df, dataset_name, fraction):

    print("\n" + "=" * 60)
    print(f"Random Forest Benchmark: {dataset_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # DATASET SIZE
    # --------------------------------------------------------

    if fraction == 1.0:
        benchmark_df = df
    else:
        benchmark_df = df.sample(
            withReplacement=False,
            fraction=fraction,
            seed=SEED
        )

    benchmark_df = benchmark_df.cache()

    row_count = benchmark_df.count()

    print(f"Selected rows: {row_count:,}")

    # --------------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------------

    print("\nPreparing features...")

    preparation_start = time.perf_counter()

    prepared_df = prepare_data(benchmark_df)
    prepared_df = prepared_df.cache()

    prepared_count = prepared_df.count()

    preparation_time = time.perf_counter() - preparation_start

    print(f"Prepared rows: {prepared_count:,}")
    print(f"Preparation time: {preparation_time:.4f} seconds")

    # --------------------------------------------------------
    # TRAIN / TEST
    # --------------------------------------------------------

    print("\nSplitting dataset...")

    train_df, test_df = prepared_df.randomSplit(
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

    train_time = time.perf_counter() - train_start

    print(f"Training time: {train_time:.4f} seconds")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print("\nRunning predictions...")

    prediction_start = time.perf_counter()

    predictions = model.transform(test_df)

    prediction_count = predictions.count()

    prediction_time = time.perf_counter() - prediction_start

    print(f"Predictions: {prediction_count:,}")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating...")

    metrics = evaluate_model(predictions)

    total_ml_time = train_time + prediction_time

    result = {
        "algorithm": "Random Forest",
        "dataset_size": dataset_name,
        "fraction": fraction,
        "rows": row_count,
        "prepared_rows": prepared_count,
        "train_rows": train_count,
        "test_rows": test_count,
        "preparation_time_sec": preparation_time,
        "training_time_sec": train_time,
        "prediction_time_sec": prediction_time,
        "total_ml_time_sec": total_ml_time,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"]
    }

    print("\nResult:")
    print("-" * 40)
    print(f"Rows:          {row_count:,}")
    print(f"Train time:    {train_time:.4f} s")
    print(f"Prediction:    {prediction_time:.4f} s")
    print(f"Total ML:      {total_ml_time:.4f} s")
    print(f"Accuracy:      {metrics['accuracy']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print(f"F1:            {metrics['f1']:.4f}")
    print(f"ROC AUC:       {metrics['roc_auc']:.4f}")

    prepared_df.unpersist()
    benchmark_df.unpersist()

    return result


def save_results(results):

    results_path = Path(RESULTS_PATH)

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "algorithm",
        "dataset_size",
        "fraction",
        "rows",
        "prepared_rows",
        "train_rows",
        "test_rows",
        "preparation_time_sec",
        "training_time_sec",
        "prediction_time_sec",
        "total_ml_time_sec",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {results_path}")


def main():

    print("=" * 60)
    print("CyberScaleData - Random Forest Scalability Benchmark")
    print("=" * 60)

    spark = create_spark_session()

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

    load_start = time.perf_counter()

    df = spark.read.parquet(str(file))

    total_rows = df.count()

    load_time = time.perf_counter() - load_start

    print(f"Total rows: {total_rows:,}")
    print(f"Load time: {load_time:.4f} seconds")

    results = []

    benchmark_start = time.perf_counter()

    for dataset_name, fraction in DATASET_SIZES.items():

        result = run_benchmark(
            df,
            dataset_name,
            fraction
        )

        results.append(result)

    total_benchmark_time = (
        time.perf_counter() - benchmark_start
    )

    save_results(results)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("RANDOM FOREST SCALABILITY BENCHMARK SUMMARY")
    print("=" * 90)

    print(
        f"{'Size':<10}"
        f"{'Rows':>12}"
        f"{'Train(s)':>12}"
        f"{'Predict(s)':>12}"
        f"{'Total(s)':>12}"
        f"{'Accuracy':>12}"
        f"{'F1':>10}"
        f"{'AUC':>10}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['dataset_size']:<10}"
            f"{result['rows']:>12,}"
            f"{result['training_time_sec']:>12.2f}"
            f"{result['prediction_time_sec']:>12.2f}"
            f"{result['total_ml_time_sec']:>12.2f}"
            f"{result['accuracy']:>12.4f}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>10.4f}"
        )

    print("-" * 90)
    print(
        f"Total benchmark time: "
        f"{total_benchmark_time:.2f} seconds"
    )

    print("=" * 90)
    print("Random Forest benchmark completed successfully")
    print("=" * 90)

    spark.stop()


if __name__ == "__main__":
    main()