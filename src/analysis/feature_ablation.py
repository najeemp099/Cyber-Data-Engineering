from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)


# ============================================================
# CONFIGURATION
# ============================================================

GOLD_PATH = "data/gold/cse-cic-ids2018"

OUTPUT_DIR = Path("results/feature_ablation")
OUTPUT_FILE = OUTPUT_DIR / "feature_ablation_results.csv"

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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CyberScaleData - Feature Ablation Analysis")
    print("=" * 70)

    start = time.perf_counter()

    # --------------------------------------------------------
    # SPARK
    # --------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-Feature-Ablation")
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
    # LOAD GOLD DATA
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

    print(f"Unique rows:   {unique_rows:,}")
    print(
        f"Removed rows:  {original_rows - unique_rows:,}"
    )

    # --------------------------------------------------------
    # SELECT REQUIRED COLUMNS
    # --------------------------------------------------------

    print("\nSelecting features...")

    df = df.select(
        *FEATURES,
        col(LABEL).cast("double").alias(LABEL)
    )

    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    print("\nCleaning numeric values...")

    for feature in FEATURES:
        df = df.filter(
            col(feature).isNotNull()
        )

    df = df.filter(
        col(LABEL).isNotNull()
    )

    # --------------------------------------------------------
    # DETERMINISTIC ORDERED SPLIT
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Do NOT use RDD.zipWithIndex().
    #
    # The previous implementation converted the Spark DataFrame
    # to an RDD and then back to a DataFrame. On this Windows +
    # PySpark environment, that caused:
    #
    # Python worker exited unexpectedly
    #
    # Instead, monotonically_increasing_id() keeps the operation
    # inside Spark's DataFrame engine.
    # --------------------------------------------------------

    print("\nCreating deterministic ordered train/test split...")

    indexed = df.withColumn(
        "_row_id",
        monotonically_increasing_id()
    )

    indexed = indexed.cache()

    total_rows = indexed.count()

    split_id = int(total_rows * 0.8)

    train = (
        indexed
        .filter(col("_row_id") < split_id)
        .drop("_row_id")
        .cache()
    )

    test = (
        indexed
        .filter(col("_row_id") >= split_id)
        .drop("_row_id")
        .cache()
    )

    train_rows = train.count()
    test_rows = test.count()

    print(f"Training rows: {train_rows:,}")
    print(f"Testing rows:  {test_rows:,}")

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nTraining class distribution:")

    train.groupBy(LABEL).count().orderBy(LABEL).show()

    print("Testing class distribution:")

    test.groupBy(LABEL).count().orderBy(LABEL).show()

    # --------------------------------------------------------
    # EVALUATORS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ABLATION SETS
    # --------------------------------------------------------

    experiments = [
        ("Full Model", FEATURES)
    ]

    for feature in FEATURES:

        remaining_features = [
            f for f in FEATURES
            if f != feature
        ]

        experiments.append(
            (
                f"Without {feature}",
                remaining_features
            )
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FEATURE ABLATION EXPERIMENTS")
    print("=" * 70)

    for experiment_name, selected_features in experiments:

        print("\n" + "-" * 70)
        print(experiment_name)
        print("-" * 70)

        print(
            f"Features used: {len(selected_features)}"
        )

        print(
            ", ".join(selected_features)
        )

        # ----------------------------------------------------
        # FEATURE VECTOR
        # ----------------------------------------------------

        assembler = VectorAssembler(
            inputCols=selected_features,
            outputCol="features",
            handleInvalid="skip"
        )

        train_vector = assembler.transform(train).select(
            "features",
            col(LABEL).alias("label")
        )

        test_vector = assembler.transform(test).select(
            "features",
            col(LABEL).alias("label")
        )

        # ----------------------------------------------------
        # RANDOM FOREST
        # ----------------------------------------------------

        rf = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=50,
            maxDepth=10,
            seed=42
        )

        print("\nTraining Random Forest...")

        train_start = time.perf_counter()

        model = rf.fit(train_vector)

        training_time = (
            time.perf_counter() - train_start
        )

        print(
            f"Training time: {training_time:.4f}s"
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        print("Running predictions...")

        prediction_start = time.perf_counter()

        predictions = model.transform(test_vector)

        predictions = predictions.cache()

        prediction_count = predictions.count()

        prediction_time = (
            time.perf_counter() - prediction_start
        )

        print(
            f"Prediction time: {prediction_time:.4f}s"
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_evaluator.evaluate(
            predictions
        )

        f1 = f1_evaluator.evaluate(
            predictions
        )

        roc_auc = roc_auc_evaluator.evaluate(
            predictions
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX COUNTS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nPerformance:")

        print(
            f"Accuracy:       {accuracy:.4f}"
        )

        print(
            f"Precision:      {precision:.4f}"
        )

        print(
            f"Recall:         {recall:.4f}"
        )

        print(
            f"F1 Score:       {f1:.4f}"
        )

        print(
            f"ROC AUC:        {roc_auc:.4f}"
        )

        print(
            f"Training Time:  {training_time:.4f}s"
        )

        print(
            f"Prediction Time:{prediction_time:.4f}s"
        )

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "experiment": experiment_name,
                "removed_feature": (
                    ""
                    if experiment_name == "Full Model"
                    else experiment_name.replace(
                        "Without ",
                        ""
                    )
                ),
                "features_used": len(
                    selected_features
                ),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "training_time_seconds": training_time,
                "prediction_time_seconds": prediction_time,
                "test_rows": prediction_count,
            }
        )

        predictions.unpersist()

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ABLATION RESULTS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    import csv

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment",
                "removed_feature",
                "features_used",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "training_time_seconds",
                "prediction_time_seconds",
                "test_rows",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(
        f"\nResults saved to: {OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # DISPLAY COMPARISON
    # --------------------------------------------------------

    print("\nPerformance comparison:")
    print("-" * 70)

    print(
        f"{'Experiment':<40}"
        f"{'F1':>10}"
        f"{'ROC AUC':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['experiment']:<40}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>12.4f}"
        )

    # --------------------------------------------------------
    # FIND MOST IMPORTANT FEATURE
    # --------------------------------------------------------

    baseline = results[0]

    ablation_results = results[1:]

    feature_impacts = []

    for result in ablation_results:

        f1_drop = (
            baseline["f1"] -
            result["f1"]
        )

        roc_drop = (
            baseline["roc_auc"] -
            result["roc_auc"]
        )

        feature_impacts.append(
            (
                result["removed_feature"],
                f1_drop,
                roc_drop
            )
        )

    feature_impacts.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "=" * 70)
    print("FEATURE IMPACT")
    print("=" * 70)

    print(
        f"{'Removed Feature':<35}"
        f"{'F1 Drop':>12}"
        f"{'ROC AUC Drop':>16}"
    )

    print("-" * 70)

    for feature, f1_drop, roc_drop in feature_impacts:

        print(
            f"{feature:<35}"
            f"{f1_drop:>12.6f}"
            f"{roc_drop:>16.6f}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_time = (
        time.perf_counter() - start
    )

    top_feature = (
        feature_impacts[0][0]
        if feature_impacts
        else "N/A"
    )

    print("\n" + "=" * 70)
    print("FEATURE ABLATION ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Original rows:       {original_rows:,}"
    )

    print(
        f"Unique rows:         {unique_rows:,}"
    )

    print(
        f"Training rows:       {train_rows:,}"
    )

    print(
        f"Testing rows:        {test_rows:,}"
    )

    print(
        f"Features tested:     {len(FEATURES)}"
    )

    print(
        f"Experiments:         {len(experiments)}"
    )

    print(
        f"Baseline F1:         {baseline['f1']:.4f}"
    )

    print(
        f"Baseline ROC AUC:    {baseline['roc_auc']:.4f}"
    )

    print(
        f"Most impactful:      {top_feature}"
    )

    print(
        f"Total execution:     {total_time:.2f}s"
    )

    print("=" * 70)
    print(
        "Feature ablation analysis completed successfully"
    )
    print("=" * 70)

    train.unpersist()
    test.unpersist()
    indexed.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()