import os
import time
import csv
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ============================================================
# CONFIGURATION
# ============================================================

def find_project_root():
    """
    Find the project root by looking for the data directory.
    This avoids problems caused by different src folder depths.
    """

    current = Path(__file__).resolve()

    # Check current file directory and all parent directories
    for path in [current.parent] + list(current.parents):

        if (path / "data" / "gold").exists():
            return path

    # Fallback to current working directory
    cwd = Path.cwd()

    if (cwd / "data" / "gold").exists():
        return cwd

    raise FileNotFoundError(
        "\nCould not find project root containing data/gold.\n"
        f"Script location: {current}\n"
        f"Current directory: {cwd}\n"
    )


PROJECT_ROOT = find_project_root()

GOLD_DIR = PROJECT_ROOT / "data" / "gold"

RESULTS_DIR = PROJECT_ROOT / "results" / "benchmark" / "scalability"
RESULTS_FILE = RESULTS_DIR / "scalability_results.csv"


# Dataset percentages to benchmark
DATASET_SIZES = [
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    1.00,
]


# Random Forest configuration
NUM_TREES = 50
MAX_DEPTH = 10
SEED = 42


# Features used by the cybersecurity detection model
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
    "Packet_Length_Std",
]

LABEL_COLUMN = "is_attack"


# ============================================================
# SPARK SESSION
# ============================================================

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("CyberScaleData-Scalability-Benchmark")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.sql.execution.pyspark.udf.faulthandler.enabled",
            "true"
        )
        .config(
            "spark.python.worker.faulthandler.enabled",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# FIND GOLD DATASET
# ============================================================

def find_gold_dataset():
    """
    Locate the Gold Parquet dataset inside data/gold.
    """

    print(f"\nSearching Gold directory:")
    print(GOLD_DIR)

    if not GOLD_DIR.exists():

        raise FileNotFoundError(
            f"\nGold directory does not exist:\n{GOLD_DIR}\n\n"
            "Please check that your Gold dataset is inside "
            "data/gold."
        )

    parquet_files = list(GOLD_DIR.rglob("*.parquet"))

    if not parquet_files:

        raise FileNotFoundError(
            f"\nNo Parquet files found inside:\n{GOLD_DIR}\n\n"
            "Please check that your Gold dataset exists."
        )

    # Prefer CICFlowMeter dataset
    cic_files = [
        f for f in parquet_files
        if "CICFlowMeter" in f.name
    ]

    if cic_files:
        return cic_files[0]

    return parquet_files[0]


# ============================================================
# DATA CLEANING
# ============================================================

def prepare_data(df):

    print("\nSelecting required features...")

    available_columns = df.columns

    missing = [
        column
        for column in FEATURE_COLUMNS + [LABEL_COLUMN]
        if column not in available_columns
    ]

    if missing:

        raise ValueError(
            "\nRequired columns are missing from the Gold dataset:\n"
            + "\n".join(missing)
        )

    df = df.select(
        FEATURE_COLUMNS + [LABEL_COLUMN]
    )

    print("Cleaning numeric values...")

    for feature in FEATURE_COLUMNS:

        df = df.withColumn(
            feature,
            col(feature).cast("double")
        )

    df = df.withColumn(
        LABEL_COLUMN,
        col(LABEL_COLUMN).cast("double")
    )

    # Replace null, NaN and infinity values
    for feature in FEATURE_COLUMNS:

        df = df.withColumn(
            feature,
            when(
                col(feature).isNull()
                | (col(feature) == float("inf"))
                | (col(feature) == float("-inf")),
                0.0
            ).otherwise(col(feature))
        )

    # Remove missing labels
    df = df.filter(
        col(LABEL_COLUMN).isNotNull()
    )

    # Remove duplicates
    print("\nRemoving exact duplicates...")

    before = df.count()

    df = df.dropDuplicates()

    after = df.count()

    print(f"Rows before duplicate removal: {before:,}")
    print(f"Rows after duplicate removal:  {after:,}")
    print(f"Duplicates removed:            {before - after:,}")

    return df


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

def train_model(train_df):

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip"
    )

    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol=LABEL_COLUMN,
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        numTrees=NUM_TREES,
        maxDepth=MAX_DEPTH,
        seed=SEED
    )

    pipeline = Pipeline(
        stages=[
            assembler,
            classifier
        ]
    )

    start_time = time.perf_counter()

    model = pipeline.fit(train_df)

    training_time = time.perf_counter() - start_time

    return model, training_time


# ============================================================
# PREDICTION + EVALUATION
# ============================================================

def evaluate_model(model, test_df):

    start_time = time.perf_counter()

    predictions = model.transform(test_df)

    # Force Spark evaluation
    predictions.select(
        "prediction",
        "probability",
        LABEL_COLUMN
    ).count()

    prediction_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    total = predictions.count()

    correct = predictions.filter(
        col("prediction") == col(LABEL_COLUMN)
    ).count()

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    tp = predictions.filter(
        (col(LABEL_COLUMN) == 1.0)
        & (col("prediction") == 1.0)
    ).count()

    tn = predictions.filter(
        (col(LABEL_COLUMN) == 0.0)
        & (col("prediction") == 0.0)
    ).count()

    fp = predictions.filter(
        (col(LABEL_COLUMN) == 0.0)
        & (col("prediction") == 1.0)
    ).count()

    fn = predictions.filter(
        (col(LABEL_COLUMN) == 1.0)
        & (col("prediction") == 0.0)
    ).count()

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COLUMN,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    roc_auc = evaluator.evaluate(predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "prediction_time": prediction_time,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset_fraction",
                "dataset_percent",
                "rows",
                "train_rows",
                "test_rows",
                "num_features",
                "num_trees",
                "max_depth",
                "training_time",
                "prediction_time",
                "total_time",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "tp",
                "tn",
                "fp",
                "fn",
            ]
        )

        writer.writeheader()

        writer.writerows(results)

    print("\nResults saved to:")
    print(RESULTS_FILE)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n")
    print("=" * 100)
    print("SCALABILITY BENCHMARK RESULTS")
    print("=" * 100)

    print(
        f"{'Dataset':<12}"
        f"{'Rows':>12}"
        f"{'Train':>12}"
        f"{'Test':>12}"
        f"{'Train(s)':>12}"
        f"{'Predict(s)':>12}"
        f"{'F1':>10}"
        f"{'ROC AUC':>10}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['dataset_percent']:>6.0f}%"
            f"{result['rows']:>12,}"
            f"{result['train_rows']:>12,}"
            f"{result['test_rows']:>12,}"
            f"{result['training_time']:>12.2f}"
            f"{result['prediction_time']:>12.2f}"
            f"{result['f1']:>10.4f}"
            f"{result['roc_auc']:>10.4f}"
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CyberScaleData - Random Forest Scalability Benchmark")
    print("=" * 70)

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nGold directory:")
    print(GOLD_DIR)

    spark = create_spark_session()

    # IMPORTANT:
    # Initialize df before try so finally can safely use it.
    df = None

    try:

        # ----------------------------------------------------
        # Load Gold dataset
        # ----------------------------------------------------

        print("\nLoading Gold dataset...")

        gold_path = find_gold_dataset()

        print("\nGold dataset found:")
        print(gold_path)

        df = spark.read.parquet(
            str(gold_path)
        )

        original_rows = df.count()

        print(
            f"Original rows: {original_rows:,}"
        )

        # ----------------------------------------------------
        # Prepare data
        # ----------------------------------------------------

        df = prepare_data(df)

        total_rows = df.count()

        print(
            f"\nUsable rows: {total_rows:,}"
        )

        print("\nClass distribution:")

        df.groupBy(
            LABEL_COLUMN
        ).count().orderBy(
            LABEL_COLUMN
        ).show()

        # Cache dataset
        df = df.cache()

        # Force cache
        df.count()

        # ----------------------------------------------------
        # Scalability experiments
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("SCALABILITY EXPERIMENTS")
        print("=" * 70)

        results = []

        for fraction in DATASET_SIZES:

            print("\n")
            print("-" * 70)

            percent = fraction * 100

            print(
                f"Dataset size: {percent:.0f}%"
            )

            # ------------------------------------------------
            # Create deterministic dataset subset
            # ------------------------------------------------

            if fraction >= 1.0:

                experiment_df = df

            else:

                experiment_df = df.sample(
                    withReplacement=False,
                    fraction=fraction,
                    seed=SEED
                )

            # Cache subset
            experiment_df = experiment_df.cache()

            rows = experiment_df.count()

            print(
                f"Rows selected: {rows:,}"
            )

            if rows < 100:

                print(
                    "Skipping because dataset is too small."
                )

                if experiment_df is not df:
                    experiment_df.unpersist()

                continue

            # ------------------------------------------------
            # Train/test split
            # ------------------------------------------------

            print(
                "Creating train/test split..."
            )

            train_df, test_df = experiment_df.randomSplit(
                [0.8, 0.2],
                seed=SEED
            )

            train_df = train_df.cache()
            test_df = test_df.cache()

            train_rows = train_df.count()
            test_rows = test_df.count()

            print(
                f"Training rows: {train_rows:,}"
            )

            print(
                f"Testing rows:  {test_rows:,}"
            )

            # ------------------------------------------------
            # Train model
            # ------------------------------------------------

            print(
                "\nTraining Random Forest..."
            )

            model, training_time = train_model(
                train_df
            )

            print(
                f"Training time: {training_time:.4f}s"
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            print(
                "Running predictions..."
            )

            metrics = evaluate_model(
                model,
                test_df
            )

            total_time = (
                training_time
                + metrics["prediction_time"]
            )

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print("\nPerformance:")

            print(
                f"Accuracy:        {metrics['accuracy']:.4f}"
            )

            print(
                f"Precision:       {metrics['precision']:.4f}"
            )

            print(
                f"Recall:          {metrics['recall']:.4f}"
            )

            print(
                f"F1 Score:        {metrics['f1']:.4f}"
            )

            print(
                f"ROC AUC:         {metrics['roc_auc']:.4f}"
            )

            print(
                f"Training Time:   {training_time:.4f}s"
            )

            print(
                f"Prediction Time: {metrics['prediction_time']:.4f}s"
            )

            print(
                f"Total Time:      {total_time:.4f}s"
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append({
                "dataset_fraction": fraction,
                "dataset_percent": percent,
                "rows": rows,
                "train_rows": train_rows,
                "test_rows": test_rows,
                "num_features": len(FEATURE_COLUMNS),
                "num_trees": NUM_TREES,
                "max_depth": MAX_DEPTH,
                "training_time": training_time,
                "prediction_time": metrics["prediction_time"],
                "total_time": total_time,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            })

            # ------------------------------------------------
            # Release memory
            # ------------------------------------------------

            train_df.unpersist()
            test_df.unpersist()

            if experiment_df is not df:
                experiment_df.unpersist()

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        save_results(results)

        # ----------------------------------------------------
        # Print final comparison
        # ----------------------------------------------------

        print_results(results)

        print("\n")
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        print(
            f"Original rows:        {original_rows:,}"
        )

        print(
            f"Usable rows:          {total_rows:,}"
        )

        print(
            f"Dataset sizes tested: {len(results)}"
        )

        print(
            f"Features:              {len(FEATURE_COLUMNS)}"
        )

        print(
            f"Random Forest trees:   {NUM_TREES}"
        )

        print(
            f"Maximum depth:         {MAX_DEPTH}"
        )

        if results:

            fastest = min(
                results,
                key=lambda x: x["training_time"]
            )

            largest = max(
                results,
                key=lambda x: x["rows"]
            )

            print(
                f"\nFastest training: "
                f"{fastest['training_time']:.2f}s "
                f"({fastest['dataset_percent']:.0f}% dataset)"
            )

            print(
                f"Largest benchmark: "
                f"{largest['rows']:,} rows"
            )

        print("=" * 70)
        print(
            "Scalability benchmark completed successfully"
        )
        print("=" * 70)

    finally:

        # ----------------------------------------------------
        # Safe cleanup
        # ----------------------------------------------------

        if df is not None:
            df.unpersist()

        spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
