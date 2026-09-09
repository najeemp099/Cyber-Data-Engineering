from pathlib import Path
from time import perf_counter
import csv

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when


# ============================================================
# CyberScaleData - Cross-Dataset Generalization
# Class-Weighted Random Forest
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "cse-cic-ids2018"
    / "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "cse-cic-ids2018"
    / "Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_dataset"

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


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_dataset(spark, path, dataset_name):

    print("\n" + "=" * 70)
    print(f"Loading {dataset_name} dataset")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}"
        )

    print(f"File: {path.name}")

    df = spark.read.parquet(str(path))

    original_rows = df.count()

    print(f"Original rows: {original_rows:,}")

    # --------------------------------------------------------
    # Remove exact duplicates
    # --------------------------------------------------------

    print("\nRemoving exact duplicates...")

    df = df.dropDuplicates()

    unique_rows = df.count()

    duplicates_removed = original_rows - unique_rows

    print(f"Unique rows: {unique_rows:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\nValidating required columns...")
    print("All required columns are present.")

    # --------------------------------------------------------
    # Prepare numerical features
    # --------------------------------------------------------

    print("\nPreparing features...")

    prepared = df.select(
        *[
            col(c).cast("double").alias(c)
            for c in FEATURE_COLUMNS
        ],
        col(LABEL_COLUMN).cast("double").alias("label"),
    )

    # Remove null / invalid feature rows
    prepared = prepared.na.drop()

    prepared_rows = prepared.count()

    print(f"Prepared rows: {prepared_rows:,}")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    (
        prepared
        .groupBy("label")
        .count()
        .orderBy("label")
        .show()
    )

    return (
        prepared,
        original_rows,
        unique_rows,
        duplicates_removed,
        prepared_rows,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    spark = (
        SparkSession.builder
        .appName(
            "CyberScaleData-Cross-Dataset-ClassWeighted"
        )
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("=" * 70)
    print("CyberScaleData - Cross-Dataset Generalization")
    print("Class-Weighted Random Forest")
    print("=" * 70)

    # ========================================================
    # 1. LOAD TRAINING DATASET
    # ========================================================

    (
        train_df,
        train_original,
        train_unique,
        train_duplicates,
        train_prepared,
    ) = prepare_dataset(
        spark,
        TRAIN_PATH,
        "TRAINING",
    )

    # ========================================================
    # 2. LOAD COMPLETELY SEPARATE TEST DATASET
    # ========================================================

    (
        test_df,
        test_original,
        test_unique,
        test_duplicates,
        test_prepared,
    ) = prepare_dataset(
        spark,
        TEST_PATH,
        "TEST",
    )

    # ========================================================
    # 3. CROSS-DATASET SETUP
    # ========================================================

    print("\n" + "=" * 70)
    print("CROSS-DATASET SETUP")
    print("=" * 70)

    print(f"Training dataset: {TRAIN_PATH.name}")
    print(f"Testing dataset:  {TEST_PATH.name}")

    print(f"\nTraining rows: {train_prepared:,}")
    print(f"Testing rows:  {test_prepared:,}")

    print("\nNo rows from the test dataset are used during training.")
    print("No random train/test split is used.")
    print(
        "The model is trained on one attack scenario "
        "and tested on another."
    )

    # ========================================================
    # 4. CALCULATE CLASS WEIGHTS
    # ========================================================

    print("\n" + "=" * 70)
    print("CALCULATING CLASS WEIGHTS")
    print("=" * 70)

    class_counts = (
        train_df
        .groupBy("label")
        .count()
        .collect()
    )

    class_count_dict = {
        float(row["label"]): row["count"]
        for row in class_counts
    }

    normal_count = class_count_dict.get(0.0, 0)
    attack_count = class_count_dict.get(1.0, 0)

    if normal_count == 0 or attack_count == 0:
        raise ValueError(
            "Training dataset must contain both normal and attack samples."
        )

    total_training = normal_count + attack_count

    normal_weight = total_training / (2.0 * normal_count)
    attack_weight = total_training / (2.0 * attack_count)

    print(f"Normal samples: {normal_count:,}")
    print(f"Attack samples: {attack_count:,}")

    print(f"Normal class weight: {normal_weight:.6f}")
    print(f"Attack class weight: {attack_weight:.6f}")

    # --------------------------------------------------------
    # Add weight column
    # --------------------------------------------------------

    train_weighted = train_df.withColumn(
        "classWeight",
        when(
            col("label") == 1.0,
            attack_weight,
        ).otherwise(
            normal_weight
        ),
    )

    print("\nClass weights added to training dataset.")

    # ========================================================
    # 5. RANDOM FOREST PIPELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    assembler = VectorAssembler(
        inputCols=FEATURE_COLUMNS,
        outputCol="features",
        handleInvalid="skip",
    )

    rf = RandomForestClassifier(
        labelCol="label",
        featuresCol="features",
        weightCol="classWeight",
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

    # ========================================================
    # 6. TRAIN MODEL
    # ========================================================

    print("\nTraining class-weighted Random Forest...")

    train_start = perf_counter()

    model = pipeline.fit(train_weighted)

    training_time = perf_counter() - train_start

    print(
        f"Training time: {training_time:.4f}s"
    )

    # ========================================================
    # 7. PREDICTIONS
    # ========================================================

    print("\nRunning predictions on unseen dataset...")

    prediction_start = perf_counter()

    predictions = model.transform(test_df)

    prediction_count = predictions.count()

    prediction_time = perf_counter() - prediction_start

    print(f"Predictions: {prediction_count:,}")
    print(
        f"Prediction time: {prediction_time:.4f}s"
    )

    # ========================================================
    # 8. CONFUSION MATRIX
    # ========================================================

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    confusion_rows = (
        predictions
        .groupBy("label", "prediction")
        .count()
        .collect()
    )

    tn = 0
    fp = 0
    fn = 0
    tp = 0

    for row in confusion_rows:

        actual = float(row["label"])
        predicted = float(row["prediction"])
        count = row["count"]

        if actual == 0.0 and predicted == 0.0:
            tn = count

        elif actual == 0.0 and predicted == 1.0:
            fp = count

        elif actual == 1.0 and predicted == 0.0:
            fn = count

        elif actual == 1.0 and predicted == 1.0:
            tp = count

    print(
        "\n+----------------+------------+"
    )
    print(
        "|                | Predicted  |"
    )
    print(
        "+----------------+------------+"
    )
    print(
        f"| Actual Normal  | TN={tn:,} |"
    )
    print(
        f"| Actual Attack  | TP={tp:,} |"
    )
    print(
        "+----------------+------------+"
    )

    print(f"\nTrue Negative  (TN): {tn:,}")
    print(f"False Positive (FP): {fp:,}")
    print(f"False Negative (FN): {fn:,}")
    print(f"True Positive  (TP): {tp:,}")

    # ========================================================
    # 9. MANUAL ATTACK METRICS
    # ========================================================

    total = tn + fp + fn + tp

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    attack_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    attack_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if attack_precision + attack_recall > 0:

        attack_f1 = (
            2
            * attack_precision
            * attack_recall
            / (attack_precision + attack_recall)
        )

    else:
        attack_f1 = 0.0

    # ========================================================
    # 10. ROC AUC
    # ========================================================

    print("\n" + "=" * 70)
    print("CROSS-DATASET MODEL PERFORMANCE")
    print("=" * 70)

    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )

    roc_auc = roc_evaluator.evaluate(predictions)

    # ========================================================
    # 11. PR AUC
    # ========================================================

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    pr_auc = pr_evaluator.evaluate(predictions)

    print(
        f"\nAccuracy:           {accuracy:.4f}"
    )

    print(
        f"Attack Precision:   {attack_precision:.4f}"
    )

    print(
        f"Attack Recall:      {attack_recall:.4f}"
    )

    print(
        f"Attack F1 Score:    {attack_f1:.4f}"
    )

    print(
        f"ROC AUC:            {roc_auc:.4f}"
    )

    print(
        f"PR AUC:             {pr_auc:.4f}"
    )

    # ========================================================
    # 12. GENERALIZATION INTERPRETATION
    # ========================================================

    print("\n" + "-" * 70)
    print("GENERALIZATION INTERPRETATION")
    print("-" * 70)

    if tp == 0:

        print(
            "WARNING: The model detected none of the attack "
            "samples in the unseen dataset."
        )

        print(
            "This indicates poor attack-class generalization."
        )

    elif attack_recall < 0.50:

        print(
            "WARNING: Attack recall is below 50%."
        )

        print(
            "The model has limited generalization "
            "to the unseen attack scenario."
        )

    else:

        print(
            "The model detected attack samples in the "
            "unseen dataset."
        )

        print(
            "Cross-dataset attack generalization is present."
        )

    # ========================================================
    # 13. FEATURE IMPORTANCE
    # ========================================================

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

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

    # ========================================================
    # 14. SAVE RESULTS
    # ========================================================

    print("\nSaving results...")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_ml_time = (
        training_time
        + prediction_time
    )

    results_path = (
        RESULTS_DIR
        / "cross_dataset_generalization_results.csv"
    )

    with open(
        results_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "train_dataset",
            "test_dataset",

            "train_original_rows",
            "train_unique_rows",
            "train_duplicates_removed",
            "train_prepared_rows",

            "test_original_rows",
            "test_unique_rows",
            "test_duplicates_removed",
            "test_prepared_rows",

            "training_time_seconds",
            "prediction_time_seconds",
            "total_ml_time_seconds",

            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",

            "accuracy",
            "attack_precision",
            "attack_recall",
            "attack_f1_score",

            "roc_auc",
            "pr_auc",
        ])

        writer.writerow([
            TRAIN_PATH.name,
            TEST_PATH.name,

            train_original,
            train_unique,
            train_duplicates,
            train_prepared,

            test_original,
            test_unique,
            test_duplicates,
            test_prepared,

            training_time,
            prediction_time,
            total_ml_time,

            tn,
            fp,
            fn,
            tp,

            accuracy,
            attack_precision,
            attack_recall,
            attack_f1,

            roc_auc,
            pr_auc,
        ])

    print(f"Saved: {results_path}")

    # ========================================================
    # 15. SAVE FEATURE IMPORTANCE
    # ========================================================

    importance_path = (
        RESULTS_DIR
        / "cross_dataset_feature_importance.csv"
    )

    with open(
        importance_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "feature",
            "importance",
        ])

        for feature, importance in feature_importance:

            writer.writerow([
                feature,
                importance,
            ])

    print(f"Saved: {importance_path}")

    # ========================================================
    # 16. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("CROSS-DATASET GENERALIZATION SUMMARY")
    print("=" * 70)

    print(
        f"Training dataset: {TRAIN_PATH.name}"
    )

    print(
        f"Testing dataset:  {TEST_PATH.name}"
    )

    print(
        f"Training rows:    {train_prepared:,}"
    )

    print(
        f"Testing rows:     {prediction_count:,}"
    )

    print(
        f"Training time:    {training_time:.4f}s"
    )

    print(
        f"Prediction time:  {prediction_time:.4f}s"
    )

    print(
        f"Total ML time:    {total_ml_time:.4f}s"
    )

    print("\nConfusion Matrix:")

    print(
        f"TN: {tn:,} | "
        f"FP: {fp:,} | "
        f"FN: {fn:,} | "
        f"TP: {tp:,}"
    )

    print("\nAttack Metrics:")

    print(
        f"Precision: {attack_precision:.4f}"
    )

    print(
        f"Recall:    {attack_recall:.4f}"
    )

    print(
        f"F1 Score:  {attack_f1:.4f}"
    )

    print(
        f"ROC AUC:   {roc_auc:.4f}"
    )

    print(
        f"PR AUC:    {pr_auc:.4f}"
    )

    print("\nResults saved to:")

    print(results_path)
    print(importance_path)

    print("=" * 70)

    print(
        "Cross-dataset generalization validation "
        "completed successfully"
    )

    print("=" * 70)

    spark.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()