import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = (
    "data/gold/cse-cic-ids2018/"
    "Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet"
)

OUTPUT_DIR = "results/strict_generalization"

FEATURES = [
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

TEST_RATIO = 0.20


# ============================================================
# Spark
# ============================================================

spark = (
    SparkSession.builder
    .appName("CyberScaleData-Strict-Generalization")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("=" * 70)
print("CyberScaleData - Strict Generalization Validation")
print("=" * 70)


# ============================================================
# Load dataset
# ============================================================

print("\nLoading Gold dataset...")
print(f"File: {os.path.basename(INPUT_FILE)}")

df = spark.read.parquet(INPUT_FILE)

original_rows = df.count()

print(f"Original rows: {original_rows:,}")


# ============================================================
# Identify label column
# ============================================================

if "label" in df.columns:
    label_column = "label"
elif "is_attack" in df.columns:
    label_column = "is_attack"
else:
    raise ValueError(
        "Could not find target column. Expected 'label' or 'is_attack'."
    )

print(f"Target column: {label_column}")


# ============================================================
# Validate required features
# ============================================================

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )


# ============================================================
# Select required columns
# ============================================================

df = df.select(FEATURES + [label_column])


# ============================================================
# Remove exact duplicates
# ============================================================

print("\nRemoving exact duplicates...")

df = df.dropDuplicates()

unique_rows = df.count()

duplicates_removed = original_rows - unique_rows

print(f"Unique rows:         {unique_rows:,}")
print(f"Duplicates removed:  {duplicates_removed:,}")


# ============================================================
# Prepare features
# ============================================================

print("\nPreparing features...")

for feature in FEATURES:
    df = df.withColumn(
        feature,
        col(feature).cast("double")
    )

df = df.withColumn(
    "label",
    col(label_column).cast("double")
)

df = df.na.fill(0, subset=FEATURES)

# Keep only features + label.
# This also removes the original is_attack column
# when label_column is is_attack.
df = df.select(FEATURES + ["label"])

prepared_rows = df.count()

print(f"Prepared rows:       {prepared_rows:,}")


# ============================================================
# Strict ordered split
# ============================================================

print("\nCreating STRICT ordered train/test split...")
print("No random sampling is used.")

split_position = int(
    prepared_rows * (1 - TEST_RATIO)
)

# Create deterministic row identifier
df = df.withColumn(
    "_row_id",
    monotonically_increasing_id()
)

# Order by the generated row ID
df = df.orderBy("_row_id")

# Use row_number instead of subtract().
# This guarantees that train/test have exactly the
# intended row counts and avoids the column mismatch.
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window = Window.orderBy("_row_id")

df = df.withColumn(
    "_position",
    row_number().over(window)
)

train_df = (
    df
    .filter(col("_position") <= split_position)
    .drop("_row_id", "_position")
)

test_df = (
    df
    .filter(col("_position") > split_position)
    .drop("_row_id", "_position")
)

train_rows = train_df.count()
test_rows = test_df.count()

print(f"Training rows:       {train_rows:,}")
print(f"Testing rows:        {test_rows:,}")


# ============================================================
# Class distribution
# ============================================================

print("\nTraining class distribution:")
train_df.groupBy("label").count().orderBy("label").show()

print("Testing class distribution:")
test_df.groupBy("label").count().orderBy("label").show()


# ============================================================
# Feature vector
# ============================================================

print("\nCreating feature vectors...")

assembler = VectorAssembler(
    inputCols=FEATURES,
    outputCol="features"
)

train_vector = assembler.transform(train_df)
test_vector = assembler.transform(test_df)


# ============================================================
# Random Forest
# ============================================================

print("=" * 70)
print("STRICT GENERALIZATION RANDOM FOREST TEST")
print("=" * 70)

rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=100,
    maxDepth=10,
    seed=42
)

print("\nTraining Random Forest...")

train_start = time.time()

model = rf.fit(train_vector)

training_time = time.time() - train_start

print(f"Training time: {training_time:.4f}s")


# ============================================================
# Prediction
# ============================================================

print("\nRunning predictions...")

prediction_start = time.time()

predictions = model.transform(test_vector)

prediction_count = predictions.count()

prediction_time = time.time() - prediction_start

print(f"Predictions: {prediction_count:,}")
print(f"Prediction time: {prediction_time:.4f}s")


# ============================================================
# Evaluation
# ============================================================

print("\nEvaluating model...")

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


# ============================================================
# Results
# ============================================================

print("\nModel Performance")
print("-" * 40)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")


# ============================================================
# Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

(
    predictions
    .groupBy("label", "prediction")
    .count()
    .orderBy("label", "prediction")
    .show()
)


# ============================================================
# Feature Importance
# ============================================================

print("\nFeature Importance")
print("-" * 40)

feature_importance = model.featureImportances

importance_data = []

for feature, importance in zip(
    FEATURES,
    feature_importance
):
    importance_data.append(
        (feature, float(importance))
    )

importance_data.sort(
    key=lambda x: x[1],
    reverse=True
)

for feature, importance in importance_data:
    print(
        f"{feature:<35} {importance:.6f}"
    )


# ============================================================
# Save results
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

results = spark.createDataFrame(
    [
        (
            original_rows,
            unique_rows,
            duplicates_removed,
            train_rows,
            test_rows,
            training_time,
            prediction_time,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
        )
    ],
    [
        "original_rows",
        "unique_rows",
        "duplicates_removed",
        "training_rows",
        "testing_rows",
        "training_time_sec",
        "prediction_time_sec",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
)

results_path = os.path.join(
    OUTPUT_DIR,
    "strict_generalization_results.csv"
)

results.toPandas().to_csv(
    results_path,
    index=False
)


# ============================================================
# Save feature importance
# ============================================================

importance_df = spark.createDataFrame(
    importance_data,
    ["feature", "importance"]
)

importance_path = os.path.join(
    OUTPUT_DIR,
    "strict_feature_importance.csv"
)

importance_df.toPandas().to_csv(
    importance_path,
    index=False
)


# ============================================================
# Summary
# ============================================================

total_time = (
    training_time
    + prediction_time
)

print("\n" + "=" * 70)
print("STRICT GENERALIZATION SUMMARY")
print("=" * 70)

print(f"Original rows:       {original_rows:,}")
print(f"Unique rows:         {unique_rows:,}")
print(f"Duplicates removed:  {duplicates_removed:,}")
print(f"Training rows:       {train_rows:,}")
print(f"Testing rows:        {test_rows:,}")

print(f"Training time:       {training_time:.4f}s")
print(f"Prediction time:     {prediction_time:.4f}s")

print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1 Score:            {f1:.4f}")
print(f"ROC AUC:             {roc_auc:.4f}")

print(f"Total ML time:       {total_time:.4f}s")

print("\nResults saved to:")
print(results_path)

print(importance_path)

print("=" * 70)
print("Strict generalization validation completed successfully")
print("=" * 70)


# ============================================================
# Stop Spark
# ============================================================

spark.stop()