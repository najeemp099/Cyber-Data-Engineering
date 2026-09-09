# Cyber-Data-Engineering

A data engineering and benchmarking project for working with large network traffic datasets and checking how well machine learning models perform as the amount of data increases.

The main focus of this project is not only detecting attacks. It is also about building a proper data pipeline, storing the data in different stages, processing it with Apache Spark, and measuring the time and performance of the ML pipeline.

## What I worked on

The project uses the CSE-CIC-IDS2018 network traffic dataset.

The pipeline follows three main data stages:

```text
Raw Data
   ↓
Bronze
   ↓
Silver
   ↓
Gold
   ↓
Spark Processing
   ↓
ML Training
   ↓
Benchmarking
   ↓
Results
```

### Data stages

**Bronze**

Raw network traffic data is kept here after ingestion.

**Silver**

The data is cleaned and prepared for further processing. This includes removing duplicate records, handling invalid values, and preparing the required columns.

**Gold**

Processed Parquet files are stored here and used for the analysis and machine learning experiments.

## Dataset

The project uses the CSE-CIC-IDS2018 dataset and includes separate traffic files for different attack scenarios.

For the cross-dataset experiment, I used:

* Training: `Botnet-Friday-02-03-2018`
* Testing: `Bruteforce-Wednesday-14-02-2018`

The important part of this experiment is that the test dataset is completely separate from the training dataset.

There is no random train/test split in this experiment.

## Machine Learning

Two Random Forest experiments were implemented.

### 1. Strict Generalization

The first experiment uses a normal train/test split from the prepared data to establish a baseline result.

The experiment produced:

* Prepared rows: 579,640
* Training rows: 34,778
* Testing rows: 544,862
* Accuracy: 99.83%
* Precision: 99.83%
* Recall: 99.83%
* F1 Score: 99.83%
* ROC AUC: 99.98%

The results are stored in:

```text
results/strict_generalization/
```

### 2. Cross-Dataset Generalization

The second experiment is more important for checking generalization.

The model is trained using the Botnet traffic dataset and then tested on the separate Bruteforce dataset.

Results:

```text
Training rows: 579,640
Testing rows: 550,328

Training time:   27.09 seconds
Prediction time: 1.19 seconds
Total ML time:   28.28 seconds

Accuracy:   0.8315
Precision:  0.0005
Recall:     0.0000
F1 Score:   0.0000
ROC AUC:    0.9796
PR AUC:     0.6464
```

Confusion matrix:

```text
TN: 457,574
FP:   1,910
FN:  90,843
TP:       1
```

The result shows an important problem: although the ROC AUC is high, the model does not generalize well to the attack class in the unseen Bruteforce dataset.

I kept this result because it is useful for understanding the difference between performance on familiar data and performance on a completely different attack scenario.

The cross-dataset results are stored in:

```text
results/cross_dataset/
```

## Feature Analysis

Feature importance was also calculated using the Random Forest model.

Some of the important features were:

```text
Flow_Packets_per_s
Flow_Duration
Protocol
Total_Fwd_Packets
Fwd_Packets_Length_Total
Packet_Length_Mean
Total_Backward_Packets
Flow_Bytes_per_s
Bwd_Packets_Length_Total
Packet_Length_Std
```

The feature importance results are saved as CSV files under the results directories.

## Benchmarking

The project also contains Spark benchmarking scripts for measuring how the processing time changes with different data sizes.

The benchmark results include:

* Dataset size
* Processing time
* Training time
* Prediction time
* Model performance

The generated benchmark results and plots are available under:

```text
results/benchmark/
results/plots/
```

## Project Structure

```text
Cyber-Data-Engineering/
│
├── config/
│   ├── config.yaml
│   └── dataset.yaml
│
├── data/
│   ├── raw/
│   │   └── cse-cic-ids2018/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── docs/
│   └── architecture.md
│
├── results/
│   ├── benchmark/
│   ├── cross_dataset/
│   ├── feature_ablation/
│   ├── feature_correlation/
│   ├── feature_separability/
│   ├── plots/
│   └── strict_generalization/
│
├── src/
│   ├── analysis/
│   ├── benchmarking/
│   ├── features/
│   ├── ingestion/
│   ├── models/
│   ├── preprocessing/
│   └── spark/
│
├── requirements.txt
└── README.md
```

## Main Technologies

* Python
* Apache Spark
* PySpark
* Pandas
* NumPy
* PyArrow
* Parquet
* Scikit-learn
* Matplotlib
* Git

## Running the project

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

The project contains separate scripts for ingestion, preprocessing, feature analysis, benchmarking, and Spark-based model validation.

For example, the cross-dataset validation can be executed with:

```powershell
python src\analysis\validate_cross_dataset.py
```

The generated results will be written to:

```text
results/cross_dataset/
```

## Current Status

The main pipeline and experiments are completed.

The project currently has:

* Raw, Bronze, Silver and Gold data stages
* Parquet-based data storage
* Data cleaning and validation
* Feature analysis
* Spark processing
* Random Forest baseline
* Strict generalization experiment
* Cross-dataset generalization experiment
* Scalability benchmarking
* Performance visualizations
* CSV result files for the experiments

The next step is mainly project polishing, documentation, and making the experimental results easier to understand and reproduce.

## Why this project

I built this project to get practical experience with data engineering for large-scale datasets and to understand where machine learning fits into a data pipeline.

A major part of the work was measuring the pipeline instead of only looking at model accuracy. The experiments helped me compare processing time, dataset size, model performance, and generalization behaviour.
