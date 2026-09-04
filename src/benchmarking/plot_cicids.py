from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


RESULT_FILE = Path("data/gold/cicids_benchmark_results.csv")


df = pd.read_csv(RESULT_FILE)

plt.figure(figsize=(8, 5))

plt.plot(
    df["dataset_size"],
    df["training_time"],
    marker="o"
)

plt.xlabel("Dataset Size (Rows)")
plt.ylabel("Training Time (Seconds)")
plt.title("Logistic Regression Scalability on CSE-CIC-IDS2018")

plt.grid(True)

plt.tight_layout()

output = Path("data/gold/cicids_training_time.png")

plt.savefig(output, dpi=300)

print(f"Plot saved: {output}")