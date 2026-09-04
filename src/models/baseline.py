from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import pandas as pd


GOLD_FILE = Path("data/gold/network_traffic_features.parquet")


def prepare_data():
    df = pd.read_parquet(GOLD_FILE)

    X = df[
        [
            "src_port",
            "dst_port",
            "packet_count",
            "bytes",
            "duration_bytes"
        ]
    ]

    y = df["is_attack"]

    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    print("Actual:", y_test.to_numpy())
    print("Predicted:", predictions)
    print(f"Accuracy: {accuracy:.2f}")

    print("Baseline model trained")
    print("Predictions:", predictions)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print("Gold data loaded")
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y


if __name__ == "__main__":
    prepare_data()