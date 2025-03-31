import argparse
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values   # Last column
    return X, y

def load_rust_predictions(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return np.array(data["predictions"])

def verify_logistic_regression(dataset_path, rust_output_path):
    X, y = load_dataset(dataset_path)
    rust_preds = load_rust_predictions(rust_output_path)

    clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    clf.fit(X, y)
    sklearn_preds = clf.predict(X)

    sklearn_acc = accuracy_score(y, sklearn_preds)
    rust_acc = accuracy_score(y, rust_preds)
    diff = abs(sklearn_acc - rust_acc)

    print("Scikit-learn Accuracy:", sklearn_acc)
    print("RustBrain Accuracy:", rust_acc)
    print("Accuracy Difference:", diff)

    assert diff < 0.01, "Accuracy mismatch exceeds tolerance!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RustBrain vs scikit-learn Verifier")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., logistic)")
    parser.add_argument("--dataset", type=str, default="../datasets/iris.csv", help="Path to dataset CSV")
    parser.add_argument("--rust-output", type=str, default="../rust_outputs/logistic.json", help="Path to Rust output JSON")
    args = parser.parse_args()

    if args.model == "logistic":
        verify_logistic_regression(args.dataset, args.rust_output)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
