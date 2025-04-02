import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

def load_rust_output(path):
    with open(path, 'r') as f:
        data = json.load(f)
    X = np.array(data["inputs"])
    y_rust = np.array(data["predictions"])
    weights = np.array(data["parameters"]["weights"])
    biases = np.array(data["parameters"]["biases"])
    return X, y_rust, weights, biases

def verify_linear(X, y_rust):
    model = LinearRegression()
    model.fit(X, y_rust)  # using X as both train and test for now
    y_sklearn = model.predict(X)
    mse = mean_squared_error(y_rust, y_sklearn)
    print(f"✅ MSE: {mse:.6f}")
    assert mse < 1e-1, "MSE too high"

def verify_logistic(X, y_rust):
    y_rust_int = y_rust.astype(int)
    model = LogisticRegression()
    model.fit(X, y_rust_int)
    y_sklearn = model.predict(X)
    acc = accuracy_score(y_rust_int, y_sklearn)
    print(f"✅ Accuracy: {acc:.6f}")
    assert acc > 0.95, "Accuracy too low"

def verify_perceptron(X, y_rust):
    y_rust_int = y_rust.astype(int)
    model = Perceptron()
    model.fit(X, y_rust_int)
    y_sklearn = model.predict(X)
    acc = accuracy_score(y_rust_int, y_sklearn)
    print(f"✅ Accuracy: {acc:.6f}")
    assert acc > 0.95, "Accuracy too low"

def verify_sgd_regression(X, y_rust):
    model = SGDRegressor(max_iter=1000, tol=1e-3)
    model.fit(X, y_rust)
    y_sklearn = model.predict(X)
    mse = mean_squared_error(y_rust, y_sklearn)
    print(f"✅ MSE: {mse:.6f}")
    assert mse < 1.0, "MSE too high for SGD regression"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model to verify: linear, logistic, perceptron, sgd")
    parser.add_argument("--input", required=True, help="Path to Rust JSON output file")
    args = parser.parse_args()

    X, y_rust, weights, biases = load_rust_output(args.input)

    if args.model == "linear":
        verify_linear(X, y_rust)
    elif args.model == "logistic":
        verify_logistic(X, y_rust)
    elif args.model == "perceptron":
        verify_perceptron(X, y_rust)
    elif args.model == "sgd":
        verify_sgd_regression(X, y_rust)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
