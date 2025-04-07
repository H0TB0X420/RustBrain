import subprocess
import os
import json
import sys

# Define model types based on filename patterns
MODEL_MAP = {
    "linreg": "linear",
    "logreg": "logistic",
    "perceptron": "perceptron",
    "softmax": "logistic",  # Treat softmax as multiclass logistic for now
    "sgd": "sgd",
    "xor": "logistic",      # neural network XOR — not covered yet
}

def guess_model_type(filename):
    for key, model_type in MODEL_MAP.items():
        if key in filename.lower():
            return model_type
    return None

def run_test(file_path):
    filename = os.path.basename(file_path)
    model = guess_model_type(filename)

    if model in ["xor"]:
        print(f"Skipping {filename}: {model} not suitable for testing")
        return False

    if model is None:
        print(f"⚠️  Skipping {filename}: unknown model type")
        return False

    cmd = [
        sys.executable,
        "sklearn_verifier.py",
        "--model", model,
        "--input", file_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {filename} passed\n{result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {filename} failed:\n{e.stdout.strip()}\n{e.stderr.strip()}")
        return False

def main():
    base_dir = os.path.join(os.path.dirname(__file__), "../rust_outputs")
    files = [f for f in os.listdir(base_dir) if f.endswith(".json")]

    passed = 0
    failed = 0

    for f in sorted(files):
        print(f"\n▶ Running verifier for: {f}")
        success = run_test(os.path.join(base_dir, f))
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n📊 Verification Summary: {passed} ran successfully, {failed} failed to complete.")

if __name__ == "__main__":
    main()
