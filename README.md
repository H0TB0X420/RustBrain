# RustBrain

RustBrain is a machine learning library built from scratch in Rust, starting with a perceptron implementation and expanding to support multi-class classification, neural networks, and regression models. It features efficient mathematical operations using a custom-built `Vector` and `Matrix` class with Basic Linear Algebra Subprograms (BLAS) support.

## Features
- **Custom Vector & Matrix Implementations**: Built for efficiency and no external dependencies.
- **Perceptron Model**: Supports binary classification with batch learning.
- **Multi-Class Perceptron**: Implements one-vs-all classification for handling multiple classes.
- **Multi-Layer Perceptron (MLP)**: Supports deep learning architectures.
- **Softmax Classification**: Implements probability-based classification using softmax and cross-entropy loss.
- **Linear Regression**: Includes both closed-form (Normal Equation) and Stochastic Gradient Descent (SGD) methods.
- **BLAS Operations**: Includes optimized linear algebra functions, QR decomposition, and matrix operations.
- **Comprehensive Testing**: Unit tests for all major components.
- **Modular Design**: Well-structured for future expansions.

## Installation
To use RustBrain, clone the repository and build the project:
```sh
git clone https://github.com/H0TB0X420/RustBrain
cd rustbrain
cargo build
```

## Usage
### Creating and Using a Perceptron with Batch Learning
```rust
use rustbrain::perceptron::Perceptron;

fn main() {
    let mut perceptron = Perceptron::new(3); // 3 input features
    let inputs = vec![1.0, -2.0, 0.5];
    let output = perceptron.predict(&inputs);
    println!("Prediction: {}", output);
}
```

### Multi-Class Perceptron
```rust
use rustbrain::perceptron::MultiClassPerceptron;

fn main() {
    let mut perceptron = MultiClassPerceptron::new(2, 3, 0.1); // 2 input features, 3 classes
    let inputs = vec![Vector::new(vec![1.0, 0.0]), Vector::new(vec![0.0, 1.0])];
    let targets = vec![0, 1];
    perceptron.train_batch(&inputs, &targets, 0.1, 2, 100);
    let prediction = perceptron.predict(&inputs[0]);
    println!("Predicted class: {}", prediction);
}
```

### Softmax Classification
```rust
use rustbrain::classification::SoftmaxClassifier;

fn main() {
    let mut classifier = SoftmaxClassifier::new(3, 3); // 3 input features, 3 classes
    let inputs = vec![Vector::new(vec![1.0, 0.0, 2.0]), Vector::new(vec![0.0, 1.0, -1.0])];
    let targets = vec![0, 1];
    classifier.train(&inputs, &targets, 0.1, 1000);
    let prediction = classifier.predict(&inputs[0]);
    println!("Predicted class: {}", prediction);
}
```

### Linear Regression (Closed-Form Solution)
```rust
use rustbrain::regression::LinearRegression;

fn main() {
    let inputs = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])];
    let targets = Vector::new(vec![5.0, 10.0]);
    
    let mut model = LinearRegression::new();
    model.fit(&inputs, &targets);
    
    let prediction = model.predict(&Vector::new(vec![2.0, 3.0]));
    println!("Prediction: {}", prediction);
}
```

### Linear Regression (Stochastic Gradient Descent)
```rust
use rustbrain::regression::LinearRegression;

fn main() {
    let inputs = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])];
    let targets = Vector::new(vec![5.0, 10.0]);
    
    let mut model = LinearRegression::new();
    model.fit_sgd(&inputs, &targets, 0.01, 1000);
    
    let prediction = model.predict(&Vector::new(vec![2.0, 3.0]));
    println!("Prediction: {}", prediction);
}
```

## Roadmap
- [x] Implement batch learning for perceptron
- [x] Multi-class perceptron (One-vs-All)
- [x] Extend to Multi-Layer Perceptron (MLP)
- [x] Implement closed-form Linear Regression
- [x] Implement Stochastic Gradient Descent Linear Regression
- [x] Implement Softmax Classification
- [x] Add Regularization for better generalization

---

🚀 *RustBrain is designed for efficiency and learning—let's build smarter AI together!*
