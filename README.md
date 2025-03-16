# RustBrain

RustBrain is a machine learning library built from scratch in Rust, starting with a perceptron implementation and expanding to support multi-class classification and neural networks. It features efficient mathematical operations using a custom-built `Vector` and `Matrix` class with Basic Linear Algebra Subprograms (BLAS) support.

## Features
- **Custom Vector & Matrix Implementations**: Built for efficiency and no external dependencies.
- **Perceptron Model**: Supports binary classification with batch learning.
- **Multi-Class Perceptron**: Implements one-vs-all classification for handling multiple classes.
- **Softmax Classification**: (Upcoming) Uses softmax for probability-based classification.
- **BLAS Operations**: Includes optimized linear algebra functions, QR decomposition, and matrix operations.
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

### Working with Vectors
```rust
use rustbrain::math::Vector;

fn main() {
    let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
    let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
    let dot_product = v1.dot(&v2);
    println!("Dot Product: {}", dot_product);
}
```

### Working with Matrices
```rust
use rustbrain::math::Matrix;

fn main() {
    let m1 = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let m2 = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let result = m1.multiply(&m2);
    println!("Matrix Product: {:?}", result);
}
```

## Project Structure
```
rustbrain
│── Cargo.toml
│── Cargo.lock
│── .gitignore
│
├── src
│   ├── lib.rs
│   ├── main.rs (if needed)
│   ├── math
│   │   ├── mod.rs
│   │   ├── vector.rs
│   │   ├── matrix.rs
│   ├── neuralnetwork
│   │   ├── mod.rs
│   │   ├── neuralnetwork.rs
│   ├── perceptron
│   │   ├── mod.rs
│   │   ├── perceptron.rs
│   │   ├── multiclass_perceptron.rs
│   ├── utils
│   │   ├── mod.rs
|   |   ├── activation.rs
|   |   ├── layer.rs
│
├── tests
│   ├── vector_tests.rs
│   ├── matrix_tests.rs
│   ├── perceptron_tests.rs
│   ├── multiclass_perceptron_tests.rs
│   ├── lib.rs
│   ├── neuralnetwork_tests.rs

```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you’d like to improve RustBrain.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Roadmap
- [x] Implement batch learning for perceptron
- [x] Multi-class perceptron (One-vs-All)
- [ ] Implement Softmax Classification
- [x] Extend to Multi-Layer Perceptron (MLP)
- [ ] Add Regularization for better generalization

---

🚀 *RustBrain is designed for efficiency and learning—let's build smarter AI together!*
