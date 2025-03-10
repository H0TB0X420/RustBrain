# RustBrain

RustBrain is a machine learning library built from scratch in Rust, starting with a perceptron implementation and eventually expanding to support neural networks. It features efficient mathematical operations using a custom-built `Vector` and `Matrix` class with Basic Linear Algebra Subprograms (BLAS) support.

## Features
- **Custom Vector & Matrix Implementations**: Built for efficiency and no external dependencies.
- **Perceptron Model**: Implements a basic perceptron for binary classification.
- **BLAS Operations**: Includes optimized linear algebra functions.
- **Modular Design**: Well-structured for future expansions.

## Installation
To use RustBrain, clone the repository and build the project:
```sh
git clone https://github.com/yourusername/rustbrain.git
cd rustbrain
cargo build
```

## Usage
### Creating and Using a Perceptron
```rust
use rustbrain::perceptron::Perceptron;

fn main() {
    let mut perceptron = Perceptron::new(3); // 3 input features
    let inputs = vec![1.0, -2.0, 0.5];
    let output = perceptron.predict(&inputs);
    println!("Prediction: {}", output);
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
│   ├── perceptron
│   │   ├── mod.rs
│   │   ├── perceptron.rs
│   ├── utils
│   │   ├── mod.rs
│
├── tests
│   ├── vector_tests.rs
│   ├── matrix_tests.rs
│   ├── perceptron_tests.rs
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you’d like to improve RustBrain.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Roadmap
- [] Reference WorkingMemory.md

---

🚀 *RustBrain is designed for efficiency and learning—let's build smarter AI together!*

