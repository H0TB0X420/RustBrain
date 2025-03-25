pub mod math;
pub mod perceptron;
mod utils;
pub mod neuralnetwork;
pub mod linear_regression;
pub mod logistic_regression;
pub mod svm;
// Re-export key components for easier access
pub use math::{Vector, Matrix};
pub use perceptron::{Perceptron, MultiClassPerceptron};
pub use neuralnetwork::NeuralNetwork;
pub use logistic_regression::{LogisticRegression, SoftmaxRegression};
pub use svm::{HardMarginSVM, SoftMarginSVM, QPSolver};