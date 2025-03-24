use crate::math::{Vector, Matrix};
use rand::Rng;

pub struct HardMarginSVM {
    pub weights: Vector, // Weight vector
    pub bias: f64,       // Bias term
    pub learning_rate: f64,
    pub epochs: usize,
}

impl HardMarginSVM {
    /// Creates a new Hard-Margin SVM model
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: Vector::new((0..n_features).map(|_| rng.gen_range(-0.01..0.01)).collect()),
            bias: rng.gen_range(-0.01..0.01),
            learning_rate,
            epochs,
        }
    }

    /// Train Hard-Margin SVM using Stochastic Gradient Descent (SGD)
    pub fn fit(&mut self, inputs: &Vec<Vector>, targets: &Vector) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");

        for _ in 0..self.epochs {
            for (x, &y) in inputs.iter().zip(targets.data.iter()) {
                let margin = y * (self.weights.dot(x) + self.bias);
                if margin < 1.0 {
                    self.weights = self.weights.scale(1.0 - self.learning_rate)
                        .add(&x.scale(self.learning_rate * y));
                    self.bias += self.learning_rate * y;
                }
            }
        }
    }

    /// Predicts the class label (-1 or 1)
    pub fn predict(&self, input: &Vector) -> i32 {
        if self.weights.dot(input) + self.bias >= 0.0 { 1 } else { -1 }
    }
}