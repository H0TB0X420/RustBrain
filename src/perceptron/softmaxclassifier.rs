use crate::math::{Matrix, Vector};
use std::f64::EPSILON;

/// Softmax Classifier Struct
pub struct SoftmaxClassifier {
    pub weights: Matrix, // Each row represents a class
}

impl SoftmaxClassifier {
    /// Creates a new Softmax classifier with random weights.
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        let weights = Matrix::random(num_classes, input_size + 1); // Include bias
        Self { weights }
    }

    /// Applies softmax to convert raw scores into probabilities.
    pub fn softmax(logits: &Vector) -> Vector {
        let max_logit = logits.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = logits.data.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f64>();
        Vector::new(exp_values.into_iter().map(|v| v / sum_exp).collect())
    }

    /// Computes the cross-entropy loss given predictions and target labels.
    fn cross_entropy_loss(predictions: &Vector, target: usize) -> f64 {
        -predictions[target].ln().max(EPSILON) // Prevent log(0) errors
    }

    /// Predicts the class probabilities for a given input.
    pub fn predict_proba(&self, input: &Vector) -> Vector {
        let extended_input = Self::extend_with_bias(input);
        let logits = self.weights.gemv(&extended_input);
        Self::softmax(&logits)
    }

    /// Predicts the class label with the highest probability.
    pub fn predict(&self, input: &Vector) -> usize {
        let probabilities = self.predict_proba(input);
        probabilities.data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0
    }

    /// Trains the classifier using batch gradient descent.
    pub fn train_batch(
        &mut self,
        inputs: &[Vector],
        targets: &[usize], 
        learning_rate: f64, 
        batch_size: usize,
        epochs: usize,
    ) {
        assert_eq!(inputs.len(), targets.len(), "Mismatched input and target sizes!");

        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut weight_updates = Matrix::zeros(self.weights.row_count(), self.weights.cols);
            let mut batch_count = 0;

            for (x, &target) in inputs.iter().zip(targets.iter()) {
                let extended_input = Self::extend_with_bias(x);
                let probabilities = self.predict_proba(x);

                total_loss += Self::cross_entropy_loss(&probabilities, target);

                // Compute gradients for softmax loss
                let mut gradient = probabilities;
                gradient[target] -= 1.0; // ∂L/∂z = P - Y

                for i in 0..self.weights.row_count() {
                    for j in 0..self.weights.cols {
                        weight_updates[(i, j)] += gradient[i] * extended_input[j];
                    }
                }

                batch_count += 1;

                // Apply batch update
                if batch_count == batch_size {
                    weight_updates.scale(-learning_rate / batch_size as f64);
                    self.weights.add_assign(&weight_updates);
                    weight_updates = Matrix::zeros(self.weights.row_count(), self.weights.cols);
                    batch_count = 0;
                }
            }

            // Apply any remaining updates
            if batch_count > 0 {
                weight_updates.scale(-learning_rate / batch_count as f64);
                self.weights.add_assign(&weight_updates);
            }

            println!("Epoch {}: Loss = {}", epoch + 1, total_loss / inputs.len() as f64);
        }
    }

    /// Helper function: Extend input with bias (prepend 1.0)
    fn extend_with_bias(input: &Vector) -> Vector {
        let mut extended = vec![1.0]; // Bias term
        extended.extend_from_slice(&input.data);
        Vector::new(extended)
    }
}
