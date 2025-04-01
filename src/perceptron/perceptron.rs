// use rand::seq::SliceRandom;
// use rand::rng;

use crate::math::Vector;

pub struct Perceptron {
    pub weights: Vector, // Using our custom Vector struct
}

impl Perceptron {
    /// Creates a new perceptron with random weights
    pub fn new(input_size: usize) -> Self {
        Self {
            weights: Vector::random(input_size + 1), // Extra weight for bias
        }
    }

    /// Activation function (step function)
    fn step_activation(&self, value: f64) -> i32 {
        if value >= 0.0 { 1 } else { 0 }
    }

    /// Computes the perceptron output
    pub fn predict(&self, input: &Vector) -> i32 {
        let extended_input = Self::extend_with_bias(input);
        let sum = self.weights.dot(&extended_input);
        self.step_activation(sum)
    }

    /// Trains the perceptron using the Perceptron Learning Rule with improvements
    pub fn train(
        &mut self,
        inputs: &Vec<Vector>, 
        targets: &[i32], 
        learning_rate: f64, 
        max_epochs: usize
    ) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");

        for epoch in 0..max_epochs {
            let mut total_error = 0.0;
            let mut updated = false;

            for (x, &target) in inputs.iter().zip(targets.iter()) {
                let extended_input = Self::extend_with_bias(x);
                let prediction = self.predict(&extended_input);
                let error = (target - prediction) as f64;

                if error != 0.0 {
                    self.weights = self.weights.add(&extended_input.scale(error * learning_rate));
                    updated = true;
                }

                total_error += error.abs();
            }

            // Logging progress (Optional)
            println!("Epoch {}: Total Error = {}", epoch + 1, total_error);

            // Early stopping if no updates occurred
            if !updated {
                println!("Training converged after {} epochs.", epoch + 1);
                break;
            }
        }
    }

    /// Returns the perceptron's current weights
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// Helper function to extend input with a bias term (1.0)
    fn extend_with_bias(input: &Vector) -> Vector {
        let mut extended = vec![1.0]; // Bias input
        extended.extend_from_slice(&input.data);
        Vector::new(extended)
    }

    pub fn train_batch(
        &mut self,
        inputs: &Vec<Vector>, 
        targets: &[i32], 
        learning_rate: f64, 
        max_epochs: usize,
        batch_size: usize
    ) {
        assert_eq!(inputs.len(), targets.len(), "Mismatched input and target sizes!");

        for epoch in 0..max_epochs {
            let mut total_error = 0.0;
            let mut weight_updates = Vector::zeros(self.weights.len());
            let mut batch_count = 0;
            let mut updated = false;

            // Shuffle data indices for each epoch
            // indices.shuffle(&mut rng);

            for (x, &target) in inputs.iter().zip(targets.iter()) {
                let extended_input = Self::extend_with_bias(x); // Ensure bias is included
                let prediction = self.predict(&extended_input);
                let error = (target - prediction) as f64;

                if error != 0.0 {
                    weight_updates.add_assign(&extended_input.scale(error * learning_rate), 1.0);
                    updated = true;
                }

                total_error += error.abs();
                batch_count += 1;

                // Apply weight updates after a full batch
                if batch_count == batch_size {
                    self.weights.add_assign(&weight_updates.scale(1.0 / batch_size as f64), 1.0);
                    weight_updates = Vector::zeros(self.weights.len());
                    batch_count = 0;
                }
            }

            // Apply any remaining updates if the batch was incomplete
            if batch_count > 0 {
                self.weights.add_assign(&weight_updates.scale(1.0 / batch_count as f64), 1.0);
            }

            // Logging progress (Optional)
            println!("Epoch {}: Total Error = {}", epoch + 1, total_error);

            // Early stopping if no updates occurred
            if !updated {
                println!("Training converged after {} epochs.", epoch + 1);
                break;
            }
        }
    }



}