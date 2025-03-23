use crate::math::{Matrix, Vector};
use rand::prelude::SliceRandom;
pub struct LinearRegression {
    pub weights: Vector, // Model parameters (including bias)
}

impl LinearRegression {
    /// Creates an uninitialized Linear Regression model.
    pub fn new() -> Self {
        Self {
            weights: Vector::new(vec![]),
        }
    }

    /// Trains the model using the closed-form Normal Equation.
    pub fn fit(&mut self, inputs: &[Vector], targets: &Vector) {
        let x_matrix = Matrix::from_vector(
            inputs.iter().map(Self::extend_with_bias).collect()
        );
        let y_vector = targets.clone();

        // Compute (X^T X)
        let xt_x = x_matrix.transpose().gemm(&x_matrix);
        // Compute (X^T y)
        let xt_y = x_matrix.transpose().gemv(&y_vector);

        // Solve for weights using the inverse: (X^T X)^(-1) (X^T y)
        let inv_xt_x = xt_x.inverse(); 
        
        self.weights = inv_xt_x.gemv(&xt_y);
    }

    /// Predict outputs for given inputs.
    pub fn predict(&self, input: &Vector) -> f64 {
        let extended_input = Self::extend_with_bias(input);
        self.weights.dot(&extended_input)
    }

    /// Helper function: Extend input with a bias term.
    fn extend_with_bias(input: &Vector) -> Vector {
        let mut extended = vec![1.0]; // Bias term
        extended.extend_from_slice(&input.data);
        Vector::new(extended)
    }


    /// Trains the model using Stochastic Gradient Descent (SGD)
    pub fn fit_sgd(&mut self, inputs: &Vec<Vector>, targets: &Vector, learning_rate: f64, epochs: usize) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
        let mut rng = rand::rng();
        let n = inputs.len();
        
        // Initialize weights with zeros (including bias)
        self.weights = Vector::new(vec![0.0; inputs[0].len() + 1]);

        for _ in 0..epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng); // Randomize sample order
            
            for &i in indices.iter() {
                let mut x = inputs[i].clone();
                x.data.insert(0, 1.0); // Add bias term
                let prediction = self.weights.dot(&x);
                let error = targets[i] - prediction;
                let gradient = x.scale(error * learning_rate);
                self.weights = self.weights.add(&gradient);
            }
        }
        
    }
}
