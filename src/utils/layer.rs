use crate::math::{Matrix, Vector};
use rand::Rng;
/// A single neural network layer.
/// It holds a weight matrix of dimensions (num_neurons x (input_dim + 1)),
/// where the extra column accounts for the bias.
pub struct Layer {
    pub weights: Matrix, // Dimensions: neurons x (input_dim + 1)
}

impl Layer {
    /// Creates a new layer with the given input and output sizes.
    /// For simplicity, weights are initialized to zeros.
    /// (In practice you’d want to initialize randomly.)
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::rng();
        let weights = Matrix::new(
            (0..output_dim)
                .map(|_| {
                    (0..=input_dim)
                        .map(|_| rng.random_range(-1.0..1.0)) // Random values in [-1, 1]
                        .collect()
                })
                .collect(),
        );
        Self { weights }
    }

    /// Helper function: Extend a vector with a bias term (always 1.0)
    pub fn extend_with_bias(input: &Vector) -> Vector {
        let mut extended = vec![1.0];
        extended.extend_from_slice(&input.data);
        Vector::new(extended)
    }
}