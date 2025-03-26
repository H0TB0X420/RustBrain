use crate::math::{Vector, Matrix};
use rand::Rng;
use std::f64::consts::PI;

pub struct RandomFourierFeatures {
    pub weights: Matrix,
    pub biases: Vector,
    pub gamma: f64,
}

impl RandomFourierFeatures {
    /// Creates a new RFF transformer
    pub fn new(input_dim: usize, feature_dim: usize, gamma: f64) -> Self {
        let mut rng = rand::rng();
        let mut weights = Matrix::random(feature_dim, input_dim);
        weights.scale((2.0 * gamma).sqrt());
        let biases = Vector::new((0..feature_dim).map(|_| rng.random_range(0.0..(2.0 * PI))).collect());
        
        Self { weights, biases, gamma }
    }
    
    /// Transforms input data using RFF mapping
    pub fn transform(&self, input: &Vector) -> Vector {
        let projection = self.weights.gemv(input).add(&self.biases);
        Vector::new(projection.iter().map(|v| (2.0 / (self.weights.row_count() as f64)).sqrt() * v.cos()).collect())
    }
}