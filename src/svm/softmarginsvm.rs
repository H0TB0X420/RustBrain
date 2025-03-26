use crate::math::{Vector, Matrix};
use crate::QPSolver;
use rand::Rng;

pub struct SoftMarginSVM {
    pub weights: Vector, // Weight vector
    pub bias: f64,       // Bias term
    pub learning_rate: f64,
    pub epochs: usize,
    pub c: f64, // Regularization parameter
    pub alpha: Option<Vector>,
}

impl SoftMarginSVM {
    /// Creates a new Soft-Margin SVM model
    pub fn new(n_features: usize, learning_rate: f64, epochs: usize, c: f64) -> Self {
        let mut rng = rand::rng();
        Self {
            weights: Vector::new((0..n_features).map(|_| rng.random_range(-0.01..0.01)).collect()),
            bias: rng.random_range(-0.01..0.01),
            learning_rate,
            epochs,
            c,
            alpha: None,
        }
    }

    /// Train Soft-Margin SVM using Stochastic Gradient Descent (SGD)
    pub fn fit(&mut self, inputs: &Vec<Vector>, targets: &Vector) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");

        for _ in 0..self.epochs {
            for (x, &y) in inputs.iter().zip(targets.data.iter()) {
                let margin = y * (self.weights.dot(x) + self.bias);
                if margin < 1.0 {
                    self.weights = self.weights.scale(1.0 - self.learning_rate)
                        .add(&x.scale(self.learning_rate * self.c * y));
                    self.bias += self.learning_rate * self.c * y;
                } else {
                    self.weights = self.weights.scale(1.0 - self.learning_rate);
                }
            }
        }
    }
    
    /// Train Soft-Margin SVM using Quadratic Programming (QP) Solver with SMO
    pub fn fit_qp(&mut self, inputs: &Vec<Vector>, targets: &Vector) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
        let n = inputs.len();
        let mut q = Matrix::zeros(n, n);
        let p = Vector::new(vec![-1.0; n]);
        let a = Matrix::from_vector(vec![targets.clone()]);
        let b = Vector::new(vec![0.0]);
        let l = Vector::new(vec![0.0; n]);
        let u = Vector::new(vec![self.c; n]);
        
        for i in 0..n {
            for j in 0..n {
                q[(i, j)] = targets[i] * targets[j] * inputs[i].dot(&inputs[j]);
            }
        }
        
        let mut qp_solver = QPSolver::new(q, p, a, b, l, u);
        self.alpha = Some(qp_solver.solve_smo(1000, 1e-5));
        
        // Compute final weights and bias
        self.weights = Vector::zeros(inputs[0].len());
        for i in 0..n {
            self.weights = self.weights.add(&inputs[i].scale(self.alpha.as_ref().unwrap()[i] * targets[i]));
        }
        self.bias = targets[0] - self.weights.dot(&inputs[0]);
    }

    /// Predicts the class label (-1 or 1)
    pub fn predict(&self, input: &Vector) -> i32 {
        if self.weights.dot(input) + self.bias >= 0.0 { 1 } else { -1 }
    }
}
