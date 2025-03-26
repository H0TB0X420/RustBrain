use crate::math::{Vector, Matrix};
use crate::QPSolver;
use crate::utils::RandomFourierFeatures;


pub enum Kernel {
    Linear,
    Polynomial { degree: usize },
    RBF { gamma: f64 },
}

pub struct KernelSVM {
    pub alpha: Option<Vector>,
    pub bias: f64,
    pub c: f64,
    pub rff: RandomFourierFeatures,
}

impl KernelSVM {
    /// Creates a new Kernel SVM model using RFF
    pub fn new(c: f64, input_dim: usize, feature_dim: usize, gamma: f64) -> Self {
        Self {
            alpha: None,
            bias: 0.0,
            c,
            rff: RandomFourierFeatures::new(input_dim, feature_dim, gamma),
        }
    }
    
    /// Train Kernel SVM using Quadratic Programming (QP) Solver with RFF
    pub fn fit_qp(&mut self, inputs: &Vec<Vector>, targets: &Vector) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
        let transformed_inputs: Vec<Vector> = inputs.iter().map(|x| self.rff.transform(x)).collect();
        let n = transformed_inputs.len();
        let mut q = Matrix::zeros(n, n);
        let p = Vector::new(vec![-1.0; n]);
        let a = Matrix::from_vector(vec![targets.clone()]);
        let b = Vector::new(vec![0.0]);
        let l = Vector::new(vec![0.0; n]);
        let u = Vector::new(vec![self.c; n]);
        
        for i in 0..n {
            for j in 0..n {
                q[(i, j)] = targets[i] * targets[j] * transformed_inputs[i].dot(&transformed_inputs[j]);
            }
        }
        
        let mut qp_solver = QPSolver::new(q, p, a, b, l, u);
        self.alpha = Some(qp_solver.solve_smo(1000, 1e-5));
        
        // Compute bias using support vectors
        let mut bias_sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            if self.alpha.as_ref().unwrap()[i] > 1e-5 {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.alpha.as_ref().unwrap()[j] * targets[j] * transformed_inputs[j].dot(&transformed_inputs[i]);
                }
                bias_sum += targets[i] - sum;
                count += 1;
            }
        }
        if count > 0 {
            self.bias = bias_sum / count as f64;
        }
    }

    /// Predicts the class label (-1 or 1) using RFF-transformed inputs
    pub fn predict(&self, input: &Vector, support_vectors: &Vec<Vector>, targets: &Vector) -> i32 {
        let transformed_input = self.rff.transform(input);
        let mut sum = 0.0;
        for i in 0..support_vectors.len() {
            sum += self.alpha.as_ref().unwrap()[i] * targets[i] * support_vectors[i].dot(&transformed_input);
        }
        if sum + self.bias >= 0.0 { 1 } else { -1 }
    }
}