use crate::math::{Vector, Matrix};
use crate::svm::QPSolver;
use crate::utils::RandomFourierFeatures;

pub enum Kernel {
    Linear,
    Polynomial { degree: usize },
    RBF { gamma: f64 },
}

pub struct KernelSVM {
    pub alpha: Vector,             // Lagrange multipliers for all training samples
    pub bias: f64,                 // Bias term computed from support vectors
    pub c: f64,                    // Regularization parameter (soft margin)
    pub rff: RandomFourierFeatures, // RFF transformer for kernel approximation
    pub support_vectors: Vec<Vector>, // Transformed support vectors (only those with non-zero alpha)\n    pub support_targets: Vector,      // Corresponding target values for support vectors\n    pub support_alphas: Vector,       // Corresponding alpha values for support vectors\n}
    support_targets: Vector,
    support_alphas: Vector,
}

impl KernelSVM {
    /// Creates a new Kernel SVM model using RFF.
    /// Support vectors, targets, and alphas are initialized as empty.
    pub fn new(c: f64, input_dim: usize, feature_dim: usize, gamma: f64) -> Self {
        Self {
            alpha: Vector::new(vec![]),
            bias: 0.0,
            c,
            rff: RandomFourierFeatures::new(input_dim, feature_dim, gamma),
            support_vectors: Vec::new(),
            support_targets: Vector::new(vec![]),
            support_alphas: Vector::new(vec![]),
        }
    }
    
    /// Train Kernel SVM using Quadratic Programming (QP) Solver with SMO.
    /// This function computes and stores support vectors, targets, and alphas internally.
    pub fn fit_qp(&mut self, inputs: &Vec<Vector>, targets: &Vector) {
            assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
            
            // Transform all inputs using RFF
            let transformed_inputs: Vec<Vector> = inputs.iter().map(|x| self.rff.transform(x)).collect();
            // println!("Transformed Inputs: \n {:?}", transformed_inputs);
            let n = transformed_inputs.len();
            let mut q = Matrix::zeros(n, n);
            let p = Vector::new(vec![-1.0; n]);
            let a = Matrix::from_vector(vec![targets.clone()]);
            let b = Vector::new(vec![0.0]);
            let l = Vector::new(vec![0.0; n]);
            let u = Vector::new(vec![self.c; n]);
            
            // Construct Q matrix: Q[i,j] = y_i * y_j * K(x_i, x_j), with K approximated by dot product
            for i in 0..n {
                for j in 0..n {
                    q[(i, j)] = targets[i] * targets[j] * transformed_inputs[i].dot(&transformed_inputs[j]);
                }
            }
            println!("Q[0,0]: {}", q[(0,0)]);
            println!("Q[0,1]: {}", q[(0,1)]);
            let mut qp_solver = QPSolver::new(q, p, a, b, l, u, targets.clone());
            self.alpha = qp_solver.solve_smo(1000, 1e-5);
            
            // Identify support vectors (where α > threshold) and store them along with corresponding targets and alphas
            let threshold = 1e-5;
            let mut sup_vecs = Vec::new();
            let mut sup_targs = Vec::new();
            let mut sup_alphas = Vec::new();
            for i in 0..n {
                if self.alpha[i] > threshold {
                    sup_vecs.push(transformed_inputs[i].clone());
                    sup_targs.push(targets[i]);
                    sup_alphas.push(self.alpha[i]);
                }
            }
            self.support_vectors = sup_vecs;
            self.support_targets = Vector::new(sup_targs);
            self.support_alphas = Vector::new(sup_alphas);
            
            // Compute bias using support vectors (average over those support vectors)\n        
            let mut bias_sum = 0.0;
            let mut count = 0;
            for i in 0..n {
                if self.alpha[i] > threshold {
                    let mut sum = 0.0;
                    for j in 0..n {
                        sum += self.alpha[j] * targets[j] * transformed_inputs[j].dot(&transformed_inputs[i]);
                    }
                    bias_sum += targets[i] - sum;
                    count += 1;
                }
            }
            if count > 0 {
                self.bias = bias_sum / count as f64;
            }
        }
    
        /// Predicts the class label (-1 or 1) using the stored support vectors.
        pub fn predict(&self, input: &Vector) -> i32 {
            let transformed_input = self.rff.transform(input);
            let mut sum = 0.0;
            for i in 0..self.support_vectors.len() {
                sum += self.support_alphas[i] * self.support_targets[i] * self.support_vectors[i].dot(&transformed_input);
            }
            if sum + self.bias >= 0.0 { 1 } else { -1 }
        }
    }