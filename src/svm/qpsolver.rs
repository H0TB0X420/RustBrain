use crate::math::Matrix;
use crate::math::Vector;

pub struct QPSolver {
    pub q: Matrix,
    pub p: Vector,
    pub a: Matrix,
    pub b: Vector,
    pub l: Vector,
    pub u: Vector,
}

impl QPSolver {
    /// Creates a new QP Solver with given matrices
    pub fn new(q: Matrix, p: Vector, a: Matrix, b: Vector, l: Vector, u: Vector) -> Self {
        Self { q, p, a, b, l, u }
    }

    /// Solves the Quadratic Programming problem using Sequential Minimal Optimization (SMO)
    pub fn solve_smo(&mut self, max_iters: usize, tolerance: f64) -> Vector {
        let mut alpha = Vector::new(vec![0.0; self.p.len()]);
        let mut b = 0.0;
        
        for _ in 0..max_iters {
            let mut num_changed = 0;
            for i in 0..alpha.len() {
                let e_i = self.q[i].dot(&alpha) + b - self.p[i];
                if (e_i * alpha[i] < -tolerance && alpha[i] < self.u[i]) || (e_i * alpha[i] > tolerance && alpha[i] > self.l[i]) {
                    let j = (i + 1) % alpha.len(); // Select second Lagrange multiplier
                    let e_j = self.q[j].dot(&alpha) + b - self.p[j];
                    let alpha_i_old = alpha[i];
                    let alpha_j_old = alpha[j];

                    let eta = 2.0 * self.q[(i, j)] - self.q[(i, i)] - self.q[(j, j)];
                    if eta >= 0.0 {
                        continue;
                    }
                    
                    alpha[j] -= (e_i - e_j) / eta;
                    alpha[j] = alpha[j].max(self.l[j]).min(self.u[j]);

                    alpha[i] += self.q[(i, j)] * (alpha_j_old - alpha[j]);
                    alpha[i] = alpha[i].max(self.l[i]).min(self.u[i]);

                    let b1 = b - e_i - self.q[(i, i)] * (alpha[i] - alpha_i_old) - self.q[(i, j)] * (alpha[j] - alpha_j_old);
                    let b2 = b - e_j - self.q[(j, i)] * (alpha[i] - alpha_i_old) - self.q[(j, j)] * (alpha[j] - alpha_j_old);
                    
                    if alpha[i] > self.l[i] && alpha[i] < self.u[i] {
                        b = b1;
                    } else if alpha[j] > self.l[j] && alpha[j] < self.u[j] {
                        b = b2;
                    } else {
                        b = (b1 + b2) / 2.0;
                    }

                    num_changed += 1;
                }
            }
            if num_changed == 0 {
                break;
            }
        }
        alpha
    }
}
