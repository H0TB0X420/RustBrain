use crate::math::Matrix;
use crate::math::Vector;

pub struct QPSolver {
    pub q: Matrix,
    pub p: Vector,
    pub a: Matrix,
    pub b: Vector,
    pub l: Vector,
    pub u: Vector,
    pub y: Vector, // actual labels
}

impl QPSolver {
    /// Creates a new QP Solver with given matrices
    pub fn new(q: Matrix, p: Vector, a: Matrix, b: Vector, l: Vector, u: Vector, y: Vector) -> Self {
        Self { q, p, a, b, l, u, y}
    }

    /// Solves the Quadratic Programming problem using Sequential Minimal Optimization (SMO)
    pub fn solve_smo(&mut self, max_iters: usize, tolerance: f64) -> Vector {
        let mut alpha = Vector::new(vec![0.0; self.p.len()]);
        let mut b = 0.0;
        let mut prev_alpha = alpha.clone();
    
        for iter in 0..max_iters {
            let mut num_changed = 0;
            for i in 0..alpha.len() {
                // Compute error for i: E_i = f(x_i) - y_i
                let f_i = self.q[i].dot(&alpha); // this is y_i * f(x_i)
                let e_i = (f_i / self.y[i]) - self.y[i];    // compute f(x_i) - y_i

                println!("E_I: {}", e_i);
                // Check if KKT conditions are violated for i
                if (e_i * alpha[i] < -tolerance && alpha[i] < self.u[i])
                    || (e_i * alpha[i] > tolerance && alpha[i] > self.l[i])
                {
                    // Select j that maximizes |E_i - E_j|
                    let mut max_diff = 0.0;
                    let mut selected_j = None;
                    for j in 0..alpha.len() {
                        if j == i { continue; }
                        let e_j = self.q[j].dot(&alpha) + b - self.p[j];
                        let diff = (e_i - e_j).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            selected_j = Some(j);
                        }
                    }
                    // If no candidate found, skip update for i
                    let j = if let Some(j_val) = selected_j { j_val } else { continue; };
    
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
            
            // Debug: compute change in alpha and objective value
            let diff = alpha.add(&prev_alpha.scale(-1.0)).norm();
            let obj = 0.5 * alpha.dot(&self.q.gemv(&alpha)) - alpha.data.iter().sum::<f64>();
            println!("Iteration {}: Objective = {}, Change = {}", iter, obj, diff);
            
            if diff < tolerance {
                println!("Convergence reached at iteration {}.", iter);
                break;
            }
            prev_alpha = alpha.clone();
            if num_changed == 0 {
                break;
            }
        }
        alpha
    }
    
}
