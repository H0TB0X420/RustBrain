use crate::math::{Matrix, Vector};

pub struct QPSolver {
    pub q: Matrix,       // Q matrix: y_i * y_j * K(x_i, x_j)
    pub p: Vector,       // Linear term in the objective (-1 vector)
    pub a: Matrix,       // Equality constraint matrix
    pub b: Vector,       // Equality constraint target
    pub l: Vector,       // Lower bounds for alpha
    pub u: Vector,       // Upper bounds for alpha
    pub y: Vector,       // Actual target labels (+1/-1)
}

impl QPSolver {
    pub fn new(q: Matrix, p: Vector, a: Matrix, b: Vector, l: Vector, u: Vector, y: Vector) -> Self {
        Self { q, p, a, b, l, u, y }
    }

    pub fn solve(&mut self, max_iters: usize, tolerance: f64) -> Vector {
        let l = self.l.len();
        let mut alpha = Vector::zeros(l);
        let mut grad = self.p.clone(); // Gradient of the objective function
        let active_set: Vec<usize> = (0..l).collect();

        for iter in 0..max_iters {
            let mut i = usize::MAX;
            let mut j = usize::MAX;
            let mut g_max = f64::NEG_INFINITY;
            let mut g_min = f64::INFINITY;

            for &t in &active_set {
                let y_t = self.y[t];
                let g = y_t * grad[t];
                if y_t == 1.0 && alpha[t] < self.u[t] {
                    if -g >= g_max {
                        g_max = -g;
                        i = t;
                    }
                } else if y_t == -1.0 && alpha[t] > self.l[t] {
                    if g >= g_max {
                        g_max = g;
                        i = t;
                    }
                }
            }

            for &t in &active_set {
                let y_t = self.y[t];
                let g = y_t * grad[t];
                if y_t == 1.0 && alpha[t] > self.l[t] {
                    if g <= g_min {
                        g_min = g;
                        j = t;
                    }
                } else if y_t == -1.0 && alpha[t] < self.u[t] {
                    if -g <= g_min {
                        g_min = -g;
                        j = t;
                    }
                }
            }

            if g_max - g_min < tolerance {
                break;
            }

            let q_i = &self.q[i];
            let q_j = &self.q[j];
            let c_i = self.u[i];
            let c_j = self.u[j];

            let yi = self.y[i];
            let yj = self.y[j];

            let s = yi * yj;
            let l_val;
            let h_val;
            if yi != yj {
                l_val = (alpha[j] - alpha[i]).max(0.0);
                h_val = (c_j - c_i + alpha[j] - alpha[i]).min(c_j).min(c_i);
            } else {
                l_val = (alpha[j] + alpha[i] - c_i).max(0.0);
                h_val = (c_j + alpha[j] + alpha[i]).min(c_j);
            }

            if (l_val - h_val).abs() < 1e-12 {
                continue;
            }

            let eta = self.q[(i, i)] + self.q[(j, j)] - 2.0 * self.q[(i, j)];
            if eta <= 0.0 {
                continue;
            }

            let delta = (grad[i] - grad[j]) / eta;
            let mut a_i = alpha[i] + yi * delta;
            let mut a_j = alpha[j] - yj * delta;

            if a_i > h_val {
                a_i = h_val;
            } else if a_i < l_val {
                a_i = l_val;
            }

            a_j = alpha[j] + yj * (alpha[i] - a_i);
            alpha[i] = a_i;
            alpha[j] = a_j;

            for k in 0..l {
                grad[k] += (alpha[i] - a_i) * q_i[k] + (alpha[j] - a_j) * q_j[k];
            }

            println!("Iteration {}: Objective = {:.6}, Alpha Change = {:.6}", iter,
                0.5 * alpha.dot(&self.q.gemv(&alpha)) - alpha.data.iter().sum::<f64>(),
                (alpha[i] - a_i).abs() + (alpha[j] - a_j).abs());
        }

        alpha
    }
}
