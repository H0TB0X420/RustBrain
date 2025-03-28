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

    pub fn solve_smo(&mut self, max_iters: usize, tolerance: f64) -> Vector {
        let mut alpha = Vector::new(vec![0.0; self.p.len()]);
        let mut b = 0.0;
        let mut prev_alpha = alpha.clone();

        for iter in 0..max_iters {
            let mut num_changed = 0;
            for i in 0..alpha.len() {
                let f_i = self.q[i].dot(&alpha);
                let f_x_i = if self.y[i] != 0.0 { f_i / self.y[i] } else { 0.0 };
                let y_i = self.y[i];
                let e_i = f_x_i - y_i;

                let violates_kkt =
                    (y_i * f_x_i < 1.0 && alpha[i] < self.u[i]) ||
                    (y_i * f_x_i > 1.0 && alpha[i] > self.l[i]);

                if violates_kkt {
                    let mut max_diff = 0.0;
                    let mut selected_j = None;
                    for j in 0..alpha.len() {
                        if j == i { continue; }
                        let f_j = self.q[j].dot(&alpha);
                        let f_x_j = if self.y[j] != 0.0 { f_j / self.y[j] } else { 0.0 };
                        let e_j = f_x_j - self.y[j];
                        let diff = (e_i - e_j).abs();
                        if diff > max_diff {
                            max_diff = diff;
                            selected_j = Some(j);
                        }
                    }
                    let j = if let Some(j_val) = selected_j { j_val } else { continue; };

                    let f_j = self.q[j].dot(&alpha);
                    let f_x_j = if self.y[j] != 0.0 { f_j / self.y[j] } else { 0.0 };
                    let e_j = f_x_j - self.y[j];

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

                    let b1 = b - e_i
                        - self.q[(i, i)] * (alpha[i] - alpha_i_old)
                        - self.q[(i, j)] * (alpha[j] - alpha_j_old);
                    let b2 = b - e_j
                        - self.q[(j, i)] * (alpha[i] - alpha_i_old)
                        - self.q[(j, j)] * (alpha[j] - alpha_j_old);

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
