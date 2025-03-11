use super::Vector;
use std::ops::Index;
use std::ops::IndexMut;
use std::fmt;
#[derive(Debug, Clone, PartialEq)]


pub struct Matrix {
    pub rows: Vec<Vector>,
    pub cols: usize,
}

impl Matrix {
    pub fn new(input: Vec<Vec<f64>>) -> Self {
        let cols = input.first().map_or(0, |row| row.len());
        let rows = input.into_iter().map(Vector::new).collect();
        Self { rows, cols }
    }

    pub fn from_vector(input: Vec<Vector>) -> Self {
        let cols = input.first().map_or(0, |row| row.len());
        let rows = input;
        Self { rows, cols }
    }
    
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows: vec![Vector::zeros(cols); rows],
            cols,
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut mat = Self::zeros(size, size);
        for i in 0..size {
            mat.rows[i].data[i] = 1.0;
        }
        mat
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn col_count(&self) -> usize {
        self.cols
    }

    pub fn get_row(&self, index: usize) -> Option<&Vector> {
        self.rows.get(index)
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = Matrix::zeros(self.cols, self.row_count());
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                transposed[j][i] = self.rows[i][j];
            }
        }
        transposed
    }

      // In-place matrix addition
    pub fn add_assign(&mut self, other: &Self) {
        assert!(self.row_count() == other.row_count() && self.col_count() == other.col_count(), 
                "Dimension mismatch in matrix addition");
        for (row_a, row_b) in self.rows.iter_mut().zip(&other.rows) {
            row_a.add_assign(row_b, 1.0);
        }
    }

      // Returns a new matrix sum
    pub fn add(&self, other: &Self) -> Self {
        assert!(self.row_count() == other.row_count() && self.col_count() == other.col_count(), 
                "Dimension mismatch in matrix addition");
        Self {
            rows: self.rows.iter().zip(&other.rows).map(|(a, b)| a.add(b)).collect(),
            cols: self.cols,
        }
    }

    // Matrix-vector multiplication: y = Ax + y
    pub fn gemv(&self, x: &Vector) -> Option<Vector> {
        if self.col_count() != x.len() {
            return None; // Dimension mismatch
        }
        let mut result = Vector::zeros(self.row_count());
        for (row, res) in self.rows.iter().zip(result.data.iter_mut()) {
            *res = row.dot(x);
        }
        Some(result)
    }

    // General matrix-matrix multiplication: C = AB
    /// Optimized matrix multiplication using loop reordering (i-k-j) for better cache locality.
    pub fn gemm(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.col_count(), other.row_count(), "Incompatible matrix dimensions for multiplication");

        let mut result = Matrix::zeros(self.row_count(), other.cols);
        let m = self.row_count();
        let n = other.cols;
        let k = self.cols;

        // Loop reordering to improve cache efficiency
        for i in 0..m {
            for p in 0..k {
                let self_ip = self.rows[i][p]; // Cache this value
                for j in 0..n {
                    result.rows[i][j] += self_ip * other.rows[p][j];
                }
            }
        }

        result
    }

    /// Computes the determinant of a square matrix
    pub fn determinant(&self) -> f64 {
        let n = self.row_count();
        assert!(self.row_count() == self.cols, "Determinant is only defined for square matrices.");

        let (lu, parity) = self.lu_decomposition();
        let mut det = parity as f64; // Keep track of row swaps (-1 if odd swaps, +1 if even)

        // Multiply diagonal elements of U
        for i in 0..n {
            det *= lu[(i, i)];
        }

        det
    }
    /// Performs LU decomposition using Doolittle’s method
    /// Returns (LU matrix, parity of row swaps)
    fn lu_decomposition(&self) -> (Matrix, i32) {
        let n = self.row_count();
        let mut lu = self.clone();
        let mut parity = 1; // Tracks row swaps (affects determinant sign)

        for k in 0..n {
            // Find pivot (largest absolute value in column k)
            let mut pivot_row = k;
            for i in k + 1..n {
                if lu[(i, k)].abs() > lu[(pivot_row, k)].abs() {
                    pivot_row = i;
                }
            }

            // Swap rows if needed
            if pivot_row != k {
                lu.swap_rows(k, pivot_row);
                parity *= -1; // Row swap changes determinant sign
            }

            let pivot = lu[(k, k)];
            if pivot.abs() < 1e-10 {
                return (lu, 0); // Singular matrix (det = 0)
            }

            // Compute L and U factors in place
            for i in k + 1..n {
                lu[(i, k)] /= pivot;
                for j in k + 1..n {
                    lu[(i, j)] -= lu[(i, k)] * lu[(k, j)];
                }
            }
        }

        (lu, parity)
    }

     /// Swaps two rows in the matrix
     fn swap_rows(&mut self, row1: usize, row2: usize) {
        assert!(row1 < self.row_count() && row2 < self.row_count(), "Row indices out of bounds");
        self.rows.swap(row1, row2);
    }

    pub fn scale_row(&mut self, row: usize, factor: f64) {
        assert!(row < self.row_count(), "Row index out of bounds");
        for val in &mut self.rows[row] {
            *val *= factor;
        }
    }

    pub fn add_rows(&mut self, target: usize, source: usize, factor: f64) {
        assert!(target < self.row_count() && source < self.row_count(), "Row indices out of bounds");

        // Split the matrix data into mutable slices to handle target and source row
        let (first, second) = self.rows.split_at_mut(std::cmp::max(target, source));
                
        // We handle both rows separately based on the row indices
        let target_row = &mut first[target];
        let source_row = &second[0];  // Only the first slice will contain the source row

        for (t, s) in target_row.iter_mut().zip(source_row.iter()) {
            *t += factor * s;
        }
    }

    pub fn swap_columns(&mut self, c1: usize, c2: usize) {
        assert!(c1 < self.cols && c2 < self.cols, "Column indices out of bounds");
        for row in &mut self.rows {
            row.swap(c1, c2);
        }
    }

    /// Multiplies an entire column by a scalar
    pub fn scale_column(&mut self, col: usize, factor: f64) {
        assert!(col < self.cols, "Column index out of bounds");
        for row in &mut self.rows {
            row[col] *= factor;
        }
    }

    /// Computes the inverse of the matrix using LU decomposition.
    pub fn inverse(&self) -> Option<Matrix> {
        let n = self.row_count();
        assert!(self.row_count() == self.cols, "Matrix must be square to compute inverse.");

        let (lu, parity) = self.lu_decomposition();
        if parity == 0 {
            return None; // Singular matrix (no inverse)
        }

        // Identity matrix as right-hand side
        let mut identity = Matrix::identity(n);
        let mut inverse = Matrix::zeros(n, n);

        // Solve LU * X = I for each column of the identity matrix
        for col in 0..n {
            let mut b = identity.get_column(col);
            let y = lu.forward_substitution(&b);
            let x = lu.backward_substitution(&y);
            inverse.set_column(col, &x);
        }

        Some(inverse)
    }

    /// Performs forward substitution to solve L * y = b
    fn forward_substitution(&self, b: &Vector) -> Vector {
        let n = self.row_count();
        let mut y = vec![0.0; n];

        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self[(i, j)] * y[j];
            }
            y[i] = sum; // Since L has ones on the diagonal
        }

        Vector::new(y)
    }

    /// Performs backward substitution to solve U * x = y
    fn backward_substitution(&self, y: &Vector) -> Vector {
        let n = self.row_count();
        let mut x = vec![0.0; n];

        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in i + 1..n {
                sum -= self[(i, j)] * x[j];
            }
            x[i] = sum / self[(i, i)]; // Divide by diagonal element
        }

        Vector::new(x)
    }

    /// Returns a column of the matrix as a Vector
    fn get_column(&self, col: usize) -> Vector {
        Vector::new((0..self.row_count()).map(|i| self[(i, col)]).collect())
    }

    /// Sets a column in the matrix from a Vector
    fn set_column(&mut self, col: usize, v: &Vector) {
        for i in 0..self.row_count() {
            self[(i, col)] = v[i];
        }
    }
  
    /// Adds `factor * source_col` to `target_col`
    pub fn add_columns(&mut self, target: usize, source: usize, factor: f64) {
        assert!(target < self.cols && source < self.cols, "Column indices out of bounds");
        for row in &mut self.rows {
            row[target] += factor * row[source];
            }
    }
}


impl Index<usize> for Matrix {
    type Output = Vector;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.rows[row][col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (row, col) = index;
        &mut self.rows[row][col]
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for row in &self.rows {
            let formatted_row: Vec<String> = row.iter().map(|v| format!("{:8.3}", v)).collect();
            writeln!(f, "[{}]", formatted_row.join(" "))?;
        }
        Ok(())
    }
}
