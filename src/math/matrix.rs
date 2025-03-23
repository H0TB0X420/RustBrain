use super::Vector;
use std::ops::Index;
use std::ops::IndexMut;
use std::fmt;
use rand::Rng;
use std::cmp::PartialEq;

#[derive(Debug, Clone)]


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
    
        for (i, row) in self.rows.iter().enumerate() {
            for (j, &value) in row.data.iter().enumerate() {
                transposed.rows[j].data[i] = value;
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
    pub fn gemv(&self, x: &Vector) -> Vector {
        assert_eq!(self.col_count(), x.len(), "Incompatible matrix dimensions for multiplication");

        let mut result = Vector::zeros(self.row_count());
        for (row, res) in self.rows.iter().zip(result.data.iter_mut()) {
            *res = row.dot(x);
        }
        result
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
    pub fn lu_decomposition(&self) -> (Matrix, i32) {
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

    pub fn split_lu(&self) -> (Matrix, Matrix) {
        let n = self.row_count();
        let mut l = Matrix::identity(n);
        let mut u = Matrix::zeros(n, n);
        
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    l[(i, j)] = self[(i, j)];
                } else {
                    u[(i, j)] = self[(i, j)];
                }
            }
        }
        (l, u)
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

    /// Computes the inverse of the matrix using adj(A) / det(A)
    pub fn inverse(&self) -> Matrix {
        if self.row_count() != self.cols
        {
            panic!("Matrix must be square to compute the inverse matrix.");
        }
        let d = self.determinant();
        if  d == 0.0{
            panic!("Matrix inversion failed! Check for singularity.");
        }
        let d_inv = 1.0 / d;
        let mut inverse = self.cofactor_matrix().transpose();
        inverse.scale(d_inv);
        
        // A * A^-1 = I validation with rounding, should be a 1.0 or a 0.0
        let mut ident_check = self.gemm(&inverse);
        for i in 0..ident_check.row_count() {
            for j in 0..ident_check.col_count() {
                ident_check[i][j] = ident_check[i][j].round();
            }
        }
        // if ident_check != Matrix::identity(self.cols)
        // {
        //     panic!("Matrix inversion failed! Check for singularity.");
        // }
        inverse
    }

    /// Reverses the order of rows in the matrix.
    pub fn reverse_rows(&self) -> Matrix {
        let reversed_rows: Vec<Vector> = self.rows.iter().rev().cloned().collect();
        Matrix::from_vector(reversed_rows)
    }

    /// Performs forward substitution to solve L * y = b
    // fn forward_substitution(&self, b: &Vector) -> Vector {
        // let n = self.row_count();
        // let mut y = vec![0.0; n];

        // for i in 0..n {
        //     let mut sum = b[i];
        //     for j in 0..i {
        //         sum -= self[(i, j)] * y[j];
        //     }
        //     y[i] = sum; // Since L has ones on the diagonal
        // }

    //     Vector::new(y)
    // }

    /// Performs backward substitution to solve U * x = y
    // fn backward_substitution(&self, y: &Vector) -> Vector {
    //     let n = self.row_count();
    //     let mut x = vec![0.0; n];

    //     for i in (0..n).rev() {
    //         let mut sum = y[i];
    //         for j in i + 1..n {
    //             sum -= self[(i, j)] * x[j];
    //         }
    //         x[i] = sum / self[(i, i)]; // Divide by diagonal element
    //     }

    //     Vector::new(x)
    // }

    /// Returns a column of the matrix as a Vector
    pub fn get_column(&self, col: usize) -> Vector {
        Vector::new((0..self.row_count()).map(|i| self[(i, col)]).collect())
    }

    /// Sets a column in the matrix from a Vector
    // fn set_column(&mut self, col: usize, v: &Vector) {
    //     assert_eq!(self.row_count(), v.len(), "Vector length must match matrix row count.");
    
    //     for i in 0..self.row_count() {
    //         self[(i, col)] = v[i];
    //     }
    // }
  
    /// Adds `factor * source_col` to `target_col`
    pub fn add_columns(&mut self, target: usize, source: usize, factor: f64) {
        assert!(target < self.cols && source < self.cols, "Column indices out of bounds");
        for row in &mut self.rows {
            row[target] += factor * row[source];
            }
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let data: Vec<Vec<f64>> = (0..rows)
            .map(|_| (0..cols).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();
        Self::new(data)
    }

    pub fn scale(&mut self, factor: f64) {
        for row in &mut self.rows {
            for val in &mut row.data {
                *val *= factor;
            }
        }
    }

    /// Solves the system of equations represented by the augmented matrix
    /// using Gaussian elimination with partial pivoting.
    /// The matrix must be of size n x (n+1).
    pub fn gaussian_elimination(&mut self) -> Result<Vec<f64>, &'static str> {
        let n = self.row_count();
        // Check if the matrix is augmented: columns == rows + 1
        if self.cols != n + 1 {
            return Err("Matrix is not augmented with the correct number of columns");
        }
        
        // Forward Elimination
        for i in 0..n {
            // Find the pivot row for column i
            let mut max_row = i;
            for k in (i+1)..n {
                if self[(k, i)].abs() > self[(max_row, i)].abs() {
                    max_row = k;
                }
            }
            // Check for singular matrix
            if self[(max_row, i)].abs() < 1e-10 {
                return Err("Singular matrix; no unique solution exists");
            }
            // Swap the pivot row with the current row, if needed
            if max_row != i {
                self.swap_rows(i, max_row);
            }
            
            // Eliminate entries below the pivot
            for j in (i+1)..n {
                let factor = self[(j, i)] / self[(i, i)];
                for k in i..self.cols {
                    self[(j, k)] -= factor * self[(i, k)];
                }
            }
        }
        
        // Back Substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i+1)..n {
                sum += self[(i, j)] * x[j];
            }
            x[i] = (self[(i, self.cols - 1)] - sum) / self[(i, i)];
        }
        Ok(x)
    }

 /// Implements the Gram–Schmidt process to perform QR decomposition.
    /// Returns (Q, R) such that A = Q * R.
    pub fn gram_schmidt(&self) -> (Matrix, Matrix) {
        let m = self.row_count();
        let n = self.cols;
        // Q will be built column by column.
        let mut q_columns: Vec<Vector> = Vec::with_capacity(n);
        // R is an n x n upper triangular matrix.
        let mut r_data: Vec<Vec<f64>> = vec![vec![0.0; n]; n];

        for j in 0..n {
            // Extract j-th column of A.
            let a_j = self.get_column(j);
            let mut v = a_j.clone();
            // For each previous q column, subtract its component.
            for i in 0..j {
                let q_i = &q_columns[i];
                let r_ij = q_i.dot(&a_j);
                r_data[i][j] = r_ij;
                // v = v - r_ij * q_i; using scale and add (with negative scaling)
                v = v.add(&q_i.scale(-r_ij));
            }
            // The norm of v is r_jj.
            let r_jj = v.norm();
            r_data[j][j] = r_jj;
            // If r_jj is near zero, the column is linearly dependent.
            let q_j = if r_jj.abs() < 1e-10 {
                Vector::zeros(v.data.len())
            } else {
                v.scale(1.0 / r_jj)
            };
            q_columns.push(q_j);
        }
        // Form Q matrix from the q_columns.
        // Q is m x n. Each row i of Q is composed of the i-th element of each q_j.
        let mut q_data: Vec<Vec<f64>> = vec![vec![0.0; n]; m];
        for j in 0..n {
            let qj = &q_columns[j];
            for i in 0..m {
                q_data[i][j] = qj[i];
            }
        }
        let q = Matrix::new(q_data);
        let r = Matrix::new(r_data);
        
        (q, r)
    }

    pub fn cofactor_matrix(&self) -> Matrix {
        assert!(self.row_count() == self.col_count(), "Matrix must be square to compute the cofactor matrix.");
        let n = self.row_count();
        let mut cofactors = Matrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let minor = self.minor(i, j);
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                cofactors[(i, j)] = sign * minor.determinant();
            }
        }

        cofactors
    }
    
    /// Computes the minor of a matrix by removing the specified row and column.
    fn minor(&self, row: usize, col: usize) -> Matrix {
        let minor_data: Vec<Vec<f64>> = self.rows.iter().enumerate()
            .filter(|&(r, _)| r != row)
            .map(|(_, row_data)| {
                row_data.data.iter().enumerate()
                    .filter(|&(c, _)| c != col)
                    .map(|(_, &val)| val)
                    .collect()
            })
            .collect();
        Matrix::new(minor_data)
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


impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.row_count() != other.row_count() || self.col_count() != other.col_count() {
            return false;
        }
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                let a = self[i][j];
                let b = other[i][j];
                if a.abs() < f64::EPSILON && b.abs() < f64::EPSILON {
                    continue;
                }
                if (a - b).abs() > f64::EPSILON {
                    return false;
                }
            }
        }
        true
    }
}