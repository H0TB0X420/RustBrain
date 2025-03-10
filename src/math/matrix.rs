use super::Vector;
use std::ops::Index;
use std::ops::IndexMut;
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
        self.rows.swap(row1, row2);
    }

    /// Returns the minor matrix obtained by removing row `row` and column `col`
    fn minor(&self, row: usize, col: usize) -> Matrix {
        let mut minor_data = Vec::with_capacity(self.row_count() - 1);
        
        for i in 0..self.row_count() {
            if i == row { continue; } // Skip the row

            let mut new_row = Vec::with_capacity(self.cols - 1);
            for j in 0..self.cols {
                if j == col { continue; } // Skip the column
                new_row.push(self[(i, j)]);
            }
            minor_data.push(Vector::new(new_row));
        }

        Matrix::from_vector(minor_data)
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