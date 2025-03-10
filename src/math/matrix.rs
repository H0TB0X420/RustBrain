use super::Vector;
#[derive(Debug, Clone, PartialEq)]


pub struct Matrix {
    pub data: Vec<Vector>,
    pub cols: usize,
}

impl Matrix {
    pub fn new(input: Vec<Vec<f64>>) -> Self {
        let cols = input.first().map_or(0, |row| row.len());
        let data = input.into_iter().map(Vector::new).collect();
        Self { data, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![Vector::zeros(cols); rows],
            cols,
        }
    }

    pub fn identity(size: usize) -> Self {
        let mut mat = Self::zeros(size, size);
        for i in 0..size {
            mat.data[i].data[i] = 1.0;
        }
        mat
    }

    pub fn row_count(&self) -> usize {
        self.data.len()
    }

    pub fn col_count(&self) -> usize {
        self.cols
    }

    pub fn get_row(&self, index: usize) -> Option<&Vector> {
        self.data.get(index)
    }

    pub fn transpose(&self) -> Self {
        let mut transposed = vec![vec![0.0; self.row_count()]; self.col_count()];
        for i in 0..self.row_count() {
            for j in 0..self.col_count() {
                transposed[j][i] = self.data[i].data[j];
            }
        }
        Self::new(transposed)
    }

      // In-place matrix addition
    pub fn add_assign(&mut self, other: &Self) {
        assert!(self.row_count() == other.row_count() && self.col_count() == other.col_count(), 
                "Dimension mismatch in matrix addition");
        for (row_a, row_b) in self.data.iter_mut().zip(&other.data) {
            row_a.add_assign(row_b, 1.0);
        }
    }

      // Returns a new matrix sum
    pub fn add(&self, other: &Self) -> Self {
        assert!(self.row_count() == other.row_count() && self.col_count() == other.col_count(), 
                "Dimension mismatch in matrix addition");
        Self {
            data: self.data.iter().zip(&other.data).map(|(a, b)| a.add(b)).collect(),
            cols: self.cols,
        }
    }

    // Matrix-vector multiplication: y = Ax + y
    pub fn gemv(&self, x: &Vector) -> Option<Vector> {
        if self.col_count() != x.len() {
            return None; // Dimension mismatch
        }
        let mut result = Vector::zeros(self.row_count());
        for (row, res) in self.data.iter().zip(result.data.iter_mut()) {
            *res = row.dot(x);
        }
        Some(result)
    }

    // General matrix-matrix multiplication: C = AB
    pub fn gemm(&self, other: &Matrix) -> Option<Matrix> {
        if self.col_count() != other.row_count() {
            return None; // Dimension mismatch
        }

        let m = self.row_count();
        let n = other.col_count();
        let mut result = Matrix::zeros(m, n);

        for i in 0..m {
            for j in 0..n {
                let sum = (0..self.col_count()).map(|k| self.data[i].data[k] * other.data[k].data[j]).sum();
                result.data[i].data[j] = sum;
            }
        }
        Some(result)
    }

  
}
