use rand::Rng; // Import random number generator
use std::ops::IndexMut;
use std::ops::Index;
use super::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    pub fn random(size: usize) -> Self {
        let mut rng = rand::rng();
        let data: Vec<f64> = (0..size).map(|_| rng.random_range(-1.0..=1.0)).collect();
        
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        let len = self.data.len();
        let mut i = 0;

        // Process 4 elements at a time (loop unrolling)
        while i + 3 < len {
            sum += self.data[i] * other.data[i]
                + self.data[i + 1] * other.data[i + 1]
                + self.data[i + 2] * other.data[i + 2]
                + self.data[i + 3] * other.data[i + 3];
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            sum += self.data[i] * other.data[i];
            i += 1;
        }

        sum
    }

    // Scalar multiplication and addition: y = αx + y
    pub fn axpy(&mut self, alpha: f64, x: &Self) {
        for (yi, xi) in self.data.iter_mut().zip(&x.data) {
            *yi += alpha * xi;
        }
    }

    // Compute Euclidean norm
    pub fn norm(&self) -> f64 {
        self.data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // Element-wise addition with scalar multiplier: v = v + α * x, In-place addition
    pub fn add_assign(&mut self, other: &Self, alpha: f64) {
        for (vi, oi) in self.data.iter_mut().zip(&other.data) {
            *vi += alpha * oi;
        }
    }

     // Scales each element of the vector by the given scalar.
     pub fn scale(&self, scalar: f64) -> Self {
        let data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Self { data }
    }

     // In-place scalar multiplication
     pub fn scale_assign(&mut self, factor: f64) {
        for a in &mut self.data {
            *a *= factor;
        }
    }

    // Adds another vector to the current vector element-wise.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len(), "Vectors must have the same length!");
        let data: Vec<f64> = self.data.iter()
                                      .zip(&other.data)
                                      .map(|(a, b)| a + b)
                                      .collect();
        Self { data }
    }

    /// Computes the outer product of two vectors, returning a matrix.
    pub fn outer_product(&self, other: &Vector) -> Matrix {
        let rows = self.len();
        let cols = other.len();
        let mut result = Matrix::zeros(rows, cols);
        
        for i in 0..rows {
            for j in 0..cols {
                result[(i, j)] = self[i] * other[j];
            }
        }
        
        result
    }

    /// Swaps two elements in the vector
    pub fn swap(&mut self, i: usize, j: usize) {
        assert!(i < self.data.len() && j < self.data.len(), "Index out of bounds");
        self.data.swap(i, j);
    }
    
    pub fn iter(&self) -> std::slice::Iter<f64> {
        self.data.iter()
    }

    /// Returns an iterator over mutable references to the elements
    pub fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.data.iter_mut()
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<'a> IntoIterator for &'a Vector {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl IntoIterator for Vector {
    type Item = f64;
    type IntoIter = std::vec::IntoIter<f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a mut Vector {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// impl From<Vector> for Vec<f64> {
//     fn from(vec: Vector) -> Self {
//         vec.data
//     }
// }


