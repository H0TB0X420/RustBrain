#[cfg(test)]
mod tests {
    use std::vec;
    use approx::assert_relative_eq;
    use rustbrain::math::Matrix;
    use rustbrain::math::Vector;
    #[test]
    fn test_matrix_vector_multiplication() {
        let a = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ]);
        let x = Vector::new(vec![1.0, 1.0, 1.0]);
        let result = a.gemv(&x);

        assert_eq!(result, Vector::new(vec![6.0, 15.0]));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ]);
        let b = Matrix::new(vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0]
        ]);
        let result = a.gemm(&b);

        assert_eq!(result, Matrix::new(vec![
            vec![58.0, 64.0],
            vec![139.0, 154.0]
        ]));
    }

    #[test]
    fn test_add_assign() {
        let mut a = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ]);
        let b = Matrix::new(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0]
        ]);
        a.add_assign(&b);
        assert_eq!(a, Matrix::new(vec![vec![6.0, 8.0], vec![10.0, 12.0]]));
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed: Incompatible matrix dimensions for multiplication\n  left: 2\n right: 3")]
    fn test_dimension_mismatch() {
        let a = Matrix::new(vec![vec![1.0, 2.0]]);
        let b = Matrix::new(vec![vec![3.0], vec![4.0], vec![5.0]]);
        a.gemm(&b);
    }

    #[test]
    fn inverse_matrix(){
        let a = Matrix::new(vec![vec![5.0, 6.0, 6.0, 8.0],
                                                vec![2.0, 2.0, 2.0, 8.0], 
                                                vec![6.0, 6.0, 2.0, 8.0],
                                                vec![2.0, 3.0, 6.0, 7.0]]);
        println!("{}", a.determinant());
        let b = a.inverse();
        println!("{}", a);
        println!("{}", b);
        let mut i = a.gemm(&b);
        let mut j = b.gemm(&a);
        
        for a in 0..i.row_count() {
            for b in 0..i.col_count() {
                i[a][b] = i[a][b].round();
            }
        }
        for a in 0..j.row_count() {
            for b in 0..j.col_count() {
                j[a][b] = j[a][b].round();
            }
        }
        println!("{}", i);
        println!("{}", j);
        assert_eq!(i, Matrix::identity(4));
        assert_eq!(j, Matrix::identity(4));
    }

    #[test]
    fn test_gaussian_elimination() {
        let mut matrix = Matrix::new(vec![
            vec![2.0, 3.0, -1.0, 1.0],
            vec![4.0, 7.0, -3.0, 2.0],
            vec![6.0, 18.0, -5.0, 3.0],
        ]);
        println!("Test 1:");
        let solution = matrix.gaussian_elimination().expect("Gaussian elimination failed");
        for s in solution.iter(){
            println!("{}", s);
        }
        let expected_solution = vec![0.5, 0.0, 0.0];

        for (sol, expected) in solution.iter().zip(expected_solution.iter()) {
            assert!((sol - expected).abs() < 1e-6, "Solution is incorrect!");
        }

        println!("Test 2:");
        let mut matrix = Matrix::new(vec![
            vec![1.0, 9.0, -5.0, -32.0],
            vec![-3.0, -5.0, -5.0, -10.0],
            vec![-2.0, -7.0, 1.0, 13.0],
        ]);

        let solution = matrix.gaussian_elimination().expect("Gaussian elimination failed");
        for s in solution.iter(){
            println!("{}", s);
        }
        let expected_solution = vec![5.0, -3.0, 2.0];

        for (sol, expected) in solution.iter().zip(expected_solution.iter()) {
            assert!((sol - expected).abs() < 1e-6, "Solution is incorrect!");
        }
    }

    #[test]
    fn test_gram_schmidt() {
        // Define a 3x3 matrix A.
        let a = Matrix::new(vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ]);
        
        // Perform the Gram-Schmidt process to get Q and R such that A = Q * R.
        let (q, r) = a.gram_schmidt();
        
        // Check that each column of Q is normalized (has unit length).
        for j in 0..q.cols {
            let col = q.get_column(j);
            let norm = col.norm();
            assert!((norm - 1.0).abs() < 1e-6, "Column {} of Q is not normalized (norm = {})", j, norm);
        }
        
        // Check orthogonality between different columns of Q.
        for j in 0..q.cols {
            for k in (j + 1)..q.cols {
                let col_j = q.get_column(j);
                let col_k = q.get_column(k);
                let dot = col_j.dot(&col_k);
                assert!(dot.abs() < 1e-6, "Columns {} and {} of Q are not orthogonal (dot = {})", j, k, dot);
            }
        }
        
        // Check that Q * R reconstructs the original matrix A.
        let qr = q.gemm(&r);
        for i in 0..a.row_count() {
            for j in 0..a.cols {
                let diff = (a[(i, j)] - qr[(i, j)]).abs();
                assert!(diff < 1e-6, "Mismatch at ({}, {}): A = {}, Q*R = {}", i, j, a[(i, j)], qr[(i, j)]);
            }
        }
    }

    #[test]
    fn test_transpose() {
        // Test case 1: Square matrix
        let mut matrix = Matrix::zeros(3, 3);
        matrix[0][0] = 1.0;
        matrix[0][1] = 2.0;
        matrix[0][2] = 3.0;
        matrix[1][0] = 4.0;
        matrix[1][1] = 5.0;
        matrix[1][2] = 6.0;
        matrix[2][0] = 7.0;
        matrix[2][1] = 8.0;
        matrix[2][2] = 9.0;

        let transposed = matrix.transpose();

        assert_eq!(transposed[0][0], 1.0);
        assert_eq!(transposed[0][1], 4.0);
        assert_eq!(transposed[0][2], 7.0);
        assert_eq!(transposed[1][0], 2.0);
        assert_eq!(transposed[1][1], 5.0);
        assert_eq!(transposed[1][2], 8.0);
        assert_eq!(transposed[2][0], 3.0);
        assert_eq!(transposed[2][1], 6.0);
        assert_eq!(transposed[2][2], 9.0);

        // Test case 2: Non-square matrix (2x3)
        let mut rect_matrix = Matrix::zeros(2, 3);
        rect_matrix[0][0] = 1.0;
        rect_matrix[0][1] = 2.0;
        rect_matrix[0][2] = 3.0;
        rect_matrix[1][0] = 4.0;
        rect_matrix[1][1] = 5.0;
        rect_matrix[1][2] = 6.0;

        let transposed_rect = rect_matrix.transpose();

        // Check dimensions
        assert_eq!(transposed_rect.row_count(), 3);
        assert_eq!(transposed_rect.col_count(), 2);

        // Check values
        assert_eq!(transposed_rect[0][0], 1.0);
        assert_eq!(transposed_rect[0][1], 4.0);
        assert_eq!(transposed_rect[1][0], 2.0);
        assert_eq!(transposed_rect[1][1], 5.0);
        assert_eq!(transposed_rect[2][0], 3.0);
        assert_eq!(transposed_rect[2][1], 6.0);

        // Test case 3: Single row matrix (1x3)
        let mut row_matrix = Matrix::zeros(1, 3);
        row_matrix[0][0] = 1.0;
        row_matrix[0][1] = 2.0;
        row_matrix[0][2] = 3.0;

        let transposed_row = row_matrix.transpose();

        // Check dimensions
        assert_eq!(transposed_row.row_count(), 3);
        assert_eq!(transposed_row.col_count(), 1);

        // Check values
        assert_eq!(transposed_row[0][0], 1.0);
        assert_eq!(transposed_row[1][0], 2.0);
        assert_eq!(transposed_row[2][0], 3.0);

        // Test case 4: Single column matrix (3x1)
        let mut col_matrix = Matrix::zeros(3, 1);
        col_matrix[0][0] = 1.0;
        col_matrix[1][0] = 2.0;
        col_matrix[2][0] = 3.0;

        let transposed_col = col_matrix.transpose();

        // Check dimensions
        assert_eq!(transposed_col.row_count(), 1);
        assert_eq!(transposed_col.col_count(), 3);

        // Check values
        assert_eq!(transposed_col[0][0], 1.0);
        assert_eq!(transposed_col[0][1], 2.0);
        assert_eq!(transposed_col[0][2], 3.0);

        // Test case 5: Empty matrix
        let empty_matrix = Matrix::zeros(0, 0);
        let transposed_empty = empty_matrix.transpose();
        
        assert_eq!(transposed_empty.row_count(), 0);
        assert_eq!(transposed_empty.col_count(), 0);
    }

    #[test]
    fn test_double_transpose() {
        // Transposing twice should give back the original matrix
        let mut matrix = Matrix::zeros(2, 3);
        matrix[0][0] = 1.0;
        matrix[0][1] = 2.0;
        matrix[0][2] = 3.0;
        matrix[1][0] = 4.0;
        matrix[1][1] = 5.0;
        matrix[1][2] = 6.0;

        let double_transposed = matrix.transpose().transpose();

        // Check dimensions
        assert_eq!(double_transposed.row_count(), matrix.row_count());
        assert_eq!(double_transposed.col_count(), matrix.col_count());

        // Check values
        for i in 0..matrix.row_count() {
            for j in 0..matrix.col_count() {
                assert_eq!(double_transposed[i][j], matrix[i][j]);
            }
        }
    }

    #[test]
    fn test_inverse_upper_triangular() {
        // Test with upper triangular matrix
        let a = Matrix::new(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.0, 1.0, 2.0, 3.0], 
            vec![0.0, 0.0, 1.0, 2.0],
            vec![0.0, 0.0, 0.0, 1.0]
        ]);
        
        let b = a.inverse();
        
        // A * A^-1 should equal identity
        let i = a.gemm(&b);
        let j = b.gemm(&a);
        
        assert_eq!(i, Matrix::identity(4));
        assert_eq!(j, Matrix::identity(4));
    }
    
    #[test]
    fn test_inverse_diagonal() {
        // Test with diagonal matrix
        let a = Matrix::new(vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 4.0, 0.0],
            vec![0.0, 0.0, 5.0]
        ]);
        
        let expected_inverse = Matrix::new(vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.25, 0.0],
            vec![0.0, 0.0, 0.2]
        ]);
        
        let b = a.inverse();
        
        // Check each element of inverse
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(b[i][j], expected_inverse[i][j], epsilon = 1e-10);
            }
        }
        
        // Check that A * A^-1 = I
        let identity = a.gemm(&b);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(identity[i][j], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(identity[i][j], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
    
    #[test]
    fn test_inverse_general() {
        // Test with a general matrix
        let a = Matrix::new(vec![
            vec![4.0, 3.0],
            vec![3.0, 2.0]
        ]);
        
        // Expected inverse calculated manually: det = 4*2 - 3*3 = 8 - 9 = -1
        // inv = [2 -3; -3 4] / -1 = [-2 3; 3 -4]
        let expected_inverse = Matrix::new(vec![
            vec![-2.0, 3.0],
            vec![3.0, -4.0]
        ]);
        
        let b = a.inverse();
        
        // Check each element of inverse
        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(b[i][j], expected_inverse[i][j], epsilon = 1e-10);
            }
        }
        
        // Check A * A^-1 = I
        let i = a.gemm(&b);
        assert_eq!(i, Matrix::identity(2));
    }
    
    #[test]
    fn test_inverse_identity() {
        // The inverse of an identity matrix is the identity matrix
        let a = Matrix::identity(5);
        let b = a.inverse();
        
        assert_eq!(b, Matrix::identity(5));
    }
    
    #[test]
    #[should_panic(expected="Matrix inversion failed! Check for singularity.")]
    fn test_singular_matrix() {
        // Test with a singular matrix (determinant = 0)
        let singular = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0]  // Linear dependent rows
        ]);
        
        let _result = singular.inverse();
    }
    
    #[test]
    fn test_inverse_3x3() {
        // Test with 3x3 matrix
        let a = Matrix::new(vec![
            vec![-1.0, 2.0, -1.0],
            vec![2.0, -1.0, 0.0],
            vec![3.0, 0.0, -2.0]
        ]);
        
        let b = a.inverse();
        println!("{}", b);
        // Check A * A^-1 = I
        let i = a.gemm(&b);
        println!("{}", i);
        let j = b.gemm(&a);
        println!("{}", j);
        // Use approximate equality for floating point
        assert_eq!(i, Matrix::identity(3));
        assert_eq!(j, Matrix::identity(3));
    }
    
    #[test]
    fn test_inverse_commutative_property_fails() {
        // Demonstrate that matrix multiplication is not commutative
        let a = Matrix::new(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0]
        ]);
        
        let b = Matrix::new(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0]
        ]);
        
        let ab = a.gemm(&b);
        let ba = b.gemm(&a);
        
        assert_ne!(ab, ba);
    }

}
    