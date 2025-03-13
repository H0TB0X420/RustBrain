#[cfg(test)]
mod tests {
    use std::vec;

    use rustbrain::math::Matrix;
    use rustbrain::math::Vector;
    #[test]
    fn test_matrix_vector_multiplication() {
        let a = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0]
        ]);
        let x = Vector::new(vec![1.0, 1.0, 1.0]);
        let result = a.gemv(&x).unwrap();

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
        let a = Matrix::new(vec![vec![1.0, 2.0, 3.0, 4.0],
                                                vec![0.0, 1.0, 2.0, 3.0], 
                                                vec![0.0,0.0, 1.0, 2.0],
                                                vec![0.0, 0.0, 0.0, 1.0]]);
        let b = a.inverse().unwrap();
        println!("{}", a);
        println!("{}", b);
        let i = a.gemm(&b);
        let j = b.gemm(&a);
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
}
    