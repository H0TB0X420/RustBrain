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
}
    