use rand::Rng;

#[cfg(test)]
mod tests {
    use rand::Rng;
    use rustbrain::math::Vector;
    use rustbrain::linear_regression::LinearRegression;
    use approx::assert_relative_eq;

    #[test]
    fn test_new() {
        let model = LinearRegression::new();
        assert_eq!(model.weights.data.len(), 0);
    }

    #[test]
    fn test_fit_simple_case() {
        // Simple test case: y = 2x + 1
        let mut model = LinearRegression::new();
        
        // Create training data
        let x_data = vec![
            Vector::new(vec![1.0]),
            Vector::new(vec![2.0]),
            Vector::new(vec![3.0]),
            Vector::new(vec![4.0]),
        ];
        
        
        // y = 2x + 1
        let targets = Vector::new(vec![3.0, 5.0, 7.0, 9.0]);
        
        model.fit(&x_data, &targets);
        
        // Check if weights are close to [1, 2] (bias=1, slope=2)
        assert_eq!(model.weights.data.len(), 2);
        assert_relative_eq!(model.weights.data[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(model.weights.data[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_fit_multivariate() {
        // Multivariate test case: y = 1 + 2*x1 + 3*x2
        let mut model = LinearRegression::new();
        
        // Create training data with two features
        let x_data = vec![
            Vector::new(vec![1.0, 1.0]),
            Vector::new(vec![2.0, 1.0]),
            Vector::new(vec![3.0, 2.0]),
            Vector::new(vec![4.0, 2.0]),
            Vector::new(vec![5.0, 3.0]),
        ];
        
        // y = 1 + 2*x1 + 3*x2
        let targets = Vector::new(vec![6.0, 8.0, 13.0, 15.0, 20.0]);
        
        model.fit(&x_data, &targets);
        
        // Check if weights are close to [1, 2, 3] (bias=1, coef_x1=2, coef_x2=3)
        assert_eq!(model.weights.data.len(), 3);
        assert_relative_eq!(model.weights.data[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(model.weights.data[1], 2.0, epsilon = 1e-5);
        assert_relative_eq!(model.weights.data[2], 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_predict() {
        let mut model = LinearRegression::new();
        model.weights = Vector::new(vec![1.0, 2.0]); // bias=1, coefficient=2
        
        // Predict y for x = 3.0
        let input = Vector::new(vec![3.0]);
        let prediction = model.predict(&input);
        
        // Expected: y = 1 + 2*3 = 7
        assert_relative_eq!(prediction, 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_predict_multivariate() {
        let mut model = LinearRegression::new();
        model.weights = Vector::new(vec![1.0, 2.0, 3.0]); // bias=1, coef1=2, coef2=3
        
        // Predict y for x = [4.0, 5.0]
        let input = Vector::new(vec![4.0, 5.0]);
        let prediction = model.predict(&input);
        
        // Expected: y = 1 + 2*4 + 3*5 = 1 + 8 + 15 = 24
        assert_relative_eq!(prediction, 24.0, epsilon = 1e-5);
    }

    #[test]
    #[should_panic(expected = "Matrix inversion failed! Check for singularity.")]
    fn test_fit_singular_matrix() {
        let mut model = LinearRegression::new();
        
        // Creating linearly dependent data will lead to singular matrix
        let x_data = vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 4.0]),  // Linearly dependent with the first row
            Vector::new(vec![3.0, 6.0]),  // Linearly dependent with the first row
        ];
        
        let targets = Vector::new(vec![1.0, 2.0, 3.0]);
        
        // This should panic due to singular matrix
        model.fit(&x_data, &targets);
    }


    #[test]
    fn test_linear_regression_complex() {
        // Simulated dataset: y = 3x1 - 2x2 + 4x3 + 1x4 + 5 + noise
        let inputs = vec![
        Vector::new(vec![1.0, 2.0, 3.0, 4.0]), // 3 - 2 + 12 + 3 + 5 = 21
        Vector::new(vec![2.0, 3.0, 1.0, 5.0]), // 0 + 16 + 2 + 5 = 23
        Vector::new(vec![3.0, 1.0, 4.0, 2.0]),
        Vector::new(vec![4.0, 0.0, 2.0, 3.0]),
        Vector::new(vec![5.0, 2.0, 3.0, 1.0]),
        Vector::new(vec![6.0, 3.0, 4.0, 2.0]),
        Vector::new(vec![7.0, 1.0, 2.0, 3.0]),
        Vector::new(vec![8.0, 4.0, 3.0, 5.0]),
        Vector::new(vec![9.0, 2.0, 1.0, 4.0]),
        Vector::new(vec![10.0, 0.0, 5.0, 2.0]),
        Vector::new(vec![11.0, 3.0, 2.0, 1.0]),
        Vector::new(vec![12.0, 5.0, 4.0, 3.0]),
        Vector::new(vec![13.0, 1.0, 3.0, 2.0]),
        Vector::new(vec![14.0, 2.0, 0.0, 4.0]),
        Vector::new(vec![15.0, 4.0, 2.0, 5.0]),
        Vector::new(vec![16.0, 3.0, 1.0, 6.0]),
        Vector::new(vec![17.0, 2.0, 5.0, 3.0]),
        Vector::new(vec![18.0, 0.0, 4.0, 1.0]),
        Vector::new(vec![19.0, 3.0, 3.0, 2.0]),
        Vector::new(vec![20.0, 1.0, 2.0, 4.0]),
        Vector::new(vec![21.0, 5.0, 1.0, 3.0]),
        Vector::new(vec![22.0, 3.0, 4.0, 2.0]),
        Vector::new(vec![23.0, 2.0, 3.0, 1.0]),
        Vector::new(vec![24.0, 4.0, 2.0, 5.0]),
        Vector::new(vec![25.0, 1.0, 5.0, 3.0]),
    ];
    let mut rng = rand::rng();
    let targets = Vector::new(
        inputs.iter()
            .map(|x| 3.0 * x[0] - 2.0 * x[1] + 4.0 * x[2] + 1.0 * x[3] + 5.0 + rng.random_range(-0.5..0.5))
            .collect()
    );
        let mut model = LinearRegression::new();
        model.fit(&inputs, &targets);

        // Predictions
        let test_input = Vector::new(vec![6.0, 1.0, 2.0, 3.0]); // Expecting y ≈ 3(6) - 2(1) + 4(2) + 1(3) + 5 = 32
        let prediction = model.predict(&test_input);
        println!("prediction: {}", prediction);
        // // Check prediction within tolerance
        assert!(
            (prediction - 32.0).abs() < 0.5,
            "Expected ~32.0, got {}",
            prediction
        );

        // // Ensure weights are close to expected [bias ≈ 5, w1 ≈ 3, w2 ≈ -2, w3 ≈ 4, w4 ≈ 1]
        // let expected_weights = Vector::new(vec![5.0, 3.0, -2.0, 4.0, 1.0]);
        // for (w, ew) in model.weights.data.iter().zip(expected_weights.data.iter()) {
        //     assert!(
        //         (w - ew).abs() < 0.5,
        //         "Expected weight {}, got {}",
        //         ew, w
        //     );
        // }
    }

}
