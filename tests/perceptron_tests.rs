#[macro_use]
mod common;


#[cfg(test)]
mod tests {
    use rustbrain::perceptron::Perceptron;
    use rustbrain::math::Vector;
    #[test]
    // #[ignore]
    fn test_perceptron_initialization() {
        let p = Perceptron::new(3);
        assert_eq!(p.weights().len(), 4); // and should account for bias
    }

    // #[test]
    // fn test_perceptron_prediction() {
    //     let p = Perceptron::new(3);
    //     let input = Vector::new(vec![1.0, -2.0, 0.5]);
    //     let predictions: Vec<f64> = input.iter().map(|x| p.predict(x) as f64).collect();
    //     let input_data: Vec<Vec<f64>> = input.iter().map(|v| v.data.clone()).collect();
    //     let weights = vec![p.weights.data.clone()];
    //     // export_verifier_output!(
    //     //     inputs = vec![input.data],
    //     //     predictions = vec![prediction.into()],
    //     //     weights = weights,
    //     //     biases = vec![],
    //     //     file = "test_perceptron_prediction.json"
    //     // );
    //     assert!(prediction == 0 || prediction == 1);
    // }

    #[test]
    fn test_perceptron_training() {
        let mut p = Perceptron::new(2);
    
        let inputs = vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![4.0, 5.0]),
            Vector::new(vec![6.0, 7.0]),

            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![2.0, 1.0]),
            Vector::new(vec![3.0, 2.0]),
            Vector::new(vec![4.0, 3.0]),
            Vector::new(vec![5.0, 4.0]),

            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![3.0, 0.0]),
            Vector::new(vec![0.0, 6.0]),
            Vector::new(vec![1.0, 5.0]),

        ];
        let outputs = vec![1,1,1,1,1,0,0,0,0,0, 1, 0, 0, 1, 1];
    
        p.train(&inputs, &outputs, 0.2, 300);
    
        let predictions: Vec<f64> = inputs.iter().map(|x| p.predict(x) as f64).collect();
        let input_data: Vec<Vec<f64>> = inputs.iter().map(|v| v.data.clone()).collect();
        let weights = vec![p.weights.data.clone()];
    
        export_verifier_output!(
            inputs = input_data,
            predictions = predictions,
            weights = weights,
            biases = vec![],
            file = "test_perceptron_training.json"
        );
        
        // for (&pred, &expected) in predictions.iter().zip(outputs.iter()) {
        //     assert_eq!(pred ,expected as f64);
        // }
    }
}
