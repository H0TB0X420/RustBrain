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

    #[test]
    fn test_perceptron_prediction() {
        let p = Perceptron::new(3);
        let input = Vector::new(vec![1.0, -2.0, 0.5]);
        let prediction = p.predict(&input);
        let weights: Vec<Vec<f64>> = p.weights.iter().map(|v| p.weights.data.clone()).collect();
        export_verifier_output!(
            predictions = vec![prediction.into()],
            weights = weights,
            biases = vec![],
            file = "test_perceptron_prediction.json"
        );
        assert!(prediction == 0 || prediction == 1);
    }

    #[test]
    fn test_perceptron_training() {
        let mut p = Perceptron::new(3);
        let inputs = vec![Vector::new(vec![1.0, 1.0, 1.0]), Vector::new(vec![0.0, 0.0, 0.0])];
        let outputs = vec![1, 0];
        p.train(&inputs, &outputs, 0.1, 10);

        let new_input = Vector::new(vec![1.0, 1.0, 1.0]);
        let prediction = p.predict(&new_input);
        let weights: Vec<Vec<f64>> = p.weights.iter().map(|v| p.weights.data.clone()).collect();
        export_verifier_output!(
            predictions = vec![prediction.into()],
            weights = weights,
            biases = vec![],
            file = "test_perceptron_training.json"
        );
        assert_eq!(prediction, 1);
    }
}
