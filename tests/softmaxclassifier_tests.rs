#[macro_use]
mod common;

#[cfg(test)]
mod tests {
    use rustbrain::math::{Vector, Matrix};
    use rustbrain::perceptron::SoftmaxClassifier;
    
    #[test]
    fn test_softmax_function() {
        let logits = Vector::new(vec![2.0, 1.0, 0.1]);
        let softmax_output = SoftmaxClassifier::softmax(&logits);
        
        let sum: f64 = softmax_output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax probabilities should sum to 1.");
    }

    #[test]
    fn test_softmax_prediction() {
        let mut classifier = SoftmaxClassifier::new(2, 3);
        
        // Manually set weights to force predictable outputs
        classifier.weights = Matrix::new(vec![
            vec![2.0, 1.0, -1.0],  // Class 0
            vec![-1.0, 2.0, 1.0],  // Class 1
            vec![0.5, -1.0, 2.0],  // Class 2
        ]);
        
        let input = Vector::new(vec![1.0, 2.0]); // Bias is auto-added
        let predicted_class = classifier.predict(&input);

        export_verifier_output!(
            predictions = vec![predicted_class as f64],
            weights = classifier.weights.rows.iter().map(|v| v.data.clone()).collect(),
            biases = vec![],
            file = "test_softmax_prediction.json"
        );
        assert!(predicted_class < 3, "Prediction should be a valid class index.");
    }

    #[test]
    fn test_softmax_training() {
        let inputs = vec![
            Vector::new(vec![1.0, 0.0]),  // Class 0
            Vector::new(vec![0.0, 1.0]),  // Class 1
            Vector::new(vec![-1.0, -1.0]), // Class 2
        ];
        let targets = vec![0, 1, 2]; // Corresponding class labels
        
        let mut classifier = SoftmaxClassifier::new(2, 3);
        
        classifier.train_batch(&inputs, &targets, 0.1, 2, 100);

        let mut predictions = Vec::new();
        for (input, &expected_class) in inputs.iter().zip(targets.iter()) {
            let predicted = classifier.predict(input);
            assert_eq!(predicted, expected_class, "Failed on input {:?}", input.data);
            predictions.push(predicted as f64);
        }

        export_verifier_output!(
            predictions = predictions,
            weights = classifier.weights.rows.iter().map(|v| v.data.clone()).collect(),
            biases = vec![],
            file = "test_softmax_prediction.json"
        );
    }
}
