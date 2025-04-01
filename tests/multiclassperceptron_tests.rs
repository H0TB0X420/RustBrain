#[macro_use]
mod common;

#[cfg(test)]
mod tests {
    
    use rustbrain::math::Vector;
    use rustbrain::perceptron::MultiClassPerceptron;    
        
    #[test]
    fn test_multiclass_perceptron_training() {
        let inputs = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![-1.0, -1.0]),
            Vector::new(vec![2.0, 0.0]),
            Vector::new(vec![0.0, 2.0]),
            Vector::new(vec![-2.0, -2.0]),
            Vector::new(vec![100.0, 0.0]),
            Vector::new(vec![0.0, 100.0]),
            Vector::new(vec![-250.0, -800.0]),
            Vector::new(vec![50.0, 0.0]),
            Vector::new(vec![0.0, 50.0]),
            Vector::new(vec![-50.0, -25.0]),
        ];
        let targets = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];

        let mut perceptron = MultiClassPerceptron::new(2, 3);
        perceptron.train_batch(&inputs, &targets, 0.5, 2, 100);

        let mut predictions = Vec::new();
        for (input, &expected_class) in inputs.iter().zip(targets.iter()) {
            let predicted = perceptron.predict(input);
            assert_eq!(predicted, expected_class, "Failed on input {:?}", input.data);
            predictions.push(predicted.into());
        }

        // ✅ Export predictions and weights for external verification
        let weights: Vec<Vec<f64>> = perceptron
                                .classifiers
                                .iter()
                                .map(|clf| clf.weights.data.clone())
                                .collect();

        export_verifier_output!(
            predictions = &predictions,
            weights = weights,
            biases = vec![],
            file = "test_multiclass_perceptron_training.json"
        );
    }

    #[test]
    fn test_multiclass_perceptron_untrained() {
        let inputs = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![-1.0, -1.0]),
        ];
        let targets = vec![0, 1, 2];

        let perceptron = MultiClassPerceptron::new(2, 3);

        let mut incorrect = 0;
        for (input, &expected_class) in inputs.iter().zip(targets.iter()) {
            let predicted = perceptron.predict(input);
            if predicted != expected_class {
                incorrect += 1;
            }
        }
        assert!(incorrect > 0, "An untrained perceptron should not be fully correct.");
    }
}