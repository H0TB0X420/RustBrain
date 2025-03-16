#[cfg(test)]
mod tests {
    use rustbrain::math::Vector;
    use rustbrain::perceptron::MultiClassPerceptron;    
        
        #[test]
        fn test_multiclass_perceptron_training() {
            // Define a simple 3-class linearly separable dataset
            let inputs = vec![
                Vector::new(vec![1.0, 0.0]),  // Class 0
                Vector::new(vec![0.0, 1.0]),  // Class 1
                Vector::new(vec![-1.0, -1.0]),  // Class 2
                Vector::new(vec![2.0, 0.0]),  // Class 0
                Vector::new(vec![0.0, 2.0]),  // Class 1
                Vector::new(vec![-2.0, -2.0]),  // Class 2
                Vector::new(vec![100.0, 0.0]),  // Class 0
                Vector::new(vec![0.0, 100.0]),  // Class 1
                Vector::new(vec![-250.0, -800.0]),  // Class 2
                Vector::new(vec![50.0, 0.0]),  // Class 0
                Vector::new(vec![0.0, 50.0]),  // Class 1
                Vector::new(vec![-50.0, -25.0]),  // Class 2
            ];
            let targets = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]; // Corresponding class labels
    
            // Initialize MultiClassPerceptron with 2 inputs and 3 classes
            let mut perceptron = MultiClassPerceptron::new(2, 3);
            
            // Train the model
            perceptron.train_batch(&inputs, &targets, 0.5, 2, 100);
    
            // Verify predictions are correct
            for (input, &expected_class) in inputs.iter().zip(targets.iter()) {
                let predicted = perceptron.predict(input);
                assert_eq!(predicted, expected_class, "Failed on input {:?}", input.data);
            }
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
            
            // Before training, predictions should be random or incorrect
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
    