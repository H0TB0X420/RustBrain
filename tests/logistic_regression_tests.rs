#[macro_use]
mod common;

mod tests {
use std::vec;

use rustbrain::Vector;
use rustbrain::{LogisticRegression, SoftmaxRegression};

    #[test]
    fn test_logistic_regression_binary() {
        let inputs = vec![
            Vector::new(vec![0.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![1.0, 1.0]),
        ];
        let targets = Vector::new(vec![0.0, 0.0, 0.0, 1.0]); // AND gate logic

        let mut model = LogisticRegression::new(2, 0.0, 0.0);
        model.fit_sgd(&inputs, &targets, 0.1, 1000);

        assert_eq!(model.predict(&Vector::new(vec![0.0, 0.0])), 0);
        assert_eq!(model.predict(&Vector::new(vec![1.0, 1.0])), 1);

        let test_inputs = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let predictions: Vec<f64> = test_inputs.iter().map(|x| model.predict(&Vector::new(x.to_vec())) as f64).collect();

        export_verifier_output!(
            inputs = test_inputs,
            predictions = predictions,
            weights = vec![model.weights.data.clone()],
            biases = vec![],
            file = "logreg_binary.json"
        );
    }

    #[test]
    fn test_logistic_regression_with_l1() {
        let inputs = vec![
            Vector::new(vec![0.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![1.0, 1.0]),
        ];
        let targets = Vector::new(vec![0.0, 0.0, 0.0, 1.0]);

        let mut model = LogisticRegression::new(2, 0.001, 0.0); // L1 regularization only
        model.fit_sgd(&inputs, &targets, 0.1, 1000);

        assert_eq!(model.predict(&Vector::new(vec![0.0, 0.0])), 0);
        assert_eq!(model.predict(&Vector::new(vec![1.0, 1.0])), 1);

        let predictions: Vec<f64> = inputs.iter().map(|x| model.predict(x) as f64).collect();

        export_verifier_output!(
            inputs = inputs.iter().map(|v| v.data.clone()).collect(),
            predictions = predictions,
            weights = vec![model.weights.data.clone()],
            biases = vec![],
            file = "logreg_l1.json"
        );

    }

    #[test]
    fn test_logistic_regression_with_l2() {
        let inputs = vec![
            Vector::new(vec![0.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![1.0, 1.0]),
        ];
        let targets = Vector::new(vec![0.0, 0.0, 0.0, 1.0]);

        let mut model = LogisticRegression::new(2, 0.0, 0.001); // L2 regularization only
        model.fit_sgd(&inputs, &targets, 0.1, 1000);

        assert_eq!(model.predict(&Vector::new(vec![0.0, 0.0])), 0);
        assert_eq!(model.predict(&Vector::new(vec![1.0, 1.0])), 1);

        let predictions: Vec<f64> = inputs.iter().map(|x| model.predict(x) as f64).collect();

        export_verifier_output!(
            inputs = inputs.iter().map(|v| v.data.clone()).collect(),
            predictions = predictions,
            weights = vec![model.weights.data.clone()],
            biases = vec![],
            file = "logreg_l2.json"
        );

    }

    #[test]
    fn test_softmax_regression_multiclass() {
        let inputs = vec![
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 1.0]),
            Vector::new(vec![0.5, 0.5]),
        ];
        let targets = vec![0, 1, 2, 1]; // Example class labels

        let mut model = SoftmaxRegression::new(2, 3);
        model.fit_sgd(&inputs, &targets, 0.1, 1000);

        assert_eq!(model.predict(&Vector::new(vec![1.0, 0.0])), 0);
        assert_eq!(model.predict(&Vector::new(vec![0.0, 1.0])), 1);
        assert_eq!(model.predict(&Vector::new(vec![1.0, 1.0])), 2);

        let predictions: Vec<f64> = inputs.iter().map(|x| model.predict(x) as f64).collect();

        let weights: Vec<Vec<f64>> = model
            .weights
            .rows
            .iter()
            .map(|row| row.data.clone())
            .collect();

        export_verifier_output!(
            inputs = inputs.iter().map(|v| v.data.clone()).collect(),
            predictions = predictions,
            weights = weights,
            biases = vec![],
            file = "softmax_multiclass.json"
        );

    }
}