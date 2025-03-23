mod tests {
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

    let mut model = LogisticRegression::new(2);
    model.fit_sgd(&inputs, &targets, 0.1, 1000);

    assert_eq!(model.predict(&Vector::new(vec![0.0, 0.0])), 0);
    assert_eq!(model.predict(&Vector::new(vec![1.0, 1.0])), 1);
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
}
}