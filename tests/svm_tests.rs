mod tests {
    use rustbrain::math::Vector;
    use rustbrain::svm::{HardMarginSVM, SoftMarginSVM};

    #[test]
    fn test_hard_margin_svm_extended() {
        let inputs = vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![3.0, -1.0]),
            Vector::new(vec![4.0, -3.0]),
            Vector::new(vec![5.0, 5.0]),
            Vector::new(vec![6.0, -2.0]),
        ];
        let targets = Vector::new(vec![1.0, 1.0, -1.0, -1.0, 1.0, -1.0]);
        
        let mut model = HardMarginSVM::new(2, 0.1, 1000);
        model.fit(&inputs, &targets);
        
        assert_eq!(model.predict(&Vector::new(vec![1.5, 2.5])), 1);
        assert_eq!(model.predict(&Vector::new(vec![3.5, -2.0])), -1);
        assert_eq!(model.predict(&Vector::new(vec![5.5, 5.0])), 1);
        assert_eq!(model.predict(&Vector::new(vec![6.0, -1.0])), -1);
    }

    #[test]
    fn test_soft_margin_svm_extended() {
        let inputs = vec![
            Vector::new(vec![1.0, 2.0]),
            Vector::new(vec![2.0, 3.0]),
            Vector::new(vec![3.0, -1.0]),
            Vector::new(vec![4.0, -3.0]),
            Vector::new(vec![5.0, 5.0]),
            Vector::new(vec![6.0, 2.0]),
            Vector::new(vec![4.0, 3.0]),
            Vector::new(vec![5.0, -5.0]),
            Vector::new(vec![6.0, 2.0]),

        ];
        let targets = Vector::new(vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]);
        
        let mut model = SoftMarginSVM::new(2, 0.1, 1000, 1.0);
        model.fit(&inputs, &targets);
        
        assert_eq!(model.predict(&Vector::new(vec![1.5, 2.5])), 1);
        assert_eq!(model.predict(&Vector::new(vec![3.5, -2.0])), -1);
        assert_eq!(model.predict(&Vector::new(vec![5.5, 5.0])), 1);
        assert_eq!(model.predict(&Vector::new(vec![6.0, -1.0])), -1);
    }
}