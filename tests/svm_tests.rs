mod tests {
    use rand::Rng;
    use rustbrain::math::Vector;
    use rustbrain::svm::{HardMarginSVM, SoftMarginSVM, KernelSVM};
    use std::f64::consts::PI;
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

    // #[test]
    // fn test_qp_solver_basic() {
    //     let q = Matrix::identity(2);
    //     let p = Vector::new(vec![-1.0, -1.0]);
    //     let a = Matrix::new(vec![vec![1.0, 1.0]]);
    //     let b = Vector::new(vec![0.0]);
    //     let l = Vector::new(vec![0.0, 0.0]);
    //     let u = Vector::new(vec![1.0, 1.0]);
        
    //     let mut qp_solver = QPSolver::new(q, p, a, b, l, u);
    //     let solution = qp_solver.solve_smo(1000, 1e-5);
        
    //     assert!(solution[0] >= 0.0 && solution[0] <= 1.0);
    //     assert!(solution[1] >= 0.0 && solution[1] <= 1.0);
    // }

    

    #[test]
    fn test_kernel_svm_linear() {
        // Create a simple linearly separable dataset
        let inputs = vec![
            Vector::new(vec![1.0, 1.0]),
            Vector::new(vec![2.0, 2.0]),
            Vector::new(vec![1.5, 1.0]),
            Vector::new(vec![-1.0, -1.0]),
            Vector::new(vec![-2.0, -2.0]),
            Vector::new(vec![-1.5, -1.0]),
        ];
        let targets = Vector::new(vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0]);
        
        // For a nearly linear kernel, set gamma very small
        let mut model = KernelSVM::new(10.0, 2, 10, 0.5);
        model.fit_qp(&inputs, &targets);
        
        let mut correct = 0;
        for (input, &target) in inputs.iter().zip(targets.data.iter()) {
            let prediction = model.predict(input);
            if prediction as f64 == target {
                correct += 1;
            }
        }
        assert_eq!(correct, inputs.len(), "Linear Kernel SVM failed to classify all samples correctly.");
    }
    
    ////Test Kernel SVM with an RBF kernel using Random Fourier Features (RFF)
    #[test]
    fn test_kernel_svm_rbf() {
        let mut inputs = Vec::new();
        let mut targets_vec = Vec::new();
        let mut rng = rand::rng();
        
        // Generate 20 samples for class 1: points inside a circle of radius 1.0\n
        for _ in 0..50 {
            let r = rng.random_range(0.0..1.0);
            let theta = rng.random_range(0.0..(2.0 * PI));
            let x = r * theta.cos();
            let y = r * theta.sin();
            inputs.push(Vector::new(vec![x, y]));
            targets_vec.push(1.0);
        }// Generate 20 samples for class -1: points outside a circle of radius 1.5\n
        for _ in 0..50 {
            let r = rng.random_range(2.0..4.0);
            let theta = rng.random_range(0.0..(2.0 * PI));
            let x = r * theta.cos();
            let y = r * theta.sin();
            inputs.push(Vector::new(vec![x, y]));
            targets_vec.push(-1.0);
        }
        let targets = Vector::new(targets_vec);      // Use an RBF kernel via RFF with appropriate gamma for this data
        let mut model = KernelSVM::new(10.0, 2, 1000, 0.5);
        model.fit_qp(&inputs, &targets);
        
        let mut correct = 0;
        for (input, &target) in inputs.iter().zip(targets.data.iter()) {
            let prediction = model.predict(input);
            println!("{} \t {}", prediction, target);
            if prediction as f64 == target {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / inputs.len() as f64;
        assert!(accuracy > 0.7, "RBF Kernel SVM accuracy too low: {} ({} out of {})", accuracy, correct, inputs.len());
    }
    

    #[test]
    fn test_kernel_svm_rbf_rff_non_linearly_separable() {
        let inputs = vec![
            Vector::new(vec![0.0, 0.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 1.0]),
        ];
        let targets = Vector::new(vec![-1.0, 1.0, 1.0, -1.0]);

        let mut model = KernelSVM::new(1.0, 2, 100, 0.5);
        model.fit_qp(&inputs, &targets);

        let test_point = Vector::new(vec![0.5, 0.5]);
        let prediction = model.predict(&test_point);
        assert!(prediction == -1 || prediction == 1); // Should not panic or be undefined
    }
}