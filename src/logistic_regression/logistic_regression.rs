use crate::math::{Vector, Matrix};
use crate::utils::activation::sigmoid;
use rand::{prelude::SliceRandom, Rng};

pub struct LogisticRegression {
    pub weights: Vector, // Model parameters (including bias)
}

impl LogisticRegression {
    /// Creates an uninitialized Logistic Regression model.
    pub fn new(n_features: usize) -> Self {
        let mut rng = rand::rng();
        Self {
            weights: Vector::new((0..=n_features).map(|_| rng.random_range(-0.01..0.01)).collect()),
        }
    }

    /// Trains the model using Stochastic Gradient Descent (SGD)
    pub fn fit_sgd(&mut self, inputs: &Vec<Vector>, targets: &Vector, learning_rate: f64, epochs: usize) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
        let mut rng = rand::rng();
        let n = inputs.len();
        
        for _ in 0..epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            
            for &i in indices.iter() {
                let mut x = inputs[i].clone();
                x.data.insert(0, 1.0); // Add bias term
                
                let prediction = sigmoid(self.weights.dot(&x));
                let error = targets[i] - prediction;
                let gradient = x.scale(error * learning_rate);
                self.weights = self.weights.add(&gradient);
            }
        }
    }

    /// Predicts the probability of class 1
    pub fn predict_proba(&self, input: &Vector) -> f64 {
        let mut extended_input = input.clone();
        extended_input.data.insert(0, 1.0);
        sigmoid(self.weights.dot(&extended_input))
    }

    /// Predicts the binary class (0 or 1)
    pub fn predict(&self, input: &Vector) -> i32 {
        if self.predict_proba(input) >= 0.5 { 1 } else { 0 }
    }
}

pub struct SoftmaxRegression {
    pub weights: Matrix, // Model parameters for multi-class classification
}

impl SoftmaxRegression {
    /// Creates a Softmax Regression model with given input size and number of classes.
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        let weights = Matrix::random(n_classes, n_features + 1);
        Self { weights }
    }

    /// Softmax function
    fn softmax(logits: &Vector) -> Vector {
        let max_logit = logits.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = logits.data.iter().map(|&z| (z - max_logit).exp()).collect();
        let sum_exp = exp_values.iter().sum::<f64>();
        Vector::new(exp_values.iter().map(|&e| e / sum_exp).collect())
    }

    /// Trains the model using Stochastic Gradient Descent (SGD)
    pub fn fit_sgd(&mut self, inputs: &Vec<Vector>, targets: &Vec<usize>, learning_rate: f64, epochs: usize) {
        assert!(inputs.len() == targets.len(), "Mismatched input and target sizes!");
        let mut rng = rand::rng();
        let n = inputs.len();
        
        for _ in 0..epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            indices.shuffle(&mut rng);
            
            for &i in indices.iter() {
                let mut x = inputs[i].clone();
                x.data.insert(0, 1.0); // Add bias term
                
                let logits = self.weights.gemv(&x);
                let probabilities = Self::softmax(&logits);
                
                let mut error = probabilities;
                error[targets[i]] -= 1.0; // One-hot encoding error adjustment
                
                let mut gradient = error.outer_product(&x);
                gradient.scale(-1.0 * learning_rate);
                self.weights = self.weights.add(&gradient);
            }
        }
    }

    /// Predicts class probabilities
    pub fn predict_proba(&self, input: &Vector) -> Vector {
        let mut extended_input = input.clone();
        extended_input.data.insert(0, 1.0);
        let logits = self.weights.gemv(&extended_input);
        Self::softmax(&logits)
    }

    /// Predicts the most likely class
    pub fn predict(&self, input: &Vector) -> usize {
        self.predict_proba(input)
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    }
}
