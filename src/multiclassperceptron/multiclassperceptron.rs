use crate::math::Vector;
use crate::perceptron::Perceptron; 

pub struct MultiClassPerceptron {
    pub classifiers: Vec<Perceptron>, // One perceptron per class
}

impl MultiClassPerceptron {
    pub fn new(input_size: usize, num_classes: usize) -> Self {
        let classifiers = (0..num_classes)
            .map(|_| Perceptron::new(input_size))
            .collect();
        Self { classifiers }
    }

    /// Predicts the class with the highest activation.
    pub fn predict(&self, input: &Vector) -> i32 {
        self.classifiers
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.predict(input).partial_cmp(&b.predict(input)).unwrap())
            .map(|(index, _)| index as i32)
            .unwrap()
    }

    /// Train using batch updates for multi-class classification.
    pub fn train_batch(
        &mut self,
        inputs: &Vec<Vector>,
        targets: &[i32], // Class labels
        learning_rate: f64,
        batch_size: usize,
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            for (class_index, perceptron) in self.classifiers.iter_mut().enumerate() {
                // Convert multi-class labels into binary labels for one-vs-all training
                let binary_targets: Vec<i32> = targets.iter()
                    .map(|&t| if t == class_index as i32 { 1 } else { -1 }) // Use {-1, 1} for better learning
                    .collect();
                perceptron.train_batch(inputs, &binary_targets, learning_rate, 100, batch_size);
            }
            println!("Epoch {} completed", epoch + 1);
        }
    }
}
