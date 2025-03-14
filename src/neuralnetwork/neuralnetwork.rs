use crate::utils::layer::Layer;
use crate::utils::activation::{sigmoid, sigmoid_derivative};
use crate::math::Vector;
/// A multi-layer perceptron neural network.
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

impl NeuralNetwork {
    /// Create a new NeuralNetwork.
    /// The `layer_sizes` slice specifies the number of neurons in each layer,
    /// including the input layer and output layer.
    ///
    /// For example, [2, 3, 1] creates a network with:
    /// - 2 inputs,
    /// - 1 hidden layer with 3 neurons,
    /// - 1 output neuron.
    pub fn new(layer_sizes: &[usize]) -> Self {
        assert!(layer_sizes.len() >= 2, "Network must have at least input and output layers");
        let mut layers = Vec::new();
        for i in 0..(layer_sizes.len() - 1) {
            let input_dim = layer_sizes[i];
            let output_dim = layer_sizes[i + 1];
            layers.push(Layer::new(input_dim, output_dim));
        }
        Self { layers }
    }

    /// Perform a forward pass through the network.
    /// Returns a vector of activations for each layer.
    /// The last element in the returned vector is the final output.
    pub fn forward(&self, input: &Vector) -> Vec<Vector> {
        let mut activations = Vec::new();
        let mut current_input = input.clone();
        for layer in &self.layers {
            // Extend input with bias
            let extended = Layer::extend_with_bias(&current_input);
            // Compute layer output: for each neuron, compute dot(weight, extended_input)
            let mut layer_output = Vector::zeros(layer.weights.row_count());
            for i in 0..layer.weights.row_count() {
                let mut sum = 0.0;
                // For each weight (including bias weight)
                for j in 0..layer.weights.cols {
                    sum += layer.weights[(i, j)] * extended.data[j];
                }
                // Apply activation function
                layer_output.data[i] = sigmoid(sum);
            }
            activations.push(layer_output.clone());
            current_input = layer_output;
        }
        activations
    }

    /// Train the neural network using backpropagation.
    ///
    /// - `inputs`: A vector of input vectors.
    /// - `targets`: A vector of target output vectors (same ordering as inputs).
    /// - `learning_rate`: Learning rate for weight updates.
    /// - `epochs`: Number of training epochs.
    pub fn train(
        &mut self,
        inputs: &Vec<Vector>,
        targets: &Vec<Vector>,
        learning_rate: f64,
        epochs: usize,
    ) {
        assert_eq!(inputs.len(), targets.len(), "Number of inputs and targets must match");

        for epoch in 0..epochs {
            let mut total_error = 0.0;
            // Iterate over each training sample.
            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass: compute activations for each layer.
                let activations = self.forward(input);
                let output = activations.last().unwrap();
                // Compute error (target - output) for output layer.
                let mut output_error = Vector::zeros(output.data.len());
                for i in 0..output.data.len() {
                    output_error.data[i] = target.data[i] - output.data[i];
                    total_error += output_error.data[i].powi(2);
                }

                // Backpropagation: compute deltas for each layer.
                // We'll store deltas in a vector corresponding to each layer.
                let mut deltas: Vec<Vector> = Vec::with_capacity(self.layers.len());
                // Compute delta for output layer.
                let mut delta = Vector::zeros(output.data.len());
                for i in 0..output.data.len() {
                    // Using the output value to compute derivative.
                    delta.data[i] = output_error.data[i] * sigmoid_derivative(output.data[i]);
                }
                deltas.push(delta);

                // Compute deltas for hidden layers (backwards).
                // We iterate from second-last layer down to first layer.
                for l in (0..self.layers.len()-1).rev() {
                    let current_activation = &activations[l];
                    let next_layer = &self.layers[l+1];
                    let next_delta = &deltas[0]; // most recent delta (for layer l+1)
                    let mut delta_hidden = Vector::zeros(current_activation.data.len());
                    // For each neuron in the current layer:
                    for i in 0..current_activation.data.len() {
                        let mut sum = 0.0;
                        // Sum over neurons in next layer.
                        // Note: weight index 0 in the next layer corresponds to bias, so skip that.
                        for k in 0..next_layer.weights.row_count() {
                            // i+1 because next layer's weight column 0 is for bias.
                            sum += next_layer.weights[(k, i + 1)] * next_delta.data[k];
                        }
                        delta_hidden.data[i] = sum * sigmoid_derivative(current_activation.data[i]);
                    }
                    deltas.insert(0, delta_hidden); // Prepend to maintain correct order.
                }

                // Update weights for each layer.
                // The input for layer 0 is the training input, extended with bias.
                let mut layer_input = Layer::extend_with_bias(input);
                for (l, layer) in self.layers.iter_mut().enumerate() {
                    for i in 0..layer.weights.row_count() {
                        for j in 0..layer.weights.cols {
                            // Update rule: w_ij += learning_rate * delta_i * input_j
                            layer.weights[(i, j)] += learning_rate * deltas[l].data[i] * layer_input.data[j];
                        }
                    }
                    // For next layer, the input is the activation from the current layer, extended with bias.
                    layer_input = Layer::extend_with_bias(&activations[l]);
                }
            }
            println!("Epoch {}: Total Error = {}", epoch + 1, total_error);
        }
    }

    /// Perform a prediction for a given input.
    /// Returns the output of the network.
    pub fn predict(&self, input: &Vector) -> Vector {
        let activations = self.forward(input);
        activations.last().unwrap().clone()
    }
}