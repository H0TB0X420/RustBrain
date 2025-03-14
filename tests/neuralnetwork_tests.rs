#[cfg(test)]
mod tests {
    use rustbrain::math::Vector;
    use rustbrain::neuralnetwork::NeuralNetwork;
    
    #[test]
    fn test_neural_network_xor() {
        // XOR dataset: inputs and expected outputs.
        let inputs = vec![
            Vector::new(vec![0.0, 0.0]),
            Vector::new(vec![0.0, 1.0]),
            Vector::new(vec![1.0, 0.0]),
            Vector::new(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Vector::new(vec![0.0]),
            Vector::new(vec![1.0]),
            Vector::new(vec![1.0]),
            Vector::new(vec![0.0]),
        ];
        
        // Create a NeuralNetwork with 2 inputs, one hidden layer with 2 neurons, and 1 output.
        let mut nn = NeuralNetwork::new(&[2, 2, 1]);
        
        // Train the network.
        // Here we use a learning rate of 0.5 and 10,000 epochs.
        nn.train(&inputs, &targets, 0.5, 10_000);
        
        // After training, test predictions on the XOR inputs.
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = nn.predict(input);
            println!("Input: {:?} -> Output: {:?}", input.data, output.data);
            // We expect the output to be close to the target (within a tolerance)
            assert!(
                (output.data[0] - target.data[0]).abs() < 0.2,
                "For input {:?}, expected output approx {:?} but got {:?}",
                input.data,
                target.data,
                output.data
            );
        }
    }
}
