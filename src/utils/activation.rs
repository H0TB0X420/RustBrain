// Sigmoid activation and its derivative (using output value)
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}