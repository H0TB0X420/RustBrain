use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Serialize)]
pub struct ModelParameters {
    pub weights: Option<Vec<Vec<f64>>>,
    pub biases: Option<Vec<f64>>,
}

#[derive(Serialize)]
struct RustModelOutput<'a> {
    predictions: &'a [usize],
    parameters: Option<ModelParameters>,
}

/// Call this from test code to write out predictions + params
pub fn dump_to_verifier(
    predictions: &[usize],
    parameters: Option<ModelParameters>,
    filename: &str,  // e.g., "perceptron.json"
) {
    let output = RustModelOutput {
        predictions,
        parameters,
    };

    let path = Path::new("verifier/rust_outputs").join(filename);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");

    let mut file = File::create(&path).expect("Failed to create verifier output file");
    file.write_all(json.as_bytes())
        .expect("Failed to write verifier output");
}
