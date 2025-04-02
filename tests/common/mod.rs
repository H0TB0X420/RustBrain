use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Used by test cases to export model internals for verification
#[derive(Serialize)]
pub struct ModelParameters {
    pub weights: Vec<Vec<f64>>,  // Always present, even if empty
    pub biases: Vec<f64>,        // Always present, even if empty
}

#[derive(Serialize)]
struct RustModelOutput<'a> {
    inputs: &'a [Vec<f64>],
    predictions: &'a [f64],
    parameters: ModelParameters,
}

/// Call this from test code to write out predictions + params + inputs
pub fn dump_to_verifier(
    inputs: &Vec<Vec<f64>>,
    predictions: &Vec<f64>,
    parameters: ModelParameters,
    filename: &str,  // e.g., "perceptron.json"
) {
    let output = RustModelOutput {
        inputs,
        predictions,
        parameters,
    };

    let path = Path::new("verifier/rust_outputs").join(filename);
    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");

    let mut file = File::create(&path).expect("Failed to create verifier output file");
    file.write_all(json.as_bytes())
        .expect("Failed to write verifier output");
}

#[macro_export]
macro_rules! export_verifier_output {
    (
        inputs = $inputs:expr,
        predictions = $predictions:expr,
        weights = $weights:expr,
        biases = $biases:expr,
        file = $filename:expr
    ) => {
        $crate::common::dump_to_verifier(
            &$inputs,
            &$predictions,
            $crate::common::ModelParameters {
                weights: $weights,
                biases: $biases,
            },
            $filename,
        );
    };
}

#[macro_export]
macro_rules! dump_json_data {
    (
        data = $data:expr,
        file = $filename:expr
    ) => {{
        use std::fs::File;
        use std::io::Write;
        use std::path::Path;
        use serde_json;

        let path = std::path::Path::new("verifier/rust_outputs").join($filename);
        let json = serde_json::to_string_pretty(&$data).expect("Failed to serialize data");

        let mut file = File::create(&path).expect("Failed to create file");
        file.write_all(json.as_bytes()).expect("Failed to write JSON data");
    }};
}
