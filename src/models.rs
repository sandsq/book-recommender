use ndarray::{ArrayBase, Axis, Dim, Ix2, ViewRepr};
use ort::{
    Error,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn get_model_info(model_path: &str) -> ort::Result<()> {
    let session = Session::builder()?.commit_from_file(model_path)?;

    let meta = session.metadata()?;
    match meta.name() {
        Ok(x) => println!("Name: {x}"),
        Err(e) => println!("Error: {}", e),
    }
    match meta.description() {
        Ok(x) => println!("Description: {x}"),
        Err(e) => println!("Error: {}", e),
    }
    match meta.producer() {
        Ok(x) => println!("Produced by {x}"),
        Err(e) => println!("Error: {}", e),
    }

    println!("Inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!("    {i} {}: {}", input.name, input.input_type);
    }
    println!("Outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!("    {i} {}: {}", output.name, output.output_type);
    }

    Ok(())
}

fn prepare_tokenized_inputs(
    tokenizer: &Tokenizer,
    inputs: &[&str],
) -> ort::Result<(Vec<i64>, Vec<i64>, usize)> {
    // Encode our input strings.
    let encodings = tokenizer
        .encode_batch(inputs.to_vec(), true)
        .map_err(|e| Error::new(e.to_string()))?;
    // println!("Encodings: {:?}", encodings);

    let max_length = encodings
        .iter()
        .map(|enc| enc.get_ids().len())
        .max()
        .unwrap_or(0);
    // println!("Maximum token length: {}", max_length);

    // Prepare padded token IDs and attention masks
    let mut padded_ids: Vec<i64> = Vec::new();
    let mut padded_mask: Vec<i64> = Vec::new();

    for encoding in encodings.clone() {
        let mut ids: Vec<i64> = encoding
            .get_ids()
            .to_vec()
            .iter()
            .map(|i| *i as i64)
            .collect();
        let mut mask: Vec<i64> = encoding
            .get_attention_mask()
            .to_vec()
            .iter()
            .map(|i| *i as i64)
            .collect();

        // Pad the token IDs and attention masks to max_length
        while ids.len() < max_length {
            ids.push(0); // Assuming 0 is the padding token ID
        }

        while mask.len() < max_length {
            mask.push(0); // Assuming 0 indicates padding in the attention mask
        }

        padded_ids.extend(ids);
        padded_mask.extend(mask);
    }

    // Debugging output for padded IDs and masks
    // println!("Padded Token IDs: {:?}", padded_ids);
    // println!("Padded Attention Masks: {:?}", padded_mask);

    // println!("Padded token length: {}", max_length);

    // println!(
    //     "Token IDs: {:?} with length {}",
    //     padded_ids,
    //     padded_ids.len()
    // );
    // println!(
    //     "Attention Mask: {:?} with length {}",
    //     padded_mask,
    //     padded_mask.len()
    // );

    Ok((padded_ids, padded_mask, max_length))
}

// fn compute_embeddings(
//     inputs: Vec<&str>,
//     ids: Vec<i64>,
//     mask: Vec<i64>,
//     padded_token_length: usize,
//     model_path: &str,
// ) -> ort::Result<ArrayBase<ViewRepr<&f32>, Dim<[usize; 2]>>, Box<dyn std::error::Error>> {
//     let a_ids = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*ids))?;
//     let a_mask = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*mask))?;

//     let mut session = Session::builder()?
//         .with_optimization_level(GraphOptimizationLevel::Level3)?
//         .with_intra_threads(4)?
//         // .commit_from_url(
//         //     "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/all-MiniLM-L6-v2.onnx",
//         // )?;
//         .commit_from_file(model_path)?;
//     // Run the model.
//     let outputs = session.run(ort::inputs![a_ids, a_mask])?;

//     // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
//     let embeddings = outputs[1]
//         .try_extract_array::<f32>()?
//         .into_dimensionality::<Ix2>()?;

//     Ok(embeddings)
// }

pub fn query_model(
    model_path: &str,
    tokenizer_path: &str,
    inputs: Vec<&str>,
) -> ort::Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing to receive debug messages from `ort`
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load the tokenizer and encode the text.

    let tokenizer =
        // Tokenizer::from_file("/home/sand/coding/Qwen3-Embedding-0.6B-ONNX/tokenizer-minilm.json")
            // .unwrap();
    Tokenizer::from_file(tokenizer_path).unwrap();

    let (ids, mask, padded_token_length) = prepare_tokenized_inputs(&tokenizer, &inputs)?;
    // println!(
    //     "Tokenized inputs: {:?}",
    //     (ids.clone(), mask.clone(), padded_token_length)
    // );

    // let embeddings = compute_embeddings(inputs, ids, mask, padded_token_length, model_path)?;

    let a_ids = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*ids))?;
    let a_mask = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*mask))?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        // .commit_from_url(
        //     "https://cdn.pyke.io/0/pyke:ort-rs/example-models@0.0.0/all-MiniLM-L6-v2.onnx",
        // )?;
        .commit_from_file(model_path)?;
    // Run the model.
    let outputs = session.run(ort::inputs![a_ids, a_mask])?;

    // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
    let embeddings = outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix2>()?;

    println!("{:?}", embeddings);

    let test = embeddings.index_axis(Axis(0), 1).to_slice();
    println!("{:?}", test);

    println!("Similarity for '{}'", inputs[0]);
    let query = embeddings.index_axis(Axis(0), 0);

    for (embeddings, sentence) in embeddings.axis_iter(Axis(0)).zip(inputs.iter()).skip(1) {
        // Calculate cosine similarity against the 'query' sentence.
        let dot_product: f32 = query
            .iter()
            .zip(embeddings.iter())
            .map(|(a, b)| a * b)
            .sum();
        println!("\t'{}': {:.1}%", sentence, dot_product * 100.);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_tokenized_inputs() {
        // Load tokenizer from file
        let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
        let tokenizer = match Tokenizer::from_file(tokenizer_path) {
            Ok(t) => t,
            Err(_) => {
                println!(
                    "Tokenizer file not found at {}. Skipping test.",
                    tokenizer_path
                );
                return;
            }
        };

        let inputs = vec!["Hello world!", "This is a test.", "Rust is great."];

        match prepare_tokenized_inputs(&tokenizer, &inputs) {
            Ok((ids, mask, max_length)) => {
                // Verify that we got non-empty results
                assert!(!ids.is_empty(), "Token IDs should not be empty");
                assert!(!mask.is_empty(), "Attention mask should not be empty");
                assert!(max_length > 0, "Max length should be greater than 0");

                // Verify that ids and mask have the same length
                assert_eq!(
                    ids.len(),
                    mask.len(),
                    "Token IDs and attention mask should have the same length"
                );

                // Verify that the total length matches inputs count * max_length
                assert_eq!(
                    ids.len(),
                    inputs.len() * max_length,
                    "Total tokens should equal number of inputs * max_length"
                );

                // Verify that padding was applied (should have 0s for padding tokens)
                let has_padding = ids.iter().any(|&id| id == 0) || mask.iter().any(|&m| m == 0);
                println!(
                    "Test passed: max_length={}, has_padding={}",
                    max_length, has_padding
                );
            }
            Err(e) => {
                panic!("prepare_tokenized_inputs failed with error: {}", e);
            }
        }
    }
}
