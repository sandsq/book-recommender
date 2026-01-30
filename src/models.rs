use ndarray::{ArrayBase, Axis, Dim, Ix2, ViewRepr};
use ort::{
    Error,
    session::{self, Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::sync::Arc;
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

pub fn ready_model(model_path: &str) -> Result<Session, Box<dyn std::error::Error>> {
    // Initialize tracing to receive debug messages from `ort`
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(session)
}

pub fn ready_tokenizer(tokenizer_path: &str) -> Tokenizer {
    Tokenizer::from_file(tokenizer_path).unwrap()
}

pub fn query_model(
    session: &mut Session,
    tokenizer: &Tokenizer,
    inputs: Vec<&str>,
) -> ort::Result<Vec<(String, Vec<f32>)>, Box<dyn std::error::Error>> {
    // Load the tokenizer and encode the text.

    let (ids, mask, padded_token_length) = prepare_tokenized_inputs(&tokenizer, &inputs)?;
    // println!(
    //     "Tokenized inputs: {:?}",
    //     (ids.clone(), mask.clone(), padded_token_length)
    // );

    // let embeddings = compute_embeddings(inputs, ids, mask, padded_token_length, model_path)?;

    let a_ids = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*ids))?;
    let a_mask = TensorRef::from_array_view(([inputs.len(), padded_token_length], &*mask))?;

    let outputs = session.run(ort::inputs![a_ids, a_mask])?;

    // Extract our embeddings tensor and convert it to a strongly-typed 2-dimensional array.
    let embeddings = outputs[1]
        .try_extract_array::<f32>()?
        .into_dimensionality::<Ix2>()?;

    // println!("{:?}", embeddings);

    // let test = embeddings.index_axis(Axis(0), 1).to_slice();
    // println!("{:?}", test);

    // println!("Similarity for '{}'", inputs[0]);
    // let query = embeddings.index_axis(Axis(0), 0);

    // for (embeddings, sentence) in embeddings.axis_iter(Axis(0)).zip(inputs.iter()).skip(1) {
    //     // Calculate cosine similarity against the 'query' sentence.
    //     let dot_product: f32 = query
    //         .iter()
    //         .zip(embeddings.iter())
    //         .map(|(a, b)| a * b)
    //         .sum();
    //     println!("\t'{}': {:.1}%", sentence, dot_product * 100.);
    // }
    let mut all_embeddings: Vec<(String, Vec<f32>)> = vec![];
    for (embedding, sentence) in embeddings.axis_iter(Axis(0)).zip(inputs.iter()) {
        if let Some(embedding_vec) = embedding.to_slice() {
            all_embeddings.push((sentence.to_string(), embedding_vec.to_vec()));
        }
    }

    Ok(all_embeddings)
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

    #[test]
    fn test_query_model() {
        // Paths to test resources
        let model_path = "/home/sand/coding/qwen3-test/model.onnx";
        let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";

        // Check if test resources exist before running the test
        if !std::path::Path::new(model_path).exists() {
            println!("Model file not found at {}. Skipping test.", model_path);
        }

        if !std::path::Path::new(tokenizer_path).exists() {
            println!(
                "Tokenizer file not found at {}. Skipping test.",
                tokenizer_path
            );
        }

        let inputs = vec![
            "Mr. and Mrs. Bennet live with their five daughters. Jane, the eldest daughter, falls in love with Charles Bingley, a rich bachelor who moves into a house nearby with his two sisters and friend, Fitzwilliam Darcy. Darcy is attracted to the second daughter, Elizabeth, but she finds him arrogant and self-centered. When Darcy proposes to Elizabeth, she refuses. But perhaps there is more to Darcy than meets the eye.",
            "\"The Declaration of Independence of the United States of America\" by Thomas Jefferson is a historic and foundational document penned in the late 18th century during the American Revolutionary period. This work primarily serves as a formal statement declaring the thirteen American colonies' separation from British rule, asserting their right to self-governance and independence. It encapsulates the philosophical underpinnings of democracy, highlighting fundamental human rights and the social contract between the government and the governed.  The text begins with a powerful introduction that outlines the principles of equality and the unalienable rights of individuals to life, liberty, and the pursuit of happiness. It details the various grievances against King George III, illustrating how his actions have eroded the colonists' rights and justified their decision to seek independence. By listing these grievances, the document seeks to assert the colonies' legitimate claim to self-determination. The Declaration culminates in a solemn proclamation of independence, stating that the colonies are entitled to be free and independent states, free from British authority and capable of forming their own alliances, levying war, and engaging in commerce. The Declaration's closing emphasizes the signers' mutual pledge to support this cause, reinforcing the commitment of the colonists to their newly proclaimed liberty.",
            "A woman and a man fall in love. The man's friend pursues the woman's younger sister, who does not like him at first.",
            "A woman meets a man who she does not like at first, even though he likes her.",
            "A dummy piece of text",
            " Beautiful, clever, rich—and single—Emma Woodhouse is perfectly content with her life and sees no need for either love or marriage. Nothing, however, delights her more than interfering in the romantic lives of others. But when she ignores the warnings of her good friend Mr. Knightley and attempts to arrange a suitable match for her protegee Harriet Smith, her carefully laid plans soon unravel and have consequences that she never expected.",
            "A man is exiled from his home, only to come back years later to take revenge.",
        ];

        let mut session = ready_model(model_path).unwrap();
        let tokenizer = ready_tokenizer(tokenizer_path);

        // Call the function - it should not panic and should return Ok
        let result = query_model(&mut session, &tokenizer, inputs.clone()).unwrap();
        let result2 = query_model(&mut session, &tokenizer, inputs).unwrap();

        println!(
            "embedding 2 {:?}, is equal {}",
            result[1],
            result[1] == result2[1]
        );
    }
}
