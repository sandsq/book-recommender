use ndarray::{Axis, Ix2};
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

pub fn query_model(model_path: &str, tokenizer_path: &str) -> ort::Result<()> {
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

    let inputs = vec![
        "Mr. and Mrs. Bennet live with their five daughters. Jane, the eldest daughter, falls in love with Charles Bingley, a rich bachelor who moves into a house nearby with his two sisters and friend, Fitzwilliam Darcy. Darcy is attracted to the second daughter, Elizabeth, but she finds him arrogant and self-centered. When Darcy proposes to Elizabeth, she refuses. But perhaps there is more to Darcy than meets the eye.",
        "A woman and a man fall in love. The man's friend pursues the woman's younger sister, who does not like him at first.",
        "A woman meets a man who she does not like at first, even though he likes her.",
        "A dummy piece of text",
        " Beautiful, clever, rich—and single—Emma Woodhouse is perfectly content with her life and sees no need for either love or marriage. Nothing, however, delights her more than interfering in the romantic lives of others. But when she ignores the warnings of her good friend Mr. Knightley and attempts to arrange a suitable match for her protegee Harriet Smith, her carefully laid plans soon unravel and have consequences that she never expected.",
        "A man is exiled from his home, only to come back years later to take revenge.",
    ];
    // let inputs = vec![
    //     "The weather outside is lovely.",
    //     "It's so sunny outside!",
    //     "She drove to the stadium.",
    // ];

    // Encode our input strings.
    let encodings = tokenizer
        .encode_batch(inputs.clone(), true)
        .map_err(|e| Error::new(e.to_string()))?;
    println!("Encodings: {:?}", encodings);

    let max_length = encodings
        .iter()
        .map(|enc| enc.get_ids().len())
        .max()
        .unwrap_or(0);
    println!("Maximum token length: {}", max_length);

    // 2. Prepare padded token IDs and attention masks
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
    println!("Padded Token IDs: {:?}", padded_ids);
    println!("Padded Attention Masks: {:?}", padded_mask);

    // Get the padded length of each encoding.
    // let padded_token_length = encodings[0].len();
    let padded_token_length = max_length;
    println!("Padded token length: {}", padded_token_length);

    // // Get our token IDs & mask as a flattened array.
    // let mut ids: Vec<i64> = encodings
    //     .iter()
    //     .flat_map(|e| e.get_ids().iter().map(|i| *i as i64))
    //     .collect();
    // let mut mask: Vec<i64> = encodings
    //     .iter()
    //     .flat_map(|e| e.get_attention_mask().iter().map(|i| *i as i64))
    //     .collect();

    let ids = padded_ids;
    let mask = padded_mask;

    println!("Token IDs: {:?} with length {}", ids, ids.len());
    println!("Attention Mask: {:?} with length {}", mask, mask.len());

    // while (ids.len() < inputs.len() * padded_token_length) {
    //     ids.push(0);
    //     mask.push(0);
    // }

    // Convert our flattened arrays into 2-dimensional tensors of shape [N, L].
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
        .into_dimensionality::<Ix2>()
        .unwrap();

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
