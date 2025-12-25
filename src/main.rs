mod book_metadata;
mod models;

fn main() -> ort::Result<()> {
    let model_path = "/home/sand/coding/qwen3-test/model.onnx";
    let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
    models::get_model_info(model_path)?;
    models::query_model(model_path, tokenizer_path)?;
    Ok(())
}
