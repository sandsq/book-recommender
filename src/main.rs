mod book_db_handler;
mod book_metadata;
mod models;

use std::env;

use sqlx::postgres::PgPoolOptions;

#[tokio::main(flavor = "current_thread")]
async fn main() -> ort::Result<(), Box<dyn std::error::Error>> {
    // let model_path = "/home/sand/coding/qwen3-test/model.onnx";
    // let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
    // models::get_model_info(model_path)?;

    // let inputs = vec![
    //     "Mr. and Mrs. Bennet live with their five daughters. Jane, the eldest daughter, falls in love with Charles Bingley, a rich bachelor who moves into a house nearby with his two sisters and friend, Fitzwilliam Darcy. Darcy is attracted to the second daughter, Elizabeth, but she finds him arrogant and self-centered. When Darcy proposes to Elizabeth, she refuses. But perhaps there is more to Darcy than meets the eye.",
    //     "A woman and a man fall in love. The man's friend pursues the woman's younger sister, who does not like him at first.",
    //     "A woman meets a man who she does not like at first, even though he likes her.",
    //     "A dummy piece of text",
    //     " Beautiful, clever, rich—and single—Emma Woodhouse is perfectly content with her life and sees no need for either love or marriage. Nothing, however, delights her more than interfering in the romantic lives of others. But when she ignores the warnings of her good friend Mr. Knightley and attempts to arrange a suitable match for her protegee Harriet Smith, her carefully laid plans soon unravel and have consequences that she never expected.",
    //     "A man is exiled from his home, only to come back years later to take revenge.",
    // ];
    // models::query_model(model_path, tokenizer_path, inputs)?;

    let pool = PgPoolOptions::new()
        .max_connections(5)
        .connect(&env::var("DATABASE_URL")?)
        .await?;
    // "postgres://postgres:@localhost/book_recommender")

    book_db_handler::set_up_table(&pool, "test_table").await?;
    book_db_handler::run_postgres_example(&pool).await?;
    Ok(())
}
