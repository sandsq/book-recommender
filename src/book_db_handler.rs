use pgvector::Vector;
use rayon::prelude::*;
use sqlx::Row;
use sqlx::postgres::PgPool;
use std::cell::RefCell;
use std::sync::mpsc;

use crate::book_metadata::RdfFileIterator;
use crate::models::{query_model, ready_model, ready_tokenizer};

// psql -U postgres
// sudo -i -u postgres
// initdb -D /var/lib/postgres/data
// sudo systemctl start postgresql.service
//
// \l to list databases
// \dt to list tables
// \c db to connect to db
// https://www.postgresql.org/docs/current/predefined-roles.html
// https://www.postgresql.org/docs/current/sql-createrole.html
// create role role_name login;
//
// https://www.postgresql.org/docs/current/sql-grant.html
// grant all on table_name to role_name;
// GRANT ALL ON SCHEMA public TO role_name;
// if using serial:
// grant all on sequence table_name_id_seq to role_name;
// grant all on all sequences in schema public to role_name;

pub async fn set_up_metadata_table(
    pool: &PgPool,
    table_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let table_creation_string = format!(
        "
        CREATE TABLE IF NOT EXISTS {} (
        id bigint PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT NOT NULL,
        birthyear INTEGER,
        deathyear INTEGER,
        summary TEXT
        )",
        table_name
    );

    sqlx::query(table_creation_string.as_str())
        .execute(pool)
        .await?;

    // let all_metadata = process_all_rdf_files("data/cache/epub", Some((1, 10)), Some(false))?;
    let metadata_iterator = RdfFileIterator::new("data/cache/epub", None, Some(false))?;

    for metadata in metadata_iterator {
        let metadata = metadata?;
        let insert_string = format!(
            "
            INSERT INTO {} (id, title, author, birthyear, deathyear, summary) VALUES ($1, $2, $3, $4, $5, $6)
            ",
            table_name
        );
        let birthyear = match metadata.birthyear.parse::<i32>() {
            Ok(year) => Some(year),
            Err(_) => None,
        };
        let deathyear = match metadata.deathyear.parse::<i32>() {
            Ok(year) => Some(year),
            Err(_) => None,
        };
        match sqlx::query(insert_string.as_str())
            .bind(metadata.id)
            .bind(metadata.title)
            .bind(metadata.author)
            .bind(birthyear)
            .bind(deathyear)
            .bind(metadata.summary)
            .execute(pool)
            .await
        {
            Ok(_) => (),
            Err(e) => {
                println!("Error inserting metadata for book {}: {}", metadata.id, e);
                continue;
            }
        };
    }

    Ok(())
}

pub async fn set_up_vector_table(
    pool: &PgPool,
    table_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;
    println!("added pgvector extension");

    let table_creation_string = format!(
        "
        CREATE TABLE IF NOT EXISTS {} (
        id bigserial PRIMARY KEY,
        embedding vector(1024)
    )",
        table_name
    );

    sqlx::query(table_creation_string.as_str())
        .bind(table_name)
        .execute(pool)
        .await?;

    let model_path = "/home/sand/coding/qwen3-test/model.onnx";
    let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
    let mut session = ready_model(model_path)?;
    let tokenizer = ready_tokenizer(tokenizer_path);

    let metadata_iterator = RdfFileIterator::new("data/cache/epub", None, Some(false))?;

    for metadata in metadata_iterator {
        let metadata = metadata?;

        let id = metadata.id;
        let result = sqlx::query("select exists(select 1 from book_summary_vectors where id = $1)")
            .bind(id)
            .fetch_all(pool)
            .await?;
        let val: bool = result.get(0).unwrap().get(0);
        if val {
            println!("already did id {}", id);
            continue;
        }

        let summary = vec![metadata.summary.as_str()];
        let summary_vector = Vector::from(
            query_model(&mut session, &tokenizer, summary)?
                .first()
                .unwrap()
                .1
                .clone(),
        );
        let insert_string = format!(
            "
            INSERT INTO {} (id, embedding) VALUES ($1, $2)
            ",
            table_name
        );

        match sqlx::query(insert_string.as_str())
            .bind(metadata.id)
            .bind(summary_vector)
            .execute(pool)
            .await
        {
            Ok(_) => (),
            Err(e) => {
                println!("Error inserting metadata for book {}: {}", metadata.id, e);
                continue;
            }
        };
    }

    Ok(())
}

pub async fn set_up_vector_table_par(
    pool: &PgPool,
    table_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    sqlx::query("CREATE EXTENSION IF NOT EXISTS vector")
        .execute(pool)
        .await?;
    println!("added pgvector extension");

    let table_creation_string = format!(
        "
        CREATE TABLE IF NOT EXISTS {} (
        id bigserial PRIMARY KEY,
        embedding vector(1024)
    )",
        table_name
    );

    sqlx::query(table_creation_string.as_str())
        .bind(table_name)
        .execute(pool)
        .await?;

    let model_path = "/home/sand/coding/qwen3-test/model.onnx";
    let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
    let tokenizer = ready_tokenizer(tokenizer_path);

    let metadata_iterator = RdfFileIterator::new("data/cache/epub", None, Some(false))?;

    // Collect metadata that needs processing
    let mut metadata_to_process = Vec::new();
    for metadata in metadata_iterator {
        let metadata = metadata?;
        let id = metadata.id;

        let result = sqlx::query("select exists(select 1 from book_summary_vectors where id = $1)")
            .bind(id)
            .fetch_all(pool)
            .await?;
        let val: bool = result.get(0).unwrap().get(0);

        if val {
            println!("already did id {}", id);
            continue;
        }

        metadata_to_process.push(metadata);
    }

    // Process in batches: compute vectors in parallel, then insert, repeat
    let batch_size = 100;
    thread_local! {
        static MODEL_SESSION: RefCell<Option<ort::session::Session>> = RefCell::new(None);
    }

    for chunk in metadata_to_process.chunks(batch_size) {
        // Parallel processing of model inference for this batch
        let results: Vec<_> = chunk
            .par_iter()
            .map(|metadata| {
                MODEL_SESSION.with(|session_cell| {
                    let mut session_opt = session_cell.borrow_mut();
                    if session_opt.is_none() {
                        *session_opt = ready_model(model_path).ok();
                    }

                    let session = session_opt.as_mut()?;
                    let summary = vec![metadata.summary.as_str()];
                    let summary_vector = query_model(session, &tokenizer, summary)
                        .ok()?
                        .first()
                        .map(|(_, vec)| Vector::from(vec.clone()))?;

                    Some((metadata.id, summary_vector))
                })
            })
            .collect();

        // Insert this batch into the database
        for result in results {
            if let Some((id, summary_vector)) = result {
                let insert_string = format!(
                    "
                    INSERT INTO {} (id, embedding) VALUES ($1, $2)
                    ",
                    table_name
                );

                match sqlx::query(insert_string.as_str())
                    .bind(id)
                    .bind(summary_vector)
                    .execute(pool)
                    .await
                {
                    Ok(_) => (),
                    Err(e) => {
                        println!("Error inserting metadata for book {}: {}", id, e);
                        continue;
                    }
                };
            }
        }

        println!("Completed batch of {} items", chunk.len());
    }

    Ok(())
}

pub async fn query_sample_text(
    pool: &PgPool,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/sand/coding/qwen3-test/model.onnx";
    let tokenizer_path = "/home/sand/coding/qwen3-test/tokenizer.json";
    let mut session = ready_model(model_path)?;
    let tokenizer = ready_tokenizer(tokenizer_path);
    let text_embedding = Vector::from(
        query_model(&mut session, &tokenizer, vec![text])?
            .first()
            .unwrap()
            .1
            .clone(),
    );

    let query_string = format!(
        "
        with top3 as (
            SELECT *, embedding <=> $1 as distance
            FROM book_summary_vectors ORDER BY embedding <=> $1 LIMIT 3
        )
        select *
        from book_metadata m
        join top3 t
        on m.id = t.id
        "
    );

    let result = sqlx::query(query_string.as_str())
        .bind(text_embedding)
        .fetch_all(pool)
        .await?;

    println!("Query result: {:?}", result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use dotenv::dotenv;
    use std::env;

    use sqlx::postgres::PgPoolOptions;

    use super::*;

    #[tokio::test]
    async fn test_set_up_metadata_table() -> Result<(), Box<dyn std::error::Error>> {
        dotenv().ok();
        let DATABASE_URL = env::var("DATABASE_URL")?;
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&DATABASE_URL)
            .await?;
        let table_name = "book_metadata";
        match set_up_metadata_table(&pool, table_name).await {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    #[tokio::test]
    async fn test_query_sample_text() -> Result<(), Box<dyn std::error::Error>> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&env::var("DATABASE_URL")?)
            .await?;
        let table_name = "book_summary_vectors";
        // let text = "A woman and a man fall in love. The man's friend pursues the woman's younger sister, who does not like him at first.";
        // let text = "Lockwood, the new tenant of Thrushcross Grange, situated on the bleak Yorkshire moors, is forced to seek shelter one night at Wuthering Heights, the home of his landlord. There he discovers the history of the tempestuous events that took place years before; of the intense relationship between the gypsy foundling Heathcliff and Catherine Earnshaw; and how Catherine, forced to choose between passionate, tortured Heathcliff and gentle, well-bred Edgar Linton, surrendered to the expectations of her class. As Heathcliff's bitterness and vengeance at his betrayal is visited upon the next generation, their innocent heirs must struggle to escape the legacy of the past.";
        // let text = "A man finds himself stranded in the wilderness. He must use his survival skills to hold out until a rescue party arrives. In the process, he learns about himself.";
        // let text = "A man seeks revenge against someone who wronged his lover.";
        // let text = "A science fiction novel where the characters make a pilgrimage through space.";
        // let text =
        // "Sailors attempt to cross a treacherous sea but must contend with weather and pirates.";
        let text = "Sailors adventure through the seas, finding treasure while escaping bad guys.";
        match query_sample_text(&pool, text).await {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
