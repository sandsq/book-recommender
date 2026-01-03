use pgvector::Vector;
use sqlx::Row;
use sqlx::postgres::PgPool;

use crate::book_metadata::RdfFileIterator;

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

    let table_creation_string = format!(
        "
        CREATE TABLE IF NOT EXISTS {} (
        id bigserial PRIMARY KEY,
        embedding vector(3)
    )",
        table_name
    );

    sqlx::query(table_creation_string.as_str())
        .bind(table_name)
        .execute(pool)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::env;

    use sqlx::postgres::PgPoolOptions;

    use super::*;

    #[tokio::test]
    async fn test_set_up_metadata_table() -> Result<(), Box<dyn std::error::Error>> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .connect(&env::var("DATABASE_URL")?)
            .await?;
        let table_name = "book_metadata";
        match set_up_metadata_table(&pool, table_name).await {
            Ok(_) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}
