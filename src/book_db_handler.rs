use std::task::Context;

use pgvector::Vector;
use sqlx::Row;
use sqlx::postgres::PgPool;

pub async fn set_up_table(
    pool: &PgPool,
    table_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("set up user");

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

    let embedding = Vector::from(vec![1.0, 2.0, 3.0]);

    let insert_string = format!(
        "
        INSERT INTO {} (embedding) VALUES ($1)
        ",
        table_name
    );
    sqlx::query(insert_string.as_str())
        .bind(embedding)
        .execute(pool)
        .await?;

    let select_string = format!(
        "
        SELECT embedding FROM {} LIMIT 1
        ",
        table_name
    );
    let row = sqlx::query(select_string.as_str())
        .bind(table_name)
        .fetch_one(pool)
        .await?;
    let embedding2: Vector = row.try_get("embedding")?;
    println!("{:?}", embedding2);

    // let create_user = sqlx::query(
    //     r#"
    //     create user $1;
    //     "#,
    // )
    // .bind(username)
    // .execute(pool)
    // .await?;

    // println!("create user result: {:?}", create_user);

    // let grant_user = client.execute(
    //     "
    //         grant all on $2 to $1;
    //     ",
    //     &[&username, &table],
    // )?;
    // println!("grant user result: {:?}", grant_user);

    // let mut client_new = Client::connect(
    //     format!("host=localhost, user={}, dbname=book_recommender", username).as_str(),
    //     NoTls,
    // )?;

    // for row in client_new.query("SELECT * FROM $ limit 1", &[])? {
    //     println!("row is {:?}", row);
    // }

    Ok(())
}

pub async fn run_postgres_example(pool: &PgPool) -> Result<(), Box<dyn std::error::Error>> {
    // let rec = sqlx::query!(
    //     r#"
    // INSERT INTO test_table ( id, title )
    // VALUES ( 2, $1 )
    // RETURNING id
    //         "#,
    //     "sqlx test"
    // )
    // .fetch_one(pool)
    // .await?;

    // // client.batch_execute(
    // //     "
    // //     CREATE TABLE person (
    // //         id      SERIAL PRIMARY KEY,
    // //         name    TEXT NOT NULL,
    // //         data    BYTEA
    // //     )
    // // ",
    // // )?;

    // let rows = sqlx::query!("SELECT * from test_table limit 1")
    //     .fetch_all(pool)
    //     .await?;

    // for row in rows {
    //     println!("- {}: {}", row.id, &row.title,);
    // }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}
