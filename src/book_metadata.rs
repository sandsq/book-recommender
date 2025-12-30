use oxrdfio::{RdfFormat, RdfParser};
use serde::{Deserialize, Serialize};
use serde_yaml; // Add this line for YAML serialization
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BookMetadata {
    id: u32, // Unique ID derived from the filename
    author: String,
    birthdate: String,
    deathdate: String,
    title: String,   // This would be optional based on the RDF
    summary: String, // Corresponding to marc520
}

// Extracting the ID from the filename
fn extract_id_from_filename(file_path: &str) -> Result<u32, std::num::ParseIntError> {
    let id = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .strip_prefix("pg")
        .unwrap_or("")
        .parse::<u32>();

    id
}

pub fn process_rdf(file_path: &str) -> Result<BookMetadata, Box<dyn std::error::Error>> {
    let mut file = File::open(file_path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let quads = RdfParser::from_format(RdfFormat::RdfXml)
        .for_reader(contents.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;

    let book_id = extract_id_from_filename(file_path)?; // Extracts ID from filename
    let mut book_metadata = BookMetadata::default();
    book_metadata.id = book_id;

    for quad in quads {
        match quad.predicate.to_string().as_str() {
            "<http://www.gutenberg.org/2009/pgterms/name>" => {
                book_metadata.author = quad.object.to_string().replace("\"", "");
            }
            "<http://purl.org/dc/terms/title>" => {
                book_metadata.title = quad.object.to_string().replace("\"", "");
            }
            "<http://www.gutenberg.org/2009/pgterms/birthdate>" => {
                book_metadata.birthdate = quad
                    .object
                    .to_string()
                    .split("^^")
                    .next()
                    .unwrap_or("")
                    .to_string()
                    .replace("\"", "");
            }

            "<http://www.gutenberg.org/2009/pgterms/deathdate>" => {
                book_metadata.deathdate = quad
                    .object
                    .to_string()
                    .split("^^")
                    .next()
                    .unwrap_or("")
                    .to_string()
                    .replace("\"", "");
            }
            "<http://www.gutenberg.org/2009/pgterms/marc520>" => {
                book_metadata.summary = quad
                    .object
                    .to_string()
                    .replace("\"", "")
                    .replace("\\", r#"""#)
                // .replace("\\\"", "\"");
            }
            // Optionally include a title extraction if needed
            _ => {}
        }
        // println!("@@@@@@");
        // println!("{}", quad.subject.to_string());
        // println!("{}", quad.predicate.to_string());
        // println!("{}", quad.object.to_string());
    }
    println!("{:?}", book_metadata);
    println!("{}", book_metadata.summary);

    // Write book_metadata to file
    write_metadata_to_file(&book_metadata)?;
    Ok(book_metadata)
}

fn write_metadata_to_file(metadata: &BookMetadata) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = format!("book_metadata_{}.yaml", metadata.id);
    {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(&file_path)?;

        serde_yaml::to_writer(file, &metadata)?;
    }
    println!("Metadata written to {}", file_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_rdf_pg1() {
        let test_path = "data/cache/epub/1/pg1.rdf";
        match process_rdf(test_path) {
            Ok(_) => assert!(true), // Test passes if no error
            Err(e) => panic!("Test failed with error: {}", e),
        }
    }

    #[test]
    fn test_process_rdf_pg422() {
        let test_path = "data/cache/epub/422/pg422.rdf";
        match process_rdf(test_path) {
            Ok(_) => assert!(true), // Test passes if no error
            Err(e) => panic!("Test failed with error: {}", e),
        }
    }
}
