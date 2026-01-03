use oxrdfio::{RdfFormat, RdfParser};
use serde::{Deserialize, Serialize};
use serde_yaml; // Add this line for YAML serialization
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BookMetadata {
    pub id: i32,
    pub title: String,
    pub author: String,
    pub birthyear: String,
    pub deathyear: String,
    pub summary: String,
}

// Extracting the ID from the filename
fn extract_id_from_filename(file_path: &str) -> Result<i32, std::num::ParseIntError> {
    let id = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .strip_prefix("pg")
        .unwrap_or("")
        .parse::<i32>();

    id
}

pub fn process_rdf(
    file_path: &str,
    write_flag: Option<bool>,
) -> Result<BookMetadata, Box<dyn std::error::Error>> {
    println!("Processing file: {:?}", file_path);
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
                book_metadata.birthyear = quad
                    .object
                    .to_string()
                    .split("^^")
                    .next()
                    .unwrap_or("")
                    .to_string()
                    .replace("\"", "");
            }

            "<http://www.gutenberg.org/2009/pgterms/deathdate>" => {
                book_metadata.deathyear = quad
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
                    .replace("(This is an automatically generated summary.)", "")
                    .trim()
                    .to_string();
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
    // println!("{:?}", book_metadata);
    // println!("{}", book_metadata.summary);

    // Write book_metadata to file
    if let Some(true) = write_flag {
        write_metadata_to_file(&book_metadata)?;
    }
    Ok(book_metadata)
}

fn write_metadata_to_file(metadata: &BookMetadata) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = format!("data/metadata/book_metadata_{}.yaml", metadata.id);
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

/// An iterator that processes RDF files one at a time
pub struct RdfFileIterator {
    dir_queue: VecDeque<PathBuf>,
    current_rdf_files: std::fs::ReadDir,
    id_range: Option<(u32, u32)>,
    write_flag: Option<bool>,
    finished: bool,
}

impl RdfFileIterator {
    /// Create a new RDF file iterator
    pub fn new(
        epub_dir: &str,
        id_range: Option<(u32, u32)>,
        write_flag: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let entries = fs::read_dir(epub_dir)?;
        let mut dir_queue = VecDeque::new();

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(dir_name) = path.file_name().and_then(|s| s.to_str()) {
                    let book_id = match dir_name.parse::<u32>() {
                        Ok(id) => id,
                        Err(_) => continue,
                    };

                    if let Some((start, end)) = id_range {
                        if book_id < start || book_id > end {
                            continue;
                        }
                    }

                    dir_queue.push_back(path);
                }
            }
        }

        let is_empty = dir_queue.is_empty();
        let current_rdf_files = if let Some(first_dir) = dir_queue.pop_front() {
            fs::read_dir(&first_dir)?
        } else {
            // Empty directory queue - create a dummy iterator
            fs::read_dir(".")?
        };

        Ok(RdfFileIterator {
            dir_queue,
            current_rdf_files,
            id_range,
            write_flag,
            finished: is_empty,
        })
    }
}

impl Iterator for RdfFileIterator {
    type Item = Result<BookMetadata, Box<dyn std::error::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        loop {
            // Try to get the next RDF file from current directory
            match self.current_rdf_files.next() {
                Some(Ok(rdf_file)) => {
                    let rdf_path = rdf_file.path();

                    if rdf_path.extension().and_then(|s| s.to_str()) == Some("rdf") {
                        let path_str = match rdf_path.to_str() {
                            Some(s) => s.to_string(),
                            None => continue,
                        };

                        let metadata = match process_rdf(&path_str, self.write_flag) {
                            Ok(metadata) => metadata,
                            Err(e) => {
                                println!(
                                    "Failed to process RDF file: {} because of {}",
                                    path_str, e
                                );
                                return Some(Err(e));
                            }
                        };

                        return Some(Ok(metadata));
                    }
                }
                Some(Err(e)) => {
                    return Some(Err(Box::new(e)));
                }
                None => {
                    // Move to next directory in queue
                    if let Some(next_dir) = self.dir_queue.pop_front() {
                        match fs::read_dir(&next_dir) {
                            Ok(read_dir) => {
                                self.current_rdf_files = read_dir;
                                continue;
                            }
                            Err(e) => {
                                return Some(Err(Box::new(e)));
                            }
                        }
                    } else {
                        // No more directories
                        self.finished = true;
                        return None;
                    }
                }
            }
        }
    }
}

/// Process all RDF files in the epub directory using an iterator
/// This is a convenience function that collects all results. For streaming processing,
/// use `RdfFileIterator` directly.
///
/// # Arguments
/// * `epub_dir` - The path to the epub directory
/// * `id_range` - Optional range of book IDs to process. If None, processes all books.
///                Format: (start_id, end_id) inclusive
// pub fn process_all_rdf_files(
//     epub_dir: &str,
//     id_range: Option<(u32, u32)>,
//     write_flag: Option<bool>,
// ) -> Result<Vec<BookMetadata>, Box<dyn std::error::Error>> {
//     let mut all_metadata = Vec::new();
//     let iterator = RdfFileIterator::new(epub_dir, id_range, write_flag)?;

//     for result in iterator {
//         match result {
//             Ok(metadata) => all_metadata.push(metadata),
//             Err(e) => {
//                 println!("Error processing metadata: {}", e);
//                 // Optionally continue or return error based on your needs
//             }
//         }
//     }

//     println!("\nProcessed {} books successfully", all_metadata.len());

//     Ok(all_metadata)
// }

#[cfg(test)]
mod tests {

    use super::*;

    // #[test]
    // fn test_process_rdf_pg1() {
    //     let test_path = "data/cache/epub/1/pg1.rdf";
    //     match process_rdf(test_path) {
    //         Ok(_) => assert!(true), // Test passes if no error
    //         Err(e) => panic!("Test failed with error: {}", e),
    //     }
    // }

    // #[test]
    // fn test_process_rdf_pg11() {
    //     let test_path = "data/cache/epub/11/pg11.rdf";
    //     match process_rdf(test_path) {
    //         Ok(_) => assert!(true), // Test passes if no error
    //         Err(e) => panic!("Test failed with error: {}", e),
    //     }
    // }

    // #[test]
    // fn test_process_all_rdf_files_with_range() -> Result<(), Box<dyn std::error::Error>> {
    //     let epub_dir = "data/cache/epub";
    //     match process_all_rdf_files(epub_dir, Some((1, 10)), Some(true)) {
    //         Ok(_) => {
    //             println!("success")
    //         }
    //         Err(e) => panic!("Test failed with error: {}", e),
    //     }

    //     Ok(())
    // }
}
