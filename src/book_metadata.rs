use oxrdfio::{RdfFormat, RdfParser};
use std::fs::File;
use std::io::Read;

pub fn process_rdf() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("data/cache/epub/1/pg1.rdf")?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let quads = RdfParser::from_format(RdfFormat::RdfXml)
        .for_reader(contents.as_bytes())
        .collect::<Result<Vec<_>, _>>()?;

    println!("{:?}", quads);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_rdf() {
        // Call the function to test
        match process_rdf() {
            Ok(_) => assert!(true), // Test passes if no error
            Err(e) => panic!("Test failed with error: {}", e),
        }
    }
}
