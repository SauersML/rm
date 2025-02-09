use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::task;
use glob::glob;
use tokio::sync::Semaphore;
use futures::future::{try_join_all};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Collect arguments
    let args: Vec<String> = env::args().collect();
    
    // Ensure that a pattern is provided
    if args.len() < 2 {
        eprintln!("Error: A pattern must be provided as a command line argument.");
        std::process::exit(1);
    }

    let pattern = &args[1];

    // Perform the deletion process
    if let Err(err) = delete_files(pattern).await {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    } else {
        println!("Files matching the pattern '{}' have been deleted successfully!", pattern);
    }
}

async fn delete_files<P: AsRef<Path>>(pattern: P) -> Result<(), Box<dyn std::error::Error>> {
    let pattern_str = pattern.as_ref().to_string_lossy();
    let files = find_files_by_pattern(&pattern_str)?;

    // Semaphore to control concurrency in the application
    let semaphore = Arc::new(Semaphore::new(0)); // No concurrency limit at application level

    let delete_futures: Vec<_> = files.into_iter().map(|file| {
        task::spawn_blocking(move || {
            // Deleting file. This is a blocking operation so we spawn it in a blocking task
            match delete_file(&file) {
                Ok(_) => {},
                Err(e) => eprintln!("Failed to delete {}: {}", file.display(), e),
            }
        })
    }).collect();

    // Wait for all delete tasks to finish
    try_join_all(delete_futures).await?;

    Ok(())
}

fn find_files_by_pattern(pattern: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();

    // Use glob to match the pattern (supports wildcards)
    for entry in glob(pattern)? {
        match entry {
            Ok(path) => {
                if path.is_file() {
                    files.push(path);
                }
            },
            Err(e) => eprintln!("Error matching pattern {}: {}", pattern, e),
        }
    }

    Ok(files)
}

fn delete_file(file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut attempt = 0;

    loop {
        attempt += 1;
        match fs::remove_file(file) {
            Ok(_) => return Ok(()),
            Err(e) if attempt < 3 => {
                // Retry logic if file removal fails
                eprintln!("Attempt {}: Failed to delete file {}: {}. Retrying...", attempt, file.display(), e);
                std::thread::sleep(std::time::Duration::from_secs(1));  // Sleep before retrying
            },
            Err(e) => {
                // Permanent failure after retries
                eprintln!("Attempt {}: Permanent failure to delete file {}: {}", attempt, file.display(), e);
                return Err(Box::new(e));
            }
        }
    }
}
