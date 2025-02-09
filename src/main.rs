use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tokio::task;
use glob::glob;
use futures::future::{try_join_all};
use indicatif::{ProgressBar, ProgressStyle};

#[tokio::main]
async fn main() {
    // Collect arguments
    let args: Vec<String> = env::args().collect();
    
    // Make sure a pattern is provided
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

    // Create a progress bar to show progress
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg} {wide_bar} {pos}/{len} ({eta})")
        .progress_chars("##-"));

    // Spawn tasks for each file
    let delete_futures: Vec<_> = files.into_iter().map(|file| {
        let pb = pb.clone(); // Clone the progress bar for each task
        task::spawn(async move {
            match delete_file(&file).await {
                Ok(_) => pb.inc(1), // Increment progress bar after successful deletion
                Err(e) => eprintln!("Failed to delete {}: {}", file.display(), e),
            }
        })
    }).collect();

    // Wait for all delete tasks to finish
    try_join_all(delete_futures).await?;

    pb.finish_with_message("Deletion complete!");
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

async fn delete_file(file: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut attempt = 0;

    loop {
        attempt += 1;
        match fs::remove_file(file) {
            Ok(_) => return Ok(()),
            Err(e) if attempt < 3 => {
                // Retry logic with exponential backoff if file removal fails
                eprintln!("Attempt {}: Failed to delete file {}: {}. Retrying...", attempt, file.display(), e);
                tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(attempt as u32))).await;  // Exponential backoff
            },
            Err(e) => {
                // Permanent failure after retries
                eprintln!("Attempt {}: Permanent failure to delete file {}: {}", attempt, file.display(), e);
                return Err(Box::new(e));
            }
        }
    }
}
