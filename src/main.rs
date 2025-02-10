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


#[cfg(test)]
mod benchmarks {
    use super::*;
    use tempfile::tempdir;
    use std::fs::{File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{Instant};

    use std::process::Command;

    // Helper function to create test files
    fn create_files(base_dir: &PathBuf, num_files: usize, file_size_kb: usize) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for i in 0..num_files {
            let file_path = base_dir.join(format!("test_file_{}.txt", i));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(vec![0u8; file_size_kb * 1024].as_slice()).unwrap(); // Create file with specified size
            files.push(file_path);
        }
        files
    }

    // Helper function to run the Rust delete function and measure time
    async fn run_rust_delete(pattern: &str) -> std::time::Duration {
        let start_time = Instant::now();
        delete_files(pattern).await.unwrap();
        start_time.elapsed()
    }

    // Helper function to run the 'rm' command and measure time
    fn run_rm_command(pattern: &str) -> std::time::Duration {
        let start_time = Instant::now();
        Command::new("rm")
            .arg("-rf") // -rf for recursive and force delete (like our script)
            .arg(pattern)
            .output()
            .expect("Failed to execute rm command");
        start_time.elapsed()
    }


    #[tokio::test]
    async fn benchmark_large_number_small_files() {
        let temp_dir = tempdir().unwrap();
        let base_dir_path = temp_dir.path().to_path_buf();
        let num_files = 50000; // Large number of files
        let file_size_kb = 1;    // Small file size

        create_files(&base_dir_path, num_files, file_size_kb);
        let pattern = base_dir_path.join("test_file_*.txt").to_string_lossy().to_string();

        println!("Benchmark: Deleting {} small files ({}KB each)", num_files, file_size_kb);

        let rust_duration = run_rust_delete(&pattern).await;
        println!("Rust Deletion Time: {:?}", rust_duration);

        // Re-create files for rm benchmark (rm is destructive)
        create_files(&base_dir_path, num_files, file_size_kb);
        let rm_duration = run_rm_command(&pattern);
        println!("rm Command Deletion Time: {:?}", rm_duration);

        println!("Comparison:");
        if rust_duration < rm_duration {
            println!("Rust code is faster than 'rm' by {:?}", rm_duration - rust_duration);
        } else {
            let diff = rust_duration - rm_duration; // Calculate difference correctly
            println!("'rm' command is faster than Rust code by {:?}", diff);
        }
        println!("---");
    }


    #[tokio::test]
    async fn benchmark_small_number_large_files() {
        let temp_dir = tempdir().unwrap();
        let base_dir_path = temp_dir.path().to_path_buf();
        let num_files = 1000;   // Small number of files
        let file_size_kb = 10240; // Large file size (10MB)

        create_files(&base_dir_path, num_files, file_size_kb);
        let pattern = base_dir_path.join("test_file_*.txt").to_string_lossy().to_string();

        println!("Benchmark: Deleting {} large files ({}KB each)", num_files, file_size_kb);

        let rust_duration = run_rust_delete(&pattern).await;
        println!("Rust Deletion Time: {:?}", rust_duration);

        // Re-create files for rm benchmark (rm is destructive)
        create_files(&base_dir_path, num_files, file_size_kb);
        let rm_duration = run_rm_command(&pattern);
        println!("rm Command Deletion Time: {:?}", rm_duration);

        println!("Comparison:");
        if rust_duration < rm_duration {
            println!("Rust code is faster than 'rm' by {:?}", rm_duration - rust_duration);
        } else {
            let diff = rust_duration - rm_duration; // Calculate difference correctly
            println!("'rm' command is faster than Rust code by {:?}", diff);
        }
        println!("---");
    }
}

// cargo test --release -- --nocapture
