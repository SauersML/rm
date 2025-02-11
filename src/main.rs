use std::{
    env,
    ffi::CString,
    io,
    path::{Path},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use futures::{stream, StreamExt};
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use num_cpus;
use tokio::runtime::Builder;
use std::os::unix::ffi::OsStrExt;


// ===========================
// HARDCODED CONSTANTS (from empirical tests):
const T_CONVERSION_NS: f64 = 500.0;  // Adjust later... actually use this
const T_SYSCALL_NS: f64 = 1000.0; // Adjust later
const T_OVERHEAD_NS: f64 = 170.0;  // Adjust later


/// Delete a single file via a direct `unlink` (libc) call.
fn unlink_file(path: &Path) -> io::Result<()> {
    let c_str = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Path contains null byte"))?;

    let ret = unsafe { libc::unlink(c_str.as_ptr()) };
    if ret == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Compute the optimal concurrency level.
fn compute_optimal_concurrency(num_files: usize, t_syscall_ns: f64, t_overhead_ns: f64) -> usize {
    if num_files == 0 {
        return 1;
    }

    let n_cpus = num_cpus::get();

    // n_opt = sqrt( (N * T_syscall) / T_overhead )
    let raw_value = ((num_files as f64) * t_syscall_ns / t_overhead_ns).sqrt();
    let candidate_n = raw_value.round() as usize;

    // Limit concurrency to the number of CPU cores, and make sure it's at least 1.
    let n_opt = candidate_n.clamp(1, n_cpus);
    n_opt
}

/// Main async entry point.
async fn run_deletion(pattern: &str) -> io::Result<()> {
    // Gather files
    let mut files = Vec::new();
    for entry in glob(pattern).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Invalid glob pattern: {}", e),
        )
    })? {
        match entry {
            Ok(path) => {
                if path.is_file() {
                    files.push(path);
                }
            }
            Err(e) => {
                eprintln!("Glob error while matching {}: {}", pattern, e);
            }
        }
    }

    let total_files = files.len();
    if total_files == 0 {
        println!("No matching files found for pattern '{}'.", pattern);
        return Ok(());
    }

    // Compute optimal concurrency
    let concurrency = compute_optimal_concurrency(total_files, T_SYSCALL_NS, T_OVERHEAD_NS);
    println!(
        "[INFO] Deleting {} files with concurrency = {} (CPU cores = {})",
        total_files,
        concurrency,
        num_cpus::get()
    );

    // Set up progress bar
    let pb = ProgressBar::new(total_files as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] {wide_bar} {pos}/{len} (ETA {eta})")
            .progress_chars("=>-"),
    );

    // Delete files with controlled concurrency
    let completed_counter = Arc::new(AtomicUsize::new(0));
    let file_stream = stream::iter(files.into_iter());

    file_stream
        .for_each_concurrent(Some(concurrency), |path| {
            let pb = pb.clone();
            let completed_counter = completed_counter.clone();
            async move {
                match unlink_file(&path) {
                    Ok(_) => {
                        let done = completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
                        pb.set_position(done as u64);
                    }
                    Err(e) => {
                        eprintln!("Failed to delete '{}': {}", path.display(), e);
                        // Still count errors towards completion for accurate progress.
                        let done = completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
                        pb.set_position(done as u64);
                    }
                }
            }
        })
        .await;

    pb.finish_with_message("Deletion complete!");
    Ok(())
}

/// Synchronous wrapper around the async function.
fn main() {
    let runtime = Builder::new_multi_thread() // Use a multi-threaded runtime
        .enable_all()
        .build()
        .expect("Failed to build Tokio runtime");

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pattern>", args[0]);
        eprintln!("Example: {} 'some_files_*.txt'", args[0]);
        std::process::exit(1);
    }

    let pattern = &args[1];

    let result = runtime.block_on(run_deletion(pattern));

    match result {
        Ok(_) => println!("Files matching '{}' deleted successfully!", pattern),
        Err(e) => {
            eprintln!("Error during deletion: {}", e);
            std::process::exit(1);
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::{self, File},
        io::Write,
        path::PathBuf,
        process::Command,
        time::Instant,
    };
    use tempfile::tempdir;

    /// A helper to create multiple files in a given directory, each of a specified size (in kilobytes).
    fn create_test_files(dir: &Path, count: usize, size_kb: usize) -> Vec<PathBuf> {
        let mut paths = Vec::with_capacity(count);
        for i in 0..count {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut f = File::create(&file_path).unwrap();
            // Write `size_kb * 1024` bytes
            f.write_all(&vec![0u8; size_kb * 1024]).unwrap();
            paths.push(file_path);
        }
        paths
    }

    /// Measures how long our Rust tool takes to delete files matching the pattern.
    /// Returns the duration in seconds as an f64 for easy comparisons.
    fn measure_rust_deletion(pattern: &str) -> f64 {
        let start = Instant::now();
        // We create a fresh Tokio runtime to run `run_deletion`
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let result = rt.block_on(run_deletion(pattern));
        let elapsed = start.elapsed().as_secs_f64();

        if let Err(e) = result {
            panic!("Error in Rust deletion: {}", e);
        }
        elapsed
    }

    /// Measures how long the system `rm` command takes to delete the given pattern.
    /// Returns the duration in seconds as an f64.
    fn measure_rm_deletion(pattern: &str) -> f64 {
        let start = Instant::now();
        // We use `-f` to ignore nonexistent files (no prompt) and skip directories safely
        let output = Command::new("rm")
            .arg("-f")
            .arg(pattern)
            .output()
            .expect("Failed to run `rm` command");
        let elapsed = start.elapsed().as_secs_f64();

        if !output.status.success() {
            eprintln!("rm command stderr: {}", String::from_utf8_lossy(&output.stderr));
        }
        elapsed
    }

    /// Utility to format seconds in a human-friendly way for logging in tests.
    fn format_duration(seconds: f64) -> String {
        if seconds < 1.0 {
            format!("{:.3} ms", seconds * 1000.0)
        } else {
            format!("{:.3} s", seconds)
        }
    }

    #[test]
    fn test_delete_small_number_of_small_files() {
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_path_buf();
        let file_count = 10;
        let file_size_kb = 1;

        // Create files
        create_test_files(&base_path, file_count, file_size_kb);
        let pattern = base_path.join("test_file_*.dat");
        let pattern_str = pattern.to_string_lossy().to_string();

        // Rust deletion time
        let rust_time = measure_rust_deletion(&pattern_str);

        // Recreate files for rm test
        create_test_files(&base_path, file_count, file_size_kb);
        let rm_time = measure_rm_deletion(&pattern_str);

        println!(
            "[test_delete_small_number_of_small_files] Rust: {}, rm: {}",
            format_duration(rust_time),
            format_duration(rm_time)
        );
    }

    #[test]
    fn test_delete_large_number_of_small_files() {
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_path_buf();
        let file_count = 50000;
        let file_size_kb = 1; // Very small files, but many.

        // Create files
        create_test_files(&base_path, file_count, file_size_kb);
        let pattern = base_path.join("test_file_*.dat");
        let pattern_str = pattern.to_string_lossy().to_string();

        println!(
            "Benchmark: Deleting {} small files ({}KB each)",
            file_count, file_size_kb
        );

        // Rust deletion time
        let rust_time = measure_rust_deletion(&pattern_str);
        println!("Rust Deletion Time: {}", format_duration(rust_time));

        // Recreate for rm
        create_test_files(&base_path, file_count, file_size_kb);
        let rm_time = measure_rm_deletion(&pattern_str);
        println!("rm Command Deletion Time: {}", format_duration(rm_time));

        // Compare
        if rust_time < rm_time {
            println!(
                "Our Rust code is faster than 'rm' by {}",
                format_duration(rm_time - rust_time)
            );
        } else {
            println!(
                "'rm' is faster than our Rust code by {}",
                format_duration(rust_time - rm_time)
            );
        }
        println!("---");
    }

    #[test]
    fn test_delete_small_number_of_large_files() {
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_path_buf();
        let file_count = 100;
        let file_size_kb = 10240; // 10 MB each
        println!(
            "Benchmark: Deleting {} large files ({} KB each = ~{} MB total)",
            file_count,
            file_size_kb,
            file_count * (file_size_kb / 1024)
        );

        // Create files
        create_test_files(&base_path, file_count, file_size_kb);
        let pattern = base_path.join("test_file_*.dat");
        let pattern_str = pattern.to_string_lossy().to_string();

        let rust_time = measure_rust_deletion(&pattern_str);
        println!("Rust Deletion Time: {}", format_duration(rust_time));

        // Recreate for rm
        create_test_files(&base_path, file_count, file_size_kb);
        let rm_time = measure_rm_deletion(&pattern_str);
        println!("rm Command Deletion Time: {}", format_duration(rm_time));

        if rust_time < rm_time {
            println!(
                "Our Rust code is faster than 'rm' by {}",
                format_duration(rm_time - rust_time)
            );
        } else {
            println!(
                "'rm' is faster than our Rust code by {}",
                format_duration(rust_time - rm_time)
            );
        }
        println!("---");
    }

    #[test]
    fn test_delete_with_no_matches() {
        // Confirm we gracefully handle a pattern with no files.
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_string_lossy().to_string();
        let pattern = format!("{}/no_such_file_*.dat", base_path);

        // This shouldn't delete anything or error out.
        let rust_time = measure_rust_deletion(&pattern);
        println!(
            "[test_delete_with_no_matches] Rust took {} (no files to delete).",
            format_duration(rust_time)
        );
    }

    #[test]
    fn test_skips_directories() {
        // Confirm that directories are NOT removed.
        let tmp = tempdir().unwrap();
        let dir_path = tmp.path().join("my_test_dir");
        fs::create_dir(&dir_path).unwrap();

        // Also create a file that matches pattern
        let file_path = tmp.path().join("my_test_dir_file.dat");
        File::create(&file_path).unwrap();

        let pattern = tmp.path().join("my_test_dir*");
        let pattern_str = pattern.to_string_lossy().to_string();

        let rust_time = measure_rust_deletion(&pattern_str);
        println!(
            "[test_skips_directories] Rust took {}.",
            format_duration(rust_time)
        );

        // Directory should still exist
        assert!(dir_path.is_dir());
        // File should be deleted
        assert!(!file_path.exists(), "File was not deleted!");
    }
}

// cargo test --release -- --nocapture
