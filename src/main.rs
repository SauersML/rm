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
use num_cpus;
use tokio::runtime::Builder;
use std::os::unix::ffi::OsStrExt;
use lazy_static::lazy_static;
use progression::{Bar, Config};

/// Delete a single file via a direct `unlink` (libc) call.
/// The `for_each_concurrent` function, combined with the `async` block,
/// implicitly handles offloading this blocking operation to a thread pool.
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

lazy_static! {
    // Cache the number of CPUs and its f64 conversion as a static variable.
    static ref N_CPUS: usize = num_cpus::get();
    static ref N_CPUS_F: f64 = *N_CPUS as f64;
}

/// Compute the optimal concurrency level
/// Model: optimal concurrency = e^((1.6063) + (0.6350 * log(CPUs)) - (0.0909 * log((NumFiles + 1))))
fn compute_optimal_concurrency(num_files: usize) -> usize {
    let num_files_f = num_files as f64;

    // Compute the optimal concurrency using the cached N_CPUS_F.
    let optimal_concurrency = (1.6063 + 0.6350 * N_CPUS_F.ln() - 0.0909 * (num_files_f + 1.0).ln()).exp();

    // Round the result
    let candidate = optimal_concurrency.round() as usize;
    return candidate;
}

/// Main async entry point
async fn run_deletion(pattern: &str, concurrency_override: Option<usize>) -> io::Result<()> {
    // Split the provided pattern into a directory part and a filename pattern
    let (dir_path, file_pattern) = pattern.rsplit_once('/').unwrap_or((".", pattern));
    let dir = Path::new(dir_path);
    
    // Canonicalize the directory to get its absolute path for safety
    let canonical_dir = std::fs::canonicalize(dir)?;
    
    // Build a globset matcher for the filename pattern
    let mut gs_builder = globset::GlobSetBuilder::new();
    gs_builder.add(
        globset::Glob::new(file_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid glob pattern: {}", e)))?
    );
    let glob_set = gs_builder.build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("GlobSet build error: {}", e)))?;
    
    // Initial counting pass to determine the total number of matching files
    let mut total_files = 0;
    for entry in std::fs::read_dir(&canonical_dir)? {
        let entry = entry?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if glob_set.is_match(&*fname_str) && entry.file_type()?.is_file() {
            total_files += 1;
        }
    }
    if total_files == 0 {
        println!("No matching files found for pattern '{}'", pattern);
        return Ok(());
    }
    
    // Compute the concurrency level using the total file count and any override
    let concurrency = match concurrency_override {
        Some(n) => n,
        None => compute_optimal_concurrency(total_files),
    };
    println!(
        "[INFO] Deleting {} files with concurrency = {} (CPU cores = {})",
        total_files,
        concurrency,
        num_cpus::get()
    );
    
    // Set up shared state for deletion
    let completed_counter = Arc::new(AtomicUsize::new(0));
    const BATCH_SIZE: usize = 1000;
    
    // Initialize progress bar with the correct total count
    let config = Config {
        throttle_millis: 250,
        ..Default::default()
    };
    let pb = Arc::new(Bar::new(total_files as u64, config));
    
    // Clone shared state for use in the concurrent closure without moving the originals
    let pb_clone = Arc::clone(&pb);
    let completed_counter_clone = Arc::clone(&completed_counter);
    let canonical_dir_clone = canonical_dir.clone();
    let glob_set_clone = glob_set.clone();
    
    // Stream directory entries and delete matching files (second pass)
    let dir_entries = std::fs::read_dir(&canonical_dir)?;
    let file_stream = stream::iter(dir_entries.filter_map(Result::ok));
    
    file_stream.for_each_concurrent(Some(concurrency), move |entry| {
        // Clone shared state within each concurrent task from the clones above
        let pb = Arc::clone(&pb_clone);
        let completed_counter = Arc::clone(&completed_counter_clone);
        let canonical_dir = canonical_dir_clone.clone();
        let glob_set = glob_set_clone.clone();
        async move {
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            // Match using the globset matcher; dereference the Cow to &str
            if glob_set.is_match(&*fname_str) {
                match entry.file_type() {
                    Ok(ft) if ft.is_file() => {
                        // Construct the full path only when needed
                        let path = canonical_dir.join(&fname);
                        // Attempt to delete the file
                        match unlink_file(&path) {
                            Ok(_) => {
                                let completed_count = completed_counter.fetch_add(1, Ordering::Relaxed) + 1;
                                // Update progress bar in batches
                                if completed_count % BATCH_SIZE == 0 {
                                    pb.inc(BATCH_SIZE as u64);
                                }
                            }
                            Err(e) => {
                                eprintln!("Failed to delete '{}': {}", path.display(), e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Ok(_) => { /* Not a file; skip it */ }
                    Err(e) => {
                        eprintln!("Failed to get file type for '{}': {}", entry.path().display(), e);
                        std::process::exit(1);
                    }
                }
            }
        }
    }).await;
    
    // Finalize the progress bar for any remaining files in the last batch
    let remainder = completed_counter.load(Ordering::Relaxed) % BATCH_SIZE;
    if remainder > 0 {
        pb.inc(remainder as u64);
    }
    
    // Use try_unwrap to finish the progress bar; if unsuccessful, drop the Arc
    match Arc::try_unwrap(pb) {
        Ok(bar_inner) => { bar_inner.finish(); },
        Err(_) => {},
    }
    
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

    let result = runtime.block_on(run_deletion(pattern, None));

    match result {
        Ok(_) => println!("Files matching '{}' deleted successfully!", pattern),
        Err(e) => {
            eprintln!("Error during deletion: {}", e);
            std::process::exit(1);
        }
    }
}








// ====================================================================================================================================


// cargo test --release -- --nocapture performance_tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        time::Instant,
    };
    use tempfile::tempdir;
    use tokio::runtime::Builder;

    /// Creates files in `dir` using the standard naming scheme (e.g. "test_file_0.dat").
    fn create_test_files(dir: &Path, count: usize, size_kb: usize) -> Vec<PathBuf> {
        let mut paths = Vec::with_capacity(count);
        for i in 0..count {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut f = File::create(&file_path).unwrap();
            f.write_all(&vec![0u8; size_kb * 1024]).unwrap();
            paths.push(file_path);
        }
        paths
    }

    /// Converts a number to a minimal base-36 string.
    fn to_base36(mut num: usize) -> String {
        const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        if num == 0 {
            return "0".to_string();
        }
        let mut buf = Vec::new();
        while num > 0 {
            buf.push(DIGITS[num % 36]);
            num /= 36;
        }
        buf.reverse();
        String::from_utf8(buf).unwrap()
    }

    /// Creates files in `dir` using short names (e.g. "0", "1", "2", ...).
    fn create_short_test_files(dir: &Path, count: usize, size_kb: usize) -> Vec<PathBuf> {
        let mut paths = Vec::with_capacity(count);
        for i in 0..count {
            let filename = to_base36(i);
            let file_path = dir.join(filename);
            let mut f = File::create(&file_path).unwrap();
            f.write_all(&vec![0u8; size_kb * 1024]).unwrap();
            paths.push(file_path);
        }
        paths
    }

    /// Runs our Rust deletion routine for files matching `pattern` and returns the elapsed time (in seconds).
    /// After deletion, only files matching the pattern are checked.
    fn measure_rust_deletion(pattern: &str) -> f64 {
        let start = Instant::now();
        let rt = Builder::new_current_thread().enable_all().build().unwrap();
        let result = rt.block_on(run_deletion(pattern, None));
        let elapsed = start.elapsed().as_secs_f64();

        if let Err(e) = result {
            panic!("Error in Rust deletion: {}", e);
        }

        // Only count leftover files that match the pattern.
        let remaining: Vec<_> = glob::glob(pattern)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|p| p.is_file())
            .collect();
        if !remaining.is_empty() {
            panic!("Rust deletion failed: {} files remain", remaining.len());
        }
        elapsed
    }

    /// Runs the system deletion command to remove files matching `pattern` and returns a tuple:
    /// (elapsed time in seconds, the actual command that was executed).
    /// If `use_find` is true, deletion is performed using the `find` command.
    fn measure_rm_deletion(pattern: &str, use_find: bool) -> (f64, String) {
        let start = Instant::now();
        let elapsed: f64;
        let cmd: String;
        if use_find {
            let dir = Path::new(pattern)
                .parent()
                .expect("Could not determine parent directory")
                .to_string_lossy()
                .to_string();
            cmd = format!("find {} -maxdepth 1 -type f -delete", dir);
            println!("Executing system deletion command: {}", cmd);
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
                .expect("Failed to run find command");
            if !output.status.success() {
                eprintln!(
                    "find command stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            elapsed = start.elapsed().as_secs_f64();
        } else {
            cmd = format!("rm -f {}", pattern);
            println!("Executing system deletion command: {}", cmd);
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
                .expect("Failed to run rm command");
            if !output.status.success() {
                eprintln!(
                    "rm command stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            elapsed = start.elapsed().as_secs_f64();
        }

        // Verify that no matching files remain.
        let remaining: Vec<_> = glob::glob(pattern)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|p| p.is_file())
            .collect();
        if !remaining.is_empty() {
            panic!("System deletion failed: {} files remain", remaining.len());
        }
        (elapsed, cmd)
    }

    /// Formats a duration in seconds to a human-friendly string (ms if < 1 second).
    fn format_duration(seconds: f64) -> String {
        if seconds < 1.0 {
            format!("{:.3} ms", seconds * 1000.0)
        } else {
            format!("{:.3} s", seconds)
        }
    }

    /// Structure to hold benchmark results for one test scenario.
    struct BenchmarkResult {
        test_name: String,
        rust_time: f64,
        rm_time: f64,
        system_command: String,
    }

    /// Runs a benchmark scenario by creating files, measuring both Rust and system deletion times,
    /// and returning the results. The parameter `use_short_names` determines which naming scheme to use,
    /// and `use_find_for_rm` selects whether to use the `find` command for the system deletion test.
    fn run_benchmark(
        test_name: &str,
        file_count: usize,
        file_size_kb: usize,
        use_short_names: bool,
        use_find_for_rm: bool,
    ) -> BenchmarkResult {
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_path_buf();

        // Choose creation function and glob pattern.
        let (pattern_str, create_files_fn): (String, fn(&Path, usize, usize) -> Vec<PathBuf>) =
            if use_short_names {
                let pattern = base_path.join("[0-9a-z]*");
                (pattern.to_string_lossy().to_string(), create_short_test_files)
            } else {
                let pattern = base_path.join("test_file_*.dat");
                (pattern.to_string_lossy().to_string(), create_test_files)
            };

        println!("\n--- {} ---", test_name);
        println!(
            "Creating {} files ({} KB each)...",
            file_count, file_size_kb
        );
        // Create files for the Rust deletion test.
        create_files_fn(&base_path, file_count, file_size_kb);

        println!("Running Rust deletion...");
        let rust_time = measure_rust_deletion(&pattern_str);
        println!("Rust deletion completed in {}", format_duration(rust_time));

        // Recreate files for the system deletion test.
        create_files_fn(&base_path, file_count, file_size_kb);
        println!("Running system deletion (rm or find)...");
        let (rm_time, system_command) = measure_rm_deletion(&pattern_str, use_find_for_rm);
        println!("System deletion completed in {}", format_duration(rm_time));

        BenchmarkResult {
            test_name: test_name.to_string(),
            rust_time,
            rm_time,
            system_command,
        }
    }

    /// The main performance benchmark test that runs several scenarios and prints a summary.
    #[test]
    fn performance_summary() {
        println!("\n===== Starting Performance Benchmarks =====");

        let mut results = Vec::new();
        // Existing benchmarks:
        results.push(run_benchmark("One file (1 x 10 KB)", 10, 10, false, false));
        results.push(run_benchmark("Small files (10 x 1 KB)", 10, 1, false, false));
        results.push(run_benchmark("Some small files (30,000 x 1 KB)", 30_000, 1, false, false));
        results.push(run_benchmark("Many small files (100,000 x 1 KB)", 100_000, 1, true, true));
        results.push(run_benchmark(
            "A ton of small files (500,000 x 1 KB)",
            500_000,
            1,
            true,
            true,
        ));
        results.push(run_benchmark("Large files (100 x 10 MB)", 100, 10240, false, false));
        results.push(run_benchmark("Medium files (2000 x 100 KB)", 50, 100, false, false));
        results.push(run_benchmark("Huge files (10 x 50 MB)", 10, 51200, false, false));

        println!("\n===== Performance Summary =====\n");
        println!(
            "{:<40} | {:>10} | {:>10} | {:>10} | {:<30}",
            "Test Scenario", "Rust", "System", "Diff", "System Command"
        );
        println!("{}", "-".repeat(110));

        for res in results {
            let diff = (res.rust_time - res.rm_time).abs();
            let winner = if res.rust_time < res.rm_time { "Rust" } else { "System" };
            println!(
                "{:<40} | {:>10} | {:>10} | {:>10} | {:<30}",
                res.test_name,
                format_duration(res.rust_time),
                format_duration(res.rm_time),
                format_duration(diff),
                res.system_command
            );
            println!("Winner: {}\n", winner);
        }
        println!("{}", "-".repeat(110));
        println!("Note: Times are measured in seconds (or ms if < 1 second).");
    }

    /// Test to verify that attempting to delete a non-matching pattern does not error.
    #[test]
    fn test_delete_with_no_matches() {
        println!("\n--- Test: Deletion with No Matches ---");
        let tmp = tempdir().unwrap();
        let base_path = tmp.path().to_string_lossy().to_string();
        let pattern = format!("{}/no_such_file_*.dat", base_path);

        let rust_time = measure_rust_deletion(&pattern);
        println!(
            "Rust deletion with no matches completed in {}.",
            format_duration(rust_time)
        );
    }

    /// Test to make sure that directories are not removed during deletion.
    #[test]
    fn test_skips_directories() {
        println!("\n--- Test: Skipping Directories ---");
        let tmp = tempdir().unwrap();
        let dir_path = tmp.path().join("my_test_dir");
        fs::create_dir(&dir_path).unwrap();

        // Create a file that matches the pattern.
        let file_path = tmp.path().join("my_test_dir_file.dat");
        File::create(&file_path).unwrap();

        let pattern = tmp.path().join("my_test_dir*");
        let pattern_str = pattern.to_string_lossy().to_string();

        let rust_time = measure_rust_deletion(&pattern_str);
        println!(
            "Rust deletion (skipping directories) completed in {}.",
            format_duration(rust_time)
        );

        // Verify that the directory still exists and the file has been deleted.
        assert!(dir_path.is_dir(), "Directory was deleted!");
        assert!(!file_path.exists(), "File was not deleted!");
    }

    /// Test when the deletion pattern only matches some of the files in the directory.
    /// Matching files should be deleted while non-matching files remain. Non-matching files are cleaned up after timing.
    #[test]
    fn test_partial_match_deletion() {
        println!("\n--- Test: Partial Pattern Deletion ---");
        let tmp = tempdir().unwrap();
        let base = tmp.path();

        // Create files that match the pattern.
        let matching_files: Vec<PathBuf> = (0..5)
            .map(|i| {
                let path = base.join(format!("match_{}.dat", i));
                let mut f = File::create(&path).unwrap();
                f.write_all(b"test").unwrap();
                path
            })
            .collect();

        // Create files that do NOT match the deletion pattern.
        let non_matching_files: Vec<PathBuf> = (0..3)
            .map(|i| {
                let path = base.join(format!("nomatch_{}.dat", i));
                let mut f = File::create(&path).unwrap();
                f.write_all(b"test").unwrap();
                path
            })
            .collect();

        // Use a pattern that only matches the 'match_' files.
        let pattern = base.join("match_*.dat");
        let pattern_str = pattern.to_string_lossy().to_string();

        println!("Running Rust deletion on partial match pattern...");
        let rust_time = measure_rust_deletion(&pattern_str);
        println!(
            "Rust deletion (partial match) completed in {}.",
            format_duration(rust_time)
        );

        // Verify that matching files have been deleted.
        for path in matching_files {
            assert!(
                !path.exists(),
                "Matching file {} was not deleted",
                path.display()
            );
        }

        // Verify that non-matching files still exist, then clean them up.
        for path in non_matching_files {
            assert!(
                path.exists(),
                "Non-matching file {} was mistakenly deleted",
                path.display()
            );
            fs::remove_file(&path).unwrap();
        }
    }
}








#[cfg(test)]
mod empirical_tests {
    use super::*;
    use std::fs::{File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::{Instant, Duration};
    use tempfile::tempdir;
    use std::iter;

    const TEST_FILE_SIZE_KB: usize = 1;
    const TEST_FILE_COUNT: usize = 100;
    const ITERATIONS: usize = 100;

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

    fn create_files_for_measurement(dir: &Path) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for i in 0..TEST_FILE_COUNT {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&vec![0u8; TEST_FILE_SIZE_KB * 1024]).unwrap();
            paths.push(file_path);
        }
        paths
    }

    fn measure_path_conversion_time_inner() -> f64 {
        const ITERS: usize = 10_000;
        let path_str: String = format!("/tmp/long_path_for_conversion/dir1/dir2/file_{}.dat",  iter::repeat(0).take(20).map(|_| "A".to_string()).collect::<String>());
        let mut total_ns = 0u128;

        for _ in 0..ITERS {
            let start = Instant::now();
            let _ = CString::new(&path_str[..]).expect("CString conversion failed");
            total_ns += start.elapsed().as_nanos();
        }

        total_ns as f64 / ITERS as f64
    }

    #[test]
    fn measure_path_conversion_time() {
        let avg_ns = measure_path_conversion_time_inner();
        println!("[Empirical] T_conversion (Path Conversion Time) = {:.2} ns/path", avg_ns);
        assert!(avg_ns > 0.0);
    }

    fn measure_syscall_time_inner(files: &Vec<PathBuf>) -> f64 {
        let mut total_ns = 0u128;
        for path in files {
            let start = Instant::now();
            unlink_file(path).expect("Failed to unlink in test");
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / files.len() as f64
    }

    #[test]
    fn measure_syscall_time() {
        let tmp_dir = tempdir().unwrap();
        let base_files = create_files_for_measurement(tmp_dir.path());
        let avg_syscall_ns = measure_syscall_time_inner(&base_files);
        println!("[Empirical] T_syscall (System Call Time) = {:.2} ns/file", avg_syscall_ns);
        assert!(avg_syscall_ns > 0.0);
    }

    fn measure_task_overhead_inner() -> f64 {
        const TASKS: usize = 100;
        const CONCURRENCY_LEVEL: usize = 4;
        let mut total_ns = 0u128;
        for _ in 0..TASKS {
            let start = Instant::now();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let futures: Vec<_> = (0..CONCURRENCY_LEVEL)
                    .map(|_| {
                        tokio::spawn(async {
                            tokio::task::yield_now().await;
                        })
                    })
                    .collect();
                futures::future::join_all(futures).await;
            });
            total_ns += start.elapsed().as_nanos();
        }
        total_ns as f64 / TASKS as f64 / CONCURRENCY_LEVEL as f64
    }


    #[test]
    fn measure_task_overhead() {
        let avg_overhead_ns = measure_task_overhead_inner();
        println!("[Empirical] T_overhead (Task Management Overhead) = {:.2} ns/task", avg_overhead_ns);
        assert!(avg_overhead_ns > 0.0);
    }

    fn measure_concurrent_deletion_time(n: usize, files: Vec<PathBuf>) -> Duration {
        let start = Instant::now();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
    
        runtime.block_on(async {
            futures::stream::iter(files)
                .for_each_concurrent(Some(n), |path| async move {
                    if let Err(e) = unlink_file(&path) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            panic!("Failed to delete '{}': {}", path.display(), e);
                        }
                    }
                })
                .await;
        });
        start.elapsed()
    }


    /// Measure I/O Efficiency f_disk(n) and return the data
    fn measure_io_efficiency() -> Vec<(f64, f64)> {
        let tmp_dir = tempdir().unwrap();
        let base_files = create_files_for_measurement(tmp_dir.path());
        let concurrency_levels = vec![1, 2, 4, 8, 16, 32, 64, 128];

        let avg_conversion_ns = measure_path_conversion_time_inner();
        let avg_syscall_ns = measure_syscall_time_inner(&base_files);
        let avg_overhead_ns = measure_task_overhead_inner();

        println!("\n[Empirical] I/O Efficiency f_disk(n) at different concurrency levels:");
        println!("--- Empirical Constants (for f_disk calculation) ---");
        println!("T_conversion = {:.2} ns/path", avg_conversion_ns);
        println!("T_syscall = {:.2} ns/file", avg_syscall_ns);
        println!("T_overhead = {:.2} ns/task", avg_overhead_ns);
        println!("--- f_disk(n) values ---");

        let n_files = base_files.len();
        let mut f_disk_values = Vec::new();

        for &concurrency in &concurrency_levels {
            let mut total_deletion_time = Duration::new(0, 0);
            for _ in 0..ITERATIONS {
                let files = create_files_for_measurement(tmp_dir.path());
                total_deletion_time += measure_concurrent_deletion_time(concurrency, files);
            }
            let avg_total_deletion_time_ns = total_deletion_time.as_nanos() / ITERATIONS as u128;

            let t_syscall_actual_ns = (avg_total_deletion_time_ns as f64) - (n_files as f64 * avg_conversion_ns) - (concurrency as f64 * avg_overhead_ns);
            let t_syscall_ideal_ns = (n_files as f64 * avg_syscall_ns) / concurrency as f64;

            let f_disk_n = if t_syscall_actual_ns > 0.0 {
                t_syscall_ideal_ns / t_syscall_actual_ns
            } else {
                1.0
            };

            println!("f_disk({}) = {:.6}", concurrency, f_disk_n);
            f_disk_values.push((concurrency as f64, f_disk_n)); // Store for curve fitting
        }
    f_disk_values
    }


    #[test]
    fn test_fit_f_disk_function() {
        // Obtain empirical data: a vector of (n, f_disk) measurements.
        let data = measure_io_efficiency();
        if data.is_empty() {
            panic!("No empirical I/O efficiency data available.");
        }
    
        // Separate the data into two vectors:
        let n_values: Vec<f64> = data.iter().map(|&(n, _)| n).collect();
        let f_values: Vec<f64> = data.iter().map(|&(_, f)| f).collect();

        // Define the cost function (sum of squared errors).
        // For each measured data point (n, f_emp), the model predicts:
        //    f_pred = 1 / (1 + a * n^b)
        // and we accumulate (f_pred - f_emp)^2.
        let cost = |a: f64, b: f64| -> f64 {
            n_values.iter()
                .zip(f_values.iter())
                .fold(0.0, |acc, (&n, &f_emp)| {
                    let f_pred = 1.0 / (1.0 + a * n.powf(b));
                    acc + (f_pred - f_emp).powi(2)
                })
        };
    
        // Compute numerical gradients using central differences.
        let gradient = |a: f64, b: f64| -> (f64, f64) {
            let eps = 1e-6;
            let cost_a_plus = cost(a + eps, b);
            let cost_a_minus = cost(a - eps, b);
            let grad_a = (cost_a_plus - cost_a_minus) / (2.0 * eps);
            let cost_b_plus = cost(a, b + eps);
            let cost_b_minus = cost(a, b - eps);
            let grad_b = (cost_b_plus - cost_b_minus) / (2.0 * eps);
            (grad_a, grad_b)
        };
    
        // --- Optimization Settings ---
        let max_iters = 1000;
        let tol = 1e-9;
        let learning_rate = 1e-3; // initial learning rate
    
        // Initial guesses for a and b.
        let mut a = 0.01;
        let mut b = 1.0;
    
        let current_cost = cost(a, b);
        println!("Initial cost: {:.12}", current_cost);
    
        // Begin gradient descent loop.
        for iter in 0..max_iters {
            // Compute the gradient.
            let (grad_a, grad_b) = gradient(a, b);
            let grad_norm = (grad_a.powi(2) + grad_b.powi(2)).sqrt();
    
            // If the gradient is very small, assume convergence.
            if grad_norm < tol {
                println!("Convergence reached at iteration {} (grad_norm = {:.12}).", iter, grad_norm);
                break;
            }
    
            // Backtracking line search: try reducing the step if cost does not decrease.
            let mut step = learning_rate;
            let mut new_a = a - step * grad_a;
            let mut new_b = b - step * grad_b;
            // Project parameters to remain positive.
            new_a = new_a.max(1e-6);
            new_b = new_b.max(1e-6);
            let mut new_cost = cost(new_a, new_b);
    
            // Reduce the step size until the cost decreases.
            while new_cost > current_cost && step > 1e-12 {
                step *= 0.5;
                new_a = a - step * grad_a;
                new_b = b - step * grad_b;
                new_a = new_a.max(1e-6);
                new_b = new_b.max(1e-6);
                new_cost = cost(new_a, new_b);
            }
    
            // If no improvement is possible, break.
            if new_cost >= current_cost {
                println!("No improvement found at iteration {} (cost: {:.12}).", iter, current_cost);
                break;
            }
    
            // Accept the new parameters.
            a = new_a;
            b = new_b;
    
            // Check for convergence based on cost improvement.
            if (current_cost - new_cost).abs() < tol {
                println!("Cost improvement below tolerance at iteration {}.", iter);
                break;
            }
        }
    
        println!("\n--- Fitted Parameters for f_disk(n) = 1/(1 + a * n^b) ---");
        println!("F_DISK_A_FIT = {:.8}", a);
        println!("F_DISK_B_FIT = {:.8}", b);
    
        assert!(a > 0.0, "Fitted parameter 'a' should be positive; got {:.8}", a);
        assert!(b > 0.0, "Fitted parameter 'b' should be positive; got {:.8}", b);
    }

}


// cargo test --release -- --nocapture test_grid
#[cfg(test)]
mod test_grid {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use std::fs::OpenOptions;
    use std::io::BufWriter;

    const TEST_FILE_SIZE_KB: usize = 1;
    const CSV_FILE_NAME: &str = "test_results.csv";

    /// Creates a set of test files in the given directory.
    fn create_test_files(dir: &Path, count: usize) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for i in 0..count {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&vec![0u8; TEST_FILE_SIZE_KB * 1024]).unwrap();
            paths.push(file_path);
        }
        paths
    }

    /// Generates logarithmically spaced values between 1 and max_val, including max_val.
    fn generate_log_space(max_val: usize, num_points: usize) -> Vec<usize> {
        let max_val_f64 = max_val as f64;
        let log_max = max_val_f64.log10();
        let mut values = Vec::new();
        if num_points > 1 {
            for i in 0..(num_points - 1) {
                let t = i as f64 / (num_points as f64 - 1.0);
                let log_val = t * log_max;
                let val = 10f64.powf(log_val).round() as usize;
                values.push(val);
            }
        }
        values.push(max_val);
        values.dedup();
        values
    }

    #[test]
    fn test_grid_search() {
        let actual_cpus = num_cpus::get();
        let simulated_cpu_counts = generate_log_space(actual_cpus, 32);
        let file_counts = [0, 8, 40, 80, 400, 2000];
        let max_concurrency_multiplier = 4;
        println!("\n[Grid Search Test] Running...");
        let mut plot_data: Vec<(f64, f64, f64)> = Vec::new();
        let csv_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(CSV_FILE_NAME)
            .expect("Failed to open CSV file");
        let mut csv_writer = BufWriter::new(csv_file);
        if csv_writer.get_ref().metadata().unwrap().len() == 0 {
            writeln!(csv_writer, "SimulatedCPUs,NumFiles,Concurrency,TotalTime(ns)").expect("Failed to write CSV header");
        }
        for &simulated_cpus in &simulated_cpu_counts {
            let num_threads = simulated_cpus.min(actual_cpus);
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(num_threads)
                .build()
                .unwrap();
            for &num_files in &file_counts {
                let tmp_dir = tempdir().unwrap();
                let pattern = format!("{}/*", tmp_dir.path().to_string_lossy());
                let mut min_time = Duration::MAX;
                let mut optimal_concurrency = 1;

                // Generate log-spaced concurrency levels from 1 to max_concurrency
                let max_concurrency = max_concurrency_multiplier * simulated_cpus;
                let concurrency_levels = generate_log_space(max_concurrency, 32);

                for concurrency in concurrency_levels {
                    create_test_files(tmp_dir.path(), num_files);
                    let start = Instant::now();
                    runtime.block_on(run_deletion(&pattern, Some(concurrency))).unwrap();
                    let elapsed = start.elapsed();
                    println!(
                        "  Simulated CPUs: {}, Files: {}, Concurrency: {}, Time: {:?}",
                        simulated_cpus, num_files, concurrency, elapsed
                    );
                    writeln!(
                        csv_writer,
                        "{},{},{},{}",
                        simulated_cpus,
                        num_files,
                        concurrency,
                        elapsed.as_nanos()
                    ).expect("Failed to write to CSV");
                    if elapsed < min_time {
                        min_time = elapsed;
                        optimal_concurrency = concurrency;
                    }
                }
                csv_writer.flush().expect("Failed to flush CSV writer");
                plot_data.push((
                    simulated_cpus as f64,
                    optimal_concurrency as f64,
                    num_files as f64,
                ));
                println!(
                    "[Grid Search Test] Optimal Concurrency (Simulated CPUs: {}, Files: {}): {}",
                    simulated_cpus, num_files, optimal_concurrency
                );
            }
        }
        println!("[Grid Search Test] Complete.  See {}", CSV_FILE_NAME);
    }
}




// cargo test --release -- --nocapture file_count_tests
#[cfg(test)]
mod file_count_tests {
    use std::ffi::{CStr, CString};
    use std::fs::{self, File};
    use std::io::Write;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;
    use std::time::{Duration, Instant};

    use scandir::{Count, Walk, Scandir};
    use tempfile::tempdir;
    use walkdir::WalkDir;
    use rayon::prelude::*;
    use libc;

    // Declare an external binding for scandir from the C library.
    extern "C" {
        pub fn scandir(
            dir: *const libc::c_char,
            namelist: *mut *mut *mut libc::dirent,
            filter: Option<unsafe extern "C" fn(*const libc::dirent) -> libc::c_int>,
            compar: Option<unsafe extern "C" fn(*const *const libc::dirent, *const *const libc::dirent) -> libc::c_int>,
        ) -> libc::c_int;
    }

    // This helper is used by the scandir-based method.
    // It returns 1 if the entry is a regular file and not "." or "..".
    unsafe extern "C" fn file_filter(entry: *const libc::dirent) -> libc::c_int {
        if entry.is_null() {
            return 0;
        }
        let d_name_ptr = (*entry).d_name.as_ptr();
        if d_name_ptr.is_null() {
            return 0;
        }
        let name = CStr::from_ptr(d_name_ptr);
        let bytes = name.to_bytes();
        if bytes == b"." || bytes == b".." {
            return 0;
        }
        if (*entry).d_type == libc::DT_REG {
            return 1;
        }
        0
    }

    // Uses std::fs::read_dir without any extra filtering.
    fn count_using_read_dir(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let count = fs::read_dir(path)
            .unwrap()
            .filter_map(Result::ok)
            .count();
        (count, start.elapsed())
    }

    // Uses std::fs::read_dir with an explicit filter checking the file type.
    fn count_using_read_dir_with_filter(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let count = fs::read_dir(path)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
            })
            .count();
        (count, start.elapsed())
    }

    // Uses the WalkDir crate (restricted to a depth of 1).
    fn count_using_walkdir(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let count = WalkDir::new(path)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|entry| entry.file_type().is_file())
            .count();
        (count, start.elapsed())
    }

    // Uses Rayon to perform parallel iteration over the entries.
    fn count_using_rayon(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let count = fs::read_dir(path)
            .unwrap()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
            })
            .count();
        (count, start.elapsed())
    }

    // Uses direct libc calls: opendir() and readdir().
    fn count_using_opendir_readdir(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        unsafe {
            let dirp = libc::opendir(c_path.as_ptr());
            if dirp.is_null() {
                panic!("opendir failed");
            }
            let mut count = 0;
            loop {
                let entry = libc::readdir(dirp);
                if entry.is_null() {
                    break;
                }
                let name_ptr = (*entry).d_name.as_ptr();
                if name_ptr.is_null() {
                    continue;
                }
                let name = CStr::from_ptr(name_ptr).to_string_lossy();
                if name == "." || name == ".." {
                    continue;
                }
                if (*entry).d_type == libc::DT_REG {
                    count += 1;
                }
            }
            libc::closedir(dirp);
            (count, start.elapsed())
        }
    }

    // Uses the C library's scandir() function.
    fn count_using_scandir(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        unsafe {
            let mut namelist: *mut *mut libc::dirent = std::ptr::null_mut();
            let count = scandir(c_path.as_ptr(), &mut namelist, Some(file_filter), None);
            if count < 0 {
                panic!("scandir failed");
            }
            for i in 0..count {
                let entry = *namelist.add(i as usize);
                libc::free(entry as *mut libc::c_void);
            }
            libc::free(namelist as *mut libc::c_void);
            (count as usize, start.elapsed())
        }
    }

    // Uses the raw getdents64 syscall.
    fn count_using_getdents64(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        unsafe {
            let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY);
            if fd < 0 {
                panic!("Failed to open directory");
            }
            let buf_size = 8192;
            let mut buf = vec![0u8; buf_size];
            let mut count = 0;
            loop {
                let nread = libc::syscall(
                    libc::SYS_getdents64,
                    fd,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf_size,
                );
                if nread == -1 {
                    libc::close(fd);
                    panic!("getdents64 failed");
                }
                if nread == 0 {
                    break;
                }
                let mut bpos = 0;
                while bpos < nread as usize {
                    let d = buf.as_ptr().add(bpos) as *const libc::dirent64;
                    let reclen = (*d).d_reclen as usize;
                    let name_ptr = (*d).d_name.as_ptr();
                    let name = CStr::from_ptr(name_ptr);
                    if name.to_bytes() != b"." && name.to_bytes() != b".." {
                        if (*d).d_type == libc::DT_REG {
                            count += 1;
                        }
                    }
                    bpos += reclen;
                }
            }
            libc::close(fd);
            (count, start.elapsed())
        }
    }

    // Test using the Count API (for obtaining directory statistics).
    #[test]
    fn test_scandir_count_api() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory and populate it with 100 files.
        let tmp_dir = tempdir()?;
        let dir_path = tmp_dir.path();
        let num_files = 100;
        for i in 0..num_files {
            let file_path = dir_path.join(format!("file_{}.txt", i));
            fs::write(file_path, "test")?;
        }

        // Create a Count instance and collect the file statistics.
        let stats = Count::new(dir_path)?.collect()?;
        assert_eq!(stats.files, num_files, "Count API did not report the expected number of files");
        Ok(())
    }

    // Test using the Walk API (for obtaining a basic file tree).
    #[test]
    fn test_scandir_walk_api() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory and populate it with 100 files.
        let tmp_dir = tempdir()?;
        let dir_path = tmp_dir.path();
        let num_files = 100;
        for i in 0..num_files {
            let file_path = dir_path.join(format!("file_{}.txt", i));
            fs::write(file_path, "test")?;
        }

        // Create a Walk instance and collect the entries.
        let entries = Walk::new(dir_path, None)?.collect()?;
        // Walk typically returns only the files in the root (when not recursing),
        // so we expect exactly `num_files` entries.
        assert_eq!(entries.files.len(), num_files, "Walk API did not yield the expected number of file entries");
        Ok(())
    }

    // Test using the Scandir API (for obtaining detailed file metadata).
    #[test]
    fn test_scandir_scandir_api() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory and populate it with 100 files.
        let tmp_dir = tempdir()?;
        let dir_path = tmp_dir.path();
        let num_files = 100;
        for i in 0..num_files {
            let file_path = dir_path.join(format!("file_{}.txt", i));
            fs::write(file_path, "test")?;
        }

        // Create a Scandir instance and collect the detailed entries.
        let entries = Scandir::new(dir_path, None)?.collect()?;
        // Filter only entries that are files (in case directories are also returned).
        let file_count = entries.results.iter().filter(|entry| match entry {
            &scandir::ScandirResult::DirEntry(ref de) => de.is_file,
            &scandir::ScandirResult::DirEntryExt(ref de) => de.is_file,
            &scandir::ScandirResult::Error(_) => false,
        }).count();
        assert_eq!(file_count, num_files, "Scandir API did not return the expected number of file entries");
        Ok(())
    }

        // Extremely optimized file counting using Rust std::fs::read_dir.
    fn count_using_read_dir_fast(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let count = fs::read_dir(path)
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| {
                // Directly check if the entry is a file.
                entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
            })
            .count();
        (count, start.elapsed())
    }

    // Extremely optimized file counting using the raw getdents64 syscall with a 1MB buffer.
    fn count_using_getdents64_fast(path: &Path) -> (usize, Duration) {
        let start = Instant::now();
        let c_path = CString::new(path.as_os_str().as_bytes()).unwrap();
        unsafe {
            let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY);
            if fd < 0 {
                panic!("Failed to open directory");
            }
            let buf_size = 1 << 20; // 1MB buffer
            let mut buf = vec![0u8; buf_size];
            let mut count = 0;
            loop {
                let nread = libc::syscall(
                    libc::SYS_getdents64,
                    fd,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf_size,
                );
                if nread < 0 {
                    libc::close(fd);
                    panic!("getdents64 failed");
                }
                if nread == 0 {
                    break;
                }
                let mut bpos = 0;
                while bpos < nread as usize {
                    let d = buf.as_ptr().add(bpos) as *const libc::dirent64;
                    let reclen = (*d).d_reclen as usize;
                    let name_ptr = (*d).d_name.as_ptr();
                    // Fast manual check for "." and ".." without creating a CStr.
                    if *name_ptr != b'.' as i8 {
                        if (*d).d_type == libc::DT_REG {
                            count += 1;
                        }
                    } else {
                        let second = *name_ptr.add(1);
                        if second == 0 {
                            // It's ".", skip.
                        } else if second == b'.' as i8 && *name_ptr.add(2) == 0 {
                            // It's "..", skip.
                        } else if (*d).d_type == libc::DT_REG {
                            count += 1;
                        }
                    }
                    bpos += reclen;
                }
            }
            libc::close(fd);
            (count, start.elapsed())
        }
    }

    #[test]
    fn test_file_count_methods() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory.
        let tmp_dir = tempdir().unwrap();
        let dir_path = tmp_dir.path();

        // Create 100,000 files.
        let num_files = 100_000;
        for i in 0..num_files {
            let file_path = dir_path.join(format!("file_{}.dat", i));
            let mut file = File::create(file_path).unwrap();
            // Write minimal content.
            writeln!(file, "test").unwrap();
        }

        // Run each counting method, record its count and elapsed time.
        let mut results = Vec::new();

        let (c, t) = count_using_read_dir(dir_path);
        results.push(("std::fs::read_dir (unfiltered)", c, t));

        let (c, t) = count_using_read_dir_with_filter(dir_path);
        results.push(("std::fs::read_dir (with file_type filter)", c, t));

        let (c, t) = count_using_walkdir(dir_path);
        results.push(("WalkDir crate", c, t));

        let (c, t) = count_using_rayon(dir_path);
        results.push(("Rayon parallel iteration", c, t));

        let (c, t) = count_using_opendir_readdir(dir_path);
        results.push(("libc opendir/readdir", c, t));

        let (c, t) = count_using_scandir(dir_path);
        results.push(("libc scandir", c, t));

        let (c, t) = count_using_getdents64(dir_path);
        results.push(("Raw getdents64 syscall", c, t));

        let (c, t) = count_using_read_dir_fast(dir_path);
        results.push(("std::fs::read_dir fast", c, t));

        let (c, t) = count_using_getdents64_fast(dir_path);
        results.push(("Raw getdents64 fast", c, t));
        
        // Using scandir-rs's Count API:
        {
            let start = Instant::now();
            let stats = Count::new(dir_path)?.collect()?;
            let duration = start.elapsed();
            results.push(("scandir Count API", stats.files.try_into().unwrap(), duration));
        }
        
        // Using scandir-rs's Walk API:
        {
            let start = Instant::now();
            let entries = Walk::new(dir_path, None)?.collect()?;
            let count = entries.files.len();
            let duration = start.elapsed();
            results.push(("scandir Walk API", count, duration));
        }
        
        // Using scandir-rs's Scandir API:
        {
            let start = Instant::now();
            let entries = Scandir::new(dir_path, None)?.collect()?;
            let count = entries.results.iter().filter(|entry| match entry {
                &scandir::ScandirResult::DirEntry(ref de) => de.is_file,
                &scandir::ScandirResult::DirEntryExt(ref de) => de.is_file,
                &scandir::ScandirResult::Error(_) => false,
            }).count();
            let duration = start.elapsed();
            results.push(("scandir Scandir API", count, duration));
        }


        println!("File counting results (expected count: {}):", num_files);
        for (desc, count, duration) in results {
            println!(
                "{} -> Count: {}, Time: {:?} ({} µs)",
                desc,
                count,
                duration,
                duration.as_micros()
            );
            assert_eq!(
                count, num_files,
                "{} did not count {} files correctly",
                desc, num_files
            );
        }
        Ok(())
    }
}

// cargo test --release -- --nocapture
