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
use num_cpus;
use tokio::runtime::Builder;
use std::os::unix::ffi::OsStrExt;
use lazy_static::lazy_static;
use progression::{Bar, Config};

/// Delete a single file via a direct `unlink` (libc) call.

/// The `libc::unlink` function is a blocking system call.  In an asynchronous environment like Tokio,
/// executing blocking operations directly on the main async runtime threads is highly detrimental to performance.
/// Blocking a runtime thread will prevent it from processing other asynchronous tasks, potentially leading to
/// thread starvation, increased latency, and reduced overall throughput of the application.
///
/// To mitigate this, we utilize `tokio::task::spawn_blocking`. This function offloads the execution of
/// the provided closure (which contains the `unlink` call) to a dedicated thread pool managed by Tokio,
/// separate from the core async runtime's worker threads.
///
/// By using `spawn_blocking`, we ensure that the blocking `unlink` operation does not impede the progress of
/// other asynchronous tasks running on the main Tokio runtime.  This is crucial for maintaining responsiveness
/// and maximizing the efficiency of concurrent file deletions in an async context.
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

/// Compute the optimal concurrency level.
/// Model: optimal concurrency = e^((1.6063) + (0.6350 * log(CPUs)) - (0.0909 * log((NumFiles + 1))))
fn compute_optimal_concurrency(num_files: usize) -> usize {
    let num_files_f = num_files as f64;

    // Compute the optimal concurrency using the cached N_CPUS_F.
    let optimal_concurrency = (1.6063 + 0.6350 * N_CPUS_F.ln() - 0.0909 * (num_files_f + 1.0).ln()).exp();

    // Round the result and clamp it between 1 and the number of CPU cores.
    let candidate = optimal_concurrency.round() as usize;
    candidate.clamp(1, *N_CPUS)
}

/// Main async entry point.
async fn run_deletion(pattern: &str, concurrency_override: Option<usize>) -> io::Result<()> {
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

    // Compute optimal concurrency, or use override
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

    // Set up progress bar using progression (updates throttled to 250 ms max)
    let config = progression::Config {
        throttle_millis: 250,
        ..Default::default()
    };
    let pb = progression::Bar::new(total_files as u64, config);

    // Delete files with controlled concurrency
    let completed_counter = Arc::new(AtomicUsize::new(0));
    let file_stream = stream::iter(files.into_iter());

    file_stream
        .for_each_concurrent(Some(concurrency), |path| {
            let pb = pb.clone();
            let completed_counter = completed_counter.clone();
            async move {

                    //   1. Atomic Counter Operations:  Incrementing `completed_counter` (even though atomic) still involves
                    //      some overhead, especially with frequent updates.
                    //   2. Progress Bar I/O and Rendering: Updating the `indicatif::ProgressBar` involves I/O operations
                    //      to the terminal and re-rendering the progress bar display. Frequent I/O can be relatively slow.
                    //
                    // To reduce this overhead, we should implement batched progress bar updates. Instead of updating
                    // after every deletion, we can update the progress bar less frequently.
                match unlink_file(&path) {
                    Ok(_) => {
                        completed_counter.fetch_add(1, Ordering::Relaxed);
                        pb.inc(1);
                    }
                    Err(e) => {
                        eprintln!("Failed to delete '{}': {}", path.display(), e);
                        // Still count errors towards completion for accurate progress.
                        completed_counter.fetch_add(1, Ordering::Relaxed);
                        pb.inc(1);
                    }
                }
            }
        })
        .await;

    pb.finish();
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



#[cfg(test)]
mod performance_tests {
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
        let result = rt.block_on(run_deletion(pattern, None));
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
    
        // Sanity checks: ensure both parameters are positive.
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
    use rand::{Rng, thread_rng};
    use plotters::prelude::*;
    use plotters::style::colors::{BLACK, WHITE};
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
        create_3d_plot(&plot_data).expect("Failed to create 3D plot");
        println!("[Grid Search Test] Complete.  See optimal_concurrency.png and {}", CSV_FILE_NAME);
    }

    /// Creates a 3D plot of the optimal concurrency data.
    fn create_3d_plot(data: &[(f64, f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
        let root = BitMapBackend::new("optimal_concurrency.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE)?;
        let max_cpus = data.iter().map(|&(x, _, _)| x).fold(0.0, f64::max);
        let max_concurrency = data.iter().map(|&(_, y, _)| y).fold(0.0, f64::max);
        let max_files = data.iter().map(|&(_, _, z)| z).fold(0.0, f64::max);
        let mut chart = ChartBuilder::on(&root)
            .margin(5)
            .x_label_area_size(60)
            .y_label_area_size(60)
            .build_cartesian_3d(0.0..max_cpus, 0.0..max_concurrency, 0.0..max_files)?;
        chart.configure_axes().draw()?;
        let label_style = ("sans-serif", 20).into_font();
        chart
            .draw_series(std::iter::once(Text::new(
                "Simulated CPUs",
                (max_cpus + 5.0, 0.0, 0.0),
                label_style.clone(),
            )))?
            .label("Simulated CPUs")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
        chart
            .draw_series(std::iter::once(Text::new(
                "Optimal Concurrency",
                (0.0, max_concurrency + 2.5, 0.0),
                label_style.clone().transform(FontTransform::Rotate90),
            )))?
            .label("Optimal Concurrency")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x, y + 20)], &BLACK));
        chart
            .draw_series(std::iter::once(Text::new(
                "Number of Files",
                (0.0, -7.0, max_files + 1000.0),
                label_style.clone().transform(FontTransform::Rotate270),
            )))?
            .label("Number of Files")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));
        chart.draw_series(data.iter().map(|&(x, y, z)| {
            let style = plotters::style::ShapeStyle {
                color: plotters::style::RGBColor(
                    (255.0 * y / max_concurrency) as u8,
                    0,
                    (255.0 * (1.0 - y / max_concurrency)) as u8,
                ).into(),
                filled: true,
                stroke_width: 1,
            };
            Cross::new((x, y, z), 5, style)
        }))?;
        Ok(())
    }
}

// cargo test --release -- --nocapture
