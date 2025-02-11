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
const T_CONVERSION_NS: f64 = 116.56;
const T_SYSCALL_NS: f64 = 113356.65;
const T_OVERHEAD_NS: f64 = 16538.87;
const F_DISK_A_FIT: f64 = 0.31238476;
const F_DISK_B_FIT: f64 = 1.20621366;
// ===========================


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

#[cfg(test)]
mod test_prediction {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;
    use tempfile::tempdir;
    use rand::Rng;
    use plotters::prelude::*;
    use statrs::statistics::Statistics;
    use statrs::statistics::Correlation;

    const TEST_FILE_SIZE_KB: usize = 1;
    const NUM_TEST_FILES: usize = 100;
    const NUMBER_OF_SCENARIOS: usize = 500;

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
                        if e.kind() != std::io::ErrorKind::NotFound{
                            panic!("Failed to delete '{}': {}", path.display(), e);
                        }
                    }
                })
                .await;
        });
        start.elapsed()
    }

    /// Fitted disk I/O efficiency function: f_disk(n) = 1 / (1 + a * n^b)
    fn f_disk(n: usize) -> f64 {
        1.0 / (1.0 + F_DISK_A_FIT * (n as f64).powf(F_DISK_B_FIT))
    }


    #[test]
    fn test_model_prediction_accuracy() {

        let mut predicted_times: Vec<f64> = Vec::with_capacity(NUMBER_OF_SCENARIOS);
        let mut actual_times: Vec<f64> = Vec::with_capacity(NUMBER_OF_SCENARIOS);

        let tmp_dir = tempdir().unwrap();

        println!("\n[Model Accuracy Test] Running {} Scenarios...", NUMBER_OF_SCENARIOS);

        for _ in 0..NUMBER_OF_SCENARIOS {
            let n = thread_rng().gen_range(1..=num_cpus::get());
            let num_files = thread_rng().gen_range(1..=1000);
            let files = create_test_files(tmp_dir.path(), num_files, TEST_FILE_SIZE_KB);

            let predicted_time_ns = (num_files as f64 * T_CONVERSION_NS)
                + (num_files as f64 * T_SYSCALL_NS) / (n as f64 * f_disk(n))
                + (n as f64 * T_OVERHEAD_NS);

            let actual_time_duration = measure_concurrent_deletion_time(n, files);
            let actual_time_ns = actual_time_duration.as_nanos() as f64;

            predicted_times.push(predicted_time_ns / 1_000_000.0);
            actual_times.push(actual_time_ns / 1_000_000.0);

            //Manually clean up files since they are not being tracked by tempfile after being added to files vec
             for file in &files {
                if let Err(e) = std::fs::remove_file(file) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                        panic!("Failed to delete during cleanup, file: {}. Error: {}", file.display(), e);
                    }
                }
            }
        }


        let root = BitMapBackend::new("prediction_vs_actual.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let max_time = predicted_times.iter().cloned().fold(0.0, f64::max)
            .max(actual_times.iter().cloned().fold(0.0, f64::max));

        let mut chart = ChartBuilder::on(&root)
            .caption("Predicted vs. Actual Deletion Time", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0.0..max_time, 0.0..max_time)
            .unwrap();

        chart.configure_mesh()
            .x_desc("Predicted Time (ms)")
            .y_desc("Actual Time (ms)")
            .draw()
            .unwrap();

        chart.draw_series(
            predicted_times.iter().zip(actual_times.iter()).map(|(&x, &y)| {
                Circle::new((x, y), 3, BLUE.filled())
            })
        ).unwrap();

        chart.draw_series(LineSeries::new(
            (0..=max_time.ceil() as i32).map(|x| (x as f64, x as f64)),
            &RED
        )).unwrap()
        .label("Perfect Prediction (y=x)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels()
            .border_style(&BLACK)
            .background_style(&WHITE.mix(0.8))
            .draw().unwrap();

        root.present().expect("Unable to write result to file, please make sure 'plotters' feature is enabled.");
        println!("Scatter plot saved to prediction_vs_actual.png");

        // --- Statistical Analysis ---
        let correlation_coefficient = if predicted_times.len() == actual_times.len() && !predicted_times.is_empty() {
            let predicted_series = statrs::statistics::Series::from_slice(&predicted_times);
            let actual_series = statrs::statistics::Series::from_slice(&actual_times);
            
            match statrs::statistics::Correlation::pearson(&predicted_series, &actual_series) {
                Ok(coeff) => coeff,
                Err(_) => {
                    eprintln!("Error calculating Pearson correlation.");
                    f64::NAN
                }
            }
        } else {
            f64::NAN
        };
        println!("\n[Model Accuracy Test] --- Results ---");
        println!("[Model Accuracy Test] Average Predicted Time: {:.2} ms", predicted_times.iter().sum::<f64>() / NUMBER_OF_SCENARIOS as f64);
        println!("[Model Accuracy Test] Average Actual Time: {:.2} ms", actual_times.iter().sum::<f64>() / NUMBER_OF_SCENARIOS as f64);
        println!("[Model Accuracy Test] Pearson Correlation Coefficient (Predicted vs Actual Time): {:.6}", correlation_coefficient);
    }
}

// cargo test --release -- --nocapture
