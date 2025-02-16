use std::{
    ffi::CString,
    io,
    path::Path,
    sync::{
        Arc,
    },
};

use futures::StreamExt;
use num_cpus;
use lazy_static::lazy_static;
use progression::{Bar, Config};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::RawFd;
use globset::{Glob, GlobSetBuilder};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use crossbeam::channel;

lazy_static! {
    // Cache the number of CPUs and its f64 conversion as a static variable.
    static ref N_CPUS: usize = num_cpus::get();
    static ref N_CPUS_F: f64 = *N_CPUS as f64;
}

/// The `ProgressReporter` trait provides an interface for updating progress.
/// Implementors of this trait supply methods to increment the progress counter
/// and to finalize the progress display.
pub trait ProgressReporter {
    /// Increments the progress by a given number.
    fn inc(&self, n: u64);

    /// Finalizes the progress reporting.
    fn finish(&self);
}

/// `RealProgressBar` is a concrete implementation of `ProgressReporter`
/// that updates a visible progress bar using an underlying `Bar` instance.
/// The progress bar is wrapped in an `Arc` for safe sharing among threads.
#[derive(Clone)]
pub struct RealProgressBar<'a> {
    bar: Arc<Bar<'a>>,
}

impl<'a> RealProgressBar<'a> {
    /// Creates a new `RealProgressBar` from a shared `Bar`.
    pub fn new(bar: Arc<Bar<'a>>) -> Self {
        Self { bar }
    }
}

impl<'a> ProgressReporter for RealProgressBar<'a> {
    #[inline(always)]
    fn inc(&self, n: u64) {
        // Increments the underlying progress bar.
        self.bar.inc(n);
    }

    #[inline(always)]
    fn finish(&self) {
        match Arc::try_unwrap(Arc::clone(&self.bar)) {
            Ok(bar) => bar.finish(),
            Err(_) => {
                eprintln!("[WARN] Unable to get exclusive ownership of progress bar.");
            }
        }
    }
}

/// `NoOpProgressBar` is a no-operation implementation of `ProgressReporter`.
/// It performs no actions.
#[derive(Clone)]
pub struct NoOpProgressBar;

impl NoOpProgressBar {
    /// Creates a new `NoOpProgressBar`.
    #[inline(always)]
    pub const fn new() -> Self {
        Self
    }
}

impl ProgressReporter for NoOpProgressBar {
    #[inline(always)]
    fn inc(&self, _n: u64) {
        // Intentionally does nothing to avoid overhead.
    }

    #[inline(always)]
    fn finish(&self) {
        // Intentionally does nothing to avoid overhead.
    }
}

/// The `Progress` enum unifies the two progress reporting implementations
/// into a single type. It allows code to use a progress reporter without
/// knowing whether it is a real progress bar or a no-op.
#[derive(Clone)]
pub enum Progress<'a> {
    /// Variant for the no-operation progress reporter.
    NoOp(NoOpProgressBar),
    /// Variant for the real progress bar reporter.
    Real(RealProgressBar<'a>),
}

impl<'a> ProgressReporter for Progress<'a> {
    #[inline(always)]
    fn inc(&self, n: u64) {
        // Match on the variant and delegate the increment operation.
        match self {
            Progress::NoOp(p) => p.inc(n),
            Progress::Real(p) => p.inc(n),
        }
    }

    #[inline(always)]
    fn finish(&self) {
        // Match on the variant and delegate the finish operation.
        match self {
            Progress::NoOp(p) => p.finish(),
            Progress::Real(p) => p.finish(),
        }
    }
}

/// Synchronous wrapper around the async function.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <pattern> [--tokio|--rayon]", args[0]);
        eprintln!("Example: {} 'some_files_*.txt' --tokio", args[0]);
        std::process::exit(1);
    }
    let pattern = &args[1];

    // Determine deletion mode: default to tokio if not specified.
    let deletion_mode = if args.len() >= 3 {
        match args[2].as_str() {
            "--tokio" => "tokio",
            "--rayon" => "rayon",
            flag => {
                eprintln!("Unknown flag: {}. Expected --tokio or --rayon", flag);
                std::process::exit(1);
            }
        }
    } else {
        "tokio"
    };

    // Call count_matches to get the file descriptor, matched files, and total count.
    let (fd, matched_files) = match count_matches(pattern) {
        Ok(Some((fd, matched_files))) => (fd, matched_files),
        Ok(None) => {
            // No matching files found; exit gracefully.
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("Error during file matching: {}", e);
            std::process::exit(1);
        }
    };

    // Choose a progress reporter based on the total file count.
    let matched_files_number = matched_files.len();
    let progress_reporter = if matched_files_number < 1000 {
        Progress::NoOp(NoOpProgressBar::new())
    } else {
        let config = Config {
            throttle_millis: 250,
            ..Default::default()
        };
        let bar = Arc::new(Bar::new(matched_files_number as u64, config));
        Progress::Real(RealProgressBar::new(Arc::clone(&bar)))
    };

    match deletion_mode {
        "tokio" => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime");

            let result = runtime.block_on(run_deletion_tokio(
                None,
                progress_reporter,
                fd,
                matched_files,
                matched_files_number,
            ));

            match result {
                Ok(_) => println!("Files matching '{}' deleted successfully!", pattern),
                Err(e) => {
                    eprintln!("Error during deletion: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "rayon" => {
            let result = run_deletion_rayon(
                None,
                None,
                fd,
                matched_files,
                matched_files_number,
            );

            match result {
                Ok(_) => println!("Files matching '{}' deleted successfully!", pattern),
                Err(e) => {
                    eprintln!("Error during deletion: {}", e);
                    std::process::exit(1);
                }
            }
        }
        _ => unreachable!(), // We've already validated the deletion_mode.
    }
}

/// Main async entry point
async fn run_deletion_tokio<P: ProgressReporter + Clone>(
    concurrency_override: Option<usize>,
    progress_reporter: P,
    fd: RawFd,
    matched_files: Vec<CString>,
    matched_files_number: usize,
) -> io::Result<()> {
    let concurrency = match concurrency_override {
        Some(n) => n,
        None => compute_optimal_tokio(matched_files_number, num_cpus::get()).round() as usize,
    };
    println!(
        "[INFO] Deleting {} files with concurrency = {} (CPU cores = {})",
        matched_files_number,
        concurrency,
        num_cpus::get()
    );

    let file_stream = futures::stream::iter(matched_files.into_iter());
    let pr_for_tasks = progress_reporter.clone();
    file_stream
        .for_each_concurrent(Some(concurrency), move |filename_cstr| {
            {
                let progress_reporter_value = pr_for_tasks.clone();
                async move {
                    let progress_reporter = progress_reporter_value.clone();
                    let result = unsafe { libc::unlinkat(fd, filename_cstr.as_ptr(), 0) };
                    if result == 0 {
                        progress_reporter.inc(1);
                    } else {
                        let e = io::Error::last_os_error();
                        eprintln!(
                            "Failed to delete '{}': {}",
                            filename_cstr.to_string_lossy(),
                            e
                        );
                        std::process::exit(1);
                    }
                }
            }
        })
        .await;

    progress_reporter.finish();
    unsafe { libc::close(fd) };

    Ok(())
}

fn run_deletion_rayon(
    thread_pool_size: Option<usize>,
    batch_size_override: Option<usize>,
    fd: RawFd,
    matched_files: Vec<CString>,
    matched_files_number: usize,
) -> io::Result<()> {
    // Compute optimal thread pool and batch size
    let (optimal_thread_pool, optimal_batch) = compute_optimal_rayon(matched_files_number);

    // If no thread pool size was provided, use the computed optimal thread pool size (rounded to the nearest usize).
    let concurrency = thread_pool_size.unwrap_or_else(|| optimal_thread_pool.round() as usize);

    // If no batch size override was provided, use the computed optimal batch size (rounded to the nearest usize).
    let batch_size = batch_size_override.unwrap_or_else(|| optimal_batch.round() as usize);

    println!(
        "[INFO] Deleting {} files using Rayon with concurrency = {}, batch_size = {} (CPU cores = {})",
        matched_files_number,
        concurrency,
        batch_size,
        num_cpus::get()
    );

    // Setup a progress bar
    let config = Config {
        throttle_millis: 250,
        ..Default::default()
    };
    let pb = Arc::new(Bar::new(matched_files.len() as u64, config));

    // We'll send completed counts in batches to the progress bar thread.
    let (sender, receiver) = channel::unbounded::<usize>();

    // Spawn a thread to consume batch-completion counts and increment the bar.
    let pb_thread = std::thread::spawn({
        let pb = Arc::clone(&pb);
        move || {
            let mut total_done = 0;
            while let Ok(batch_count) = receiver.recv() {
                total_done += batch_count;
                pb.inc(batch_count as u64);
            }
            // If for some reason we didn't get the last partial batch, account for it here:
            let remainder = matched_files_number - total_done;
            if remainder > 0 {
                pb.inc(remainder as u64);
            }
            // We'll finish the bar later (outside this thread).
        }
    });

    // Build a custom Rayon thread pool with the desired concurrency.
    let pool = ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to build Rayon thread pool: {}", e),
            )
        })?;

    // Run the parallel deletion inside that custom pool.
    pool.install(|| {
        matched_files.par_chunks(batch_size).for_each(|chunk| {
            let mut count = 0;
            for filename_cstr in chunk {
                let result = unsafe { libc::unlinkat(fd, filename_cstr.as_ptr(), 0) };
                if result == 0 {
                    count += 1;
                } else {
                    let e = io::Error::last_os_error();
                    eprintln!(
                        "Failed to delete '{}': {}",
                        filename_cstr.to_string_lossy(),
                        e
                    );
                    std::process::exit(1);
                }
            }
            // Send how many were successfully deleted in this chunk.
            sender.send(count).unwrap();
        });
    });

    // Close the sending side, so the progress thread can end
    drop(sender);
    pb_thread.join().unwrap();

    // Now we can finish the bar because the thread is done
    match Arc::try_unwrap(pb) {
        Ok(bar) => bar.finish(),
        Err(_) => eprintln!("[WARN] Could not get exclusive ownership of progress bar."),
    }

    // Close the directory FD
    unsafe {
        libc::close(fd);
    }

    Ok(())
}

#[inline(always)]
pub fn compute_optimal_tokio(num_files: usize, simulated_cpus: usize) -> f64 {
    // Convert the number of files to a float and compute its natural logarithm
    let ln_num_files = (num_files as f64).ln();
    // Convert the number of simulated CPUs to a float and compute its natural logarithm
    let ln_simulated_cpus = (simulated_cpus as f64).ln();
    // Compute the optimal concurrency using the derived model:
    let optimal_concurrency = 23.7282581922052 * (0.138095238095238 * ln_num_files * ln_simulated_cpus - 0.552380952380952 * ln_num_files).exp();
    optimal_concurrency
}


#[inline(always)]
pub fn compute_optimal_rayon(file_count: usize) -> (f64, f64) {
    // Compute natural logarithm of file_count.
    let log_file_count = (file_count as f64).ln();
    // Precompute square of log_file_count.
    let log_file_count_squared = log_file_count * log_file_count;

    // Compute optimal thread_pool_size:
    // exp((169000.0 - 49112.0 * log_file_count) / (27.0 * log_file_count^2 - 91936.0))
    let optimal_thread_pool_size = ((169_000.0 - 49_112.0 * log_file_count) / (27.0 * log_file_count_squared - 91_936.0)).exp();

    // Compute optimal batch_size:
    // exp((9171.0 * log_file_count^2 - 29250.0 * log_file_count - 2284256.0) / (81.0 * log_file_count^2 - 275808.0))
    let optimal_batch_size = ((9_171.0 * log_file_count_squared - 29_250.0 * log_file_count - 2_284_256.0) / (81.0 * log_file_count_squared - 275_808.0)).exp();

    (optimal_thread_pool_size, optimal_batch_size)
}

/// Collects matching filenames from the given directory using `collect_matching_files`.
/// Returns a tuple of (fd, Vec<CString>) on success,
/// or `Ok(None)` if no matching files are found.
/// The caller is responsible for closing the file descriptor if matches are returned.
#[inline(always)]
fn count_matches(pattern: &str) -> io::Result<Option<(RawFd, Vec<CString>)>> {
    // Split the pattern into directory & filename parts.
    let (dir_path, file_pattern) = pattern.rsplit_once('/').unwrap_or((".", pattern));
    let dir = Path::new(dir_path);

    // Canonicalize the directory
    let canonical_dir = std::fs::canonicalize(dir)?;

    // Build the globset matcher
    let mut gs_builder = GlobSetBuilder::new();
    gs_builder.add(
        Glob::new(file_pattern)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid glob pattern: {}", e)))?
    );
    let glob_set = gs_builder.build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("GlobSet build error: {}", e)))?;

    // Collect matching files
    let (fd, matched_files) = collect_matching_files(&canonical_dir, &glob_set)?;

    if matched_files.is_empty() {
        println!("No matching files found for pattern '{}'.", pattern);
        unsafe {
            libc::close(fd);
        }
        return Ok(None);
    }

    Ok(Some((fd, matched_files)))
}


/// Collects matching filenames in the given directory using a fast,
/// `getdents64`-based approach on Linux and a `getdirentries`-based approach on macOS.
/// We do not build full paths; we just store the base names of regular files
/// that match the provided globset. The directory file descriptor (`fd`)
/// remains open so we can use it with `unlinkat` later.
///
/// # Inputs
/// - `dir`: A canonicalized directory (absolute path) in which to search.
/// - `matcher`: A reference to a `GlobSet` matcher built from the desired filename pattern.
///
/// # Output
/// Returns an `io::Result<(RawFd, Vec<CString>)>` containing:
///   1. A file descriptor (open directory) to be used with `unlinkat`.
///   2. A list of matching filenames (as CStrings), not full paths.
fn collect_matching_files(
    dir: &std::path::Path,
    matcher: &globset::GlobSet,
) -> std::io::Result<(std::os::unix::io::RawFd, Vec<std::ffi::CString>)> {
    // Open the directory using the raw syscall interface
    let c_path = std::ffi::CString::new(dir.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "Directory path contains null byte")
    })?;

    unsafe {
        let fd = libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY);
        if fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        #[cfg(target_os = "linux")]
        {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL);
        }
        #[cfg(target_os = "macos")]
        {
            libc::fcntl(fd, libc::F_RDADVISE, 0);
        }

        // Allocate a large buffer to minimize syscall overhead
        let buf_size = 1 << 26; // 64 MB
        let mut buf = vec![0u8; buf_size];
        let files = Vec::new();

        #[cfg(target_os = "linux")]
        {
            loop {
                let nread = libc::syscall(
                    libc::SYS_getdents64,
                    fd,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    buf_size,
                );
                if nread < 0 {
                    libc::close(fd);
                    return Err(std::io::Error::last_os_error());
                }
                if nread == 0 {
                    break;
                }
                let mut bpos = 0;
                while bpos < nread as usize {
                    let d = buf.as_ptr().add(bpos) as *const libc::dirent64;
                    let reclen = (*d).d_reclen as usize;
                    let name_ptr = (*d).d_name.as_ptr();

                    // Determine the length of the filename (until null terminator)
                    let mut namelen = 0;
                    while *name_ptr.add(namelen) != 0 {
                        namelen += 1;
                    }

                    // Skip entries for "." and ".."
                    if namelen > 0 {
                        let name_slice = std::slice::from_raw_parts(name_ptr as *const u8, namelen);
                        if name_slice != b"." && name_slice != b".." {
                            // Only consider regular files.
                            if (*d).d_type == libc::DT_REG {
                                let os_str = std::ffi::OsStr::from_bytes(name_slice);
                                // Use globset matcher to check if the filename matches.
                                if matcher.is_match(os_str) {
                                    // Store just the base name in the vector.
                                    match std::ffi::CString::new(name_slice) {
                                        Ok(cstr_filename) => {
                                            files.push(cstr_filename);
                                        }
                                        Err(_) => {
                                            // If there's a null byte in the filename, skip.
                                        }
                                    }
                                }
                            }
                        }
                    }
                    bpos += reclen;
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        pub mod macos_dirent {
            use std::ffi::{CStr, CString};
            use std::io;
            use std::os::unix::io::RawFd;
            use libc;
        
            extern "C" {
                /// Retrieves directory entries for the file descriptor.
                pub fn getdirentries(
                    fd: libc::c_int,
                    buf: *mut libc::c_char,
                    nbytes: libc::size_t,
                    basep: *mut libc::off_t,
                ) -> libc::ssize_t;
            }
        
            /// Represents a single directory entry.
            #[derive(Debug)]
            pub struct DirEntry {
                filename: CString,
                file_type: u8,
            }
        
            impl DirEntry {
                /// Creates a new `DirEntry` with the provided filename and file type.
                pub fn new(filename: CString, file_type: u8) -> Self {
                    println!(
                        "[DirEntry::new] Creating new DirEntry with filename: {:?} and file_type: {}",
                        filename, file_type
                    );
                    Self { filename, file_type }
                }
        
                /// Returns the directory entry's filename as a CStr.
                pub fn name(&self) -> &CStr {
                    println!(
                        "[DirEntry::name] Returning filename as CStr: {:?}",
                        self.filename
                    );
                    self.filename.as_c_str()
                }
        
                /// Returns the directory entry's file type.
                pub fn file_type(&self) -> u8 {
                    println!(
                        "[DirEntry::file_type] Returning file type: {}",
                        self.file_type
                    );
                    self.file_type
                }
            }
        
            /// An iterator over directory entries read using `getdirentries`.
            pub struct DirectoryIterator {
                fd: RawFd,
                buffer: Vec<u8>,
                buffer_pos: usize,
                bytes_read: usize,
            }
        
            impl DirectoryIterator {
                /// Creates a new `DirectoryIterator` for the given directory file descriptor.
                pub fn new(fd: RawFd) -> Self {
                    println!(
                        "[DirectoryIterator::new] Initializing DirectoryIterator with fd: {}",
                        fd
                    );
                    let buffer_size = 64 * 1024; // 64 KB
                    println!(
                        "[DirectoryIterator::new] Allocating buffer with size: {} bytes",
                        buffer_size
                    );
                    Self {
                        fd,
                        buffer: vec![0; buffer_size],
                        buffer_pos: 0,
                        bytes_read: 0,
                    }
                }
        
                /// Reads the next block of directory entries into the internal buffer.
                fn read_next_block(&mut self) -> io::Result<usize> {
                    println!(
                        "[DirectoryIterator::read_next_block] Reading next block from fd: {}",
                        self.fd
                    );
                    let mut base: libc::off_t = 0;
                    println!(
                        "[DirectoryIterator::read_next_block] Initial base value before read: {}",
                        base
                    );
                    let nread = unsafe {
                        getdirentries(
                            self.fd,
                            self.buffer.as_mut_ptr() as *mut libc::c_char,
                            self.buffer.len(),
                            &mut base as *mut libc::off_t,
                        )
                    };
                    println!(
                        "[DirectoryIterator::read_next_block] getdirentries returned: {}",
                        nread
                    );
                    if nread < 0 {
                        let err = io::Error::last_os_error();
                        println!(
                            "[DirectoryIterator::read_next_block] Error encountered: {:?}",
                            err
                        );
                        return Err(err);
                    }
                    self.bytes_read = nread as usize;
                    self.buffer_pos = 0;
                    println!(
                        "[DirectoryIterator::read_next_block] Bytes read: {}. Buffer position reset to 0.",
                        self.bytes_read
                    );
                    Ok(self.bytes_read)
                }
            }
        
            impl Iterator for DirectoryIterator {
                type Item = io::Result<DirEntry>;
        
                fn next(&mut self) -> Option<Self::Item> {
                    println!(
                        "[DirectoryIterator::next] Called next(). Current buffer_pos: {}, bytes_read: {}",
                        self.buffer_pos, self.bytes_read
                    );
                    if self.buffer_pos >= self.bytes_read {
                        println!("[DirectoryIterator::next] Buffer exhausted. Attempting to read next block.");
                        match self.read_next_block() {
                            Ok(0) => {
                                println!("[DirectoryIterator::next] End of directory reached (0 bytes read).");
                                return None;
                            }
                            Ok(n) => println!("[DirectoryIterator::next] Successfully read {} bytes.", n),
                            Err(e) => {
                                println!(
                                    "[DirectoryIterator::next] Error reading next block: {:?}",
                                    e
                                );
                                return Some(Err(e));
                            }
                        }
                    }
                    if self.buffer_pos >= self.bytes_read {
                        println!("[DirectoryIterator::next] No more data available after reading block.");
                        return None;
                    }
                    unsafe {
                        let entry_ptr = self.buffer.as_ptr().add(self.buffer_pos) as *const libc::dirent;
                        let reclen = (*entry_ptr).d_reclen as usize;
                        println!(
                            "[DirectoryIterator::next] Processing entry at buffer_pos {} with reclen: {}",
                            self.buffer_pos, reclen
                        );
                        if reclen == 0 || self.buffer_pos + reclen > self.bytes_read {
                            println!("[DirectoryIterator::next] Invalid reclen or buffer overrun detected. Ending iteration.");
                            return None;
                        }
                        let name_ptr = (*entry_ptr).d_name.as_ptr();
                        let c_str = CStr::from_ptr(name_ptr);
                        println!(
                            "[DirectoryIterator::next] Extracted CStr from entry: {:?}",
                            c_str
                        );
                        let filename_bytes = c_str.to_bytes();
                        if filename_bytes == b"." || filename_bytes == b".." {
                            println!(
                                "[DirectoryIterator::next] Skipping entry ('.' or '..'): {:?}",
                                c_str
                            );
                            self.buffer_pos += reclen;
                            println!(
                                "[DirectoryIterator::next] Advanced buffer_pos to {} after skipping entry.",
                                self.buffer_pos
                            );
                            return self.next();
                        }
                        let filename = match CString::new(filename_bytes) {
                            Ok(s) => s,
                            Err(_) => {
                                println!(
                                    "[DirectoryIterator::next] Failed to convert entry to CString. Skipping entry."
                                );
                                self.buffer_pos += reclen;
                                println!(
                                    "[DirectoryIterator::next] Advanced buffer_pos to {} after conversion failure.",
                                    self.buffer_pos
                                );
                                return self.next();
                            }
                        };
                        let file_type = (*entry_ptr).d_type;
                        println!(
                            "[DirectoryIterator::next] Creating DirEntry with filename: {:?} and file_type: {}",
                            filename, file_type
                        );
                        let dir_entry = DirEntry::new(filename, file_type);
                        self.buffer_pos += reclen;
                        println!(
                            "[DirectoryIterator::next] Advanced buffer_pos to {} after processing entry.",
                            self.buffer_pos
                        );
                        Some(Ok(dir_entry))
                    }
                }
            }
        
            /// Reads directory entries from the given file descriptor and returns a DirectoryIterator.
            pub fn read_dir(fd: RawFd) -> io::Result<DirectoryIterator> {
                println!(
                    "[macos_dirent::read_dir] Starting directory reading for fd: {}",
                    fd
                );
                let it = DirectoryIterator::new(fd);
                println!("[macos_dirent::read_dir] DirectoryIterator created successfully.");
                Ok(it)
            }
        
            impl Drop for DirectoryIterator {
                fn drop(&mut self) {
                    println!(
                        "[DirectoryIterator::drop] Dropping DirectoryIterator for fd: {}",
                        self.fd
                    );
                    unsafe {
                        if libc::close(self.fd) == 0 {
                            println!(
                                "[DirectoryIterator::drop] Successfully closed file descriptor: {}",
                                self.fd
                            );
                        } else {
                            println!(
                                "[DirectoryIterator::drop] Failed to close file descriptor: {}. Error: {:?}",
                                self.fd,
                                io::Error::last_os_error()
                            );
                        }
                    }
                }
            }
        }

        // At this point, we do NOT close(fd). We return it alongside the filenames.
        Ok((fd, files))
    }
}


// ====================================================================================================================================
// ====================================================================================================================================
// ====================================================================================================================================
// ====================================================================================================================================


// cargo test --release -- --nocapture shell_performance
#[cfg(test)]
mod shell_performance {
    use glob::glob;
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        thread::sleep,
        time::{Duration, Instant},
    };

    // Number of iterations per command
    const ITERATIONS: usize = 5;

    /// Returns the base test directory (e.g., "$HOME/tmp_test").
    fn base_test_dir() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME environment variable not set");
        PathBuf::from(home).join("tmp_test")
    }

    /// Retrieves filesystem information for the given directory using `df -T`.
    fn get_filesystem_info(dir: &Path) -> String {
        let output = Command::new("df")
            .arg("-T")
            .arg(dir)
            .output()
            .expect("Failed to get filesystem info");
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Prepares a fresh test directory for a single benchmark iteration.
    /// The directory will be created at:
    ///   BASE_TEST_DIR/<test_name>_<command_type>_iter<iteration>
    fn prepare_test_directory(test_name: &str, command_type: &str, iteration: usize) -> PathBuf {
        let dir_path = base_test_dir().join(format!("{}_{}_iter{}", test_name, command_type, iteration));

        // Remove the directory if it already exists.
        if dir_path.exists() {
            fs::remove_dir_all(&dir_path)
                .unwrap_or_else(|e| panic!("Failed to remove {}: {}", dir_path.display(), e));
        }

        // Create the new directory.
        fs::create_dir_all(&dir_path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", dir_path.display(), e));

        // Print filesystem information for transparency.
        println!(
            "Filesystem info for {}:\n{}",
            dir_path.display(),
            get_filesystem_info(&dir_path)
        );
        dir_path
    }

    /// Creates test files in the specified directory.
    ///
    /// Files are named "test_file_0.dat", "test_file_1.dat", etc., and each file
    /// contains 16 zero bytes.
    fn create_test_files(dir: &Path, count: usize) {
        for i in 0..count {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&file_path)
                .unwrap_or_else(|e| panic!("Failed to create {}: {}", file_path.display(), e));
            let content = vec![0u8; 16];
            file.write_all(&content)
                .unwrap_or_else(|e| panic!("Failed to write to {}: {}", file_path.display(), e));
        }
    }

    /// Verifies that no files matching the provided glob pattern remain.
    /// If any undeleted files are found, it prints debugging information and panics.
    fn verify_no_files(pattern: &str) {
        let mut count = 0;
        let mut undeleted_files = Vec::new();

        for entry in glob(pattern).expect("Invalid glob pattern") {
            match entry {
                Ok(path) => {
                    if path.is_file() {
                        count += 1;
                        undeleted_files.push(path);
                    }
                }
                Err(e) => {
                    eprintln!("DEBUG: Error reading entry: {}", e);
                }
            }
        }

        if count > 0 {
            println!("DEBUG: Found {} undeleted file(s) matching '{}':", count, pattern);
            for file in undeleted_files {
                println!("DEBUG: {}", file.display());
            }
            panic!("Deletion failed: {} file(s) still exist matching {}", count, pattern);
        }
    }

    /// Executes a shell command (using `sh -c`) and returns the elapsed time in seconds.
    ///
    /// This function performs all setup (pre-sync) and teardown (post-sync, sleep, verification)
    /// outside of the timed region. Only the actual execution of the command is timed.
    fn run_command(command: &str, pattern: &str) -> f64 {
        println!("Executing: {}", command);

        // Pre-command: flush I/O (setup; not timed)
        Command::new("sync")
            .status()
            .expect("Failed to sync before command");

        // Start timing just before executing the command.
        let start = Instant::now();
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .unwrap_or_else(|e| panic!("Failed to execute command `{}`: {}", command, e));
        let command_elapsed = start.elapsed();

        if !output.status.success() {
            panic!(
                "Command `{}` failed:\n{}",
                command,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Post-command: flush I/O, wait for metadata to settle, and verify deletion (not timed)
        Command::new("sync")
            .status()
            .expect("Failed to sync after command");
        sleep(Duration::from_millis(100));
        verify_no_files(pattern);

        // Return only the command execution time.
        command_elapsed.as_secs_f64()
    }

    /// Calculates statistical metrics (min, max, mean, median, stddev) for the provided time values.
    fn calculate_stats(times: &[f64]) -> (f64, f64, f64, f64, f64) {
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = times.iter().sum();
        let mean = sum / times.len() as f64;

        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
        let stddev = variance.sqrt();
        (min, max, mean, median, stddev)
    }

    /// Runs a single benchmark iteration.
    ///
    /// This function sets up the test directory, creates the test files, and then
    /// times the deletion command—isolating the measurement strictly to the command's execution.
    fn run_single_benchmark(
        test_name: &str,
        file_count: usize,
        command_type: &str,
        iteration: usize,
    ) -> f64 {
        let dir_path = prepare_test_directory(test_name, command_type, iteration);
        // Build the glob pattern to match test files.
        let pattern = format!("{}/t*.dat", dir_path.to_string_lossy());
        println!("Creating {} file(s) in {}", file_count, dir_path.display());
        create_test_files(&dir_path, file_count);

        // Determine the deletion command.
        let command = if command_type == "rust" {
            format!("target/release/del \"{}\"", pattern)
        } else {
            if file_count == 1_000_000 {
                format!("find {} -maxdepth 1 -type f -delete", dir_path.to_string_lossy())
            } else {
                format!("rm -f {}/t*.dat", dir_path.to_string_lossy())
            }
        };

        // Time only the command execution.
        let elapsed = run_command(&command, &pattern);
        println!(
            "[{}] Iteration {} ({} files, {} deletion) completed in {:.3} seconds",
            test_name,
            iteration + 1,
            file_count,
            command_type,
            elapsed
        );
        elapsed
    }

    /// Runs the benchmark for a given file count.
    ///
    /// This function executes both the Rust binary deletion and the system deletion commands
    /// over multiple iterations and then prints comprehensive statistics.
    fn run_benchmark_for_file_count(test_name: &str, file_count: usize) {
        println!("\n===== {}: {} file(s) =====", test_name, file_count);
        println!("Using base test directory: {}", base_test_dir().display());

        // Benchmark the Rust binary deletion.
        println!("--- Running Rust binary Deletion_Benchmark ---");
        let mut rust_times = Vec::with_capacity(ITERATIONS);
        for iter in 0..ITERATIONS {
            rust_times.push(run_single_benchmark(test_name, file_count, "rust", iter));
        }
        let (min_r, max_r, mean_r, median_r, stddev_r) = calculate_stats(&rust_times);
        println!(
            "[{} - Rust] min: {:.3} s, max: {:.3} s, mean: {:.3} s, median: {:.3} s, stddev: {:.3} s",
            test_name, min_r, max_r, mean_r, median_r, stddev_r
        );

        // Benchmark the system deletion command.
        println!("--- Running system Deletion_Benchmark ---");
        let mut system_times = Vec::with_capacity(ITERATIONS);
        for iter in 0..ITERATIONS {
            system_times.push(run_single_benchmark(test_name, file_count, "system", iter));
        }
        let (min_s, max_s, mean_s, median_s, stddev_s) = calculate_stats(&system_times);
        println!(
            "[{} - System] min: {:.3} s, max: {:.3} s, mean: {:.3} s, median: {:.3} s, stddev: {:.3} s",
            test_name, min_s, max_s, mean_s, median_s, stddev_s
        );
    }

    /// Executes the full benchmark suite by running both deletion methods
    /// across multiple file count scenarios.
    #[test]
    fn benchmark_shell_commands() {
        println!("=== Starting Shell Command Benchmarks ===");
        let file_counts = [1, 100, 10_000];
        for &count in &file_counts {
            run_benchmark_for_file_count("Deletion_Benchmark", count);
        }
        println!("=== Benchmarks Complete ===");
    }
}

// cargo test --release -- --nocapture t_r_performance
#[allow(non_snake_case)]
#[cfg(test)]
mod t_r_performance {
    use glob::glob;
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        thread::sleep,
        time::{Duration, Instant},
    };

    // Number of iterations per command
    const ITERATIONS: usize = 10;

    // Base directory for test runs
    fn base_test_dir() -> std::path::PathBuf {
        let home = std::env::var("HOME").expect("HOME environment variable not set");
        std::path::PathBuf::from(home).join("tmp_test")
    }


    /// Returns filesystem info (using `df -T`) for the given directory.
    fn get_filesystem_info(dir: &Path) -> String {
        let output = Command::new("df")
            .arg("-T")
            .arg(dir)
            .output()
            .expect("Failed to get filesystem info");
        String::from_utf8_lossy(&output.stdout).to_string()
    }

    /// Prepares a fresh test directory for a single benchmark iteration.
    /// The directory will be created at:
    ///   BASE_TEST_DIR/<test_name>_<command_type>_iter<iteration>
    fn prepare_test_directory(test_name: &str, command_type: &str, iteration: usize) -> PathBuf {
    let dir_path = base_test_dir()
        .join(format!("{}_{}_iter{}", test_name, command_type, iteration));

        // Remove the directory if it already exists
        if dir_path.exists() {
            fs::remove_dir_all(&dir_path)
                .unwrap_or_else(|e| panic!("Failed to remove {}: {}", dir_path.display(), e));
        }
        // Create the fresh directory
        fs::create_dir_all(&dir_path)
            .unwrap_or_else(|e| panic!("Failed to create {}: {}", dir_path.display(), e));
        // Report filesystem type/info for transparency
        println!(
            "Filesystem info for {}:\n{}",
            dir_path.display(),
            get_filesystem_info(&dir_path)
        );
        dir_path
    }

    /// Creates test files (named "test_file_0.dat", "test_file_1.dat", …)
    /// in the given directory. Each file gets 16 zero bytes.
    fn create_test_files(dir: &Path, count: usize) {
        for i in 0..count {
            let path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&path)
                .unwrap_or_else(|e| panic!("Failed to create {}: {}", path.display(), e));
            let content = vec![0u8; 16];
            file.write_all(&content)
                .unwrap_or_else(|e| panic!("Failed to write to {}: {}", path.display(), e));
        }
    }

    /// Verifies that no files matching the given glob pattern remain.
    /// Panics if any matching file is found.
    fn verify_no_files(pattern: &str) {
        let mut count = 0;
        for entry in glob(pattern).expect("Invalid glob pattern") {
            if let Ok(path) = entry {
                if path.is_file() {
                    count += 1;
                }
            }
        }
        if count > 0 {
            panic!("Deletion failed: {} file(s) still exist matching {}", count, pattern);
        }
    }

    /// Runs the provided shell command (via `sh -c`) and returns the elapsed time in seconds.
    /// Sync is executed before starting the timer and after the command completes to flush caches.
    fn run_command(command: &str, pattern: &str) -> f64 {
        println!("Executing: {}", command);
        // Force pending I/O to disk before timing
        Command::new("sync")
            .status()
            .expect("Failed to sync before command");
        let start = Instant::now();
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .unwrap_or_else(|e| panic!("Failed to execute command `{}`: {}", command, e));
        if !output.status.success() {
            panic!(
                "Command `{}` failed:\n{}",
                command,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        // Force pending I/O to disk after command execution
        Command::new("sync")
            .status()
            .expect("Failed to sync after command");
        let elapsed = start.elapsed().as_secs_f64();
        // Small delay to allow filesystem metadata to settle
        sleep(Duration::from_millis(100));
        verify_no_files(pattern);
        elapsed
    }

    /// Calculates statistical metrics (min, max, mean, median, standard deviation) for the provided times.
    fn calculate_stats(times: &[f64]) -> (f64, f64, f64, f64, f64) {
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = times.iter().sum();
        let mean = sum / times.len() as f64;
        let mut sorted = times.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let variance = times
            .iter()
            .map(|t| (t - mean).powi(2))
            .sum::<f64>()
            / times.len() as f64;
        let stddev = variance.sqrt();
        (min, max, mean, median, stddev)
    }

    /// Runs a single benchmark iteration for the given command type ("tokio" or "rayon")
    /// and returns the elapsed time in seconds.
    fn run_single_benchmark(
        test_name: &str,
        file_count: usize,
        command_type: &str,
        iteration: usize,
    ) -> f64 {
        // Prepare a fresh, isolated test directory
        let dir_path = prepare_test_directory(test_name, command_type, iteration);
        // Build the glob pattern for matching test files
        let pattern = format!("{}/t*.dat", dir_path.to_string_lossy());
        println!("Creating {} file(s) in {}", file_count, dir_path.display());
        create_test_files(&dir_path, file_count);
        // Build the command for the Rust binary deletion.
        // It uses the file glob pattern and the provided flag (--tokio or --rayon).
        let command = format!("target/release/del \"{}\" --{}", pattern, command_type);
        let elapsed = run_command(&command, &pattern);
        println!(
            "[{}] Iteration {} ({} files, {} deletion) completed in {:.3} seconds",
            test_name,
            iteration + 1,
            file_count,
            command_type,
            elapsed
        );
        elapsed
    }

    /// Approximates the error function.
    fn erf(x: f64) -> f64 {
        // Abramowitz and Stegun formula 7.1.26
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    /// Approximates the cumulative distribution function for a standard normal.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / 2f64.sqrt()))
    }

    /// Performs a Mann–Whitney U test on two groups of f64 values.
    /// Returns a tuple (U, p_value).
    fn mann_whitney_u_test(group1: &[f64], group2: &[f64]) -> (f64, f64) {
        let n1 = group1.len();
        let n2 = group2.len();
        let mut combined: Vec<(f64, u8)> = Vec::with_capacity(n1 + n2);
        for &val in group1 {
            combined.push((val, 1));
        }
        for &val in group2 {
            combined.push((val, 2));
        }
        // Sort by value (ascending)
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Assign ranks with tie correction
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i + 1;
            while j < combined.len() && (combined[j].0 - combined[i].0).abs() < 1e-10 {
                j += 1;
            }
            let rank_sum: f64 = ((i + 1)..=j).map(|r| r as f64).sum();
            let avg_rank = rank_sum / ((j - i) as f64);
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        // Sum of ranks for group1
        let mut rank_sum1 = 0.0;
        for (i, &(_, grp)) in combined.iter().enumerate() {
            if grp == 1 {
                rank_sum1 += ranks[i];
            }
        }
        let U1 = rank_sum1 - (n1 * (n1 + 1)) as f64 / 2.0;
        let U2 = (n1 * n2) as f64 - U1;
        let U = U1.min(U2);

        // Compute tie correction for variance adjustment
        let N = n1 + n2;
        let mut tie_sum = 0.0;
        let mut i = 0;
        while i < combined.len() {
            let mut j = i + 1;
            while j < combined.len() && (combined[j].0 - combined[i].0).abs() < 1e-10 {
                j += 1;
            }
            let t = (j - i) as f64;
            if t > 1.0 {
                tie_sum += t * t * t - t;
            }
            i = j;
        }
        let var_U = (n1 as f64 * n2 as f64 / 12.0)
            * ((N as f64 + 1.0) - tie_sum / ((N as f64) * (N as f64 - 1.0)));
        let mean_U = (n1 * n2) as f64 / 2.0;
        let z = if var_U > 0.0 {
            (U - mean_U) / var_U.sqrt()
        } else {
            0.0
        };
        let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));
        (U, p_value)
    }

    /// Structure to hold benchmark results for a given file count.
    struct BenchmarkResult {
        file_count: usize,
        tokio_mean: f64,
        rayon_mean: f64,
        u_stat: f64,
        p_value: f64,
        winner: &'static str,
    }

    /// Runs the benchmark for a given test scenario (file count).
    /// Both the Tokio and Rayon deletion methods are run over multiple iterations.
    /// Returns a BenchmarkResult containing mean times, Mann–Whitney U test statistic, p-value, and the winner.
    fn run_benchmark_for_file_count(test_name: &str, file_count: usize) -> BenchmarkResult {
        println!("\n===== {}: {} file(s) =====", test_name, file_count);
        println!("Using base test directory: {}", base_test_dir().display());

        // Run the Tokio deletion benchmark
        println!("--- Running Tokio Deletion_Benchmark ---");
        let mut tokio_times = Vec::new();
        for iter in 0..ITERATIONS {
            tokio_times.push(run_single_benchmark(test_name, file_count, "tokio", iter));
        }
        let (_min_tokio, _max_tokio, mean_tokio, _median_tokio, _stddev_tokio) =
            calculate_stats(&tokio_times);
        println!(
            "[{} - Tokio] mean: {:.3} s (n={})",
            test_name,
            mean_tokio,
            tokio_times.len()
        );

        // Run the Rayon deletion benchmark
        println!("--- Running Rayon Deletion_Benchmark ---");
        let mut rayon_times = Vec::new();
        for iter in 0..ITERATIONS {
            rayon_times.push(run_single_benchmark(test_name, file_count, "rayon", iter));
        }
        let (_min_rayon, _max_rayon, mean_rayon, _median_rayon, _stddev_rayon) =
            calculate_stats(&rayon_times);
        println!(
            "[{} - Rayon] mean: {:.3} s (n={})",
            test_name,
            mean_rayon,
            rayon_times.len()
        );

        // Perform Mann–Whitney U test on the two sets of times.
        let (u_stat, p_value) = mann_whitney_u_test(&tokio_times, &rayon_times);

        // Determine the winner based on lower mean time.
        let winner = if mean_tokio < mean_rayon {
            "Tokio"
        } else if mean_rayon < mean_tokio {
            "Rayon"
        } else {
            "Tie"
        };

        BenchmarkResult {
            file_count,
            tokio_mean: mean_tokio,
            rayon_mean: mean_rayon,
            u_stat,
            p_value,
            winner,
        }
    }

    /// Main performance benchmark test.
    ///
    /// Each scenario runs both the Tokio and Rayon deletion methods over multiple iterations.
    /// After all benchmarks, a final table with Mann–Whitney U test results is displayed.
    #[test]
    fn benchmark_shell_commands() {
        println!("=== Starting Shell Command Benchmarks ===");
        let file_counts = [1, 5, 100, 5_000];
        let mut results = Vec::new();
        for &count in &file_counts {
            let res = run_benchmark_for_file_count("Deletion_Benchmark", count);
            results.push(res);
        }
        println!("\n=== Final Mann–Whitney U Test Results ===");
        println!(
            "{:<10} {:>14} {:>14} {:>10} {:>12} {:>10}",
            "Files", "Tokio Mean(s)", "Rayon Mean(s)", "U", "p-value", "Winner"
        );
        println!("{}", "-".repeat(70));
        for r in results {
            println!(
                "{:<10} {:>14.3} {:>14.3} {:>10.3} {:>12.4} {:>10}",
                r.file_count, r.tokio_mean, r.rayon_mean, r.u_stat, r.p_value, r.winner
            );
        }
        println!("=== Benchmarks Complete ===");
    }
}

// cargo test --release -- --nocapture performance_tests
#[cfg(test)]
mod performance_tests {
    use std::{
        fs::{self, File},
        io::Write,
        path::{Path, PathBuf},
        process::Command,
        time::Instant,
    };
    use tempfile::tempdir;
    use tokio::runtime::Builder;
    use glob;

    use super::{count_matches, run_deletion_tokio, NoOpProgressBar, Progress};

    /// Creates test files in `dir` using a standard naming scheme (e.g. "test_file_0.dat").
    fn create_test_files(dir: &Path, count: usize, size_kb: usize) -> Vec<PathBuf> {
        let mut file_paths = Vec::with_capacity(count);
        for i in 0..count {
            let path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&path).unwrap();
            file.write_all(&vec![0u8; size_kb * 1024]).unwrap();
            file_paths.push(path);
        }
        file_paths
    }

    /// Converts a given number into a minimal base‑36 string.
    fn to_base36(mut num: usize) -> String {
        const BASE36: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
        if num == 0 {
            return "0".to_string();
        }
        let mut buffer = Vec::new();
        while num > 0 {
            buffer.push(BASE36[num % 36]);
            num /= 36;
        }
        buffer.reverse();
        String::from_utf8(buffer).unwrap()
    }

    /// Creates test files in `dir` using short names (e.g. "0", "1", "2", …) generated from base‑36.
    fn create_short_test_files(dir: &Path, count: usize, size_kb: usize) -> Vec<PathBuf> {
        let mut file_paths = Vec::with_capacity(count);
        for i in 0..count {
            let path = dir.join(to_base36(i));
            let mut file = File::create(&path).unwrap();
            file.write_all(&vec![0u8; size_kb * 1024]).unwrap();
            file_paths.push(path);
        }
        file_paths
    }

    /// Runs the Rust deletion routine for files matching `pattern` using run_deletion_tokio.
    /// It uses `count_matches` to open the directory and gather matching file names,
    /// then calls the deletion routine with a No‑op progress reporter.
    /// Returns the elapsed time (in seconds) and verifies that no matching files remain.
    fn measure_rust_deletion(pattern: &str) -> f64 {
        let start = Instant::now();
        let rt = Builder::new_current_thread().enable_all().build().unwrap();

        // Get the directory FD and matching files.
        let (fd, matched_files) = match count_matches(pattern).unwrap() {
            Some((fd, files)) => (fd, files),
            None => {
                // If there are no matches, nothing to delete.
                return start.elapsed().as_secs_f64();
            }
        };
        let matched_files_number = matched_files.len();

        // Use a No‑op progress reporter.
        let progress = Progress::NoOp(NoOpProgressBar::new());
        let result = rt.block_on(run_deletion_tokio(
            None,
            progress,
            fd,
            matched_files,
            matched_files_number,
        ));
        let elapsed = start.elapsed().as_secs_f64();

        if let Err(e) = result {
            panic!("Error during Rust deletion: {}", e);
        }

        // Verify that no matching files remain.
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

    /// Runs a system deletion command (using `find` or `rm`) for files matching `pattern`.
    /// Returns a tuple with the elapsed time (in seconds) and the executed command string.
    fn measure_rm_deletion(pattern: &str, use_find: bool) -> (f64, String) {
        let start = Instant::now();
        let (elapsed, cmd) = if use_find {
            let dir = Path::new(pattern)
                .parent()
                .expect("Unable to determine parent directory")
                .to_string_lossy()
                .to_string();
            let cmd = format!("find {} -maxdepth 1 -type f -delete", dir);
            println!("Executing system deletion command: {}", cmd);
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
                .expect("Failed to execute find command");
            if !output.status.success() {
                eprintln!(
                    "find command stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            (start.elapsed().as_secs_f64(), cmd)
        } else {
            let cmd = format!("rm -f {}", pattern);
            println!("Executing system deletion command: {}", cmd);
            let output = Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .output()
                .expect("Failed to execute rm command");
            if !output.status.success() {
                eprintln!(
                    "rm command stderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            (start.elapsed().as_secs_f64(), cmd)
        };

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

    /// Formats a duration (in seconds) into a human‑readable string.
    /// Uses milliseconds if the duration is less than one second.
    fn format_duration(seconds: f64) -> String {
        if seconds < 1.0 {
            format!("{:.3} ms", seconds * 1000.0)
        } else {
            format!("{:.3} s", seconds)
        }
    }

    /// Struct to hold the benchmark results of a test scenario.
    struct BenchmarkResult {
        test_name: String,
        rust_time: f64,
        rm_time: f64,
        system_command: String,
    }

    /// Runs a benchmark scenario: it creates files, measures both Rust and system deletion times,
    /// and returns a summary result. The parameters control the naming scheme and whether to use `find` for deletion.
    fn run_benchmark(
        test_name: &str,
        file_count: usize,
        file_size_kb: usize,
        use_short_names: bool,
        use_find_for_rm: bool,
    ) -> BenchmarkResult {
        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path().to_path_buf();

        // Choose the file creation function and matching glob pattern based on the naming scheme.
        let (pattern, create_fn): (String, fn(&Path, usize, usize) -> Vec<PathBuf>) =
            if use_short_names {
                let pattern = base_path.join("[0-9a-z]*");
                (pattern.to_string_lossy().to_string(), create_short_test_files)
            } else {
                let pattern = base_path.join("t*.dat");
                (pattern.to_string_lossy().to_string(), create_test_files)
            };

        println!("\n--- {} ---", test_name);
        println!("Creating {} files ({} KB each)...", file_count, file_size_kb);

        // Create files and run the Rust deletion routine.
        create_fn(&base_path, file_count, file_size_kb);
        println!("Running Rust deletion...");
        let rust_time = measure_rust_deletion(&pattern);
        println!("Rust deletion completed in {}", format_duration(rust_time));

        // Recreate files for the system deletion test.
        create_fn(&base_path, file_count, file_size_kb);
        println!(
            "Running system deletion (using {} command)...",
            if use_find_for_rm { "find" } else { "rm" }
        );
        let (rm_time, system_cmd) = measure_rm_deletion(&pattern, use_find_for_rm);
        println!("System deletion completed in {}", format_duration(rm_time));

        BenchmarkResult {
            test_name: test_name.to_string(),
            rust_time,
            rm_time,
            system_command: system_cmd,
        }
    }

    /// Main performance benchmark test that runs several scenarios and prints a summary.
    #[test]
    fn performance_summary() {
        println!("\n===== Starting Performance Benchmarks =====");

        let mut results = Vec::new();
        results.push(run_benchmark("One file (10 x 10 KB)", 10, 10, false, false));
        results.push(run_benchmark("Small files (10 x 1 KB)", 10, 1, false, false));
        results.push(run_benchmark("Some small files (30,000 x 1 KB)", 30_000, 1, false, false));
        results.push(run_benchmark("Many small files (100,000 x 1 KB)", 100_000, 1, true, true));
        results.push(run_benchmark("A ton of small files (500,000 x 1 KB)", 500_000, 1, true, true));
        results.push(run_benchmark("Large files (100 x 10 MB)", 100, 10240, false, false));
        results.push(run_benchmark("Medium files (50 x 100 KB)", 50, 100, false, false));
        results.push(run_benchmark("Huge files (10 x 50 MB)", 10, 51200, false, false));

        println!("\n===== Performance Summary =====\n");
        println!(
            "{:<40} | {:>10} | {:>10} | {:>10} | {:<30}",
            "Test Scenario", "Rust", "System", "Diff", "System Command"
        );
        println!("{}", "-".repeat(110));

        for result in results {
            let diff = (result.rust_time - result.rm_time).abs();
            let winner = if result.rust_time < result.rm_time {
                "Rust"
            } else {
                "System"
            };
            println!(
                "{:<40} | {:>10} | {:>10} | {:>10} | {:<30}",
                result.test_name,
                format_duration(result.rust_time),
                format_duration(result.rm_time),
                format_duration(diff),
                result.system_command
            );
            println!("Winner: {}\n", winner);
        }
        println!("{}", "-".repeat(110));
        println!("Note: Times are measured in seconds (or ms if < 1 second).");
    }

    /// Test to verify that running deletion with a pattern that matches no files does not error.
    #[test]
    fn test_delete_with_no_matches() {
        println!("\n--- Test: Deletion with No Matches ---");
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path().to_string_lossy().to_string();
        let pattern = format!("{}/no_such_file_*.dat", base);

        let elapsed = measure_rust_deletion(&pattern);
        println!(
            "Rust deletion with no matches completed in {}.",
            format_duration(elapsed)
        );
    }

    /// Test to make sure that directories are not removed during deletion.
    #[test]
    fn test_skips_directories() {
        println!("\n--- Test: Skipping Directories ---");
        let temp_dir = tempdir().unwrap();
        let dir_path = temp_dir.path().join("my_test_dir");
        fs::create_dir(&dir_path).unwrap();

        // Create a file that matches the pattern.
        let file_path = temp_dir.path().join("my_test_dir_file.dat");
        File::create(&file_path).unwrap();

        let pattern = temp_dir.path().join("my_test_dir*");
        let pattern_str = pattern.to_string_lossy().to_string();

        let elapsed = measure_rust_deletion(&pattern_str);
        println!(
            "Rust deletion (skipping directories) completed in {}.",
            format_duration(elapsed)
        );

        // Verify that the directory still exists and the file has been deleted.
        assert!(dir_path.is_dir(), "Directory was deleted!");
        assert!(!file_path.exists(), "File was not deleted!");
    }

    /// Test to make sure that when the deletion pattern only matches some files,
    /// only the matching files are removed while non‑matching files remain.
    #[test]
    fn test_partial_match_deletion() {
        println!("\n--- Test: Partial Pattern Deletion ---");
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Create files that match the pattern.
        let matching_files: Vec<PathBuf> = (0..5)
            .map(|i| {
                let path = base.join(format!("match_{}.dat", i));
                let mut file = File::create(&path).unwrap();
                file.write_all(b"test").unwrap();
                path
            })
            .collect();

        // Create files that do NOT match the deletion pattern.
        let non_matching_files: Vec<PathBuf> = (0..3)
            .map(|i| {
                let path = base.join(format!("nomatch_{}.dat", i));
                let mut file = File::create(&path).unwrap();
                file.write_all(b"test").unwrap();
                path
            })
            .collect();

        // Use a pattern that only matches the 'match_' files.
        let pattern = base.join("match_*.dat");
        let pattern_str = pattern.to_string_lossy().to_string();

        println!("Running Rust deletion on partial match pattern...");
        let elapsed = measure_rust_deletion(&pattern_str);
        println!(
            "Rust deletion (partial match) completed in {}.",
            format_duration(elapsed)
        );

        // Confirm that matching files have been deleted.
        for path in matching_files {
            assert!(
                !path.exists(),
                "Matching file {} was not deleted",
                path.display()
            );
        }

        // Confirm that non‑matching files remain, then clean them up.
        for path in non_matching_files {
            assert!(
                path.exists(),
                "Non‑matching file {} was mistakenly deleted",
                path.display()
            );
            fs::remove_file(&path).unwrap();
        }
    }
}


// cargo test --release -- --nocapture tokio_tune
#[cfg(test)]
mod tokio_tune {
    use super::{count_matches, run_deletion_tokio, NoOpProgressBar, Progress};
    use std::fs::{File, OpenOptions};
    use std::io::{BufWriter, Write};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    use tempfile::tempdir;
    use tokio::runtime::Builder;

    const TEST_FILE_SIZE_KB: usize = 1;
    const CSV_FILE_NAME: &str = "test_results.csv";

    /// Creates a set of test files in the given directory.
    fn create_test_files(dir: &Path, count: usize) -> Vec<PathBuf> {
        let mut paths = Vec::new();
        for i in 0..count {
            let file_path = dir.join(format!("test_file_{}.dat", i));
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&vec![0u8; TEST_FILE_SIZE_KB * 1024])
                .unwrap();
            paths.push(file_path);
        }
        paths
    }

    /// Generates logarithmically spaced values between 1 and max_val (inclusive).
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
            writeln!(
                csv_writer,
                "SimulatedCPUs,NumFiles,Concurrency,TotalTime(ns)"
            )
            .expect("Failed to write CSV header");
        }
        for &simulated_cpus in &simulated_cpu_counts {
            let num_threads = simulated_cpus.min(actual_cpus);
            let runtime = Builder::new_multi_thread()
                .enable_all()
                .worker_threads(num_threads)
                .build()
                .unwrap();
            for &num_files in &file_counts {
                // Create a temporary directory for this test iteration.
                let tmp_dir = tempdir().unwrap();
                let pattern = format!("{}/*", tmp_dir.path().to_string_lossy());
                let mut min_time = Duration::MAX;
                let mut optimal_concurrency = 1;

                // Generate log‑spaced concurrency levels from 1 to max_concurrency.
                let max_concurrency = max_concurrency_multiplier * simulated_cpus;
                let concurrency_levels = generate_log_space(max_concurrency, 32);

                for concurrency in concurrency_levels {
                    // (Re)create test files.
                    create_test_files(tmp_dir.path(), num_files);
                    let start = Instant::now();
                    // Obtain the directory file descriptor and matching file names.
                    let (fd, matched_files) = match count_matches(&pattern).unwrap() {
                        Some((fd, files)) => (fd, files),
                        None => continue,
                    };
                    let num_matches = matched_files.len();
                    // Use a no‑op progress reporter.
                    let progress = Progress::NoOp(NoOpProgressBar::new());
                    // Call the deletion routine with the proper arguments.
                    runtime
                        .block_on(run_deletion_tokio(
                            Some(concurrency),
                            progress,
                            fd,
                            matched_files,
                            num_matches,
                        ))
                        .unwrap();
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
                    )
                    .expect("Failed to write to CSV");
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
        println!(
            "[Grid Search Test] Complete.  See {} for results.",
            CSV_FILE_NAME
        );
    }
}



// cargo test --release -- --nocapture file_count_tests
#[cfg(target_os = "linux")]
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
            let buf_size = 1 << 26; // 64MB buffer
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
            let buf_size = 1 << 26; // 64MB buffer
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

        // Create 100,000 files
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

// cargo test --release -- --nocapture glob_tests
#[cfg(test)]
mod glob_tests {
    use glob::{glob, glob_with, MatchOptions};
    use globset::{Glob, GlobBuilder, GlobSetBuilder};
    use std::fs::{self, File};
    use std::io;
    use std::path::{Path};
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    // Define an enum to select the matching method.
    #[derive(Clone)]
    enum Method {
        Glob,                        // glob crate, case-sensitive
        GlobSetFilename,             // globset matching on file names, case-sensitive
        GlobSetFullPath,             // globset matching on full paths, case-sensitive
        GlobCaseInsensitive,         // glob crate, case-insensitive
        GlobSetCaseInsensitive,      // globset matching on file names, case-insensitive
        GlobSetFullPathCaseInsensitive, // globset matching on full paths, case-insensitive
    }

    /// Creates temporary files in the given directory.
    /// If `use_subdirs` is true, creates two subdirectories and distributes files between them.
    /// Half of the files are named to match patterns (e.g. "testmatch...") and the other half to not match.
    fn create_temp_files(dir: &Path, num_files: usize, use_subdirs: bool) -> io::Result<()> {
        if use_subdirs {
            let subdirs = ["subdir1", "subdir2"];
            for sub in &subdirs {
                fs::create_dir_all(dir.join(sub))?;
            }
            for i in 0..num_files {
                let sub = if i % 2 == 0 { "subdir1" } else { "subdir2" };
                let file_name = if i % 2 == 0 {
                    // Files that should match patterns that require "testmatch" prefix.
                    format!("testmatch{}_{}.txt", sub, i)
                } else {
                    format!("nomatch{}_{}.txt", sub, i)
                };
                let file_path = dir.join(sub).join(file_name);
                File::create(file_path)?;
            }
        } else {
            for i in 0..num_files {
                let file_name = if i % 2 == 0 {
                    format!("testmatch{}.txt", i)
                } else {
                    format!("nomatch{}.txt", i)
                };
                let file_path = dir.join(file_name);
                File::create(file_path)?;
            }
        }
        Ok(())
    }

    /// Runs a single benchmark for the specified method.
    /// Returns a tuple of (elapsed duration, number of files matched).
    fn run_benchmark(
        num_files: usize,
        pattern: &str,
        use_subdirs: bool,
        method: Method,
    ) -> io::Result<(Duration, usize)> {
        // Create a fresh temporary directory for this test run.
        let temp_dir = TempDir::new()?;
        let dir_path = temp_dir.path();

        // Populate the directory with files.
        create_temp_files(dir_path, num_files, use_subdirs)?;

        // Build the glob pattern strings.
        let glob_pattern = dir_path.join(pattern).to_string_lossy().to_string();
        let full_pattern = format!("{}/{}", dir_path.to_string_lossy(), pattern);

        let start = Instant::now();
        let mut count = 0;
        match method {
            Method::Glob => {
                // Use the glob crate (case-sensitive).
                for entry in glob(&glob_pattern).expect("Failed to read glob pattern") {
                    if let Ok(path) = entry {
                        if path.is_file() {
                            count += 1;
                        }
                    }
                }
            }
            Method::GlobSetFilename => {
                // Use globset matching on file names (case-sensitive).
                let mut builder = GlobSetBuilder::new();
                let compiled = Glob::new(pattern)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                builder.add(compiled);
                let set = builder.build().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                for entry in fs::read_dir(dir_path)? {
                    let entry = entry?;
                    if entry.file_type()?.is_file() && set.is_match(entry.file_name()) {
                        count += 1;
                    }
                }
            }
            Method::GlobSetFullPath => {
                // Use globset matching on full paths (case-sensitive).
                let mut builder = GlobSetBuilder::new();
                let compiled = Glob::new(&full_pattern)
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                builder.add(compiled);
                let set = builder.build().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                for entry in fs::read_dir(dir_path)? {
                    let entry = entry?;
                    if entry.file_type()?.is_file() && set.is_match(&entry.path()) {
                        count += 1;
                    }
                }
            }
            Method::GlobCaseInsensitive => {
                // Use glob_with with case-insensitive options.
                let options = MatchOptions {
                    case_sensitive: false,
                    require_literal_separator: false,
                    require_literal_leading_dot: false,
                };
                for entry in glob_with(&glob_pattern, options).expect("Failed to read glob pattern") {
                    if let Ok(path) = entry {
                        if path.is_file() {
                            count += 1;
                        }
                    }
                }
            }
            Method::GlobSetCaseInsensitive => {
                // Use globset on file names, case-insensitive.
                let mut builder = GlobSetBuilder::new();
                let compiled = GlobBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                builder.add(compiled);
                let set = builder.build().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                for entry in fs::read_dir(dir_path)? {
                    let entry = entry?;
                    if entry.file_type()?.is_file() && set.is_match(entry.file_name()) {
                        count += 1;
                    }
                }
            }
            Method::GlobSetFullPathCaseInsensitive => {
                // Use globset on full paths, case-insensitive.
                let mut builder = GlobSetBuilder::new();
                let compiled = GlobBuilder::new(&full_pattern)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                builder.add(compiled);
                let set = builder.build().map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                for entry in fs::read_dir(dir_path)? {
                    let entry = entry?;
                    if entry.file_type()?.is_file() && set.is_match(&entry.path()) {
                        count += 1;
                    }
                }
            }
        }
        let duration = start.elapsed();
        Ok((duration, count))
    }

    #[test]
    fn comprehensive_benchmark() -> io::Result<()> {
        // Define a set of file counts to test.
        let file_counts = vec![100, 1_000, 10_000, 100_000];
        // Define a set of glob patterns to test.
        let patterns = vec![
            "*.txt",                // simple pattern
            "testmatch*.txt",       // specific pattern requiring a "testmatch" prefix
            "test[0-9]match*.txt",   // pattern with a character range
        ];
        // Test both without subdirectories and with subdirectories.
        let subdir_options = vec![false, true];
        // Define the methods to test (both case-sensitive and case-insensitive).
        let methods = vec![
            ("glob", Method::Glob),
            ("globset filename", Method::GlobSetFilename),
            ("globset fullpath", Method::GlobSetFullPath),
            ("glob (case-insensitive)", Method::GlobCaseInsensitive),
            ("globset (case-insensitive, filename)", Method::GlobSetCaseInsensitive),
            ("globset (case-insensitive, fullpath)", Method::GlobSetFullPathCaseInsensitive),
        ];

        // Loop over each combination of file count, pattern, and subdirectory option.
        for &count in &file_counts {
            for pattern in &patterns {
                for &use_subdirs in &subdir_options {
                    println!("===========================================");
                    println!(
                        "Test: {} files, Pattern: '{}', Subdirectories: {}",
                        count, pattern, use_subdirs
                    );
                    for (method_name, method) in &methods {
                        let (duration, matched) = run_benchmark(count, pattern, use_subdirs, method.clone())?;
                        println!(
                            "{:40} => Duration: {:?}, Matched: {}",
                            method_name, duration, matched
                        );
                    }
                    println!("===========================================");
                }
            }
        }
        Ok(())
    }
}


// cargo test --release -- --nocapture collect_tests
#[cfg(test)]
mod collect_tests {
    use super::*; // Import everything from the parent module, including collect_matching_files.
    use globset::{Glob, GlobSetBuilder};
    use std::{
        fs,
        io,
        os::unix::fs::symlink,
        path::{Path},
    };
    use tempfile::TempDir;

    /// Helper function: Build a globset matcher from a given pattern.
    fn build_matcher(pattern: &str) -> globset::GlobSet {
        let mut gs_builder = GlobSetBuilder::new();
        // Note: Glob::new returns a Result; unwrap here with expect.
        gs_builder.add(Glob::new(pattern).expect("Failed to create glob"));
        gs_builder.build().expect("Failed to build globset")
    }

    /// Test that files matching a simple pattern are correctly collected.
    #[test]
    fn test_collect_basic_matching() -> io::Result<()> {
        // Create a fresh temporary directory.
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        // Create some files.
        // Files that should match "testmatch*.txt"
        let matching_files = ["testmatch1.txt", "testmatch_extra.txt"];
        // Files that should not match.
        let non_matching_files = ["nomatch1.txt", "other.dat"];

        for fname in matching_files.iter().chain(non_matching_files.iter()) {
            fs::write(dir.join(fname), b"dummy")?;
        }

        // Build a matcher for "testmatch*.txt".
        let matcher = build_matcher("testmatch*.txt");

        // Call the function under test.
        let mut collected = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        collected.sort();

        // Build expected full paths.
        let mut expected: Vec<CString> = matching_files.iter().map(|s| {
            let dir_bytes = dir.as_os_str().as_bytes();
            let mut full_path_bytes = Vec::with_capacity(dir_bytes.len() + 1 + s.len());
            full_path_bytes.extend_from_slice(dir_bytes);
            full_path_bytes.push(b'/');
            full_path_bytes.extend_from_slice(s.as_bytes());
            CString::new(full_path_bytes).unwrap()
        }).collect();
        expected.sort();


        assert_eq!(
            collected, expected,
            "Collected files do not match expected files"
        );
        Ok(())
    }

    /// Test that an empty directory yields an empty vector.
    #[test]
    fn test_collect_empty_directory() -> io::Result<()> {
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        let matcher = build_matcher("*.txt");
        let files = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        assert!(files.is_empty(), "Expected no files in an empty directory");
        Ok(())
    }

    /// Test that when no file matches the given pattern, an empty vector is returned.
    #[test]
    fn test_collect_non_matching_pattern() -> io::Result<()> {
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        // Create files that do not match "*.txt"
        for fname in &["file1.dat", "image.png"] {
            fs::write(dir.join(fname), b"dummy")?;
        }
        let matcher = build_matcher("*.txt");
        let files = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        assert!(files.is_empty(), "Expected no matches for pattern '*.txt'");
        Ok(())
    }

    /// Test that the function does not collect "." or ".." entries.
    #[test]
    fn test_dot_entries_not_collected() -> io::Result<()> {
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        // Create a couple of normal files.
        fs::write(dir.join("file.txt"), b"dummy")?;
        fs::write(dir.join(".hidden.txt"), b"dummy")?;

        let matcher = build_matcher("*.txt");
        let mut collected = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        collected.sort();

        let mut expected: Vec<CString> = vec![
            {
                let mut v = Vec::new();
                v.extend_from_slice(dir.as_os_str().as_bytes());
                v.push(b'/');
                v.extend_from_slice(b"file.txt");
                CString::new(v).unwrap()
            },
            {
                let mut v = Vec::new();
                v.extend_from_slice(dir.as_os_str().as_bytes());
                v.push(b'/');
                v.extend_from_slice(b".hidden.txt");
                CString::new(v).unwrap()
            }
        ];
        expected.sort();

        assert_eq!(
            collected, expected,
            "Dot entries were incorrectly included or excluded"
        );
        Ok(())
    }

    /// Test that only regular files are collected (directories and symlinks are skipped).
    #[test]
    fn test_file_type_filtering() -> io::Result<()> {
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        // Create a regular file.
        fs::write(dir.join("regular.txt"), b"dummy")?;
        // Create a subdirectory.
        fs::create_dir(dir.join("subdir"))?;
        // Create a symlink (assumes Unix).
        let target = dir.join("regular.txt");
        let symlink_path = dir.join("link.txt");
        symlink(&target, &symlink_path)?;

        let matcher = build_matcher("*.txt");
        let files = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        assert_eq!(files.len(), 1, "Expected only one regular file to be collected (directories and symlinks should be skipped)");
        let target_c = CString::new(target.as_os_str().as_bytes()).unwrap();
        assert_eq!(files[0], target_c, "Collected file is not the expected regular file");
        Ok(())
    }

    /// Test error handling when a nonexistent directory is passed.
    #[test]
    fn test_nonexistent_directory() {
        let fake_dir = Path::new("/this/dir/should/not/exist");
        let matcher = build_matcher("*.txt");
        let result = collect_matching_files(fake_dir, &matcher);
        assert!(result.is_err(), "Expected an error for a nonexistent directory");
    }

    /// Test error handling when directory permission is denied.
    #[cfg(unix)]
    #[test]
    fn test_permission_denied() -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();

        // Create a file so the directory is not empty.
        fs::write(dir.join("file.txt"), b"dummy")?;
        // Get current permissions.
        let metadata = fs::metadata(dir)?;
        let mut perms = metadata.permissions();
        perms.set_mode(0o000);
        fs::set_permissions(dir, perms.clone())?;

        let matcher = build_matcher("*.txt");
        let result = collect_matching_files(dir, &matcher);
        // Restore permissions so TempDir can be cleaned up.
        perms.set_mode(0o755);
        fs::set_permissions(dir, perms)?;
        assert!(result.is_err(), "Expected a permission denied error");
        Ok(())
    }

    /// Creates many files and prints the elapsed time for collecting matching files.
    #[test]
    fn test_performance_large_directory() -> io::Result<()> {
        let temp_dir = TempDir::new()?;
        let dir = temp_dir.path();
        let num_files = 100_000; // Adjust as needed.
        // Create 100_000 files; half match "testmatch*.txt".
        for i in 0..num_files {
            let fname = if i % 2 == 0 {
                format!("testmatch{:06}.txt", i)
            } else {
                format!("nomatch{:06}.dat", i)
            };
            fs::write(dir.join(&fname), b"dummy")?;
        }
        let matcher = build_matcher("testmatch*.txt");

        let start = std::time::Instant::now();
        let files = Vec::<CString>::new();
        collect_matching_files(dir, &matcher)?;
        let elapsed = start.elapsed();
        println!(
            "Performance test: Collected {} matching files out of {} in {:?}",
            files.len(),
            num_files,
            elapsed
        );
        // For 100_000 files (around 50_000 matches), expect a very short runtime.
        Ok(())
    }
}

// cargo test --release -- --nocapture rayon_tune
#[cfg(test)]
mod rayon_tune {
    use super::{count_matches, run_deletion_rayon, N_CPUS_F};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::Instant;
    use tempfile::TempDir;

    #[test]
    fn grid_search_rayon() {
        // CSV file to append results to
        let csv_path = "rayon_deletion_benchmark.csv";

        // These are the file counts to test
        let file_counts = [8, 80, 800, 4000, 20_000, 80_000, 250_000];

        // Thread pool size factors to test (relative to CPU count)
        let concurrency_factors = [0.2, 0.4, 0.6, 1.4, 1.0, 3.0, 5.0];

        // Various batch sizes to try
        let batch_sizes = [200, 400, 800, 1000, 2000, 3000, 5000];

        // Loop over each file count.
        for &fc in &file_counts {
            // For each concurrency factor...
            for &factor in &concurrency_factors {
                // Convert the factor to an integer concurrency value.
                let concurrency = (factor * (*N_CPUS_F)).ceil() as usize;

                // And for each batch size...
                for &bsize in &batch_sizes {
                    // Create a fresh temporary directory for this test run.
                    let tmp_dir = TempDir::new().expect("Failed to create temp dir");

                    // Generate `fc` files matching our pattern.
                    for i in 0..fc {
                        let file_path = tmp_dir.path().join(format!("testfile_{}.tmp", i));
                        std::fs::write(&file_path, b"some test data").unwrap();
                    }

                    // Construct the glob pattern to match the generated files.
                    let pattern = format!("{}/testfile_*.tmp", tmp_dir.path().display());

                    println!(
                        "[INFO] Deleting {} files using Rayon with concurrency = {}, batch_size = {} (from {})",
                        fc,
                        concurrency,
                        bsize,
                        tmp_dir.path().display()
                    );

                    // Before deletion, obtain the file descriptor and matching file list.
                    let (fd, matched_files) = match count_matches(&pattern).unwrap() {
                        Some((fd, files)) => (fd, files),
                        None => {
                            println!("No matching files found for pattern: {}", pattern);
                            continue;
                        }
                    };
                    let num_matches = matched_files.len();

                    // Start timing the deletion.
                    let start = Instant::now();

                    // Run the Rayon-based deletion with the chosen concurrency and batch size.
                    // Note that run_deletion_rayon takes 5 arguments:
                    // thread_pool_size, batch_size_override, fd, matched_files, matched_files_number.
                    let res = run_deletion_rayon(
                        Some(concurrency),
                        Some(bsize),
                        fd,
                        matched_files,
                        num_matches,
                    );

                    let elapsed = start.elapsed();
                    // Make sure deletion succeeded.
                    res.expect("Deletion failed in grid search test");

                    // Append the test result to the CSV file.
                    let mut file = OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(csv_path)
                        .expect("Could not open CSV for appending");

                    // CSV format: file_count, time_in_seconds, thread_pool_size, batch_size
                    writeln!(
                        file,
                        "{},{},{},{}",
                        fc,
                        elapsed.as_secs_f64(),
                        concurrency,
                        bsize
                    )
                    .expect("Failed to write to CSV");
                    // tmp_dir is dropped here and its contents are cleaned up.
                }
            }
        }
    }
}


// cargo test --release -- --nocapture
