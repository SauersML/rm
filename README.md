# `del`: Faster File Deletion

This is a Rust-based command-line utility that deletes files matching a given pattern in the filesystem, which aims to be faster than the "rm" command. It uses glob patterns (wildcards) for matching filenames and performs the deletions in parallel.

## Features
- Accepts a glob pattern (Unix-style) to find files in the filesystem.
- Deletes files matching the pattern.
- Has a progress bar when there are many files.
- Implements retry logic for file deletions (up to 3 attempts).
- Uses parallel operations for efficient parallel file deletions.
- Logs success and error messages to the standard output.

## What `del` does not do
- Shell expansion of wildcards resulting in "too many arguments" errors.
- Pattern-matching on directories: you must specify the single directory you wish to delete files in.
- Recursion.
- Directory deletion.
- Warnings: `del` trusts that your input is what you intended.
- Run on Windows.

## Speed
- `del` aims to be faster than or equivalent to rm on Linux universally, and usually faster on MacOS.
- `del` uses parallel execution with a thread pool size and batch size calculated from the number of files with a formula derived from empirical optimization.

## Installation

To install the utility globally, run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo install del
```

Or download the binary directly:
```
wget https://github.com/SauersML/rm/releases/download/v0.1.1/del-x86_64-unknown-linux-gnu.tar.gz && tar -xvzf del-x86_64-unknown-linux-gnu.tar.gz && chmod +x del
```

Or build the program:
```
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Usage

Run the utility with the glob pattern of the files you want to delete:

```bash
del "<pattern>"
```

Replace `<pattern>` with the glob pattern (e.g., `*.log` to delete all `.log` files).

Make sure to add the "./" if you installed via binary download.
