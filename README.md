# File Deletion Utility

This is a Rust-based command-line utility that deletes files matching a given pattern in the filesystem. It uses glob patterns (wildcards) for matching filenames and performs the deletions in parallel.

## Features
- Accepts a glob pattern to find files in the filesystem.
- Deletes files matching the pattern.
- Implements retry logic for file deletions (up to 3 attempts).
- Uses async operations for efficient parallel file deletions.
- Logs success and error messages to the standard output.

## Installation

To install the utility globally, run:

```bash
cargo install del
```

## Usage

Run the utility with the glob pattern of the files you want to delete:

```bash
del "<pattern>"
```

Replace `<pattern>` with the glob pattern (e.g., `*.log` to delete all `.log` files).
