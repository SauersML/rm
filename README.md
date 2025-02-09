# File Deletion Utility

This is a Rust-based command-line utility that deletes files matching a given pattern in the filesystem. The utility supports glob patterns (such as wildcards) to match filenames, and it attempts to delete the files in parallel.

## Features
- Accepts a glob pattern to find files in the filesystem.
- Deletes the files that match the given pattern.
- Implements retry logic for file deletions (up to 3 attempts).
- Uses async operations and spawning blocking tasks for efficient concurrent file deletions.
- Logs errors and success messages to the standard output and error.
