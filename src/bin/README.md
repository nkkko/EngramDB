# EngramDB CLI

A command-line interface for managing EngramDB databases.

## Overview

EngramDB CLI provides a comprehensive command-line and interactive shell interface for working with EngramDB databases. It supports all EngramDB storage engines and enables users to create, view, edit, and delete memory nodes, as well as manage the connections between them.

## Features

- **Memory Management**: Create, view, edit, and delete memory nodes
- **Search**: Search through memories based on string attributes
- **Connection Management**: Create and remove connections between memories
- **Multiple Storage Types**: Support for in-memory, multi-file, and single-file storage
- **Interactive Shell**: User-friendly shell interface for database management
- **Command Line Interface**: Scriptable command-line operations

## Installation

Build the CLI using Cargo:

```bash
# Build the CLI
cargo build --bin engramdb-cli

# Run the CLI
cargo run --bin engramdb-cli
```

## Usage

### Command-Line Options

```bash
USAGE:
    engramdb-cli [OPTIONS] [SUBCOMMAND]

OPTIONS:
    -d, --database <FILE>        Path to database file or directory
    -h, --help                   Print help
    -m, --memory                 Use in-memory database
    -s, --storage-type <TYPE>    Storage type: multi-file, single-file, or memory
                                 [default: multi-file]
    -V, --version                Print version
```

### Subcommands

```
SUBCOMMANDS:
    connect       Add a connection between memories
    create        Create a new memory
    delete        Delete a memory
    disconnect    Remove a connection between memories
    edit          Edit an existing memory
    help          Print this message or the help of the given subcommand(s)
    list          List all memories
    search        Search for memories
    shell         Interactive shell mode
    view          View a specific memory
```

### Examples

#### Basic Operations

```bash
# Start interactive shell (default action)
cargo run --bin engramdb-cli

# Use in-memory database (data is lost when program exits)
cargo run --bin engramdb-cli -- --memory

# Use specific database file/directory
cargo run --bin engramdb-cli -- --database /path/to/database

# Use single-file storage
cargo run --bin engramdb-cli -- --database /path/to/database.engramdb --storage-type single-file
```

#### Managing Memories

```bash
# List all memories
cargo run --bin engramdb-cli list

# View a specific memory
cargo run --bin engramdb-cli view <memory-id>

# Create a new memory
cargo run --bin engramdb-cli create --description "Sample memory" --importance 0.8 --tags "sample,test"

# Edit a memory
cargo run --bin engramdb-cli edit <memory-id> --description "Updated description"

# Delete a memory
cargo run --bin engramdb-cli delete <memory-id>
```

#### Managing Connections

```bash
# Create a connection between memories
cargo run --bin engramdb-cli connect --source <source-id> --target <target-id> --relationship "Association" --strength 0.9

# Remove a connection
cargo run --bin engramdb-cli disconnect --source <source-id> --target <target-id>
```

#### Searching

```bash
# Search memories
cargo run --bin engramdb-cli search "search term"
```

### Interactive Shell

The interactive shell provides an easy-to-use interface for managing EngramDB databases:

```bash
$ cargo run --bin engramdb-cli
EngramDB Interactive Shell
Type 'help' for a list of commands, 'exit' to quit
engramdb> help
```

Available shell commands:

- `list`: List all memories
- `view <id>`: View a specific memory
- `create`: Create a new memory (interactive prompts)
- `edit <id>`: Edit a memory (interactive prompts)
- `delete <id>`: Delete a memory
- `search <query>`: Search for memories
- `connect`: Add a connection (interactive prompts)
- `disconnect`: Remove a connection (interactive prompts)
- `exit`, `quit`: Exit the shell

## Shell Examples

```
engramdb> list
Found 2 memories:
╔══════════════════════════════════════════════════════════════════════════════════════════
║                  ID                  │             Description              │ Created
╠══════════════════════════════════════════════════════════════════════════════════════════
║ 89178284-e169-46c2-9a94-05cb9042ff8b │ Target memory                        │ 2025-04-07 04:04:32
║ 74a7482a-7fd4-4419-a8fa-5ed0e5a0c808 │ Source memory                        │ 2025-04-07 04:04:32
╚══════════════════════════════════════════════════════════════════════════════════════════

engramdb> view 89178284-e169-46c2-9a94-05cb9042ff8b
╔══════════════════════════════════════════════════════════════
║ Memory ID: 89178284-e169-46c2-9a94-05cb9042ff8b
╠══════════════════════════════════════════════════════════════
║ Attributes:
║   importance: 0.6000000238418579
║   tags: target,test
║   description: Target memory
║   creation_timestamp: 1743998672
║ Embeddings: [0.10, 0.20, 0.30, ... 7 more values]
║ Connections:
║   → 74a7482a-7fd4-4419-a8fa-5ed0e5a0c808 (Association, strength: 0.90)
╚══════════════════════════════════════════════════════════════
```

## Storage Types

EngramDB CLI supports three storage types:

- **Memory Storage**: In-memory storage that doesn't persist data after the program exits. Useful for testing and temporary operations.
- **Multi-File Storage**: File-based storage that stores each memory node in a separate file within a directory hierarchy.
- **Single-File Storage**: Stores all memory nodes in a single file, which can be easier to manage for small databases.

## Important Notes

- When using in-memory storage, data is not persisted between CLI invocations. This is by design.
- For persistent storage, use either multi-file or single-file storage with the `--database` option.
- UUIDs are used for all memory node identifiers. Make sure to use the correct UUIDs when referencing memories in commands.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.