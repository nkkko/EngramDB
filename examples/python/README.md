# EngramDB Python Examples

This directory contains Python examples demonstrating how to use the EngramDB Python bindings (`engramdb-py`).

## Prerequisites

- Python 3.8+
- `engramdb-py` package installed

## Installation

You can install the EngramDB Python bindings using pip:

```bash
pip install engramdb-py
```

## Examples

Here's an overview of the provided examples:

1. **basic_usage.py**: Shows the basic operations such as creating, saving, and retrieving memories, as well as basic vector similarity search.

2. **memory_and_file_storage.py**: Demonstrates how to work with both in-memory and file-based storage in EngramDB.

3. **memory_graph.py**: Shows how to create a knowledge graph using memory nodes and relationships between them, representing a network of connected information.

4. **single_file_storage.py**: Demonstrates using a single file for storage and performing various operations on it.

5. **unified_database.py**: Shows how to create a unified database containing different types of memories (notes, tasks, contacts, documents) and searching across all of them.

6. **agent_memory.py**: Provides a simple implementation of an agent memory system using EngramDB, demonstrating how to store conversation messages, facts, and tasks, and how to perform semantic search across this memory.

## Running the Examples

You can run any example with Python directly:

```bash
python basic_usage.py
python memory_graph.py
```

## Agent Example

For a more sophisticated example, check out the Flask website generator agent in the `../agent/` directory, which demonstrates using EngramDB for agent memory.

## Additional Resources

- [EngramDB Documentation](https://engramdb.mintlify.app/)
- [Rust Examples](../rust/)