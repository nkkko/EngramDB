# EngramDB Python Bindings

Python bindings for the EngramDB database.

## Overview

This package provides Python bindings for the EngramDB database, allowing Python applications to leverage the specialized agent memory database system. It uses PyO3 to create efficient Rust-Python bindings with minimal overhead.

## Installation

```bash
pip install engramdb-py
```

## Usage

### Basic Usage

```python
import engramdb

# Create an in-memory database
db = engramdb.Database.in_memory()

# Create a memory node
memory = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.4])
memory.set_attribute("title", "Important information")
memory.set_attribute("importance", 0.8)

# Save to database
memory_id = db.save(memory)
print(f"Saved memory with ID: {memory_id}")

# Search for similar memories
results = db.search_similar([0.15, 0.25, 0.35, 0.45], limit=5, threshold=0.7)
for memory_id, score in results:
    memory = db.load(memory_id)
    print(f"Found similar memory: {memory.get_attribute('title')} (score: {score:.2f})")
```

### Persistent Storage

```python
# Create a file-based database
db = engramdb.Database.file_based("./my_database")
```

### Memory Connections (Graph Features)

```python
# Create related memories
memory1 = engramdb.MemoryNode([0.1, 0.2, 0.3])
memory1.set_attribute("title", "Project Plan")

memory2 = engramdb.MemoryNode([0.2, 0.3, 0.4])
memory2.set_attribute("title", "Meeting Notes")

# Save both memories
memory1_id = db.save(memory1)
memory2_id = db.save(memory2)

# Create a connection
db.connect(
    memory1_id,
    memory2_id,
    "related_to",
    0.8
)
```

### Thread Safety for Multi-Agent Systems

For systems with multiple agents accessing the same database concurrently:

```python
# Create a thread-safe database
db = engramdb.ThreadSafeDatabase.in_memory()

# Operations are the same, but thread-safe
memory = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.4])
memory_id = db.save(memory)

# For connection pools in shared environments
pool = engramdb.ThreadSafeDatabasePool.new("./shared_db")
db_conn = pool.get_connection()
```

## API Reference

### Database

- `Database.in_memory()` - Create an in-memory database
- `Database.file_based(path)` - Create a file-based database
- `Database.save(memory)` - Save a memory node
- `Database.load(id)` - Load a memory node by ID
- `Database.delete(id)` - Delete a memory node
- `Database.search_similar(vector, limit, threshold)` - Search for similar memory nodes
- `Database.connect(source_id, target_id, relationship_type, strength)` - Create connection between nodes

### MemoryNode

- `MemoryNode(embeddings)` - Create a new memory node
- `MemoryNode.id` - Get the node ID
- `MemoryNode.set_attribute(key, value)` - Set an attribute
- `MemoryNode.get_attribute(key)` - Get an attribute
- `MemoryNode.get_embeddings()` - Get the vector embeddings

### ThreadSafeDatabase

- `ThreadSafeDatabase.in_memory()` - Create an in-memory thread-safe database
- `ThreadSafeDatabase.file_based(path)` - Create a file-based thread-safe database
- `ThreadSafeDatabase.in_memory_with_hnsw()` - Create database with HNSW index

### ThreadSafeDatabasePool

- `ThreadSafeDatabasePool.new(path)` - Create a database connection pool
- `ThreadSafeDatabasePool.get_connection()` - Get a connection from the pool

## LLM Context Generation

This package includes tools to generate XML context files for LLMs:

```bash
# Generate basic llms.txt file
python scripts/build_llms_txt.py

# Generate full context with examples
python scripts/build_llms_txt.py --full

# Convert to XML context format
python scripts/llms_txt2ctx.py llms-full-engramdb.txt --optional > llms-full-engramdb.md
```

## Development

### Building from Source

To build the package from source:

```bash
git clone https://github.com/nkkko/engramdb.git
cd engramdb/python
pip install maturin
maturin develop
```

### Cross-Platform Building

See [PUBLISHING.md](PUBLISHING.md) for detailed instructions on building wheels for different platforms.

### Running Tests

```bash
pytest tests/
```