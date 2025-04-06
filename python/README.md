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

# Query the database
results = db.query()\
    .with_vector([0.15, 0.25, 0.35, 0.45])\
    .execute()

# Process results
for node in results:
    print(f"Found memory: {node.get_attribute('title')}")
```

### Persistent Storage

```python
# Create a file-based database
db = engramdb.Database.file_based("./my_database")

# Initialize the database (loads existing memories)
db.initialize()
```

### Advanced Queries

```python
# Create attribute filters
importance_filter = engramdb.AttributeFilter.greater_than("importance", 0.7)
category_filter = engramdb.AttributeFilter.equals("category", "work")

# Create a temporal filter for recent memories
time_filter = engramdb.TemporalFilter.within_last(days=7)

# Execute combined query
results = db.query()\
    .with_vector([0.1, 0.3, 0.5, 0.1])\
    .with_attribute_filter(importance_filter)\
    .with_attribute_filter(category_filter)\
    .with_temporal_filter(time_filter)\
    .with_limit(10)\
    .execute()
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
    relationship_type=engramdb.RelationshipType.ASSOCIATION,
    strength=0.8
)
```

## API Reference

### Database

- `Database.in_memory()` - Create an in-memory database
- `Database.file_based(path)` - Create a file-based database
- `Database.initialize()` - Initialize the database
- `Database.save(memory)` - Save a memory node
- `Database.load(id)` - Load a memory node by ID
- `Database.delete(id)` - Delete a memory node
- `Database.query()` - Create a query builder

### MemoryNode

- `MemoryNode(embeddings)` - Create a new memory node
- `MemoryNode.id` - Get the node ID
- `MemoryNode.embeddings` - Get/set the vector embeddings
- `MemoryNode.set_attribute(key, value)` - Set an attribute
- `MemoryNode.get_attribute(key)` - Get an attribute

### QueryBuilder

- `QueryBuilder.with_vector(vector)` - Add vector similarity search
- `QueryBuilder.with_attribute_filter(filter)` - Add attribute filter
- `QueryBuilder.with_temporal_filter(filter)` - Add temporal filter
- `QueryBuilder.with_limit(limit)` - Set maximum results
- `QueryBuilder.execute()` - Execute the query

## Development

### Building from Source

To build the package from source:

```bash
git clone https://github.com/nkkko/engramdb.git
cd engramdb/python
pip install maturin
maturin develop
```

### Running Tests

```bash
pytest tests/
```