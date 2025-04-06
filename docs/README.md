# EngramDB Documentation

Welcome to the EngramDB documentation. EngramDB is a specialized database system designed specifically for agent memory management, providing efficient storage, retrieval, and querying of agent memories using a unified memory representation model.

## What is EngramDB?

EngramDB is a database system that combines vector, graph, and temporal properties in a single primitive called a MemoryNode. This unified approach allows for rich, multi-dimensional representation of agent memories, enabling complex queries that can leverage semantic similarity, relational connections, and temporal evolution.

## Core Features

- **Unified Memory Representation**: Combines graph, vector, and temporal properties in a single primitive
- **Vector Similarity Search**: Find memories with similar semantic content
- **Flexible Storage Options**: In-memory database for testing and development, file-based for persistence
- **Query API**: Rich querying with vector similarity, attribute filters, and temporal constraints
- **Memory Evolution**: Track changes to memories over time with temporal layers
- **Python Bindings**: First-class Python API for integration with ML and AI applications
- **Web Interface**: Browser-based UI for visualization and interaction with the database

## Documentation Contents

- [Getting Started](getting-started.md) - Installation and basic usage
- [Core Concepts](core-concepts.md) - Explanation of key concepts
- API Reference
  - [MemoryNode](api-reference/memory-node.md) - MemoryNode API
  - [Database](api-reference/database.md) - Database API
  - [Query](api-reference/query.md) - Query API
  - [Vector Index](api-reference/vector-index.md) - Vector index API
- Examples
  - [Rust Examples](examples/rust-examples.md) - Rust examples
  - [Python Examples](examples/python-examples.md) - Python examples
- Advanced Topics
  - [Temporal Layers](advanced-topics/temporal-layers.md) - Working with temporal layers
  - [Connections](advanced-topics/connections.md) - Working with connections
  - [Storage Engines](advanced-topics/storage-engines.md) - Storage engine details
- [Contributing](contributing.md) - Guide for contributors

## License

This project is licensed under the MIT License - see the LICENSE file for details.
