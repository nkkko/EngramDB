# Core Concepts

EngramDB is built around several key concepts that work together to provide a comprehensive memory management system for agents. This document explains these core concepts and how they relate to each other.

## Memory Node

The `MemoryNode` is the fundamental unit of storage in EngramDB. It combines multiple aspects of memory representation:

- **Vector Embeddings**: Numerical vectors that represent the semantic content of the memory, enabling similarity search
- **Attributes**: Key-value pairs for structured data associated with the memory
- **Connections**: Graph-like relationships to other memory nodes
- **Temporal Layers**: Historical versions of the memory that track how it evolves over time

A memory node is identified by a unique UUID and includes metadata such as creation timestamp and access patterns.

## Vector Embeddings

Vector embeddings are numerical representations of semantic content. In EngramDB, these embeddings enable similarity search, allowing you to find memories that are semantically related even if they don't share the same attributes.

The vector index maintains these embeddings and provides efficient similarity search using cosine similarity.

## Attributes

Attributes are flexible key-value pairs associated with a memory node. They can store various types of data:

- Strings
- Integers
- Floating-point numbers
- Booleans
- Lists
- Maps (nested key-value pairs)

Attributes provide a way to add structured metadata to memories and enable filtering in queries.

## Connections

Connections represent relationships between memory nodes, forming a graph structure. Each connection has:

- A target memory node ID
- A relationship type (e.g., Association, Causation, Sequence)
- A strength value (between 0.0 and 1.0)
- A creation timestamp

Connections allow memories to reference and relate to each other, enabling graph-based queries and traversals.

## Temporal Layers

Temporal layers track how memories evolve over time. Each layer represents a version of the memory at a specific point in time and includes:

- A timestamp
- Optionally modified embeddings
- Optionally modified attributes
- A reason for the change

This temporal dimension allows EngramDB to maintain a history of how memories change and evolve, which is crucial for agent systems that learn and update their knowledge over time.

## Database

The `Database` is the main interface for interacting with EngramDB. It provides methods for:

- Saving and loading memory nodes
- Searching for similar memories
- Querying with complex filters
- Managing connections between memories

The database combines a storage engine (for persistence) with a vector index (for similarity search).

## Storage Engines

EngramDB supports multiple storage backends:

- **Memory Storage**: Volatile storage for testing and development
- **File Storage**: Persistent storage for production use

The storage engine is responsible for saving and loading memory nodes, while the database coordinates between storage and indexing.

## Query System

The query system provides a flexible way to search for memories based on multiple criteria:

- Vector similarity
- Attribute filters
- Temporal filters
- Connection filters

Queries can combine these different dimensions to find memories that match complex criteria.

## Putting It All Together

In EngramDB, these concepts work together to provide a comprehensive memory system:

1. You create `MemoryNode` instances with embeddings and attributes
2. You save them to the `Database`
3. The database stores them using the `StorageEngine` and indexes their embeddings in the `VectorIndex`
4. You can create `Connection`s between nodes to form a memory graph
5. You can update nodes over time, creating `TemporalLayer`s that track their evolution
6. You can query the database using vector similarity, attribute filters, and temporal constraints

This unified approach allows for rich, multi-dimensional representation and querying of agent memories.
